from __future__ import division
from __future__ import print_function
import os
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
import cv2
import tensorflow as tf

from generator.EOT_simulation import transformation
from generator.attack_method.base_logic import ODDLogic
from generator.attack_method.coordinates import Coordinates
from generator.layers.eot_layer import EOTLayer

from generator.object_detectors.tiny_yolo.tiny_yolo import TinyYOLO
from generator.object_detectors.tiny_yolo.config import TinyYOLOConfig
from generator.object_detectors.tiny_yolo_v3.models import TinyYoloV3
from generator.object_detectors.tiny_yolo_v4.yolov4 import TinyYoloV4
from generator.object_detectors.tiny_yolo_v4 import utils


class EOTAttack(ODDLogic):
    """
    This class contains the implementation for the EOT attack necessary to generate
    adversarial stickers.
    """

    def __init__(self, model):
        super().__init__(model)
        # init global variable
        self.filewrite_img = False
        self.filewrite_txt = False
        self.tofile_img = os.path.join(self.path, 'output.jpg')
        self.tofile_txt = os.path.join(self.path, 'output.txt')

        if model == TinyYOLO:
            self.logo_size = 448
        else:
            self.logo_size = 416

        # optimization settings
        self.inter = None
        self.eot_transforms = None
        self.num_of_eot_transforms = None
        self.learning_rate = 1e-2
        self.punishment_value = 0.001
        self.smoothness_punishment_value = 0.6
        self.steps = 5

        self.alpha = 0.1
        self.threshold = 0.2
        self.iou_threshold = 0.5
        self.num_class = 20
        self.eot_split = 1
        self.classes = ["aeroplane",
                        "bicycle",
                        "bird",
                        "boat",
                        "bottle",
                        "bus",
                        "car",
                        "cat",
                        "chair",
                        "cow",
                        "diningtable",
                        "dog",
                        "horse",
                        "motorbike",
                        "person",
                        "pottedplant",
                        "sheep",
                        "sofa",
                        "train",
                        "tvmonitor"]
        self.precision = 'float32'
        self.minimise = True
        self.init_eot_variables()

    def init_eot_variables(self):
        """
        Initializes the variable that controls the adversarial image
        :return:
        """
        self.eot_transforms = np.array(transformation.target_sample())
        self.num_of_eot_transforms = len(self.eot_transforms)

        init_inter = tf.constant(
            0.7 * np.random.normal(scale=0.8, size=[1, self.logo_size, self.logo_size, 3]),
            dtype=self.precision
        )

        self.inter = tf.Variable(
            init_inter,
            name='inter',
            shape=[1, self.logo_size, self.logo_size, 3],
            dtype=self.precision,
            trainable=True
        )

    def arg_parser(self, args):
        """
        Parses arguments
        :param args:
        :return:
        """
        self.fromfile = args.input_images
        self.fromlogofile = args.input_logo
        self.frommaskfile = args.mask_file

        if args.steps:
            self.steps = args.steps

        if args.punishment:
            self.punishment_value = args.punishment

        if args.smoothness_punishment:
            self.smoothness_punishment_value = args.smoothness_punishment

        if args.skip_pdb is not None:
            self.skip_pdb = args.skip_pdb

        if args.minimise is not None:
            self.minimise = args.minimise

        if args.learning_rate:
            self.learning_rate = args.learning_rate

        if args.num_batches:
            self.eot_split = args.num_batches

    def extract_confidence_from_yolo_output(self, outputs):
        """
        Used for Yolo v3/v4. From the output of the network, fetches the confidence and
        class probabilities in order to compute score. The global maximum is then fetched.
        :param outputs:
        :return:
        """
        box, confs, probs = [], [], []
        for out in outputs:
            box.append(tf.reshape(out[0], (tf.shape(out[0])[0], -1, tf.shape(out[0])[-1])))
            confs.append(tf.reshape(out[1], (tf.shape(out[1])[0], -1, tf.shape(out[1])[-1])))
            probs.append(tf.reshape(out[2], (tf.shape(out[2])[0], -1, tf.shape(out[2])[-1])))

        bbox = tf.concat(box, axis=1)
        confidence = tf.concat(confs, 1)
        class_probs = tf.concat(probs, 1)
        class_probs = class_probs[:, :, 0] # Person is first class in COCO
        class_probs = tf.reshape(class_probs, (tf.shape(class_probs)[0], tf.shape(class_probs)[1], 1))

        scores = confidence * class_probs

        if not self.minimise:
            return tf.reduce_sum(tf.reduce_max(scores[:, :, 0], axis=1))

        target_confidence = 0
        confidence_threshold = 0.1
        iou_threshold = 0.4

        for i in range(scores.shape[0]):
            threshold_indices = tf.where(confidence[i, :, 0] > confidence_threshold)
            if threshold_indices.shape[0] > 0:
                threshold_indices = threshold_indices[:, 0]
                bbox_filter = tf.gather(bbox[i, :, :], threshold_indices, axis=0)
                scores_filter = tf.gather(scores[i, :, 0], threshold_indices)
                indices = tf.image.non_max_suppression(bbox_filter,
                                                        scores_filter,
                                                        100,
                                                        iou_threshold=iou_threshold)

                nms_scores = tf.gather(scores_filter, indices)

                target_confidence += tf.reduce_sum(nms_scores)

        return target_confidence

    def generate_splits(self):
        """
        Depending on the chosen batching size, separates the eot transforms into splits.
        :return:
        """
        indexes = list(range(len(self.eot_transforms)))
        splitter = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]

        return splitter(indexes, int(len(self.eot_transforms) / self.eot_split))

    def train_step(self, input_image, input_mask, options: dict):
        """
        Performs the training step for one iteration. In case the model is Keras, calls the
        Model class directly. Otherwise, it calls the `build_graph` method, which should
        be present in any model for this to work.
        :param input_image:
        :param input_mask:
        :param options:
        :return:
        """
        chunks = self.generate_splits()

        with tf.GradientTape() as tape:
            final_loss = 0
            eot_image_final = None
            total_conf = 0
            for chunk in chunks:
                eot_image, partial_loss = EOTLayer(
                    self.punishment_value, self.smoothness_punishment_value, self.inter
                )((input_image[chunk], input_mask, self.eot_transforms[chunk]))

                if options['is_keras_model']:
                    output = options['object_detector'](eot_image)
                else:
                    output = options['object_detector'].build_graph(eot_image)

                if options['extract_confidence']:
                    confidence_target = self.extract_confidence_from_yolo_output(output)
                else:
                    confidence_target = output

                final_loss += (confidence_target + partial_loss)
                total_conf += confidence_target

                if eot_image_final is None:
                    eot_image_final = eot_image[0]

            if not self.minimise:
                final_loss = - final_loss

        print('Confidence: ' + str(total_conf.numpy()))

        grads = tape.gradient(final_loss, [self.inter])
        options['optimizer'].apply_gradients(zip(grads, [self.inter]))

        return eot_image_final

    def attack_optimize(self, img_list, mask, logo_mask=None, resized_logo_mask=None, **kwargs):
        """
        Contains initialization of models and variables and calls the training step in order
        to obtained the adversarial image.
        :param img_list:
        :param mask:
        :param logo_mask:
        :param resized_logo_mask:
        :return:
        """
        options = {}

        inputs = np.zeros(
            (self.num_of_eot_transforms, self.logo_size, self.logo_size, 3), dtype=self.precision
        )
        inputs_mask = np.zeros(
            (1, self.logo_size, self.logo_size, 3), dtype=self.precision
        )
        inputs_mask[0] = cv2.resize(mask, (self.logo_size, self.logo_size))

        if self.model == TinyYOLO:
            options['object_detector'] = TinyYOLO({
                'all_variables': self.load_weights_tf2(),
                'iou_threshold': self.iou_threshold,
                'threshold': self.threshold,
                'classes': self.classes,
                'minimise': self.minimise
            })
            options['is_keras_model'] = False
            options['extract_confidence'] = False
        elif self.model == TinyYoloV3:
            options['object_detector'] = TinyYoloV3().build_graph(size=self.logo_size, classes=80)
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                '../object_detectors/tiny_yolo_v3/checkpoints/yolov3-tiny.tf'
            )
            options['object_detector'].load_weights(checkpoint_path)
            options['is_keras_model'] = True
            options['extract_confidence'] = True
        elif self.model == TinyYoloV4:
            options['object_detector'] = TinyYoloV4().build_graph(self.logo_size, 80)
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                '../object_detectors/tiny_yolo_v4/checkpoints/yolov4-tiny.weights'
            )
            utils.load_weights(options['object_detector'], checkpoint_path, is_tiny=True)
            options['is_keras_model'] = True
            options['extract_confidence'] = True

        options['optimizer'] = tf.optimizers.Adam(self.learning_rate)
        eot_image = None

        for i in range(self.steps):
            for j in range(self.num_of_eot_transforms):
                choose = np.random.randint(len(img_list))
                img_with_logo = self._init_sticker_area(img_list[choose][1], resized_logo_mask)

                img_resized = cv2.resize(img_with_logo, (self.logo_size, self.logo_size))

                inputs[j] = (img_resized / 255.0) * 2.0 - 1.0

            print('Step: ' + str(i + 1))

            eot_image = self.train_step(
                inputs, inputs_mask, options
            )

        return eot_image

    @classmethod
    def generate_paste_coords(cls, resized_logo_mask, coords: Coordinates):
        """
        Creates the necessary paste coordinates to generate the adversarial image
        :param resized_logo_mask:
        :param coords:
        :return:
        """
        ad_area_center_x = old_div((coords.xmin + coords.xmax), 2)
        ad_area_center_y = old_div((coords.ymin + coords.ymax), 2)

        # cv2.resize only eats integer
        resized_width = resized_logo_mask.shape[1]
        resized_height = resized_logo_mask.shape[0]

        paste_xmin = int(ad_area_center_x - old_div(resized_width, 2))
        paste_ymin = int(ad_area_center_y - old_div(resized_height, 2))
        paste_xmax = paste_xmin + resized_width
        paste_ymax = paste_ymin + resized_height

        return paste_xmin, paste_ymin, paste_xmax, paste_ymax

    # add logo on input
    def _init_sticker_area(self, pic_in_numpy_0_255, resized_logo_mask=None):
        """
        Initialize the sticker
        :param pic_in_numpy_0_255:
        :param logo_mask:
        :param resized_logo_mask:
        :return:
        """
        pic_in_numpy_0_255_copy = np.array(pic_in_numpy_0_255)
        _object = self.mask_list[0]
        coords = self.get_mask_coordination(_object)

        pic_in_numpy_0_255_copy[coords.ymin:coords.ymax, coords.xmin:coords.xmax] = [164, 83, 57]

        if resized_logo_mask is not None:
            paste_xmin, paste_ymin, paste_xmax, paste_ymax = EOTAttack.generate_paste_coords(
                resized_logo_mask, coords
            )

            # can also write as np.where(cond, v1, v2)
            for i in range(paste_xmin, paste_xmax):
                for j in range(paste_ymin, paste_ymax):
                    if resized_logo_mask[j - paste_ymin, i - paste_xmin, 0] == self.very_small:
                        # plot logo
                        pic_in_numpy_0_255_copy[j, i] = 255

        return pic_in_numpy_0_255_copy

    def load_weights_tf2(self):
        """
        Load weights for tiny_yolo in tf2 fashion
        :return:
        """
        config = TinyYOLOConfig(self.precision)
        var_configs = config.get_config()

        saver = tf.compat.v1.train.Saver(var_configs)
        saver.restore(None, '../weights/YOLO_tiny.ckpt')

        return var_configs
