"""
Full implementation of the Tiny Yolo V3 model
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


class TinyYoloV3:
    """
    Encapsulates operations that need to be done to build the network.
    """

    def build_graph(self, size=None, channels=3, anchors=yolo_tiny_anchors,
                    masks=yolo_tiny_anchor_masks, classes=80, training=False):
        """
        Using the other operations, builds a Keras Model to be used

        :param size:
        :param channels:
        :param anchors:
        :param masks:
        :param classes:
        :param training:
        :return:
        """
        x = inputs = Input([size, size, channels], name='input')

        x_8, x = self.darknet_tiny(name='yolo_darknet')(x)

        x = self.yolo_conv_tiny_wrapper(256, name='yolo_conv_0')(x)
        output_0 = self.yolo_output_wrapper(256, len(masks[0]), classes, name='yolo_output_0')(x)

        x = self.yolo_conv_tiny_wrapper(128, name='yolo_conv_1')((x, x_8))
        output_1 = self.yolo_output_wrapper(128, len(masks[1]), classes, name='yolo_output_1')(x)

        if training:
            return Model(inputs, (output_0, output_1), name='yolov3')

        boxes_0 = Lambda(lambda x: self.yolo_boxes(x, anchors[masks[0]], classes),
                         name='yolo_boxes_0')(output_0)
        boxes_1 = Lambda(lambda x: self.yolo_boxes(x, anchors[masks[1]], classes),
                         name='yolo_boxes_1')(output_1)
        # outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
        #                  name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
        outputs = (boxes_0[:3], boxes_1[:3])

        return Model(inputs, outputs=outputs, name='yolov3_tiny')

    @classmethod
    def darknet_conv(cls, x, filters, size, strides=1, batch_norm=True):
        """
        Darknet base convolution

        :param x:
        :param filters:
        :param size:
        :param strides:
        :param batch_norm:
        :return:
        """
        if strides == 1:
            padding = 'same'
        else:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
            padding = 'valid'
        x = Conv2D(filters=filters, kernel_size=size,
                   strides=strides, padding=padding,
                   use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
        return x

    def darknet_tiny(self, name=None):
        """
        Series of convolutions and maxpools, according to the model definition

        :param name:
        :return:
        """
        x = inputs = Input([None, None, 3])
        x = self.darknet_conv(x, 16, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.darknet_conv(x, 32, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.darknet_conv(x, 64, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.darknet_conv(x, 128, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = x_8 = self.darknet_conv(x, 256, 3)  # skip connection
        x = MaxPool2D(2, 2, 'same')(x)
        x = self.darknet_conv(x, 512, 3)
        x = MaxPool2D(2, 1, 'same')(x)
        x = self.darknet_conv(x, 1024, 3)
        return tf.keras.Model(inputs, (x_8, x), name=name)

    def yolo_conv_tiny_wrapper(self, filters, name=None):
        """
        High-level wrapper to build the convolutions
        :param filters:
        :param name:
        :return:
        """
        def yolo_conv(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                x, x_skip = inputs

                # concat with skip connection
                x = self.darknet_conv(x, filters, 1)
                x = UpSampling2D(2)(x)
                x = Concatenate()([x, x_skip])
            else:
                x = inputs = Input(x_in.shape[1:])
                x = self.darknet_conv(x, filters, 1)

            return Model(inputs, x, name=name)(x_in)
        return yolo_conv

    def yolo_output_wrapper(self, filters, anchors, classes, name=None):
        """
        High-level wrapper for the output
        :param filters:
        :param anchors:
        :param classes:
        :param name:
        :return:
        """
        def yolo_output(x_in):
            x = inputs = Input(x_in.shape[1:])
            x = self.darknet_conv(x, filters * 2, 3)
            x = self.darknet_conv(x, anchors * (classes + 5), 1, batch_norm=False)
            x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2],
                                                anchors, classes + 5)))(x)
            return tf.keras.Model(inputs, x, name=name)(x_in)
        return yolo_output

    @classmethod
    def yolo_boxes(cls, pred, anchors, classes):
        """
        Obtains the boxes and probabilities from the network output

        :param pred:
        :param anchors:
        :param classes:
        :return:
        """
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(pred)[1:3]
        box_xy, box_wh, objectness, class_probs = tf.split(
            pred, (2, 2, 1, classes), -1)

        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_box = tf.concat((box_xy, box_wh), -1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, -1), 2)  # [gx, gy, 1, 2]

        box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
            tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * anchors

        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_box
