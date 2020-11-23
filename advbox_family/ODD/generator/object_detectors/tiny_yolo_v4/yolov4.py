"""
Contains the code for building the tiny yolo v4 model
"""

import tensorflow as tf
from . import backbone
from . import common
from . import utils


class TinyYoloV4:
    """
    Tiny Yolo V4
    """
    def build_graph(self, image_size, num_classes):
        """
        Builds the complete graph
        :param image_size:
        :param num_classes:
        :return:
        """
        input_layer = tf.keras.layers.Input((image_size, image_size, 3))

        strides, anchors, xy_scale = utils.load_config()
        feature_maps = self.tiny_yolo_v4(input_layer, num_classes)

        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = self.decode_train(fm, image_size // 16, num_classes, strides, anchors,
                                                i, xy_scale)
            else:
                bbox_tensor = self.decode_train(fm, image_size // 32, num_classes, strides, anchors,
                                                i, xy_scale)
            bbox_tensors.append(bbox_tensor)

        model = tf.keras.Model(input_layer, bbox_tensors)

        return model

    @classmethod
    def tiny_yolo_v4(cls, input_layer, num_classes):
        """
        Builds the tiny yolo v4 base graph
        :param input_layer:
        :param num_classes:
        :return:
        """
        route_1, conv = backbone.cspdarknet53_tiny(input_layer)

        conv = common.convolutional(conv, (1, 1, 512, 256))

        conv_lobj_branch = common.convolutional(conv, (3, 3, 256, 512))
        conv_lbbox = common.convolutional(conv_lobj_branch, (1, 1, 512, 3 * (num_classes + 5)),
                                          activate=False, batch_n=False)

        conv = common.convolutional(conv, (1, 1, 256, 128))
        conv = common.upsample(conv)
        conv = tf.concat([conv, route_1], -1)

        conv_mobj_branch = common.convolutional(conv, (3, 3, 128, 256))
        conv_mbbox = common.convolutional(conv_mobj_branch, (1, 1, 256, 3 * (num_classes + 5)),
                                          activate=False, batch_n=False)

        return [conv_mbbox, conv_lbbox]

    @classmethod
    def decode_train(cls, conv_output, output_size, num_classes,
                     strides, anchors, i=0, xy_scale=[1, 1, 1]):
        """
        Transforms the network output into probabilities
        :param conv_output:
        :param output_size:
        :param num_classes:
        :param strides:
        :param anchors:
        :param i:
        :param xy_scale:
        :return:
        """
        conv_output = tf.reshape(conv_output,
                                 (tf.shape(conv_output)[0], output_size, output_size, 3,
                                  5 + num_classes))

        conv_raw_dxdy, conv_raw_dwdh, conv_raw_conf, conv_raw_prob = tf.split(
            conv_output, (2, 2, 1, num_classes), -1
        )

        xy_grid = tf.meshgrid(tf.range(output_size), tf.range(output_size))
        xy_grid = tf.expand_dims(tf.stack(xy_grid, axis=-1), axis=2)  # [gx, gy, 1, 2]
        xy_grid = tf.tile(tf.expand_dims(xy_grid, axis=0), [tf.shape(conv_output)[0], 1, 1, 3, 1])

        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = ((tf.sigmoid(conv_raw_dxdy) * xy_scale[i])
                   - 0.5 * (xy_scale[i] - 1) + xy_grid) * strides[i]
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors[i])
        pred_xywh = tf.concat([pred_xy, pred_wh], -1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return pred_xywh, pred_conf, pred_prob
