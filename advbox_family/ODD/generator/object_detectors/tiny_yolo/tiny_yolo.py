"""
Contains the implementation of the Tiny Yolo model, courtesy of gliese581gg
https://github.com/gliese581gg/YOLO_tensorflow
"""
from __future__ import division
from __future__ import print_function
from builtins import str
import tensorflow as tf
import numpy as np


class TinyYOLO:
    """
    Class containing the methods required to build the Tiny Yolo model
    """

    ADVERSARIAL_OBJECT_INDEX: int = 14  # Person

    model_input = None
    mode = None
    disp_console = None
    model_output = None
    yolo_variables = None
    model = None

    sess = None
    weights_file = '../weights/YOLO_tiny.ckpt'

    alpha = 0.1

    def __init__(self, init_dict):
        self.all_variables = init_dict['all_variables']
        self.iou_threshold = init_dict['iou_threshold']
        self.threshold = init_dict['threshold']
        self.classes = init_dict['classes']
        self.minimise = init_dict['minimise']

    def build_graph(self, input_image):
        """
        Builds the layers of the network
        :param input_image:
        :return:
        """
        net = self.conv_layer(1, input_image, 3, 1, 'Variable:0', 'Variable_1:0')
        net = self.pooling_layer(2, net, 2, 2)
        net = self.conv_layer(3, net, 3, 1, 'Variable_2:0', 'Variable_3:0')
        net = self.pooling_layer(4, net, 2, 2)
        net = self.conv_layer(5, net, 3, 1, 'Variable_4:0', 'Variable_5:0')
        net = self.pooling_layer(6, net, 2, 2)
        net = self.conv_layer(7, net, 3, 1, 'Variable_6:0', 'Variable_7:0')
        net = self.pooling_layer(8, net, 2, 2)
        net = self.conv_layer(9, net, 3, 1, 'Variable_8:0', 'Variable_9:0')
        net = self.pooling_layer(10, net, 2, 2)
        net = self.conv_layer(11, net, 3, 1, 'Variable_10:0', 'Variable_11:0')
        net = self.pooling_layer(12, net, 2, 2)
        net = self.conv_layer(13, net, 3, 1, 'Variable_12:0', 'Variable_13:0')
        net = self.conv_layer(14, net, 3, 1, 'Variable_14:0', 'Variable_15:0')
        net = self.conv_layer(15, net, 3, 1, 'Variable_16:0', 'Variable_17:0')
        net = self.fc_layer(16, net, 'Variable_18:0', 'Variable_19:0', flat=True, linear=False)
        net = self.fc_layer(17, net, 'Variable_20:0', 'Variable_21:0', flat=False,
                            linear=False)
        # skip dropout_18
        fc_19 = self.fc_layer(19, net, 'Variable_22:0', 'Variable_23:0', flat=False,
                              linear=True)

        conf = tf.reshape(fc_19[:, 0:980], (fc_19[:, 0:980].shape.as_list()[0], 7, 7, 20))
        scale = tf.reshape(fc_19[:, 980:1078], (fc_19[:, 980:1078].shape.as_list()[0], 7, 7, 2))

        prob1 = tf.multiply(conf[:, :, :, self.ADVERSARIAL_OBJECT_INDEX], scale[:, :, :, 0])
        prob2 = tf.multiply(conf[:, :, :, self.ADVERSARIAL_OBJECT_INDEX], scale[:, :, :, 1])

        final_p = tf.stack([prob1, prob2], axis=0)

        if self.minimise:
            batch_p = tf.reduce_max(input_tensor=final_p, axis=2)
        else:
            batch_p = tf.reduce_max(input_tensor=final_p, axis=[0, 2, 3])

        c_target = tf.reduce_sum(input_tensor=batch_p)

        return c_target

    def conv_layer(self,
                   idx,
                   inputs,
                   size,
                   stride,
                   weight_name,
                   biases_name):
        """
        Builds the convolutional layer (using low-level operations). Uses the pre-loaded variables
        :param idx:
        :param inputs:
        :param size:
        :param stride:
        :param weight_name:
        :param biases_name:
        :return:
        """
        pad_size = size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        inputs_pad = tf.pad(inputs, pad_mat, constant_values=0)

        conv = tf.nn.conv2d(inputs_pad, self.all_variables[weight_name[:-2]],
                            [1, stride, stride, 1], 'VALID',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, self.all_variables[biases_name[:-2]],
                             name=str(idx) + '_conv_biased')

        return tf.maximum(self.alpha * conv_biased, conv_biased, name=str(idx) + '_leaky_relu')

    @classmethod
    def pooling_layer(cls,
                      idx,
                      inputs,
                      size,
                      stride
                      ):
        """
        Builds the pooling layer (using low-level operations)
        :param idx:
        :param inputs:
        :param size:
        :param stride:
        :return:
        """
        return tf.nn.max_pool2d(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1],
                                padding='SAME',
                                name=str(idx) + '_pool')

    def fc_layer(self,
                 idx,
                 inputs,
                 weight_name,
                 biases_name,
                 flat=False,
                 linear=False):
        """
        Builds the fully connected layer (using low-level operations)
        :param idx:
        :param inputs:
        :param weight_name:
        :param biases_name:
        :param flat:
        :param linear:
        :return:
        """
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_transposed = tf.transpose(a=inputs, perm=(0, 3, 1, 2))
            inputs_processed = tf.reshape(inputs_transposed, [-1, dim])
        else:
            inputs_processed = inputs

        if linear:
            return tf.add(tf.matmul(
                inputs_processed, self.all_variables[weight_name[:-2]]),
                self.all_variables[biases_name[:-2]],
                name=str(idx) + '_fc'
            )

        final = tf.add(tf.matmul(inputs_processed, self.all_variables[weight_name[:-2]]),
                       self.all_variables[biases_name[:-2]])
        return tf.maximum(self.alpha * final, final, name=str(idx) + '_fc')
