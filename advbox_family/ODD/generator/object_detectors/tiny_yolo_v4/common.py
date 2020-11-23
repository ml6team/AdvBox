"""
Contains common neural operations (convolutions, residuals etc.)
"""

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization as Keras_BatchNormalization


class BatchNormalization(Keras_BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


def convolutional(input_layer, filters_shape, downsample=False, activate=True, batch_n=True,
                  activate_type='leaky'):
    """
    Custom conv Layer

    :param input_layer:
    :param filters_shape:
    :param downsample:
    :param activate:
    :param batch_n:
    :param activate_type:
    :return:
    """
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0],
                                  strides=strides, padding=padding,
                                  use_bias=not batch_n,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    if batch_n:
        conv = BatchNormalization()(conv)

    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv


def mish(x):
    """
    Mish activation function
    :param x:
    :return:
    """
    return x * tf.math.tanh(tf.math.softplus(x))


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    """
    Residual block
    :param input_layer:
    :param input_channel:
    :param filter_num1:
    :param filter_num2:
    :param activate_type:
    :return:
    """
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1),
                         activate_type=activate_type)
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2),
                         activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output


def route_group(input_layer, groups, group_id):
    """
    Group routes
    :param input_layer:
    :param groups:
    :param group_id:
    :return:
    """
    convs = tf.split(input_layer, groups, -1)
    return convs[group_id]


def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2),
                           method='bilinear')
