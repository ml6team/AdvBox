"""
Config necessary to load TF1 weights
"""

import tensorflow as tf


class TinyYOLOConfig:
    """
    Contains the config necessary to load TF1 weights into TF2
    """

    def __init__(self, dtype=tf.float32):
        self.config = {}
        self.dtype = dtype
        self.build_config()

    def build_config(self):
        """
        Generates TF2 variables for Tiny Yolo in order to load the weights from a TF1 weight file
        :return:
        """
        conv_filters = [16, 32, 64, 128, 256, 512, 1024, 1024, 1024]
        conv_size = 3
        conv_channels = [3] + conv_filters

        conv_weight_names = ['Variable_' + str(i) for i in range(0, 17, 2)]
        conv_weight_names[0] = 'Variable'

        conv_bias_names = ['Variable_' + str(i) for i in range(1, 18, 2)]

        for i, weight_name in enumerate(conv_weight_names):
            self.config[weight_name] = tf.Variable(initial_value=tf.random.truncated_normal(
                [conv_size, conv_size, conv_channels[i], conv_filters[i]],
                name=weight_name, dtype=self.dtype
            ))

        for i, bias_name in enumerate(conv_bias_names):
            self.config[bias_name] = tf.Variable(
                initial_value=tf.constant(0.1, shape=[conv_filters[i]], dtype=self.dtype),
                name=bias_name, dtype=self.dtype)

        # pylint: disable=unexpected-keyword-arg
        self.config['Variable_18'] = tf.Variable(
            initial_value=tf.random.truncated_normal([50176, 256], stddev=0.1, dtype=self.dtype),
            name='Variable_18', dtype=self.dtype
        )
        self.config['Variable_19'] = tf.Variable(
            initial_value=tf.constant(0.1, shape=[256], dtype=self.dtype), name='Variable_19',
            dtype=self.dtype
        )
        self.config['Variable_20'] = tf.Variable(
            initial_value=tf.random.truncated_normal([256, 4096], stddev=0.1, dtype=self.dtype),
            name='Variable_20', dtype=self.dtype
        )
        # pylint: enable=unexpected-keyword-arg

        self.config['Variable_21'] = tf.Variable(
            initial_value=tf.constant(0.1, shape=[4096], dtype=self.dtype), name='Variable_21',
            dtype=self.dtype
        )
        # pylint: disable=unexpected-keyword-arg
        self.config['Variable_22'] = tf.Variable(
            initial_value=tf.random.truncated_normal([4096, 1470], stddev=0.1, dtype=self.dtype),
            name='Variable_22', dtype=self.dtype
        )
        # pylint: enable=unexpected-keyword-arg
        self.config['Variable_23'] = tf.Variable(
            initial_value=tf.constant(0.1, shape=[1470], dtype=self.dtype), name='Variable_23',
            dtype=self.dtype
        )

    def get_config(self):
        """
        Returns the config
        :return:
        """
        return self.config
