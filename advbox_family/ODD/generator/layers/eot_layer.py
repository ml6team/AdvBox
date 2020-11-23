"""
Contains layer that applies EOT
"""
import tensorflow as tf
import tensorflow_addons as tfa

import numpy as np


class EOTLayer(tf.keras.layers.Layer):
    """
    Inherits Keras Layer in order to implement a layer that does EOT
    """

    def __init__(self, punishment, smoothness_punishment, inter):
        super().__init__(dtype=tf.float32)
        self.punishment = np.array([punishment])
        self.smoothness_punishment = np.array([smoothness_punishment])

        self.inter = inter

    def call(self, inputs):
        """
        Takes an image as input, applies transformations based on inter variable, which is
        the only variable that is being changed depending on the gradient
        Structure
        - 0: image
        - 1: mask
        - 2: EOT_transforms
        :param inputs:
        :return:
        """
        image = inputs[0]
        mask = inputs[1]
        num_of_eot_transforms = len(inputs[2])
        w_image = tf.atanh(image)
        # add mask
        masked_inter = tf.multiply(mask, self.inter)

        masked_inter_batch = masked_inter

        for i in range(num_of_eot_transforms):
            if i == num_of_eot_transforms - 1:
                break

            masked_inter_batch = tf.concat([masked_inter_batch, masked_inter], 0)

        # interpolation choices "NEAREST", "BILINEAR"
        masked_inter_batch = tfa.image.transform(masked_inter_batch,
                                                 inputs[2],
                                                 interpolation='BILINEAR')

        # self.w_image [1,448,448,3] broadcast into [num_of_eot_transforms, 448, 448, 3]
        shuru = tf.add(w_image, masked_inter_batch)
        constrained = tf.tanh(shuru)

        perturbation = image - constrained
        distance_l2 = tf.norm(tensor=perturbation, ord=2)

        # non-smoothness
        sub_lala1_2 = masked_inter[0:-1, 0:-1] - masked_inter[1:, 1:]
        non_smoothness = tf.norm(tensor=sub_lala1_2, ord=2)

        partial_loss = self.punishment * distance_l2 + self.smoothness_punishment * non_smoothness

        return constrained, partial_loss
