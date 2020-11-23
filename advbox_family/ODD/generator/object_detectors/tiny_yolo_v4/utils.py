"""
Various utils to load weights and other config params
"""

import numpy as np
from .config import cfg


def load_weights(model, weights_file, model_name='yolov4', is_tiny=False):
    """
    Loads the weights for the model from the file
    :param model:
    :param weights_file:
    :param model_name:
    :param is_tiny:
    :return:
    """
    if is_tiny:
        if model_name == 'yolov3':
            layer_size = 13
            output_pos = [9, 12]
        else:
            layer_size = 21
            output_pos = [17, 20]
    else:
        if model_name == 'yolov3':
            layer_size = 75
            output_pos = [58, 66, 74]
        else:
            layer_size = 110
            output_pos = [93, 101, 109]
    weights_file = open(weights_file, 'rb')
    _, _, _, _, _ = np.fromfile(weights_file, dtype=np.int32, count=5)

    j = 0
    for i in range(layer_size):
        conv_layer_name = 'conv2d_%d' % i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' % j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in output_pos:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(weights_file, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(weights_file, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(weights_file, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in output_pos:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    # assert len(weights_file.read()) == 0, 'failed to read all data'
    weights_file.close()


def load_config():
    """
    Loads appropriate config from config.py file
    :return:
    """
    strides = np.array(cfg.YOLO.STRIDES_TINY)
    anchors = get_anchors(cfg.YOLO.ANCHORS_TINY, True)
    xyscale = cfg.YOLO.XYSCALE_TINY

    return strides, anchors, xyscale


def get_anchors(anchors_path, tiny=False):
    """
    Reshapes anchors
    :param anchors_path:
    :param tiny:
    :return:
    """
    anchors = np.array(anchors_path)
    if tiny:
        return anchors.reshape(2, 3, 2)

    return anchors.reshape(3, 3, 2)
