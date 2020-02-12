"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras.backend as K
import tensorflow as tf


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """
    Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes: np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean: The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std: The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = K.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size.

    Args
        shape: Shape to shift the anchors over. (h,w)
        stride: Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.

    Returns
        shifted_anchors: (fh * fw * num_anchors, 4)
    """
    shift_x = (K.arange(0, shape[1], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
    shift_y = (K.arange(0, shape[0], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = K.reshape(shift_x, [-1])
    shift_y = K.reshape(shift_y, [-1])

    # (4, fh * fw)
    shifts = K.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)
    # (fh * fw, 4)
    shifts = K.transpose(shifts)
    number_anchors = K.shape(anchors)[0]

    # number of base points = fh * fw
    k = K.shape(shifts)[0]

    # (k=fh*fw, num_anchors, 4)
    shifted_anchors = K.reshape(anchors, [1, number_anchors, 4]) + K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
    # (k * num_anchors, 4)
    shifted_anchors = K.reshape(shifted_anchors, [k * number_anchors, 4])

    return shifted_anchors


def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)
