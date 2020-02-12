import tensorflow as tf
import keras.backend as K


def xyxy2cxcywh(xyxy):
    """
    Convert [x1 y1 x2 y2] box format to [cx cx w h] format.
    """
    return tf.concat((0.5 * (xyxy[:, 0:2] + xyxy[:, 2:4]), xyxy[:, 2:4] - xyxy[:, 0:2]), axis=-1)


def cxcywh2xyxy(xywh):
    """
    Convert [cx cy w y] box format to [x1 y1 x2 y2] format.
    """
    return tf.concat((xywh[:, 0:2] - 0.5 * xywh[:, 2:4], xywh[:, 0:2] + 0.5 * xywh[:, 2:4]), axis=-1)


def prop_box_graph(boxes, scale, width, height):
    """
    Compute proportional box coordinates.

    Box centers are fixed. Box w and h scaled by scale.
    """
    prop_boxes = xyxy2cxcywh(boxes)
    prop_boxes = tf.concat((prop_boxes[:, :2], prop_boxes[:, 2:] * scale), axis=-1)
    prop_boxes = cxcywh2xyxy(prop_boxes)
    x1 = tf.floor(prop_boxes[:, 0])
    y1 = tf.floor(prop_boxes[:, 1])
    x2 = tf.ceil(prop_boxes[:, 2])
    y2 = tf.ceil(prop_boxes[:, 3])
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    x2 = tf.cast(tf.clip_by_value(x2, 1, width - 1), tf.int32)
    y2 = tf.cast(tf.clip_by_value(y2, 1, height - 1), tf.int32)
    x1 = tf.cast(tf.clip_by_value(x1, 0, tf.cast(x2, tf.float32) - 1), tf.int32)
    y1 = tf.cast(tf.clip_by_value(y1, 0, tf.cast(y2, tf.float32) - 1), tf.int32)

    return x1, y1, x2, y2


def prop_box_graph_2(boxes, scale, width, height):
    """
    Compute proportional box coordinates.

    Box centers are fixed. Box w and h scaled by scale.
    """
    prop_boxes = xyxy2cxcywh(boxes)
    prop_boxes = tf.concat((prop_boxes[:, :2], prop_boxes[:, 2:] * scale), axis=-1)
    prop_boxes = cxcywh2xyxy(prop_boxes)
    # (n, 1)
    x1 = tf.floor(prop_boxes[:, 0:1])
    y1 = tf.floor(prop_boxes[:, 1:2])
    x2 = tf.ceil(prop_boxes[:, 2:3])
    y2 = tf.ceil(prop_boxes[:, 3:4])
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)
    x2 = tf.cast(tf.clip_by_value(x2, 1, width - 1), tf.int32)
    y2 = tf.cast(tf.clip_by_value(y2, 1, height - 1), tf.int32)
    x1 = tf.cast(tf.clip_by_value(x1, 0, tf.cast(x2, tf.float32) - 1), tf.int32)
    y1 = tf.cast(tf.clip_by_value(y1, 0, tf.cast(y2, tf.float32) - 1), tf.int32)

    return x1, y1, x2, y2


def trim_zeros_graph(boxes, name='trim_zeros'):
    """
    Often boxes are represented with matrices of shape [N, 4] and are padded with zeros.
    This removes zero boxes.

    Args:
        boxes: [N, 4] matrix of boxes.
        name: name of tensor

    Returns:

    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


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
