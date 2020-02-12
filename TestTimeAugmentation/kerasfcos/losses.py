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

import tensorflow as tf
import keras.backend as K


def focal(alpha=0.25, gamma=2.0):
    """
    Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(y_true, y_pred):
        """
        Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # compute the focal loss
        location_state = y_true[:, :, -1]
        labels = y_true[:, :, :-1]
        alpha_factor = K.ones_like(labels) * alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        # (1 - 0.99) ** 2 = 1e-4, (1 - 0.9) ** 2 = 1e-2
        focal_weight = tf.where(K.equal(labels, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma
        cls_loss = focal_weight * K.binary_crossentropy(labels, y_pred)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(K.equal(location_state, 1))
        normalizer = K.cast(K.shape(normalizer)[0], K.floatx())
        normalizer = K.maximum(K.cast_to_floatx(1.0), normalizer)

        return K.sum(cls_loss) / normalizer

    return _focal


def iou():
    def iou_(y_true, y_pred):
        location_state = y_true[:, :, -1]
        indices = tf.where(K.equal(location_state, 1))
        if tf.size(indices) == 0:
            return tf.constant(0.0)
        y_regr_pred = tf.gather_nd(y_pred, indices)
        y_true = tf.gather_nd(y_true, indices)
        y_regr_true = y_true[:, :4]
        y_centerness_true = y_true[:, 4]

        # (num_pos, )
        pred_left = y_regr_pred[:, 0]
        pred_top = y_regr_pred[:, 1]
        pred_right = y_regr_pred[:, 2]
        pred_bottom = y_regr_pred[:, 3]

        # (num_pos, )
        target_left = y_regr_true[:, 0]
        target_top = y_regr_true[:, 1]
        target_right = y_regr_true[:, 2]
        target_bottom = y_regr_true[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        w_intersect = tf.minimum(pred_left, target_left) + tf.minimum(pred_right, target_right)
        h_intersect = tf.minimum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        # (num_pos, )
        losses = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        losses = tf.reduce_sum(losses * y_centerness_true) / (tf.reduce_sum(y_centerness_true) + 1e-8)
        return losses

    return iou_


def bce():
    def bce_(y_true, y_pred):
        location_state = y_true[:, :, -1]
        indices = tf.where(K.equal(location_state, 1))
        if tf.size(indices) == 0:
            return tf.constant(0.0)
        y_centerness_pred = tf.gather_nd(y_pred, indices)
        y_true = tf.gather_nd(y_true, indices)
        y_centerness_true = y_true[:, 0:1]
        loss = K.switch(tf.size(y_centerness_true) > 0,
                        K.binary_crossentropy(target=y_centerness_true, output=y_centerness_pred),
                        tf.constant(0.0))
        loss = K.mean(loss)
        return loss

    return bce_
