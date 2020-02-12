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

import keras
import keras.backend as K
from .utils import anchors as utils_anchors
from . import utils_graph

import numpy as np
import tensorflow as tf


class Anchors(keras.layers.Layer):
    """
    Keras layer for generating anchors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """
        Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to AnchorParameters.default.ratios).
            scales: The scales of the anchors to generate (defaults to AnchorParameters.default.scales).
        """
        self.size = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios = utils_anchors.AnchorParameters.default.ratios
        elif isinstance(ratios, list):
            self.ratios = np.array(ratios)
        if scales is None:
            self.scales = utils_anchors.AnchorParameters.default.scales
        elif isinstance(scales, list):
            self.scales = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors = K.variable(utils_anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        # 一个 feature map
        feature = inputs
        feature_shape = K.shape(feature)

        # generate proposals from bbox deltas and shifted anchors
        if K.image_data_format() == 'channels_first':
            anchors = utils_graph.shift(feature_shape[2:4], self.stride, self.anchors)
        else:
            # (fh * fw * num_anchors, 4)
            anchors = utils_graph.shift(feature_shape[1:3], self.stride, self.anchors)
        # (b, fh * fw * num_anchors, 4)
        anchors = K.tile(K.expand_dims(anchors, axis=0), (feature_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            if K.image_data_format() == 'channels_first':
                total = np.prod(input_shape[2:4]) * self.num_anchors
            else:
                total = np.prod(input_shape[1:3]) * self.num_anchors

            return input_shape[0], total, 4
        else:
            return input_shape[0], None, 4

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size': self.size,
            'stride': self.stride,
            'ratios': self.ratios.tolist(),
            'scales': self.scales.tolist(),
        })

        return config


class Locations(keras.layers.Layer):
    """
    Keras layer for generating anchors for a given shape.
    """

    def __init__(self, strides, *args, **kwargs):
        """
        Initializer for an Anchors layer.

        Args
            strides: The strides mapping to the feature maps.
        """
        self.strides = strides

        super(Locations, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        feature_shapes = [K.shape(feature)[1:3] for feature in features]
        locations_per_feature = []
        for feature_shape, stride in zip(feature_shapes, self.strides):
            h = feature_shape[0]
            w = feature_shape[1]
            # [0, 8, 16]
            shifts_x = K.arange(0, w * stride, step=stride, dtype=np.float32)
            # [0, 8, 16, 24]
            shifts_y = K.arange(0, h * stride, step=stride, dtype=np.float32)
            # shape 为 (h, w)
            # shift_x 为 [[0, 8, 16], [0, 8, 16], [0, 8, 16], [0, 8, 16]
            # shift_y 为 [[0, 0, 0], [8, 8, 8], [16, 16, 16], [24, 24, 24]]
            shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
            # (h * w, )
            shift_x = K.reshape(shift_x, (-1,))
            # (h * w, )
            shift_y = K.reshape(shift_y, (-1,))
            locations = K.stack((shift_x, shift_y), axis=1) + stride // 2
            locations_per_feature.append(locations)
        # (sum(h * w), 2)
        locations = K.concatenate(locations_per_feature, axis=0)
        # (batch, sum(h * w), 2)
        locations = K.tile(K.expand_dims(locations, axis=0), (K.shape(inputs[0])[0], 1, 1))
        return locations

    def compute_output_shape(self, input_shapes):
        feature_shapes = [feature_shape[1:3] for feature_shape in input_shapes]
        total = 1
        for feature_shape in feature_shapes:
            if None not in feature_shape:
                total = total * feature_shape[0] * feature_shape[1]
            else:
                return input_shapes[0][0], None, 2
        return input_shapes[0][0], total, 2

    def get_config(self):
        config = super(Locations, self).get_config()
        config.update({
            'strides': self.strides,
        })
        return config


class UpsampleLike(keras.layers.Layer):
    """
    Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return utils_graph.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    """
    Keras layer for applying regression values to boxes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializer for the RegressBoxes layer.

        """
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        locations, regression = inputs
        x1 = locations[:, :, 0] - regression[:, :, 0]
        y1 = locations[:, :, 1] - regression[:, :, 1]
        x2 = locations[:, :, 0] + regression[:, :, 2]
        y2 = locations[:, :, 1] + regression[:, :, 3]
        # (batch_size, num_locations, 4)
        bboxes = K.stack([x1, y1, x2, y2], axis=-1)
        return bboxes

    def compute_output_shape(self, input_shape):
        return input_shape[1]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()

        return config


class ClipBoxes(keras.layers.Layer):
    """
    Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = K.cast(K.shape(image), K.floatx())
        height = shape[1]
        width = shape[2]
        x1 = tf.clip_by_value(boxes[:, :, 0], 0, width)
        y1 = tf.clip_by_value(boxes[:, :, 1], 0, height)
        x2 = tf.clip_by_value(boxes[:, :, 2], 0, width)
        y2 = tf.clip_by_value(boxes[:, :, 3], 0, height)

        return K.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def filter_detections(
        boxes,
        classification,
        centerness,
        class_specific_filter=True,
        nms=True,
        score_threshold=0.05,
        max_detections=300,
        nms_threshold=0.5
):
    """
    Filter detections using the boxes and classification values.

    Args
        boxes: Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification: Tensor of shape (num_boxes, num_classes) containing the classification scores.
        centerness: Tensor of shape (num_boxes, 1) to filter along with the boxes and classification scores.
        class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
        nms: Flag to enable/disable non maximum suppression.
        score_threshold: Threshold used to prefilter the boxes with.
        max_detections: Maximum number of detections to keep.
        nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.

    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """

    def _filter_detections(scores_, labels_):
        """
        Args:
            scores_: (num_boxes, )
            labels_: (num_boxes, )

        Returns:

        """
        # threshold based on score
        # (num_score_keeps, 1)
        indices_ = tf.where(keras.backend.greater(scores_, score_threshold))

        if nms:
            # (num_score_keeps, 4)
            filtered_boxes = tf.gather_nd(boxes, indices_)
            # In [4]: scores = np.array([0.1, 0.5, 0.4, 0.2, 0.7, 0.2])
            # In [5]: tf.greater(scores, 0.4)
            # Out[5]: <tf.Tensor: id=2, shape=(6,), dtype=bool, numpy=array([False,  True, False, False,  True, False])>
            # In [6]: tf.where(tf.greater(scores, 0.4))
            # Out[6]:
            # <tf.Tensor: id=7, shape=(2, 1), dtype=int64, numpy=
            # array([[1],
            #        [4]])>
            #
            # In [7]: tf.gather(scores, tf.where(tf.greater(scores, 0.4)))
            # Out[7]:
            # <tf.Tensor: id=15, shape=(2, 1), dtype=float64, numpy=
            # array([[0.5],
            #        [0.7]])>
            filtered_scores = keras.backend.gather(scores_, indices_)[:, 0]
            filtered_centerness = tf.gather_nd(centerness, indices_)[:, 0]
            filtered_scores = K.sqrt(filtered_scores * filtered_centerness)
            # perform NMS
            # (x1, y1, x2, y2) --> (y1, x1, y2, x2)
            filtered_boxes_2 = tf.stack([filtered_boxes[:, 1], filtered_boxes[:, 0],
                                         filtered_boxes[:, 3], filtered_boxes[:, 2]], axis=1)
            nms_indices = tf.image.non_max_suppression(filtered_boxes_2, filtered_scores, max_output_size=max_detections,
                                                       iou_threshold=nms_threshold)
            # nms_indices = tf.Print(nms_indices, [nms_indices], '\nnms_indices', summarize=1000)
            # filter indices based on NMS
            # (num_score_nms_keeps, 1)
            indices_ = keras.backend.gather(indices_, nms_indices)

        # add indices to list of all indices
        # (num_score_nms_keeps, )
        labels_ = tf.gather_nd(labels_, indices_)
        # (num_score_nms_keeps, 2)
        indices_ = keras.backend.stack([indices_[:, 0], labels_], axis=1)

        return indices_

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            # (num_boxes, )
            scores = classification[:, c]
            # (num_boxes, )
            labels = c * tf.ones((keras.backend.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        # (concatenated_num_score_nms_keeps, 2)
        indices = keras.backend.concatenate(all_indices, axis=0)
    else:
        scores = keras.backend.max(classification, axis=1)
        labels = keras.backend.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    # (m, c) * (m, 1)
    classification = classification * centerness
    classification = K.sqrt(classification)
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=keras.backend.minimum(max_detections, keras.backend.shape(scores)[0]))

    # filter input using the final set of indices
    indices = keras.backend.gather(indices[:, 0], top_indices)
    boxes = keras.backend.gather(boxes, indices)
    labels = keras.backend.gather(labels, top_indices)

    # zero pad the outputs
    pad_size = keras.backend.maximum(0, max_detections - keras.backend.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = keras.backend.cast(labels, 'int32')

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])

    return [boxes, scores, labels]


class FilterDetections(keras.layers.Layer):
    """
    Keras layer for filtering detections using score threshold and NMS.
    """

    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.05,
            max_detections=300,
            parallel_iterations=32,
            **kwargs
    ):
        """
        Filters detections using score threshold, NMS and selecting the top-k detections.

        Args
            nms: Flag to enable/disable NMS.
            class_specific_filter: Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold: Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold: Threshold used to prefilter the boxes with.
            max_detections: Maximum number of detections to keep.
            parallel_iterations: Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        Constructs the NMS graph.

        Args
            inputs : List of [boxes, classification, centerness] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        centerness = inputs[2]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes_ = args[0]
            classification_ = args[1]
            centerness_ = args[2]

            return filter_detections(
                boxes_,
                classification_,
                centerness_,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold,
            )

        # call filter_detections on each batch item
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, centerness],
            dtype=[keras.backend.floatx(), keras.backend.floatx(), 'int32'],
            parallel_iterations=self.parallel_iterations
        )

        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].

        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections),
        ]

    def compute_mask(self, inputs, mask=None):
        """
        This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms': self.nms,
            'class_specific_filter': self.class_specific_filter,
            'nms_threshold': self.nms_threshold,
            'score_threshold': self.score_threshold,
            'max_detections': self.max_detections,
            'parallel_iterations': self.parallel_iterations,
        })

        return config
