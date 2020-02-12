from keras.layers import Layer
import tensorflow as tf
from .util_graphs import trim_zeros_graph, prop_box_graph, prop_box_graph_2
import keras.backend as K
from .losses import focal, iou
from .configure import MAX_NUM_GT_BOXES, STRIDES, POS_SCALE, IGNORE_SCALE


def level_select(cls_pred, regr_pred, gt_boxes, feature_shapes, strides, pos_scale=0.2):
    """

    Args:
        cls_pred: (sum(fh * fw), num_classes)
        regr_pred:  (sum(fh * fw), 4)
        gt_boxes:  (MAX_NUM_GT_BOXES, 5)
        feature_shapes: (5, 2)
        strides:
        pos_scale:

    Returns:

    """
    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)
    gt_boxes = gt_boxes[:, :4]
    focal_loss = focal()
    iou_loss = iou()
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes)
    num_gt_boxes = tf.shape(gt_boxes)[0]
    gt_labels = tf.boolean_mask(gt_labels, non_zeros)
    level_losses = []
    for level_id in range(len(strides)):
        stride = strides[level_id]
        fh = feature_shapes[level_id][0]
        fw = feature_shapes[level_id][1]
        fa = tf.reduce_prod(feature_shapes, axis=-1)
        start_idx = tf.reduce_sum(fa[:level_id])
        end_idx = start_idx + fh * fw
        cls_pred_i = tf.reshape(cls_pred[start_idx:end_idx], (fh, fw, tf.shape(cls_pred)[-1]))
        regr_pred_i = tf.reshape(regr_pred[start_idx:end_idx], (fh, fw, tf.shape(regr_pred)[-1]))
        proj_boxes = gt_boxes / stride
        x1, y1, x2, y2 = prop_box_graph(proj_boxes, pos_scale, fw, fh)

        def compute_gt_box_loss(args):
            x1_ = args[0]
            y1_ = args[1]
            x2_ = args[2]
            y2_ = args[3]
            gt_box = args[4]
            gt_label = args[5]
            locs_cls_pred_i = cls_pred_i[y1_:y2_, x1_:x2_, :]
            locs_cls_pred_i = tf.reshape(locs_cls_pred_i, (-1, tf.shape(locs_cls_pred_i)[-1]))
            locs_cls_true_i = tf.zeros_like(locs_cls_pred_i)
            gt_label_col = tf.ones_like(locs_cls_true_i[:, 0:1])
            locs_cls_true_i = tf.concat([locs_cls_true_i[:, :gt_label],
                                         gt_label_col,
                                         locs_cls_true_i[:, gt_label + 1:],
                                         ], axis=-1)
            loss_cls = focal_loss(K.expand_dims(locs_cls_true_i, axis=0), K.expand_dims(locs_cls_pred_i, axis=0))
            locs_regr_pred_i = regr_pred_i[y1_:y2_, x1_:x2_, :]
            locs_regr_pred_i = tf.reshape(locs_regr_pred_i, (-1, tf.shape(locs_regr_pred_i)[-1]))

            locs_x = K.arange(x1_, x2_, dtype=tf.float32)
            locs_y = K.arange(y1_, y2_, dtype=tf.float32)
            shift_x = (locs_x + 0.5) * stride
            shift_y = (locs_y + 0.5) * stride
            shift_xx, shift_yy = tf.meshgrid(shift_x, shift_y)
            shift_xx = tf.reshape(shift_xx, (-1,))
            shift_yy = tf.reshape(shift_yy, (-1,))
            shifts = K.stack((shift_xx, shift_yy, shift_xx, shift_yy), axis=-1)
            l = shifts[:, 0] - gt_box[0]
            t = shifts[:, 1] - gt_box[1]
            r = gt_box[2] - shifts[:, 2]
            b = gt_box[3] - shifts[:, 3]
            locs_regr_true_i = tf.stack([l, t, r, b], axis=-1)
            locs_regr_true_i /= 4.0
            loss_regr = iou_loss(K.expand_dims(locs_regr_true_i, axis=0), K.expand_dims(locs_regr_pred_i, axis=0))
            return loss_cls + loss_regr

        level_loss = tf.map_fn(
            compute_gt_box_loss,
            elems=[x1, y1, x2, y2, gt_boxes, gt_labels],
            dtype=tf.float32
        )
        level_losses.append(level_loss)
    losses = tf.stack(level_losses, axis=-1)
    gt_box_levels = tf.argmin(losses, axis=-1)
    padding_gt_box_levels = tf.ones((MAX_NUM_GT_BOXES - num_gt_boxes), dtype=tf.int64) * -1
    gt_box_levels = tf.concat([gt_box_levels, padding_gt_box_levels], axis=0)
    return gt_box_levels


class LevelSelect(Layer):
    def __init__(self, **kwargs):
        super(LevelSelect, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        batch_cls_pred = inputs[0]
        batch_regr_pred = inputs[1]
        feature_shapes = inputs[2][0]
        batch_gt_boxes = inputs[3]

        def _level_select(args):
            cls_pred = args[0]
            regr_pred = args[1]
            gt_boxes = args[2]

            return level_select(
                cls_pred,
                regr_pred,
                gt_boxes,
                feature_shapes=feature_shapes,
                strides=STRIDES,
                pos_scale=POS_SCALE
            )

        outputs = tf.map_fn(
            _level_select,
            elems=[batch_cls_pred, batch_regr_pred, batch_gt_boxes],
            dtype=tf.int64,
        )
        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of shapes of [batch_cls_pred, batch_regr_pred, feature_shapes, batch_gt_boxes].

        Returns
            shape of batch_gt_box_levels
        """
        # return input_shape[0][0], config.MAX_NUM_GT_BOXES
        return input_shape[0][0], None

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(LevelSelect, self).get_config()
        return config


def build_fsaf_target(gt_box_levels, gt_boxes, feature_shapes, num_classes, strides, pos_scale, ignore_scale):
    gt_labels = tf.cast(gt_boxes[:, 4], tf.int32)
    gt_boxes = gt_boxes[:, :4]
    cls_target = tf.zeros((0, num_classes))
    cls_mask = tf.zeros((0,), dtype=tf.bool)
    cls_num_pos = tf.zeros((0,))
    regr_target = tf.zeros((0, 4))
    regr_mask = tf.zeros((0,), dtype=tf.bool)
    for level_id in range(len(strides)):
        feature_shape = feature_shapes[level_id]
        stride = strides[level_id]
        fh = feature_shape[0]
        fw = feature_shape[1]
        level_gt_box_indices = tf.where(tf.equal(gt_box_levels, level_id))

        def do_level_has_gt_boxes():
            level_gt_boxes = tf.gather(gt_boxes, level_gt_box_indices[:, 0])
            level_proj_boxes = level_gt_boxes / stride
            level_gt_labels = tf.gather_nd(gt_labels, level_gt_box_indices)
            ign_x1, ign_y1, ign_x2, ign_y2 = prop_box_graph_2(level_proj_boxes, ignore_scale, fw, fh)
            pos_x1, pos_y1, pos_x2, pos_y2 = prop_box_graph_2(level_proj_boxes, pos_scale, fw, fh)

            def build_single_gt_box_fsaf_target(args):
                ign_x1_ = args[0]
                ign_y1_ = args[1]
                ign_x2_ = args[2]
                ign_y2_ = args[3]
                pos_x1_ = args[4]
                pos_y1_ = args[5]
                pos_x2_ = args[6]
                pos_y2_ = args[7]
                gt_box = args[8]
                gt_label = args[9]
                level_box_cls_target = tf.zeros((pos_y2_[0] - pos_y1_[0], pos_x2_[0] - pos_x1_[0], num_classes))
                level_box_gt_label_col = tf.ones((pos_y2_[0] - pos_y1_[0], pos_x2_[0] - pos_x1_[0], 1))
                level_box_cls_target = tf.concat((level_box_cls_target[..., :gt_label],
                                                  level_box_gt_label_col,
                                                  level_box_cls_target[..., gt_label + 1:]), axis=-1)
                level_box_cls_pos_mask = tf.ones((pos_y2_[0] - pos_y1_[0], pos_x2_[0] - pos_x1_[0])) * 2.
                ign_top_bot = tf.concat((pos_y1_ - ign_y1_, ign_y2_ - pos_y2_), axis=0)
                ign_lef_rit = tf.concat((pos_x1_ - ign_x1_, ign_x2_ - pos_x2_), axis=0)
                ign_pad = tf.stack([ign_top_bot, ign_lef_rit], axis=0)
                level_box_cls_ign_mask = tf.pad(level_box_cls_pos_mask, ign_pad)
                other_top_bot = tf.concat((ign_y1_, fh - ign_y2_), axis=0)
                other_lef_rit = tf.concat((ign_x1_, fw - ign_x2_), axis=0)
                other_pad = tf.stack([other_top_bot, other_lef_rit], axis=0)
                level_box_cls_mask = tf.pad(level_box_cls_ign_mask, other_pad, constant_values=-1.)
                level_box_cls_target = tf.pad(level_box_cls_target,
                                              tf.concat((ign_pad + other_pad, tf.constant([[0, 0]])), axis=0))
                locs_x = K.arange(pos_x1_[0], pos_x2_[0], dtype=tf.float32)
                locs_y = K.arange(pos_y1_[0], pos_y2_[0], dtype=tf.float32)
                shift_x = (locs_x + 0.5) * stride
                shift_y = (locs_y + 0.5) * stride
                shift_xx, shift_yy = tf.meshgrid(shift_x, shift_y)
                shifts = K.stack((shift_xx, shift_yy, shift_xx, shift_yy), axis=-1)
                l = shifts[:, :, 0] - gt_box[0]
                t = shifts[:, :, 1] - gt_box[1]
                r = gt_box[2] - shifts[:, :, 2]
                b = gt_box[3] - shifts[:, :, 3]
                deltas = K.stack((l, t, r, b), axis=-1)
                level_box_regr_pos_target = deltas / 4.0
                level_box_regr_pos_mask = tf.ones((pos_y2_[0] - pos_y1_[0], pos_x2_[0] - pos_x1_[0]))
                level_box_regr_mask = tf.pad(level_box_regr_pos_mask, ign_pad + other_pad)
                level_box_regr_target = tf.pad(level_box_regr_pos_target,
                                               tf.concat((ign_pad + other_pad, tf.constant([[0, 0]])), axis=0))
                level_box_pos_area = (l + r) * (t + b)
                level_box_area = tf.pad(level_box_pos_area, ign_pad + other_pad, constant_values=1e7)
                return level_box_cls_target, level_box_cls_mask, level_box_regr_target, level_box_regr_mask, level_box_area

            level_cls_target, level_cls_mask, level_regr_target, level_regr_mask, level_area = tf.map_fn(
                build_single_gt_box_fsaf_target,
                elems=[
                    ign_x1, ign_y1, ign_x2, ign_y2,
                    pos_x1, pos_y1, pos_x2, pos_y2,
                    level_gt_boxes, level_gt_labels],
                dtype=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32)
            )
            level_min_area_box_indices = tf.argmin(level_area, axis=0, output_type=tf.int32)
            level_min_area_box_indices = tf.reshape(level_min_area_box_indices, (-1,))
            locs_x = K.arange(0, fw)
            locs_y = K.arange(0, fh)
            locs_xx, locs_yy = tf.meshgrid(locs_x, locs_y)
            locs_xx = tf.reshape(locs_xx, (-1,))
            locs_yy = tf.reshape(locs_yy, (-1,))
            level_indices = tf.stack((level_min_area_box_indices, locs_yy, locs_xx), axis=-1)
            level_cls_target_ = tf.gather_nd(level_cls_target, level_indices)
            level_regr_target_ = tf.gather_nd(level_regr_target, level_indices)
            level_cls_num_pos_ = tf.reduce_sum(tf.cast(tf.equal(tf.reduce_max(level_cls_mask, axis=0), 2), tf.float32))
            level_cls_mask = tf.equal(tf.reduce_max(level_cls_mask, axis=0), 2) | tf.equal(
                tf.reduce_max(level_cls_mask, axis=0),
                -1)
            level_cls_mask_ = tf.reshape(level_cls_mask, (fh * fw,))
            level_regr_mask = tf.reduce_sum(level_regr_mask, axis=0) > 0
            level_regr_mask_ = tf.reshape(level_regr_mask, (fh * fw,))
            return level_cls_target_, level_cls_mask_, level_cls_num_pos_, level_regr_target_, level_regr_mask_

        def do_level_has_no_gt_boxes():
            level_cls_target_ = tf.zeros((fh * fw, num_classes))
            level_cls_mask_ = tf.ones((fh * fw,), dtype=tf.bool)
            level_cls_num_pos_ = tf.zeros(())
            level_regr_target_ = tf.zeros((fh * fw, 4))
            level_regr_mask_ = tf.zeros((fh * fw,), dtype=tf.bool)
            return level_cls_target_, level_cls_mask_, level_cls_num_pos_, level_regr_target_, level_regr_mask_

        level_cls_target, level_cls_mask, level_cls_num_pos, level_regr_target, level_regr_mask = tf.cond(
            tf.equal(tf.size(level_gt_box_indices), 0),
            do_level_has_no_gt_boxes,
            do_level_has_gt_boxes)

        cls_target = tf.concat([cls_target, level_cls_target], axis=0)
        cls_mask = tf.concat([cls_mask, level_cls_mask], axis=0)
        cls_num_pos = tf.concat([cls_num_pos, level_cls_num_pos[None]], axis=0)
        regr_target = tf.concat([regr_target, level_regr_target], axis=0)
        regr_mask = tf.concat([regr_mask, level_regr_mask], axis=0)
    cls_num_pos = tf.reduce_sum(cls_num_pos)
    return [cls_target, cls_mask, cls_num_pos, regr_target, regr_mask]


class FSAFTarget(Layer):
    def __init__(self, num_classes, **kwargs):
        super(FSAFTarget, self).__init__(**kwargs)
        self.num_classes = num_classes

    def call(self, inputs, **kwargs):
        batch_gt_box_levels = inputs[0]
        feature_shapes = inputs[1][0]
        batch_gt_boxes = inputs[2]

        def _build_fsaf_target(args):
            gt_box_levels = args[0]
            gt_boxes = args[1]

            return build_fsaf_target(
                gt_box_levels,
                gt_boxes,
                feature_shapes=feature_shapes,
                num_classes=self.num_classes,
                strides=STRIDES,
                pos_scale=POS_SCALE,
                ignore_scale=IGNORE_SCALE,
            )

        outputs = tf.map_fn(
            _build_fsaf_target,
            elems=[batch_gt_box_levels, batch_gt_boxes],
            dtype=[tf.float32, tf.bool, tf.float32, tf.float32, tf.bool],
        )
        return outputs

    def compute_output_shape(self, input_shape):
        """
        Computes the output shapes given the input shapes.

        Args
            input_shape : List of shapes of [batch_gt_box_levels, feature_shapes, batch_gt_boxes].

        Returns
            List of tuples representing the shapes of [batch_cls_target, batch_cls_mask, batch_num_pos, batch_regr_target, batch_regr_mask]
        """
        batch_size = input_shape[0][0]
        return [[batch_size, None, self.num_classes], [batch_size, None], [batch_size, ], [batch_size, None, 4],
                [batch_size, None]]

    def get_config(self):
        """
        Gets the configuration of this layer.

        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FSAFTarget, self).get_config()
        config.update({'num_classes': self.num_classes})
        return config


class Locations(Layer):
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

        super(Locations, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        feature_shapes = [tf.shape(feature)[1:3] for feature in features]
        locations_per_feature = []
        strides_per_feature = []
        for feature_shape, stride in zip(feature_shapes, self.strides):
            fh = feature_shape[0]
            fw = feature_shape[1]
            shifts_x = K.arange(0, fw * stride, step=stride, dtype=tf.float32)
            shifts_y = K.arange(0, fh * stride, step=stride, dtype=tf.float32)
            shift_x, shift_y = tf.meshgrid(shifts_x, shifts_y)
            shift_x = K.reshape(shift_x, (-1,))
            shift_y = K.reshape(shift_y, (-1,))
            locations = K.stack((shift_x, shift_y), axis=1) + stride // 2
            locations_per_feature.append(locations)

            strides = tf.ones((fh, fw)) * stride
            strides = tf.reshape(strides, (-1,))
            strides_per_feature.append(strides)
        locations = K.concatenate(locations_per_feature, axis=0)
        locations = tf.tile(tf.expand_dims(locations, axis=0), (tf.shape(inputs[0])[0], 1, 1))
        strides = tf.concat(strides_per_feature, axis=0)
        strides = tf.tile(tf.expand_dims(strides, axis=0), (tf.shape(inputs[0])[0], 1))
        return [locations, strides]

    def compute_output_shape(self, input_shapes):
        feature_shapes = [feature_shape[1:3] for feature_shape in input_shapes]
        total = 1
        for feature_shape in feature_shapes:
            if None not in feature_shape:
                total = total * feature_shape[0] * feature_shape[1]
            else:
                return [[input_shapes[0][0], None, 2], [input_shapes[0][0], None]]
        return [[input_shapes[0][0], total, 2], [input_shapes[0][0], total]]

    def get_config(self):
        base_config = super(Locations, self).get_config()
        base_config.update({'strides': self.strides})
        return base_config


class RegressBoxes(Layer):
    """
    Keras layer for applying regression values to boxes.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializer for the RegressBoxes layer.

        """
        super(RegressBoxes, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        locations, strides, regression = inputs
        x1 = locations[:, :, 0] - regression[:, :, 0] * 4.0
        y1 = locations[:, :, 1] - regression[:, :, 1] * 4.0
        x2 = locations[:, :, 0] + regression[:, :, 2] * 4.0
        y2 = locations[:, :, 1] + regression[:, :, 3] * 4.0
        bboxes = K.stack([x1, y1, x2, y2], axis=-1)
        return bboxes

    def compute_output_shape(self, input_shape):
        return input_shape[2]

    def get_config(self):
        base_config = super(RegressBoxes, self).get_config()

        return base_config
