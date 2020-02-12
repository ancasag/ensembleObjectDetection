from keras.layers import Add, BatchNormalization, Concatenate, Conv2D, Input, Reshape
from keras.layers import Lambda, LeakyReLU, UpSampling2D, ZeroPadding2D, Activation
from keras.regularizers import l2
from keras.models import Model
from functools import reduce

from layers import FilterDetections, ClipBoxes
from losses import focal_with_mask, iou_with_mask
from yolo import config
from yolo.fsaf_layers import FSAFTarget, LevelSelect, Locations, RegressBoxes


def compose(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def darknet_conv2d(*args, **kwargs):
    """
    Wrapper to set Darknet parameters for Convolution2D.
    """
    darknet_conv_kwargs = dict({'kernel_regularizer': l2(5e-4)})
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def darknet_conv2d_bn_leaky(*args, **kwargs):
    """
    Darknet Convolution2D followed by BatchNormalization and LeakyReLU.
    """
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        darknet_conv2d(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """
    A series of resblocks starting with a downsampling Convolution2D
    """
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = darknet_conv2d_bn_leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            darknet_conv2d_bn_leaky(num_filters // 2, (1, 1)),
            darknet_conv2d_bn_leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """
    Darknet body having 52 Convolution2D layers
    """
    x = darknet_conv2d_bn_leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """
    6 conv2d_bn_leaky layers followed by a conv2d layer
    """
    x = compose(darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)))(x)
    y = compose(darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(num_classes=20, score_threshold=0.01):
    """
    Create YOLO_V3 model CNN body in Keras.

    Args:
        num_classes:
        score_threshold:

    Returns:

    """
    image_input = Input(shape=(None, None, 3), name='image_input')
    darknet = Model([image_input], darknet_body(image_input))
    ##################################################
    # build fsaf head
    ##################################################
    x, y1 = make_last_layers(darknet.output, 512, 4 + num_classes)

    x = compose(darknet_conv2d_bn_leaky(256, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, 4 + num_classes)
    x = compose(darknet_conv2d_bn_leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, 4 + num_classes)
    y1_ = Reshape((-1, 4 + num_classes))(y1)
    y2_ = Reshape((-1, 4 + num_classes))(y2)
    y3_ = Reshape((-1, 4 + num_classes))(y3)
    y = Concatenate(axis=1)([y1_, y2_, y3_])
    batch_cls_pred = Lambda(lambda x: x[..., 4:])(y)
    batch_regr_pred = Lambda(lambda x: x[..., :4])(y)
    batch_cls_pred = Activation('sigmoid')(batch_cls_pred)
    batch_regr_pred = Activation('relu')(batch_regr_pred)

    gt_boxes_input = Input(shape=(config.MAX_NUM_GT_BOXES, 5), name='gt_boxes_input')
    grid_shapes_input = Input((len(config.STRIDES), 2), dtype='int32', name='grid_shapes_input')
    batch_gt_box_levels = LevelSelect(name='level_select')(
        [batch_cls_pred, batch_regr_pred, grid_shapes_input, gt_boxes_input])
    batch_cls_target, batch_cls_mask, batch_cls_num_pos, batch_regr_target, batch_regr_mask = FSAFTarget(
        num_classes=num_classes,
        name='fsaf_target')(
        [batch_gt_box_levels, grid_shapes_input, gt_boxes_input])
    focal_loss_graph = focal_with_mask()
    iou_loss_graph = iou_with_mask()
    cls_loss = Lambda(focal_loss_graph,
                      output_shape=(1,),
                      name="cls_loss")(
        [batch_cls_target, batch_cls_pred, batch_cls_mask, batch_cls_num_pos])
    regr_loss = Lambda(iou_loss_graph,
                       output_shape=(1,),
                       name="regr_loss")([batch_regr_target, batch_regr_pred, batch_regr_mask])
    model = Model(inputs=[image_input, gt_boxes_input, grid_shapes_input],
                  outputs=[cls_loss, regr_loss],
                  name='fsaf')

    # compute the anchors
    features = [y1, y2, y3]

    locations, strides = Locations(strides=config.STRIDES)(features)

    # apply predicted regression to anchors
    boxes = RegressBoxes(name='boxes')([locations, strides, batch_regr_pred])
    boxes = ClipBoxes(name='clipped_boxes')([image_input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = FilterDetections(
        nms=True,
        class_specific_filter=True,
        name='filtered_detections',
        score_threshold=score_threshold
    )([boxes, batch_cls_pred])

    prediction_model = Model(inputs=image_input, outputs=detections, name='fsaf_detection')
    return model, prediction_model
