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
import sys
sys.path.append("../")

import keras
import keras.backend as K
from .. import initializers
from .. import layers
from ..utils.anchors import AnchorParameters
from . import assert_training_model


def default_shared_model(
        pyramid_feature_size=256,
        classification_feature_size=256,
        name='shared_submodel'
):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_shared_{}'.format(i),
            kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_classification_model(
        num_classes,
        shared_model,
        pyramid_feature_size=256,
        prior_probability=0.01,
        name='classification_submodel'
):
    """
    Creates the default classification submodel.

    Args
        num_classes: Number of classes to predict a score for at each feature level.
        shared_model:
        pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
        name: The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = shared_model(inputs)

    outputs = keras.layers.Conv2D(
        filters=num_classes,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_centerness_model(
        shared_model,
        pyramid_feature_size=256,
        name='centerness_submodel'
):
    """
    Creates the default centerness submodel.

    Args
        pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
        name: The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    outputs = shared_model(inputs)

    outputs = keras.layers.Conv2D(
        filters=1,
        kernel_initializer=keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        bias_initializer='zeros',
        name='pyramid_centerness',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = keras.layers.Reshape((-1, 1), name='pyramid_centerness_reshape')(outputs)
    outputs = keras.layers.Activation('sigmoid', name='pyramid_centerness_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values=4, pyramid_feature_size=256, regression_feature_size=256,
                             name='regression_submodel'):
    """
    Creates the default regression submodel.

    Args
        num_values: Number of values to regress.
        num_anchors: Number of anchors to regress for each feature level.
        pyramid_feature_size: The number of filters to expect from the feature pyramid levels.
        regression_feature_size : The number of filters to use in the layers in the regression submodel.
        name: The name of the submodel.

    Returns
        A keras.models.Model that predicts regression values for each anchor.
    """
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': keras.initializers.normal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = keras.layers.Conv2D(num_values, name='pyramid_regression', **options)(outputs)
    # (b, num_anchors_this_feature_map, num_values)
    outputs = keras.layers.Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)
    # added for fcos
    outputs = keras.layers.Lambda(lambda x: K.exp(x))(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=256):
    """
    Creates the FPN layers on top of the backbone features.

    Args
        C3: Feature stage C3 from the backbone.
        C4: Feature stage C4 from the backbone.
        C5: Feature stage C5 from the backbone.
        feature_size: The feature size to use for the resulting feature levels.

    Returns
        A list of feature levels [P3, P4, P5, P6, P7].
    """
    # upsample C5 to get P5 from the FPN paper
    P5 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = layers.UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # add P5 elementwise to C4
    P4 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = layers.UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # add P4 elementwise to C3
    P3 = keras.layers.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return [P3, P4, P5, P6, P7]


def default_submodels(num_classes):
    """
    Create a list of default submodels used for object detection.

    The default submodels contains a regression submodel and a classification submodel.

    Args
        num_classes: Number of classes to use.

    Returns
        A list of tuple, where the first element is the name of the submodel and the second element is the submodel itself.
    """
    shared_model = default_shared_model(pyramid_feature_size=256, classification_feature_size=256)
    return [
        ('regression', default_regression_model(num_values=4)),
        ('classification', default_classification_model(num_classes=num_classes, shared_model=shared_model)),
        ('centerness', default_centerness_model(shared_model=shared_model))
    ]


def __build_model_pyramid(model_name, model, features):
    """
    Applies a single submodel to each FPN level.

    Args
        model_name: Name of the submodel.
        model: The submodel to evaluate.
        features: The FPN features. [P3, P4, P5, P6, P7]

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=model_name)([model(f) for f in features])


def __build_pyramid(models, features):
    """
    Applies all submodels to each FPN level.

    Args
        models: List of sumodels to run on each pyramid level (by default only regression, classifcation).
        features: The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(model_name, model, features) for model_name, model in models]


def __build_locations(anchor_params, features):
    """
    Builds anchors for the shape of the features from FPN.

    Args
        anchor_params: Parameters that determine how anchors are generated.
        features: The FPN features.

    Returns
        A tensor containing the anchors for the FPN features.

        The shape is:
        ```
        (batch_size, total_locations, 2)
        ```
    """
    locations = layers.Locations(
            # (8, 16, 32, 64, 128)
            strides=anchor_params.strides,
            name='locations'
    )(features)  # [P3, P4, P5, P6, P7]

    # (batch_size, total_locations, 2)
    return locations


def retinanet(
        inputs,
        backbone_layers,
        num_classes,
        create_pyramid_features=__create_pyramid_features,
        submodels=None,
        name='retinanet'
):
    """
    Construct a RetinaNet model on top of a backbone.

    This model is the minimum model necessary for training (with the unfortunate exception of anchors as output).

    Args
        inputs: keras.layers.Input (or list of) for the input to the model.
        num_classes: Number of classes to classify.
        num_anchors: Number of base anchors.
        create_pyramid_features : Functor for creating pyramid features given the features C3, C4, C5 from the backbone.
        submodels: Submodels to run on each feature map (default is regression and classification submodels).
        name: Name of the model.

    Returns
        A keras.models.Model which takes an image as input and outputs generated anchors and the result from each submodel on every pyramid level.

        The order of the outputs is as defined in submodels:
        ```
        [
            regression, classification, other[0], other[1], ...
        ]
        ```
    """
    if submodels is None:
        submodels = default_submodels(num_classes)

    C3, C4, C5 = backbone_layers

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    # [P3, P4, P5, P6, P7]
    features = create_pyramid_features(C3, C4, C5)

    # for all pyramid levels, run available submodels
    # [(b, sum(fh * fw), 4), (b, sum(fh * fw), num_classes), (b, sum(fh * fw)]
    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


def retinanet_bbox(
        model=None,
        nms=True,
        class_specific_filter=True,
        name='retinanet-bbox',
        anchor_params=None,
        **kwargs
):
    """
    Construct a RetinaNet model on top of a backbone and adds convenience functions to output boxes directly.

    This model uses the minimum retinanet model and appends a few layers to compute boxes within the graph.
    These layers include applying the regression values to the anchors and performing NMS.

    Args
        model: RetinaNet model to append bbox layers to. If None, it will create a RetinaNet model using **kwargs.
        nms: Whether to use non-maximum suppression for the filtering step.
        class_specific_filter: Whether to use class specific filtering or filter for the best scoring class only.
        name: Name of the model.
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        *kwargs: Additional kwargs to pass to the minimal retinanet model.

    Returns
        A keras.models.Model which takes an image as input and outputs the detections on the image.

        The order is defined as follows:
        ```
        [
            boxes, scores, labels, other[0], other[1], ...
        ]
        ```
    """

    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    assert_training_model(model)

    # compute the anchors
    features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    # we expect the anchors, regression and classification values as first output
    # (b, m, 4)
    regression = model.outputs[0]
    # (b, m, num_classes)
    classification = model.outputs[1]
    # (b, m, 1)
    centerness = model.outputs[2]
    # (b, m, 2)
    locations = __build_locations(anchor_params, features)
    # return keras.models.Model(inputs=model.inputs, outputs=locations, name=name)
    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([locations, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections'
    )([boxes, classification, centerness])

    # construct the model
    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
