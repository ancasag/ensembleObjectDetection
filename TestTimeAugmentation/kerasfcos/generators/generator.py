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

import numpy as np
import random
import warnings

import keras

from utils.anchors import (
    anchors_for_shape,
    guess_shapes,
    compute_locations,
    compute_interest_sizes,
)
from utils.config import parse_anchor_parameters
from utils.image import (
    TransformParameters,
    adjust_transform_for_image,
    apply_transform,
    preprocess_image,
    resize_image,
)
from utils.transform import transform_aabb


class Generator(keras.utils.Sequence):
    """
    Abstract generator class.
    """

    def __init__(
            self,
            transform_generator=None,
            visual_effect_generator=None,
            batch_size=1,
            group_method='ratio',  # one of 'none', 'random', 'ratio'
            shuffle_groups=True,
            image_min_side=800,
            image_max_side=1333,
            transform_parameters=None,
            compute_shapes=guess_shapes,
            compute_locations=compute_locations,
            compute_interest_sizes=compute_interest_sizes,
            preprocess_image=preprocess_image,
            config=None
    ):
        """
        Initialize Generator object.

        Args
            transform_generator: A generator used to randomly transform images and annotations.
            batch_size: The size of the batches to generate.
            group_method: Determines how images are grouped together (defaults to 'ratio', one of ('none', 'random', 'ratio')).
            shuffle_groups: If True, shuffles the groups each epoch.
            image_min_side: After resizing the minimum side of an image is equal to image_min_side.
            image_max_side: If after resizing the maximum side is larger than image_max_side, scales down further so that the max side is equal to image_max_side.
            transform_parameters: The transform parameters used for data augmentation.
            compute_shapes: Function handler for computing the shapes of the pyramid for a given input.
            compute_locations: Function handler for computing center point of grid cells in all feature map
            compute_interest_sizes: Function handler for computing size limit for each location
            preprocess_image: Function handler for preprocessing an image (scaling / normalizing) for passing through a network.
        """
        self.transform_generator = transform_generator
        self.visual_effect_generator = visual_effect_generator
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.image_min_side = image_min_side
        self.image_max_side = image_max_side
        self.image_min_sides = (600, 700, 800, 900, 1000)
        self.image_max_sides = (1000, 1166, 1333, 1500, 1666)
        self.transform_parameters = transform_parameters or TransformParameters()
        self.compute_shapes = compute_shapes
        self.compute_locations = compute_locations
        self.compute_interest_sizes = compute_interest_sizes
        self.preprocess_image = preprocess_image
        self.config = config
        self.groups = None
        self.current_index = 0

        # Define groups
        self.group_images()

        # Shuffle when initializing
        if self.shuffle_groups:
            random.shuffle(self.groups)

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)
        self.current_index = 0

    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations,
                               dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(
                type(annotations))
            assert (
                    'labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert (
                    'bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)
            if annotations['bboxes'].shape[0] == 0:
                warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
                    group[index],
                    image.shape,
                ))

        return image_group, annotations_group

    def clip_transformed_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        filtered_image_group = []
        filtered_annotations_group = []
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image_height = image.shape[0]
            image_width = image.shape[1]
            # x1
            annotations['bboxes'][:, 0] = np.clip(annotations['bboxes'][:, 0], 0, image_width - 2)
            # y1
            annotations['bboxes'][:, 1] = np.clip(annotations['bboxes'][:, 1], 0, image_height - 2)
            # x2
            annotations['bboxes'][:, 2] = np.clip(annotations['bboxes'][:, 2], 1, image_width - 1)
            # y2
            annotations['bboxes'][:, 3] = np.clip(annotations['bboxes'][:, 3], 1, image_height - 1)
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            small_indices = np.where(
                (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0] < 15) |
                (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1] < 15)
            )[0]

            # delete invalid indices
            if len(small_indices):
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], small_indices, axis=0)
                # import cv2
                # for invalid_index in small_indices:
                #     x1, y1, x2, y2 = annotations['bboxes'][invalid_index]
                #     label = annotations['labels'][invalid_index]
                #     class_name = self.labels[label]
                #     print('width: {}'.format(x2 - x1))
                #     print('height: {}'.format(y2 - y1))
                #     cv2.rectangle(image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 2)
                #     cv2.putText(image, class_name, (int(round(x1)), int(round(y1))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
            if annotations_group[index]['bboxes'].shape[0] != 0:
                filtered_image_group.append(image)
                filtered_annotations_group.append(annotations_group[index])
            else:
                warnings.warn('Image with id {} (shape {}) contains no valid boxes after transform'.format(
                    group[index],
                    image.shape,
                ))

        return filtered_image_group, filtered_annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def random_visual_effect_group_entry(self, image, annotations):
        """
        Randomly transforms image and annotation.
        """
        visual_effect = next(self.visual_effect_generator)
        # apply visual effect
        image = visual_effect(image)
        return image, annotations

    def random_visual_effect_group(self, image_group, annotations_group):
        """
        Randomly apply visual effect on each image.
        """
        assert (len(image_group) == len(annotations_group))

        if self.visual_effect_generator is None:
            # do nothing
            return image_group, annotations_group

        for index in range(len(image_group)):
            # apply effect on a single group entry
            image_group[index], annotations_group[index] = self.random_visual_effect_group_entry(
                image_group[index], annotations_group[index]
            )

        return image_group, annotations_group

    def random_transform_group_entry(self, image, annotations, transform=None):
        """
        Randomly transforms image and annotation.
        """

        # randomly transform both image and annotations
        if transform is not None or self.transform_generator:
            if transform is None:
                transform = adjust_transform_for_image(next(self.transform_generator), image,
                                                       self.transform_parameters.relative_translation)

            # apply transformation to image
            image = apply_transform(transform, image, self.transform_parameters)

            # Transform the bounding boxes in the annotations.
            annotations['bboxes'] = annotations['bboxes'].copy()
            for index in range(annotations['bboxes'].shape[0]):
                annotations['bboxes'][index, :] = transform_aabb(transform, annotations['bboxes'][index, :])

        return image, annotations

    def random_transform_group(self, image_group, annotations_group):
        """
        Randomly transforms each image and its annotations.
        """

        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # transform a single group entry
            image_group[index], annotations_group[index] = self.random_transform_group_entry(image_group[index],
                                                                                             annotations_group[index])

        return image_group, annotations_group

    def resize_image(self, image):
        """
        Resize an image using image_min_side and image_max_side.
        """
        # random_side_index = random.randint(0, 4)
        # return resize_image(image,
        #                     min_side=self.image_min_sides[random_side_index],
        #                     max_side=self.image_max_sides[random_side_index])
        return resize_image(image, min_side=self.image_min_side, max_side=self.image_max_side)

    def preprocess_group_entry(self, image, annotations):
        """
        Preprocess image and its annotations.
        """

        # preprocess the image
        image = self.preprocess_image(image)

        # resize image
        image, image_scale = self.resize_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= image_scale

        # convert to the wanted keras floatx
        image = keras.backend.cast_to_floatx(image)

        return image, annotations

    def preprocess_group(self, image_group, annotations_group):
        """
        Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index])

        return image_group, annotations_group

    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images

        order = list(range(self.size()))
        if self.group_method == 'random':
            random.shuffle(order)
        elif self.group_method == 'ratio':
            order.sort(key=lambda x: self.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group):
        """
        Compute inputs for the network using an image_group.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        # construct an image batch object
        image_batch = np.zeros((len(image_group),) + max_shape, dtype=keras.backend.floatx())

        # copy all images to the upper left part of the image batch object
        for image_index, image in enumerate(image_group):
            image_batch[image_index, :image.shape[0], :image.shape[1], :image.shape[2]] = image

        return image_batch

    def generate_anchors(self, image_shape):
        anchor_params = None
        if self.config and 'anchor_parameters' in self.config:
            anchor_params = parse_anchor_parameters(self.config)
        return anchors_for_shape(image_shape, anchor_params=anchor_params, shapes_callback=self.compute_shapes)

    def compute_targets(self, image_group, annotations_group):
        """
        Compute target outputs for the network using images and their annotations.
        """
        INF = 1e8
        assert (len(image_group) == len(
            annotations_group)), "The length of the images and annotations need to be equal."
        assert (len(annotations_group) > 0), "No data received to compute anchor targets for."
        for annotations in annotations_group:
            assert ('bboxes' in annotations), "Annotations should contain bboxes."
            assert ('labels' in annotations), "Annotations should contain labels."
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))
        feature_shapes = self.compute_shapes(max_shape, pyramid_levels=(3, 4, 5, 6, 7))
        # list of np.array
        locations = self.compute_locations(feature_shapes)
        num_locations_each_layer = [location.shape[0] for location in locations]
        # (m, 2) m=sum(fh*fw)
        locations = np.concatenate(locations, axis=0)
        # (m, 2)
        interest_sizes = self.compute_interest_sizes(num_locations_each_layer)
        batch_size = len(image_group)
        num_classes = self.num_classes()
        batch_regression = np.zeros((batch_size, locations.shape[0], 4 + 1 + 1), dtype=keras.backend.floatx())
        batch_classification = np.zeros((batch_size, locations.shape[0], num_classes + 1), dtype=keras.backend.floatx())
        batch_centerness = np.zeros((batch_size, locations.shape[0], 1 + 1), dtype=keras.backend.floatx())
        # (m, ), (m, )
        cx, cy = locations[:, 0], locations[:, 1]
        for batch_item_id, annotations in enumerate(annotations_group):
            # (n, 4)
            bboxes = annotations['bboxes']
            assert bboxes.shape[0] != 0, 'There should be no such annotations going into training'
            # (n, )
            bbox_areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            # (n, )
            labels = annotations['labels']
            # (m, 1) - (1, n) --> (m, n)
            l = cx[:, None] - bboxes[:, 0][None]
            t = cy[:, None] - bboxes[:, 1][None]
            # (1, n) - (m, 1) --> (m, n)
            r = bboxes[:, 2][None] - cx[:, None]
            b = bboxes[:, 3][None] - cy[:, None]
            # (m, n, 4)
            regr_targets = np.stack([l, t, r, b], axis=2)
            # (m, n)
            is_in_bbox = regr_targets.min(axis=2) > 0
            # (m, n)
            max_regr_target = regr_targets.max(axis=2)
            # limit the regression range for each location
            # (m, n)
            is_cared_in_level = (max_regr_target >= interest_sizes[:, 0:1]) & (max_regr_target <= interest_sizes[:, 1:2])
            locations_to_gt_areas = np.tile(bbox_areas[None], (len(locations), 1))
            locations_to_gt_areas[~is_in_bbox] = INF
            locations_to_gt_areas[~is_cared_in_level] = INF
            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area = locations_to_gt_areas.min(axis=1)
            pos_location_indices = np.where(locations_to_min_area != INF)[0]
            if len(pos_location_indices) == 0:
                warnings.warn('no pos locations')
            locations_to_min_area_ind = locations_to_gt_areas.argmin(axis=1)
            # (m, 4)
            regr_targets = regr_targets[range(len(locations)), locations_to_min_area_ind]
            # (m, 2)
            left_right = regr_targets[:, [0, 2]]
            top_bottom = regr_targets[:, [1, 3]]
            # (m, )
            centerness = (left_right.min(axis=-1) / left_right.max(axis=-1)) * \
                         (top_bottom.min(axis=-1) / top_bottom.max(axis=-1))
            centerness_targets = np.sqrt(np.abs(centerness))
            # (m, )
            location_labels = labels[locations_to_min_area_ind]
            pos_location_labels = location_labels[pos_location_indices]
            batch_regression[batch_item_id, :, :4] = regr_targets
            batch_regression[batch_item_id, :, 4] = centerness_targets
            batch_regression[batch_item_id, pos_location_indices, -1] = 1
            batch_classification[batch_item_id, pos_location_indices, pos_location_labels] = 1
            batch_classification[batch_item_id, pos_location_indices, -1] = 1
            batch_centerness[batch_item_id, :, 0] = centerness_targets
            batch_centerness[batch_item_id, pos_location_indices, -1] = 1

        return [batch_regression, batch_classification, batch_centerness]
        # return [locations, batch_regression, batch_classification, batch_centerness]

    def compute_input_output(self, group):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group, group)

        if len(image_group) == 0:
            return None, None

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def compute_input_output_test(self, group):
        """
        Compute inputs and target outputs for the network.
        """

        # load images and annotations
        # list
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group, group)

        # randomly apply visual effect
        # image_group, annotations_group = self.random_visual_effect_group(image_group, annotations_group)

        # randomly transform data
        # image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # perform preprocessing steps
        # image_group, annotations_group = self.preprocess_group(image_group, annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        return image_group, annotations_group, targets

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[self.current_index]
        inputs, targets = self.compute_input_output(group)
        while inputs is None:
            current_index = self.current_index + 1
            if current_index >= len(self.groups):
                current_index = current_index % (len(self.groups))
            self.current_index = current_index
            group = self.groups[self.current_index]
            inputs, targets = self.compute_input_output(group)
        current_index = self.current_index + 1
        if current_index >= len(self.groups):
            current_index = current_index % (len(self.groups))
        self.current_index = current_index
        return inputs, targets
        # image_group, annotation_group, targets = self.compute_input_output_test(group)
        # return image_group, annotation_group, targets
