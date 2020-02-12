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

from utils.compute_overlap import compute_overlap
from utils.visualization import draw_detections, draw_annotations

import keras
import numpy as np
import os
import cv2
import progressbar
import pickle

assert (callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, score_threshold=0.05, max_detections=100, visualize=False):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save the images with visualized detections to.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in
                      range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        raw_image = generator.load_image(i)
        image = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))[:3]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        # (n, 4)
        image_boxes = boxes[0, indices[scores_sort], :]
        # (n, )
        image_scores = scores[scores_sort]
        # (n, )
        image_labels = labels[0, indices[scores_sort]]
        # (n, 6)
        image_detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if visualize:
            draw_annotations(raw_image, generator.load_annotations(i), label_to_name=generator.label_to_name)
            draw_detections(raw_image, image_boxes[:5], image_scores[:5], image_labels[:5], label_to_name=generator.label_to_name,
                            score_threshold=score_threshold)

            # cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)
            cv2.namedWindow('{}'.format(i), cv2.WINDOW_NORMAL)
            cv2.imshow('{}'.format(i), raw_image)
            cv2.waitKey(0)

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]

    return all_detections


def _get_annotations(generator):
    """
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        A list of lists containing the annotations for each image in the generator.

    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
        generator,
        model,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        visualize=False,
        epoch=0
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.

    Returns:
        A dict mapping class names to mAP scores.

    """
    # gather all detections and annotations
    all_detections = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections,
                                     visualize=visualize)
    all_annotations = _get_annotations(generator)
    average_precisions = {}

    # all_detections = pickle.load(open('all_detections_{}.pkl'.format(epoch + 1), 'rb'))
    # all_annotations = pickle.load(open('all_annotations_{}.pkl'.format(epoch + 1), 'rb'))
    # pickle.dump(all_detections, open('all_detections_{}.pkl'.format(epoch + 1), 'wb'))
    # pickle.dump(all_annotations, open('all_annotations_{}.pkl'.format(epoch + 1), 'wb'))

    # process detections and annotations
    for label in range(generator.num_classes()):
        if not generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections = all_detections[i][label]
            annotations = all_annotations[i][label]
            num_annotations += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    continue
                overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

    return average_precisions


if __name__ == '__main__':
    from generators.voc_generator import PascalVocGenerator
    from utils.image import preprocess_image
    import models
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    common_args = {
        'batch_size': 1,
        'image_min_side': 800,
        'image_max_side': 1333,
        'preprocess_image': preprocess_image,
    }
    # generator = PascalVocGenerator(
    #     'datasets/voc_trainval/VOC0712',
    #     'val',
    #     shuffle_groups=False,
    #     skip_truncated=False,
    #     skip_difficult=True,
    #     **common_args
    # )
    generator = PascalVocGenerator(
        'datasets/voc_test/VOC2007',
        'test',
        shuffle_groups=False,
        skip_truncated=False,
        skip_difficult=True,
        **common_args
    )
    model_path = '/home/adam/workspace/github/xuannianz/carrot/fsaf/snapshots/2019-10-05/resnet101_pascal_47_0.7652.h5'
    # load retinanet model
    # import keras.backend as K
    # K.set_learning_phase(1)
    from models.resnet import resnet_fsaf
    from models.retinanet import fsaf_bbox
    fsaf = resnet_fsaf(num_classes=20, backbone='resnet101')
    model = fsaf_bbox(fsaf)
    model.load_weights(model_path, by_name=True)
    average_precisions = evaluate(generator, model, visualize=False)
    # compute per class average precision
    total_instances = []
    precisions = []
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations), generator.label_to_name(label),
              'with average precision: {:.4f}'.format(average_precision))
        total_instances.append(num_annotations)
        precisions.append(average_precision)
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    print('mAP: {:.4f}'.format(mean_ap))
