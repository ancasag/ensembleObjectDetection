import cv2
import numpy as np
from generators.voc_generator import PascalVocGenerator
from utils.transform import random_transform_generator
from utils.image import random_visual_effect_generator
from utils.image import preprocess_image


def show_annotations():
    generator = PascalVocGenerator(data_dir='datasets/voc_trainval/VOC0712', set_name='val')
    for image_group, annotation_group, targets in generator:
        locations = targets[0]
        batch_regr_targets = targets[1]
        batch_cls_targets = targets[2]
        batch_centerness_targets = targets[3]
        for image, annotation, regr_targets, cls_targets, centerness_targets in zip(image_group, annotation_group,
                                                                                    batch_regr_targets,
                                                                                    batch_cls_targets,
                                                                                    batch_centerness_targets):
            gt_boxes = annotation['bboxes']
            for gt_box in gt_boxes:
                gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box
                cv2.rectangle(image, (int(gt_xmin), int(gt_ymin)), (int(gt_xmax), int(gt_ymax)), (0, 255, 0), 2)
            pos_indices = np.where(centerness_targets[:, 1] == 1)[0]
            for pos_index in pos_indices:
                cx, cy = locations[pos_index]
                l, t, r, b, *_ = regr_targets[pos_index]
                xmin = cx - l
                ymin = cy - t
                xmax = cx + r
                ymax = cy + b
                class_id = np.argmax(cls_targets[pos_index])
                centerness = centerness_targets[pos_index][0]
                # cv2.putText(image, '{:.2f}'.format(centerness), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 255), 2)
                cv2.putText(image, str(class_id), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
                cv2.circle(image, (round(cx), round(cy)), 5, (255, 0, 0), -1)
                cv2.rectangle(image, (round(xmin), round(ymin)), (round(xmax), round(ymax)), (0, 0, 255), 2)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image', image)
            cv2.waitKey(0)


def verify_no_negative_regr():
    transform_generator = random_transform_generator(
        min_rotation=-0.1,
        max_rotation=0.1,
        min_translation=(-0.1, -0.1),
        max_translation=(0.1, 0.1),
        min_shear=-0.1,
        max_shear=0.1,
        min_scaling=(0.9, 0.9),
        max_scaling=(1.1, 1.1),
        flip_x_chance=0.5,
        flip_y_chance=0.5,
    )
    visual_effect_generator = random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)
    )
    common_args = {
        'batch_size': 1,
        'image_min_side': 800,
        'image_max_side': 1333,
        'preprocess_image': preprocess_image,
    }
    generator = PascalVocGenerator(
        'datasets/voc_trainval/VOC0712',
        'trainval',
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        skip_difficult=True,
        **common_args
    )
    i = 0
    for image_group, targets in generator:
        i += 1
        if i > 20000:
            break


verify_no_negative_regr()
