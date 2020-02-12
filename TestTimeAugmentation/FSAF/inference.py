import keras
import models
from utils.image import read_image_bgr, preprocess_image, resize_image
from utils.visualization import draw_box, draw_caption
from utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import glob
import os.path as osp

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.set_session(get_session())
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = '/home/adam/workspace/github/xuannianz/carrot/fsaf/snapshots/2019-10-05/resnet101_pascal_47_0.7652.h5'

# load retinanet model
# model = models.load_model(model_path, backbone_name='resnet101')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
from models.resnet import resnet_fsaf
from models.retinanet import fsaf_bbox
fsaf = resnet_fsaf(num_classes=20, backbone='resnet101')
model = fsaf_bbox(fsaf)
model.load_weights(model_path, by_name=True)
# load label to names mapping for visualization purposes
voc_classes = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}
labels_to_names = {}
for key, value in voc_classes.items():
    labels_to_names[value] = key
# load image
image_paths = glob.glob('datasets/voc_test/VOC2007/JPEGImages/*.jpg')
for image_path in image_paths:
    print('Handling {}'.format(image_path))
    image = read_image_bgr(image_path)

    # copy to draw on
    draw = image.copy()

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    # locations, feature_shapes = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    labels_to_locations = {}
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break
        start_x = int(box[0])
        start_y = int(box[1])
        end_x = int(box[2])
        end_y = int(box[3])
        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', draw)
    key = cv2.waitKey(0)
    if int(key) == 121:
        image_fname = osp.split(image_path)[-1]
        cv2.imwrite('test/{}'.format(image_fname), draw)

