import cv2
import glob
import keras
import numpy as np
import os
import os.path as osp
import tensorflow as tf
import time

from utils.visualization import draw_box, draw_caption
from utils.colors import label_color
from yolo.model import yolo_body


# set tf backend to allow memory to grow, instead of claiming everything
def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def preprocess_image(image, image_size=416):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size
    image = cv2.resize(image, (resized_width, resized_height))
    new_image = np.ones((image_size, image_size, 3), dtype=np.float32) * 128.
    offset_h = (image_size - resized_height) // 2
    offset_w = (image_size - resized_width) // 2
    new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image.astype(np.float32)
    new_image /= 255.
    return new_image, scale, offset_h, offset_w


# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.set_session(get_session())

model_path = 'pascal_18_6.4112_6.5125_0.8319_0.8358.h5'

model, prediction_model = yolo_body(num_classes=20)

prediction_model.load_weights(model_path, by_name=True)

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
    image = cv2.imread(image_path)

    # copy to draw on
    draw = image.copy()

    # preprocess image for network
    image, scale, offset_h, offset_w = preprocess_image(image)

    # process image
    start = time.time()
    # locations, feature_shapes = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes, scores, labels = prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    # correct boxes for image scale
    boxes[0, :, [0, 2]] -= offset_w
    boxes[0, :, [1, 3]] -= offset_h
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
