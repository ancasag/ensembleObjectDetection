import keras
import kerasfcos.models
from kerasfcos.utils.image import read_image_bgr, preprocess_image, resize_image
from kerasfcos.utils.visualization import draw_box, draw_caption
from kerasfcos.utils.colors import label_color
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths

# import miscellaneous modules
import cv2
import os
import os.path as osp
import numpy as np
import time
import glob

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from kerasfcos.utils.anchors import guess_shapes, compute_locations


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

#confidence=0.5

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generateXML(filename,outputPath,w,h,d,boxes,classes):
    top = ET.Element('annotation')
    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'
    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = filename[0:filename.rfind(".")]
    childPath = ET.SubElement(top, 'path')
    childPath.text = outputPath + "/" + filename
    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'
    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(w)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(h)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = str(d)
    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = str(0)
    for (box,score) in boxes:
        category = classes[box[0]]
        box = box[1].astype("int")
        (x,y,xmax,ymax) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = category
        childScore = ET.SubElement(childObject, 'confidence')
        childScore.text = str(score)
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(x)
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(y)
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(xmax)
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(ymax)
    return prettify(top)

def mainDataset(dataset,output, confidence, weights, fichClass):
    f = open(fichClass)
    LABELS = f.read().strip().split("\n")
    LABELS = [label.split(',')[0] for label in LABELS]
    f.close()

    weighted_bifpn = False
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = weights
    model = kerasfcos.models.load_model(model_path, backbone_name='resnet50')
    model = kerasfcos.models.convert_model(model)


    imagePaths = list(paths.list_images(dataset))
    # loop over the input image paths
    for (i, image_path) in enumerate(imagePaths):
      image = read_image_bgr(image_path)
      image = preprocess_image(image)
      image, scale = resize_image(image)
      boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
      boxes /= scale
      w, h, d = image.shape
      boxes1 = []
      for (box, score, label) in zip(boxes[0], scores[0], labels[0]):
        if score < confidence:
          continue
        boxes1.append(([label,box],score))
      file = open(image_path[0:image_path.rfind(".")]+".xml", "w")
      file.write(generateXML(image_path[0:image_path.rfind(".")],image_path,h, w, d, boxes1,LABELS))
      file.close()
