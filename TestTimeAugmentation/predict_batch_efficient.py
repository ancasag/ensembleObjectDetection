from EfficientDet.model import efficientdet
import cv2
import os
import numpy as np
import time
from EfficientDet.utils import preprocess_image
from EfficientDet.utils.anchors import anchors_for_shape
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths

#confidence=0.5
def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generateXML(filename,outputPath,w,h,d,boxes,scores,labels,classes):
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
    for (category,box,score) in zip(labels,boxes,scores):
        (x,y,xmax,ymax) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = classes[category]
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
        childXmin.text = str(int(x))
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(int(y))
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(int(xmax))
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(int(ymax))
    return prettify(top)


def mainDataset(dataset, output,weights, fichClass,confidence,phi=0):
    f = open(fichClass)
    LABELS = f.read().strip().split("\n")
    LABELS = {int(L.split(",")[1]): L.split(",")[0] for L in LABELS}
    f.close()

    weighted_bifpn = False
    model_path = weights
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]

    num_classes = len(LABELS)
    score_threshold = confidence
    model, prediction_model = efficientdet(phi=phi,
                                           weighted_bifpn=weighted_bifpn,
                                           num_classes=num_classes,
                                           score_threshold=score_threshold)
    prediction_model.load_weights(model_path, by_name=True)

    imagePaths = list(os.scandir(dataset))
    # loop over the input image paths
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(dataset+'/'+imagePath.name)
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale, offset_h, offset_w = preprocess_image(image, image_size=image_size)
        inputs = np.expand_dims(image, axis=0)
        anchors = anchors_for_shape((image_size, image_size))
        boxes, scores, labels = prediction_model.predict_on_batch([np.expand_dims(image, axis=0),
                                                                   np.expand_dims(anchors, axis=0)])
        boxes[0, :, [0, 2]] = boxes[0, :, [0, 2]] - offset_w
        boxes[0, :, [1, 3]] = boxes[0, :, [1, 3]] - offset_h
        boxes /= scale
        boxes[0, :, 0] = np.clip(boxes[0, :, 0], 0, w - 1)
        boxes[0, :, 1] = np.clip(boxes[0, :, 1], 0, h - 1)
        boxes[0, :, 2] = np.clip(boxes[0, :, 2], 0, w - 1)
        boxes[0, :, 3] = np.clip(boxes[0, :, 3], 0, h - 1)
        indices = np.where(scores[0, :] > score_threshold)[0]
        boxes = boxes[0, indices]
        scores = scores[0, indices]
        labels = labels[0, indices]

        # parse the filename from the input image path, construct the
        # path to the output image, and write the image to disk
        filename = imagePath.split(os.path.sep)[-1]
        ext = os.path.splitext(imagePath)
        file = open(ext[0] + ".xml", "w")
        file.write(generateXML(ext[0], imagePath.name, hI, wI, d, boxes1))
        file.close()
