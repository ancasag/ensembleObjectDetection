# USAGE
# python predict_batch.py --input logos/images --output output

# import the necessary packages
import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# TODO:
# Allow option for --input to be a .txt file OR a directory. Check if
# file, and if so, presume keras-retinanet set of images + labels
#confidence=0.25

def mainDataset(dataset,output,confidence,name,weights,fichClass):
    classes=[]
    f = open(fichClass)
    for linea in f:
      classes.append(str((linea.rstrip("\n")).strip()))
    f.close()
    net = gcv.model_zoo.get_model(name, classes=classes, pretrained_base=False)
    net.load_parameters(weights)
    imagePaths = list(paths.list_images(dataset))
    # loop over the input image paths
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image (in BGR order), clone it, and preprocess it
        image = cv2.imread(imagePath)
        (hI, wI, d) = image.shape
        # detect objects in the input image and correct for the image scale
        x, image = gcv.data.transforms.presets.ssd.load_test(imagePath,min(wI,hI))
        cid, score, bbox = net(x)
        boxes1 = []
        for (box, score, cid) in zip(bbox[0], score[0], cid[0]):
            if score < confidence:
                continue     
            boxes1.append((box,score, net.classes[cid[0].asnumpy()[0].astype('int')]))
    
        # parse the filename from the input image path, construct the
        # path to the output image, and write the image to disk
        filename = imagePath.split(os.path.sep)[-1]
        file = open(imagePath[0:imagePath.rfind(".")]+".xml", "w")
        file.write(generateXML(imagePath[0:imagePath.rfind(".")],imagePath,wI, hI, d, boxes1))
        file.close()

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def generateXML(filename,outputPath,w,h,d,boxes):
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
    for (box,score, label) in boxes:
        box = box.astype("int")
        (x,y,xmax,ymax) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = label
        childScore = ET.SubElement(childObject, 'confidence')
        childScore.text = str(score.asscalar())
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(max(x.asscalar(),1))
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(max(y.asscalar(),1))
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(min(xmax.asscalar(),w-1))
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(min(ymax.asscalar(),h-1))
    return prettify(top)


