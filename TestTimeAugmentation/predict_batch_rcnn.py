# USAGE
# python predict_batch.py --input logos/images --output output

# import the necessary packages
import numpy as np
import mxnet as mx
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from mrcnn import utils

#confidence=0.25

# TODO:
# Allow option for --input to be a .txt file OR a directory. Check if
# file, and if so, presume keras-retinanet set of images + labels


##### J. Esto hay que cambiarlo dependiendo de cada problema


class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     #BACKBONE = "resnet50"
     #IMAGE_RESIZE_MODE = "square"
     #IMAGE_MIN_DIM = 512
     #IMAGE_MAX_DIM = 512
     IMAGES_PER_GPU = 1
     ##### J. Esto hay que cambiarlo dependiendo de cada problema
     NUM_CLASSES = 0 #+ len(classes)
 
# define the model

# load coco model weights



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
    #boxes tiene que contener labels
    for (box,score) in boxes:
        # Cambiar categoria por label
        category = box[0]
        box = box[1].astype("int")
        ####### 
        # Cuidado esto está cambiado con respecto a lo que es habitualmente
        #######  
        (y,x,ymax,xmax) = box
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

# loop over the input image paths

def mainDataset(dataset,output, confidence,weights,fichClass):
    LABELS = open(fichClass).read().strip().split("\n")
    classes =[L.split(",")[0] for L in LABELS]
    conf= TestConfig()
    conf.NUM_CLASSES = 1+ len(classes)
    conf.IMAGE_META_SIZE=1+3+3+4+1+conf.NUM_CLASSES
    
    rcnn = MaskRCNN(mode='inference', model_dir='./', config=conf)
    rcnn.load_weights(weights, by_name=True)#, exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    imagePaths = list(paths.list_images(dataset))
    for (i, imagePath) in enumerate(imagePaths):
	    # load the input image (in BGR order), clone it, and preprocess it
	    #print("[INFO] predicting on image {} of {}".format(i + 1,
	    #	len(imagePaths)))

	    # load the input image (in BGR order), clone it, and preprocess it
	    img = load_img(imagePath)
	    img = img_to_array(img)
	    (hI, wI, d) = img.shape
	

	    # detect objects in the input image and correct for the image scale
        # Poner short=512
	    results = rcnn.detect([img], verbose=0)
	    r = results[0]
	    boxes1 = []
	    for (box, score,cid) in zip(r['rois'], r['scores'],r['class_ids']):
		    if score < confidence:
			    continue
                # Añadir label que sera con net.classes[cid]
		    boxes1.append(([classes[cid-1],box],score))

	    # parse the filename from the input image path, construct the
	    # path to the output image, and write the image to disk
	    filename = imagePath.split(os.path.sep)[-1]
	    #outputPath = os.path.sep.join([args["output"], filename])
	    file = open(imagePath[0:imagePath.rfind(".")]+".xml", "w")
	    file.write(generateXML(imagePath[0:imagePath.rfind(".")],imagePath,wI, hI, d, boxes1))
	    file.close()

	
	#cv2.imwrite(outputPath, output)
