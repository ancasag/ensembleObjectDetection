import cv2 as cv 
import numpy as np
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths
import sys
import os
import urllib.request
from tqdm import tqdm
import glob
import testTimeAugmentation as test
import generateXML

#confThreshold = 0.4  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, conf):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    newboxes = []
    newconfidences = []
    newclassIds = []
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf, nmsThreshold)
    for i in indices:
        i = i[0]
        newboxes.append(boxes[i])
        newconfidences.append(confidences[i])
        newclassIds.append(classIds[i])
    return newboxes,newconfidences, newclassIds



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)



def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def generateXML(filename,outputPath,w,h,d,boxes,confidences, classIds,classes):
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
    for box,con, categoryID in zip(boxes,confidences, classIds):
        category = classes[categoryID]
        (x,y,wb,hb) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = category
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childConfidence = ET.SubElement(childObject, 'confidence')
        childConfidence.text = str(con)
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(x)
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(y)
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(x+wb)
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(y+hb)
    return prettify(top)

def mainImage(imagePath):
    generateXMLFromImage(imagePath)

def mainDataset(imagesPath, outputDataset, conf, pathPesos, fichNames, fichCfg):
    # Give the configuration and weight files for the model and
    # load the network using them.
    if os.path.exists(outputDataset) == False:
        os.mkdir(outputDataset)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    #fichNames = glob.glob(pathPesos+"/*.names")
    #classesFile = fichNames[0]
    classes = None
    with open(fichNames, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    #fichCfg = glob.glob(pathPesos + "/*.cfg")
    #modelConfiguration = fichCfg[0]
    #modelWeights = glob.glob(pathPesos+"/*.weights")
    

    # Invocamos a la funcion con dichos parametros y mostramos el resultado por pantalla
    images = list(paths.list_files(imagesPath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")))
    #for peso in modelWeights:
    net = cv.dnn.readNetFromDarknet(fichCfg, pathPesos)#peso)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    #nomPeso = os.path.basename( pathPesos)#peso)
    #nombre = os.path.splitext(nomPeso)[0]
    for image in images:
        generateXMLFromImage(image, outputDataset, net,classes, conf)#+'/'+nombre, net)


def generateXMLFromImage(imagePath, output, net, classes, conf):
    im = cv.VideoCapture(imagePath)
    hasFrame, frame = im.read()
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    if os.path.exists(output) == False:
        os.mkdir(output)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    boxes,confidences, classIds = postprocess(frame, outs, conf)
    wI, hI, d = frame.shape
    filename = os.path.basename(imagePath)
    file = open(output+'/'+os.path.splitext(filename)[0] + ".xml", "w")
    file.write(generateXML(imagePath[0:imagePath.rfind(".")], "", wI, hI, d, boxes,confidences,classIds,classes))
    file.close()
