import cv2 as cv #s
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from imutils import paths
import os
from gluoncv import model_zoo
from mxnet import autograd, gluon
import gluoncv as gcv
from tqdm import tqdm


confThreshold = 0.25  #Confidence threshold
nmsThreshold = 0.45   #Non-maximum suppression threshold
inpWidth = 1248       #Width of network's input image
inpHeight = 1248      #Height of network's input image


# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]



# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
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
            if confidence > confThreshold:
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
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return boxes,confidences



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
        childXmin.text = str(x.asscalar())
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(y.asscalar())
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(xmax.asscalar())
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(ymax.asscalar())
    return prettify(top)

def mainImage(imagePath):
    generateXMLFromImage(imagePath)

def mainDataset(imagesPath, outputDataset, weights):
    # Give the configuration and weight files for the model and
    # load the network using them.
    # Load names of classes
    if os.path.exists(outputDataset) == False:
        os.mkdir(outputDataset)
    dirPath = os.path.dirname(os.path.realpath(__file__))
    classes = None
    modelWeights = weights;
    images = list(paths.list_files(imagesPath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")))
    net = model_zoo.get_model(weights, pretrained=True)
    for image in images:
        generateXMLFromImage(image, outputDataset, net)


def generateXMLFromImage(imagePath, output, net):
    image = cv.imread(imagePath)
    (hI, wI, d) = image.shape
    
    # detect objects in the input image and correct for the image scale
    x, image = gcv.data.transforms.presets.ssd.load_test(imagePath,min(wI,hI))
    cid, score, bbox = net(x)
    boxes1 = []
    for (box, score, cid) in zip(bbox[0], score[0], cid[0]):
        if score < 0.5:
            continue    
        boxes1.append((box,score, net.classes[cid[0].asnumpy()[0].astype('int')]))

    # parse the filename from the input image path, construct the
    # path to the output image, and write the image to disk
    filename = imagePath.split(os.path.sep)[-1]
    file = open(imagePath[0:imagePath.rfind(".")]+".xml", "w")
    file.write(generateXML(imagePath[0:imagePath.rfind(".")],imagePath,wI, hI, d, boxes1))
    file.close()