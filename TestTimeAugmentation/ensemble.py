import os
from lxml import etree
import glob
import numpy as np

def listarCuadrados(pathCarpeta):

    boxesAllXmls = []#list that stores all the lists of boxes of all xml
    boxes = []  # list that will contain all the squares of each xml
    prob = 0.5
    listDirectorios = os.scandir(path=pathCarpeta)#we list the files in the last folder
    for files in listDirectorios:
        if files.is_dir():
            break


    for fichero in os.listdir(files):#We go through the files in the folder
        (nombreFichero, extension) = os.path.splitext(fichero)

        if (extension == ".xml"):#we stay with those who are xmls and we go through them looking for a box
            boxes=[]
            fichIguales = glob.glob(pathCarpeta+'/*/'+fichero)#
            for f in fichIguales:
                j = 0  # declaration of variables

                doc = etree.parse(f)
                filename = doc.getroot()  # we look for the root of our xml
                objetos = filename.findall("object")
                while j < len(objetos):
                    name = objetos[j].find("name").text
                    ymax = float(objetos[j].find("bndbox").find("ymax").text)
                    ymin = float(objetos[j].find("bndbox").find("ymin").text)
                    xmax = float(objetos[j].find("bndbox").find("xmax").text)
                    xmin = float(objetos[j].find("bndbox").find("xmin").text)
                    prob = "{0:.2f}".format(float(objetos[j].find("confidence").text))
                    boxes.append([name, xmin, ymin, xmax, ymax, prob])
                    j = j+1
        boxesAllXmls.append((nombreFichero,boxes))
    return boxesAllXmls


def uneBoundingBoxes(boxesAllXmls):

    boundingBox=[]
    listBox = []
    l=len(boxesAllXmls)
    while(l>0):
        boxPrim=boxesAllXmls[0]

        listBox.append(boxPrim)
        boxesAllXmls1=boxesAllXmls[1:]
        boxesAllXmls.remove(boxPrim)
        for box in boxesAllXmls1:
            if boxPrim[0]==box[0] and bb_intersection_over_union(boxPrim[1:5], box[1:5]) > 0.5:
                listBox.append(box)
                boxesAllXmls.remove(box)


        boundingBox.append(listBox)
        listBox = []
        l=len(boxesAllXmls)
    return boundingBox



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def nonMaximumSuppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [0]
    # initialize the list of picked indexes
    pick = []
    probFinal = 0
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 1].astype(float)
    y1 = boxes[:, 2].astype(float)
    x2 = boxes[:, 3].astype(float)
    y2 = boxes[:, 4].astype(float)
    prob = boxes[:, 5].astype(float)
    for l in prob:
        probFinal = probFinal+l
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick], probFinal