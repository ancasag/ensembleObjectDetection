import ensemble
import argparse
import numpy as np
import generateXML
import glob
from lxml import etree
import os
import math

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to the dataset of images")
ap.add_argument("-o", "--option", required=True,help="option to the ensemble: affirmative, consensus or unanimous")
args = vars(ap.parse_args())

#read the arguments
datasetPath= args["dataset"]
option = args["option"]

#we get a list that contains as many pairs as there are xmls in the first folder, these pairs indicate first the
#name of the xml file and then contains a list with all the objects of the xmls
boxes = ensemble.listBoxes(datasetPath)

for nombre,lis in boxes:
    pick = []
    resul = []

    #we check if the output folder exists
    equalFiles = glob.glob(datasetPath + '/*/' + nombre+'.xml')
    if os.path.exists(datasetPath+"/output") == False:
        os.mkdir(datasetPath+"/output")
    file = open(datasetPath+"/output/"+nombre+".xml", "w")
    numFich = len(equalFiles) -1
    if equalFiles[0].find("/output/")>0:
        doc = etree.parse(equalFiles[1])
    else:
        doc = etree.parse(equalFiles[0])
    filename = doc.getroot()  # we look for the root of our xml
    wI = filename.find("size").find("width").text
    hI = filename.find("size").find("height").text
    d = filename.find("size").find("depth").text
    box = ensemble.uneBoundingBoxes(lis)
    #apply the corresponging technique
    for rectangles in box:
        list1 = []

        for rc in rectangles:
            list1.append(rc)
        pick = []

        if option == 'consensus':
            if len(np.array(list1))>=math.ceil(numFich/2):
                pick,prob = ensemble.nonMaximumSuppression(np.array(list1), 0.3)
                pick[0][5] = prob/numFich


        elif option == 'unanimous':
            if len(np.array(list1))==numFich:
                pick,prob = ensemble.nonMaximumSuppression(np.array(list1), 0.3)
                pick[0][5] = prob / numFich

        elif option == 'affirmative':
            pick,prob = ensemble.nonMaximumSuppression(np.array(list1), 0.3)
            pick[0][5] = prob / numFich

        if len(pick)!=0:
            resul.append(list(pick[0]))
    file.write(generateXML.generateXML(nombre, "", wI, hI, d, resul))
    file.close()
