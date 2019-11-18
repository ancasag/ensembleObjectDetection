import ensemble
import argparse
import numpy as np
import generateXML
import predict_batch
import glob
from lxml import etree
import os
import math




def ensembleOptions(datasetPath, option):

    #we get a list that contains as many pairs as there are xmls in the first folder, 
    #these pairs indicate first the name of the xml file and then contain a list with all the objects of the xmls
    boxes = ensemble.listarCuadrados(datasetPath)

    #we separate by images and we get a list that groups the objects by the iou> 0.5
    for nombre,lis in boxes:
        pick = []
        resul = []
    
        #we check if the output folder where we are going to store the xmls exists
        if os.path.exists(datasetPath+"/output") == False:
            os.mkdir(datasetPath+"/output")
        
        #we look for the width, height and depth of the image
        fichIguales = glob.glob(datasetPath + '/*/' + nombre+'.xml')
        file = open(datasetPath+"/output/"+nombre+".xml", "w")
        numFich = len(fichIguales)
        doc = etree.parse(fichIguales[0])
        filename = doc.getroot()  # we look for the root of our xml
        wI = filename.find("size").find("width").text
        hI = filename.find("size").find("height").text
        d = filename.find("size").find("depth").text
        box = ensemble.uneBoundingBoxes(lis)
        #now we pass the non-maximunSupression to each list within the list obtained
        for rectangles in box:
            lista = []
    
            for rc in rectangles:
                lista.append(rc)
            pick = []
    
            if option == 'consensus':
                if len(np.array(lista))>=math.ceil(numFich/2):#if the number of boxes is greater than half the number of files
                    pick,prob = ensemble.nonMaximumSuppression(np.array(lista), 0.3)
                    pick[0][5] = prob/numFich
    
    
            elif option == 'unanimous':
                if len(np.array(lista))==numFich:#if the number of boxes is greater than half the number of files
                    pick,prob = ensemble.nonMaximumSuppression(np.array(lista), 0.3)
                    pick[0][5] = prob / numFich
    
            elif option == 'affirmative':
                pick,prob = ensemble.nonMaximumSuppression(np.array(lista), 0.3)
                pick[0][5] = prob / numFich
    
            if len(pick)!=0:
                resul.append(list(pick[0]))
        file.write(generateXML.generateXML(nombre, "", wI, hI, d, resul))
        file.close()
