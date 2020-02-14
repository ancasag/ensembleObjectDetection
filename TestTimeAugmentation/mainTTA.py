import ensemble.testTimeAugmentation
import ensemble.function
import os
import shutil
import sys
import argparse
import ensemble.ensembleOptionsTTA
from imutils import paths
from PyQt5 import QtCore, QtGui, QtWidgets
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    from PyQt5 import QtGui, QtCore
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip

        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

def tta(model,myTechniques,pathImg,option, conf):
    fichs = os.listdir(pathImg)
    # 1. Create tmp folder
    os.mkdir(pathImg+'/tmp')
    # move imgs to tmp
    for fich in fichs:
        shutil.copy(pathImg+'/'+fich, pathImg+'/tmp')
    imgFolder = pathImg

    os.mkdir(pathImg+'/../salida')
    # 3. Classification
    for technique in myTechniques:
        ensemble.function.clasification(imgFolder,technique)
    # we get all the folders we have created
    listDirOut = []
    for filename in os.listdir(pathImg+'/../salida'):
        if os.path.isdir(pathImg+'/../salida/'+filename) == True:
            listDirOut.append(pathImg+'/../salida/'+filename)


    for dir in listDirOut:
        for img in os.listdir(dir+'/tmp'):
            img1 = img[img.find("_")+1:]
            img2 = img1[img1.find("_")+1:]
            shutil.move(dir+'/tmp/'+img, dir+'/'+img2)
        os.rmdir(dir+'/tmp')

    # 4. Generate xml

    for dir in listDirOut:
        model.predict(dir, dir,conf)



    # 5. Detection
    for dir in listDirOut:
        tec = dir.split("/")
        ensemble.function.detection(dir, tec[len(tec)-1])

    for dir in listDirOut:
        for img in os.listdir(dir):
            if os.path.isdir(dir+'/'+img)== False:
                os.remove(dir+'/'+img)
        for img in os.listdir(dir+'/tmp'):
            img1 = img[img.find("_") + 1:]
            img2 = img1[img1.find("_") + 1:]
            shutil.move(dir+'/tmp/'+img, dir+'/'+img2)
        os.rmdir(dir+'/tmp')

    # 6. Ensemble
    for dirOut in os.listdir(pathImg+'/../salida/'):
        for file in list(paths.list_files(pathImg+'/../salida/'+dirOut, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))):
            os.remove(file)

    ensemble.ensembleOptionsTTA.ensembleOptions(pathImg+'/../salida/', option)
    for xml in os.listdir(pathImg+'/../salida/output/'):
        shutil.copy(pathImg+'/../salida/output/'+xml,pathImg+'/')
    shutil.rmtree(pathImg+'/tmp')
    shutil.rmtree(pathImg+'/../salida/')
