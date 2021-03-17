import testTimeAugmentation
import function
import os
import shutil
import sys
import argparse
import ensembleOptions
from imutils import paths
notebook = True
def tta(model,myTechniques,pathImg,option):
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
        function.clasification(imgFolder,technique)
    # we get all the folders we have created

    listDirOut = []
    for filename in os.scandir(pathImg+'/../salida'):
        if os.path.isdir(pathImg+'/../salida/'+filename.name) == True:
            listDirOut.append(pathImg+'/../salida/'+filename.name)


    for dir in listDirOut:
        for img in os.scandir(dir+'/tmp'):
            img1 = img.name[(img.name).find("_")+1:]
            img2 = img1[img1.find("_")+1:]
            shutil.move(dir+'/tmp/'+img.name, dir+'/'+img2)
        os.rmdir(dir+'/tmp')

    # 4. Generate xml
    for dir in listDirOut:
        model.predict(dir, dir)

    # 5. Detection
    for dir in listDirOut:
        tec = dir.split("/")
        function.detection(dir, tec[len(tec)-1])

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

    ensembleOptions.ensembleOptions(pathImg+'/../salida/', option)
    for xml in os.scandir(pathImg + '/../salida/output/'):
        shutil.copy(pathImg + '/../salida/output/' + xml.name, pathImg + '/')
    if notebook is False:
        shutil.rmtree(pathImg+'/../salida/')
    shutil.rmtree(pathImg + '/tmp')

if __name__== "__main__":
    #Enter the path of the folder that will contain the images
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to the dataset of images")
    ap.add_argument("-o", "--option",  default='consensus', help="option to the ensemble: affirmative, consensus or unanimous")
    notebook = False
    args = vars(ap.parse_args())
    pathImg = args["dataset"]

    option = args["option"]
    imgFolder = pathImg
    # the user define configurations fichs
    yoloDarknet = testTimeAugmentation.DarknetYoloPred('/home/ancasag/Codigo/General/ensembleObjectDetection/peso/yolov3.weights', '/home/ancasag/Codigo/General/ensembleObjectDetection/peso/coco.names','/home/ancasag/Codigo/General/ensembleObjectDetection/peso/yolov3.cfg', 0.7)
    # ssdResnet = testTimeAugmentation.MXnetSSD512Pred('weights/ssd_512_resnet50_v1_voc-9c8b225a.params', 'weights/classesMXnet.txt',0.7)
    # fasterResnet = testTimeAugmentation.MXnetFasterRCNNPred('weights/Desktop/peso/faster_rcnn_resnet50_v1b_voc-447328d8.params', 'weights/classesMXnet.txt',0.7)
    # yoloResnet = testTimeAugmentation.MXnetYoloPred('weights/Desktop/peso/yolo3_darknet53_voc-f5ece5ce.params', 'weights/classesMXnet.txt',0.7)
    # retinaResnet50 = testTimeAugmentation.RetinaNetResnet50Pred('weights/resnet50_coco_best_v2.1.0.h5', 'weights/coco.csv',0.7)
    # maskRcnn = testTimeAugmentation.MaskRCNNPred('weights/mask_rcnn_coco.h5', 'weights/coco.names',0.7)

    myTechniques = ["histo", "hflip", "none"]
    tta(yoloDarknet, myTechniques, pathImg, option)