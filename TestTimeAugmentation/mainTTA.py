import testTimeAugmentation
import function
import os
import shutil
import argparse
import ensembleOptions
from imutils import paths


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
        
        #yoloDarknet.predict(dir, dir)
        #ssdResnet.predict(dir, dir)
        #fasterResnet.predict(dir, dir)
        #yoloResnet.predict(dir, dir)
        model.predict(dir, dir)
        #maskRcnn.predict(dir,dir)


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


#Enter the path of the folder that will contain the images
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the dataset of images")
ap.add_argument("-o", "--option",  default='consensus', help="option to the ensemble: affirmative, consensus or unanimous")

args = vars(ap.parse_args())
pathImg= args["dataset"]
option = args["option"]

#2. the user define the techniques and configurations fichs
myTechniques = [ "histo","vflip","gamma"]

yoloDarknet = testTimeAugmentation.DarknetYoloPred('/home/master/Desktop/peso/yolov3.weights', '../peso/coco.names','../peso/yolov3.cfg')
ssdResnet = testTimeAugmentation.MXnetSSD512Pred('/home/master/Desktop/peso/ssd_512_resnet50_v1_voc-9c8b225a.params', '../peso/classesMXnet.txt')
fasterResnet = testTimeAugmentation.MXnetFasterRCNNPred('/home/master/Desktop/peso/faster_rcnn_resnet50_v1b_voc-447328d8.params', '../peso/classesMXnet.txt')
yoloResnet = testTimeAugmentation.MXnetYoloPred('/home/master/Desktop/peso/yolo3_darknet53_voc-f5ece5ce.params', '../peso/classesMXnet.txt')
retinaResnet50 = testTimeAugmentation.RetinaNetResnet50Pred('/home/master/Desktop/peso/resnet50_coco_best_v2.1.0.h5', '../peso/coco.csv')
maskRcnn = testTimeAugmentation.MaskRCNNPred('/home/master/Desktop/peso/mask_rcnn_coco.h5', '../peso/coco.names')

tta(yoloDarknet,myTechniques,pathImg,option)