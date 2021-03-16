import testTimeAugmentation
import function
import os
import shutil
import argparse
import ensembleOptions
from imutils import paths

def models(listaModels,pathImg,option):
    # 1. First we create the folder where we will store the resulting images and create as many folders as we have models
    os.mkdir(pathImg+'/../salida')
    for model in listaModels:
        os.mkdir(pathImg+'/../salida/'+os.path.splitext(os.path.basename(model.pathPesos))[0])

    # 2. We create a list with the folders we have created
    listDirOut = []
    for filename in os.listdir(pathImg+'/../salida'):
        if os.path.isdir(pathImg+'/../salida/'+filename) == True:
            listDirOut.append(pathImg+'/../salida/'+filename)


    # 3. we copy the images from the initial folder to each of the created folders
    for dire in listDirOut:
        for fich in os.listdir(pathImg):
            shutil.copy(pathImg+'/'+fich, dire+'/')


    # 4. Generate xml
    for model in listaModels:
        #If the model matches the name of the folder, we will predict it is only folder
        for dir in os.listdir(pathImg+'/../salida/'):
            if (os.path.splitext(os.path.basename(pathImg+'/../salida/'+model.pathPesos))[0]) == dir:
                #Then we list the files in that folder
                images = os.listdir(pathImg+'/../salida/'+dir)
                model.predict(pathImg+'/../salida/'+dir, pathImg+'/../salida/'+dir)#,confidence)



    # 5. We perform the ensemble method
    for dirOut in os.listdir(pathImg+'/../salida/'):
        #for file in list(paths.list_files(pathImg+'/../salida/'+dirOut, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"))):
        for file in list(os.scandir(pathImg+'/../salida/'+dirOut)):
            if file.name.endswith('.jpg') or file.name.endswith('.jpeg') or file.name.endswith('.png') or file.name.endswith('.tif') or file.name.endswith('.tiff') or file.name.endswith('.bmp'):
                os.remove(file)

    ensembleOptions.ensembleOptions(pathImg+'/../salida/', option)

    for xml in os.scandir(pathImg+'/../salida/output/'):
        shutil.copy(pathImg+'/../salida/output/'+xml.name,pathImg+'/')
    shutil.rmtree(pathImg+'/../salida/')


if __name__== "__main__":
    #Enter the path of the folder that will contain the images
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="path to the dataset of images")
    ap.add_argument("-o", "--option",  default='consensus', help="option to the ensemble: affirmative, consensus or unanimous")

    args = vars(ap.parse_args())
    pathImg= args["dataset"]

    option = args["option"]

    #fichs = os.listdir(pathImg)

    imgFolder = pathImg
    #the user define configurations fichs

    yoloDarknet = testTimeAugmentation.DarknetYoloPred('/home/ancasag/Codigo/General/ensembleObjectDetection-masterGit/peso/AlvaroPrueba1_600train_65000.weights', '../peso/vocEstomas.names','../peso/yolov3Estomas.cfg',0.7)
    ssdResnet = testTimeAugmentation.MXnetSSD512Pred('/home/ancasag/Codigo/General/ensembleObjectDetection-masterGit/peso/ssd_512_resnet50_v1_voc-9c8b225a.params', '../peso/classesMXnet.txt',0.7)
    fasterResnet = testTimeAugmentation.MXnetFasterRCNNPred('/home/ancasag/Codigo/General/ensembleObjectDetection-masterGit/peso/faster_rcnn_resnet50_v1b_voc-447328d8.params', '../peso/classesMXnet.txt',0.7)
    yoloResnet = testTimeAugmentation.MXnetYoloPred('/home/ancasag/Codigo/General/ensembleObjectDetection-masterGit/peso/yolo3_darknet53_voc-f5ece5ce.params', '../peso/classesMXnet.txt',0.7)
    retinaResnet50 = testTimeAugmentation.RetinaNetResnet50Pred('/home/ancasag/Codigo/General/ensembleObjectDetection-masterGit/peso/resnet50_coco_best_v2.1.0.h5', '../peso/coco.csv',0.7)
    maskRcnn = testTimeAugmentation.MaskRCNNPred('/home/ancasag/Codigo/General/ensembleObjectDetection-masterGit/peso/mask_rcnn_coco.h5', '../peso/coco.names',0.7)

    listaModels = [retinaResnet50, maskRcnn,yoloResnet,yoloDarknet,fasterResnet,ssdResnet]

    models(listaModels,pathImg,option)
