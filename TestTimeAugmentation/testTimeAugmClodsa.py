import techniques
import os
import shutil
import detectStoma
import argparse

#Enter the path of the folder that will contain the images
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to the dataset of images")

ap.add_argument("-m", "--model", required=True,
    help="choose model")

args = vars(ap.parse_args())
pathImg= args["dataset"]
fichs = os.listdir(pathImg)
# 1. Create tmp folder
os.mkdir(pathImg+'/tmp')
# move imgs to tmp
for fich in fichs:
    shutil.copy(pathImg+'/'+fich, pathImg+'/tmp')
imgFolder = pathImg
#2. the user definethe techniques
mytechniques = [ "hflip","rotation10","histo","gamma", "none"]
os.mkdir(pathImg+'/../out')
# 3. Clasification
for technique in mytechniques:
    techniques.clasification(imgFolder,technique)
# we get all the folders we have created
listDirOut = []
for filename in os.listdir(pathImg+'/../out'):
    if os.path.isdir(pathImg+'/../out/'+filename) == True:
        listDirOut.append(pathImg+'/../out/'+filename)


for dir in listDirOut:
    for img in os.listdir(dir+'/tmp'):
        img1 = img[img.find("_")+1:]
        img2 = img1[img1.find("_")+1:]
        shutil.move(dir+'/tmp/'+img, dir+'/'+img2)
    os.rmdir(dir+'/tmp')

# 4. Generate xml
for dir in listDirOut:
    detectStoma.mainDataset(dir, dir, args["model"])

# 5. Detection

for dir in listDirOut:
    tec = dir.split("/")
    techniques.detection(dir, tec[len(tec)-1])

for dir in listDirOut:
    for img in os.listdir(dir):
        if os.path.isdir(dir+'/'+img)== False:
            os.remove(dir+'/'+img)
    for img in os.listdir(dir+'/tmp'):
        img1 = img[img.find("_") + 1:]
        img2 = img1[img1.find("_") + 1:]
        shutil.move(dir+'/tmp/'+img, dir+'/'+img2)
    os.rmdir(dir+'/tmp')



