# Ensemble methods for object detection

In this repository, we provide the code for ensembling the output of object detection models, and applying test-time augmentation for object detection. This library has been designed to be applicable to any object detection model independently of the underlying algorithm and the framework employed to implement it. A draft describing the techniques implemented in this repository are available in the following [article](https://drive.google.com/file/d/1ku8X8lHs6lethEa5Adhj7frzV44NTbl4/view?usp=sharing).

0. [Installation](#installation-and-requirements)
1. [Ensemble of models](#ensemble-of-models)
   * [Ensemble options](#ensemble-options)
   * [Execution](#execution)
2. [Test-time augmentation for object detection](#test-time-augmentation-for-object-detection)
   * [Ensemble options](#ensemble-options)
   * [Techniques for TTA](#techniques-for-tta)
   * [Execution](#execution)
3. [Adding new models](#adding-new-models)
   * [Available models](#available-models)
4. [Experiments](#experiments)
   * [Pascal VOC](#pascal-voc)
   * [Stomata](#stomata)
   * [Tables](#tables)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Installation and Requirements

This library requires Python 3.6 and the packages listed in ```requirements.txt```.

Installation:
1. Clone this repository
2. Install dependencies

```bash
pip3 install -r requirements.txt
```

## Ensemble of models 

In the following image, we show an example of the workflow of our ensemble algorithm. Three methods have been applied to detect the objects in the original image: the first method has detected the person and the horse; the second, the person and the dog; and, the third, the person, the dog, and an undefined region. The first step of our ensemble method groups the overlapping regions. Subsequently, a voting strategy is applied to discard some of those groups. The final predictions are obtained using the NMs algorithm.

![TestTimeAugmentation](images/ensemble.jpg)

### Ensemble Options
Three different voting strategies can be applied with our ensemble algorithm:
*   Affirmative. This means that whenever one of the methods that produce the initial predictions says that a region contains an object, such a detection is considered as valid.
*   Consensus. This means that the majority of the initial methods must agree to consider that a region contains an object. The consensus strategy is analogous to the majority voting strategy commonly applied in ensemble methods for images classification.
*   Unanimous. This means that all the methods must agree to consider that a region contains an object.

### Execution

In order to run the ensemble algorithm, you can edit the file mainModel.py from the TestTimeAugmentation folder to configure the models to use and then invoke the following command where ```pathOfDataset``` is the path where the images are saved, and ```option```  is the voting strategy (affirmative, consensus or unanimous).

```bash
python mainModel.py -d pathOfDataset -o option
```

A simpler way to use our this method is provided in the following notebook. 

- [Notebook for ensembling models](https://colab.research.google.com/drive/1Tg9WaI_Cd-lPXDMuj6tHDlqakxo4-CLK)

## Test-time augmentation for object detection

In the following image, we show an example of the workflow of test-time augmentation (from now on, TTA) for object detectors. First, we apply three transformations to the original image: a histogram equalisation, a horizontal flip, and a none transformation (that does not modify the image). Subsequently, we detect the objects in the new images, and apply the corresponding detection transformation to locate the objects in the correct position for the original image. Finally, the detections are ensembled using the consensus strategy.

![TestTimeAugmentation](images/testTimeAugm.jpg)

### Ensemble Options
As indicated previously, three different voting strategies can be applied for TTA:
*   Affirmative. This means that whenever one of the methods that produce the initial predictions says that a region contains an object, such a detection is considered as valid.
*   Consensus. This means that the majority of the initial methods must agree to consider that a region contains an object. The consensus strategy is analogous to the majority voting strategy commonly applied in ensemble methods for images classification.
*   Unanimous. This means that all the methods must agree to consider that a region contains an object.

### Techniques for TTA

These are all the techniques that we have defined to use in the TTA process. The first column corresponds with the name assigned to the technique, and the second column describes the technique.

- "avgBlur": Average blurring
- "bilaBlur": Bilateral blurring 
- "blur": Blurring
- "chanHsv": Change to hsv colour space
- "chanLab": Blurring
- "crop": Crop
- "dropOut": Dropout
- "elastic": Elastic deformation
- "histo": Equalize histogram
- "vflip": Vertical flip
- "hflip": Horizontal flip
- "hvflip": Vertical and horizontal flip
- "gamma": Gamma correction
- "blurGau": Gaussian blurring
- "avgNoise": Add Gaussian noise
- "invert": Invert
- "medianblur": Median blurring
- "none": None 
- "raiseBlue": Raise blue channel
- "raiseGreen": Raise green channel
- "raiseHue": Raise hue
- "raiseRed": Raise red
- "raiseSatu": Raise saturation
- "raiseValue": Raise value
- "resize": Resize
- "rotation10": Rotate 10º
- "rotation90": Rotate 90º
- "rotation180": Rotate 180º
- "rotation270": Rotate 270º
- "saltPeper": Add salt and pepper noise
- "sharpen": Sharpen
- "shiftChannel": Shift channel
- "shearing": Shearing
- "translation": Translation

### Execution

In order to run the ensemble algorithm, you can edit the mainTTA.py file from the TestTimeAugmentation folder to configure the model to use and the transformation techniques. Then, you can invoke the following command where ```pathOfDataset``` is the path where the images are saved, and ```option```  is the voting strategy (affirmative, consensus or unanimous).

```bash
python mainTTA.py -d pathOfDataset -o option
```

A simpler way to use our this method is provided in the following notebook. 

- [Test Time Augmentation Notebook](https://colab.research.google.com/drive/1T1mn85AedRlaTNHeJW_QeTy0I5wOy14J)

## Adding new models

This open source library can be extended to work with any object detection model regardless of the algorithm and framework used to build it. To do this, it is necessary to create a new class that extends the ```IPredictor``` class of the following diagram:

![DiagramModels](images/diagramaClases.jpg)

Several examples of classes extending the ```IPredictor``` class can be seen in the [testTimeAugmentation.py](testTimeAugmentation.py) file. Namely, it is necessary to define a class with a ```predict``` method that takes as input the path to a folder containing the images, and stores the predictions in the Pascal VOC format in the same folder. Once this new class has been created, it can be applied both for the ensemble of models and for TTA. 

### Available models 

Currently, they library can work with models constructed with the following models:
- YOLO models constructed with the [Darknet](https://pjreddie.com/darknet/yolo/) library. To use models constructed with this library create an object of the ```DarknetYoloPred``` class. The constructor of this class takes as input the path to the weights of the model, the path to the file with the names of the classes for the model, and the configuration file. 
- Faster-RCNN models constructed with the [MxNet](https://gluon-cv.mxnet.io/) library. To use models constructed with this library create an object of the ```MXnetFasterRCNNPred``` class. The constructor of this class takes as input the path to the weights of the model and the path to the file with the names of the classes for the model.
- SSD models constructed with the [MxNet](https://gluon-cv.mxnet.io/) library. To use models constructed with this library create an object of the ```MXnetSSD512Pred``` class. The constructor of this class takes as input the path to the weights of the model and the path to the file with the names of the classes for the model.
- YOLO models constructed with the [MxNet](https://gluon-cv.mxnet.io/) library. To use models constructed with this library create an object of the ```MXnetYoloPred``` class. The constructor of this class takes as input the path to the weights of the model and the path to the file with the names of the classes for the model.
- Retinanet models with the Resnet 50 backbone constucted with the [Keras Retinanet](https://github.com/fizyr/keras-retinanet) library. To use models constructed with this library create an object of the ```RetinaNetResnet50Pred``` class. The constructor of this class takes as input the path to the weights of the model and the path to the file with the names of the classes for the model.
- MaskRCNN models with the Resnet 101 backbone constucted with the [Keras MaskRCNN](https://github.com/matterport/Mask_RCNN/) library. To use models constructed with this library create an object of the ```MaskRCNNPred``` class. The constructor of this class takes as input the path to the weights of the model and the path to the file with the names of the classes for the model.

You can see several examples of these models in the [notebook for ensembling models](https://colab.research.google.com/drive/1Tg9WaI_Cd-lPXDMuj6tHDlqakxo4-CLK).

## Experiments
Several experiments were conducted to test this library and the results are presented in the [article](https://drive.google.com/file/d/1ku8X8lHs6lethEa5Adhj7frzV44NTbl4/view?usp=sharing). Here, we provide the datasets and models used for those experiments.
### Pascal VOC

For the experiments of Section 4.1 of the paper, we employed the [test set](https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar) of [The PASCAL Visual Object Classes Challenge](http://host.robots.ox.ac.uk/pascal/VOC/); and, the [pre-trained models](https://gluon-cv.mxnet.io/model_zoo/detection.html#pascal-voc) provided by the [MXNet library](https://gluon-cv.mxnet.io).


### Stomata

For the experiments of Section 4.2 of the paper, we employed two stomata datasets:
- [A fully annotated dataset]().
- [A partially annotated dataset]().

Using these datasets, we trained YOLO models using the [Darknet framework](https://pjreddie.com/darknet/yolo/):
- [Model trained using the fully annotated dataset](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EakmwB-DyndNtqjy44QPkV8BiWSRr1IdzH3HMUXqJ-CZDg?e=dbrjb3).
- [Model trained with the partially annotated dataset using data distillation, all the transformations, and affirmative strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EXKT7thc4xRMj9_XpI_W2CQBaGH1qLxoeRijoRkoSE0POg?e=LQB0Zw).
- [Model trained with the partially annotated dataset using data distillation, all the transformations, and consensus strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EZjK7mUFw7VKmxCw09NMtrkBFHQhQ7DVjsCYiv8cl4uCHw?e=oSjG1a).
- [Model trained with the partially annotated dataset using data distillation, all the transformations, and unanimous strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EbOnYhJQnQNFhzjYqvO7bZgBdWRXRthJ0objTnDtHKBG6w?e=vL8WlA).
- [Model trained with the partially annotated dataset using data distillation, colour transformations, and affirmative strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/Ea4UPMu2xgJOt3cCwmheM8MBRCNIzZGec3u12B6FHpPYXw?e=1SVTOt).
- [Model trained with the partially annotated dataset using data distillation, colour transformations, and consensus strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EQOYgVS68QtJlBDatQsp0SMBl-MuY_efkkGAN1hti2vfSw?e=Q0mj0T).
- [Model trained with the partially annotated dataset using data distillation, colour transformations, and unanimous strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EdQAX0hpDZdBnSt00w8kQRIB9ASjBVk3DfMq9DYOPHioOQ?e=84Q5T7).
- [Model trained with the partially annotated dataset using data distillation, flip transformations, and affirmative strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EQrfZZBG-6tEgtaauu4DAS8BmTH-PvUYZtJbcelu6sXJwA?e=UMsIAk).
- [Model trained with the partially annotated dataset using data distillation, flip transformations, and consensus strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EXqjV82vSLlDqEFAPhnVjBMB88gHC6vnRusG8iquHjZZ1A?e=eypIoM).
- [Model trained with the partially annotated dataset using data distillation, flip transformations, and unanimous strategy](https://unirioja-my.sharepoint.com/:u:/g/personal/ancasag_unirioja_es/EcNU8c-5LuxHh3jx0AqGgrgBTNzLqSkU3FzLkxCPSqEO7g?e=cKEeWg).


### Tables
For the experiments of Section 4.3 of the paper, we employed two table datasets:
- [The ICDAR 2013 datset](https://www.dropbox.com/s/ky97l6zvaiv51ep/icdar2013.zip?dl=1).
- [The Word part of the TableBank dataset](https://github.com/doc-analysis/TableBank).

Using these datasets, we trained several models for the ICDAR 2013:
- [Mask RCNN model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EakIU3Ex0_xAkCB2tlHwYr0B7NFsGzPzvIOyVbibMzS_wA?e=deO0eX) trained using the [Keras MaskRCNN](https://github.com/matterport/Mask_RCNN/) 
- [SSD Model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/Ee1Spx8pFkZIoPmPBwMAVroB8gmxhKz2ZHYpaPk3Dd-2-Q?e=ZPR2QX) trained using the [MXNet framework](https://gluon-cv.mxnet.io).
- [YOLO Model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/ESuy3N0fOkVNra63HNtPm8kBVc6qq86yPKeTsAvHdY9r6Q?e=1FuNLR) trained using the [Darknet framework](https://pjreddie.com/darknet/yolo/).


And also applying model distillation:
- [Mask RCNN model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EVkhZOnvWuRJuACkEfBt3tMBhVfxncchunZYvmuImOEc2A?e=gpayiE) using affirmative strategy.
- [Mask RCNN model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EVur0-Ksy51Ikqlc7xZrPZoBtyS2iL7pCF_Fm7Nwf1bNAQ?e=OG4Tyj) using consensus strategy.
- [Mask RCNN model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/Ea2J-_ZthxZCscBUHbShms8BWJNgiFFBSBLsnSNKv0ckZA?e=eg9W5e) using unanimous strategy.
- [SSD model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EYqgC3Sp1jtDoDNnagKSiTsB3yRHfGr9EGUK4LYk6MEeCw?e=XSXOcV) using affirmative strategy.
- [SSD model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EYbEwuvXWg1CgIwiOleApYYBfA8TPlWIUrGgwytfoWd5JQ?e=ilBID3) using consensus strategy.
- [SSD model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EZRaBdzsODFNscizYwOqe3ABm4yqBLqu_GahhlIByhvMpQ?e=4t6iO3) using unanimous strategy.
- [YOLO model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EX9jQC6iOqhBvc1qGCbr8xcBYq8WgPut2TmHqszkESNh1g?e=dmQmbC) using affirmative strategy.
- [YOLO model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/Ebn2UwG7DvtEsmFTTTTzGMUBW2BPUeMnE5oTCIY720qyEw?e=DGHYbl) using consensus strategy.
- [YOLO model](https://unirioja-my.sharepoint.com/:u:/g/personal/joheras_unirioja_es/EVAQYXSVAzBOnxviio5vpAEBDbICFfjgHedtvarVFanYVA?e=FF4xcI) using unanimous strategy.


## Citation

Use this bibtex to cite this work:

```
@misc{CasadoGarcia19,
  title={Ensemble Methods for Object Detection},
  author={A. Casado-García and J. Heras},
  year={2019},
  note={\url{https://github.com/ancasag/ensembleObjectDetection}},
}
```

## Acknowledgments
This work was partially supported by Ministerio de Economía y Competitividad [MTM2017-88804-P], Ministerio de Ciencia, Innovación y Universidades [RTC-2017-6640-7], Agencia de Desarrollo Económico de La Rioja [2017-I-IDD-00018], and the computing facilities of Extremadura Research Centre for Advanced Technologies (CETA-CIEMAT), funded by the European Regional Development Fund (ERDF). CETA-CIEMAT belongs to CIEMAT and the Government of Spain. We also thank Álvaro San-Sáez for providing us with the stomata datasets.
