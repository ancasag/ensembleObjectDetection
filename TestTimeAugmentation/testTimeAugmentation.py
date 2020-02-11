import abc
from abc import ABC
import os

#abstract class
class IPredictor(ABC):
    #constructor
    def __init__(self, weightPath):
        self.pathPesos = weightPath

    @abc.abstractmethod
    def predict(self,imgPath):
        pass

#heritage
class DarknetYoloPred(IPredictor):
    import detect
    def __init__(self,weightPath,fichNames, fichCfg):
        IPredictor.__init__(self, weightPath)
        self.fichNames = fichNames
        self.fichCfg = fichCfg

    def predict(self, imgPath, output):
        detect.mainDataset(imgPath, output, self.pathPesos, self.fichNames, self.fichCfg)

class MXnetYoloPred(IPredictor):
    import predict_batch
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output):
        predict_batch.mainDataset(imgPath, output,'yolo3_darknet53_custom', self.pathPesos, self.classes)

class MXnetSSD512Pred(IPredictor):
    import predict_batch
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output):
        predict_batch.mainDataset(imgPath, output,'ssd_512_resnet50_v1_custom',self.pathPesos, self.classes)

class MXnetFasterRCNNPred(IPredictor):
    import predict_batch
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output):
        predict_batch.mainDataset(imgPath, output,'faster_rcnn_resnet50_v1b_custom', self.pathPesos, self.classes)

class RetinaNetResnet50Pred(IPredictor):
    import predict_batch_retinanet
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output):
        predict_batch_retinanet.mainDataset(imgPath, output,'resnet50_v1', self.pathPesos, self.classes)

class MaskRCNNPred(IPredictor):
    import predict_batch_rcnn
    def __init__(self,weightPath,classes):
        IPredictor.__init__(self, weightPath)
        self.classes=classes

    def predict(self, imgPath, output):
        predict_batch_rcnn.mainDataset(imgPath, output, self.pathPesos, self.classes)
 
class Efficient(IPredictor):
    import predict_batch_efficient
    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output):
        predict_batch_efficient.mainDataset(imgPath, output, self.pathPesos, self.classes)

class FSAF(IPredictor):
    import predict_batch_FSAF
    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output):
        predict_batch_FSAF.mainDataset(imgPath, output, self.pathPesos, self.classes)

class FCOS(IPredictor):
    import predict_batch_FCOS
    def __init__(self, weightPath, classes):
        IPredictor.__init__(self, weightPath)
        self.classes = classes

    def predict(self, imgPath, output):
        predict_batch_FCOS.mainDataset(imgPath, output, self.pathPesos, self.classes)
