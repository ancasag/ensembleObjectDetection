import abc
from abc import ABC

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
    
    def __init__(self,weightPath,fichNames, fichCfg, conf):
        IPredictor.__init__(self, weightPath)
        self.fichNames = fichNames
        self.fichCfg = fichCfg
        self.conf = conf

    def predict(self, imgPath, output):
        import detect
        detect.mainDataset(imgPath, output, self.pathPesos, self.fichNames, self.fichCfg, self.conf)


class MXnetYoloPred(IPredictor):

    def __init__(self,weightPath,classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes=classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch
        predict_batch.mainDataset(imgPath, output,'yolo3_darknet53_custom', self.pathPesos, self.classes, self.conf)

class MXnetSSD512Pred(IPredictor):

    def __init__(self,weightPath,classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes=classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch
        predict_batch.mainDataset(imgPath, output,'ssd_512_resnet50_v1_custom',self.pathPesos, self.classes, self.conf)

class MXnetFasterRCNNPred(IPredictor):
    
    def __init__(self,weightPath,classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes=classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch
        predict_batch.mainDataset(imgPath, output,'faster_rcnn_resnet50_v1b_custom', self.pathPesos, self.classes, self.conf)

class RetinaNetResnet50Pred(IPredictor):
    
    def __init__(self,weightPath,classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes=classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch_retinanet
        predict_batch_retinanet.mainDataset(imgPath, output,'resnet50_v1', self.pathPesos, self.classes, self.conf)

class MaskRCNNPred(IPredictor):
    
    def __init__(self,weightPath,classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes=classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch_rcnn
        predict_batch_rcnn.mainDataset(imgPath, output, self.pathPesos, self.classes, self.conf)


class Efficient(IPredictor):

    def __init__(self, weightPath, classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes = classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch_efficient
        predict_batch_efficient.mainDataset(imgPath, output, self.pathPesos, self.classes, self.conf)

class FSAF(IPredictor):

    def __init__(self, weightPath, classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes = classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch_FSAF
        predict_batch_FSAF.mainDataset(imgPath, output, self.pathPesos, self.classes, self.conf)

class FCOS(IPredictor):

    def __init__(self, weightPath, classes, conf):
        IPredictor.__init__(self, weightPath)
        self.classes = classes
        self.conf = conf

    def predict(self, imgPath, output):
        import predict_batch_FCOS
        predict_batch_FCOS.mainDataset(imgPath, output, self.pathPesos, self.classes, self.conf)
