from clodsa.augmentors.augmentorFactory import createAugmentor
from clodsa.transformers.transformerFactory import transformerGenerator
import techniques
import os

def clasification(imgFolder,technique):
    technique2 = technique
    technique = techniques.techniques[technique][0]
    augmentor = createAugmentor("classification","folders","folders","linear",imgFolder,
                                {"outputPath":imgFolder+'/../salida/'+technique2+"/"})
    transformer = transformerGenerator("classification")
    augmentor.addTransformer(transformer(technique))
    augmentor.applyAugmentation()





def detection(imgFolder,technique):
    technique2 = technique
    technique = techniques.techniques[technique][1]
    os.mkdir(imgFolder+"/tmp")
    augmentor = createAugmentor("detection","pascalvoc","pascalvoc","linear",imgFolder,
                                {"outputPath":imgFolder+"/tmp"})
    transformer = transformerGenerator("detection")
    augmentor.addTransformer(transformer(technique))
    augmentor.applyAugmentation()