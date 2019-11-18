# Ensemble Object Detection
In this project, we show the benefits of using an ensemble algorithm that can be applied with any object detection model independently of the underlying algorithm. In addition, our ensemble method has been employed to define a test-time augmentation procedure for object detection models.We have tested our methods with several datasets and algorithms.

### Colab Notebooks for prediction
You can use the methods with the following notebooks.

- [Test Time Augmentation Notebook](https://colab.research.google.com/drive/1T1mn85AedRlaTNHeJW_QeTy0I5wOy14J)
- [Ensemble Model Notebook](https://colab.research.google.com/drive/1Tg9WaI_Cd-lPXDMuj6tHDlqakxo4-CLK)


## Test-Time Augmentation and Model Ensemble
We provides the necessary tools to apply ensemble methods and test-time augmentation (TTA). This open-source library canbe  extended  to  work  with  any  object  detection  model  indepen-dently of the algorithm and framework employed to construct it.
### Tecnhiques of TTA
- "avgBlur": (createTechnique("average_blurring", {"kernel" : 5}), createTechnique("none", {})),
- "bilaBlur": (createTechnique("average_blurring", {"diameter" : 11, "sigmaColor": 21, "sigmaSpace":7}), createTechnique("none", {})),
- "blur": (createTechnique("blurring", {"ksize" : 5}), createTechnique("none", {})),
- "chanHsv": (createTechnique("change_to_hsv",{}), createTechnique("none", {})),
- "chanLab": (createTechnique("blurring", {"ksize" : 5}), createTechnique("none", {})),
- "crop": (createTechnique("crop",{"percentage":0.8,"startFrom": "TOPLEFT"}), createTechnique("none", {})),
- "dropOut": (createTechnique("dropout",{"percentage":0.05}), createTechnique("none", {})),
- "elastic": (createTechnique("elastic",{"alpha":5,"sigma":0.05}), createTechnique("none", {})),
- "histo": (createTechnique("equalize_histogram",{}), createTechnique("none", {})),
- "vflip": (createTechnique("flip", {"flip": 0}), createTechnique("flip", {"flip": 0})),
- "hflip": (createTechnique("flip", {"flip": 1}), createTechnique("flip", {"flip": 1})),
- "hvflip": (createTechnique("flip", {"flip": -1}), createTechnique("flip", {"flip": -1})),
- "gamma": (createTechnique("gamma",{"gamma":1.5}), createTechnique("none", {})),
- "blurGau": (createTechnique("gaussian_blur", {"kernel" : 5}), createTechnique("none", {})),
- "avgNoise": (createTechnique("gaussian_noise", {"mean":0, "sigma":10}), createTechnique("none", {})),
- "invert": (createTechnique("invert",{}), createTechnique("none", {})),
- "medianblur": (createTechnique("median_blur", {"kernel" : 5}), createTechnique("none", {})),
- "none": (createTechnique("none", {}), createTechnique("none", {})),
- "raiseBlue": (createTechnique("raise_blue", {"power" : 0.9}), createTechnique("none", {})),
- "raiseGreen": (createTechnique("raise_green", {"power" : 0.9}), createTechnique("none", {})),
- "raiseHue": (createTechnique("raise_hue", {"power" : 0.9}), createTechnique("none", {})),
- "raiseRed": (createTechnique("raise_red", {"power" : 0.9}), createTechnique("none", {})),
- "raiseSatu": (createTechnique("raise_saturation", {"power" : 0.9}), createTechnique("none", {})),
- "raiseValue": (createTechnique("raise_value", {"power" : 0.9}), createTechnique("none", {})),
- "resize": (createTechnique("resize", {"percentage" : 0.9,"method":"INTER_NEAREST"}), createTechnique("none", {})),
- "rotation10": (createTechnique("rotate", {"angle": 10}), createTechnique("rotate", {"angle": -10})),
- "rotation90": (createTechnique("rotate", {"angle": 90}), createTechnique("rotate", {"angle": -90})),
- "rotation180": (createTechnique("rotate", {"angle": 180}), createTechnique("rotate", {"angle": -180})),
- "rotation270": (createTechnique("rotate", {"angle": 270}), createTechnique("rotate", {"angle": -270})),
- "saltPeper": (createTechnique("salt_and_pepper", {"low" : 0,"up":25}), createTechnique("none", {})),
- "sharpen": (createTechnique("sharpen", {}), createTechnique("none", {})),
- "shiftChannel": (createTechnique("shift_channel", {"shift":0.2}), createTechnique("none", {})),
- "shearing": (createTechnique("shearing", {"a":0.5}), createTechnique("none", {})),
- "translation": (createTechnique("translation", {"x":10,"y":10}), createTechnique("none", {}))
    
### Model Ensemble
As we have said before, this open source library can be expanded to work with any object detection model regardless of the algorithm and framework used to build it. As we can see in the following diagram:
![DiagramModels](diagramaClases.jpg)

### Ensemble Options
You can be taken using three different voting strategies:
*   Affirmative. This means that whenever one of the methods that produce the 
initial predictions says that a region contains an object, such a detection is considered as valid.
*   Consensus. This means that the majority of the initial methods must agree to consider that a region contains an object. The consensus strategy is analogous to the majority voting strategy commonly applied in ensemble methods for images classification.
*   Unanimous. This means that all the methods must agree to consider that a region contains an object.

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
