# Test-Time Augmentation and Ensemble models

This code allows us to apply the ensemble method and test-time Augmentation with the ensemble methos given a folder containing images.

### Ensemble Options
You can be taken using three different voting strategies:
*   Affirmative. This means that whenever one of the methods that produce the 
initial predictions says that a region contains an object, such a detection is considered as valid.
*   Consensus. This means that the majority of the initial methods must agree to consider that a region contains an object. The consensus strategy is analogous to the majority voting strategy commonly applied in ensemble methods for images classification.
*   Unanimous. This means that all the methods must agree to consider that a region contains an object.

### Executed
To execute the code we use the following instruction.

*   In case of applying Test-Time Augmentation:
```bash
python mainTTA.py -d pathOfDataset -o option
```
*   In case of applying Ensemble models:
```bash
python mainModel.py -d pathOfDataset -o option
```
### Example
We can see an example of the use of both in the notebooks found on the home page.
