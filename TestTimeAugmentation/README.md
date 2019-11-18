# Test-Time Augmentation

This code allows us to apply the ensemble method given a folder containing folders with the corresponding xmls and indicating 
the option we want to use.

### Ensemble Options
You can be taken using three different voting strategies:
*   Affirmative. This means that whenever one of the methods that produce the 
initial predictions says that a region contains an object, such a detection is considered as valid.
*   Consensus. This means that the majority of the initial methods must agree to consider that a region contains an object. The consensus strategy is analogous to the majority voting strategy commonly applied in ensemble methods for images classification.
*   Unanimous. This means that all the methods must agree to consider that a region contains an object.

### Executed
To execute the code we use the following instruction.
```bash
python main.py -d pathOfDataset -o option
```
### Example
An example of its use would be the following. Given the examples folder that in turn contains folders with the xmls files, we would execute the following instruction (suppose we choose the consensus option).
```bash
python main.py -d examples -o consensus
```
