# DL Framework Hardcoded

This project was made to completement theory that I learned while following Andrew Ng's [DeepLearning.ai course](https://www.deeplearning.ai/)

The project is a Deep learning framework where it is possible to make custom architectures and train them for various tasks.  
Made while relying solely on libarries/modules like numpy, matplotlib and math

Currently has support limited to classification tasks for images, but it is my intention to add other modules for support of other extended tasks...

## Files

The project contains the following files:

- `func.py`: Contains a combination of functions used internal by various layers, and other end-user functions such as losses, division being clearly stated in the file
- `layers.py`: Contains all necessary layers, and activation functions to build a model
- `model.py`: Has the base model class definition, to be used for defining custom models, training and prediction...
- `requirements.txt`: Dependencies to be installed for running the scripts
- `sandbox.ipynb`: A testing environment used during development, does not hold significance for building custom models
- `templates.ipynb`: Contains templates associated with the expected structure of layers and loss fucntions etc...

## Prerequisites

The required libaries can be installed using the requirements file

Currenlty the Prerequisites include:

-Numpy
-scipy
-ipython


```bash
pip install -r requirements.txt
```

### Current Progress

Currenlty the implemented layers and loss functions include:

-Conv2d
-Linear
-Self Attention
-Sigmoid
-Cross entropy loss using logits

There is a bug with softmax backwards function  (which is also used in self attention) that is being worked on locally.

## Usage

The current relevant files to build the model can be found in model.py, layers.py and func.py.

All classes in the layers file are to be used directly, but most the files in functions files are used internally for different layers.

In the future, once more loss functions, optimizers, and activations are added, futher subdivision will be made into their own respective files...

```python
from model import Model
import func
import layers
```

To define a new model, a list containing the layer information needs to be passed while initializing the model.  
Later, information about learning rate, loss function and optimizer should be passed into the model using .compile method.  
Finally the model can be trained using the .train method, with predictions made with the .predict method.  

The layer information in the list is to be passed in as instances of their respective layer classes.  
They can be thought of lazy versions of themselves, we only need to specify the internal parameters of each layer,  
while the input and output dimensions will be inferred on their own.  