# Implementation of neural networks from scratch

## Done/TODO

### Implemnted so far:
- General: Create a model from a specific architecture, train and evaluate it;
- Layers: Fully connected, ReLU, BatchNorm1d, Dropout;

### Todo:
- Option to train model with other optimizers;
- Add support to train Convolutional Neural Networks;

## Example Code

Here's a minimal example of defining 3-layer neural network and training it to classify the MNIST dataset:

```python
# imports
import numpy as np
import keras.datasets

from scratchlib import scratchlib as sl

if __name__ == '__main__':
    # get data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    img_size = 28*28
    # preprocess data
    x_train, y_train = x_train.reshape(-1, img_size) / 255.0, y_train.flatten()
    x_test, y_test = x_test.reshape(-1, img_size) / 255.0, y_test.flatten()

    model_layers = [sl.LinearLayer(img_size, 32),
                    sl.ReLU(),
                    sl.LinearLayer(32, 64),
                    sl.ReLU(),
                    sl.LinearLayer(64, 10)]

    # epochs=5, batch_size=128, lr=1e-3
    myModel = sl.Model(model_layers)
    for e in range(5):
        batch = 128
        for it, i in enumerate(range(0, x_train.shape[0], batch)):
            if i + batch > x_train.shape[0]:
                batch = x_train.shape[0] - i
            logits = myModel.forward(x_train[i:i+batch], y_train[i:i+batch])
            myModel.backward()
            myModel.update_weights(1e-3)
        myModel.evaluate(x_test, y_test)
```
## References

[CS231n Classes](https://cs231n.github.io/)

[Vector, Matrix, Tensor Derivatives](http://cs231n.stanford.edu/vecDerivs.pdf)

[Batch Normalization Backpropagation](https://kevinzakka.github.io/2016/09/14/batch_normalization/)


## Known issues
- Number overflows on exp. operations and divisions by zero;

