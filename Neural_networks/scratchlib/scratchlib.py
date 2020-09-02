from IPython.core.debugger import set_trace
import math

import numpy as np
from tensorflow.keras.utils import to_categorical

np.random.seed(44)


def cross_entropy_loss(logits, targets):
    # One hot-enconding targets
    targets_ohe = to_categorical(targets, 10)

    # softmax
    e = np.exp(logits-np.max(logits))
    dist = e / (np.sum(e, 1, keepdims=True))

    # Cross entropy
    E_average = -np.sum(targets_ohe*np.log(dist)) / dist.shape[0]
    delta_logits = dist - targets_ohe

    # trick to reduce calculations
    # E = -np.sum(targets_ohe * (logits -
    #                            np.log(epsilon + np.sum(np.exp(logits) + epsilon, axis=1, keepdims=True))))
    # E_average = E / logits.shape[0]
    return E_average, delta_logits


class LinearLayer():
    def __init__(self, input_features, neurons, bias=True, activation='relu', **kwargs):
        super(LinearLayer, self).__init__()

        self.type = 'linear'
        self.input_features = input_features
        self.neurons = neurons
        self.bias = bias
        self.activation = activation

        self.is_training = True

        for k, v in kwargs:
            setattr(self, k, v)

        self.w = np.random.randn(input_features, neurons) * np.sqrt(2/neurons)
        if bias:
            # self.b = np.random.randn(neurons)
            self.b = np.zeros(neurons)

    def forward(self, x):
        self.x = x

        self.matmul = np.matmul(self.x, self.w)

        if self.bias:
            self.z = self.matmul + self.b
        else:
            self.z = self.matmul

        # if self.activation == 'relu':
        #     self.a = np.maximum(0, self.z)

        return self.z

    def backward(self, x):

        # self.delta_z = np.array(delta_E, copy=True)
        # self.delta_z[self.z <= 0] = 0

        self.delta_b = np.sum(x * 1, axis=0)
        self.delta_matmul = x * 1

        self.delta_w = np.transpose(
            np.matmul(np.transpose(self.delta_matmul), self.x))
        self.delta_x = np.matmul(self.delta_matmul, np.transpose(self.w))

        return self.delta_x


class ReLU():
    def __init__(self):
        super(ReLU, self).__init__()
        self.type = 'relu'

    def forward(self, x):
        self.x = x
        return np.maximum(0, self.x)

    def backward(self, x):
        x[self.x <= 0] = 0
        return x


class Dropout():
    def __init__(self, p: float):
        super(Dropout, self).__init__()
        if p <= 0 or p >= 1:
            raise Exception('0 < p < 1')
        else:
            self.p = p

        self.is_training = True
        self.type = 'dropout'

    def forward(self, x):
        if self.is_training:
            p_matrix = np.random.random((x.shape[0], 1))
            self.p_matrix = np.where(p_matrix < self.p, 0, 1)
            X = np.dot(x * p_matrix, 1/1-self.p)
            return x
        else:
            return x

    def backward(self, x):
        x = np.dot(x, 1/1-self.p)
        x = x * self.p_matrix
        return x


class BatchNorm1d():
    def __init__(self, num_features: int, momentum=0.1, epsilon=1e-5,  **kwargs):
        super(BatchNorm1d, self).__init__()

        self.type = "bn1d"
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = np.zeros(num_features)
        self.beta = np.ones(num_features)

        # running variables
        self.running_mean = np.zeros(num_features)
        self.running_std = np.zeros(num_features)

        self.is_training = True

        for k, v in kwargs:
            setattr(self, k, v)

    def forward(self, x):
        if self.is_training:
            batch_mean = np.mean(x, axis=0)
            batch_std = np.std(x, axis=0)

            self.running_mean = self.momentum * \
                self.running_mean + (1-self.momentum) * batch_mean
            self.running_std = self.momentum * \
                self.running_std + (1-self.momentum) * batch_std

            self.denominator = np.sqrt(batch_std + self.epsilon)

            self.x_norm = (x - batch_mean) / self.denominator
            return self.gamma * self.x_norm + self.beta
        else:
            self.x_norm = (x-self.running_mean) / \
                np.sqrt(self.running_std + self.epsilon)
            return self.gamma * self.x_norm + self.beta

    def backward(self, x):
        self.delta_beta = (x * 1).sum(axis=0)
        self.delta_matmul = x * 1
        self.delta_gamma = (self.delta_matmul * self.x_norm).sum(axis=0)

        self.delta_x_norm = self.delta_matmul * self.gamma

        self.delta_x = (1. / x.shape[-1]) / self.denominator * (x.shape[-1] * self.delta_x_norm -
                                                                self.delta_x_norm.sum(axis=0) - self.x_norm * (self.delta_x_norm * self.x_norm).sum(axis=0))
        return self.delta_x


class Model():
    def __init__(self, layers: [], **kwargs):
        super(Model, self).__init__()
        self.layers = layers
        self.delta_E = None

        for k, v in kwargs:
            setattr(self, k, v)

    def train(self, train=True):
        for layer in self.layers:
            layer.is_training = train

    def forward(self, x, y=None):

        out = self.layers[0].forward(x)
        for layer in self.layers[1:]:
            out = layer.forward(out)
        if y is not None:
            _, self.delta_E = cross_entropy_loss(out, y)

        return out

    def backward(self):
        for layer in reversed(self.layers):
            # print(layer.type)
            # print(np.sum(self.delta_E))
            # if layer.type == 'linear':
            self.delta_E = layer.backward(self.delta_E)

    def evaluate(self, x, y):
        self.train(False)

        logits = self.forward(x)
        loss, _ = cross_entropy_loss(logits, y)

        accuracy = (np.argmax(logits, axis=1).__eq__(y).sum() / len(y)) * 100

        print(f"Accuracy {accuracy:.2f} | Loss {loss:.4f}")
        self.train(True)

    def update_weights(self, optimizer='SGD', lr=1e-3):
        for layer in self.layers:
            if layer.type == 'linear':
                layer.w -= lr * layer.delta_w
                layer.b -= lr * layer.delta_b
            elif layer.type == "bn1d":
                layer.gamma -= lr * layer.delta_gamma
                layer.beta -= lr*layer.delta_beta
            else:
                pass
