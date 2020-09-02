from IPython.core.debugger import set_trace
import math

import numpy as np
import keras.datasets
import matplotlib.pyplot as plt

from src import scratchlib as sl

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    img_size = 32*32*3
    x_train, y_train = x_train.reshape(-1, img_size) / 255.0, y_train.flatten()
    x_test, y_test = x_test.reshape(-1, img_size) / 255.0, y_test.flatten()

    # model_layers = [sl.LinearLayer(img_size, 32),
    #                 sl.BatchNorm1d(32),
    #                 sl.ReLU(),
    #                 sl.LinearLayer(32, 64),
    #                 sl.BatchNorm1d(64),
    #                 sl.ReLU(),
    #                 sl.LinearLayer(64, 10)]

    model_layers = [sl.LinearLayer(img_size, 32),
                    # sl.BatchNorm1d(32),
                    sl.ReLU(),
                    sl.LinearLayer(32, 64),
                    # sl.BatchNorm1d(64),
                    sl.ReLU(),
                    sl.LinearLayer(64, 128),
                    # sl.BatchNorm1d(128),
                    sl.ReLU(),
                    sl.LinearLayer(128, 64),
                    # sl.BatchNorm1d(64),
                    sl.ReLU(),
                    sl.LinearLayer(64, 10)]

    # myModel = sl.Model(model_layers)
    # for e in range(30):
    #     batch = 128
    #     for it, i in enumerate(range(0, x_train.shape[0], batch)):
    #         if i + batch > x_train.shape[0]:
    #             batch = x_train.shape[0] - i
    #         logits = myModel.forward(x_train[i:i+batch], y_train[i:i+batch])
    #         myModel.backward()
    #         myModel.update_weights(1e-3)
    #     # myModel.evaluate(x_train, y_train)
    #     myModel.evaluate(x_test, y_test)

    myModel = sl.Model(model_layers)
    x, y = x_train[:30, :], y_train[:30]
    myModel.evaluate(x, y)
    myModel.forward(x, y)
    myModel.backward()
    myModel.update_weights(1e-3)
    myModel.evaluate(x, y)
