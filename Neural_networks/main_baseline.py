import keras.datasets
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader

np.random.seed(44)
torch.manual_seed(44)


class Module_(nn.Module):
    def __init__(self):
        super(Module_, self).__init__()

        self.myModel = nn.Sequential(nn.Linear(28*28, 32),
                                     nn.ReLU(),
                                     nn.Linear(32, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 10))

    def forward(self, x):
        return self.myModel(x)


if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = [x.reshape(-1, 28*28) / 255.0 for x in [x_train, x_test]]

    x_train, y_train, x_test, y_test = [torch.from_numpy(
        x) for x in [x_train, y_train, x_test, y_test]]
    train_ds, test_ds = [TensorDataset(x.float(), y.long()) for x, y in [
        [x_train, y_train], [x_test, y_test]]]

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=256)

    myModel = Module_()
    optim = torch.optim.SGD(myModel.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(10):
        correct = 0
        losses = []
        for x, y in train_loader:
            optim.zero_grad()
            out = myModel(x)
            loss = criterion(out, y)
            loss.backward()
            with torch.no_grad():
                optim.step()

        for x, y in test_loader:
            with torch.no_grad():
                out = myModel(x)
                loss = criterion(out, y)
                losses.append(loss.detach().item())
                correct += torch.argmax(out, 1).eq(y).sum().item()
        print(
            f"Epoch {e} - Loss {np.mean(losses)} - Acc {correct / 10000}")
