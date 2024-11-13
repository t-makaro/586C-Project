import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TorchNeuralNet(nn.Module):
    def __init__(self, layers):
        super(TorchNeuralNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))

    def load_np_params(self, weights, biases):
        for i, (w, b) in enumerate(zip(weights, biases)):
            self.layers[i].weight.data = torch.tensor(w, dtype=torch.float32)
            self.layers[i].bias.data = torch.tensor(b, dtype=torch.float32)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.sigmoid(layer(x))
        return self.layers[-1](x)

    def evaluate(self, x, y):
        preds = self(x)
        correct = (preds.argmax(dim=1) == y).sum().item()
        return correct / len(y)
