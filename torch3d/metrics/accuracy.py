import torch
from .metric import Metric


__all__ = ["BinaryAccuracy", "Accuracy"]


class BinaryAccuracy(Metric):
    name = "accuracy"

    def __init__(self, activation=torch.sigmoid):
        self.activation = activation
        self.reset()

    def reset(self):
        self.count = 0
        self.total = 0

    def update(self, output, target):
        if self.activation is not None:
            output = self.activation(output)
        output = output.view(-1) >= 0.5
        target = target.view(-1) >= 0.5
        correct = torch.eq(output, target).type(torch.float32)
        self.count += torch.sum(correct)
        self.total += target.numel()

    def score(self):
        value = self.count / self.total
        return value.item()

    def mean(self):
        return self.score()


class Accuracy(Metric):
    name = "accuracy"

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.count = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)

    def update(self, x, y):
        x = torch.argmax(x, dim=1).view(-1)
        y = y.view(-1)
        for k in range(self.num_classes):
            index = y == k
            correct = torch.eq(x[index], y[index]).type(torch.float32)
            self.count[k] += torch.sum(correct)
            self.total[k] += torch.sum(index)

    def score(self):
        value = torch.sum(self.count) / torch.sum(self.total)
        return value.item()

    def mean(self):
        value = torch.mean(self.count / self.total)
        return value.item()
