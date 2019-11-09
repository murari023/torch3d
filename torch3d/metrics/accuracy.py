import torch
from .metric import Metric


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
            indices = y == k
            correct = torch.eq(x[indices], y[indices]).type(torch.float32)
            self.count[k] += torch.sum(correct)
            self.total[k] += torch.sum(indices)

    def score(self):
        value = torch.sum(self.count) / torch.sum(self.total)
        return value.item()

    def mean(self):
        value = torch.mean(self.count / self.total)
        return value.item()
