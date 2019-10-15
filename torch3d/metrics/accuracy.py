import torch
from .metric import Metric


class Accuracy(Metric):
    name = "accuracy"

    def __init__(self, num_classes, transform=None):
        self.transform = transform
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.count = torch.zeros(self.num_classes)
        self.total = torch.zeros(self.num_classes)

    def update(self, output, target):
        if self.transform is not None:
            output, target = self.transform(output, target)
        target = target.view(-1)
        pred = torch.argmax(output, dim=1).view(-1)
        for k in range(self.num_classes):
            indices = (target == k)
            correct = torch.eq(pred[indices], target[indices])
            correct = correct.type(torch.float32)
            self.count[k] += torch.sum(correct)
            self.total[k] += torch.sum(indices)

    def score(self):
        val = torch.sum(self.count) / torch.sum(self.total)
        return val.item()
