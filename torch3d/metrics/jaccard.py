import torch
from .metric import Metric


class Jaccard(Metric):
    name = "jaccard"

    def __init__(self, num_classes, transform=None):
        self.transform = transform
        self.num_classes = num_classes
        self.smooth = 1.0
        self.reset()

    def reset(self):
        self.inter = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)

    def update(self, output, target):
        if self.transform is not None:
            output, target = self.trasnform(output, target)
        target = target.view(-1)
        predict = torch.argmax(output, dim=1).view(-1)
        for k in range(self.num_classes):
            a = (predict == k)
            b = (target == k)
            self.inter[k] += torch.sum(a & b)
            self.union[k] += torch.sum(a | b)

    def score(self):
        values = (self.inter + self.smooth) / (self.union + self.smooth)
        return torch.mean(values).item()
