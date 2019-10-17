import torch
from .metric import Metric


class Jaccard(Metric):
    name = "jaccard"

    def __init__(self, num_classes, smooth=0.0):
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self):
        self.inter = torch.zeros(self.num_classes) + self.smooth
        self.union = torch.zeros(self.num_classes) + self.smooth

    def update(self, x, y):
        x = torch.argmax(x, dim=1).view(-1)
        y = target.view(-1)
        for k in range(self.num_classes):
            a = (x == k)
            b = (y == k)
            self.inter[k] += torch.sum(a & b)
            self.union[k] += torch.sum(a | b)

    def score(self):
        value = torch.mean(self.inter / self.union)
        return value.item()
