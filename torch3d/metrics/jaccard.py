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

    def report(self, names=None):
        print("----------------------------------------------------------------")
        print("{:<25}  {:>10} {:>10} {:>10}".format("Category", "IoU", "Inter.", "Union"))
        print("================================================================")
        values = (self.inter + self.smooth) / (self.union + self.smooth)
        if names is None:
            names = list(range(self.num_classes))
        for i in range(self.num_classes):
            print("{:<25}  {:>10.3f} {:>10.0f} {:>10.0f}".format(names[i], values[i], self.inter[i], self.union[i]))
        print("================================================================")
        mean = torch.mean(values).item()
        print("Mean IoU: {:.3f}".format(mean))
