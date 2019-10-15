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

    def report(self, names=None):
        print("----------------------------------------------------------------")
        print("{:<25}  {:>10} {:>10} {:>10}".format("Category", "Accuracy", "Correct", "Total"))
        print("================================================================")
        values = self.count / self.total
        if names is None:
            names = list(range(self.num_classes))
        for i in range(self.num_classes):
            print("{:<25}  {:>10.3f} {:>10.0f} {:>10.0f}".format(names[i], values[i], self.count[i], self.total[i]))
        print("================================================================")
        mean = torch.mean(values).item()
        print("Mean accuracy: {:.3f}".format(mean))
        print("Overall accuracy: {:.3f}".format(self.score()))
