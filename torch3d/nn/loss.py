import torch
import torch.nn as nn
import torch3d.ops as ops


__all__ ["ChamferLoss"]


def ChamferLoss(nn.Module):
    def __init__(self, transform=None):
        super(ChamferLoss, self).__init__()
        self.transform = transform

    def forward(x, y):
        if self.transform is not None:
            x, y = self.transform(x, y)
        return ops.chamfer(x, y)
