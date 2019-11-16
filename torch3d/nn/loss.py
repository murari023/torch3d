import torch
import torch.nn as nn
import torch3d.nn.functional as F


__all__ = ["ChamferLoss"]


class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(input, target):
        _, _, sqdist1, sqdist2 = F.chamfer_distance(input, target)
        return torch.mean(sqdist1) + torch.mean(sqdist2)
