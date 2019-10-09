import torch
import torch.nn as nn
from collections import OrderedDict


__all__ = ['Downsample']


class Downsample(nn.Module):
    def __init__(self, num_points, mode='random'):
        super(Downsample, self).__init__()
        self.num_points = num_points
        self.mode = mode

    def forward(self, p, x=None):
        if self.mode == 'random':
            choice = torch.randperm(p
