import torch
import torch.nn as nn
import torch3d.nn.functional as F


__all__ = ["Downsample"]


class Downsample(nn.Module):
    def __init__(self, num_samples, mode="random"):
        super(Downsample, self).__init__()
        self.num_samples = num_samples
        self.mode = mode

    def forward(self, p, x=None):
        if self.mode == "random":
            return F.random_sample(p, x, self.num_samples)
