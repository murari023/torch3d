import torch
import torch.nn as nn


__all__ = [
    'Downsample'
]


class Downsample(nn.Module):
    def __init__(self, num_samples, mode='random'):
        super(Downsample, self).__init__()
        self.num_samples = num_samples
        self.mode = mode

    def forward(self, p, x=None):
        num_points = p.shape[2]
        if self.mode == 'random':
            choice = torch.randperm(num_points)[:self.num_samples]
            q = p[:, :, choice]
            return q, x
