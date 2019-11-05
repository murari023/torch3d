import torch
import torch.nn as nn
import torch3d.nn.functional as F


__all__ = [
    "RandomPointSample",
    "FarthestPointSample",
]


class RandomPointSample(nn.Module):
    def __init__(self, num_samples):
        super(RandomPointSample, self).__init__()
        self.num_samples = num_samples

    def forward(self, p, x=None):
        indices = F.random_point_sample(p, self.num_samples)
        if x is not None:
            x = x[:, :, indices]
        p = p[:, indices]
        return p, x


class FarthestPointSample(nn.Module):
    def __init__(self, num_samples):
        super(FarthestPointSample, self).__init__()
        self.num_samples = num_samples

    def forward(self, p, x=None):
        indices = F.farthest_point_sample(p, self.num_samples)
        if x is not None:
            x = x.permute(0, 2, 1)
            x = F.gather_points(x, indices)
            x = x.permute(0, 2, 1)
        p = F.gather_points(p, indices)
        return p, x
