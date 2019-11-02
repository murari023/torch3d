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
            x = torch.stack([x[b, :, i] for b, i in enumerate(indices)], dim=0)
        p = torch.stack([p[b, i, :] for b, i in enumerate(indices)], dim=0)
        return p, x
