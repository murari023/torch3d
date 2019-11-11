import torch
import numpy as np
from . import functional as F


__all__ = [
    "Compose",
    "ToTensor",
    "Shuffle",
    "RandomSample",
]


class Compose(object):
    """
    Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.Shuffle(),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pcd):
        for t in self.transforms:
            pcd = t(pcd)
        return pcd


class ToTensor(object):
    def __call__(self, pcd):
        return F.to_tensor(pcd)


class Shuffle(object):
    @staticmethod
    def get_params(pcd):
        n = len(pcd)
        assert n > 0
        return np.random.permutation(n)

    def __call__(self, pcd):
        perm = self.get_params(pcd)
        return F.shuffle(pcd, perm)


class RandomSample(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    @staticmethod
    def get_params(pcd, num_samples):
        n = len(pcd)
        assert n > 0
        if n >= num_samples:
            samples = np.random.choice(n, num_samples, replace=False)
        else:
            m = num_samples - n
            samples = np.random.choice(n, m, replace=True)
            samples = list(range(n)) + list(samples)
        return samples

    def __call__(self, pcd):
        samples = self.get_params(pcd, self.num_samples)
        return F.sample(pcd, samples)
