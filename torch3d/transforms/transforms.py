import torch
import numpy as np
from . import functional as F


__all__ = [
    'Compose',
    'ToTensor',
    'Shuffle',
    'Downsample'
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

    def __call__(self, sample, target):
        for t in self.transforms:
            sample, target = t(sample, target)
        return sample, target


class ToTensor(object):
    def __call__(self, sample, target):
        return F.to_tensor(sample), target

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Shuffle(object):
    def __call__(self, sample, target):
        indices = np.random.permutation(sample.shape[0])
        sample = F.select(sample, indices)
        target = F.select(target, indices)
        return sample, target


class Downsample(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, sample, target):
        indices = np.random.choice(len(sample), self.num_samples, replace=False)
        sample = F.select(sample, indices)
        target = F.select(target, indices)
        return sample, target
