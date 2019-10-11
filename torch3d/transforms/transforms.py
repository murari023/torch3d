import torch
import numpy as np
from . import functional as F


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
        sample = F.to_tensor(sample)
        target = torch.from_numpy(np.asarray(target, dtype=np.int64))
        return sample, target


class Shuffle(object):
    def __call__(self, sample, target):
        n = len(sample)
        choice = np.random.permutation(n)
        sample = sample[choice]
        if isinstance(target, np.ndarray):
            target = target[choice]
        return sample, target


class Downsample(object):
    def __init__(self, num_samples, mode='random'):
        self.num_samples = num_samples
        self.mode = mode

    def __call__(self, sample, target):
        if self.mode == 'random':
            n = len(sample)
            choice = np.random.choice(n, self.num_samples, replace=False)

        sample = sample[choice]
        if isinstance(target, np.ndarray):
            target = target[choice]
        return sample, target
