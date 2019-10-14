import torch
import numpy as np
from . import functional as F


__all__ = [
    "Compose",
    "ToTensor",
    "Shuffle",
    "Downsample",
    "RandomRotate",
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
    def __init__(self, num_samples, mode="random"):
        self.num_samples = num_samples
        self.mode = mode

    def __call__(self, sample, target):
        if self.mode == "random":
            n = len(sample)
            choice = np.random.choice(n, self.num_samples, replace=False)

        sample = sample[choice]
        if isinstance(target, np.ndarray):
            target = target[choice]
        return sample, target


class Jitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample, target):
        sample = F.jitter(sample, sigma, clip)
        return sample, target


class Rotate(object):
    def __init__(self, axis="z"):
        if axis == "x":
            self.axis = np.array([1.0, 0.0, 0.0])
        elif axis == "y":
            self.axis = np.array([0.0, 1.0, 0.0])
        elif axis == "z":
            self.axis = np.array([0.0, 0.0, 1.0])
        else:
            self.axis = axis

    def __call__(self, sample, target):
        angle = np.random.uniform(-np.pi, np.pi)
        sample = F.rotate(sample, angle, axis)
        return sample, target
