import random
import torch
import numpy as np
from . import functional as F


__all__ = [
    "Compose",
    "First",
    "ToTensor",
    "Shuffle",
    "RandomDownsample"
]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class First(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x, *args):
        x = self.transform(x)
        return (x,) + args


class ToTensor(object):
    def __call__(self, pcd):
        return F.to_tensor(pcd)


class Shuffle(object):
    def __init__(self):
        self.params = None

    @staticmethod
    def get_params(pcd):
        num_points = pcd.shape[0]
        return np.random.permutation(num_points)

    def __call__(self, pcd):
        if self.params is None:
            self.params = self.get_params(pcd)
        perm = self.params
        # XXX: Should we reset params?
        self.params = None
        return pcd[perm]


class RandomDownsample(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.params = None

    @staticmethod
    def get_params(pcd, num_samples):
        num_points = pcd.shape[0]
        return random.sample(range(num_points), num_samples)

    def __call__(self, pcd):
        if self.params is None:
            self.params = self.get_params(pcd, self.num_samples)
        samples = self.params
        self.params = None
        return pcd[samples]
