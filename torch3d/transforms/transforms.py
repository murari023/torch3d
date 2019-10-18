import random
import torch
import numpy as np
import torch3d.transforms.functional as F


__all__ = [
    "Compose",
    "ToTensor",
    "Shuffle",
    "RandomSample",
    "Jitter"
]


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, points, target):
        for t in self.transforms:
            points, target = t(points, target)
        return points, target


class ToTensor(object):
    def __call__(self, points, target):
        points = F.to_tensor(points)
        return points, target


class Shuffle(object):
    def __call__(self, points, target):
        n = points.shape[0]
        perm = np.random.permutation(n)
        return points[perm], target


class RandomSample(object):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, points, target):
        n = points.shape[0]
        samples = random.sample(range(n), self.num_samples)
        return points[samples], target


class Jitter(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, points, target):
        points = F.jitter(points, self.sigma)
        return points, target


class RandomRotate(object):
    pass
