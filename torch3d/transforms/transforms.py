import torch
import numpy as np
from . import functional as F


__all__ = [
    "ToTensor",
    "Shuffle"
]


class Compose(object):
    def __call__(self, *args):
        pass


class ToTensor(object):
    def __call__(self, pcd):
        return F.to_tensor(pcd)


class Shuffle(object):
    def __init__(self):
        self.params = None

    @staticmethod
    def get_params(pcd):
        num_points = pcd.shape[0]
        perm = np.random.permutation(num_points)
        return perm

    def __call__(self, pcd):
        if self.params is None:
            self.params = self.get_params(pcd)
        perm = self.params
        # XXX: Should we reset params?
        self.params = None
        return pcd[perm]
