import torch
import numpy as np
from . import functional as F


__all__ = [
    "ToTensor",
]


class ToTensor(object):
    def __call__(self, pcd):
        return F.to_tensor(pcd)
