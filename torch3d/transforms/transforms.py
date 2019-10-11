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
