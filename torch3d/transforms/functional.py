import torch
import numpy as np


def _is_numpy(pcd):
    return isinstance(pcd, np.ndarray)


def _is_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def to_tensor(pcd):
    """
    Convert a ``numpy.ndarray`` to `Tensor`.
    """

    if not _is_numpy(pcd) or pcd.ndim == 1:
        return pcd
    if pcd.ndim == 2:
        tensor = torch.tensor(pcd.transpose(1, 0))
        return tensor


def to_point_cloud(tensor):
    """
    Convert a ``Tensor`` to ``numpy.ndarray``.
    """

    if not _is_tensor(tensor):
        raise TypeError('Input should be a Tensor. Got {}'.format(type(tensor)))

    if tensor.ndim == 2:
        pcd = tensor.transpose(1, 0).cpu().numpy()
    return pcd


def select(pcd, indices):
    """
    Select samples from a ``numpy.ndarray``
    """

    if not _is_numpy(pcd):
        return pcd
    return pcd[indices]
