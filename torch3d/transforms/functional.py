import torch
import numpy as np


def _is_numpy(pcd):
    return isinstance(pcd, np.ndarray)


def _is_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def _is_numpy_pcd(pcd):
    return pcd.ndim == 2


def to_tensor(pcd):
    """
    Convert a ``numpy.ndarray`` point cloud to `Tensor`.

    """

    if not _is_numpy(pcd):
        raise TypeError("pcd should be an ndarray. Got {}.".format(type(pcd)))

    if not _is_numpy_pcd(pcd):
        raise ValueError(
            "pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim)
        )

    pcd = torch.tensor(pcd.T)
    return pcd


def sample(pcd, samples):
    if not _is_numpy(pcd):
        raise TypeError("pcd should be an ndarray. Got {}.".format(type(pcd)))

    if not _is_numpy_pcd(pcd):
        raise ValueError(
            "pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim)
        )

    return pcd[samples]


def shuffle(pcd, perm):
    if not _is_numpy(pcd):
        raise TypeError("pcd should be an ndarray. Got {}.".format(type(pcd)))

    if not _is_numpy_pcd(pcd):
        raise ValueError(
            "pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim)
        )

    return pcd[perm]
