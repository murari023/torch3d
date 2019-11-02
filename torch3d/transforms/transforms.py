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
        raise ValueError("pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim))

    pcd = torch.tensor(pcd.transpose((1, 0)))
    return pcd


def jitter(pcd):
    if not _is_numpy(pcd):
        raise TypeError("pcd should be an ndarray. Got {}.".format(type(pcd)))

    if not _is_numpy_pcd(pcd):
        raise ValueError("pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim))

    noise = np.random.rand(*pcd.shape)
    return pcd + noise


def random_sample(pcd, num_samples):
    if not _is_numpy(pcd):
        raise TypeError("pcd should be an ndarray. Got {}.".format(type(pcd)))

    if not _is_numpy_pcd(pcd):
        raise ValueError("pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim))

    n = pcd.shape[0]
    if n >= num_samples:
        samples = np.random.choice(n, num_samples, replace=False)
    else:
        m = num_samples - n
        samples = np.random.choice(n, m, replace=True)
        samples = list(range(n)) + list(samples)
    return samples


def shuffle(pcd):
    if not _is_numpy(pcd):
        raise TypeError("pcd should be an ndarray. Got {}.".format(type(pcd)))

    if not _is_numpy_pcd(pcd):
        raise ValueError("pcd should be 2 dimensional. Got {} dimensions.".format(pcd.ndim))

    n = pcd.shape[0]
    return np.random.permutation(n)
