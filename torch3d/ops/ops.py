import torch


__all__ = ["cdist", "knn", "chamfer"]


def cdist(x, y):
    xx = x.pow(2).sum(dim=2, keepdim=True)
    yy = y.pow(2).sum(dim=2, keepdim=True).transpose(2, 1)
    sqdists = torch.baddbmm(yy, x, y.transpose(2, 1), alpha=-2).add_(xx)
    return sqdists


def knn(x, y, k):
    sqdists = cdist(x, y)
    return torch.topk(sqdists, k, dim=2, largest=False)


def chamfer(x, y):
    sqdists = cdist(x, y)
    return torch.mean(sqdists.min(1)[0]) + torch.mean(sqdists.min(2)[0])
