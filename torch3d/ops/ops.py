import torch


__all__ = [
    "knn",
]


def knn(x, y, k):
    xx = x.pow(2).sum(dim=2, keepdim=True)
    yy = y.pow(2).sum(dim=2, keepdim=True)
    sqdists = torch.baddbmm(yy.transpose(2, 1), x, y.transpose(2, 1), alpha=-2).add_(xx)
    return torch.topk(sqdists, k, dim=2, largest=False)
