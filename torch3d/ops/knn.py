import torch

__all__ = ['knn']


def knn(p, q, k, radius=True):
    xx = p.pow(2).sum(dim=2, keepdim=True)
    yy = q.pow(2).sum(dim=2, keepdim=True)
    sqdists = torch.baddbmm(yy.transpose(2, 1), p, q.transpose(2, 1), alpha=-2).add_(xx)
    return torch.topk(sqdists, k, dim=2, largest=False)
