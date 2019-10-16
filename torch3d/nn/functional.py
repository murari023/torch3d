import torch
import torch3d.ops as ops


def chamfer_loss(x, y):
    sqdists = ops.cdist(x, y)
    return torch.mean(sqdists.min(1)[0]) + torch.mean(sqdists.min(2)[0])
