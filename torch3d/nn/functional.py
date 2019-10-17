import torch


def cdist(x, y):
    xx = x.pow(2).sum(dim=2, keepdim=True)
    yy = y.pow(2).sum(dim=2, keepdim=True).transpose(2, 1)
    sqdist = torch.baddbmm(yy, x, y.transpose(2, 1), alpha=-2).add_(xx)
    return sqdist


def knn(q, p, k):
    sqdist = cdist(q, p)
    return torch.topk(sqdist, k, dim=2, largest=False)


def chamfer_loss(x, y):
    sqdist = cdist(x, y)
    return torch.mean(sqdist.min(1)[0]) + torch.mean(sqdist.min(2)[0])


def random_sample(p, x, num_samples):
    num_points = p.shape[1]
    indices = torch.randperm(num_points)[:num_samples]
    p = p[:, indices]
    if x is not None:
        x = x[:, :, indices]
    return p, x
