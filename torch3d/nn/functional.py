import torch


def cdist(x, y):
    xx = x.pow(2).sum(dim=2, keepdim=True)
    yy = y.pow(2).sum(dim=2, keepdim=True).transpose(2, 1)
    sqdist = torch.baddbmm(yy, x, y.transpose(2, 1), alpha=-2).add_(xx)
    return sqdist


def gather_nd(x, indices):
    x = [x[b, i, :] for b, i in enumerate(torch.unbind(indices, dim=0))]
    x = torch.stack(x, dim=0)
    return x


def knn(q, p, k):
    sqdist = cdist(q, p)
    return torch.topk(sqdist, k, dim=2, largest=False)


def chamfer_loss(x, y):
    sqdist = cdist(x, y)
    return torch.mean(sqdist.min(1)[0]) + torch.mean(sqdist.min(2)[0])


def random_point_sample(p, x, num_samples):
    num_points = p.shape[1]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")

    indices = torch.randperm(num_points)[:num_samples]
    p = p[:, indices]
    if x is not None:
        x = x[:, :, indices]
    return p, x


def farthest_point_sample(p, x, num_samples):
    pass
