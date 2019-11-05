import torch
from torch3d.extension import _lazy_import


def cdist(x, y):
    xx = x.pow(2).sum(dim=2, keepdim=True)
    yy = y.pow(2).sum(dim=2, keepdim=True).transpose(2, 1)
    sqdist = torch.baddbmm(yy, x, y.transpose(2, 1), alpha=-2).add_(xx)
    return sqdist


def knn(p, q, k):
    sqdist = cdist(q, p)
    return torch.topk(sqdist, k, dim=-1, largest=False)


def ball_point(p, q, radius, k):
    _C = _lazy_import()
    return _C.ball_point(p, q, radius, k)


def chamfer_loss(x, y):
    sqdist = cdist(x, y)
    return torch.mean(sqdist.min(1)[0]) + torch.mean(sqdist.min(2)[0])


def random_point_sample(p, num_samples):
    num_points = p.shape[1]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")
    return torch.randperm(num_points)[:num_samples]


def farthest_point_sample(p, num_samples):
    num_points = p.shape[1]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")
    _C = _lazy_import()
    return _C.farthest_point_sample(p, num_samples)


def gather_points(p, indices):
    return _GatherPoints.apply(p, indices)


def gather_groups(p, indices):
    batch_size = indices.shape[0]
    m = indices.shape[1]
    k = indices.shape[2]
    indices = indices.view(batch_size, -1)
    output = _GatherPoints.apply(p, indices)
    output = output.view(batch_size, m, k, -1)
    return output


class _GatherPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, indices):
        n = points.shape[1]
        ctx.for_backwards = (indices, n)
        _C = _lazy_import()
        return _C.gather_points(points, indices)

    @staticmethod
    def backward(ctx, grad):
        indices, n = ctx.for_backwards
        _C = _lazy_import()
        output = _C.gather_points_backward(grad, indices, n)
        return output, None
