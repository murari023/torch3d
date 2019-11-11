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


def gather_points(p, index):
    return _GatherPoints.apply(p, index)


def gather_groups(p, index):
    batch_size = index.shape[0]
    m = index.shape[1]
    k = index.shape[2]
    index = index.view(batch_size, -1)
    output = _GatherPoints.apply(p, index)
    output = output.view(batch_size, m, k, -1)
    return output


def interpolate(input, index, weight):
    return _Interpolate.apply(input, index, weight)


class _GatherPoints(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, index):
        n = points.shape[1]
        ctx.for_backwards = (index, n)
        _C = _lazy_import()
        return _C.gather_points(points, index)

    @staticmethod
    def backward(ctx, grad):
        index, n = ctx.for_backwards
        _C = _lazy_import()
        output = _C.gather_points_grad(grad, index, n)
        return output, None


class _Interpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, index, weight):
        n = input.shape[2]
        ctx.for_backwards = (index, weight, n)
        _C = _lazy_import()
        return _C.interpolate(input, index, weight)

    @staticmethod
    def backward(ctx, grad):
        index, weight, n = ctx.for_backwards
        _C = _lazy_import()
        output = _C.interpolate_grad(grad, index, weight, n)
        return output, None, None
