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


def batched_index_select(input, dim, index):
    views = [input.shape[0]]
    views += [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


class PointInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, index, weight):
        n = input.shape[2]
        ctx.for_backwards = (index, weight, n)
        _C = _lazy_import()
        return _C.point_interpolate(input, index, weight)

    @staticmethod
    def backward(ctx, grad):
        index, weight, n = ctx.for_backwards
        _C = _lazy_import()
        output = _C.point_interpolate_grad(grad, index, weight, n)
        return output, None, None


point_interpolate = PointInterpolate.apply
