import torch
from torch3d.extension import _lazy_import


def cdist(x, y):
    xx = x.pow(2).sum(dim=2, keepdim=True)
    yy = y.pow(2).sum(dim=2, keepdim=True).transpose(2, 1)
    sqdist = torch.baddbmm(yy, x, y.transpose(2, 1), alpha=-2).add_(xx)
    return sqdist


def knn(input, query, k):
    sqdist = cdist(query, input)
    return torch.topk(sqdist, k, dim=-1, largest=False)


def ball_point(input, query, radius, k):
    _C = _lazy_import()
    return _C.ball_point(input, query, radius, k)


def chamfer_loss(input, target):
    sqdist = cdist(input, target)
    return torch.mean(sqdist.min(1)[0]) + torch.mean(sqdist.min(2)[0])


def random_point_sample(input, num_samples):
    num_points = input.shape[1]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")
    return torch.randperm(num_points)[:num_samples]


def farthest_point_sample(input, num_samples):
    num_points = input.shape[1]
    if num_samples > num_points:
        raise ValueError("num_samples should be less than input size.")
    _C = _lazy_import()
    return _C.farthest_point_sample(input, num_samples)


def batched_index_select(input, dim, index):
    views = [input.shape[0]]
    views += [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(input, dim, index)


def point_interpolate(input, index, weight):
    return PointInterpolate.apply(input, index, weight)


def chamfer_distance(input, target):
    return ChamferDistance.apply(input, target)


class PointInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, index, weight):
        ctx.num_points = input.shape[2]
        ctx.save_for_backward(index, weight)
        _C = _lazy_import()
        return _C.point_interpolate(input, index, weight)

    @staticmethod
    def backward(ctx, grad):
        num_points = ctx.num_points
        index, weight = ctx.saved_tensors
        _C = _lazy_import()
        output = _C.point_interpolate_grad(grad, index, weight, num_points)
        return output, None, None


class ChamferDistance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target):
        _C = _lazy_import()
        index1, index2, sqdist1, sqdist2 = _C.chamfer_distance(input, target)
        ctx.save_for_backward(input, target, index1, index2)
        return index1, index2, sqdist1, sqdist2

    def forward(ctx, grad1, grad2):
        input, target, sqdist1, sqdist2 = ctx.saved_tensors
        output1, output2 = _C.chamfer_distance_grad(
            grad1, grad2, input, target, index1, index2
        )
        return output1, output2
