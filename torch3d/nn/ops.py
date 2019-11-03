import torch
from torch3d.extension import _lazy_import


class Gather1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, indices):
        assert x.is_contiguous()
        assert indices.is_contiguous()
        _C = _lazy_import()
        n = x.shape[1]
        output = _C.gather1d(x, indices)
        ctx.for_backwards = (indices, n)

    @staticmethod
    def backward(ctx, grad):
        _C = _lazy_import()
        indices, n = ctx.for_backwards
        output = _C.gather1d_backward(grad, indices)
        return output, None
