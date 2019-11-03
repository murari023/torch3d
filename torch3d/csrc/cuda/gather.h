// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor gather1d_cuda(at::Tensor x, at::Tensor indices);
at::Tensor gather1d_backward_cuda(at::Tensor grad, at::Tensor indices, int n);
