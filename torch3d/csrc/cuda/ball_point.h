// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor ball_point_cuda(at::Tensor points, at::Tensor queries, float radius, int k);
