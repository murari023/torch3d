// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor farthest_point_sample(at::Tensor points, int num_samples);
at::Tensor ball_point(at::Tensor points, at::Tensor queries, float radius, int k);
at::Tensor gather1d(at::Tensor x, at::Tensor indices);
at::Tensor gather1d_backward(at::Tensor grad, at::Tensor indices, int n);
