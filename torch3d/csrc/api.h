// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor farthest_point_sample(const at::Tensor& points, int num_samples);
at::Tensor ball_point(const at::Tensor& points, const at::Tensor& queries, float radius, int k);
