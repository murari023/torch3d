// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor farthest_point_sample_cpu(const at::Tensor& points, int num_samples);
at::Tensor ball_point_cpu(const at::Tensor& points, const at::Tensor& queries, float radius, int k);
at::Tensor gather_points_cpu(const at::Tensor& points, const at::Tensor& indices);
at::Tensor gather_points_backward_cpu(const at::Tensor& grad, const at::Tensor& indices, int n);
