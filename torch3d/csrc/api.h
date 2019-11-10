// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor farthest_point_sample(const at::Tensor& points, int num_samples);
at::Tensor ball_point(const at::Tensor& points, const at::Tensor& queries, float radius, int k);
at::Tensor gather_points(const at::Tensor& points, const at::Tensor& indices);
at::Tensor gather_points_grad(const at::Tensor& grad, const at::Tensor& indices, int n);
at::Tensor interpolate(const at::Tensor& input, const at::Tensor& indices, const at::Tensor& weight);
at::Tensor interpolate_grad(const at::Tensor& grad, const at::Tensor& indices, const at::Tensor& weight, int n);
