// -*- mode: c++ -*-
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


at::Tensor farthest_point_sample_cuda(const at::Tensor& points, int num_samples);
at::Tensor ball_point_cuda(const at::Tensor& points, const at::Tensor& queries, float radius, int k);
at::Tensor gather_points_cuda(const at::Tensor& points, const at::Tensor& index);
at::Tensor gather_points_grad_cuda(const at::Tensor& grad, const at::Tensor& index, int n);
at::Tensor interpolate_cuda(const at::Tensor& input, const at::Tensor& index, const at::Tensor& weight);
at::Tensor interpolate_grad_cuda(const at::Tensor& grad, const at::Tensor& index, const at::Tensor& weight, int n);
