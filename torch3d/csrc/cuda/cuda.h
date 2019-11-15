// -*- mode: c++ -*-
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>


at::Tensor farthest_point_sample_cuda(const at::Tensor& input, int m);

at::Tensor ball_point_cuda(
    const at::Tensor& input,
    const at::Tensor& query,
    float radius,
    int k);

at::Tensor point_interpolate_cuda(
    const at::Tensor& input,
    const at::Tensor& index,
    const at::Tensor& weight);
at::Tensor point_interpolate_grad_cuda(
    const at::Tensor& grad,
    const at::Tensor& index,
    const at::Tensor& weight,
    int n);

std::vector<at::Tensor> chamfer_distance_cuda(
    const at::Tensor& input,
    const at::Tensor& target);
std::vector<at::Tensor> chamfer_distance_grad_cuda(
    const at::Tensor& grad1,
    const at::Tensor& grad2,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& index1,
    const at::Tensor& index2);
