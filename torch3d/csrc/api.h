// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor farthest_point_sample(const at::Tensor& input, int m);

at::Tensor ball_point(
    const at::Tensor& input,
    const at::Tensor& query,
    float radius,
    int k);

at::Tensor point_interpolate(
    const at::Tensor& input,
    const at::Tensor& index,
    const at::Tensor& weight);
at::Tensor point_interpolate_grad(
    const at::Tensor& grad,
    const at::Tensor& index,
    const at::Tensor& weight,
    int n);
