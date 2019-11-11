#include "api.h"
#include "cuda/cuda.h"


at::Tensor farthest_point_sample(const at::Tensor& points, int num_samples)
{
    if (points.type().is_cuda()) {
        return farthest_point_sample_cuda(points, num_samples);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor ball_point(const at::Tensor& points, const at::Tensor& queries, float radius, int k)
{
    if (points.type().is_cuda()) {
        return ball_point_cuda(points, queries, radius, k);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor gather_points(const at::Tensor& points, const at::Tensor& index)
{
    if (points.type().is_cuda()) {
        return gather_points_cuda(points, index);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor gather_points_grad(const at::Tensor& grad, const at::Tensor& index, int n)
{
    if (grad.type().is_cuda()) {
        return gather_points_grad_cuda(grad, index, n);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor interpolate(const at::Tensor& input, const at::Tensor& index, const at::Tensor& weight)
{
    if (input.type().is_cuda()) {
        return interpolate_cuda(input, index, weight);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor interpolate_grad(const at::Tensor& grad, const at::Tensor& index, const at::Tensor& weight, int n)
{
    if (grad.type().is_cuda()) {
        return interpolate_grad_cuda(grad, index, weight, n);
    }
    AT_ERROR("Not compiled with GPU support");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sample", &farthest_point_sample);
    m.def("ball_point", &ball_point);
    m.def("gather_points", &gather_points);
    m.def("gather_points_grad", &gather_points_grad);
    m.def("interpolate", &interpolate);
    m.def("interpolate_grad", &interpolate_grad);
}
