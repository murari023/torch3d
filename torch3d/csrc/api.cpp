#include "api.h"
#include "cuda/cuda.h"


at::Tensor farthest_point_sample(const at::Tensor& input, int m)
{
    if (input.type().is_cuda()) {
        return farthest_point_sample_cuda(input, m);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor ball_point(
    const at::Tensor& input,
    const at::Tensor& query,
    float radius,
    int k)
{
    if (input.type().is_cuda()) {
        return ball_point_cuda(input, query, radius, k);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor point_interpolate(
    const at::Tensor& input,
    const at::Tensor& index,
    const at::Tensor& weight)
{
    if (input.type().is_cuda()) {
        return point_interpolate_cuda(input, index, weight);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor point_interpolate_grad(
    const at::Tensor& grad,
    const at::Tensor& index,
    const at::Tensor& weight,
    int n)
{
    if (grad.type().is_cuda()) {
        return point_interpolate_grad_cuda(grad, index, weight, n);
    }
    AT_ERROR("Not compiled with GPU support");
}


std::vector<at::Tensor> chamfer_distance(
    const at::Tensor& input,
    const at::Tensor& target)
{
    if (input.type().is_cuda()) {
        return chamfer_distance_cuda(input, target);
    }
    AT_ERROR("Not compiled with GPU support");
}


std::vector<at::Tensor> chamfer_distance_grad(
    const at::Tensor& grad1,
    const at::Tensor& grad2,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& index1,
    const at::Tensor& index2)
{
    if (grad1.type().is_cuda()) {
        return chamfer_distance_grad_cuda(grad1, grad2, input, target, index1, index2);
    }
    AT_ERROR("Not compiled with GPU support");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("farthest_point_sample", &farthest_point_sample);
    m.def("ball_point", &ball_point);
    m.def("point_interpolate", &point_interpolate);
    m.def("point_interpolate_grad", &point_interpolate_grad);
    m.def("chamfer_distance", &chamfer_distance);
    m.def("chamfer_distance_grad", &chamfer_distance_grad);
}
