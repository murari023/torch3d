#include "api.h"
#include "cuda/cuda.h"


at::Tensor farthest_point_sample(at::Tensor points, int num_samples)
{
    if (points.type().is_cuda()) {
        return farthest_point_sample_cuda(points, num_samples);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor ball_point(at::Tensor points, at::Tensor queries, float radius, int k)
{
    if (points.type().is_cuda()) {
        return ball_point_cuda(points, queries, radius, k);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor gather_points(at::Tensor points, at::Tensor indices)
{
    if (points.type().is_cuda()) {
        return gather_points_cuda(points, indices);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor gather_points_backward(at::Tensor grad, at::Tensor indices, int n)
{
    if (grad.type().is_cuda()) {
        return gather_points_backward_cuda(grad, indices, n);
    }
    AT_ERROR("Not compiled with GPU support");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sample", &farthest_point_sample);
    m.def("ball_point", &ball_point);
    m.def("gather_points", &gather_points);
    m.def("gather_points_backward", &gather_points_backward);
}
