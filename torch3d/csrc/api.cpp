#include "api.h"
#include "cpu/cpu.h"
#ifdef WITH_CUDA
#include "cuda/cuda.h"
#endif


at::Tensor farthest_point_sample(const at::Tensor& points, int num_samples)
{
    if (points.type().is_cuda()) {
#ifdef WITH_CUDA
        return farthest_point_sample_cuda(points, num_samples);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return farthest_point_sample_cpu(points, num_samples);
}


at::Tensor ball_point(const at::Tensor& points, const at::Tensor& queries, float radius, int k)
{
    if (points.type().is_cuda()) {
#ifdef WITH_CUDA
        return ball_point_cuda(points, queries, radius, k);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return ball_point_cpu(points, queries, radius, k);
}


at::Tensor gather_points(const at::Tensor& points, const at::Tensor& indices)
{
    if (points.type().is_cuda()) {
#ifdef WITH_CUDA
        return gather_points_cuda(points, indices);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return gather_points_cpu(points, indices);
}


at::Tensor gather_points_backward(const at::Tensor& grad, const at::Tensor& indices, int n)
{
    if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
        return gather_points_backward_cuda(grad, indices, n);
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }
    return gather_points_backward_cpu(grad, indices, n);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sample", &farthest_point_sample);
    m.def("ball_point", &ball_point);
    m.def("gather_points", &gather_points);
    m.def("gather_points_backward", &gather_points_backward);
}
