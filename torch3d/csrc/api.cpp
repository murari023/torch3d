#include "api.h"
#include "cuda/cuda.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")


at::Tensor farthest_point_sample(at::Tensor points, int num_samples)
{
    CHECK_CONTIGUOUS(points);
    if (points.type().is_cuda()) {
        return farthest_point_sample_cuda(points, num_samples);
    }
    AT_ERROR("Not compiled with GPU support");
}


at::Tensor ball_point(at::Tensor points, at::Tensor queries, float radius, int k)
{
    CHECK_CONTIGUOUS(points);
    CHECK_CONTIGUOUS(queries);
    if (points.type().is_cuda()) {
        return ball_point_cuda(points, queries, radius, k);
    }
    AT_ERROR("Not compiled with GPU support");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sample", &farthest_point_sample);
    m.def("ball_point", &ball_point);
}
