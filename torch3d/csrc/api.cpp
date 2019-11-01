#include "api.h"
#include "cuda/sample.h"


at::Tensor farthest_point_sample(at::Tensor points, int num_samples)
{
    if (points.type().is_cuda()) {
        return farthest_point_sample_cuda(points, num_samples);
    }
    AT_ERROR("Not compiled with GPU support");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("farthest_point_sample", &farthest_point_sample);
}
