#include "cpu.h"


template <typename T>
void farthest_point_sample_impl(
    const T* points,
    int batch_size,
    int num_points,
    int num_samples,
    int channels,
    T* sqdists,
    int64_t* indices)
{

}


at::Tensor farthest_point_sample_cpu(const at::Tensor& points, int num_samples)
{
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int channels = points.size(2);
    at::Tensor indices = at::zeros({batch_size, num_samples}, points.options().dtype(at::kLong));
    at::Tensor sqdists = at::zeros({batch_size, num_points}, points.options()).fill_(1e10);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "farthest_point_sample_cpu", [&] {
        farthest_point_sample_impl<scalar_t>(
            points.contiguous().data<scalar_t>(),
            batch_size,
            num_points,
            num_samples,
            channels,
            sqdists.data<scalar_t>(),
            indices.data<int64_t>());
    });

    return indices;
}
