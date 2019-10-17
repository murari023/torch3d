#include "sample.h"


constexpr int num_threads = 256;


template <typename T>
__global__ void farthest_point_sample_kernel(
    const T* __restrict__ points,
    int batch_size,
    int num_points,
    int channels,
    int num_samples,
    int* __restrict__ indices)
{
}


void farthest_point_sample_cuda(at::Tensor points, int num_samples)
{
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int channels = points.size(2);
    at::Tensor indices = at::zeros({batch_size, num_samples}, points.options().dtype(at::kInt));

    AT_DISPATCH_FLOATING_TYPES(p.type(), "farthest_point_sample_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        furthest_point_sample_kernel<scalar_t><<<grid, block, 0, stream>>>(
            points.data<scalar_t>(),
            batch_size,
            num_points,
            channels,
            num_samples,
            indices.data<scalar_t>());
    });

    return indices;
}
