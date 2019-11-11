#include "cuda.h"

constexpr int num_threads = 256;

template <typename T>
__global__ void gather_points_kernel(
    const T* __restrict__ points,
    const int64_t* __restrict__ index,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ output) {
    int b = blockIdx.x;

    points += b * n * channels;
    index += b * m;
    output += b * m * channels;

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        int64_t k = index[i];
        for (int c = 0; c < channels; ++c)
            output[i * channels + c] = points[k * channels + c];
    }
}

at::Tensor gather_points_cuda(const at::Tensor& points, const at::Tensor& index) {
    int batch_size = points.size(0);
    int n = points.size(1);
    int m = index.size(1);
    int channels = points.size(2);
    at::Tensor output = at::zeros({batch_size, m, channels}, points.options());

    AT_DISPATCH_FLOATING_TYPES(points.type(), "gather_points_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        gather_points_kernel<scalar_t><<<grid, block, 0, stream>>>(
            points.contiguous().data<scalar_t>(),
            index.contiguous().data<int64_t>(),
            batch_size,
            n,
            m,
            channels,
            output.data<scalar_t>());
    });

    return output;
}

template <typename T>
__global__ void gather_points_grad_kernel(
    const T* __restrict__ grad,
    const int64_t* __restrict__ index,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ output) {
    int b = blockIdx.x;

    grad += b * n * channels;
    index += b * m;
    output += b * n * channels;

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        int64_t k = index[i];
        for (int c = 0; c < channels; ++c)
            atomicAdd(output + k * channels + c, grad[i * channels + c]);
    }
}

at::Tensor gather_points_grad_cuda(
    const at::Tensor& grad,
    const at::Tensor& index,
    int n) {
    int batch_size = grad.size(0);
    int m = grad.size(1);
    int channels = grad.size(2);
    at::Tensor output = at::zeros({batch_size, n, channels}, grad.options());

    AT_DISPATCH_FLOATING_TYPES(grad.type(), "gather_points_grad_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        gather_points_grad_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad.contiguous().data<scalar_t>(),
            index.contiguous().data<int64_t>(),
            batch_size,
            n,
            m,
            channels,
            output.data<scalar_t>());
    });

    return output;
}
