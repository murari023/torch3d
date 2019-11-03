#include "ball_point.h"
#include <ATen/cuda/CUDAContext.h>


template <typename T>
__global__ void gather1d_kernel(
    const T* __restrict__ x,
    const int64_t* __restrict__ indices,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ output)
{
    int b = blockDim.x;

    x += b * n * channels;
    indices += b * m;
    output += b * m * channels;

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        int64_t k = indices[i];
        for (int c = 0; c < channels; ++c)
            output[i * channels + c] = x[k * channels + c]
    }
}


template <typename T>
__global__ void gather1d_backward_kernel(
    const T* __restrict__ grad,
    const int64_t* __restrict__ indices,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ output)
{
    int b = blockDim.x;

    grad += b * n * channels;
    indices += b * m;
    output += b * n * channels;

    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        int64_t k = indices[i];
        for (int c = 0; c < channels; ++c)
            atomicAdd(output + k * channels + c, grad[i * channels + c])
    }
}


at::Tensor gather1d_cuda(at::Tensor x, at::Tensor indices)
{
    int batch_size = x.size(0);
    int n = x.size(1);
    int m = indices.size(1);
    int channels = x.size(2);
    at::Tensor output = at::zeros({batch_size, m, channels}, x.options());

    AT_DISPATCH_FLOATING_TYPES(x.type(), "gather1d_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        gather1d_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x.data<scalar_t>(),
            indices.data<int64_t>(),
            batch_size,
            n,
            m,
            channels,
            output.data<scalar_t>());
    });

    return output;
}


at::Tensor gather1d_backward_cuda(at::Tensor grad, at::Tensor indices, int n)
{
    int batch_size = grad.size(0);
    int m = grad.size(1);
    int channels = grad.size(2);
    at::Tensor output = at::zeros({batch_size, n, channels}, grad.options());

    AT_DISPATCH_FLOATING_TYPES(grad.type(), "gather1d_backward_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        gather1d_backward_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad.data<scalar_t>(),
            indices.data<int64_t>(),
            batch_size,
            n,
            m,
            channels,
            output.data<scalar_t>());
    });

    return output;
}
