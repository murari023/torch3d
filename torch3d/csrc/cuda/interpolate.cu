#include "cuda.h"


constexpr int num_threads = 256;


template <typename T>
__global__ void interpolate_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ index,
    const T* __restrict__ weight,
    int batch_size,
    int n,
    int m,
    int channels,
    int kernel_size,
    T* __restrict__ output)
{
    int b = blockIdx.x;

    input += b * channels * n;
    index += b * m * kernel_size;
    weight += b * m * kernel_size;
    output += b * channels * m;

    int tid = threadIdx.x;
    for (int i = tid; i < channels * m; i += num_threads) {
        int j = i % m;
        int c = i / m;

        for (int k = 0; k < kernel_size; ++k) {
            T w = weight[j * kernel_size + k];
            int64_t idx = index[j * kernel_size + k];
            output[i] += input[c * n + idx] * w;
        }
    }
}


at::Tensor interpolate_cuda(const at::Tensor& input, const at::Tensor& index, const at::Tensor& weight)
{
    int batch_size = input.size(0);
    int channels = input.size(1);
    int n = input.size(2);
    int m = index.size(1);
    int kernel_size = index.size(2);
    at::Tensor output = at::zeros({batch_size, channels, m}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "interpolate_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        interpolate_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.contiguous().data<scalar_t>(),
            index.contiguous().data<int64_t>(),
            weight.contiguous().data<scalar_t>(),
            batch_size,
            n,
            m,
            channels,
            kernel_size,
            output.data<scalar_t>());
    });

    return output;
}


template <typename T>
__global__ void interpolate_grad_kernel(
    const T* __restrict__ grad,
    const int64_t* __restrict__ index,
    const T* __restrict__ weight,
    int batch_size,
    int n,
    int m,
    int channels,
    int kernel_size,
    T* __restrict__ output)
{
    int b = blockIdx.x;

    grad += b * channels * m;
    index += b * m * kernel_size;
    weight += b * m * kernel_size;
    output += b * channels * n;

    int tid = threadIdx.x;
    for (int i = tid; i < channels * m; i += num_threads) {
        int j = i % m;
        int c = i / m;

        for (int k = 0; k < kernel_size; ++k) {
            T w = weight[j * kernel_size + k];
            int64_t idx = index[j * kernel_size + k];
            atomicAdd(output + c * n + idx, grad[i] * w);
        }
    }
}


at::Tensor interpolate_grad_cuda(const at::Tensor& grad, const at::Tensor& index, const at::Tensor& weight, int n)
{
    int batch_size = grad.size(0);
    int channels = grad.size(1);
    int m = grad.size(2);
    int kernel_size = index.size(2);
    at::Tensor output = at::zeros({batch_size, channels, n}, grad.options());

    AT_DISPATCH_FLOATING_TYPES(grad.type(), "interpolate_grad_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        interpolate_grad_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad.contiguous().data<scalar_t>(),
            index.contiguous().data<int64_t>(),
            weight.contiguous().data<scalar_t>(),
            batch_size,
            n,
            m,
            channels,
            kernel_size,
            output.data<scalar_t>());
    });

    return output;
}
