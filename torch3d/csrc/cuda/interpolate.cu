#include "cuda.h"


constexpr int num_threads = 256;


template <typename T>
__global__ void interpolate_kernel(
    const T* __restrict__ input,
    const int64_t* __restrict__ indices,
    const T* __restrict__ weight,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ output)
{
    int b = blockIdx.x;

    input += b * channels * n;
    indices += b * m * 3;
    weight += b * m * 3;
    output += b * channels * m;

    int tid = threadIdx.x;
    for (int i = tid; i < channels * m; i += num_threads) {
        int k = i % m;
        int c = i / m;

        T w1 = weight[k * 3 + 0];
        T w2 = weight[k * 3 + 1];
        T w3 = weight[k * 3 + 2];
        int64_t i1 = indices[k * 3 + 0];
        int64_t i2 = indices[k * 3 + 1];
        int64_t i3 = indices[k * 3 + 2];

        output[i] = input[c * n + i1] * w1 + input[c * n + i2] * w2 + input[c * n + i3] * w3;
    }
}


at::Tensor interpolate_cuda(const at::Tensor& input, const at::Tensor& indices, const at::Tensor& weight)
{
    int batch_size = input.size(0);
    int channels = input.size(1);
    int n = input.size(2);
    int m = indices.size(1);
    at::Tensor output = at::zeros({batch_size, channels, m}, input.options());

    AT_DISPATCH_FLOATING_TYPES(input.type(), "interpolate_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        interpolate_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.contiguous().data<scalar_t>(),
            indices.contiguous().data<int64_t>(),
            weight.contiguous().data<scalar_t>(),
            batch_size,
            n,
            m,
            channels,
            output.data<scalar_t>());
    });

    return output;
}


template <typename T>
__global__ void interpolate_grad_kernel(
    const T* __restrict__ grad,
    const int64_t* __restrict__ indices,
    const T* __restrict__ weight,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ output)
{
    int b = blockIdx.x;

    grad += b * channels * m;
    indices += b * m * 3;
    weight += b * m * 3;
    output += b * channels * n;

    int tid = threadIdx.x;
    for (int i = tid; i < channels * m; i += num_threads) {
        int k = i % m;
        int c = i / m;

        T w1 = weight[k * 3 + 0];
        T w2 = weight[k * 3 + 1];
        T w3 = weight[k * 3 + 2];
        int64_t i1 = indices[k * 3 + 0];
        int64_t i2 = indices[k * 3 + 1];
        int64_t i3 = indices[k * 3 + 2];

        atomicAdd(output + c * n + i1, grad[i] * w1);
        atomicAdd(output + c * n + i2, grad[i] * w2);
        atomicAdd(output + c * n + i3, grad[i] * w3);
    }
}


at::Tensor interpolate_grad_cuda(const at::Tensor& grad, const at::Tensor& indices, const at::Tensor& weight, int n)
{
    int batch_size = grad.size(0);
    int channels = grad.size(1);
    int m = grad.size(2);
    at::Tensor output = at::zeros({batch_size, channels, n}, grad.options());

    AT_DISPATCH_FLOATING_TYPES(grad.type(), "interpolate_grad_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        interpolate_grad_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad.contiguous().data<scalar_t>(),
            indices.contiguous().data<int64_t>(),
            weight.contiguous().data<scalar_t>(),
            batch_size,
            n,
            m,
            channels,
            output.data<scalar_t>());
    });

    return output;
}
