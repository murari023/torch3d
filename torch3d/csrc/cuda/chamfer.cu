#include "cuda.h"


constexpr int num_threads = 256;


template <typename T>
__global__ void chamfer_distance_kernel(
    const T* input,
    const T* target,
    int batch_size,
    int n,
    int m,
    int channels,
    int64_t* __restrict__ index,
    T* __restrict__ sqdist)
{
    __shared__ T smem[num_threads * 3];

    int b = blockIdx.x;

    input += b * n * 3;
    target += b * m * 3;
    index += b * n;
    sqdist += b * n;

    int tid = threadIdx.x;
    int base = 0;
    for (int i = tid; i < m; i += num_threads) {
        smem[tid * 3 + 0] = target[i * 3 + 0];
        smem[tid * 3 + 1] = target[i * 3 + 1];
        smem[tid * 3 + 2] = target[i * 3 + 2];
        __syncthreads();

        int kmax = min(m, base + num_threads) - base;
        for (int j = 0; j < n; ++j) {
            T x1 = input[j * 3 + 0];
            T y1 = input[j * 3 + 1];
            T z1 = input[j * 3 + 2];

            T minval = sqdist[j];
            int64_t argmin = index[j];
            for (int k = 0; k < kmax; ++k) {
                T x2 = smem[k * 3 + 0] - x1;
                T y2 = smem[k * 3 + 1] - y1;
                T z2 = smem[k * 3 + 2] - z1;
                T dist = x2 * x2 + y2 * y2 + z2 * z2;
                if (dist < minval) {
                    minval = dist;
                    argmin = base + k;
                }
            }

            if (minval < sqdist[j]) {
                sqdist[j] = minval;
                index[j] = argmin;
            }
        }
        base += num_threads;
        __syncthreads();
    }
}


std::vector<at::Tensor> chamfer_distance_cuda(
    const at::Tensor& input,
    const at::Tensor& target)
{
    int batch_size = input.size(0);
    int n = input.size(1);
    int m = target.size(1);
    int channels = input.size(2);
    at::Tensor index1 = at::zeros({batch_size, n}, input.options().dtype(at::kLong));
    at::Tensor index2 = at::zeros({batch_size, m}, input.options().dtype(at::kLong));
    at::Tensor sqdist1 = at::zeros({batch_size, n}, input.options()).fill_(1e10);
    at::Tensor sqdist2 = at::zeros({batch_size, m}, input.options()).fill_(1e10);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "chamfer_distance_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        chamfer_distance_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.contiguous().data<scalar_t>(),
            target.contiguous().data<scalar_t>(),
            batch_size,
            n,
            m,
            channels,
            index1.data<int64_t>(),
            sqdist1.data<scalar_t>());
        chamfer_distance_kernel<scalar_t><<<grid, block, 0, stream>>>(
            target.contiguous().data<scalar_t>(),
            input.contiguous().data<scalar_t>(),
            batch_size,
            m,
            n,
            channels,
            index2.data<int64_t>(),
            sqdist2.data<scalar_t>());
    });

    return {index1, index2, sqdist1, sqdist2};
}


template <typename T>
__global__ void chamfer_distance_grad_kernel(
    const T* __restrict__ grad,
    const T* input,
    const T* target,
    const int64_t* __restrict__ index,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ output1,
    T* __restrict__ output2)
{
    int b = blockIdx.x;

    grad += b * n;
    input += b * n * 3;
    target += b * m * 3;
    index += b * n;
    output1 += b * n * 3;
    output2 += b * m * 3;

    for (int i = 0; i < n; i += num_threads) {
        T x1 = input[i + 0];
        T y1 = input[i + 1];
        T z1 = input[i + 2];

        int64_t j = index[i];
        T x2 = target[j + 0];
        T y2 = target[j + 1];
        T z2 = target[j + 2];

        T g = grad[i] * 2;
        atomicAdd(output1 + i * 3 + 0, g * (x1 - x2));
        atomicAdd(output1 + i * 3 + 1, g * (y1 - y2));
        atomicAdd(output1 + i * 3 + 2, g * (z1 - z2));
        atomicAdd(output2 + j * 3 + 0, g * (x2 - x1));
        atomicAdd(output2 + j * 3 + 1, g * (y2 - y1));
        atomicAdd(output2 + j * 3 + 2, g * (z2 - z1));
    }
}


std::vector<at::Tensor> chamfer_distance_grad_cuda(
    const at::Tensor& grad1,
    const at::Tensor& grad2,
    const at::Tensor& input,
    const at::Tensor& target,
    const at::Tensor& index1,
    const at::Tensor& index2)
{
    int batch_size = input.size(0);
    int n = input.size(1);
    int m = target.size(1);
    int channels = input.size(2);
    at::Tensor output1 = at::zeros_like(input);
    at::Tensor output2 = at::zeros_like(target);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "chamfer_distance_grad_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        chamfer_distance_grad_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad1.contiguous().data<scalar_t>(),
            input.contiguous().data<scalar_t>(),
            target.contiguous().data<scalar_t>(),
            index1.data<int64_t>(),
            batch_size,
            n,
            m,
            channels,
            output1.data<scalar_t>(),
            output2.data<scalar_t>());
        chamfer_distance_grad_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad2.contiguous().data<scalar_t>(),
            target.contiguous().data<scalar_t>(),
            input.contiguous().data<scalar_t>(),
            index2.data<int64_t>(),
            batch_size,
            m,
            n,
            channels,
            output2.data<scalar_t>(),
            output1.data<scalar_t>());
    });

    return {output1, output2};
}
