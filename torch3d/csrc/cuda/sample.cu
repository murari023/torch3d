#include "cuda.h"


constexpr int num_threads = 256;


template <typename T>
__device__ void __update(
    T* __restrict__ sdist,
    int64_t* __restrict__ sdist_i,
    int64_t i,
    int64_t j)
{
    const T v1 = sdist[i];
    const T v2 = sdist[j];
    const int64_t i1 = sdist_i[i];
    const int64_t i2 = sdist_i[j];
    sdist[i] = max(v1, v2);
    sdist_i[i] = v2 > v1 ? i2 : i1;
}


template <typename T>
__global__ void farthest_point_sample_kernel(
    const T* __restrict__ input,
    int batch_size,
    int n,
    int m,
    int channels,
    T* __restrict__ sqdist,
    int64_t* __restrict__ index)
{
    __shared__ T sdist[num_threads];
    __shared__ int64_t sdist_i[num_threads];

    int b = blockIdx.x;

    input += b * n * channels;
    sqdist += b * n;
    index += b * m;

    int64_t prev = 0;
    int tid = threadIdx.x;
    if (tid == 0)
        index[prev] = 0;

    __syncthreads();
    for (int64_t i = 1; i < m; ++i) {
        T maxval = 0;
        int argmax = 0;
        for (int64_t k = tid; k < n; k += num_threads) {
            T dist = 0;
            for (int64_t c = 0; c < channels; ++c) {
                T d = input[k * channels + c] - input[prev * channels + c];
                dist += d * d;
            }
            dist = min(dist, sqdist[k]);
            sqdist[k] = dist;
            argmax = dist > maxval ? k : argmax;
            maxval = dist > maxval ? dist : maxval;
        }
        sdist[tid] = maxval;
        sdist_i[tid] = argmax;
        __syncthreads();

        if (num_threads >= 512) {
            if (tid < 256) {
                __update(sdist, sdist_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (num_threads >= 256) {
            if (tid < 128) {
                __update(sdist, sdist_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (num_threads >= 128) {
            if (tid < 64) {
                __update(sdist, sdist_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (num_threads >= 64) {
            if (tid < 32) {
                __update(sdist, sdist_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (num_threads >= 32) {
            if (tid < 16) {
                __update(sdist, sdist_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (num_threads >= 16) {
            if (tid < 8) {
                __update(sdist, sdist_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (num_threads >= 8) {
            if (tid < 4) {
                __update(sdist, sdist_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (num_threads >= 4) {
            if (tid < 2) {
                __update(sdist, sdist_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (num_threads >= 2) {
            if (tid < 1) {
                __update(sdist, sdist_i, tid, tid + 1);
            }
            __syncthreads();
        }

        prev = sdist_i[0];
        if (tid == 0)
            index[i] = prev;
    }
}


at::Tensor farthest_point_sample_cuda(const at::Tensor& input, int m)
{
    int batch_size = input.size(0);
    int n = input.size(1);
    int channels = input.size(2);
    at::Tensor index = at::zeros({batch_size, m}, input.options().dtype(at::kLong));
    at::Tensor sqdist = at::zeros({batch_size, n}, input.options()).fill_(1e10);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "farthest_point_sample_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        farthest_point_sample_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.contiguous().data<scalar_t>(),
            batch_size,
            n,
            m,
            channels,
            sqdist.data<scalar_t>(),
            index.data<int64_t>());
    });

    return index;
}
