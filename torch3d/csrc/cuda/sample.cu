#include "cuda.h"


constexpr int num_threads = 256;


template <typename T>
__device__ void __update(T* __restrict__ sdist, int64_t* __restrict__ sdist_i, int64_t i, int64_t j)
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
    const T* __restrict__ points,
    int batch_size,
    int num_points,
    int num_samples,
    int channels,
    T* __restrict__ sqdists,
    int64_t* __restrict__ indices)
{
    __shared__ T sdist[num_threads];
    __shared__ int64_t sdist_i[num_threads];

    int b = blockIdx.x;

    points += b * num_points * channels;
    sqdists += b * num_points;
    indices += b * num_samples;

    int64_t prev = 0;
    int tid = threadIdx.x;
    if (tid == 0)
        indices[prev] = 0;

    __syncthreads();
    for (int64_t i = 1; i < num_samples; ++i) {
        int argmax = 0;
        T max_dist = 0;
        for (int64_t k = tid; k < num_points; k += num_threads) {
            T dist = 0;
            for (int64_t c = 0; c < channels; ++c) {
                T d = points[k * channels + c] - points[prev * channels + c];
                dist += d * d;
            }
            dist = min(dist, sqdists[k]);
            sqdists[k] = dist;
            argmax = dist > max_dist ? k : argmax;
            max_dist = dist > max_dist ? dist : max_dist;
        }
        sdist[tid] = max_dist;
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
            indices[i] = prev;
    }
}


at::Tensor farthest_point_sample_cuda(at::Tensor points, int num_samples)
{
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int channels = points.size(2);
    at::Tensor indices = at::zeros({batch_size, num_samples}, points.options().dtype(at::kLong));
    at::Tensor sqdists = at::zeros({batch_size, num_points}, points.options()).fill_(1e10);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "farthest_point_sample_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        farthest_point_sample_kernel<scalar_t><<<grid, block, 0, stream>>>(
            points.data<scalar_t>(),
            batch_size,
            num_points,
            num_samples,
            channels,
            sqdists.data<scalar_t>(),
            indices.data<int64_t>());
    });

    return indices;
}
