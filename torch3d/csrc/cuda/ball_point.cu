#include "cuda.h"

constexpr int num_threads = 256;

template <typename T>
__global__ void ball_point_kernel(
    const T* points,
    const T* queries,
    int batch_size,
    int num_points,
    int num_queries,
    int channels,
    float radius,
    int k,
    int64_t* __restrict__ index) {
    int b = blockIdx.x;

    points += b * num_points * channels;
    queries += b * num_queries * channels;
    index += b * num_queries * k;

    int tid = threadIdx.x;
    float r2 = radius * radius;

    for (int i = tid; tid < num_queries; tid += num_threads) {
        int count = 0;
        for (int j = 0; j < num_points; ++j) {
            T dist = 0;
            for (int c = 0; c < channels; ++c) {
                T d = queries[i * channels + c] - points[j * channels + c];
                dist += d * d;
            }

            if (dist < r2) {
                if (count == 0) {
                    for (int l = 0; l < k; ++l)
                        index[i * k + l] = j;
                }
                index[i * k + count] = j;
                ++count;
            }
            if (count >= k)
                break;
        }
    }
}

at::Tensor ball_point_cuda(
    const at::Tensor& points,
    const at::Tensor& queries,
    float radius,
    int k) {
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int num_queries = queries.size(1);
    int channels = points.size(2);
    at::Tensor index =
        at::zeros({batch_size, num_queries, k}, points.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES(points.type(), "ball_point_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ball_point_kernel<scalar_t><<<grid, block, 0, stream>>>(
            points.contiguous().data<scalar_t>(),
            queries.contiguous().data<scalar_t>(),
            batch_size,
            num_points,
            num_queries,
            channels,
            radius,
            k,
            index.data<int64_t>());
    });

    return index;
}
