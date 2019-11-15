#include "cuda.h"


constexpr int num_threads = 256;


template <typename T>
__global__ void ball_point_kernel(
    const T* input,
    const T* query,
    int batch_size,
    int n,
    int m,
    int channels,
    float radius,
    int k,
    int64_t* __restrict__ index)
{
    int b = blockIdx.x;

    input += b * n * channels;
    query += b * m * channels;
    index += b * m * k;

    int tid = threadIdx.x;
    float r2 = radius * radius;

    for (int i = tid; tid < m; tid += num_threads) {
        int count = 0;
        for (int j = 0; j < n; ++j) {
            T dist = 0;
            for (int c = 0; c < channels; ++c) {
                T d = query[i * channels + c] - input[j * channels + c];
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
    const at::Tensor& input,
    const at::Tensor& query,
    float radius,
    int k)
{
    int batch_size = input.size(0);
    int n = input.size(1);
    int m = query.size(1);
    int channels = input.size(2);
    at::Tensor index = at::zeros({batch_size, m, k}, input.options().dtype(at::kLong));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "ball_point_cuda", [&] {
        dim3 block(num_threads);
        dim3 grid(batch_size);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        ball_point_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.contiguous().data<scalar_t>(),
            query.contiguous().data<scalar_t>(),
            batch_size,
            n,
            m,
            channels,
            radius,
            k,
            index.data<int64_t>());
    });

    return index;
}
