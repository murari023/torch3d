#include "cpu.h"


at::Tensor ball_point_cpu(const at::Tensor& points, const at::Tensor& queries, float radius, int k)
{
    int batch_size = points.size(0);
    int num_points = points.size(1);
    int num_queries = queries.size(1);
    int channels = points.size(2);
    at::Tensor indices = at::zeros({batch_size, num_queries, k}, points.options().dtype(at::kLong));

    return indices;
}
