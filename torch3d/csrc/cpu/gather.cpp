#include "cpu.h"


at::Tensor gather_points_cpu(const at::Tensor& points, const at::Tensor& indices)
{
    int batch_size = points.size(0);
    int n = points.size(1);
    int m = indices.size(1);
    int channels = points.size(2);
    at::Tensor output = at::zeros({batch_size, m, channels}, points.options());

    return output;
}


at::Tensor gather_points_backward_cpu(const at::Tensor& grad, const at::Tensor& indices, int n)
{
    int batch_size = grad.size(0);
    int m = grad.size(1);
    int channels = grad.size(2);
    at::Tensor output = at::zeros({batch_size, n, channels}, grad.options());

    return output;
}
