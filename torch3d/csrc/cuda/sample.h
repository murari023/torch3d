// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor farthest_point_sample_cuda(at::Tensor points, int num_samples);
