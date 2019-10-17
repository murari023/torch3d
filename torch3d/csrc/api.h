// -*- mode: c++ -*-
#include <torch/extension.h>


at::Tensor farthest_point_sample(at::Tensor points, int num_samples);
