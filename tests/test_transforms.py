import torch
import numpy as np
import torch3d.transforms as transforms


class TestTransforms:
    def test_to_tensor(self):
        channels = 3
        num_points = 1024
        transform = transforms.ToTensor()
        sample = np.random.rand(num_points, channels)
        target = np.random.randint(0, 255, (num_points,)).astype(np.uint8)
        sample, target = transform(sample, target)
        assert type(sample) == torch.Tensor
        assert type(target) == torch.Tensor
        assert target.dtype == torch.int64
