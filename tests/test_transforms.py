import torch
import numpy as np
import torch3d.transforms as transforms


class TestTransforms:
    def test_to_tensor(self):
        channels = 3
        num_points = 1024
        transform = transforms.ToTensor()
        points = np.random.rand(num_points, channels)
        points = transform(points)
        assert type(points) == torch.Tensor
