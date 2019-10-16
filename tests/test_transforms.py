import torch
import numpy as np
import torch3d.transforms as transforms



class TestTransforms:
    num_points = 2048
    in_channels = 3

    def test_random_downsample(self):
        num_samples = 1024
        transform = transforms.RandomDownsample(num_samples)
        x = torch.rand([self.num_points, self.in_channels])
        y = transform(x)
        assert y.shape == torch.Size([num_samples, self.in_channels])
