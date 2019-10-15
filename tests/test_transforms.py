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

    def test_shuffle(self):
        channels = 3
        num_points = 1024
        transform = transforms.Shuffle()
        x = np.random.rand(num_points, channels)
        perm = transform.get_params(x)
        transform.params = perm
        y = transform(x)
        assert np.all(np.equal(x[perm], y))
