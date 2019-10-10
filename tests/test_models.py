import torch
import torch3d.models as models


class TestModels:
    def test_pointnet(self):
        in_channels = 3
        num_classes = 50
        model = models.PointNet(in_channels, num_classes)
        model.eval()
        x = torch.rand([1, 3, 1024])
        x = model(x)
        assert x.shape[-1] == num_classes
