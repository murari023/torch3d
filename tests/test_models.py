import torch
import torch3d.models as models


class TestModels:
    def test_pointnet(self):
        in_channels = 3
        num_points = 1024
        num_classes = 50
        model = models.PointNet(in_channels, num_classes)
        model.eval()
        x = torch.rand([1, in_channels, num_points])
        y = model(x)
        assert y.shape[-1] == num_classes

    def test_pointcnn(self):
        pass

    def test_pointnet_segmentation(self):
        in_channels = 3
        num_points = 1024
        num_classes = 50
        model = models.segmentation.PointNet(in_channels, num_classes)
        model.eval()
        x = torch.rand([1, in_channels, num_points])
        y = model(x)
        assert y.shape[1] == num_classes
        assert y.shape[2] == num_points
