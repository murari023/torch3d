import torch
import torch3d.models as models


class TestPointNet:
    batch_size = 8
    num_points = 2048
    in_channels = 3
    num_classes = 100
    model = models.PointNet(in_channels, num_classes)

    def test_forward(self):
        self.model.eval()
        x = torch.rand([self.batch_size, self.in_channels, self.num_points])
        y = self.model(x)
        assert y.shape == torch.Size([batch_size, num_classes])
