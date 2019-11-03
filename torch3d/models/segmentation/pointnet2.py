import torch
import torch.nn as nn
from torch3d.nn import SetAbstraction, FarthestPointSample


__all__ = ["PointNetSSG"]


class PointNetSSG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointNetSSG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.down1 = FarthestPointSample(1024)
        self.down2 = FarthestPointSample(256)
        self.down3 = FarthestPointSample(64)
        self.down4 = FarthestPointSample(16)
        self.sa1 = SetAbstraction([self.in_channels + 3, 32, 32, 64], 0.1, 32)
        self.sa2 = SetAbstraction([64 + 3, 64, 64, 128], 0.2, 32)
        self.sa3 = SetAbstraction([128 + 3, 128, 128, 256], 0.4, 32)
        self.sa4 = SetAbstraction([256 + 3, 256, 256, 512], 0.8, 32)

    def forward(self, p, x=None):
        q, _ = self.down1(p)
        p, x = self.sa1(p, q, x)
        q, _ = self.down2(p)
        p, x = self.sa2(p, q, x)
        q, _ = self.down3(p)
        p, x = self.sa3(p, q, x)
        q, _ = self.down4(p)
        p, x = self.sa4(p, q, x)
