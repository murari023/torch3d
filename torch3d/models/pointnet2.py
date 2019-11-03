import torch
import torch.nn as nn
from torch3d.nn import SetAbstraction, FarthestPointSample


__all__ = ["PointNetSSG"]


class PointNetSSG(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointNetSSG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.down1 = FarthestPointSample(512)
        self.down2 = FarthestPointSample(128)
        self.sa1 = SetAbstraction([self.in_channels + 3, 64, 64, 128], 0.2, 32)
        self.sa2 = SetAbstraction([128 + 3, 128, 128, 256], 0.4, 64)
        self.sa3 = SetAbstraction([256 + 3, 256, 512, 1024], None, 128)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(256, self.num_classes)

    def forward(self, p, x=None):
        q, _ = self.down1(p)
        p, x = self.sa1(p, q, x)
        q, _ = self.down2(p)
        p, x = self.sa2(p, q, x)
        _, x = self.sa3(p, None, x)
        x = x.squeeze(2)
        x = self.mlp(x)
        x = self.fc(x)
        return x
