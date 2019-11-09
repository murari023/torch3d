import torch
import torch.nn as nn
from torch3d.nn import SetAbstraction, FeaturePropagation, FarthestPointSample


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
        self.sa1 = SetAbstraction(
            self.in_channels + 3, [32, 32, 64], 0.1, 32, bias=False
        )
        self.sa2 = SetAbstraction(64 + 3, [64, 64, 128], 0.2, 32, bias=False)
        self.sa3 = SetAbstraction(128 + 3, [128, 128, 256], 0.4, 32, bias=False)
        self.sa4 = SetAbstraction(256 + 3, [256, 256, 512], 0.8, 32, bias=False)
        self.fp1 = FeaturePropagation(768, [256, 256], bias=False)
        self.fp2 = FeaturePropagation(384, [256, 256], bias=False)
        self.fp3 = FeaturePropagation(320, [256, 128], bias=False)
        self.fp4 = FeaturePropagation(128, [128, 128], bias=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.fc = nn.Conv1d(128, self.num_classes, 1)

    def forward(self, p, x=None):
        q1, _ = self.down1(p)
        p1, x1 = self.sa1(p, q1, x)
        q2, _ = self.down2(p1)
        p2, x2 = self.sa2(p1, q2, x1)
        q3, _ = self.down3(p2)
        p3, x3 = self.sa3(p2, q3, x2)
        q4, _ = self.down4(p3)
        p4, x4 = self.sa4(p3, q4, x3)
        p3, x3 = self.fp1(p4, q3, x4, x3)
        p2, x2 = self.fp2(p3, q2, x3, x2)
        p1, x1 = self.fp3(p2, q1, x2, x1)
        p, x = self.fp4(p1, p, x1)
        x = self.mlp(x)
        x = self.fc(x)
        return x
