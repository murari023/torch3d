import torch
import torch.nn as nn
from torch3d.nn import XConv, RandomPointSample


__all__ = ["PointCNN"]


class PointCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.down1 = RandomPointSample(1024)
        self.down2 = RandomPointSample(384)
        self.down3 = RandomPointSample(128)
        self.down4 = RandomPointSample(128)
        self.conv1 = XConv(self.in_channels, 48, 8, dilation=1, bias=False)
        self.conv2 = XConv(48, 96, 12, dilation=2, bias=False)
        self.conv3 = XConv(96, 192, 16, dilation=2, bias=False)
        self.conv4 = XConv(192, 384, 16, dilation=3, bias=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(384, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2)
        )
        self.fc = nn.Conv1d(128, self.num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, p, x=None):
        q, _ = self.down1(p)
        p, x = self.conv1(p, q, x)
        q, _ = self.down2(p)
        p, x = self.conv2(p, q, x)
        q, _ = self.down3(p)
        p, x = self.conv3(p, q, x)
        q, _ = self.down3(p)
        p, x = self.conv4(p, q, x)
        x = self.mlp(x)
        x = self.fc(x)
        x = self.avgpool(x).squeeze(2)
        return x
