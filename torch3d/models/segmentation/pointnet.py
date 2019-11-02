import torch
import torch.nn as nn


__all__ = ["PointNet"]


class PointNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.fc = nn.Conv1d(128, num_classes, 1)

    def forward(self, x):
        num_points = x.shape[2]
        f = self.mlp1(x)
        x = self.mlp2(f)
        x = self.maxpool(x).repeat(1, 1, num_points)
        x = torch.cat([f, x], dim=1)
        x = self.mlp3(x)
        x = self.fc(x)
        return x
