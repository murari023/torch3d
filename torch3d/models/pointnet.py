import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['PointNet']


class PointNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mlp1 = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(256, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.maxpool(x).squeeze(2)
        x = self.mlp3(x)
        x = self.fc(x)
        return x
