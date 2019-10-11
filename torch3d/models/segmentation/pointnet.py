import torch
import torch.nn as nn


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
            nn.ReLU(),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp3 = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.fc = nn.Conv1d(128, num_classes, 1)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        num_points = x.shape[2]
        f = self.mlp1(x)
        x = self.mlp2(f)
        x = self.maxpool(x).repeat(1, 1, num_points)
        x = torch.cat([f, x], dim=1)
        x = self.mlp3(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
