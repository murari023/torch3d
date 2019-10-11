import torch
import torch.nn as nn
from torch3d.nn import XConv, Downsample


__all__ = ['PointCNN']


class PointCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointCNN, self).__init__()
        self.down1 = Downsample(1024, mode='random')
        self.conv1 = XConv(in_channels, 48, 8, dilation=1)
        self.down2 = Downsample(384, mode='random')
        self.conv2 = XConv(48, 96, 12, dilation=2)
        self.down3 = Downsample(128, mode='random')
        self.conv3 = XConv(96, 192, 16, dilation=2)
        self.down4 = Downsample(128, mode='random')
        self.conv4 = XConv(192, 384, 16, dilation=3)
        self.mlp = nn.Sequential(
            nn.Conv1d(384, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc = nn.Conv1d(128, num_classes, 1)
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


if __name__ == '__main__':
    model = PointCNN(0, 40)
    x = torch.rand([1, 3, 1024])
    y = model(x)
    print(y.shape)
