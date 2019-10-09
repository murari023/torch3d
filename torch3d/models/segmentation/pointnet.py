import torch
import torch.nn as nn
from collections import OrderedDict
from .. import pointnet


__all__ = ['PointNet']


class PointNet(pointnet.PointNet):
    def __init__(self, in_channels, num_classes):
        super(PointNet, self).__init__(in_channels, num_classes)
        self.mlp3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(1088, 512, 1, bias=False)),
            ('bn1', nn.BatchNorm1d(512)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv1d(512, 256, 1, bias=False)),
            ('bn2', nn.BatchNorm1d(256)),
            ('relu2', nn.ReLU()),
            ('conv3', nn.Conv1d(256, 128, 1, bias=False)),
            ('bn3', nn.BatchNorm1d(128)),
            ('relu3', nn.ReLU()),
        ]))
        self.fc = nn.Conv1d(128, num_classes, 1)
        self._initialize_weights()

    def forward(self, x):
        num_points = x.shape[2]
        f = self.mlp1(x)
        x = self.mlp2(f)
        x = self.maxpool(x).squeeze(2)
        x = x.view(-1, 1024, 1).repeat(1, 1, num_points)
        x = torch.cat([f, x], dim=1)
        x = self.mlp3(x)
        x = self.fc(x)
        return x
