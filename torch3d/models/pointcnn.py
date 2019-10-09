import math
import torch
import torch.nn as nn
from collections import OrderedDict
from torch3d.nn import XConv


__all__ = ['PointCNN']


class PointCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointCNN, self).__init__()
        self.conv1 = XConv(1024, in_channels, 48, 8, dilation=1)
        self.conv2 = XConv(384, 48, 96, 12, dilation=2)
        self.conv3 = XConv(128, 96, 192, 16, dilation=2)
        self.conv4 = XConv(128, 192, 384, 16, dilation=3)
        self.mlp = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(384, 256, 1)),
            ('bn1', nn.BatchNorm1d(256)),
            ('selu1', nn.SELU()),
            ('dropout1', nn.Dropout(0.2)),
            ('conv2', nn.Conv1d(256, 128, 1)),
            ('bn2', nn.BatchNorm1d(128)),
            ('selu2', nn.SELU()),
            ('dropout2', nn.Dropout(0.2))
        ]))
        self.fc = nn.Conv1d(128, num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self._initialize_weights()

    def forward(self, p, x=None):
        p, x = self.conv1(p, x)
        p, x = self.conv2(p, x)
        p, x = self.conv3(p, x)
        p, x = self.conv4(p, x)
        x = self.mlp(x)
        x = self.fc(x)
        x = self.avgpool(x).squeeze(2)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                stddev = 1.0 / math.sqrt(m.weight.size(1))
                nn.init.normal_(m.weight, 0, stddev)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                stddev = 1.0 / math.sqrt(m.weight.size(1))
                nn.init.normal_(m.weight, 0, stddev)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
