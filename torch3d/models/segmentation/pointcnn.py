import math
import torch
import torch.nn as nn
from collections import OrderedDict
from torch3d.nn import XConv


__all__ = ['PointCNN']


class PointCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointCNN, self).__init__()
        self.conv1 = XConv(2048, in_channels, 256, 8, sampling='furthest')
        self.conv2 = XConv(768, 256, 256, 12, dilation=2, sampling='furthest')
        self.conv3 = XConv(384, 256, 512, 16, dilation=2, sampling='furthest')
        self.conv4 = XConv(128, 512, 1024, 16, dilation=6, sampling='furthest')
        self.dconv3 = XConv(384, 1024, 512, 16, dilation=6)
        self.dconv2 = XConv(768, 512, 256, 12, dilation=4)
        self.dconv1 = XConv(384, 256, 256, 8, dilation=4)
        self.mlp = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(256, 256, 1)),
            ('bn1', nn.BatchNorm1d(256)),
            ('selu1', nn.SELU()),
            ('conv2', nn.Conv1d(256, 128, 1)),
            ('bn2', nn.BatchNorm1d(128)),
            ('selu2', nn.SELU())
        ]))
        self.fc = nn.Conv1d(128, num_classes, 1)
        self._initialize_weights()

    def forward(self, p, x=None):
        p1, x = self.conv1(p, x)
        p2, x = self.conv2(p1, x)
        p3, x = self.conv3(p2, x)
        q4, x = self.conv4(p3, x)
        q3, x = self.dconv3(q4, x, p3.transpose(2, 1))
        q2, x = self.dconv2(q3, x, p2.transpose(2, 1))
        q1, x = self.dconv1(q2, x, p1.transpose(2, 1))
        x = self.mlp(x)
        x = self.fc(x)
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
