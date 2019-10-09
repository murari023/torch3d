import torch
import torch.nn as nn
from collections import OrderedDict


__all__ = ['PointNet']


class PointNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mlp1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(self.in_channels, 64, 1, bias=False)),
            ('bn1', nn.BatchNorm1d(64)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv1d(64, 64, 1, bias=False)),
            ('bn2', nn.BatchNorm1d(64)),
            ('relu2', nn.ReLU()),
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(64, 128, 1, bias=False)),
            ('bn1', nn.BatchNorm1d(128)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv1d(128, 1024, 1, bias=False)),
            ('bn2', nn.BatchNorm1d(1024)),
            ('relu2', nn.ReLU())
        ]))
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.mlp3 = nn.Sequential(OrderedDict([
            ('dense1', nn.Linear(1024, 512, bias=False)),
            ('bn1', nn.BatchNorm1d(512)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('dense2', nn.Linear(512, 256, bias=False)),
            ('bn2', nn.BatchNorm1d(256)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
        ]))
        self.fc = nn.Linear(256, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.maxpool(x).squeeze(2)
        x = self.mlp3(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
