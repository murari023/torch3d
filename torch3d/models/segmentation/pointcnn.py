import torch
import torch.nn as nn
from torch3d.nn import XConv, Downsample


__all__ = ["PointCNN"]


class PointCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PointCNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.down1 = Downsample(2048, mode="farthest")
        self.conv1 = XConv(in_channels, 256, 8, dilation=1)
        self.down2 = Downsample(768, mode="farthest")
        self.conv2 = XConv(256, 256, 12, dilation=2)
        self.down3 = Downsample(384, mode="farthest")
        self.conv3 = XConv(256, 512, 16, dilation=2)
        self.down4 = Downsample(128, mode="farthest")
        self.conv4 = XConv(512, 1024, 16, dilation=4)
        self.conv5 = XConv(1024, 512, 16, dilation=6)
        self.conv6 = XConv(512, 256, 12, dilation=4)
        self.conv7 = XConv(256, 256, 8, dilation=4)

    def forward(self, p, x=None):
        q1, _ = self.down1(p)
        p1, x = self.conv1(p, q1, x)
        q2, _ = self.down2(p1)
        p2, x = self.conv2(p1, q2, x)
        q3, _ = self.down3(p2)
        p3, x = self.conv3(p2, q3, x)
        q4, _ = self.down4(p3)
        p4, x = self.conv4(p3, q4, x)
        p3, x = self.conv5(p4, p3, x)
        p2, x = self.conv6(p3, p2, x)
        p1, x = self.conv7(p2, p1, x)
        print(p1.shape, x.shape)
        return x


if __name__ == "__main__":
    p = torch.rand(1, 2048, 3)
