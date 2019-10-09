import torch
import torch.nn as nn
from collections import OrderedDict
from torch3d.ops import furthest_point_sample, knn


__all__ = ['XConv']


class XConv(nn.Module):
    def __init__(self,
                 num_points,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation=1,
                 sampling='random'):
        super(XConv, self).__init__()
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sampling = sampling
        self.mid_channels = out_channels // 4
        self.mlp1 = nn.Sequential(OrderedDict([
            ('dense1', nn.Conv2d(3, self.mid_channels, 1, bias=False)),
            ('bn1', nn.BatchNorm2d(self.mid_channels)),
            ('selu1', nn.SELU()),
            ('dense2', nn.Conv2d(self.mid_channels, self.mid_channels, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(self.mid_channels)),
            ('selu2', nn.SELU())
        ]))
        self.mlp2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, self.kernel_size ** 2, [1, self.kernel_size], bias=False)),
            ('bn1', nn.BatchNorm2d(self.kernel_size ** 2)),
            ('selu1', nn.SELU()),
            ('conv2', nn.Conv2d(self.kernel_size ** 2, self.kernel_size ** 2, 1, bias=False)),
            ('bn2', nn.BatchNorm2d(self.kernel_size ** 2)),
            ('selu2', nn.SELU()),
            ('conv3', nn.Conv2d(self.kernel_size ** 2, self.kernel_size ** 2, 1, bias=False))
        ]))
        self.conv = nn.Conv2d(
            self.in_channels + self.mid_channels,
            self.out_channels,
            [1, self.kernel_size],
            bias=False
        )
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.activation = nn.SELU()

    def forward(self, p, x=None, q=None):
        batch_size = p.shape[0]
        p = p.permute(0, 2, 1).contiguous()

        if q is None:
            # random downsample
            if self.sampling == 'random':
                choice = torch.randperm(p.shape[1])[:self.num_points]
                q = p[:, choice, :]
            elif self.sampling == 'furthest':
                choice = furthest_point_sample(p, self.num_points)
                q = self._region_select(p, choice)

        # find k-nearest neighbors
        _, indices = knn(q, p, self.kernel_size * self.dilation)
        indices = indices[..., ::self.dilation]

        p = self._region_select(p, indices)
        p_hat = p - q.unsqueeze(2)  # move p to local coordinates of q
        p_hat = p_hat.permute(0, 3, 1, 2)
        x_hat = self.mlp1(p_hat)  # lifted features
        x_hat = x_hat.permute(0, 2, 3, 1)

        if x is not None:
            x = x.permute(0, 2, 1).contiguous()
            x = self._region_select(x, indices)
            x_hat = torch.cat([x_hat, x], dim=-1)

        # X-transform
        T = self.mlp2(p_hat)
        T = T.view(batch_size, self.kernel_size, self.kernel_size, -1)
        T = T.permute(0, 3, 1, 2)
        x_hat = torch.matmul(T, x_hat)

        x = x_hat
        x = x.permute(0, 3, 1, 2)
        x = self.bn(self.conv(x))
        x = x.squeeze(3)
        x = self.activation(x)
        q = q.permute(0, 2, 1).contiguous()
        return q, x

    def _region_select(self, x, indices):
        x = [x[b, i, :] for b, i in enumerate(torch.unbind(indices, dim=0))]
        x = torch.stack(x, dim=0)
        return x
