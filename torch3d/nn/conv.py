import torch
import torch.nn as nn
import torch3d.nn.functional as F


__all__ = ["XConv"]


class XConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(XConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels // 4
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.mlp = nn.Sequential(
            nn.Conv2d(3, self.mid_channels, 1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(True),
            nn.Conv2d(self.mid_channels, self.mid_channels, 1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(True),
        )
        self.stn = nn.Sequential(
            nn.Conv2d(3, self.kernel_size ** 2, [1, self.kernel_size]),
            nn.BatchNorm2d(self.kernel_size ** 2),
            nn.ReLU(True),
            nn.Conv2d(self.kernel_size ** 2, self.kernel_size ** 2, 1),
            nn.BatchNorm2d(self.kernel_size ** 2),
            nn.ReLU(True),
            nn.Conv2d(self.kernel_size ** 2, self.kernel_size ** 2, 1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channels + self.mid_channels,
                      self.out_channels,
                      [1, self.kernel_size]),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, p, q, x=None):
        batch_size = p.shape[0]
        _, indices = F.knn(q, p, self.kernel_size * self.dilation)
        indices = indices[..., ::self.dilation]
        p = self._gather_nd(p, indices)
        p_hat = p - q.unsqueeze(2)
        p_hat = p_hat.permute(0, 3, 1, 2)
        x_hat = self.mlp(p_hat)
        x_hat = x_hat.permute(0, 2, 3, 1)
        if x is not None:
            x = x.permute(0, 2, 1)
            x = self._gather_nd(x, indices)
            x_hat = torch.cat([x_hat, x], dim=-1)
        T = self.stn(p_hat)
        T = T.view(batch_size, self.kernel_size, self.kernel_size, -1)
        T = T.permute(0, 3, 1, 2)
        x_hat = torch.matmul(T, x_hat)
        x = x_hat
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.squeeze(3)
        return q, x

    def _gather_nd(self, x, indices):
        x = [x[b, i, :] for b, i in enumerate(torch.unbind(indices, dim=0))]
        x = torch.stack(x, dim=0)
        return x
