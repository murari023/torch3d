import torch
import torch.nn as nn
import torch3d.nn.functional as F


__all__ = ["SetAbstraction"]


class SetAbstraction(nn.Module):
    def __init__(self, mlp, radius=None, k=None):
        super(SetAbstraction, self).__init__()
        self.radius = radius
        self.k = k
        modules = []
        last_channels = mlp[0]
        for channels in mlp[1:]:
            modules.append(nn.Conv2d(last_channels, channels, 1, bias=False))
            modules.append(nn.BatchNorm2d(channels))
            modules.append(nn.ReLU(True))
            last_channels = channels
        self.mlp = nn.Sequential(*modules)
        self.maxpool = nn.MaxPool2d([1, k])

    def forward(self, p, q, x=None):
        if self.radius is not None:
            indices = F.ball_point(p, q, self.radius, self.k)
            p = torch.stack([p[b, i, :] for b, i in enumerate(indices)], dim=0)
            p_hat = p - q.unsqueeze(2)
            x_hat = p_hat
        else:
            x_hat = p.unsqueeze(1)
        if x is not None:
            x = x.permute(0, 2, 1)
            if self.radius is not None:
                x = torch.stack([x[b, i, :] for b, i in enumerate(indices)], dim=0)
            else:
                x = x.unsqueeze(1)
            x_hat = torch.cat([x_hat, x], dim=-1)
        x = x_hat.permute(0, 3, 1, 2)
        x = self.mlp(x)
        x = self.maxpool(x).squeeze(3)
        return q, x
