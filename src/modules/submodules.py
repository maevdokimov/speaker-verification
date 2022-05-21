import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvNorm(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, forward_padding=0, side_padding=0, pad_value=0.0, **kwargs
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")

        self.pad_value = pad_value
        self.pad_size = forward_padding, 0, side_padding, side_padding

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): batch x channels x D x T
        """
        x = F.pad(x, self.pad_size, "constant", self.pad_value)

        return self.conv(x)


class SELayer(nn.Module):
    def __init__(self, n_features, reduction):
        super().__init__()
        self.n_features = n_features
        self.reduction = reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(n_features, n_features // reduction),
            nn.ReLU(),
            nn.Linear(n_features // reduction, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): B x C x T x D
        """
        inp = x

        x = self.avg_pool(x).reshape(x.shape[0], x.shape[1])  # B x C
        x = self.excitation(x)
        x = x.reshape(*x.shape, 1, 1)

        return x * inp
