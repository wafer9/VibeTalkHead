"""Pure PyTorch fallback matching LIA-X's fused leaky-ReLU semantics."""

import torch
from torch import nn
from torch.nn import functional as F


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(
            input, self.bias, self.negative_slope, self.scale
        )


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        input = input + bias.view(1, bias.shape[0], *rest_dim)
    return F.leaky_relu(input, negative_slope=negative_slope) * scale
