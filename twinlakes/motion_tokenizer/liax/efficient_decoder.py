"""Latency-oriented LIA-X decoder prototype.

This module is intentionally separate from the training decoder.  It keeps the
reference-feature warp and LIA-X motion direction, while replacing the deep
StyleGAN-style flow/RGB stacks with a small ConvNeXt-FiLM synthesis trunk.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .ops import Direction


class ConvNeXtFiLMBlock(nn.Module):
    """A compact channels-last ConvNeXt block with stage-shared FiLM."""

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        kernel_size: int = 5,
        layer_scale_init: float = 5e-2,
    ):
        super().__init__()
        hidden = channels * int(expansion)
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=padding, groups=channels
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, hidden)
        self.project = nn.Linear(hidden, channels)
        self.layer_scale = nn.Parameter(
            torch.full((channels,), float(layer_scale_init))
        )

    def forward(
        self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = F.gelu(x, approximate="tanh")
        x = self.project(x)
        x = x * (1.0 + scale) + shift
        x = x * self.layer_scale
        return residual + x.permute(0, 3, 1, 2)


class FiLMStage(nn.Module):
    """Project motion once and reuse the modulation across all stage blocks."""

    def __init__(
        self,
        channels: int,
        style_dim: int,
        depth: int,
        expansion: int = 2,
        kernel_size: int = 5,
        film_init_std: float = 1e-2,
        layer_scale_init: float = 5e-2,
    ):
        super().__init__()
        self.film = nn.Linear(style_dim, channels * 2)
        self.blocks = nn.ModuleList(
            ConvNeXtFiLMBlock(
                channels, expansion, kernel_size,
                layer_scale_init=layer_scale_init,
            )
            for _ in range(int(depth))
        )
        nn.init.normal_(self.film.weight, std=float(film_init_std))
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(style).chunk(2, dim=1)
        scale = scale[:, None, None, :]
        shift = shift[:, None, None, :]
        for block in self.blocks:
            x = block(x, scale, shift)
        return x


class StyleFlowHead(nn.Module):
    """Predict a bounded residual flow from explicitly style-conditioned features."""

    def __init__(
        self,
        channels: int,
        style_dim: int,
        weight_scale: float = 1e-3,
        style_init_std: float = 1e-2,
    ):
        super().__init__()
        self.style = nn.Linear(style_dim, channels * 2)
        self.output = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.normal_(self.style.weight, std=float(style_init_std))
        nn.init.zeros_(self.style.bias)
        with torch.no_grad():
            self.output.weight.mul_(float(weight_scale))
            self.output.bias.zero_()

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.style(style).chunk(2, dim=1)
        conditioned = x * (1.0 + scale[:, :, None, None])
        conditioned = conditioned + shift[:, :, None, None]
        return self.output(F.silu(conditioned))


class GatedConcatFusion(nn.Module):
    """Preserve synthesis features while injecting warped reference residuals."""

    def __init__(
        self,
        channels: int,
        style_dim: int,
        expansion: int = 2,
        kernel_size: int = 5,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.compress = nn.Conv2d(channels * 2 + 3, channels, 1)
        self.refine = FiLMStage(
            channels,
            style_dim,
            depth=1,
            expansion=expansion,
            kernel_size=kernel_size,
        )
        self.gate = nn.Parameter(torch.full((1, channels, 1, 1), gate_init))

    def forward(
        self,
        x: torch.Tensor,
        warped: torch.Tensor,
        mask: torch.Tensor,
        offset: torch.Tensor,
        style: torch.Tensor,
    ) -> torch.Tensor:
        fusion_input = torch.cat(
            (x, warped - x, mask, offset.to(dtype=x.dtype)), dim=1
        )
        residual = self.compress(fusion_input)
        residual = self.refine(residual, style)
        return x + self.gate * residual


class EfficientWarpDecoder(nn.Module):
    """Four-warp ConvNeXt-FiLM student for decoder latency experiments.

    Reference projections are explicitly separated through ``prepare_reference``
    because a talking-head video reuses one reference across all target frames.
    Their one-time cost should not be charged to per-frame decoder RTF.
    """

    resolutions = (8, 16, 32, 64, 128, 256, 512)
    warp_stage_indices = (3, 4, 5, 6)

    def __init__(
        self,
        style_dim: int = 512,
        motion_dim: int = 40,
        source_channels: Sequence[int] = (512, 512, 512, 512, 256, 128, 64),
        channels: Sequence[int] = (512, 512, 384, 256, 128, 64, 32),
        depths: Sequence[int] = (2, 2, 2, 2, 1, 1, 1),
        expansion: int = 2,
        kernel_size: int = 5,
        flow_limits: Sequence[float] = (0.10, 0.05, 0.025, 0.0125),
    ):
        super().__init__()
        if len(source_channels) != 7 or len(channels) != 7 or len(depths) != 7:
            raise ValueError("source_channels, channels and depths must have 7 entries")
        self.style_dim = int(style_dim)
        self.motion_dim = int(motion_dim)
        self.channels = tuple(int(value) for value in channels)
        self.depths = tuple(int(value) for value in depths)
        if len(flow_limits) != len(self.warp_stage_indices):
            raise ValueError("flow_limits must match the number of warp stages")
        self.flow_limits = tuple(float(value) for value in flow_limits)
        self.direction = Direction(self.style_dim, self.motion_dim)
        self.constant = nn.Parameter(torch.randn(1, self.channels[0], 4, 4))

        previous = self.channels[0]
        self.upsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        for channel, depth in zip(self.channels, self.depths):
            self.upsamples.append(nn.Conv2d(previous, channel, 3, padding=1))
            self.stages.append(
                FiLMStage(
                    channel, self.style_dim, depth,
                    expansion=expansion, kernel_size=kernel_size,
                )
            )
            previous = channel

        self.reference_projections = nn.ModuleList(
            nn.Conv2d(int(source_channels[index]), self.channels[index], 1)
            for index in self.warp_stage_indices
        )
        self.flow_heads = nn.ModuleList(
            StyleFlowHead(self.channels[index], self.style_dim)
            for index in self.warp_stage_indices
        )
        self.fusions = nn.ModuleList(
            GatedConcatFusion(
                self.channels[index], self.style_dim,
                expansion=expansion, kernel_size=kernel_size,
            )
            for index in self.warp_stage_indices
        )
        self.to_rgb = nn.Conv2d(self.channels[-1], 3, 1)

        for resolution in self.resolutions:
            # Pixel centers for grid_sample(..., align_corners=False).
            axis = (torch.arange(resolution, dtype=torch.float32) + 0.5)
            axis = axis.mul_(2.0 / resolution).sub_(1.0)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
            self.register_buffer(f"base_grid_{resolution}", grid, persistent=False)

    def prepare_reference(
        self, features: Sequence[torch.Tensor]
    ) -> list[torch.Tensor]:
        if len(features) != 7:
            raise ValueError(f"expected 7 reference features, got {len(features)}")
        return [
            projection(features[index])
            for projection, index in zip(
                self.reference_projections, self.warp_stage_indices
            )
        ]

    def navigation(self, source_style: torch.Tensor, alpha) -> torch.Tensor:
        if alpha is None:
            return source_style
        if len(alpha) == 1:
            return source_style + self.direction(alpha[0])
        target = self.direction(alpha[0])
        source = self.direction(alpha[1])
        start = self.direction(alpha[2])
        return source_style + (target - start) + source

    @staticmethod
    def _expand_reference(feature: torch.Tensor, batch: int) -> torch.Tensor:
        if feature.shape[0] == batch:
            return feature
        if feature.shape[0] != 1:
            raise ValueError(
                f"cannot expand reference batch {feature.shape[0]} to target batch {batch}"
            )
        return feature.expand(batch, -1, -1, -1)

    def forward(
        self,
        source_style: torch.Tensor,
        alpha,
        prepared_features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(prepared_features) != len(self.warp_stage_indices):
            raise ValueError(
                "prepared_features must contain the 64/128/256/512 features"
            )
        style = self.navigation(source_style, alpha)
        batch = style.shape[0]
        x = self.constant.expand(batch, -1, -1, -1)
        accumulated_offset = None
        warp_index = 0

        for index, (resolution, upsample, stage) in enumerate(
            zip(self.resolutions, self.upsamples, self.stages)
        ):
            x = F.interpolate(x, size=(resolution, resolution), mode="nearest")
            x = upsample(x)
            x = stage(x, style)
            if index not in self.warp_stage_indices:
                continue

            raw = self.flow_heads[warp_index](x, style)
            residual_offset = torch.tanh(raw[:, :2]).float()
            residual_offset = residual_offset * self.flow_limits[warp_index]
            if accumulated_offset is None:
                accumulated_offset = residual_offset
            else:
                accumulated_offset = F.interpolate(
                    accumulated_offset,
                    size=(resolution, resolution),
                    mode="bilinear",
                    align_corners=False,
                ) + residual_offset
            mask = torch.sigmoid(raw[:, 2:3])
            base_grid = getattr(self, f"base_grid_{resolution}")
            grid = base_grid + accumulated_offset.permute(0, 2, 3, 1)
            reference = self._expand_reference(
                prepared_features[warp_index], batch
            )
            warped = F.grid_sample(
                reference, grid, mode="bilinear", padding_mode="zeros",
                align_corners=False,
            )
            x = self.fusions[warp_index](
                x, warped, mask, accumulated_offset, style
            )
            warp_index += 1

        return self.to_rgb(x)
