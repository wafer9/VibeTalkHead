"""Strict two-branch motion autoencoder.

The target frame is allowed to enter the renderer only through a compact motion
vector.  All spatial appearance features come from a single cached reference
image.  This is intentionally different from the original LIA-X shared style
space: the structural separation is what makes the exported code useful as an
identity-agnostic target for an audio-conditioned generative model.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    for value in (32, 16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, kernel // 2)
        self.norm = nn.GroupNorm(_groups(out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, downsample: bool = False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, stride)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm2(self.conv2(h))
        return F.silu((h + self.skip(x)) * (2.0 ** -0.5))


class MotionEncoder(nn.Module):
    """Frame-wise deterministic motion encoder."""

    def __init__(
        self,
        motion_dim: int = 64,
        base_channels: int = 32,
        blocks_per_stage: int = 2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if blocks_per_stage < 1:
            raise ValueError("motion blocks_per_stage must be at least 1")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        channels = [base_channels, base_channels * 2, base_channels * 4,
                    base_channels * 8, base_channels * 8]
        self.stem = ConvNormAct(3, channels[0], 7, 2)
        stages: List[nn.Module] = []
        in_ch = channels[0]
        for out_ch in channels[1:]:
            blocks: List[nn.Module] = [ResidualBlock(in_ch, out_ch, downsample=True)]
            blocks.extend(
                ResidualBlock(out_ch, out_ch)
                for _ in range(blocks_per_stage - 1)
            )
            stages.append(nn.Sequential(*blocks))
            in_ch = out_ch
        # Keep the historical flat key layout when blocks_per_stage=2 so old
        # 256 checkpoints continue to load strictly.
        self.blocks = nn.Sequential(*(block for stage in stages for block in stage))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], channels[-1]),
            nn.SiLU(),
            nn.Linear(channels[-1], motion_dim),
        )
        # Keep the initial latent compact without bounding its eventual range.
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.blocks, x, use_reentrant=False)
        else:
            x = self.blocks(x)
        return self.head(x)


class ReferenceEncoder(nn.Module):
    """Appearance-only spatial pyramid, evaluated once per generated video."""

    def __init__(
        self,
        base_channels: int = 32,
        blocks_per_stage: int = 1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if blocks_per_stage < 0:
            raise ValueError("reference blocks_per_stage must be non-negative")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        channels = [base_channels, base_channels * 2, base_channels * 4,
                    base_channels * 8, base_channels * 8]
        self.channels = channels
        self.stem = ConvNormAct(3, channels[0], 7, 2)
        stages: List[nn.Module] = []
        for index in range(1, len(channels)):
            modules: List[nn.Module] = [
                ResidualBlock(channels[index - 1], channels[index], downsample=True)
            ]
            modules.extend(
                ResidualBlock(channels[index], channels[index])
                for _ in range(blocks_per_stage)
            )
            stages.append(nn.Sequential(*modules))
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = [self.stem(x)]
        for stage in self.stages:
            if self.gradient_checkpointing and self.training:
                features.append(checkpoint(stage, features[-1], use_reentrant=False))
            else:
                features.append(stage(features[-1]))
        return features  # fine -> coarse, H/2 ... H/32


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, motion_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels, affine=False)
        self.norm2 = nn.GroupNorm(_groups(channels), channels, affine=False)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.affine1 = nn.Linear(motion_dim, channels * 2)
        self.affine2 = nn.Linear(motion_dim, channels * 2)
        nn.init.zeros_(self.affine1.weight)
        nn.init.zeros_(self.affine1.bias)
        nn.init.zeros_(self.affine2.weight)
        nn.init.zeros_(self.affine2.bias)

    @staticmethod
    def _modulate(x: torch.Tensor, affine: nn.Linear, motion: torch.Tensor) -> torch.Tensor:
        scale, shift = affine(motion).chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        h = self._modulate(self.norm1(x), self.affine1, motion)
        h = self.conv1(F.silu(h))
        h = self._modulate(self.norm2(h), self.affine2, motion)
        h = self.conv2(F.silu(h))
        return (x + h) * (2.0 ** -0.5)


class CausalMotionBlock(nn.Module):
    def __init__(self, motion_dim: int, kernel_size: int = 5, layers: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.depthwise = nn.ModuleList()
        self.pointwise = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            self.depthwise.append(nn.Conv1d(
                motion_dim, motion_dim, kernel_size, groups=motion_dim, padding=0
            ))
            self.pointwise.append(nn.Conv1d(motion_dim, motion_dim * 2, 1))
            self.norms.append(nn.LayerNorm(motion_dim))
        self.out = nn.Linear(motion_dim, motion_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, motion: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        # motion: [B,T,D].  Left-only padding makes every block streamable.
        h = motion
        for depthwise, pointwise, norm in zip(self.depthwise, self.pointwise, self.norms):
            y = h.transpose(1, 2)
            y = F.pad(y, (self.kernel_size - 1, 0))
            y = depthwise(y)
            value, gate = pointwise(y).chunk(2, dim=1)
            y = value * torch.sigmoid(gate)
            h = norm(h + y.transpose(1, 2))
        residual = self.out(h)
        return motion + float(strength) * torch.tanh(self.gate) * residual


def _base_grid(batch: int, height: int, width: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)


def warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp x by normalized coordinate offsets in flow [B,2,H,W]."""
    grid = _base_grid(x.shape[0], x.shape[-2], x.shape[-1], x.device, x.dtype)
    return F.grid_sample(
        x, grid + flow.permute(0, 2, 3, 1), mode="bilinear",
        padding_mode="border", align_corners=True,
    )


class FlowRenderStage(nn.Module):
    def __init__(self, in_ch: int, ref_ch: int, out_ch: int, motion_dim: int,
                 max_flow: float, upsample: bool, blocks_per_stage: int = 2):
        super().__init__()
        if blocks_per_stage < 2:
            raise ValueError("renderer blocks_per_stage must be at least 2")
        self.upsample = upsample
        self.max_flow = max_flow
        self.up = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.ref_proj = nn.Conv2d(ref_ch, out_ch, 1)
        self.pre = nn.Conv2d(out_ch * 2, out_ch, 3, 1, 1)
        self.motion_bias = nn.Linear(motion_dim, out_ch)
        self.block1 = FiLMResidualBlock(out_ch, motion_dim)
        self.block2 = FiLMResidualBlock(out_ch, motion_dim)
        self.extra_blocks = nn.ModuleList(
            FiLMResidualBlock(out_ch, motion_dim)
            for _ in range(blocks_per_stage - 2)
        )
        self.flow_head = nn.Sequential(
            nn.GroupNorm(_groups(out_ch), out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, 2, 3, 1, 1),
        )
        self.mask_head = nn.Sequential(
            nn.GroupNorm(_groups(out_ch), out_ch), nn.SiLU(),
            nn.Conv2d(out_ch, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)
        nn.init.zeros_(self.mask_head[-1].weight)
        nn.init.constant_(self.mask_head[-1].bias, 1.0)

    def forward(
        self,
        h: torch.Tensor,
        ref: torch.Tensor,
        motion: torch.Tensor,
        flow: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.upsample:
            h = F.interpolate(h, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up(h)
        h = h + self.motion_bias(motion)[:, :, None, None]
        if flow is None:
            flow = torch.zeros(
                h.shape[0], 2, h.shape[-2], h.shape[-1],
                device=h.device, dtype=h.dtype,
            )
        else:
            flow = F.interpolate(flow, size=h.shape[-2:], mode="bilinear", align_corners=False)
        residual_flow = torch.tanh(self.flow_head(h)) * self.max_flow
        flow = flow + residual_flow
        warped_ref = warp(ref, flow)
        warped_ref = self.ref_proj(warped_ref)
        h = self.pre(torch.cat([h, warped_ref], dim=1))
        h = self.block2(self.block1(h, motion), motion)
        for block in self.extra_blocks:
            h = block(h, motion)
        mask = torch.sigmoid(self.mask_head(h))
        h = mask * warped_ref + (1.0 - mask) * h
        return h, flow, mask


class SourceAnchoredRenderer(nn.Module):
    def __init__(
        self,
        ref_channels: Sequence[int],
        motion_dim: int = 64,
        blocks_per_stage: int = 2,
        full_resolution_stage: bool = False,
        full_resolution_channels: Optional[int] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.full_resolution_stage = bool(full_resolution_stage)
        # Work coarse -> fine. Normalized flow limits become finer by scale.
        decoder_channels = list(reversed(ref_channels))
        reference_channels = list(reversed(ref_channels))
        max_flows = [0.20, 0.15, 0.10, 0.075, 0.05]
        self.input = nn.Linear(motion_dim, decoder_channels[0] * 4 * 4)
        stages = []
        in_ch = decoder_channels[0]
        for index, (ref_ch, out_ch) in enumerate(zip(reference_channels, decoder_channels)):
            stages.append(FlowRenderStage(
                in_ch, ref_ch, out_ch, motion_dim,
                max_flow=max_flows[index], upsample=index > 0,
                blocks_per_stage=blocks_per_stage,
            ))
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.full_stage: Optional[FlowRenderStage]
        if self.full_resolution_stage:
            full_ch = int(full_resolution_channels or decoder_channels[-1])
            self.full_stage = FlowRenderStage(
                in_ch, 3, full_ch, motion_dim, max_flow=0.025, upsample=True,
                blocks_per_stage=blocks_per_stage,
            )
            final_ch = full_ch
        else:
            self.full_stage = None
            final_ch = decoder_channels[-1]
        self.to_render = nn.Sequential(
            ConvNormAct(final_ch, final_ch),
            nn.Conv2d(final_ch, 3, 3, 1, 1),
        )
        self.to_mask = nn.Sequential(
            ConvNormAct(final_ch, final_ch),
            nn.Conv2d(final_ch, 1, 3, 1, 1),
        )
        nn.init.zeros_(self.to_mask[-1].weight)
        nn.init.constant_(self.to_mask[-1].bias, 1.5)

    def forward(
        self,
        reference_rgb: torch.Tensor,
        reference_features: Sequence[torch.Tensor],
        motion: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch = motion.shape[0]
        coarse_size = reference_features[-1].shape[-2:]
        h = self.input(motion).reshape(batch, -1, 4, 4)
        h = F.interpolate(h, size=coarse_size, mode="bilinear", align_corners=False)
        flow = None
        masks = []
        flows = []
        for stage, ref in zip(self.stages, reversed(reference_features)):
            h, flow, mask = self._run_stage(stage, h, ref, motion, flow)
            masks.append(mask)
            flows.append(flow)

        if self.full_stage is not None:
            h, flow, mask = self._run_stage(
                self.full_stage, h, reference_rgb, motion, flow
            )
            masks.append(mask)
            flows.append(flow)
        else:
            h = F.interpolate(
                h, size=reference_rgb.shape[-2:], mode="bilinear", align_corners=False
            )
        render = torch.tanh(self.to_render(h))
        final_mask = torch.sigmoid(self.to_mask(h))
        final_flow = F.interpolate(flow, size=reference_rgb.shape[-2:], mode="bilinear", align_corners=False)
        warped_rgb = warp(reference_rgb, final_flow)
        image = final_mask * warped_rgb + (1.0 - final_mask) * render
        return {
            "image": image,
            "flow": final_flow,
            "mask": final_mask,
            "pyramid_flows": flows,
            "pyramid_masks": masks,
        }

    def _run_stage(
        self,
        stage: FlowRenderStage,
        h: torch.Tensor,
        reference: torch.Tensor,
        motion: torch.Tensor,
        flow: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.gradient_checkpointing or not self.training:
            return stage(h, reference, motion, flow)
        if flow is None:
            return checkpoint(
                lambda a, b, c: stage(a, b, c, None),
                h, reference, motion, use_reentrant=False,
            )
        return checkpoint(
            stage, h, reference, motion, flow, use_reentrant=False
        )


class MotionNormalizer(nn.Module):
    """Dataset-level statistics with distributed-safe finalization."""

    def __init__(self, dim: int, std_floor: float = 1e-3):
        super().__init__()
        self.dim = dim
        self.std_floor = std_floor
        self.register_buffer("sum", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("sq_sum", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("std", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("ready", torch.zeros((), dtype=torch.uint8))

    @torch.no_grad()
    def update(self, motion: torch.Tensor) -> None:
        if bool(self.ready.item()):
            return
        flat = motion.detach().reshape(-1, self.dim).double()
        self.sum.add_(flat.sum(dim=0))
        self.sq_sum.add_(flat.square().sum(dim=0))
        self.count.add_(flat.shape[0])

    @torch.no_grad()
    def finalize(self) -> None:
        total_sum = self.sum.clone()
        total_sq = self.sq_sum.clone()
        total_count = self.count.clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_sum)
            dist.all_reduce(total_sq)
            dist.all_reduce(total_count)
        count = total_count.clamp_min(1.0)
        mean = total_sum / count
        var = (total_sq / count - mean.square()).clamp_min(self.std_floor ** 2)
        self.mean.copy_(mean.float())
        self.std.copy_(var.sqrt().float())
        self.ready.fill_(1)

    def normalize(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean.to(motion)) / self.std.to(motion)

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std.to(motion) + self.mean.to(motion)


def structured_motion_noise(x: torch.Tensor, max_sigma: float, mode: str = "mixed") -> torch.Tensor:
    """Add DiT-like errors in normalized motion coordinates."""
    if max_sigma <= 0:
        return x
    batch, time, dim = x.shape
    sigma = torch.rand(batch, 1, 1, device=x.device, dtype=x.dtype) * max_sigma

    iid = torch.randn_like(x)
    bias = torch.randn(batch, 1, dim, device=x.device, dtype=x.dtype).expand_as(x)
    ar = torch.empty_like(x)
    ar[:, 0] = torch.randn_like(x[:, 0])
    rho = 0.92
    innovation = math.sqrt(1.0 - rho * rho)
    for index in range(1, time):
        ar[:, index] = rho * ar[:, index - 1] + innovation * torch.randn_like(ar[:, index])

    if mode == "iid":
        noise = iid
    elif mode == "bias":
        noise = bias
    elif mode == "drift":
        noise = ar
    elif mode == "mixed":
        weights = torch.rand(batch, 3, 1, 1, device=x.device, dtype=x.dtype)
        weights = weights / weights.square().sum(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        noise = weights[:, 0] * iid + weights[:, 1] * bias + weights[:, 2] * ar
    else:
        raise ValueError(f"unknown motion noise mode: {mode}")
    return x + sigma * noise


class MotionTokenizer(nn.Module):
    def __init__(
        self,
        motion_dim: int = 64,
        base_channels: int = 32,
        motion_blocks_per_stage: int = 2,
        reference_blocks_per_stage: int = 1,
        renderer_blocks_per_stage: int = 2,
        full_resolution_stage: bool = False,
        full_resolution_channels: Optional[int] = None,
        gradient_checkpointing: bool = False,
        motion_input_size: int = 256,
        causal_kernel_size: int = 5,
        causal_layers: int = 3,
    ):
        super().__init__()
        self.motion_dim = motion_dim
        self.motion_input_size = motion_input_size
        self.motion_encoder = MotionEncoder(
            motion_dim, base_channels, motion_blocks_per_stage,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.reference_encoder = ReferenceEncoder(
            base_channels, reference_blocks_per_stage,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.motion_adapter = CausalMotionBlock(motion_dim, causal_kernel_size, causal_layers)
        self.renderer = SourceAnchoredRenderer(
            self.reference_encoder.channels,
            motion_dim,
            blocks_per_stage=renderer_blocks_per_stage,
            full_resolution_stage=full_resolution_stage,
            full_resolution_channels=full_resolution_channels,
            gradient_checkpointing=gradient_checkpointing,
        )
        self.normalizer = MotionNormalizer(motion_dim)

    def encode_motion(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != (self.motion_input_size, self.motion_input_size):
            image = F.interpolate(
                image, size=(self.motion_input_size, self.motion_input_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
        return self.motion_encoder(image)

    @staticmethod
    def _expand_reference(x: torch.Tensor, count: int) -> torch.Tensor:
        return x[:, None].expand(-1, count, *x.shape[1:]).reshape(-1, *x.shape[1:])

    def _render_sequence(
        self,
        reference: torch.Tensor,
        reference_features: Sequence[torch.Tensor],
        delta: torch.Tensor,
        render_chunk: int,
    ) -> Dict[str, torch.Tensor]:
        batch, time, _ = delta.shape
        images, flows, masks = [], [], []
        render_chunk = time if render_chunk <= 0 else render_chunk
        for start in range(0, time, render_chunk):
            stop = min(time, start + render_chunk)
            count = stop - start
            motion = delta[:, start:stop].reshape(batch * count, -1)
            ref_rgb = self._expand_reference(reference, count)
            ref_feats = [self._expand_reference(feature, count) for feature in reference_features]
            result = self.renderer(ref_rgb, ref_feats, motion)
            images.append(result["image"].reshape(batch, count, *result["image"].shape[1:]))
            flows.append(result["flow"].reshape(batch, count, *result["flow"].shape[1:]))
            masks.append(result["mask"].reshape(batch, count, *result["mask"].shape[1:]))
        return {
            "image": torch.cat(images, dim=1),
            "flow": torch.cat(flows, dim=1),
            "mask": torch.cat(masks, dim=1),
        }

    def forward(
        self,
        reference: torch.Tensor,
        frames: torch.Tensor,
        *,
        cross_reference: Optional[torch.Tensor] = None,
        causal_strength: float = 1.0,
        noise_sigma: float = 0.0,
        noise_mode: str = "mixed",
        render_chunk: int = 0,
        return_clean_with_noise: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch, time = frames.shape[:2]
        reference_motion = self.encode_motion(reference)
        target_motion = self.encode_motion(frames.flatten(0, 1)).reshape(batch, time, -1)
        clean_delta = target_motion - reference_motion[:, None]
        clean_delta = self.motion_adapter(clean_delta, causal_strength)
        reference_features = self.reference_encoder(reference)

        clean_render = self._render_sequence(
            reference, reference_features, clean_delta, render_chunk
        )
        output: Dict[str, torch.Tensor] = {
            "reconstruction": clean_render["image"],
            "flow": clean_render["flow"],
            "mask": clean_render["mask"],
            "reference_motion": reference_motion,
            "target_motion": target_motion,
            "normalized_motion": self.normalizer.normalize(target_motion),
        }

        if noise_sigma > 0:
            if not bool(self.normalizer.ready.item()):
                raise RuntimeError("motion normalizer must be finalized before noise training")
            normalized = self.normalizer.normalize(target_motion)
            noisy_normalized = structured_motion_noise(normalized, noise_sigma, noise_mode)
            noisy_motion = self.normalizer.denormalize(noisy_normalized)
            noisy_delta = noisy_motion - reference_motion[:, None]
            noisy_delta = self.motion_adapter(noisy_delta, causal_strength)
            noisy_render = self._render_sequence(
                reference, reference_features, noisy_delta, render_chunk
            )
            output["noisy_reconstruction"] = noisy_render["image"]
            output["noisy_flow"] = noisy_render["flow"]
            output["noisy_mask"] = noisy_render["mask"]
            if not return_clean_with_noise:
                output["reconstruction"] = output["noisy_reconstruction"]
                output["flow"] = output["noisy_flow"]
                output["mask"] = output["noisy_mask"]

        if cross_reference is not None:
            cross_features = self.reference_encoder(cross_reference)
            cross_reference_motion = self.encode_motion(cross_reference)
            # One random-ish central frame is enough for the expensive cycle branch.
            selected_motion = target_motion[:, time // 2]
            cross_target_delta = selected_motion - cross_reference_motion
            adapted_cross_delta = self.motion_adapter(
                cross_target_delta[:, None], causal_strength
            )
            cross_render = self._render_sequence(
                cross_reference, cross_features, adapted_cross_delta, render_chunk=1
            )["image"][:, 0]
            cross_cycle_motion = self.encode_motion(cross_render)
            # Compare changes relative to the same cross reference.  The two
            # cross_reference_motion terms make the cycle gradient invariant to
            # a common encoder offset, while preserving the absolute target-pose
            # semantics used by rendering and exported motion codes.
            cross_cycle_delta = cross_cycle_motion - cross_reference_motion
            output.update({
                "cross_reconstruction": cross_render,
                "cross_reference": cross_reference,
                "cross_reference_motion": cross_reference_motion,
                "cross_target_motion": selected_motion,
                "cross_cycle_motion": cross_cycle_motion,
                "cross_target_delta": cross_target_delta,
                "cross_cycle_delta": cross_cycle_delta,
            })
        return output

    @torch.inference_mode()
    def render_normalized(
        self,
        reference: torch.Tensor,
        normalized_motion: torch.Tensor,
        causal_strength: float = 1.0,
        render_chunk: int = 0,
    ) -> torch.Tensor:
        """Render exported normalized motion [B,T,D] from a reference image."""
        if not bool(self.normalizer.ready.item()):
            raise RuntimeError("checkpoint has no finalized corpus normalization statistics")
        target_motion = self.normalizer.denormalize(normalized_motion)
        reference_motion = self.encode_motion(reference)
        delta = self.motion_adapter(
            target_motion - reference_motion[:, None], causal_strength
        )
        features = self.reference_encoder(reference)
        return self._render_sequence(reference, features, delta, render_chunk)["image"]
