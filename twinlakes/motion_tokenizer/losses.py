"""Losses and discriminators for the motion tokenizer."""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.utils.checkpoint import checkpoint

from .liax.ops import ConvLayer, EqualLinear


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


class LaplacianPyramidLoss(nn.Module):
    """Download-free perceptual-ish image pyramid loss."""

    def __init__(self, levels: int = 4):
        super().__init__()
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(3, 1, 1, 1))
        self.levels = levels

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.kernel.to(x), groups=3)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.flatten(0, 1) if prediction.ndim == 5 else prediction
        target = target.flatten(0, 1) if target.ndim == 5 else target
        total = prediction.new_zeros(())
        weight = 1.0
        for _ in range(self.levels):
            pred_blur = self._blur(prediction)
            target_blur = self._blur(target)
            total = total + weight * charbonnier(prediction - pred_blur, target - target_blur)
            prediction = F.avg_pool2d(pred_blur, 2)
            target = F.avg_pool2d(target_blur, 2)
            weight *= 0.5
        return total


class LocalVGGPerceptualLoss(nn.Module):
    """Frozen VGG feature loss with optional LIA-style image pyramids.

    Weights are always loaded from a local file.  The LIA setting uses VGG19
    features after relu1_1..relu5_1 and sums feature L1 over 256/128/64/32
    inputs, matching the perceptual pyramid described in the paper.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        model_name: str = "vgg16",
        pyramid_sizes: Optional[Iterable[int]] = None,
        feature_weights: Optional[Iterable[float]] = None,
    ):
        super().__init__()
        self.enabled = bool(weights_path)
        self.model_name = str(model_name).lower()
        self.pyramid_sizes = tuple(int(size) for size in (pyramid_sizes or ()))
        if not self.enabled:
            self.blocks = nn.ModuleList()
            return
        if not os.path.isfile(str(weights_path)):
            raise FileNotFoundError(f"VGG weights not found: {weights_path}")
        if self.model_name == "vgg19":
            from torchvision.models import vgg19
            model = vgg19(weights=None)
            boundaries = (2, 7, 12, 21, 30)
        elif self.model_name == "vgg16":
            from torchvision.models import vgg16
            model = vgg16(weights=None)
            boundaries = (4, 9, 16, 23)
        else:
            raise ValueError(f"unsupported VGG model: {model_name}")
        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict) and "state_dict" in payload:
            payload = payload["state_dict"]
        model.load_state_dict(payload)
        features = model.features
        starts = (0,) + boundaries[:-1]
        self.blocks = nn.ModuleList(
            features[start:stop] for start, stop in zip(starts, boundaries)
        )
        if feature_weights is None:
            feature_weights = [1.0] * len(self.blocks)
        self.feature_weights = tuple(float(weight) for weight in feature_weights)
        if len(self.feature_weights) != len(self.blocks):
            raise ValueError(
                "feature_weights must have one value per VGG feature block: "
                f"expected {len(self.blocks)}, got {len(self.feature_weights)}"
            )
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _single_scale(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = (
            prediction.clamp(-1, 1).add(1).mul(0.5) - self.mean.to(prediction)
        ) / self.std.to(prediction)
        target = (
            target.clamp(-1, 1).add(1).mul(0.5) - self.mean.to(target)
        ) / self.std.to(target)
        total = prediction.new_zeros(())
        x, y = prediction, target
        for block, weight in zip(self.blocks, self.feature_weights):
            x = block(x)
            with torch.no_grad():
                y = block(y)
            total = total + weight * F.l1_loss(x, y)
        return total

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return prediction.new_zeros(())
        prediction = prediction.flatten(0, 1) if prediction.ndim == 5 else prediction
        target = target.flatten(0, 1) if target.ndim == 5 else target
        if not self.pyramid_sizes:
            return self._single_scale(prediction, target)
        total = prediction.new_zeros(())
        for size in self.pyramid_sizes:
            pred_scale = F.interpolate(
                prediction, size=(size, size), mode="bilinear",
                align_corners=False, antialias=True,
            )
            target_scale = F.interpolate(
                target, size=(size, size), mode="bilinear",
                align_corners=False, antialias=True,
            )
            total = total + self._single_scale(pred_scale, target_scale)
        return total


class TorchScriptIdentityLoss(nn.Module):
    """Optional frozen ArcFace-like model loaded from a local TorchScript file."""

    def __init__(self, model_path: Optional[str], image_size: int = 112):
        super().__init__()
        self.enabled = bool(model_path)
        self.image_size = image_size
        if self.enabled:
            self.model = torch.jit.load(model_path, map_location="cpu").eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False
        else:
            self.model = None

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"identity images must be NCHW, got shape {tuple(image.shape)}")
        image = F.interpolate(
            image, size=(self.image_size, self.image_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
        # Frozen TorchScript constants remain FP32 and do not participate in
        # eager autocast dispatch. Keep ArcFace in FP32 while preserving the
        # gradient back to a BF16 renderer output.
        with torch.autocast(device_type=image.device.type, enabled=False):
            result = self.model(image.float())
        if isinstance(result, (tuple, list)):
            result = result[0]
        return F.normalize(result.flatten(1).float(), dim=-1)

    def forward(self, prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return prediction.new_zeros(())
        if prediction.ndim == 5:
            batch, frames = prediction.shape[:2]
            prediction = prediction.flatten(0, 1)
            if reference.ndim == 4 and reference.shape[0] == batch:
                reference = reference[:, None].expand(-1, frames, -1, -1, -1).flatten(0, 1)
        if reference.ndim == 5:
            reference = reference.flatten(0, 1)
        if prediction.shape[0] != reference.shape[0]:
            raise ValueError(
                "identity prediction/reference batch mismatch: "
                f"{prediction.shape[0]} versus {reference.shape[0]}"
            )
        prediction_embedding = self.embed(prediction)
        with torch.no_grad():
            reference_embedding = self.embed(reference)
        return (1 - (prediction_embedding * reference_embedding).sum(dim=-1)).mean()


class FrozenFANMouthVelocityLoss(nn.Module):
    """Differentiable mouth dynamics from a frozen 68-point FAN teacher.

    Only one random adjacent pair is evaluated per clip. GT landmarks are
    detached, while soft-argmax landmarks on the reconstruction retain the
    gradient to the renderer. No face detector is used: tokenizer inputs are
    already aligned face crops.
    """

    def __init__(
        self,
        weights_path: Optional[str],
        package_path: Optional[str] = None,
        temperature: float = 20.0,
        confidence_threshold: float = 0.20,
        smooth_l1_beta: float = 0.01,
        use_checkpoint: bool = True,
        input_grad_max_norm: float = 0.02,
    ):
        super().__init__()
        self.enabled = bool(weights_path)
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.smooth_l1_beta = float(smooth_l1_beta)
        self.use_checkpoint = bool(use_checkpoint)
        self.input_grad_max_norm = float(input_grad_max_norm)
        self.register_buffer("last_input_grad_norm", torch.zeros(()), persistent=False)
        self.register_buffer("last_input_grad_clipped", torch.zeros(()), persistent=False)
        if not self.enabled:
            self.fan = None
            return
        if not os.path.isfile(str(weights_path)):
            raise FileNotFoundError(f"FAN weights not found: {weights_path}")
        if package_path:
            package_path = os.path.abspath(package_path)
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
        try:
            from face_alignment.models import FAN
        except ImportError as error:
            raise ImportError(
                "face_alignment is required for mouth velocity loss; "
                f"package_path={package_path!r}"
            ) from error
        self.fan = FAN(4)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.fan.load_state_dict(state, strict=True)
        self.fan.requires_grad_(False).eval()

    def train(self, mode: bool = True):
        super().train(False)
        if self.fan is not None:
            self.fan.eval()
        return self

    def _fan_forward(self, image: torch.Tensor) -> torch.Tensor:
        result = self.fan(image)
        return result[-1] if isinstance(result, (tuple, list)) else result

    def _landmarks(self, image: torch.Tensor, keep_input_gradient: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        image = F.interpolate(
            image.float(), size=(256, 256), mode="bilinear",
            align_corners=False, antialias=True,
        ).add(1).mul(0.5).clamp(0, 1)
        with torch.autocast(device_type=image.device.type, enabled=False):
            if keep_input_gradient and self.use_checkpoint:
                heatmaps = checkpoint(self._fan_forward, image, use_reentrant=False)
            else:
                heatmaps = self._fan_forward(image)
        heatmaps = heatmaps.float()
        flat = heatmaps.flatten(2)
        probability = F.softmax(flat * self.temperature, dim=-1)
        height, width = heatmaps.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=image.device),
            torch.linspace(0, 1, width, device=image.device), indexing="ij",
        )
        grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        landmarks = probability @ grid
        confidence = flat.amax(dim=-1)
        return landmarks, confidence

    def _clip_input_gradient(self, tensor: torch.Tensor) -> None:
        """Bound only the FAN-loss gradient entering each generated image."""
        if not tensor.requires_grad or self.input_grad_max_norm <= 0:
            return

        def hook(gradient: torch.Tensor) -> torch.Tensor:
            flat = gradient.float().flatten(1)
            norms = flat.norm(dim=1)
            limit = self.input_grad_max_norm
            scale = (limit / norms.clamp_min(1e-12)).clamp(max=1.0)
            self.last_input_grad_norm.copy_(norms.mean().detach())
            self.last_input_grad_clipped.copy_((norms > limit).float().mean().detach())
            shape = [gradient.shape[0]] + [1] * (gradient.ndim - 1)
            return gradient * scale.to(gradient).reshape(shape)

        tensor.register_hook(hook)

    @staticmethod
    def _face_normalization(landmarks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a detached, robust face coordinate system from GT landmarks."""
        left_eye = landmarks[:, 36:42].mean(dim=1)
        right_eye = landmarks[:, 42:48].mean(dim=1)
        center = (left_eye + right_eye) * 0.5
        # FAN coordinates are in [0,1]. A valid aligned face has an eye distance
        # around 0.2--0.35; 0.10 prevents a bad GT prediction from amplifying
        # the renderer gradient while still allowing normal scale variation.
        scale = (left_eye - right_eye).norm(dim=-1).clamp_min(0.10)
        return center.detach(), scale.detach()

    @staticmethod
    def _mouth_geometry(
        landmarks: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normalized = (landmarks - center[:, None]) / scale[:, None, None]
        mouth = normalized[:, 48:68]
        upper = normalized[:, [61, 62, 63]]
        lower = normalized[:, [67, 66, 65]]
        openness = (upper - lower).norm(dim=-1).mean(dim=-1)
        return mouth, openness

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        zero = prediction.new_zeros(())
        if not self.enabled or prediction.shape[1] < 2:
            return {
                "mouth_landmark_velocity": zero,
                "mouth_openness_velocity": zero,
                "mouth_landmark_confidence": zero,
                "mouth_landmark_valid": zero,
                "mouth_input_grad_norm": zero,
                "mouth_input_grad_clipped": zero,
            }
        start = int(torch.randint(prediction.shape[1] - 1, (), device=prediction.device).item())
        pred_pair = prediction[:, start:start + 2].flatten(0, 1)
        self._clip_input_gradient(pred_pair)
        target_pair = target[:, start:start + 2].flatten(0, 1)
        pred_landmarks, _ = self._landmarks(pred_pair, keep_input_gradient=True)
        with torch.no_grad():
            target_landmarks, target_confidence = self._landmarks(
                target_pair, keep_input_gradient=False,
            )
        # Crucially, prediction and GT share the coordinate system computed
        # from detached GT. Using predicted eye distance in the denominator can
        # approach zero on a blurry frame and caused 10^2--10^3 grad spikes.
        center, scale = self._face_normalization(target_landmarks)
        pred_mouth, pred_open = self._mouth_geometry(pred_landmarks, center, scale)
        target_mouth, target_open = self._mouth_geometry(target_landmarks, center, scale)
        batch = prediction.shape[0]
        pred_mouth = pred_mouth.reshape(batch, 2, 20, 2)
        target_mouth = target_mouth.reshape(batch, 2, 20, 2)
        pred_open = pred_open.reshape(batch, 2)
        target_open = target_open.reshape(batch, 2)

        target_confidence = target_confidence[:, 48:68].mean(dim=-1).reshape(batch, 2)
        confidence = target_confidence.amin(dim=1)
        valid = (confidence >= self.confidence_threshold).float()
        denominator = valid.sum().clamp_min(1.0)
        landmark = F.smooth_l1_loss(
            pred_mouth[:, 1] - pred_mouth[:, 0],
            target_mouth[:, 1] - target_mouth[:, 0],
            beta=self.smooth_l1_beta, reduction="none",
        ).mean(dim=(1, 2))
        openness = F.smooth_l1_loss(
            pred_open[:, 1] - pred_open[:, 0],
            target_open[:, 1] - target_open[:, 0],
            beta=self.smooth_l1_beta, reduction="none",
        )
        return {
            "mouth_landmark_velocity": (landmark * valid).sum() / denominator,
            "mouth_openness_velocity": (openness * valid).sum() / denominator,
            "mouth_landmark_confidence": confidence.mean().detach(),
            "mouth_landmark_valid": valid.mean().detach(),
            # These buffers are updated by the hook during backward, before
            # training scalars are logged.
            "mouth_input_grad_norm": self.last_input_grad_norm,
            "mouth_input_grad_clipped": self.last_input_grad_clipped,
        }


def _spatial_gradient(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x[..., 1:, :] - x[..., :-1, :], x[..., :, 1:] - x[..., :, :-1]


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_y, pred_x = _spatial_gradient(prediction)
    target_y, target_x = _spatial_gradient(target)
    return charbonnier(pred_y, target_y) + charbonnier(pred_x, target_x)


def region_weighted_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Extra weight for the central eye and lower-face areas of aligned crops."""
    height, width = prediction.shape[-2:]
    yy = torch.linspace(0, 1, height, device=prediction.device, dtype=prediction.dtype)
    xx = torch.linspace(0, 1, width, device=prediction.device, dtype=prediction.dtype)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")
    mouth = torch.exp(-(((xx - 0.50) / 0.20) ** 2 + ((yy - 0.69) / 0.13) ** 2) * 2.0)
    eyes = torch.exp(-(((xx - 0.50) / 0.30) ** 2 + ((yy - 0.39) / 0.10) ** 2) * 2.0)
    weight = (mouth + 0.5 * eyes)[None, None]
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(0)
    leading = prediction.numel() // (prediction.shape[-3] * height * width)
    denominator = weight.sum() * prediction.shape[-3] * leading
    return (weight * (prediction - target).abs()).sum() / (denominator + 1e-6)


def temporal_relation_loss(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape[1] < 2:
        zero = prediction.new_zeros(())
        return zero, zero
    pred_velocity = prediction[:, 1:] - prediction[:, :-1]
    target_velocity = target[:, 1:] - target[:, :-1]
    velocity = charbonnier(pred_velocity, target_velocity)
    if prediction.shape[1] < 3:
        return velocity, prediction.new_zeros(())
    pred_accel = pred_velocity[:, 1:] - pred_velocity[:, :-1]
    target_accel = target_velocity[:, 1:] - target_velocity[:, :-1]
    return velocity, charbonnier(pred_accel, target_accel)


def flow_total_variation(flow: torch.Tensor) -> torch.Tensor:
    dy, dx = _spatial_gradient(flow)
    return dy.abs().mean() + dx.abs().mean()


def covariance_loss(motion: torch.Tensor) -> torch.Tensor:
    flat = motion.reshape(-1, motion.shape[-1]).float()
    if flat.shape[0] < 2:
        return flat.new_zeros(())
    flat = flat - flat.mean(dim=0, keepdim=True)
    flat = flat / flat.std(dim=0, keepdim=True).clamp_min(1e-4)
    covariance = flat.T @ flat / max(flat.shape[0] - 1, 1)
    eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=torch.bool)
    return covariance.masked_fill(eye, 0).square().mean()


def motion_moment_loss(motion: torch.Tensor, target_std: float = 0.20) -> torch.Tensor:
    """Anchor raw motion deltas before corpus normalization.

    Renderer weights can compensate for arbitrary delta scale. Per-dimension
    centering and scale targets keep the representation stationary until the
    late-window MotionNormalizer freezes.
    """
    if target_std <= 0:
        raise ValueError(f"target_std must be positive, got {target_std}")
    flat = motion.reshape(-1, motion.shape[-1]).float()
    if flat.shape[0] < 2:
        return flat.new_zeros(())
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False)
    scale = float(target_std)
    center_loss = (mean / scale).square().mean()
    scale_loss = (std / scale - 1.0).square().mean()
    return center_loss + scale_loss


def color_statistics_loss(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    prediction = F.avg_pool2d(prediction, 16)
    reference = F.avg_pool2d(reference, 16)
    pred_mean = prediction.mean(dim=(-2, -1))
    ref_mean = reference.mean(dim=(-2, -1))
    pred_std = prediction.std(dim=(-2, -1))
    ref_std = reference.std(dim=(-2, -1))
    return F.l1_loss(pred_mean, ref_mean) + F.l1_loss(pred_std, ref_std)


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        base_channels: int = 64,
        max_channels: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("PatchDiscriminator num_layers must be at least 2")
        channels = [min(base_channels * (2 ** index), max_channels)
                    for index in range(num_layers)]
        layers: List[nn.Module] = []
        in_ch = 3
        for index, out_ch in enumerate(channels):
            layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, out_ch, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)
        self.head = spectral_norm(nn.Conv2d(in_ch, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        logits = self.head(x)
        return (logits, features) if return_features else logits


class MultiScaleImageDiscriminator(nn.Module):
    def __init__(
        self,
        scales: int = 2,
        base_channels: int = 64,
        max_channels: int = 512,
        num_layers: int = 4,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList(
            PatchDiscriminator(base_channels, max_channels, num_layers)
            for _ in range(scales)
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x, return_features))
            x = F.avg_pool2d(x, 2)
        return outputs


class LIAImageDiscriminator(nn.Module):
    """Full-image StyleGAN discriminator used by the released LIA trainer."""

    def __init__(self, image_size: int = 512, channel_multiplier: int = 1):
        super().__init__()
        image_size = int(image_size)
        if image_size < 8 or image_size & (image_size - 1):
            raise ValueError("image_size must be a power of two and at least 8")
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        if image_size not in channels:
            raise ValueError(f"unsupported discriminator image_size: {image_size}")
        layers: List[nn.Module] = [ConvLayer(3, channels[image_size], 1)]
        in_channel = channels[image_size]
        for exponent in range(int(math.log2(image_size)), 2, -1):
            out_channel = channels[2 ** (exponent - 1)]
            layers.append(_LIAResidualDownBlock(in_channel, out_channel))
            in_channel = out_channel
        self.convs = nn.Sequential(*layers)
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, image: torch.Tensor):
        out = self.convs(image)
        batch, channels, height, width = out.shape
        group = min(batch, 4)
        while batch % group != 0:
            group -= 1
        stddev = out.view(group, -1, 1, channels, height, width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean((2, 3, 4), keepdim=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat((out, stddev), dim=1)
        out = self.final_conv(out).reshape(batch, -1)
        return [self.final_linear(out)]


class _LIAResidualDownBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True,
            activate=False, bias=False,
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return (self.conv2(self.conv1(tensor)) + self.skip(tensor)) / math.sqrt(2)


class VideoDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32, max_channels: int = 256):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, max_channels]
        layers: List[nn.Module] = []
        in_ch = 3
        for index, out_ch in enumerate(channels):
            temporal_stride = 1 if index == 0 else 2
            layers.append(nn.Sequential(
                spectral_norm(nn.Conv3d(
                    in_ch, out_ch, kernel_size=(3, 4, 4),
                    stride=(temporal_stride, 2, 2), padding=(1, 1, 1),
                )),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)
        self.head = spectral_norm(nn.Conv3d(in_ch, 1, 3, 1, 1))

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = video.permute(0, 2, 1, 3, 4)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def discriminator_hinge(real_logits: Iterable[torch.Tensor], fake_logits: Iterable[torch.Tensor]) -> torch.Tensor:
    losses = []
    for real, fake in zip(real_logits, fake_logits):
        if isinstance(real, tuple):
            real = real[0]
        if isinstance(fake, tuple):
            fake = fake[0]
        losses.append(F.relu(1 - real).mean() + F.relu(1 + fake).mean())
    return sum(losses) / max(len(losses), 1)


def generator_hinge(fake_logits: Iterable[torch.Tensor]) -> torch.Tensor:
    logits = [value[0] if isinstance(value, tuple) else value for value in fake_logits]
    return -sum(value.mean() for value in logits) / max(len(logits), 1)


def discriminator_logistic(
    real_logits: Iterable[torch.Tensor], fake_logits: Iterable[torch.Tensor]
) -> torch.Tensor:
    """Non-saturating logistic discriminator objective used by StyleGAN/LIA."""
    losses = []
    for real, fake in zip(real_logits, fake_logits):
        if isinstance(real, tuple):
            real = real[0]
        if isinstance(fake, tuple):
            fake = fake[0]
        losses.append(F.softplus(-real).mean() + F.softplus(fake).mean())
    return sum(losses) / max(len(losses), 1)


def generator_nonsaturating(fake_logits: Iterable[torch.Tensor]) -> torch.Tensor:
    """Non-saturating logistic generator objective."""
    logits = [value[0] if isinstance(value, tuple) else value for value in fake_logits]
    return sum(F.softplus(-value).mean() for value in logits) / max(len(logits), 1)


def feature_matching_loss(real_outputs, fake_outputs) -> torch.Tensor:
    total = fake_outputs[0][0].new_zeros(())
    count = 0
    for (_, real_features), (_, fake_features) in zip(real_outputs, fake_outputs):
        for real, fake in zip(real_features, fake_features):
            total = total + F.l1_loss(fake, real.detach())
            count += 1
    return total / max(count, 1)


def set_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
    if module is None:
        return
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)
