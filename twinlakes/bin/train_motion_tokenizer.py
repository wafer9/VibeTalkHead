#!/usr/bin/env python
"""Train the strict two-branch talking-head motion tokenizer.

Single GPU:
  python -m twinlakes.bin.train_motion_tokenizer --config conf/motion_tokenizer.yaml

Distributed:
  torchrun --nproc_per_node=8 -m twinlakes.bin.train_motion_tokenizer \
      --config conf/motion_tokenizer.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from twinlakes.motion_tokenizer.data import (
    JsonlVideoClipDataset,
    build_jsonl_index,
    seed_worker,
)
from twinlakes.motion_tokenizer.losses import (
    FrozenFANMouthVelocityLoss,
    LaplacianPyramidLoss,
    LocalVGGPerceptualLoss,
    MultiScaleImageDiscriminator,
    TorchScriptIdentityLoss,
    VideoDiscriminator,
    charbonnier,
    color_statistics_loss,
    covariance_loss,
    discriminator_hinge,
    feature_matching_loss,
    flow_total_variation,
    generator_hinge,
    gradient_loss,
    motion_moment_loss,
    region_weighted_loss,
    set_requires_grad,
    temporal_relation_loss,
)
from twinlakes.motion_tokenizer.model import MotionTokenizer


LOG = logging.getLogger("motion_tokenizer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume", default=None, help="checkpoint path, 'latest', or empty")
    parser.add_argument("--max_steps", type=int, default=None, help="override for smoke/short runs")
    parser.add_argument("--train_manifest", default=None)
    parser.add_argument("--val_manifest", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None, help="override per-rank batch size")
    parser.add_argument("--render_chunk", type=int, default=None, help="override renderer micro-chunk")
    parser.add_argument("--log_interval", type=int, default=None, help="override logging interval")
    parser.add_argument(
        "--skip_final_save", action="store_true",
        help="skip the terminal checkpoint (intended for smoke tests only)",
    )
    return parser.parse_args()


def distributed_info() -> Tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def setup_distributed() -> Tuple[int, int, int, torch.device]:
    rank, local_rank, world_size = distributed_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl":
            dist.init_process_group(backend, device_id=device)
        else:
            dist.init_process_group(backend)
    return rank, local_rank, world_size, device


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def seed_everything(seed: int, rank: int) -> torch.Generator:
    seed = seed + rank * 100003
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def raw_module(module):
    return module.module if isinstance(module, DDP) else module


def model_parameter_groups(model: MotionTokenizer, config: Dict) -> list:
    motion_ids = {id(parameter) for parameter in model.motion_encoder.parameters()}
    motion = [parameter for parameter in model.parameters() if id(parameter) in motion_ids]
    main = [parameter for parameter in model.parameters() if id(parameter) not in motion_ids]
    lr = float(config["lr"])
    return [
        {"params": main, "lr": lr, "initial_lr": lr, "name": "main"},
        {
            "params": motion,
            "lr": lr * float(config.get("motion_lr_multiplier", 1.0)),
            "initial_lr": lr * float(config.get("motion_lr_multiplier", 1.0)),
            "name": "motion",
        },
    ]


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(step + 1, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def set_learning_rates(optimizer, step: int, training: Dict, normalizer_ready: bool) -> None:
    scale = cosine_lr(step, int(training["max_steps"]), int(training.get("warmup_steps", 0)))
    min_scale = float(training.get("min_lr_ratio", 0.05))
    scale = min_scale + (1 - min_scale) * scale
    freeze_motion = bool(training.get("freeze_motion_after_normalizer", True)) and normalizer_ready
    for group in optimizer.param_groups:
        if group.get("name") == "motion" and freeze_motion:
            group["lr"] = 0.0
        else:
            group["lr"] = group["initial_lr"] * scale


def stage_strength(step: int, start: int, ramp: int = 0) -> float:
    if start < 0 or step < start:
        return 0.0
    if ramp <= 0:
        return 1.0
    return min((step - start + 1) / ramp, 1.0)


def autocast_context(device: torch.device, dtype: str):
    if device.type != "cuda" or dtype == "fp32":
        return contextlib.nullcontext()
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def flatten_selected_frames(video: torch.Tensor, maximum: int) -> torch.Tensor:
    flat = video.flatten(0, 1)
    if maximum > 0 and flat.shape[0] > maximum:
        indices = torch.randperm(flat.shape[0], device=flat.device)[:maximum]
        flat = flat[indices]
    return flat


def compute_generator_losses(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    reference: torch.Tensor,
    loss_config: Dict,
    laplacian: LaplacianPyramidLoss,
    perceptual: LocalVGGPerceptualLoss,
    identity: TorchScriptIdentityLoss,
    mouth_velocity: FrozenFANMouthVelocityLoss,
    image_discriminator: Optional[torch.nn.Module],
    video_discriminator: Optional[torch.nn.Module],
    image_gan_active: bool,
    video_gan_active: bool,
) -> Dict[str, torch.Tensor]:
    reconstruction = output["reconstruction"]
    losses: Dict[str, torch.Tensor] = {}
    zero = reconstruction.new_zeros(())
    losses["reconstruction"] = charbonnier(reconstruction, target)
    losses["laplacian"] = (
        laplacian(reconstruction, target)
        if float(loss_config.get("laplacian", 0.0)) != 0 else zero
    )
    losses["gradient"] = (
        gradient_loss(reconstruction, target)
        if float(loss_config.get("gradient", 0.0)) != 0 else zero
    )
    losses["region"] = (
        region_weighted_loss(reconstruction, target)
        if float(loss_config.get("region", 0.0)) != 0 else zero
    )
    if (
        float(loss_config.get("temporal_velocity", 0.0)) != 0
        or float(loss_config.get("temporal_acceleration", 0.0)) != 0
    ):
        velocity, acceleration = temporal_relation_loss(reconstruction, target)
        losses["temporal_velocity"] = velocity
        losses["temporal_acceleration"] = acceleration
    else:
        losses["temporal_velocity"] = zero
        losses["temporal_acceleration"] = zero
    if mouth_velocity.enabled and (
        float(loss_config.get("mouth_landmark_velocity", 0.0)) != 0
        or float(loss_config.get("mouth_openness_velocity", 0.0)) != 0
    ):
        losses.update(mouth_velocity(reconstruction, target))
    else:
        losses.update({
            "mouth_landmark_velocity": zero,
            "mouth_openness_velocity": zero,
            "mouth_landmark_confidence": zero,
            "mouth_landmark_valid": zero,
            "mouth_input_grad_norm": zero,
            "mouth_input_grad_clipped": zero,
        })
    losses["flow_tv"] = (
        flow_total_variation(output["flow"])
        if float(loss_config.get("flow_tv", 0.0)) != 0 else zero
    )
    losses["covariance"] = (
        covariance_loss(output["target_motion"])
        if float(loss_config.get("covariance", 0.0)) != 0 else zero
    )
    if float(loss_config.get("motion_moment", 0.0)) != 0:
        motion_samples = torch.cat([
            output["reference_motion"][:, None], output["target_motion"]
        ], dim=1)
        losses["motion_moment"] = motion_moment_loss(
            motion_samples, float(loss_config.get("motion_target_std", 0.20))
        )
    else:
        losses["motion_moment"] = zero
    perceptual_weight = float(loss_config.get("perceptual", 0.0))
    if perceptual.enabled and perceptual_weight != 0:
        prediction_frames = reconstruction.flatten(0, 1)
        target_frames = target.flatten(0, 1)
        maximum = int(loss_config.get("max_perceptual_frames", 4))
        if maximum > 0 and prediction_frames.shape[0] > maximum:
            indices = torch.randperm(prediction_frames.shape[0], device=prediction_frames.device)[:maximum]
            prediction_frames = prediction_frames[indices]
            target_frames = target_frames[indices]
        losses["perceptual"] = perceptual(prediction_frames, target_frames)
    else:
        losses["perceptual"] = reconstruction.new_zeros(())

    # Same-identity reconstruction identity; evaluate only one frame per clip.
    if identity.enabled and float(loss_config.get("identity", 0.0)) != 0:
        losses["identity"] = identity(
            reconstruction[:, reconstruction.shape[1] // 2], reference
        )
    else:
        losses["identity"] = reconstruction.new_zeros(())

    if "noisy_reconstruction" in output:
        noisy = output["noisy_reconstruction"]
        losses["noisy_reconstruction"] = charbonnier(noisy, target)
        losses["noise_consistency"] = charbonnier(noisy, reconstruction.detach())
        noisy_velocity, _ = temporal_relation_loss(noisy, target)
        losses["noisy_temporal"] = noisy_velocity
    else:
        zero = reconstruction.new_zeros(())
        losses.update({
            "noisy_reconstruction": zero,
            "noise_consistency": zero,
            "noisy_temporal": zero,
        })

    if "cross_reconstruction" in output:
        losses["cross_motion_cycle"] = F.smooth_l1_loss(
            output["cross_cycle_delta"], output["cross_target_delta"].detach()
        )
        losses["cross_appearance"] = color_statistics_loss(
            output["cross_reconstruction"], output["cross_reference"]
        )
        if identity.enabled and float(loss_config.get("cross_identity", 0.0)) != 0:
            losses["cross_identity"] = identity(
                output["cross_reconstruction"], output["cross_reference"]
            )
        else:
            losses["cross_identity"] = reconstruction.new_zeros(())
    else:
        zero = reconstruction.new_zeros(())
        losses.update({
            "cross_motion_cycle": zero,
            "cross_appearance": zero,
            "cross_identity": zero,
        })

    if image_gan_active and image_discriminator is not None:
        maximum = int(loss_config.get("max_gan_frames", 8))
        fake = flatten_selected_frames(reconstruction, maximum)
        real = flatten_selected_frames(target, maximum)
        with torch.no_grad():
            real_outputs = image_discriminator(real, return_features=True)
        fake_outputs = image_discriminator(fake, return_features=True)
        losses["image_adversarial"] = generator_hinge(fake_outputs)
        losses["feature_matching"] = feature_matching_loss(real_outputs, fake_outputs)
    else:
        zero = reconstruction.new_zeros(())
        losses["image_adversarial"] = zero
        losses["feature_matching"] = zero

    if video_gan_active and video_discriminator is not None:
        spatial = int(loss_config.get("video_gan_size", 128))
        fake_video = F.interpolate(
            reconstruction.flatten(0, 1), size=(spatial, spatial),
            mode="bilinear", align_corners=False, antialias=True,
        ).reshape(*reconstruction.shape[:3], spatial, spatial)
        losses["video_adversarial"] = -video_discriminator(fake_video).mean()
    else:
        losses["video_adversarial"] = reconstruction.new_zeros(())

    total = reconstruction.new_zeros(())
    for name, value in losses.items():
        total = total + float(loss_config.get(name, 0.0)) * value
    losses["total"] = total
    return losses


def compute_discriminator_losses(
    output: Dict[str, torch.Tensor],
    target: torch.Tensor,
    loss_config: Dict,
    image_discriminator: Optional[torch.nn.Module],
    video_discriminator: Optional[torch.nn.Module],
    image_active: bool,
    video_active: bool,
) -> Dict[str, torch.Tensor]:
    reconstruction = output["reconstruction"].detach()
    result = {"image_discriminator": reconstruction.new_zeros(()),
              "video_discriminator": reconstruction.new_zeros(())}
    if image_active and image_discriminator is not None:
        maximum = int(loss_config.get("max_gan_frames", 8))
        fake = flatten_selected_frames(reconstruction, maximum)
        real = flatten_selected_frames(target, maximum)
        result["image_discriminator"] = discriminator_hinge(
            image_discriminator(real), image_discriminator(fake)
        )
    if video_active and video_discriminator is not None:
        spatial = int(loss_config.get("video_gan_size", 128))
        fake = F.interpolate(
            reconstruction.flatten(0, 1), size=(spatial, spatial),
            mode="bilinear", align_corners=False, antialias=True,
        ).reshape(*reconstruction.shape[:3], spatial, spatial)
        real = F.interpolate(
            target.flatten(0, 1), size=(spatial, spatial),
            mode="bilinear", align_corners=False, antialias=True,
        ).reshape(*target.shape[:3], spatial, spatial)
        real_logits = video_discriminator(real)
        fake_logits = video_discriminator(fake)
        result["video_discriminator"] = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
    result["total"] = result["image_discriminator"] + result["video_discriminator"]
    return result


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
    config: Dict,
    causal_strength: float,
) -> Dict[str, float]:
    model.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    limit = int(config["training"].get("validation_batches", 8))
    for batch_index, batch in enumerate(loader):
        if batch_index >= limit:
            break
        reference = batch["reference"].to(device, non_blocking=True)
        target = batch["frames"].to(device, non_blocking=True)
        with autocast_context(device, config["training"].get("dtype", "bf16")):
            output = model(
                reference, target, causal_strength=causal_strength,
                render_chunk=int(config["training"].get("render_chunk", 4)),
            )
        prediction = output["reconstruction"].float()
        target = target.float()
        l1 = (prediction - target).abs().mean()
        mse = (prediction - target).square().mean()
        velocity, acceleration = temporal_relation_loss(prediction, target)
        totals += torch.tensor([l1, mse, velocity, acceleration, 1.0], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    count = totals[-1].clamp_min(1)
    mse = float((totals[1] / count).item())
    model.train()
    return {
        "l1": float((totals[0] / count).item()),
        "psnr": -10.0 * math.log10(max(mse / 4.0, 1e-12)),  # images are in [-1,1]
        "velocity": float((totals[2] / count).item()),
        "acceleration": float((totals[3] / count).item()),
    }


def latest_checkpoint(output_dir: str) -> Optional[str]:
    checkpoints = []
    for path in Path(output_dir).glob("step_*.pt"):
        match = re.search(r"step_(\d+)\.pt$", path.name)
        if match:
            checkpoints.append((int(match.group(1)), str(path)))
    return max(checkpoints)[1] if checkpoints else None


def save_checkpoint(
    path: str,
    model,
    image_discriminator,
    video_discriminator,
    generator_optimizer,
    discriminator_optimizer,
    scaler,
    step: int,
    epoch: int,
    config: Dict,
) -> None:
    payload = {
        "model": raw_module(model).state_dict(),
        "image_discriminator": raw_module(image_discriminator).state_dict() if image_discriminator else None,
        "video_discriminator": raw_module(video_discriminator).state_dict() if video_discriminator else None,
        "generator_optimizer": generator_optimizer.state_dict(),
        "discriminator_optimizer": discriminator_optimizer.state_dict() if discriminator_optimizer else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "config": config,
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: str,
    model,
    image_discriminator,
    video_discriminator,
    generator_optimizer,
    discriminator_optimizer,
    scaler,
) -> Tuple[int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_module(model).load_state_dict(payload["model"], strict=True)
    if image_discriminator is not None and payload.get("image_discriminator") is not None:
        raw_module(image_discriminator).load_state_dict(payload["image_discriminator"])
    if video_discriminator is not None and payload.get("video_discriminator") is not None:
        raw_module(video_discriminator).load_state_dict(payload["video_discriminator"])
    generator_optimizer.load_state_dict(payload["generator_optimizer"])
    if discriminator_optimizer is not None and payload.get("discriminator_optimizer") is not None:
        discriminator_optimizer.load_state_dict(payload["discriminator_optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("step", 0)), int(payload.get("epoch", 0))


def save_preview(path: str, reference: torch.Tensor, target: torch.Tensor,
                 prediction: torch.Tensor, maximum_frames: int = 6) -> None:
    count = min(target.shape[1], maximum_frames)
    indices = torch.linspace(0, target.shape[1] - 1, count, device=target.device).long()
    images = [reference[0]]
    for index in indices:
        images.extend([target[0, index], prediction[0, index]])
    grid = make_grid(torch.stack(images).add(1).mul(0.5).clamp(0, 1), nrow=1 + 2 * count)
    save_image(grid, path)


def main() -> None:
    args = parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.train_manifest:
        config["data"]["train_manifest"] = args.train_manifest
    if args.val_manifest:
        config["data"]["val_manifest"] = args.val_manifest
    if args.num_workers is not None:
        config["data"]["num_workers"] = args.num_workers
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.render_chunk is not None:
        config["training"]["render_chunk"] = args.render_chunk
    if args.log_interval is not None:
        config["training"]["log_interval"] = args.log_interval
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps

    rank, local_rank, world_size, device = setup_distributed()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    generator = seed_everything(int(config.get("seed", 2026)), rank)
    output_dir = os.path.abspath(config["output_dir"])
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train.yaml"), "w") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)

    # The 609 MB manifest is indexed once on rank zero, then mmap'ed by all workers.
    if rank == 0:
        for key in ("train_manifest", "val_manifest"):
            build_jsonl_index(config["data"][key])
    barrier()

    cross_enabled = int(config["stages"].get("cross_identity_start", -1)) >= 0
    common_data = dict(
        video_root=config["data"]["video_root"],
        clip_length=int(config["data"]["clip_length"]),
        image_size=int(config["data"]["image_size"]),
        target_fps=float(config["data"].get("target_fps", 25)),
    )
    train_dataset = JsonlVideoClipDataset(
        config["data"]["train_manifest"], training=True,
        first_frame_reference_prob=float(config["data"].get("first_frame_reference_prob", 0.5)),
        photometric_strength=float(config["data"].get("photometric_strength", 0.12)),
        horizontal_flip_prob=float(config["data"].get("horizontal_flip_prob", 0.5)),
        # Cross references are formed from another local sample or another DDP
        # rank below, avoiding a second VideoReader open for every training item.
        return_cross_reference=False,
        max_retries=int(config["data"].get("max_retries", 12)),
        max_samples=int(config["data"].get("max_train_samples", 0)),
        **common_data,
    )
    val_dataset = JsonlVideoClipDataset(
        config["data"]["val_manifest"], training=False,
        return_cross_reference=False, photometric_strength=0,
        horizontal_flip_prob=0, max_retries=int(config["data"].get("max_retries", 12)),
        max_samples=int(config["data"].get("max_val_samples", 0)),
        **common_data,
    )
    train_sampler = DistributedSampler(
        train_dataset, world_size, rank, shuffle=True, seed=int(config.get("seed", 2026))
    ) if world_size > 1 else None
    val_sampler = DistributedSampler(
        val_dataset, world_size, rank, shuffle=False
    ) if world_size > 1 else None
    loader_kwargs = dict(
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=bool(config["data"].get("pin_memory", True)),
        persistent_workers=int(config["data"].get("num_workers", 4)) > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, shuffle=train_sampler is None,
        drop_last=True, **loader_kwargs,
    )
    val_loader_kwargs = copy.copy(loader_kwargs)
    val_loader_kwargs["batch_size"] = int(config["data"].get("val_batch_size", 1))
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, shuffle=False, drop_last=False,
        **val_loader_kwargs,
    )

    model = MotionTokenizer(**config["model"]).to(device)
    image_discriminator = MultiScaleImageDiscriminator(
        **config.get("image_discriminator", {})
    ).to(device) if int(config["stages"].get("image_gan_start", -1)) >= 0 else None
    video_discriminator = VideoDiscriminator(
        **config.get("video_discriminator", {})
    ).to(device) if int(config["stages"].get("video_gan_start", -1)) >= 0 else None

    training = config["training"]
    if bool(training.get("freeze_motion_encoder", False)):
        model.motion_encoder.requires_grad_(False)
    generator_optimizer = torch.optim.AdamW(
        model_parameter_groups(model, training),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    discriminator_parameters = []
    if image_discriminator is not None:
        discriminator_parameters.extend(image_discriminator.parameters())
    if video_discriminator is not None:
        discriminator_parameters.extend(video_discriminator.parameters())
    discriminator_optimizer = torch.optim.AdamW(
        discriminator_parameters,
        lr=float(training.get("discriminator_lr", training["lr"])),
        betas=tuple(training.get("discriminator_betas", [0.0, 0.99])),
    ) if discriminator_parameters else None

    dtype = training.get("dtype", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == "fp16")
    loss_config = config["loss"]
    laplacian = LaplacianPyramidLoss(int(loss_config.get("laplacian_levels", 4))).to(device)
    perceptual_enabled = float(loss_config.get("perceptual", 0.0)) != 0
    identity_enabled = (
        float(loss_config.get("identity", 0.0)) != 0
        or float(loss_config.get("cross_identity", 0.0)) != 0
    )
    mouth_enabled = (
        float(loss_config.get("mouth_landmark_velocity", 0.0)) != 0
        or float(loss_config.get("mouth_openness_velocity", 0.0)) != 0
    )
    perceptual = LocalVGGPerceptualLoss(
        loss_config.get("vgg_weights_path") if perceptual_enabled else None
    ).to(device)
    identity = TorchScriptIdentityLoss(
        loss_config.get("identity_model_path") if identity_enabled else None
    ).to(device)
    mouth_velocity = FrozenFANMouthVelocityLoss(
        loss_config.get("fan_weights_path") if mouth_enabled else None,
        package_path=loss_config.get("face_alignment_path"),
        temperature=float(loss_config.get("fan_softargmax_temperature", 20.0)),
        confidence_threshold=float(loss_config.get("fan_confidence_threshold", 0.20)),
        smooth_l1_beta=float(loss_config.get("mouth_velocity_beta", 0.01)),
        use_checkpoint=bool(loss_config.get("fan_gradient_checkpoint", True)),
        input_grad_max_norm=float(loss_config.get("fan_input_grad_max_norm", 0.02)),
    ).to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=bool(training.get("find_unused_parameters", True)),
        )
        if image_discriminator is not None:
            image_discriminator = DDP(
                image_discriminator, device_ids=[local_rank] if device.type == "cuda" else None
            )
        if video_discriminator is not None:
            video_discriminator = DDP(
                video_discriminator, device_ids=[local_rank] if device.type == "cuda" else None
            )

    step, start_epoch = 0, 0
    resume = args.resume or config.get("resume")
    if resume == "latest":
        resume = latest_checkpoint(output_dir)
    if resume:
        step, start_epoch = load_checkpoint(
            resume, model, image_discriminator, video_discriminator,
            generator_optimizer, discriminator_optimizer, scaler,
        )
        # A 512 fine-tune deliberately uses a new LR even though optimizer moments
        # are restored from the 256 run.
        base_lr = float(training["lr"])
        for group in generator_optimizer.param_groups:
            multiplier = float(training.get("motion_lr_multiplier", 1.0)) if group.get("name") == "motion" else 1.0
            group["initial_lr"] = base_lr * multiplier
            group["lr"] = group["initial_lr"]
        if discriminator_optimizer is not None:
            for group in discriminator_optimizer.param_groups:
                group["lr"] = float(training.get("discriminator_lr", base_lr))
        LOG.info("resumed %s at step %d", resume, step)

    writer = SummaryWriter(output_dir) if rank == 0 else None
    if rank == 0:
        parameters = sum(parameter.numel() for parameter in raw_module(model).parameters())
        LOG.info("device=%s world_size=%d train=%d val=%d parameters=%.2fM",
                 device, world_size, len(train_dataset), len(val_dataset), parameters / 1e6)
        if not perceptual.enabled:
            LOG.warning("VGG weights are not configured; using Laplacian/gradient losses only")
        if not identity.enabled and cross_enabled:
            LOG.warning("identity_model_path is empty; cross-ID uses motion-cycle + color statistics only")
        if bool(training.get("freeze_motion_encoder", False)):
            LOG.info("motion encoder is explicitly frozen for this fine-tune")
        if cross_enabled and world_size == 1 and int(config["data"]["batch_size"]) < 2:
            LOG.warning("cross-ID is inactive with one GPU and batch_size=1; use batch_size>=2 or DDP")

    accum_steps = int(training.get("accumulation_steps", 1))
    max_steps = int(training["max_steps"])
    normalizer_step = int(config["stages"].get("normalizer_freeze_step", -1))
    normalizer_start = int(config["stages"].get("normalizer_start_step", 0))
    causal_start = int(config["stages"].get("causal_start", -1))
    cross_start = int(config["stages"].get("cross_identity_start", -1))
    image_gan_start = int(config["stages"].get("image_gan_start", -1))
    video_gan_start = int(config["stages"].get("video_gan_start", -1))
    noise_start = int(config["stages"].get("noise_start", -1))
    render_chunk = int(training.get("render_chunk", 4))
    log_interval = int(training.get("log_interval", 20))
    eval_interval = int(training.get("eval_interval", 2000))
    save_interval = int(training.get("save_interval", 2000))
    preview_interval = int(training.get("preview_interval", 500))
    last_time = time.time()

    model.train()
    generator_optimizer.zero_grad(set_to_none=True)
    if discriminator_optimizer is not None:
        discriminator_optimizer.zero_grad(set_to_none=True)

    epoch = start_epoch
    while step < max_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        for micro_index, batch in enumerate(train_loader):
            if step >= max_steps:
                break
            raw = raw_module(model)
            if normalizer_step >= 0 and step >= normalizer_step and not bool(raw.normalizer.ready.item()):
                raw.normalizer.finalize()
                barrier()
                if rank == 0:
                    LOG.info("froze motion normalization at step %d: std=[%.4f, %.4f]",
                             step, raw.normalizer.std.min().item(), raw.normalizer.std.max().item())

            set_learning_rates(generator_optimizer, step, training, bool(raw.normalizer.ready.item()))
            causal_strength = stage_strength(
                step, causal_start, int(config["stages"].get("causal_ramp_steps", 10000))
            )
            cross_active = cross_start >= 0 and step >= cross_start
            image_gan_active = image_gan_start >= 0 and step >= image_gan_start
            video_gan_active = video_gan_start >= 0 and step >= video_gan_start
            noise_active = noise_start >= 0 and step >= noise_start
            noise_strength = stage_strength(
                step, noise_start, int(config["stages"].get("noise_ramp_steps", 10000))
            ) * float(config["stages"].get("max_noise_sigma", 0.10))

            reference = batch["reference"].to(device, non_blocking=True)
            target = batch["frames"].to(device, non_blocking=True)
            cross_reference = None
            if cross_active and reference.shape[0] > 1:
                cross_reference = reference.roll(1, dims=0)
            elif cross_active and world_size > 1:
                gathered = [torch.empty_like(reference) for _ in range(world_size)]
                dist.all_gather(gathered, reference.detach())
                cross_reference = gathered[(rank + 1) % world_size]

            set_requires_grad(image_discriminator, False)
            set_requires_grad(video_discriminator, False)
            with autocast_context(device, dtype):
                output = model(
                    reference, target, cross_reference=cross_reference,
                    causal_strength=causal_strength,
                    noise_sigma=noise_strength if noise_active else 0.0,
                    noise_mode=config["stages"].get("noise_mode", "mixed"),
                    render_chunk=render_chunk,
                )
                losses = compute_generator_losses(
                    output, target, reference, config["loss"], laplacian,
                    perceptual, identity, mouth_velocity,
                    image_discriminator, video_discriminator,
                    image_gan_active, video_gan_active,
                )
                generator_loss = losses["total"] / accum_steps
            scaler.scale(generator_loss).backward()

            # Corpus stats use only the configured late-training window, so
            # early encoder drift does not contaminate the exported scale.
            if step >= normalizer_start:
                raw.normalizer.update(torch.cat([
                    output["reference_motion"].detach()[:, None],
                    output["target_motion"].detach(),
                ], dim=1))

            discriminator_losses = {"total": target.new_zeros(())}
            if discriminator_optimizer is not None and (image_gan_active or video_gan_active):
                set_requires_grad(image_discriminator, image_gan_active)
                set_requires_grad(video_discriminator, video_gan_active)
                with autocast_context(device, dtype):
                    discriminator_losses = compute_discriminator_losses(
                        output, target, config["loss"], image_discriminator,
                        video_discriminator, image_gan_active, video_gan_active,
                    )
                    discriminator_loss = discriminator_losses["total"] / accum_steps
                scaler.scale(discriminator_loss).backward()

            accumulation_boundary = (micro_index + 1) % accum_steps == 0
            if not accumulation_boundary:
                continue

            scaler.unscale_(generator_optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("grad_clip", 1.0))
            )
            scaler.step(generator_optimizer)
            if discriminator_optimizer is not None and (image_gan_active or video_gan_active):
                scaler.unscale_(discriminator_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    discriminator_parameters, float(training.get("discriminator_grad_clip", 5.0))
                )
                scaler.step(discriminator_optimizer)
            scaler.update()
            generator_optimizer.zero_grad(set_to_none=True)
            if discriminator_optimizer is not None:
                discriminator_optimizer.zero_grad(set_to_none=True)
            step += 1

            if rank == 0 and step % log_interval == 0:
                elapsed = max(time.time() - last_time, 1e-6)
                last_time = time.time()
                scalars = {name: float(value.detach().float().item()) for name, value in losses.items()}
                scalars.update({
                    f"d_{name}": float(value.detach().float().item())
                    for name, value in discriminator_losses.items()
                })
                with torch.no_grad():
                    logged_motion = torch.cat([
                        output["reference_motion"].detach()[:, None],
                        output["target_motion"].detach(),
                    ], dim=1).flatten(0, 1).float()
                    motion_mean = logged_motion.mean(dim=0)
                    motion_std = logged_motion.std(dim=0, unbiased=False)
                    scalars.update({
                        "motion_raw_mean_rms": float(motion_mean.square().mean().sqrt().item()),
                        "motion_raw_std_mean": float(motion_std.mean().item()),
                        "motion_raw_std_max": float(motion_std.max().item()),
                    })
                LOG.info(
                    "step=%d loss=%.4f rec=%.4f vel=%.4f mvel=%.4f ovel=%.4f "
                    "mconf=%.3f mvalid=%.2f mgn=%.3e mgclip=%.2f noise=%.3f causal=%.2f "
                    "mstd=%.3f grad=%.3f lr=%.2e mem=%.2fGB steps/s=%.3f",
                    step, scalars["total"], scalars["reconstruction"],
                    scalars["temporal_velocity"], scalars["mouth_landmark_velocity"],
                    scalars["mouth_openness_velocity"], scalars["mouth_landmark_confidence"],
                    scalars["mouth_landmark_valid"], scalars["mouth_input_grad_norm"],
                    scalars["mouth_input_grad_clipped"], noise_strength, causal_strength,
                    scalars["motion_raw_std_mean"],
                    float(grad_norm), generator_optimizer.param_groups[0]["lr"],
                    torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0,
                    log_interval / elapsed,
                )
                for name, value in scalars.items():
                    writer.add_scalar(f"train/{name}", value, step)
                writer.add_scalar("train/causal_strength", causal_strength, step)
                writer.add_scalar("train/noise_sigma", noise_strength, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                for group in generator_optimizer.param_groups:
                    writer.add_scalar(f"lr/{group['name']}", group["lr"], step)

            if rank == 0 and preview_interval > 0 and step % preview_interval == 0:
                save_preview(
                    os.path.join(output_dir, f"preview_{step:09d}.jpg"),
                    reference.detach(), target.detach(), output["reconstruction"].detach(),
                )

            if eval_interval > 0 and step % eval_interval == 0:
                metrics = validate(model, val_loader, device, config, causal_strength)
                if rank == 0:
                    LOG.info("validation step=%d %s", step, json.dumps(metrics, sort_keys=True))
                    for name, value in metrics.items():
                        writer.add_scalar(f"validation/{name}", value, step)
                barrier()

            if save_interval > 0 and step % save_interval == 0:
                barrier()
                if rank == 0:
                    save_checkpoint(
                        os.path.join(output_dir, f"step_{step:09d}.pt"),
                        model, image_discriminator, video_discriminator,
                        generator_optimizer, discriminator_optimizer, scaler,
                        step, epoch, config,
                    )
                barrier()
        epoch += 1

    barrier()
    if rank == 0 and not args.skip_final_save:
        save_checkpoint(
            os.path.join(output_dir, f"step_{step:09d}.pt"),
            model, image_discriminator, video_discriminator,
            generator_optimizer, discriminator_optimizer, scaler,
            step, epoch, config,
        )
    if rank == 0 and writer is not None:
        writer.close()
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
