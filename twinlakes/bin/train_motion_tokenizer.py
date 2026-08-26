#!/usr/bin/env python
"""Train the vendored official LIA-X encoder/decoder."""

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

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from twinlakes.motion_tokenizer.data import JsonlVideoClipDataset, build_jsonl_index, seed_worker
from twinlakes.motion_tokenizer.losses import (
    LocalVGGPerceptualLoss,
    MultiScaleImageDiscriminator,
    discriminator_hinge,
    generator_hinge,
    set_requires_grad,
)
from twinlakes.motion_tokenizer.model import MotionTokenizer

LOG = logging.getLogger("liax_motion_tokenizer")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir")
    parser.add_argument("--resume")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--train_manifest")
    parser.add_argument("--val_manifest")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--render_chunk", type=int)
    parser.add_argument("--accumulation_steps", type=int)
    parser.add_argument("--image_gan_start", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--skip_final_save", action="store_true")
    return parser.parse_args()


def distributed_setup():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if backend == "nccl":
            dist.init_process_group(backend, device_id=device)
        else:
            dist.init_process_group(backend)
    return rank, local_rank, world_size, device


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def raw_module(module):
    return module.module if isinstance(module, DDP) else module


def seed_all(seed, rank):
    seed += rank * 100003
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def autocast_context(device, dtype):
    if device.type != "cuda" or dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if dtype == "bf16" else torch.float16,
    )


def set_lr(optimizer, step, training):
    warmup = int(training.get("warmup_steps", 0))
    maximum = int(training["max_steps"])
    if step < warmup:
        scale = max(step + 1, 1) / max(warmup, 1)
    else:
        progress = min(max((step - warmup) / max(maximum - warmup, 1), 0.0), 1.0)
        scale = 0.5 * (1 + math.cos(math.pi * progress))
    minimum = float(training.get("min_lr_ratio", 0.1))
    lr = float(training["lr"]) * (minimum + (1 - minimum) * scale)
    for group in optimizer.param_groups:
        group["lr"] = lr


def select_pairs(prediction, target, maximum):
    prediction = prediction.flatten(0, 1)
    target = target.flatten(0, 1)
    if maximum > 0 and prediction.shape[0] > maximum:
        indices = torch.randperm(prediction.shape[0], device=prediction.device)[:maximum]
        prediction = prediction[indices]
        target = target[indices]
    return prediction, target


def generator_losses(output, target, config, perceptual, discriminator, gan_active):
    prediction = output["reconstruction"]
    zero = prediction.new_zeros(())
    losses = {
        "reconstruction": F.l1_loss(prediction, target),
        "perceptual": zero,
        "image_adversarial": zero,
    }
    if float(config.get("perceptual", 0.0)) != 0:
        fake, real = select_pairs(
            prediction, target, int(config.get("max_perceptual_frames", 8))
        )
        losses["perceptual"] = perceptual(fake, real)
    if gan_active:
        fake, _ = select_pairs(prediction, target, int(config.get("max_gan_frames", 8)))
        losses["image_adversarial"] = generator_hinge(discriminator(fake))
    losses["total"] = sum(
        float(config.get(name, 0.0)) * value for name, value in losses.items()
    )
    return losses


def discriminator_loss(output, target, config, discriminator):
    fake, real = select_pairs(
        output["reconstruction"].detach(),
        target,
        int(config.get("max_gan_frames", 8)),
    )
    return discriminator_hinge(discriminator(real), discriminator(fake))


def temporal_errors(prediction, target):
    if prediction.shape[1] < 2:
        zero = prediction.new_zeros(())
        return zero, zero
    pred_v = prediction[:, 1:] - prediction[:, :-1]
    target_v = target[:, 1:] - target[:, :-1]
    velocity = F.l1_loss(pred_v, target_v)
    if prediction.shape[1] < 3:
        return velocity, prediction.new_zeros(())
    return velocity, F.l1_loss(
        pred_v[:, 1:] - pred_v[:, :-1],
        target_v[:, 1:] - target_v[:, :-1],
    )


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    for index, batch in enumerate(loader):
        if index >= int(config["training"].get("validation_batches", 8)):
            break
        reference = batch["reference"].to(device, non_blocking=True)
        target = batch["frames"].to(device, non_blocking=True)
        with autocast_context(device, config["training"].get("dtype", "bf16")):
            prediction = model(
                reference,
                target,
                render_chunk=int(config["training"].get("render_chunk", 1)),
            )["reconstruction"]
        prediction, target = prediction.float(), target.float()
        velocity, acceleration = temporal_errors(prediction, target)
        totals += torch.stack([
            F.l1_loss(prediction, target).double(),
            F.mse_loss(prediction, target).double(),
            velocity.double(),
            acceleration.double(),
            prediction.new_ones(()).double(),
        ])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    count = totals[-1].clamp_min(1)
    mse = float((totals[1] / count).item())
    model.train()
    return {
        "l1": float((totals[0] / count).item()),
        "psnr": -10 * math.log10(max(mse / 4.0, 1e-12)),
        "velocity": float((totals[2] / count).item()),
        "acceleration": float((totals[3] / count).item()),
    }


@torch.no_grad()
def reduce_scalars(values):
    names = list(values)
    packed = torch.stack([values[name].detach().float().reshape(()) for name in names])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed)
        packed.div_(dist.get_world_size())
    return {name: float(value.item()) for name, value in zip(names, packed)}


def latest_checkpoint(output_dir):
    found = []
    for path in Path(output_dir).glob("step_*.pt"):
        match = re.search(r"step_(\d+)\.pt$", path.name)
        if match:
            found.append((int(match.group(1)), str(path)))
    return max(found)[1] if found else None


def save_checkpoint(path, model, discriminator, g_optimizer, d_optimizer, scaler, step, epoch, config):
    payload = {
        "model": raw_module(model).state_dict(),
        "image_discriminator": raw_module(discriminator).state_dict() if discriminator else None,
        "generator_optimizer": g_optimizer.state_dict(),
        "discriminator_optimizer": d_optimizer.state_dict() if d_optimizer else None,
        "scaler": scaler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config,
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, discriminator, g_optimizer, d_optimizer, scaler):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_module(model).load_state_dict(payload["model"], strict=True)
    if discriminator and payload.get("image_discriminator") is not None:
        raw_module(discriminator).load_state_dict(payload["image_discriminator"], strict=True)
    g_optimizer.load_state_dict(payload["generator_optimizer"])
    if d_optimizer and payload.get("discriminator_optimizer") is not None:
        d_optimizer.load_state_dict(payload["discriminator_optimizer"])
    if payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("step", 0)), int(payload.get("epoch", 0))


def save_preview(path, reference, target, prediction):
    count = min(target.shape[1], 6)
    indices = torch.linspace(0, target.shape[1] - 1, count, device=target.device).long()
    images = [reference[0]]
    for index in indices:
        images.extend([target[0, index], prediction[0, index]])
    save_image(
        make_grid(torch.stack(images).add(1).mul(0.5).clamp(0, 1), nrow=1 + 2 * count),
        path,
    )


def build_loaders(config, rank, world_size, generator):
    data = config["data"]
    common = dict(
        video_root=data["video_root"],
        clip_length=int(data["clip_length"]),
        image_size=int(data["image_size"]),
        target_fps=float(data.get("target_fps", 25)),
        max_retries=int(data.get("max_retries", 12)),
    )
    train_set = JsonlVideoClipDataset(
        data["train_manifest"],
        training=True,
        first_frame_reference_prob=float(data.get("first_frame_reference_prob", 0.5)),
        photometric_strength=float(data.get("photometric_strength", 0.08)),
        horizontal_flip_prob=float(data.get("horizontal_flip_prob", 0.5)),
        return_cross_reference=False,
        max_samples=int(data.get("max_train_samples", 0)),
        **common,
    )
    val_set = JsonlVideoClipDataset(
        data["val_manifest"],
        training=False,
        photometric_strength=0,
        horizontal_flip_prob=0,
        return_cross_reference=False,
        max_samples=int(data.get("max_val_samples", 0)),
        **common,
    )
    train_sampler = (
        DistributedSampler(train_set, world_size, rank, shuffle=True, seed=int(config.get("seed", 2026)))
        if world_size > 1 else None
    )
    val_sampler = (
        DistributedSampler(val_set, world_size, rank, shuffle=False)
        if world_size > 1 else None
    )
    workers = int(data.get("num_workers", 4))
    kwargs = dict(
        batch_size=int(data["batch_size"]),
        num_workers=workers,
        pin_memory=bool(data.get("pin_memory", True)),
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if workers:
        kwargs["prefetch_factor"] = int(data.get("prefetch_factor", 2))
    train_loader = DataLoader(
        train_set, sampler=train_sampler, shuffle=train_sampler is None,
        drop_last=True, **kwargs,
    )
    val_kwargs = copy.copy(kwargs)
    val_kwargs["batch_size"] = int(data.get("val_batch_size", 1))
    val_loader = DataLoader(
        val_set, sampler=val_sampler, shuffle=False, drop_last=False, **val_kwargs,
    )
    return train_set, val_set, train_sampler, train_loader, val_loader


def main():
    args = parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    for argument, section, key in (
        ("output_dir", None, "output_dir"),
        ("train_manifest", "data", "train_manifest"),
        ("val_manifest", "data", "val_manifest"),
        ("num_workers", "data", "num_workers"),
        ("batch_size", "data", "batch_size"),
        ("render_chunk", "training", "render_chunk"),
        ("accumulation_steps", "training", "accumulation_steps"),
        ("image_gan_start", "stages", "image_gan_start"),
        ("log_interval", "training", "log_interval"),
        ("max_steps", "training", "max_steps"),
    ):
        value = getattr(args, argument)
        if value is not None:
            if section is None:
                config[key] = value
            else:
                config[section][key] = value

    rank, local_rank, world_size, device = distributed_setup()
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    generator = seed_all(int(config.get("seed", 2026)), rank)
    output_dir = os.path.abspath(config["output_dir"])
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train.yaml"), "w") as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        for key in ("train_manifest", "val_manifest"):
            build_jsonl_index(config["data"][key])
    barrier()

    train_set, val_set, train_sampler, train_loader, val_loader = build_loaders(
        config, rank, world_size, generator
    )
    model = MotionTokenizer(**config["model"]).to(device)
    training, loss_config = config["training"], config["loss"]
    gan_start = int(config["stages"].get("image_gan_start", -1))
    discriminator = (
        MultiScaleImageDiscriminator(**config["image_discriminator"]).to(device)
        if gan_start >= 0 else None
    )
    g_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["lr"]),
        betas=tuple(training.get("betas", [0.9, 0.95])),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    d_optimizer = (
        torch.optim.AdamW(
            discriminator.parameters(),
            lr=float(training.get("discriminator_lr", training["lr"])),
            betas=tuple(training.get("discriminator_betas", [0.0, 0.99])),
            weight_decay=float(training.get("discriminator_weight_decay", 0.0)),
        )
        if discriminator else None
    )
    dtype = training.get("dtype", "bf16")
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and dtype == "fp16")
    perceptual = LocalVGGPerceptualLoss(
        loss_config.get("vgg_weights_path")
        if float(loss_config.get("perceptual", 0.0)) != 0 else None
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
        if discriminator:
            discriminator = DDP(discriminator, device_ids=[local_rank])

    step, epoch = 0, 0
    resume = args.resume or config.get("resume")
    if resume == "latest":
        resume = latest_checkpoint(output_dir)
    if resume:
        step, epoch = load_checkpoint(
            resume, model, discriminator, g_optimizer, d_optimizer, scaler
        )
        LOG.info("resumed %s at step %d", resume, step)

    writer = SummaryWriter(output_dir) if rank == 0 else None
    if rank == 0:
        parameters = sum(p.numel() for p in raw_module(model).parameters())
        LOG.info(
            "device=%s world_size=%d train=%d val=%d parameters=%.3fM",
            device, world_size, len(train_set), len(val_set), parameters / 1e6,
        )

    maximum = int(training["max_steps"])
    accumulation = int(training.get("accumulation_steps", 1))
    render_chunk = int(training.get("render_chunk", 1))
    log_interval = int(training.get("log_interval", 20))
    preview_interval = int(training.get("preview_interval", 500))
    eval_interval = int(training.get("eval_interval", 1000))
    save_interval = int(training.get("save_interval", 5000))
    last_log = time.time()
    model.train()
    g_optimizer.zero_grad(set_to_none=True)
    if d_optimizer:
        d_optimizer.zero_grad(set_to_none=True)

    while step < maximum:
        if train_sampler:
            train_sampler.set_epoch(epoch)
        for micro_index, batch in enumerate(train_loader):
            if step >= maximum:
                break
            set_lr(g_optimizer, step, training)
            gan_active = gan_start >= 0 and step >= gan_start
            reference = batch["reference"].to(device, non_blocking=True)
            target = batch["frames"].to(device, non_blocking=True)
            accumulation_boundary = (micro_index + 1) % accumulation == 0

            set_requires_grad(discriminator, False)
            generator_sync = (
                model.no_sync()
                if isinstance(model, DDP) and not accumulation_boundary
                else contextlib.nullcontext()
            )
            with generator_sync:
                with autocast_context(device, dtype):
                    output = model(reference, target, render_chunk=render_chunk)
                    losses = generator_losses(
                        output, target, loss_config, perceptual, discriminator, gan_active
                    )
                scaler.scale(losses["total"] / accumulation).backward()

            d_loss = target.new_zeros(())
            if gan_active:
                set_requires_grad(discriminator, True)
                discriminator_sync = (
                    discriminator.no_sync()
                    if isinstance(discriminator, DDP) and not accumulation_boundary
                    else contextlib.nullcontext()
                )
                with discriminator_sync:
                    with autocast_context(device, dtype):
                        d_loss = discriminator_loss(
                            output, target, loss_config, discriminator
                        )
                    scaler.scale(d_loss / accumulation).backward()

            if not accumulation_boundary:
                continue
            scaler.unscale_(g_optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training.get("grad_clip", 1.0))
            )
            scaler.step(g_optimizer)
            if gan_active:
                scaler.unscale_(d_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    float(training.get("discriminator_grad_clip", 5.0)),
                )
                scaler.step(d_optimizer)
            scaler.update()
            g_optimizer.zero_grad(set_to_none=True)
            if d_optimizer:
                d_optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % log_interval == 0:
                with torch.no_grad():
                    alpha = output["target_motion"].detach().flatten(0, 1).float()
                    alpha_std = alpha.std(dim=0, unbiased=False)
                    scalars = reduce_scalars({
                        **losses,
                        "image_discriminator": d_loss,
                        "alpha_std_mean": alpha_std.mean(),
                        "alpha_std_min": alpha_std.min(),
                        "alpha_std_max": alpha_std.max(),
                        "alpha_active_fraction": (alpha_std > 0.01).float().mean(),
                    })
                if rank == 0:
                    elapsed = max(time.time() - last_log, 1e-6)
                    last_log = time.time()
                    parts = [
                        f"step={step}",
                        f"loss={scalars['total']:.4f}",
                        f"rec={scalars['reconstruction']:.4f}",
                        f"perc={scalars['perceptual']:.4f}",
                    ]
                    if gan_active:
                        parts += [
                            f"g_adv={scalars['image_adversarial']:.4f}",
                            f"d_img={scalars['image_discriminator']:.4f}",
                        ]
                    parts += [
                        f"astd={scalars['alpha_std_mean']:.3f}",
                        f"active={scalars['alpha_active_fraction']:.2f}",
                        f"grad={float(grad_norm):.3f}",
                        f"lr={g_optimizer.param_groups[0]['lr']:.2e}",
                        f"mem={torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.0:.2f}GB",
                        f"steps/s={log_interval / elapsed:.3f}",
                    ]
                    LOG.info(" ".join(parts))
                    for name, value in scalars.items():
                        writer.add_scalar(f"train/{name}", value, step)
                    writer.add_scalar("train/grad_norm", float(grad_norm), step)
                    writer.add_scalar("lr/generator", g_optimizer.param_groups[0]["lr"], step)

            if rank == 0 and preview_interval > 0 and step % preview_interval == 0:
                save_preview(
                    os.path.join(output_dir, f"preview_{step:09d}.jpg"),
                    reference.detach(), target.detach(), output["reconstruction"].detach(),
                )
            if eval_interval > 0 and step % eval_interval == 0:
                metrics = validate(model, val_loader, device, config)
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
                        model, discriminator, g_optimizer, d_optimizer,
                        scaler, step, epoch, config,
                    )
                barrier()
        epoch += 1

    barrier()
    if rank == 0 and not args.skip_final_save:
        save_checkpoint(
            os.path.join(output_dir, f"step_{step:09d}.pt"),
            model, discriminator, g_optimizer, d_optimizer,
            scaler, step, epoch, config,
        )
    if writer:
        writer.close()
    barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
