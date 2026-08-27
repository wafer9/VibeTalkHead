#!/usr/bin/env python
"""Measure how strongly a trained LIA-X decoder responds to target alpha."""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from twinlakes.motion_tokenizer.model import MotionTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def preview_pair(path: str, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
    padding = 2
    tiles = []
    for index in range(2):
        left = padding + index * (image_size + padding)
        tiles.append(image[:, padding:padding + image_size, left:left + image_size])
    return tuple(tile.float().div(127.5).sub(1).unsqueeze(0) for tile in tiles)


@torch.inference_mode()
def main():
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = MotionTokenizer(**config["model"])
    model.load_state_dict(payload["model"], strict=True)
    del payload
    model = model.cuda().eval()

    image_size = int(config["data"]["image_size"])
    reference, target = preview_pair(args.preview, image_size)
    reference, target = reference.cuda(), target.cuda()

    style, features = model.encode_reference(reference)
    reference_alpha = model.encoder.enc_r2t(style)
    target_alpha = model.encode_motion(target)
    direction_reference = model.decoder.direction(reference_alpha)
    direction_target = model.decoder.direction(target_alpha)

    source_reconstruction = model.decoder(style, None, features)
    reference_reconstruction = model.decoder(style, [reference_alpha], features)
    reference_stats = model.decoder.last_flow_stats.clone()
    target_prediction = model.decoder(style, [target_alpha], features)
    target_stats = model.decoder.last_flow_stats.clone()
    midpoint_alpha = reference_alpha + 0.5 * (target_alpha - reference_alpha)
    midpoint_prediction = model.decoder(style, [midpoint_alpha], features)

    alpha_delta = target_alpha - reference_alpha
    metrics = {
        "target_reference_l1": (target - reference).abs().mean().item(),
        "prediction_target_l1": (target_prediction - target).abs().mean().item(),
        "prediction_reference_l1": (target_prediction - reference).abs().mean().item(),
        "reference_reconstruction_l1": (reference_reconstruction - reference).abs().mean().item(),
        "source_zero_alpha_l1": (source_reconstruction - reference).abs().mean().item(),
        "alpha_delta_l1": alpha_delta.abs().mean().item(),
        "alpha_delta_l2": alpha_delta.norm(dim=-1).mean().item(),
        "reference_alpha_l2": reference_alpha.norm(dim=-1).mean().item(),
        "target_alpha_l2": target_alpha.norm(dim=-1).mean().item(),
        "direction_delta_l2": (direction_target - direction_reference).norm(dim=-1).mean().item(),
        "direction_to_style_ratio": (
            (direction_target - direction_reference).norm(dim=-1)
            / style.norm(dim=-1).clamp_min(1e-8)
        ).mean().item(),
        "decoder_target_vs_reference_l1": (
            target_prediction - reference_reconstruction
        ).abs().mean().item(),
        "decoder_target_vs_zero_alpha_l1": (
            target_prediction - source_reconstruction
        ).abs().mean().item(),
        "decoder_midpoint_vs_reference_l1": (
            midpoint_prediction - reference_reconstruction
        ).abs().mean().item(),
        "reference_flow": reference_stats[:, 0].mean().item(),
        "target_flow": target_stats[:, 0].mean().item(),
        "reference_mask": reference_stats[:, 1].mean().item(),
        "target_mask": target_stats[:, 1].mean().item(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_image(
        torch.cat([
            reference, target, source_reconstruction, reference_reconstruction,
            midpoint_prediction, target_prediction,
        ]).add(1).mul(0.5).clamp(0, 1),
        args.output,
        nrow=6,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.8f}")


if __name__ == "__main__":
    main()
