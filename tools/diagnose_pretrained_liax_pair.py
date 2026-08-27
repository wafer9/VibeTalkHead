#!/usr/bin/env python
"""Compare an official pretrained LIA-X reconstruction with source copying."""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_index", type=int, default=0)
    return parser.parse_args()


def preview_tiles(path):
    image = torch.from_numpy(__import__("numpy").array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
    _, height, width = image.shape
    count, padding = 9, 2
    tile_width = (width - padding * (count + 1)) // count
    tile_height = height - 2 * padding
    return torch.stack([
        image[:, padding:padding + tile_height,
              padding + index * (tile_width + padding):padding + index * (tile_width + padding) + tile_width]
        for index in range(count)
    ]).float().div(127.5).sub(1)


@torch.inference_mode()
def main():
    args = parse_args()
    source_root = "/nfs-speech-cfs/wangzhou/s2s/Talker-T2AV"
    sys.path.insert(0, source_root)
    from lia_x.networks.generator import Generator

    tiles = preview_tiles(args.preview)
    reference = tiles[0:1].cuda()
    target = tiles[1 + 2 * args.target_index:2 + 2 * args.target_index].cuda()

    model = Generator(style_dim=512, motion_dim=40, scale=2).cuda().eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)

    reference_style, features = model.enc.enc_2r(reference)
    reference_alpha = model.enc.enc_r2t(reference_style)
    target_alpha = model.get_alpha(target)
    prediction = model.dec(reference_style, [target_alpha], features)
    reference_reconstruction = model.dec(reference_style, [reference_alpha], features)

    metrics = {
        "target_reference_l1": (target - reference).abs().mean().item(),
        "prediction_reference_l1": (prediction - reference).abs().mean().item(),
        "prediction_target_l1": (prediction - target).abs().mean().item(),
        "target_vs_reference_alpha_l1": (target_alpha - reference_alpha).abs().mean().item(),
        "output_motion_response_l1": (prediction - reference_reconstruction).abs().mean().item(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_image(
        torch.cat([reference, target, reference_reconstruction, prediction]).add(1).mul(0.5).clamp(0, 1),
        args.output,
        nrow=4,
    )
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
