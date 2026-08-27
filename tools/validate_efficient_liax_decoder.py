#!/usr/bin/env python
"""Numerical sanity checks for the standalone efficient decoder prototype."""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twinlakes.motion_tokenizer.model import MotionTokenizer
from twinlakes.motion_tokenizer.liax.efficient_decoder import EfficientWarpDecoder


def read_reference(path: str) -> torch.Tensor:
    capture = cv2.VideoCapture(path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame).permute(2, 0, 1).float()[None]
    image = F.interpolate(
        image, (512, 512), mode="bilinear", align_corners=False, antialias=True
    )
    return image.div_(127.5).sub_(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    teacher = MotionTokenizer(**checkpoint["config"]["model"])
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher = teacher.to(device).eval()
    decoder = EfficientWarpDecoder(
        style_dim=teacher.style_dim,
        motion_dim=teacher.motion_dim,
        source_channels=getattr(
            teacher.decoder, "channels", (512, 512, 512, 512, 256, 128, 64)
        ),
        channels=(512, 512, 512, 512, 256, 128, 64),
        depths=(3, 3, 3, 2, 2, 1, 1),
    ).to(device).train()

    reference = read_reference(args.image).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        source_style, features = teacher.encode_reference(reference)
        alpha = teacher.encoder.enc_r2t(source_style)
    source_style = source_style.detach()
    alpha = alpha.detach()
    features = [feature.detach() for feature in features]
    del teacher, checkpoint
    torch.cuda.empty_cache()

    raw_flows: list[torch.Tensor] = []
    hooks = [
        head.register_forward_hook(
            lambda _module, _inputs, output: raw_flows.append(output.detach().float())
        )
        for head in decoder.flow_heads
    ]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prepared = decoder.prepare_reference(features)
        output = decoder(source_style, [alpha], prepared)
        loss = output.float().square().mean()
    loss.backward()
    for hook in hooks:
        hook.remove()

    finite_gradient_tensors = 0
    gradient_tensors = 0
    for parameter in decoder.parameters():
        if parameter.grad is None:
            continue
        gradient_tensors += 1
        finite_gradient_tensors += int(torch.isfinite(parameter.grad).all())

    print(f"output_finite={int(torch.isfinite(output).all())}")
    print(f"output_abs_mean={output.detach().float().abs().mean().item():.8f}")
    print(f"grad_tensors={finite_gradient_tensors}/{gradient_tensors}")
    print(f"direction_grad_norm={decoder.direction.weight.grad.float().norm().item():.8e}")
    for index, (raw, limit) in enumerate(zip(raw_flows, decoder.flow_limits)):
        offset = torch.tanh(raw[:, :2]) * limit
        mask = torch.sigmoid(raw[:, 2:3])
        print(
            f"flow_stage={index} resolution={decoder.resolutions[decoder.warp_stage_indices[index]]} "
            f"offset_abs_mean={offset.abs().mean().item():.8e} "
            f"offset_abs_max={offset.abs().max().item():.8e} "
            f"mask_mean={mask.mean().item():.8f}"
        )

    decoder.eval()
    raw_flows.clear()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prepared = decoder.prepare_reference(features)
        output_a = decoder(source_style, [alpha], prepared)
        output_b = decoder(source_style, [alpha + 0.01], prepared)
    print(
        "motion_response_abs_mean="
        f"{(output_b.float() - output_a.float()).abs().mean().item():.8e}"
    )
    for resolution in decoder.resolutions:
        grid = getattr(decoder, f"base_grid_{resolution}")
        expected = 1.0 - 1.0 / resolution
        error = max(
            abs(grid[0, 0, 0, 0].item() + expected),
            abs(grid[0, -1, -1, 0].item() - expected),
        )
        print(f"grid_resolution={resolution} endpoint_error={error:.3e}")


if __name__ == "__main__":
    main()
