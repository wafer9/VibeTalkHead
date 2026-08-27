#!/usr/bin/env python
"""Compare the trained LIA-X decoder with an untrained efficient prototype."""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twinlakes.motion_tokenizer.model import MotionTokenizer
from twinlakes.motion_tokenizer.liax.efficient_decoder import EfficientWarpDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--batch_sizes", default="1,4,8")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--variant", choices=("lite", "balanced"), default="lite")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def read_reference(path: str) -> torch.Tensor:
    capture = cv2.VideoCapture(path)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot read {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(frame).permute(2, 0, 1).float()[None]
    image = F.interpolate(
        image, (512, 512), mode="bilinear", align_corners=False,
        antialias=True,
    )
    return image.div_(127.5).sub_(1)


def timed_forward(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return time.perf_counter() - start


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MotionTokenizer(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    variant = {
        "lite": {
            "channels": (512, 512, 384, 256, 128, 64, 32),
            "depths": (2, 2, 2, 2, 1, 1, 1),
        },
        "balanced": {
            "channels": (512, 512, 512, 512, 256, 128, 64),
            "depths": (3, 3, 3, 2, 2, 1, 1),
        },
    }[args.variant]
    candidate = EfficientWarpDecoder(
        style_dim=model.style_dim,
        motion_dim=model.motion_dim,
        source_channels=model.decoder.channels if hasattr(model.decoder, "channels") else (
            512, 512, 512, 512, 256, 128, 64
        ),
        channels=variant["channels"],
        depths=variant["depths"],
    ).to(device).eval()

    reference = read_reference(args.image).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        style, features = model.encode_reference(reference)
        alpha = model.encoder.enc_r2t(style)
        prepared = candidate.prepare_reference(features)

    original = model.decoder
    if args.compile:
        original = torch.compile(original, mode="max-autotune", fullgraph=False)
        candidate = torch.compile(candidate, mode="max-autotune", fullgraph=False)

    original_params = sum(parameter.numel() for parameter in model.decoder.parameters())
    candidate_params = sum(parameter.numel() for parameter in candidate.parameters())
    print(f"variant={args.variant}")
    print(f"original_params_m={original_params / 1e6:.3f}")
    print(f"candidate_params_m={candidate_params / 1e6:.3f}")

    batches = [int(value) for value in args.batch_sizes.split(",") if value]
    for batch in batches:
        style_batch = style.expand(batch, -1)
        alpha_batch = alpha.expand(batch, -1)
        original_features = [feature.expand(batch, -1, -1, -1) for feature in features]
        candidate_features = [
            feature.expand(batch, -1, -1, -1) for feature in prepared
        ]

        def run_original():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return original(style_batch, [alpha_batch], original_features)

        def run_candidate():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return candidate(style_batch, [alpha_batch], candidate_features)

        original_seconds = timed_forward(run_original, args.warmup, args.repeats)
        candidate_seconds = timed_forward(run_candidate, args.warmup, args.repeats)
        for name, seconds in (
            ("original", original_seconds), ("candidate", candidate_seconds)
        ):
            frames = batch * args.repeats
            milliseconds = seconds * 1000.0 / frames
            rtf = seconds / frames * args.fps
            print(
                f"name={name} batch={batch} ms_per_frame={milliseconds:.3f} "
                f"fps={1000.0 / milliseconds:.3f} rtf={rtf:.4f}"
            )
        print(f"speedup_batch_{batch}={original_seconds / candidate_seconds:.3f}x")


if __name__ == "__main__":
    main()
