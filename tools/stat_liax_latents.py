#!/usr/bin/env python
"""Statistics of released LIA-X alpha and first-frame-relative delta codes."""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


T2AV = "/nfs-speech-cfs/wangzhou/s2s/Talker-T2AV"
sys.path.insert(0, T2AV)
from lia_x_recon import load_liax, read_video_frames, tensorize  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def describe(values):
    mean = values.mean(axis=0)
    variance = values.var(axis=0)
    std = np.sqrt(variance)
    return {
        "num_vectors": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "global_mean": float(values.mean()),
        "global_variance": float(values.var()),
        "global_std": float(values.std()),
        "mean_abs_dimension_mean": float(np.abs(mean).mean()),
        "dimension_variance_mean": float(variance.mean()),
        "dimension_std_mean": float(std.mean()),
        "dimension_std_min": float(std.min()),
        "dimension_std_max": float(std.max()),
        "mean_per_dimension": mean.tolist(),
        "variance_per_dimension": variance.tolist(),
        "std_per_dimension": std.tolist(),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    with open(args.manifest) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if args.limit > 0:
        records = records[:args.limit]
    model, _ = load_liax(args.checkpoint, motion_dim=40, scale=2, device=args.device)
    alpha_all = []
    delta_all = []
    for record in tqdm(records, desc="LIA-X latent stats"):
        frames_np, _ = read_video_frames(
            record["video_path"], args.resolution, max_frames=None,
            target_fps=args.fps,
        )
        parts = []
        for start in range(0, len(frames_np), args.batch):
            frames = tensorize(frames_np[start:start + args.batch], args.device)
            style, _ = model.enc.enc_2r(frames)
            parts.append(model.enc.enc_r2t(style).float().cpu().numpy())
        alpha = np.concatenate(parts, axis=0)
        alpha_all.append(alpha)
        delta_all.append(alpha - alpha[:1])
    alpha_all = np.concatenate(alpha_all, axis=0).astype(np.float64)
    delta_all = np.concatenate(delta_all, axis=0).astype(np.float64)
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "manifest": os.path.abspath(args.manifest),
        "num_videos": len(records),
        "resolution": args.resolution,
        "fps": args.fps,
        "absolute_alpha": describe(alpha_all),
        "first_frame_relative_delta": describe(delta_all),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
