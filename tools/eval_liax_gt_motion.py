#!/usr/bin/env python
"""Batch LIA-X GT-motion reconstruction for renderer upper-bound evaluation.

For every JSONL sample, the first video frame supplies identity/appearance and
each GT frame supplies only its 40-D LIA-X motion code.  The reconstructed mp4
keeps the input basename so it can be evaluated directly by eval_fid_fvd.py.
"""
import argparse
import json
import os
import sys
import zlib

import cv2
import numpy as np
import torch
from tqdm import tqdm


T2AV = "/nfs-speech-cfs/wangzhou/s2s/Talker-T2AV"
VIBEHEAD = "/nfs-speech-cfs/wangzhou/s2s/vibehead"
sys.path.insert(0, T2AV)
from lia_x_recon import load_liax, read_video_frames, tensorize  # noqa: E402


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_data", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--ckpt", default=os.path.join(T2AV, "deps/LIA-X/lia-x.pt"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--noise_sigma", type=float, default=0.0,
                   help="Gaussian perturbation std in normalized motion space")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--motion_std", default=os.path.join(
        VIBEHEAD, "twinlakes/dataset/stats/motion_std.npy"))
    return p.parse_args()


def detensorize(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1).byte().cpu().numpy()


@torch.inference_mode()
def reconstruct(model, video_path, out_path, args, motion_std, rng):
    frames_np, _ = read_video_frames(
        video_path, args.resolution, max_frames=None, target_fps=args.fps)
    frames = tensorize(frames_np, args.device)

    # Appearance is fixed to frame zero. Only alpha_t changes with GT motion.
    z_ref, feats_ref = model.enc.enc_2r(frames[:1])
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.resolution, args.resolution))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create video: {out_path}")

    try:
        for start in range(0, len(frames), args.batch):
            x = frames[start:start + args.batch]
            z_target, _ = model.enc.enc_2r(x)
            alpha = model.enc.enc_r2t(z_target)
            if args.noise_sigma > 0:
                # alpha is raw LIA-X motion. Scale each dimension by its
                # training-set std so sigma has a consistent normalized meaning.
                eps = torch.from_numpy(
                    rng.standard_normal(tuple(alpha.shape)).astype(np.float32))
                alpha = alpha + args.noise_sigma * motion_std * eps.to(alpha.device)
            motion_direction = model.dec.direction(alpha)
            batch_size = len(x)
            feats = [f.expand(batch_size, -1, -1, -1) for f in feats_ref]
            recon = model.dec(z_ref.expand(batch_size, -1) + motion_direction,
                              None, feats)
            for rgb in detensorize(recon):
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return len(frames)


def main():
    args = get_args()
    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.test_data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        records = records[:args.limit]

    model, _ = load_liax(args.ckpt, motion_dim=40, scale=2, device=args.device)
    motion_std = torch.from_numpy(np.load(args.motion_std).astype(np.float32)).to(args.device)
    motion_std = motion_std.clamp_min(1e-6).unsqueeze(0)
    print(f"[noise] sigma={args.noise_sigma:g} normalized units, seed={args.seed}, "
          f"raw per-dim std mean={motion_std.mean().item():.5f}", flush=True)
    completed = skipped = failed = 0
    for record in tqdm(records, desc="LIA-X GT motion"):
        video_path = record.get("video_path") or record.get("video")
        key = record.get("sample_id") or record.get("key") or os.path.splitext(
            os.path.basename(video_path))[0]
        out_path = os.path.join(args.result_dir, f"{key}.mp4")
        if os.path.isfile(out_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            # Stable per-video noise: identical epsilon is used when comparing
            # different sigma values, independent of processing/resume order.
            sample_seed = (args.seed + zlib.crc32(key.encode("utf-8"))) & 0xFFFFFFFF
            rng = np.random.default_rng(sample_seed)
            n = reconstruct(model, video_path, out_path, args, motion_std, rng)
            completed += 1
            print(f"[done] {key}: {n} frames", flush=True)
        except Exception as exc:
            failed += 1
            print(f"[fail] {key}: {exc!r}", flush=True)
            if os.path.exists(out_path):
                os.unlink(out_path)

    print(f"[summary] completed={completed}, skipped={skipped}, failed={failed}, "
          f"result_dir={args.result_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
