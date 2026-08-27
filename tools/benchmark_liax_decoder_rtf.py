#!/usr/bin/env python
"""Benchmark decoder-only RTF for the local tokenizer or released LIA-X."""

import argparse
import os
import sys

import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twinlakes.motion_tokenizer.model import MotionTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("local", "official"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="video or image used to build reference features")
    parser.add_argument("--batch_sizes", default="1,4")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def read_reference(path):
    cap = cv2.VideoCapture(path)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
    x = F.interpolate(x, (512, 512), mode="bilinear", align_corners=False,
                      antialias=True)
    return x.div_(127.5).sub_(1)


def load_model(args, device):
    if args.kind == "local":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = MotionTokenizer(**checkpoint["config"]["model"])
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        t2av = "/nfs-speech-cfs/wangzhou/s2s/Talker-T2AV"
        sys.path.insert(0, t2av)
        from lia_x_recon import load_liax
        model, _ = load_liax(args.checkpoint, motion_dim=40, scale=2, device=device)
        return model.eval()
    return model.to(device).eval()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda")
    model = load_model(args, device)
    reference = read_reference(args.image).to(device)
    autocast_enabled = args.dtype == "bf16"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        if args.kind == "local":
            style, features = model.encode_reference(reference)
            alpha = model.encoder.enc_r2t(style)
            decoder = model.decoder
        else:
            style, features = model.enc.enc_2r(reference)
            alpha = model.enc.enc_r2t(style)
            decoder = model.dec

    params = sum(p.numel() for p in decoder.parameters())
    print(f"kind={args.kind} dtype={args.dtype} decoder_params={params / 1e6:.3f}M")
    for batch in [int(v) for v in args.batch_sizes.split(",") if v]:
        style_b = style.expand(batch, -1)
        alpha_b = alpha.expand(batch, -1)
        features_b = [f.expand(batch, -1, -1, -1) for f in features]

        def decode():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                return decoder(style_b, [alpha_b], features_b)

        for _ in range(args.warmup):
            decode()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            decode()
        end.record()
        torch.cuda.synchronize()
        total_s = start.elapsed_time(end) / 1000.0
        frames = batch * args.repeats
        sec_per_frame = total_s / frames
        fps = 1.0 / sec_per_frame
        rtf = sec_per_frame * args.fps
        print(f"batch={batch} repeats={args.repeats} total_s={total_s:.6f} "
              f"ms_per_frame={sec_per_frame * 1000:.3f} fps={fps:.3f} rtf={rtf:.4f}")


if __name__ == "__main__":
    main()
