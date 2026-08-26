#!/usr/bin/env python
"""Render the GT-motion upper bound (and normalized-noise stress tests)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

from twinlakes.motion_tokenizer.data import resolve_video_path
from twinlakes.motion_tokenizer.model import MotionTokenizer, structured_motion_noise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--video_root", default="/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards"
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--resolution", type=int, default=0,
        help="output resolution; 0 uses the checkpoint training resolution",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--encode_batch", type=int, default=64)
    parser.add_argument("--render_chunk", type=int, default=4)
    parser.add_argument("--noise_sigma", type=float, default=0.0)
    parser.add_argument("--noise_mode", choices=["iid", "bias", "drift", "mixed"], default="iid")
    parser.add_argument("--causal_strength", type=float, default=1.0)
    parser.add_argument(
        "--mux_audio", action="store_true",
        help="mux each manifest wav_path/audio into the reconstructed mp4",
    )
    parser.add_argument(
        "--audio_root", default=None,
        help="base directory for relative wav_path/audio entries (defaults to manifest directory)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def video_reader_and_indices(path: str, fps: float):
    reader = VideoReader(path, ctx=cpu(0), num_threads=4)
    source_fps = float(reader.get_avg_fps() or fps)
    indices = np.rint(np.arange(0, len(reader), source_fps / fps)).astype(np.int64)
    indices = np.unique(np.clip(indices, 0, len(reader) - 1))
    return reader, indices


def decode_frames(reader, indices, resolution: int):
    frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).float()
    frames = frames.div_(127.5).sub_(1)
    if frames.shape[-2:] != (resolution, resolution):
        frames = F.interpolate(
            frames, size=(resolution, resolution), mode="bilinear",
            align_corners=False, antialias=True,
        )
    return frames


@torch.inference_mode()
def reconstruct_to_file(model, video_path: str, output_path: str, args, device):
    reader, indices = video_reader_and_indices(video_path, args.fps)
    reference = decode_frames(reader, indices[:1], args.resolution).to(device)
    motion_parts = []
    for start in range(0, len(indices), args.encode_batch):
        chunk = decode_frames(
            reader, indices[start:start + args.encode_batch], model.motion_input_size
        ).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            motion = model.encode_motion(chunk)
        motion_parts.append(motion)
    absolute_motion = torch.cat(motion_parts, dim=0)
    motion_delta = absolute_motion - absolute_motion[:1]
    normalizer_ready = bool(model.normalizer.ready.item())
    normalized = (
        model.normalizer.normalize_delta(motion_delta)
        if normalizer_ready else motion_delta
    ).unsqueeze(0)
    if args.noise_sigma > 0:
        if not normalizer_ready:
            raise RuntimeError(
                "--noise_sigma requires a finalized motion normalizer"
            )
        normalized = structured_motion_noise(normalized, args.noise_sigma, args.noise_mode)

    # Apply the causal adapter to the full (tiny) latent sequence once, then
    # render/write bounded chunks. This avoids holding a long 512p video in RAM
    # or GPU memory while preserving causal state across chunk boundaries.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        motion_delta = (
            model.normalizer.denormalize_delta(normalized)
            if normalizer_ready else normalized
        )
        delta = model.motion_adapter(motion_delta, args.causal_strength)
        reference_features = model.reference_encoder(reference)
        if model.shared_motion_encoder:
            _, reference_motion = model._shared_style_and_motion(reference_features)
        else:
            reference_motion = model.encode_motion(reference)

    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
        (args.resolution, args.resolution),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {output_path}")
    try:
        for start in range(0, delta.shape[1], args.render_chunk):
            stop = min(delta.shape[1], start + args.render_chunk)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model._render_sequence(
                    reference, reference_features, delta[:, start:stop], args.render_chunk,
                    reference_motion=reference_motion,
                )["image"][0]
            rgb = prediction.float().cpu().add(1).mul(127.5).clamp(0, 255)
            rgb = rgb.byte().permute(0, 2, 3, 1).numpy()
            for frame in rgb:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def resolve_audio_path(record, manifest: str, audio_root: str | None) -> str:
    path = record.get("wav_path") or record.get("audio")
    if not path:
        raise ValueError("record has no wav_path/audio for --mux_audio")
    if os.path.isabs(path):
        return path
    root = audio_root or os.path.dirname(os.path.abspath(manifest))
    return os.path.join(root, path)


def mux_audio(
    video_path: str, source_video_path: str, audio_path: str, output_path: str
) -> None:
    """Mux the exact GT audio track, falling back to the manifest waveform."""
    source_command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-i", source_video_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy",
        "-shortest", "-movflags", "+faststart", output_path,
    ]
    try:
        subprocess.run(source_command, check=True)
        return
    except subprocess.CalledProcessError:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(audio_path)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", video_path, "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-shortest", "-movflags", "+faststart", output_path,
    ], check=True)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MotionTokenizer(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    if not bool(model.normalizer.ready.item()):
        print(
            "checkpoint has no finalized normalizer; clean GT-motion reconstruction "
            "will use raw reference-relative delta"
        )
    if args.resolution <= 0:
        args.resolution = int(
            checkpoint["config"].get("data", {}).get(
                "image_size", model.motion_input_size
            )
        )
    model = model.to(device).eval()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.manifest) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if args.limit > 0:
        records = records[:args.limit]
    completed = skipped = failed = 0
    for record in tqdm(records, desc="motion upper bound"):
        key = record.get("sample_id") or os.path.splitext(os.path.basename(record["video_path"]))[0]
        output_path = os.path.join(args.output_dir, key + ".mp4")
        if os.path.isfile(output_path) and not args.overwrite:
            skipped += 1
            continue
        try:
            video_path = resolve_video_path(record["video_path"], args.video_root)
            if args.mux_audio:
                silent_path = os.path.join(args.output_dir, key + ".noaudio.mp4")
                try:
                    reconstruct_to_file(model, video_path, silent_path, args, device)
                    audio_path = resolve_audio_path(record, args.manifest, args.audio_root)
                    mux_audio(silent_path, video_path, audio_path, output_path)
                finally:
                    if os.path.isfile(silent_path):
                        os.remove(silent_path)
            else:
                reconstruct_to_file(model, video_path, output_path, args, device)
            completed += 1
        except Exception as exc:
            print(f"failed {key}: {exc!r}", flush=True)
            failed += 1
    print(f"completed={completed} skipped={skipped} failed={failed} output={args.output_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
