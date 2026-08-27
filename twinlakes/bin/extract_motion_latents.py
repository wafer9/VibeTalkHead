#!/usr/bin/env python
"""Extract first-frame-relative motion deltas learned by the LIA-X tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from decord import VideoReader, cpu
from tqdm import tqdm

from twinlakes.motion_tokenizer.data import build_jsonl_index, resolve_video_path
from twinlakes.motion_tokenizer.model import MotionTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--video_root", default="/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards"
    )
    parser.add_argument("--batch_frames", type=int, default=64)
    parser.add_argument("--target_fps", type=float, default=25.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def distributed_setup():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def read_record(stream, offset: int):
    stream.seek(int(offset))
    return json.loads(stream.readline())


@torch.inference_mode()
def encode_video(model, path: str, target_fps: float, batch_frames: int,
                 image_size: int, device):
    reader = VideoReader(path, ctx=cpu(0), num_threads=4)
    source_fps = float(reader.get_avg_fps() or target_fps)
    step = source_fps / target_fps
    indices = np.rint(np.arange(0, len(reader), step)).astype(np.int64)
    indices = np.unique(np.clip(indices, 0, len(reader) - 1))
    chunks = []
    for start in range(0, len(indices), batch_frames):
        frames = torch.from_numpy(reader.get_batch(indices[start:start + batch_frames]).asnumpy())
        frames = frames.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(127.5).sub_(1)
        frames = F.interpolate(
            frames, size=(image_size, image_size),
            mode="bilinear", align_corners=False, antialias=True,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            motion = model.encode_motion(frames)
        chunks.append(motion.float().cpu())
    return torch.cat(chunks, dim=0)


def main():
    args = parse_args()
    rank, _, world_size, device = distributed_setup()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MotionTokenizer(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    image_size = int(checkpoint["config"]["data"]["image_size"])

    if rank == 0:
        build_jsonl_index(args.manifest)
        os.makedirs(args.output_root, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    offsets = np.load(args.manifest + ".idx.npy", mmap_mode="r")
    total = min(len(offsets), args.limit) if args.limit > 0 else len(offsets)
    indices = range(rank, total, world_size)
    manifest_out = os.path.join(args.output_root, f"manifest.rank{rank:03d}.jsonl")
    completed = skipped = failed = 0
    with open(args.manifest, "rb") as source, open(manifest_out, "w", buffering=1) as destination:
        for index in tqdm(indices, total=(total + world_size - 1 - rank) // world_size,
                          disable=rank != 0, desc="extract motion"):
            record = read_record(source, offsets[index])
            key = record.get("sample_id", f"sample_{index:09d}")
            dataset_source = record.get("dataset_source", "unknown")
            output_dir = os.path.join(args.output_root, dataset_source)
            output_path = os.path.join(output_dir, key + ".pt")
            if os.path.isfile(output_path) and not args.overwrite:
                skipped += 1
            else:
                try:
                    video_path = resolve_video_path(record["video_path"], args.video_root)
                    motion = encode_video(
                        model, video_path, args.target_fps, args.batch_frames,
                        image_size, device,
                    )
                    motion = motion - motion[:1]
                    os.makedirs(output_dir, exist_ok=True)
                    temporary = output_path + f".tmp.{os.getpid()}"
                    torch.save({
                        "motion": motion.half(),
                        "fps": args.target_fps,
                        "motion_dim": model.motion_dim,
                        "normalized": False,
                        "representation": "liax_first_frame_relative_delta",
                        "source_checkpoint": os.path.abspath(args.checkpoint),
                    }, temporary)
                    os.replace(temporary, output_path)
                    completed += 1
                except Exception as exc:
                    print(f"[rank {rank}] failed {key}: {exc!r}", flush=True)
                    failed += 1
                    continue
            enriched = dict(record)
            enriched["motion_latent"] = output_path
            destination.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    print(f"[rank {rank}] completed={completed} skipped={skipped} failed={failed} "
          f"manifest={manifest_out}", flush=True)
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        merged_manifest = os.path.join(args.output_root, "manifest.jsonl")
        with open(merged_manifest, "wb") as destination:
            for shard_rank in range(world_size):
                shard = os.path.join(args.output_root, f"manifest.rank{shard_rank:03d}.jsonl")
                with open(shard, "rb") as source:
                    shutil.copyfileobj(source, destination)
        print(f"merged manifest={merged_manifest}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
