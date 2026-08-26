#!/usr/bin/env python
"""Probe a motion encoder against cached, speaker-disjoint FAN targets.

This avoids repeating expensive landmark extraction when only the motion
encoder checkpoint changes.  The label cache is produced by
``tools/probe_motion_latent.py`` and must contain GT mouth geometry, clip IDs,
speaker IDs, and frame indices.  Renderer measurements in that cache are
intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from probe_motion_latent import (
    fit_mlp,
    fit_ridge,
    group_split,
    load_cache,
    metrics,
    read_records,
    speaker_id,
    video_frames,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinlakes.motion_tokenizer.data import resolve_video_path
from twinlakes.motion_tokenizer.model import MotionTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--video_root",
        default="/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--encode_batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite_cache", action="store_true")
    return parser.parse_args()


def load_motion_tokenizer(
    checkpoint_path: str, device: torch.device
) -> tuple[MotionTokenizer, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]["model"]
    config = dict(config)
    config["gradient_checkpointing"] = False
    model = MotionTokenizer(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    input_size = int(config.get("motion_input_size", 256))
    del checkpoint
    return model.to(device).eval(), input_size


@torch.inference_mode()
def encode_frames(
    model: MotionTokenizer,
    frames: np.ndarray,
    batch_size: int,
    input_size: int,
    device: torch.device,
) -> np.ndarray:
    values = []
    for start in range(0, len(frames), batch_size):
        image = torch.from_numpy(frames[start:start + batch_size]).to(device)
        image = image.permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        if image.shape[-2:] != (input_size, input_size):
            image = F.interpolate(
                image,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            values.append(model.encode_motion(image).float().cpu())
    motion = torch.cat(values)
    return (motion - motion[:1]).numpy().astype(np.float32)


def extract_features(args: argparse.Namespace, labels: dict, output: Path) -> np.ndarray:
    device = torch.device(args.device)
    model, input_size = load_motion_tokenizer(args.checkpoint, device)

    records = read_records(args.manifest, 0)
    record_by_key = {
        record.get("sample_id") or Path(record["video_path"]).stem: record
        for record in records
    }
    clips = list(dict.fromkeys(labels["clip"].tolist()))
    features = np.empty(
        (len(labels["clip"]), model.motion_dim), dtype=np.float32
    )
    for key in tqdm(clips, desc="15k motion extraction"):
        if key not in record_by_key:
            raise KeyError(f"cached clip is absent from manifest: {key}")
        mask = labels["clip"] == key
        frame_indices = labels["frame"][mask].astype(np.int64)
        record = record_by_key[key]
        path = resolve_video_path(record["video_path"], args.video_root)
        frames = video_frames(path, args.fps, args.resolution)
        if frame_indices.max(initial=-1) >= len(frames):
            raise IndexError(
                f"cached frame index exceeds decoded video for {key}: "
                f"{frame_indices.max()} >= {len(frames)}"
            )
        features[mask] = encode_frames(
            model,
            frames[frame_indices],
            args.encode_batch,
            input_size,
            device,
        )
    np.save(output, features)
    return features


def evaluate(x: np.ndarray, target: np.ndarray, groups: np.ndarray, seed: int) -> tuple[dict, dict]:
    masks = group_split(groups, seed)
    ridge, alpha = fit_ridge(x, target, masks["train"], masks["val"], masks["test"])
    mlp, iterations = fit_mlp(
        x, target, masks["train"], masks["val"], masks["test"], seed
    )
    held_out = target[masks["test"]]
    result = {
        "zero": metrics(np.zeros_like(held_out), held_out),
        "ridge": metrics(ridge, held_out),
        "mlp": metrics(mlp, held_out),
    }
    metadata = {
        "train_frames": int(masks["train"].sum()),
        "val_frames": int(masks["val"].sum()),
        "test_frames": int(masks["test"].sum()),
        "ridge_alpha": alpha,
        "mlp_iterations": iterations,
    }
    return result, metadata


def ridge_robustness(
    x: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    seed: int,
    splits: int = 10,
) -> dict:
    collected: dict[str, list[float]] = {
        "mouth_nme": [],
        "openness_rmse": [],
        "openness_r2": [],
        "openness_corr": [],
    }
    for offset in range(splits):
        masks = group_split(groups, seed + offset)
        prediction, _ = fit_ridge(
            x, target, masks["train"], masks["val"], masks["test"]
        )
        value = metrics(prediction, target[masks["test"]])
        for name in collected:
            collected[name].append(value[name])
    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in collected.items()
    }


def report_table(name: str, values: dict) -> list[str]:
    lines = [
        name,
        "method     mouth_nme   openness_rmse   openness_r2   openness_corr",
    ]
    for method in ("zero", "ridge", "mlp"):
        item = values[method]
        lines.append(
            f"{method:<10} {item['mouth_nme']:.5f}     {item['openness_rmse']:.5f}"
            f"          {item['openness_r2']:.4f}        {item['openness_corr']:.4f}"
        )
    return lines


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_cache(Path(args.labels_cache))
    feature_path = output_dir / "motion_delta.npy"
    if feature_path.is_file() and not args.overwrite_cache:
        x = np.load(feature_path)
    else:
        x = extract_features(args, labels, feature_path)
    if len(x) != len(labels["gt"]):
        raise ValueError(f"feature/label length mismatch: {len(x)} vs {len(labels['gt'])}")

    position_result, position_meta = evaluate(
        x, labels["gt"], labels["speaker"], args.seed
    )
    consecutive = labels["frame"] > 0
    indices = np.flatnonzero(consecutive)
    previous = indices - 1
    same_clip = labels["clip"][indices] == labels["clip"][previous]
    indices, previous = indices[same_clip], previous[same_clip]
    velocity_result, velocity_meta = evaluate(
        x[indices] - x[previous],
        labels["gt"][indices] - labels["gt"][previous],
        labels["speaker"][indices],
        args.seed,
    )
    robustness = {
        "position_ridge": ridge_robustness(
            x, labels["gt"], labels["speaker"], args.seed
        ),
        "velocity_ridge": ridge_robustness(
            x[indices] - x[previous],
            labels["gt"][indices] - labels["gt"][previous],
            labels["speaker"][indices],
            args.seed,
        ),
    }

    result = {
        "checkpoint": args.checkpoint,
        "labels_cache": args.labels_cache,
        "frames": int(len(x)),
        "clips": int(len(np.unique(labels["clip"]))),
        "speakers": int(len(np.unique(labels["speaker"]))),
        "position": position_result,
        "velocity": velocity_result,
        "robustness_10_splits": robustness,
        "metadata": {"position": position_meta, "velocity": velocity_meta},
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "15k cached-label motion latent probe",
        f"checkpoint: {args.checkpoint}",
        f"frames/clips/speakers: {result['frames']}/{result['clips']}/{result['speakers']}",
        "",
        *report_table("Reference-relative mouth position", position_result),
        "",
        *report_table("Mouth velocity", velocity_result),
        "",
        "Ridge robustness over 10 speaker-disjoint splits",
        "task       mouth_nme             openness_r2          openness_corr",
        (
            "position   "
            f"{robustness['position_ridge']['mouth_nme']['mean']:.5f} ± "
            f"{robustness['position_ridge']['mouth_nme']['std']:.5f}     "
            f"{robustness['position_ridge']['openness_r2']['mean']:.4f} ± "
            f"{robustness['position_ridge']['openness_r2']['std']:.4f}     "
            f"{robustness['position_ridge']['openness_corr']['mean']:.4f} ± "
            f"{robustness['position_ridge']['openness_corr']['std']:.4f}"
        ),
        (
            "velocity   "
            f"{robustness['velocity_ridge']['mouth_nme']['mean']:.5f} ± "
            f"{robustness['velocity_ridge']['mouth_nme']['std']:.5f}     "
            f"{robustness['velocity_ridge']['openness_r2']['mean']:.4f} ± "
            f"{robustness['velocity_ridge']['openness_r2']['std']:.4f}     "
            f"{robustness['velocity_ridge']['openness_corr']['mean']:.4f} ± "
            f"{robustness['velocity_ridge']['openness_corr']['std']:.4f}"
        ),
        "",
    ]
    report = "\n".join(lines)
    (output_dir / "report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
