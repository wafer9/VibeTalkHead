#!/usr/bin/env python
"""Probe blink information in a motion latent and compare renderer oracles.

The probe deliberately uses the same frozen FAN, soft-argmax temperature and
GT-derived normalization assumptions as the training-time mouth dynamics loss.
Eye openness is the standard two-eye EAR, normalized by each clip's open-eye
90th percentile so thresholds transfer across identities.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinlakes.motion_tokenizer.losses import FrozenFANMouthVelocityLoss
from twinlakes.motion_tokenizer.model import MotionTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--renderer", action="append", default=[], metavar="NAME=DIR",
        help="renderer reconstruction directory; may be repeated",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--video_root", default="/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards"
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--encode_batch", type=int, default=64)
    parser.add_argument("--fan_batch", type=int, default=64)
    parser.add_argument("--blink_threshold", type=float, default=0.65)
    parser.add_argument("--confidence_threshold", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument(
        "--face_alignment_path",
        default=str(PROJECT_ROOT / "checkpoints/motion_tokenizer/probe_deps"),
    )
    parser.add_argument(
        "--fan_weights_path",
        default=str(PROJECT_ROOT / "checkpoints/motion_tokenizer/probe_weights/hub/checkpoints/2DFAN4-11f355bf06.pth.tar"),
    )
    return parser.parse_args()


def parse_renderers(values: Iterable[str]) -> Dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--renderer must be NAME=DIR, got {value!r}")
        name, directory = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"invalid/duplicate renderer name: {name!r}")
        result[name] = Path(directory)
    return result


def read_records(path: str, limit: int) -> list[dict]:
    with open(path) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return records[:limit] if limit > 0 else records


def speaker_id(sample_id: str) -> str:
    parts = sample_id.rsplit("_", 3)
    return parts[0] if len(parts) == 4 else sample_id


def resolve_video_path(path: str, video_root: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(video_root, path)


def video_frames(path: str, fps: float, resolution: int) -> np.ndarray:
    reader = VideoReader(path, ctx=cpu(0), num_threads=4)
    source_fps = float(reader.get_avg_fps() or fps)
    indices = np.rint(np.arange(0, len(reader), source_fps / fps)).astype(np.int64)
    indices = np.unique(np.clip(indices, 0, len(reader) - 1))
    frames = reader.get_batch(indices).asnumpy()
    if frames.shape[1:3] != (resolution, resolution):
        frames = np.stack([
            cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_AREA)
            for frame in frames
        ])
    return frames


@torch.inference_mode()
def encode_delta(
    model: MotionTokenizer, frames: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    values = []
    for start in range(0, len(frames), batch_size):
        image = torch.from_numpy(frames[start:start + batch_size]).to(device)
        image = image.permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            values.append(model.encode_motion(image).float().cpu())
    motion = torch.cat(values)
    return (motion - motion[:1]).numpy().astype(np.float32)


@torch.inference_mode()
def fan_landmarks(
    teacher: FrozenFANMouthVelocityLoss,
    frames: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    landmarks, confidence = [], []
    for start in range(0, len(frames), batch_size):
        image = torch.from_numpy(frames[start:start + batch_size]).to(device)
        image = image.permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        points, score = teacher._landmarks(image, keep_input_gradient=False)
        landmarks.append(points.cpu())
        confidence.append(score.cpu())
    return torch.cat(landmarks).numpy(), torch.cat(confidence).numpy()


def eye_aspect_ratio(points: np.ndarray) -> np.ndarray:
    def distance(a: int, b: int) -> np.ndarray:
        return np.linalg.norm(points[:, a] - points[:, b], axis=-1)

    left = (distance(37, 41) + distance(38, 40)) / (
        2.0 * np.maximum(distance(36, 39), 1e-4)
    )
    right = (distance(43, 47) + distance(44, 46)) / (
        2.0 * np.maximum(distance(42, 45), 1e-4)
    )
    return np.stack([left, right], axis=-1).astype(np.float32)


def build_cache(args: argparse.Namespace, renderers: Dict[str, Path], path: Path) -> Dict[str, np.ndarray]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MotionTokenizer(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    teacher = FrozenFANMouthVelocityLoss(
        args.fan_weights_path,
        package_path=args.face_alignment_path,
        temperature=20.0,
        confidence_threshold=args.confidence_threshold,
        use_checkpoint=False,
    ).to(device).eval()

    items: Dict[str, list] = {
        "x": [], "gt_eye": [], "clip": [], "speaker": [], "frame": [],
        "gt_confidence": [], "eye_baseline": [],
    }
    for name in renderers:
        items[f"renderer_{name}"] = []
        items[f"renderer_{name}_confidence"] = []
    failures = []

    for record in tqdm(read_records(args.manifest, args.limit), desc="blink latent/FAN"):
        key = record.get("sample_id") or Path(record["video_path"]).stem
        try:
            gt = video_frames(
                resolve_video_path(record["video_path"], args.video_root),
                args.fps, args.resolution,
            )
            generated = {
                name: video_frames(str(directory / f"{key}.mp4"), args.fps, args.resolution)
                for name, directory in renderers.items()
            }
            count = min([len(gt), *[len(value) for value in generated.values()]])
            gt = gt[:count]
            generated = {name: value[:count] for name, value in generated.items()}
            delta = encode_delta(model, gt, args.encode_batch, device)
            gt_points, gt_score = fan_landmarks(teacher, gt, args.fan_batch, device)
            gt_eye = eye_aspect_ratio(gt_points)
            baseline = np.maximum(np.quantile(gt_eye, 0.90, axis=0), 1e-4)
            gt_eye = gt_eye / baseline[None]

            items["x"].append(delta)
            items["gt_eye"].append(gt_eye.astype(np.float32))
            items["eye_baseline"].append(
                np.repeat(baseline[None].astype(np.float32), count, axis=0)
            )
            items["clip"].append(np.repeat(key, count))
            items["speaker"].append(np.repeat(speaker_id(key), count))
            items["frame"].append(np.arange(count, dtype=np.int32))
            items["gt_confidence"].append(gt_score[:, 36:48].mean(axis=1).astype(np.float32))
            for name, frames in generated.items():
                points, score = fan_landmarks(teacher, frames, args.fan_batch, device)
                items[f"renderer_{name}"].append((eye_aspect_ratio(points) / baseline[None]).astype(np.float32))
                items[f"renderer_{name}_confidence"].append(
                    score[:, 36:48].mean(axis=1).astype(np.float32)
                )
        except Exception as error:
            failures.append(f"{key}: {type(error).__name__}: {error}")

    if not items["x"]:
        raise RuntimeError("no clips extracted: " + "; ".join(failures[:5]))
    arrays = {name: np.concatenate(value) for name, value in items.items()}
    arrays["failures"] = np.asarray(failures, dtype=str)
    np.savez_compressed(path, **arrays)
    return arrays


def append_renderer_cache(
    args: argparse.Namespace,
    data: Dict[str, np.ndarray],
    renderers: Dict[str, Path],
    path: Path,
) -> Dict[str, np.ndarray]:
    """Append new renderer FAN outputs without recomputing GT or motion latents."""
    if "eye_baseline" not in data:
        raise RuntimeError(
            "existing cache predates eye_baseline support; rerun with --overwrite_cache"
        )
    device = torch.device(args.device)
    teacher = FrozenFANMouthVelocityLoss(
        args.fan_weights_path,
        package_path=args.face_alignment_path,
        temperature=20.0,
        confidence_threshold=args.confidence_threshold,
        use_checkpoint=False,
    ).to(device).eval()
    values = {name: [] for name in renderers}
    scores = {name: [] for name in renderers}
    failures = data.get("failures", np.asarray([], dtype=str)).tolist()
    for record in tqdm(read_records(args.manifest, args.limit), desc="append renderer FAN"):
        key = record.get("sample_id") or Path(record["video_path"]).stem
        mask = data["clip"] == key
        count = int(mask.sum())
        if count == 0:
            continue
        baseline = data["eye_baseline"][np.flatnonzero(mask)[0]]
        try:
            for name, directory in renderers.items():
                frames = video_frames(
                    str(directory / f"{key}.mp4"), args.fps, args.resolution,
                )
                if len(frames) < count:
                    raise RuntimeError(f"{name} has {len(frames)} frames, expected {count}")
                points, confidence = fan_landmarks(
                    teacher, frames[:count], args.fan_batch, device,
                )
                values[name].append(
                    (eye_aspect_ratio(points) / baseline[None]).astype(np.float32)
                )
                scores[name].append(
                    confidence[:, 36:48].mean(axis=1).astype(np.float32)
                )
        except Exception as error:
            raise RuntimeError(f"cannot append renderer for {key}: {error}") from error
    for name in renderers:
        if not values[name]:
            raise RuntimeError(f"no frames appended for renderer {name}")
        data[f"renderer_{name}"] = np.concatenate(values[name])
        data[f"renderer_{name}_confidence"] = np.concatenate(scores[name])
        if len(data[f"renderer_{name}"]) != len(data["x"]):
            raise RuntimeError(
                f"renderer {name} cache length mismatch: "
                f"{len(data[f'renderer_{name}'])} versus {len(data['x'])}"
            )
    data["failures"] = np.asarray(failures, dtype=str)
    np.savez_compressed(path, **data)
    return data


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def group_split(groups: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique = unique[rng.permutation(len(unique))]
    train_end = max(1, int(round(len(unique) * 0.70)))
    val_end = max(train_end + 1, int(round(len(unique) * 0.85)))
    val_end = min(val_end, len(unique) - 1)
    split = {
        "train": unique[:train_end],
        "val": unique[train_end:val_end],
        "test": unique[val_end:],
    }
    return {name: np.isin(groups, values) for name, values in split.items()}


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(prediction - target))),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
        "r2": float(r2_score(target, prediction, multioutput="variance_weighted")),
        "corr": correlation(prediction, target),
    }


def fit_ridge(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, float]:
    xs = StandardScaler().fit(x[train])
    ys = StandardScaler().fit(y[train])
    x_all, y_all = xs.transform(x), ys.transform(y)
    best_alpha, best_error = 1.0, float("inf")
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(x_all[train], y_all[train])
        error = float(np.mean((model.predict(x_all[val]) - y_all[val]) ** 2))
        if error < best_error:
            best_alpha, best_error = alpha, error
    model = Ridge(alpha=best_alpha).fit(x_all[train | val], y_all[train | val])
    return ys.inverse_transform(model.predict(x_all[test])).astype(np.float32), best_alpha


def fit_mlp(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
    test: np.ndarray, seed: int,
) -> Tuple[np.ndarray, int]:
    xs = StandardScaler().fit(x[train])
    ys = StandardScaler().fit(y[train])
    x_all, y_all = xs.transform(x), ys.transform(y)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 128), activation="relu", solver="adam",
        alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
        max_iter=250, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed,
    )
    model.fit(x_all[train | val], y_all[train | val])
    return ys.inverse_transform(model.predict(x_all[test])).astype(np.float32), int(model.n_iter_)


def classification_metrics(probability: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = probability >= 0.5
    result = {
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "positive_rate": float(target.mean()),
    }
    result["auroc"] = float(roc_auc_score(target, probability)) if len(np.unique(target)) > 1 else 0.0
    return result


def fit_blink_classifier(
    x: np.ndarray, target: np.ndarray, masks: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, float]:
    scaler = StandardScaler().fit(x[masks["train"]])
    values = scaler.transform(x)
    best_c, best_f1 = 1.0, -1.0
    for c in (0.01, 0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
        model.fit(values[masks["train"]], target[masks["train"]])
        score = f1_score(
            target[masks["val"]], model.predict(values[masks["val"]]), zero_division=0,
        )
        if score > best_f1:
            best_c, best_f1 = c, score
    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=1000)
    model.fit(values[masks["train"] | masks["val"]], target[masks["train"] | masks["val"]])
    return model.predict_proba(values[masks["test"]])[:, 1], best_c


def consecutive_indices(data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(data["frame"] > 0)
    previous = indices - 1
    keep = data["clip"][indices] == data["clip"][previous]
    return indices[keep], previous[keep]


def renderer_metrics(
    generated: np.ndarray,
    target: np.ndarray,
    clips: np.ndarray,
    frames: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    generated_mean = generated.mean(axis=1)
    target_mean = target.mean(axis=1)
    gt_blink = target_mean < threshold
    pred_blink = generated_mean < threshold
    cls = classification_metrics(
        pred_blink[valid].astype(np.float32), gt_blink[valid],
    )

    indices = np.flatnonzero(frames > 0)
    previous = indices - 1
    keep = (clips[indices] == clips[previous]) & valid[indices] & valid[previous]
    indices, previous = indices[keep], previous[keep]
    gt_velocity = target[indices] - target[previous]
    pred_velocity = generated[indices] - generated[previous]
    velocity_error = np.abs(pred_velocity - gt_velocity).mean(axis=1)
    frame_error = np.abs(generated - target).mean(axis=1)
    metrics = {
        "openness_mae": float(frame_error[valid].mean()),
        "openness_corr": correlation(generated[valid], target[valid]),
        "velocity_mae": float(velocity_error.mean()),
        "velocity_corr": correlation(pred_velocity, gt_velocity),
        "velocity_amplitude_ratio": float(
            np.abs(pred_velocity).mean() / max(float(np.abs(gt_velocity).mean()), 1e-8)
        ),
        "blink_precision": cls["precision"],
        "blink_recall": cls["recall"],
        "blink_f1": cls["f1"],
        "blink_frames": int(gt_blink[valid].sum()),
        "valid_frames": int(valid.sum()),
    }
    per_clip = {}
    for clip in np.unique(clips):
        mask = (clips == clip) & valid
        vmask = clips[indices] == clip
        per_clip[str(clip)] = np.asarray([
            frame_error[mask].mean() if mask.any() else np.nan,
            velocity_error[vmask].mean() if vmask.any() else np.nan,
        ], dtype=np.float32)
    return metrics, per_clip


def paired_bootstrap(
    first: Dict[str, np.ndarray], second: Dict[str, np.ndarray], seed: int
) -> Dict[str, Dict[str, float]]:
    keys = sorted(set(first) & set(second))
    a = np.stack([first[key] for key in keys])
    b = np.stack([second[key] for key in keys])
    rng = np.random.default_rng(seed)
    result = {}
    for index, name in enumerate(("openness_mae", "velocity_mae")):
        delta = b[:, index] - a[:, index]
        delta = delta[np.isfinite(delta)]
        samples = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        result[name] = {
            "delta_second_minus_first": float(delta.mean()),
            "second_win_rate": float((delta < 0).mean()),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return result


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    renderers = parse_renderers(args.renderer)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "features.npz"
    if cache_path.is_file() and not args.overwrite_cache:
        data = load_cache(cache_path)
        missing = {
            name: directory for name, directory in renderers.items()
            if f"renderer_{name}" not in data
        }
        if missing:
            data = append_renderer_cache(args, data, missing, cache_path)
    else:
        data = build_cache(args, renderers, cache_path)

    valid = data["gt_confidence"] >= args.confidence_threshold
    position_masks = group_split(data["speaker"], args.seed)
    # Reference-relative openness is the exact information a motion delta must carry.
    reference = np.empty_like(data["gt_eye"])
    for clip in np.unique(data["clip"]):
        mask = data["clip"] == clip
        reference[mask] = data["gt_eye"][np.flatnonzero(mask)[0]]
    relative_eye = data["gt_eye"] - reference

    usable = valid.copy()
    masks = {name: value & usable for name, value in position_masks.items()}
    ridge, alpha = fit_ridge(data["x"], relative_eye, masks["train"], masks["val"], masks["test"])
    mlp, iterations = fit_mlp(
        data["x"], relative_eye, masks["train"], masks["val"], masks["test"], args.seed,
    )
    test_target = relative_eye[masks["test"]]
    position_results = {
        "zero": regression_metrics(np.zeros_like(test_target), test_target),
        "ridge": regression_metrics(ridge, test_target),
        "mlp": regression_metrics(mlp, test_target),
    }

    current, previous = consecutive_indices(data)
    velocity_valid = valid[current] & valid[previous]
    current, previous = current[velocity_valid], previous[velocity_valid]
    velocity_x = data["x"][current] - data["x"][previous]
    velocity_y = data["gt_eye"][current] - data["gt_eye"][previous]
    velocity_groups = data["speaker"][current]
    velocity_masks = group_split(velocity_groups, args.seed)
    velocity_ridge, velocity_alpha = fit_ridge(
        velocity_x, velocity_y,
        velocity_masks["train"], velocity_masks["val"], velocity_masks["test"],
    )
    velocity_mlp, velocity_iterations = fit_mlp(
        velocity_x, velocity_y,
        velocity_masks["train"], velocity_masks["val"], velocity_masks["test"], args.seed,
    )
    velocity_target = velocity_y[velocity_masks["test"]]
    velocity_results = {
        "zero": regression_metrics(np.zeros_like(velocity_target), velocity_target),
        "ridge": regression_metrics(velocity_ridge, velocity_target),
        "mlp": regression_metrics(velocity_mlp, velocity_target),
    }

    blink = data["gt_eye"].mean(axis=1) < args.blink_threshold
    # The reference eye state is available to the real renderer, so include it as
    # a fair two-dimensional covariate; compare against reference-only explicitly.
    blink_x = np.concatenate([data["x"], reference], axis=1)
    blink_probability, blink_c = fit_blink_classifier(blink_x, blink, masks)
    reference_probability, reference_c = fit_blink_classifier(reference, blink, masks)
    blink_results = {
        "reference_only": classification_metrics(reference_probability, blink[masks["test"]]),
        "latent_plus_reference": classification_metrics(blink_probability, blink[masks["test"]]),
    }

    renderer_results, per_clip = {}, {}
    for name in renderers:
        renderer_results[name], per_clip[name] = renderer_metrics(
            data[f"renderer_{name}"], data["gt_eye"], data["clip"], data["frame"],
            valid, args.blink_threshold,
        )
    comparisons = {}
    names = list(renderers)
    if len(names) >= 2:
        for first_index in range(len(names) - 1):
            for second_index in range(first_index + 1, len(names)):
                first, second = names[first_index], names[second_index]
                comparisons[f"{first}_vs_{second}"] = paired_bootstrap(
                    per_clip[first], per_clip[second], args.seed,
                )

    result = {
        "checkpoint": args.checkpoint,
        "frames": int(len(data["x"])),
        "clips": int(len(np.unique(data["clip"]))),
        "speakers": int(len(np.unique(data["speaker"]))),
        "failures": data.get("failures", np.asarray([])).tolist(),
        "mean_gt_eye_confidence": float(data["gt_confidence"].mean()),
        "confidence_threshold": args.confidence_threshold,
        "blink_threshold": args.blink_threshold,
        "blink_frames": int(blink.sum()),
        "blink_rate": float(blink.mean()),
        "position": position_results,
        "velocity": velocity_results,
        "blink_classification": blink_results,
        "renderer_oracle": renderer_results,
        "renderer_comparison": comparisons,
        "metadata": {
            "ridge_alpha": alpha,
            "mlp_iterations": iterations,
            "velocity_ridge_alpha": velocity_alpha,
            "velocity_mlp_iterations": velocity_iterations,
            "blink_logistic_c": blink_c,
            "reference_logistic_c": reference_c,
            "test_frames": int(masks["test"].sum()),
            "velocity_test_pairs": int(velocity_masks["test"].sum()),
        },
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# 42k blink latent probe and causal oracle", "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- frames/clips/speakers: {result['frames']}/{result['clips']}/{result['speakers']}",
        f"- valid GT eye confidence mean: {result['mean_gt_eye_confidence']:.4f}",
        f"- eye confidence threshold: {args.confidence_threshold:.2f}",
        f"- blink threshold/rate/frames: {args.blink_threshold:.2f}/{result['blink_rate']:.4f}/{result['blink_frames']}",
        f"- extraction failures: {len(result['failures'])}", "",
        "## Speaker-disjoint latent regression", "",
        "| task | method | MAE ↓ | RMSE ↓ | R² ↑ | corr ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task, values in (("relative openness", position_results), ("openness velocity", velocity_results)):
        for method, metric in values.items():
            lines.append(
                f"| {task} | {method} | {metric['mae']:.6f} | {metric['rmse']:.6f} | "
                f"{metric['r2']:.4f} | {metric['corr']:.4f} |"
            )
    lines.extend([
        "", "## Speaker-disjoint blink classification", "",
        "| method | precision ↑ | recall ↑ | F1 ↑ | AUROC ↑ |",
        "|---|---:|---:|---:|---:|",
    ])
    for method, metric in blink_results.items():
        lines.append(
            f"| {method} | {metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{metric['f1']:.4f} | {metric['auroc']:.4f} |"
        )
    lines.extend([
        "", "## Renderer blink oracle (all clips)", "",
        "| renderer | openness MAE ↓ | velocity MAE ↓ | velocity corr ↑ | amplitude/GT | blink P/R/F1 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, metric in renderer_results.items():
        lines.append(
            f"| {name} | {metric['openness_mae']:.6f} | {metric['velocity_mae']:.6f} | "
            f"{metric['velocity_corr']:.4f} | {metric['velocity_amplitude_ratio']:.4f} | "
            f"{metric['blink_precision']:.3f}/{metric['blink_recall']:.3f}/{metric['blink_f1']:.3f} |"
        )
    if comparisons:
        lines.extend(["", "## Paired clip bootstrap", "", "```json", json.dumps(comparisons, indent=2), "```"])
    report = "\n".join(lines) + "\n"
    (output / "report.md").write_text(report)
    print(report)
    print(f"cache={cache_path}")
    print(f"metrics={output / 'metrics.json'}")


if __name__ == "__main__":
    main()
