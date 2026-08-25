#!/usr/bin/env python
"""Probe whether a frozen motion latent retains mouth geometry and velocity.

The script extracts the exact reference-relative delta consumed by the
causal-strength=0 renderer, obtains 68-point FAN landmarks from GT and an
existing oracle reconstruction, then fits speaker-disjoint Ridge and MLP
probes.  Extraction is cached so probe fitting can be repeated cheaply.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from twinlakes.motion_tokenizer.data import resolve_video_path
from twinlakes.motion_tokenizer.model import MotionTokenizer


MOUTH = np.arange(48, 68)
STABLE = np.concatenate([np.arange(27, 36), np.arange(36, 48)])


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--renderer_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--video_root", default="/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards"
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--encode_batch", type=int, default=64)
    parser.add_argument("--fan_batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument(
        "--face_alignment_path",
        default=str(root / "checkpoints/motion_tokenizer/probe_deps"),
    )
    parser.add_argument(
        "--torch_home",
        default=str(root / "checkpoints/motion_tokenizer/probe_weights"),
    )
    return parser.parse_args()


def read_records(path: str, limit: int) -> list[dict]:
    with open(path) as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    return records[:limit] if limit > 0 else records


def speaker_id(sample_id: str) -> str:
    parts = sample_id.rsplit("_", 3)
    return parts[0] if len(parts) == 4 else sample_id


def video_frames(path: str, fps: float, resolution: int) -> np.ndarray:
    reader = VideoReader(path, ctx=cpu(0), num_threads=4)
    source_fps = float(reader.get_avg_fps() or fps)
    indices = np.rint(np.arange(0, len(reader), source_fps / fps)).astype(np.int64)
    indices = np.unique(np.clip(indices, 0, len(reader) - 1))
    frames = reader.get_batch(indices).asnumpy()
    if frames.shape[1:3] != (resolution, resolution):
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(
            tensor, size=(resolution, resolution), mode="bilinear",
            align_corners=False, antialias=True,
        )
        frames = tensor.clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
    return frames


@torch.inference_mode()
def encode_delta(
    model: MotionTokenizer, frames: np.ndarray, batch_size: int, device: torch.device
) -> np.ndarray:
    motion = []
    for start in range(0, len(frames), batch_size):
        tensor = torch.from_numpy(frames[start:start + batch_size]).to(device)
        tensor = tensor.permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            motion.append(model.encode_motion(tensor).float().cpu())
    target = torch.cat(motion)
    return (target - target[:1]).numpy().astype(np.float32)


def largest_face(boxes: Iterable[np.ndarray]) -> np.ndarray:
    boxes = list(boxes)
    if not boxes:
        raise RuntimeError("FAN detector found no face in reference frame")
    return max(boxes, key=lambda b: float((b[2] - b[0]) * (b[3] - b[1])))


@torch.inference_mode()
def fan_landmarks_batch(fan, frames: np.ndarray, box: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    # Use one GT-reference box for the full clip and its reconstruction.  This
    # removes per-frame detector jitter from the velocity target/comparison.
    from face_alignment.utils import crop, get_preds_fromhm

    center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    center[1] -= (box[3] - box[1]) * 0.12
    scale = (box[2] - box[0] + box[3] - box[1]) / fan.face_detector.reference_scale
    crops = np.stack([crop(frame, center, scale) for frame in frames])
    landmarks, scores = [], []
    for start in range(0, len(crops), batch_size):
        x = torch.from_numpy(crops[start:start + batch_size]).permute(0, 3, 1, 2)
        x = x.to(fan.device, dtype=fan.dtype).div_(255.0)
        heatmaps = fan.face_alignment_net(x)
        if isinstance(heatmaps, list):
            heatmaps = heatmaps[-1]
        heatmaps = heatmaps.float().cpu().numpy()
        points, _, confidence = get_preds_fromhm(heatmaps)
        landmarks.append(points * 4.0)
        scores.append(confidence)
    return np.concatenate(landmarks).astype(np.float32), np.concatenate(scores).astype(np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((*points.shape[:-1], 1), dtype=points.dtype)
    return np.concatenate([points, ones], axis=-1) @ matrix.T


def align_landmarks(sequence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    aligned = np.empty_like(sequence)
    for index, points in enumerate(sequence):
        matrix, _ = cv2.estimateAffinePartial2D(
            points[STABLE], reference[STABLE], method=cv2.LMEDS,
        )
        if matrix is None:
            matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        aligned[index] = transform_points(points, matrix.astype(np.float32))
    return aligned


def geometry_targets(gt: np.ndarray, generated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    reference = gt[0]
    gt = align_landmarks(gt, reference)
    generated = align_landmarks(generated, reference)
    left_eye = reference[36:42].mean(axis=0)
    right_eye = reference[42:48].mean(axis=0)
    interocular = max(float(np.linalg.norm(left_eye - right_eye)), 1.0)
    reference_mouth = reference[MOUTH]
    reference_open = float(np.linalg.norm(reference[62] - reference[66])) / interocular

    def targets(points: np.ndarray) -> np.ndarray:
        mouth = (points[:, MOUTH] - reference_mouth[None]) / interocular
        openness = np.linalg.norm(points[:, 62] - points[:, 66], axis=-1) / interocular
        openness = openness - reference_open
        return np.concatenate([mouth.reshape(len(points), -1), openness[:, None]], axis=1)

    return targets(gt).astype(np.float32), targets(generated).astype(np.float32)


def build_cache(args: argparse.Namespace, cache_path: Path) -> Dict[str, np.ndarray]:
    sys.path.insert(0, os.path.abspath(args.face_alignment_path))
    os.environ["TORCH_HOME"] = os.path.abspath(args.torch_home)
    import face_alignment

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = MotionTokenizer(**checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()
    fan = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        device=str(device), flip_input=False, compile=False,
    )

    items: Dict[str, list] = {
        "x": [], "gt": [], "generated": [], "clip": [], "speaker": [],
        "frame": [], "gt_score": [], "generated_score": [],
    }
    failures = []
    records = read_records(args.manifest, args.limit)
    for record in tqdm(records, desc="latent/FAN extraction"):
        key = record.get("sample_id") or Path(record["video_path"]).stem
        generated_path = Path(args.renderer_dir) / f"{key}.mp4"
        try:
            gt_path = resolve_video_path(record["video_path"], args.video_root)
            if not generated_path.is_file():
                raise FileNotFoundError(generated_path)
            gt_frames = video_frames(gt_path, args.fps, args.resolution)
            generated_frames = video_frames(str(generated_path), args.fps, args.resolution)
            count = min(len(gt_frames), len(generated_frames))
            gt_frames, generated_frames = gt_frames[:count], generated_frames[:count]
            delta = encode_delta(model, gt_frames, args.encode_batch, device)
            box = largest_face(fan.face_detector.detect_from_image(gt_frames[0].copy()))
            gt_landmarks, gt_scores = fan_landmarks_batch(
                fan, gt_frames, box, args.fan_batch,
            )
            generated_landmarks, generated_scores = fan_landmarks_batch(
                fan, generated_frames, box, args.fan_batch,
            )
            gt_target, generated_target = geometry_targets(gt_landmarks, generated_landmarks)
            items["x"].append(delta)
            items["gt"].append(gt_target)
            items["generated"].append(generated_target)
            items["clip"].append(np.repeat(key, count))
            items["speaker"].append(np.repeat(speaker_id(key), count))
            items["frame"].append(np.arange(count, dtype=np.int32))
            items["gt_score"].append(gt_scores[:, MOUTH].mean(axis=1))
            items["generated_score"].append(generated_scores[:, MOUTH].mean(axis=1))
        except Exception as error:
            failures.append(f"{key}: {type(error).__name__}: {error}")

    if not items["x"]:
        raise RuntimeError("no clips were extracted: " + "; ".join(failures[:5]))
    arrays = {key: np.concatenate(value) for key, value in items.items()}
    arrays["failures"] = np.asarray(failures, dtype=str)
    np.savez_compressed(cache_path, **arrays)
    return arrays


def load_cache(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def group_split(groups: np.ndarray, seed: int) -> Dict[str, np.ndarray]:
    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    unique = unique[rng.permutation(len(unique))]
    train_end = max(1, int(round(len(unique) * 0.70)))
    val_end = max(train_end + 1, int(round(len(unique) * 0.85)))
    val_end = min(val_end, len(unique) - 1)
    split_groups = {
        "train": unique[:train_end],
        "val": unique[train_end:val_end],
        "test": unique[val_end:],
    }
    return {name: np.isin(groups, values) for name, values in split_groups.items()}


def fit_ridge(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray
) -> Tuple[np.ndarray, float]:
    x_scaler = StandardScaler().fit(x[train])
    y_scaler = StandardScaler().fit(y[train])
    xs = x_scaler.transform(x)
    ys = y_scaler.transform(y)
    best_alpha, best_error = None, float("inf")
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(xs[train], ys[train])
        error = float(np.mean((model.predict(xs[val]) - ys[val]) ** 2))
        if error < best_error:
            best_alpha, best_error = alpha, error
    model = Ridge(alpha=best_alpha).fit(xs[train | val], ys[train | val])
    return y_scaler.inverse_transform(model.predict(xs[test])).astype(np.float32), float(best_alpha)


def fit_mlp(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, test: np.ndarray, seed: int
) -> Tuple[np.ndarray, int]:
    x_scaler = StandardScaler().fit(x[train])
    y_scaler = StandardScaler().fit(y[train])
    xs = x_scaler.transform(x)
    ys = y_scaler.transform(y)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 128), activation="relu", solver="adam",
        alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
        max_iter=250, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=seed,
    )
    model.fit(xs[train | val], ys[train | val])
    prediction = y_scaler.inverse_transform(model.predict(xs[test]))
    return prediction.astype(np.float32), int(model.n_iter_)


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    mouth_prediction = prediction[:, :40].reshape(-1, 20, 2)
    mouth_target = target[:, :40].reshape(-1, 20, 2)
    return {
        "mouth_nme": float(np.linalg.norm(mouth_prediction - mouth_target, axis=-1).mean()),
        "openness_rmse": float(np.sqrt(np.mean((prediction[:, 40] - target[:, 40]) ** 2))),
        "openness_r2": float(r2_score(target[:, 40], prediction[:, 40])),
        "openness_corr": correlation(target[:, 40], prediction[:, 40]),
    }


def evaluate_task(
    name: str, x: np.ndarray, gt: np.ndarray, generated: np.ndarray,
    groups: np.ndarray, seed: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    masks = group_split(groups, seed)
    ridge, alpha = fit_ridge(x, gt, masks["train"], masks["val"], masks["test"])
    mlp, iterations = fit_mlp(x, gt, masks["train"], masks["val"], masks["test"], seed)
    target = gt[masks["test"]]
    results = {
        "zero": metrics(np.zeros_like(target), target),
        "renderer": metrics(generated[masks["test"]], target),
        "ridge": metrics(ridge, target),
        "mlp": metrics(mlp, target),
    }
    metadata = {
        "train_frames": int(masks["train"].sum()),
        "val_frames": int(masks["val"].sum()),
        "test_frames": int(masks["test"].sum()),
        "ridge_alpha": alpha,
        "mlp_iterations": iterations,
    }
    return results, metadata


def markdown_report(
    args: argparse.Namespace, data: Dict[str, np.ndarray], results: Dict, metadata: Dict
) -> str:
    failures = data.get("failures", np.asarray([], dtype=str))
    clips = np.unique(data["clip"])
    speakers = np.unique(data["speaker"])
    lines = [
        "# 35k motion-latent mouth probe", "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- frames/clips/speakers: {len(data['x'])}/{len(clips)}/{len(speakers)}",
        f"- extraction failures: {len(failures)}",
        f"- mean FAN mouth confidence (GT/renderer): "
        f"{data['gt_score'].mean():.4f}/{data['generated_score'].mean():.4f}", "",
    ]
    for task, label in (("position", "Reference-relative mouth position"),
                        ("velocity", "Mouth velocity")):
        lines.extend([
            f"## {label}", "",
            "| method | mouth NME ↓ | openness RMSE ↓ | openness R² ↑ | openness corr ↑ |",
            "|---|---:|---:|---:|---:|",
        ])
        for method in ("zero", "renderer", "ridge", "mlp"):
            value = results[task][method]
            lines.append(
                f"| {method} | {value['mouth_nme']:.5f} | {value['openness_rmse']:.5f} | "
                f"{value['openness_r2']:.4f} | {value['openness_corr']:.4f} |"
            )
        lines.extend(["", f"metadata: `{json.dumps(metadata[task], ensure_ascii=False)}`", ""])
    if len(failures):
        lines.extend(["## Failures", "", *[f"- {item}" for item in failures.tolist()]])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "features.npz"
    if cache_path.is_file() and not args.overwrite_cache:
        data = load_cache(cache_path)
    else:
        data = build_cache(args, cache_path)

    position_x = data["x"]
    position_gt = data["gt"]
    position_generated = data["generated"]
    position_groups = data["speaker"]
    position_results, position_meta = evaluate_task(
        "position", position_x, position_gt, position_generated, position_groups, args.seed,
    )

    consecutive = (data["frame"] > 0)
    indices = np.flatnonzero(consecutive)
    previous = indices - 1
    same_clip = data["clip"][indices] == data["clip"][previous]
    indices, previous = indices[same_clip], previous[same_clip]
    velocity_x = data["x"][indices] - data["x"][previous]
    velocity_gt = data["gt"][indices] - data["gt"][previous]
    velocity_generated = data["generated"][indices] - data["generated"][previous]
    velocity_groups = data["speaker"][indices]
    velocity_results, velocity_meta = evaluate_task(
        "velocity", velocity_x, velocity_gt, velocity_generated, velocity_groups, args.seed,
    )

    results = {"position": position_results, "velocity": velocity_results}
    metadata = {"position": position_meta, "velocity": velocity_meta}
    report = markdown_report(args, data, results, metadata)
    (output_dir / "report.md").write_text(report)
    (output_dir / "metrics.json").write_text(json.dumps(
        {"results": results, "metadata": metadata}, indent=2, ensure_ascii=False,
    ) + "\n")
    print(report)
    print(f"cache={cache_path}")
    print(f"report={output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
