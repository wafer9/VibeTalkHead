"""JSONL-backed continuous video clip dataset for motion-tokenizer training."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from decord import VideoReader, cpu
except ImportError as exc:  # pragma: no cover - gives a useful startup error
    raise ImportError("decord is required; use the existing `vibe` conda environment") from exc


DEFAULT_VIDEO_ROOT = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards"


def build_jsonl_index(manifest: str, index_path: Optional[str] = None) -> str:
    """Build uint64 byte offsets without loading a 600 MB manifest into RAM."""
    index_path = index_path or manifest + ".idx.npy"
    manifest_stat = os.stat(manifest)
    if os.path.isfile(index_path) and os.path.getmtime(index_path) >= manifest_stat.st_mtime:
        return index_path

    offsets = []
    offset = 0
    with open(manifest, "rb") as stream:
        for line in stream:
            if line.strip():
                offsets.append(offset)
            offset += len(line)
    target = Path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = str(target) + f".tmp.{os.getpid()}"
    with open(temporary, "wb") as stream:
        np.save(stream, np.asarray(offsets, dtype=np.uint64))
    os.replace(temporary, target)
    return str(target)


def resolve_video_path(path: str, video_root: str = DEFAULT_VIDEO_ROOT) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(video_root, path)


def _read_record(manifest: str, offsets: np.ndarray, index: int) -> Dict[str, Any]:
    with open(manifest, "rb") as stream:
        stream.seek(int(offsets[index]))
        return json.loads(stream.readline())


def _resize_video(frames: torch.Tensor, image_size: int) -> torch.Tensor:
    # [T,H,W,3] uint8 -> [T,3,S,S] float in [0,1]
    frames = frames.permute(0, 3, 1, 2).float().div_(255.0)
    if frames.shape[-2:] != (image_size, image_size):
        frames = F.interpolate(
            frames, size=(image_size, image_size), mode="bilinear",
            align_corners=False, antialias=True,
        )
    return frames


def _photometric(frames: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return frames
    device, dtype = frames.device, frames.dtype
    brightness = 1.0 + (torch.rand((), device=device, dtype=dtype) * 2 - 1) * strength
    contrast = 1.0 + (torch.rand((), device=device, dtype=dtype) * 2 - 1) * strength
    saturation = 1.0 + (torch.rand((), device=device, dtype=dtype) * 2 - 1) * strength
    gray = frames.mean(dim=1, keepdim=True)
    frames = gray + saturation * (frames - gray)
    channel_mean = frames.mean(dim=(-2, -1), keepdim=True)
    frames = channel_mean + contrast * (frames - channel_mean)
    return (frames * brightness).clamp_(0, 1)


class JsonlVideoClipDataset(Dataset):
    """Random continuous clips with a fixed same-video reference.

    Invalid/corrupt records are retried locally so one damaged video does not
    terminate a multi-day distributed job.
    """

    def __init__(
        self,
        manifest: str,
        *,
        video_root: str = DEFAULT_VIDEO_ROOT,
        index_path: Optional[str] = None,
        clip_length: int = 16,
        image_size: int = 256,
        target_fps: float = 25.0,
        sampling_mode: str = "clip",
        training: bool = True,
        first_frame_reference_prob: float = 0.5,
        photometric_strength: float = 0.12,
        horizontal_flip_prob: float = 0.5,
        return_cross_reference: bool = False,
        max_retries: int = 12,
        max_samples: int = 0,
    ):
        self.manifest = os.path.abspath(manifest)
        self.video_root = video_root
        self.index_path = build_jsonl_index(self.manifest, index_path)
        self.offsets = np.load(self.index_path, mmap_mode="r")
        self.clip_length = int(clip_length)
        self.image_size = int(image_size)
        self.target_fps = float(target_fps)
        self.sampling_mode = str(sampling_mode)
        if self.sampling_mode not in {"clip", "random_pair"}:
            raise ValueError(
                f"sampling_mode must be 'clip' or 'random_pair', got {self.sampling_mode!r}"
            )
        if self.sampling_mode == "random_pair" and self.clip_length != 1:
            raise ValueError("random_pair sampling requires clip_length=1")
        self.training = bool(training)
        self.first_frame_reference_prob = float(first_frame_reference_prob)
        self.photometric_strength = float(photometric_strength)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.return_cross_reference = bool(return_cross_reference)
        self.max_retries = int(max_retries)
        self.max_samples = min(max_samples, len(self.offsets)) if max_samples > 0 else len(self.offsets)

    def __len__(self) -> int:
        return self.max_samples

    def _record(self, index: int) -> Dict[str, Any]:
        return _read_record(self.manifest, self.offsets, index % len(self.offsets))

    def _sample_indices(self, length: int, fps: float) -> tuple[np.ndarray, int]:
        if self.sampling_mode == "random_pair":
            if length < 2:
                raise ValueError("video too short for a distinct source/target pair")
            if self.training:
                reference, target = random.sample(range(length), 2)
            else:
                reference, target = 0, length // 2
                if target == reference:
                    target = length - 1
            return np.asarray([target], dtype=np.int64), reference

        stride = max(fps / self.target_fps, 1.0)
        span = int(round((self.clip_length - 1) * stride)) + 1
        if length < span:
            raise ValueError(f"video too short: {length} frames for span {span}")
        max_start = length - span
        start = random.randint(0, max_start) if self.training else max_start // 2
        target = np.rint(start + np.arange(self.clip_length) * stride).astype(np.int64)
        target = np.clip(target, 0, length - 1)

        if not self.training or random.random() < self.first_frame_reference_prob:
            reference = 0
        else:
            before = list(range(0, max(0, int(target[0]) - 1)))
            after = list(range(min(length, int(target[-1]) + 2), length))
            candidates = before + after
            reference = random.choice(candidates) if candidates else 0
        return target, reference

    def _decode_clip(
        self, record: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, str, bool]:
        path = resolve_video_path(record["video_path"], self.video_root)
        reader = VideoReader(path, ctx=cpu(0), num_threads=2)
        fps = float(record.get("fps") or reader.get_avg_fps() or self.target_fps)
        target_indices, reference_index = self._sample_indices(len(reader), fps)
        all_indices = np.concatenate([[reference_index], target_indices])
        decoded = torch.from_numpy(reader.get_batch(all_indices).asnumpy())
        decoded = _resize_video(decoded, self.image_size)
        reference, frames = decoded[:1], decoded[1:]

        if self.training:
            # Geometry and photometric augmentation are shared across the whole
            # sample.  Independent reference/target color jitter would force the
            # only target-side path (motion delta) to carry brightness and color.
            if random.random() < self.horizontal_flip_prob:
                reference = reference.flip(-1)
                frames = frames.flip(-1)
            augmented = _photometric(
                torch.cat([reference, frames], dim=0), self.photometric_strength
            )
            reference, frames = augmented[:1], augmented[1:]
        return (
            reference[0].mul(2).sub(1), frames.mul(2).sub(1), path,
            reference_index == 0,
        )

    def _decode_single_reference(self, index: int) -> torch.Tensor:
        record = self._record(index)
        path = resolve_video_path(record["video_path"], self.video_root)
        reader = VideoReader(path, ctx=cpu(0), num_threads=1)
        if len(reader) < 1:
            raise ValueError("empty cross-reference video")
        frame_index = random.randrange(len(reader)) if self.training else 0
        frame = torch.from_numpy(reader.get_batch([frame_index]).asnumpy())
        frame = _resize_video(frame, self.image_size)
        if self.training:
            frame = _photometric(frame, self.photometric_strength)
        return frame[0].mul(2).sub(1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        candidate = int(index)
        for _ in range(self.max_retries):
            try:
                record = self._record(candidate)
                reference, frames, path, reference_is_first = self._decode_clip(record)
                sample: Dict[str, Any] = {
                    "key": record.get("sample_id", str(candidate)),
                    "source": record.get("dataset_source", "unknown"),
                    "video_path": path,
                    "reference": reference,
                    "reference_is_first": reference_is_first,
                    "frames": frames,
                }
                if self.return_cross_reference:
                    cross_index = random.randrange(len(self.offsets))
                    if cross_index == candidate:
                        cross_index = (cross_index + 1) % len(self.offsets)
                    sample["cross_reference"] = self._decode_single_reference(cross_index)
                return sample
            except Exception as exc:
                last_error = exc
                candidate = random.randrange(len(self.offsets))
        raise RuntimeError(
            f"failed to decode a valid clip after {self.max_retries} attempts; "
            f"last error: {last_error!r}"
        )


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)
