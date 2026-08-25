#!/usr/bin/env python
"""Download and prepare the frozen VGG16/ArcFace training backbones.

The ArcFace model in InsightFace's official ``buffalo_l`` release is ONNX.
ONNXRuntime cannot propagate gradients to generated pixels, so this utility
converts it once to a frozen TorchScript graph.  Training only needs the
resulting TorchScript file; onnx/onnx2torch are preparation-time dependencies.

InsightFace pretrained models are restricted to non-commercial research use.
See https://github.com/deepinsight/insightface/tree/master/model_zoo.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

import torch


VGG_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
VGG_SHA256 = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
BUFFALO_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
BUFFALO_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
ARCFACE_MEMBER = "w600k_r50.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/motion_tokenizer")
    parser.add_argument(
        "--install-converter-deps",
        action="store_true",
        help="install onnx/onnx2torch into OUTPUT_DIR/python_deps if unavailable",
    )
    parser.add_argument("--force-convert", action="store_true")
    return parser.parse_args()


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checked_download(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.is_file() and sha256(destination) == expected_sha256:
        print(f"verified {destination}")
        return
    if destination.exists():
        raise RuntimeError(
            f"existing file has the wrong SHA256: {destination}; remove it and retry"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, partial)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink()
        raise RuntimeError(f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual}")
    partial.replace(destination)


def install_converter_dependencies(output_dir: Path) -> Path:
    dependency_dir = output_dir / "python_deps"
    dependency_dir.mkdir(parents=True, exist_ok=True)
    pip_tmp = output_dir / "pip_tmp"
    pip_tmp.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["TMPDIR"] = str(pip_tmp.resolve())
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(dependency_dir),
            "--no-cache-dir",
            "--no-deps",
            "onnx==1.18.0",
            "onnx2torch==1.5.15",
        ],
        env=environment,
    )
    return dependency_dir


def load_converter_dependencies(output_dir: Path, install: bool):
    dependency_dir = output_dir / "python_deps"
    if dependency_dir.is_dir():
        sys.path.insert(0, str(dependency_dir.resolve()))
    try:
        onnx = importlib.import_module("onnx")
        convert = importlib.import_module("onnx2torch").convert
        return onnx, convert
    except (ImportError, AttributeError) as error:
        if not install:
            raise RuntimeError(
                "ArcFace conversion needs onnx==1.18.0 and onnx2torch==1.5.15. "
                "Rerun with --install-converter-deps; they are installed only "
                "under the output directory and are not needed during training."
            ) from error
    dependency_dir = install_converter_dependencies(output_dir)
    sys.path.insert(0, str(dependency_dir.resolve()))
    onnx = importlib.import_module("onnx")
    convert = importlib.import_module("onnx2torch").convert
    return onnx, convert


def validate_vgg(path: Path) -> None:
    from torchvision.models import vgg16

    state = torch.load(path, map_location="cpu", weights_only=True)
    model = vgg16(weights=None)
    model.load_state_dict(state, strict=True)
    print(f"VGG16 strict state-dict validation passed ({len(state)} tensors)")


def convert_arcface(onnx_path: Path, output_path: Path, output_dir: Path, install: bool) -> None:
    onnx, convert = load_converter_dependencies(output_dir, install)
    graph = onnx.load(str(onnx_path))
    onnx.checker.check_model(graph)
    model = convert(graph).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # RGB inputs produced by the dataset already use ArcFace's expected
    # (pixel - 127.5) / 127.5 range, i.e. [-1, 1].
    generator = torch.Generator().manual_seed(2026)
    example = torch.rand(1, 3, 112, 112, generator=generator).mul_(2).sub_(1)
    with torch.no_grad():
        expected = model(example)
        traced = torch.jit.trace(model, example, strict=False).eval()
        actual = traced(example)
    if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-4):
        error = (actual - expected).abs().max().item()
        raise RuntimeError(f"TorchScript conversion changed ArcFace output (max error {error})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output_path))

    # Do not call torch.jit.freeze(): that would inline weights as CPU constants,
    # after which an enclosing loss module cannot move them with .to(cuda).
    # requires_grad=False freezes parameters without sacrificing device moves.
    # Verify dynamic batch behavior and, critically, a gradient all the way to
    # input pixels. Frozen parameters do not prevent input gradients.
    loaded = torch.jit.load(str(output_path), map_location="cpu").eval()
    batch = torch.rand(2, 3, 112, 112, generator=generator).mul_(2).sub_(1)
    batch.requires_grad_(True)
    embeddings = loaded(batch)
    if embeddings.shape != (2, 512):
        raise RuntimeError(f"unexpected ArcFace output shape: {tuple(embeddings.shape)}")
    embeddings.float().square().mean().backward()
    if batch.grad is None or not torch.isfinite(batch.grad).all() or batch.grad.abs().sum() == 0:
        raise RuntimeError("ArcFace TorchScript does not propagate a valid input gradient")
    print(f"ArcFace TorchScript and input-gradient validation passed: {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    download_dir = output_dir / "downloads"
    vgg_path = output_dir / "vgg16-397923af.pth"
    archive_path = download_dir / "buffalo_l.zip"
    onnx_path = download_dir / ARCFACE_MEMBER
    arcface_path = output_dir / "arcface_w600k_r50.ts"

    checked_download(VGG_URL, vgg_path, VGG_SHA256)
    checked_download(BUFFALO_URL, archive_path, BUFFALO_SHA256)
    validate_vgg(vgg_path)
    if not onnx_path.exists():
        download_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            if ARCFACE_MEMBER not in archive.namelist():
                raise RuntimeError(f"{ARCFACE_MEMBER} is absent from {archive_path}")
            archive.extract(ARCFACE_MEMBER, download_dir)

    if args.force_convert or not arcface_path.exists():
        convert_arcface(
            onnx_path, arcface_path, output_dir,
            install=bool(args.install_converter_deps),
        )
    else:
        print(f"using existing {arcface_path}")

    metadata = {
        "vgg_path": str(vgg_path),
        "vgg_sha256": sha256(vgg_path),
        "arcface_path": str(arcface_path),
        "arcface_sha256": sha256(arcface_path),
        "arcface_source": BUFFALO_URL + "#" + ARCFACE_MEMBER,
        "arcface_input": "RGB float32 in [-1, 1], NCHW, resized to 112x112 by loss",
        "arcface_output": "512-D embedding",
        "license_note": "InsightFace pretrained models: non-commercial research only",
    }
    metadata_path = output_dir / "weights.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
