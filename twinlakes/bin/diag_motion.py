# Copyright (c) Kyutai, all rights reserved.
"""诊断: 模型生成的 motion 到底怎么了 —— 不渲染, 只打统计。

回答两个问题:
  1) 模型输出 motion 在归一化空间的 std 是多少? (训练目标 ≈1; 远小于1 = 条件坍缩到均值)
  2) 换不同音频, 输出 motion 变不变? (不变 = 模型忽略 condition; 变 = condition 有用)
  3) 同音频、不同初始噪声两次, 差多少? (大 = 条件分布宽/未收敛)
  4) 相邻帧差 (frame-diff ratio), 对比 GT 基线 ~0.29

用法:
  CUDA_VISIBLE_DEVICES=0 python -m twinlakes.bin.diag_motion \
      --config exp/s3/train.yaml --checkpoint exp/s3/9_140000.pt \
      --test_data data/talker/dev.json --num_steps 20 --limit 3
"""
import argparse
import json
import os

import numpy as np
import torch

from twinlakes.models.rq_transformer import LMModel
from twinlakes.utils.checkpoint import load_checkpoint
from twinlakes.dataset.processor import MOTION_MEAN, MOTION_STD
from twinlakes.bin.infer_video import (
    load_wav_16k, audio_to_condition, denoise_motion, SHARDS_ROOT, _resolve,
)


def _stats(name, m):
    """m: (T,40) tensor 归一化空间."""
    m = m.float()
    T = m.shape[0]
    per_dim_std = m.std(dim=0)                       # 每维时间方向 std
    global_std = m.std().item()
    mean_abs = m.abs().mean().item()
    # frame-diff ratio: mean|Δ帧| / mean(per-dim std over time)
    if T > 1:
        fd = (m[1:] - m[:-1]).abs().mean().item()
        denom = per_dim_std.mean().item() + 1e-8
        fd_ratio = fd / denom
    else:
        fd_ratio = float("nan")
    print(f"  [{name}] T={T}  global_std={global_std:.4f}  mean|x|={mean_abs:.4f}  "
          f"per-dim std [{per_dim_std.min():.3f},{per_dim_std.max():.3f}]  "
          f"frame-diff-ratio={fd_ratio:.3f}")
    return global_std, fd_ratio


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--limit", type=int, default=3)
    p.add_argument("--shards_root", default=SHARDS_ROOT)
    p.add_argument("--max_seconds", type=float, default=6.0)
    return p.parse_args()


def main():
    args = get_args()
    import yaml
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    n_mels = configs["dataset_conf"]["fbank_conf"]["num_mel_bins"]

    model = LMModel.from_audio_text_pretrained(configs)
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda:0")
    model = model.to(device).eval()

    with open(args.test_data) as f:
        lines = [l.strip() for l in f if l.strip()][: args.limit]

    print("=" * 90)
    print("训练目标: motion 归一化后 global_std ≈ 1.0, per-dim std ≈ 1.0; GT frame-diff-ratio ≈ 0.29")
    print("=" * 90)

    conds = []
    for i, line in enumerate(lines):
        obj = json.loads(line)
        key = obj.get("sample_id", obj.get("key", f"s{i}"))
        wav_path = _resolve(args.shards_root, obj.get("wav_path") or obj.get("audio"))
        wav = load_wav_16k(wav_path, args.max_seconds)
        cond = audio_to_condition(model, wav, device, n_mels=n_mels)  # (T,1280)
        conds.append((key, cond))
        c = cond.float()
        print(f"\n[{key}] condition: shape={tuple(cond.shape)} "
              f"global_std={c.std().item():.4f} "
              f"frame-diff={((c[1:]-c[:-1]).abs().mean()/(c.std()+1e-8)).item():.4f}")

        torch.manual_seed(0)
        m0 = denoise_motion(model, cond, args.num_steps, device, shared_noise=False)
        _stats("run0 (seed0)", m0)
        torch.manual_seed(1)
        m1 = denoise_motion(model, cond, args.num_steps, device, shared_noise=False)
        _stats("run1 (seed1)", m1)
        diff = (m0 - m1).abs().mean().item()
        rel = diff / (m0.abs().mean().item() + 1e-8)
        print(f"  >> 同音频不同噪声: mean|Δ|={diff:.4f}  相对={rel:.2f}  "
              f"(小=收敛稳定, 大=条件分布宽)")

    # 换音频对比: 用第一条的噪声种子, 跑不同 condition, 看输出是否变
    if len(conds) >= 2:
        print("\n" + "=" * 90)
        print("换音频测试 (固定 seed=0, 只换 condition): 若输出几乎不变 = 模型忽略 condition")
        print("=" * 90)
        outs = []
        Tmin = min(c.shape[0] for _, c in conds)
        for key, cond in conds:
            torch.manual_seed(0)
            m = denoise_motion(model, cond, args.num_steps, device, shared_noise=False)
            outs.append((key, m[:Tmin]))
            _stats(f"audio={key}", m[:Tmin])
        base_key, base = outs[0]
        for key, m in outs[1:]:
            d = (m - base).abs().mean().item()
            r = d / (base.abs().mean().item() + 1e-8)
            print(f"  >> {key} vs {base_key}: mean|Δ|={d:.4f}  相对={r:.2f}  "
                  f"(接近0=忽略音频; 大=音频驱动有效)")


if __name__ == "__main__":
    with torch.no_grad():
        main()
