"""批量: 从 vivi (VividHead) 的 train.list 逐条视频提取 LIA-X 40 维 raw motion,
存成与 talker/train.json 完全一致的 (T,40) fp16 .pt 格式, 并写出 train_liax.json。

提取核心复用 Talker-T2AV/lia_x_recon.py:
  motion = get_alpha(frame) = enc_r2t(enc_2r(frame)[0])   # 与 vibehead infer_video 同路径
  存 raw(未归一化)codes, 与 talker 的 motion .pt 口径一致 (归一化在训练/推理时再做)。

vivi 视频已是 512x512 @25fps, 无需 resize/resample (走快路径, 仍按 target_fps=25 保险)。

输出:
  .pt   -> <out_root>/motion/<key>.pt        (T,40) fp16, T=帧数@25fps
  json  -> data/vivi/train_liax.json         每行一个样本, 字段对齐 talker/train.json

断点续跑: 已存在且可正常 load 的 .pt 跳过。json 每 N 条 flush 一次(崩溃可从 .pt 重建)。

用法 (vibe 环境):
    CUDA_VISIBLE_DEVICES=6 python extract_vivi_liax.py \
        --list data/vivi/train.list \
        --out_root /nfs-speech-cfs/wangzhou/data/tts/VividHead/lia-x \
        --out_json data/vivi/train_liax.json \
        --batch 16
    # 冒烟: 加 --limit 200
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

# 复用 Talker-T2AV 的 LIA-X 加载 + 帧读取 + tensor 化 (与 recon 脚本同一实现)
T2AV = "/nfs-speech-cfs/wangzhou/s2s/Talker-T2AV"
sys.path.insert(0, T2AV)
from lia_x_recon import load_liax, read_video_frames, tensorize  # noqa: E402

DEFAULT_CKPT = os.path.join(T2AV, "deps", "LIA-X", "lia-x.pt")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--list", default="data/vivi/train.list")
    p.add_argument("--out_root", default="/nfs-speech-cfs/wangzhou/data/tts/VividHead/lia-x",
                   help="motion .pt 落盘根目录 (会建 <out_root>/motion/)")
    p.add_argument("--out_json", default="data/vivi/train_liax.json")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--fps", type=float, default=25.0, help="重采样目标 fps (vivi 本就 25)")
    p.add_argument("--batch", type=int, default=16, help="编码器一次前向的帧数")
    p.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 条 (冒烟)")
    p.add_argument("--flush_every", type=int, default=500, help="每 N 条重写一次 json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"],
                   help="编码器 autocast 精度; bf16 约 1.4x 提速且省显存, 对 codes MAE~4e-4 (可忽略)")
    # 多卡分片: entries[shard::num_shards]。每个 worker 独立进程, 各写自己的 out_json
    # (.pt 按 key 命名, 全部 worker 共写同一 motion/ 目录, 无冲突)。跑完 merge 成总 json。
    p.add_argument("--shard", type=int, default=0, help="本 worker 分片编号 [0, num_shards)")
    p.add_argument("--num_shards", type=int, default=1, help="总分片数 (= 总 GPU 数)")
    return p.parse_args()


_DTYPE = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def _is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


@torch.no_grad()
def _encode_frames(model, frames, batch, amp_dtype):
    """(T,3,H,W)[-1,1] -> (T,40) fp32 numpy。单次尝试, 指定 batch。

    amp_dtype!=None 时用 torch.autocast: 权重仍 fp32, matmul/conv 走半精度 (bf16 约 1.4x
    且省显存, 对 40 维 codes MAE~4e-4 可忽略), 精度敏感 op 由 autocast 自动留 fp32。
    编码路径只有 Encoder2R 卷积 + enc_r2t(Sequential), 无 decoder 的 QR/geqrf, 半精度安全。
    """
    T = frames.shape[0]
    codes = []
    for start in range(0, T, batch):
        x = frames[start:start + batch]
        if amp_dtype is None:
            z_s2r, _ = model.enc.enc_2r(x)
            alpha = model.enc.enc_r2t(z_s2r)
        else:
            with torch.autocast("cuda", dtype=amp_dtype):
                z_s2r, _ = model.enc.enc_2r(x)
                alpha = model.enc.enc_r2t(z_s2r)    # (B,40) == get_alpha(x)
        codes.append(alpha.float().cpu().numpy())
    return np.concatenate(codes, axis=0).astype(np.float32)   # (T,40)


def extract_motion(model, device, video_path, resolution, fps, batch, amp_dtype):
    """视频 -> (T,40) raw motion。共享 GPU 上遇 CUDA OOM 自动 empty_cache + 减半 batch 重试
    (直到 batch=1), 避免瞬时显存尖峰把整条丢掉。仍失败才向上抛。"""
    frames_np, _ = read_video_frames(video_path, resolution, max_frames=None, target_fps=fps)
    frames = tensorize(frames_np, device)          # (T,3,H,W) in [-1,1]
    b = batch
    while True:
        try:
            return _encode_frames(model, frames, b, amp_dtype)
        except Exception as e:
            if _is_oom(e) and b > 1:
                torch.cuda.empty_cache()
                b = max(1, b // 2)
                continue
            raise


def load_existing_pt(pt_path):
    """已存在 .pt 且能正常 load -> 返回帧数 T, 否则 None (需重算)。"""
    try:
        t = torch.load(pt_path, map_location="cpu")
        if t.ndim == 2 and t.shape[1] == 40 and t.shape[0] > 0:
            return int(t.shape[0])
    except Exception:
        return None
    return None


def main():
    args = get_args()
    motion_dir = os.path.join(args.out_root, "motion")
    os.makedirs(motion_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)

    with open(args.list) as f:
        entries = [json.loads(l) for l in f if l.strip()]
    total_all = len(entries)
    if args.num_shards > 1:
        entries = entries[args.shard::args.num_shards]   # stride 切片, 各 worker 不重叠
    if args.limit > 0:
        entries = entries[: args.limit]
    print(f"[vivi] shard {args.shard}/{args.num_shards}: {len(entries)}/{total_all} 条待处理 "
          f"-> pt: {motion_dir}  json: {args.out_json}", flush=True)

    amp_dtype = _DTYPE[args.dtype]
    print(f"[liax] loading {args.ckpt} ... (autocast={args.dtype})", flush=True)
    model, _ = load_liax(args.ckpt, motion_dim=40, scale=2, device=args.device)  # returns (model, device)

    records = []
    done = skip = fail = 0
    t0 = time.time()
    for i, d in enumerate(entries):
        key = str(d["key"])
        video = d["video"]
        audio = d.get("audio", "")
        pt_path = os.path.abspath(os.path.join(motion_dir, key + ".pt"))

        T = load_existing_pt(pt_path)
        if T is not None:
            skip += 1
        else:
            # OOM (哪怕 batch=1) 不丢条: empty_cache 后等待重试, 让别的进程释放显存。
            # 最多等 ~5 分钟 (共享卡瞬时挤满通常几十秒恢复), 之后才算真失败。
            oom_wait = 0
            while True:
                try:
                    codes = extract_motion(model, args.device, video, args.resolution, args.fps, args.batch, amp_dtype)
                    torch.save(torch.from_numpy(codes).half(), pt_path)
                    T = int(codes.shape[0])
                    done += 1
                    break
                except Exception as e:
                    if _is_oom(e) and oom_wait < 300:
                        torch.cuda.empty_cache()
                        time.sleep(15)
                        oom_wait += 15
                        continue
                    fail += 1
                    print(f"[fail] {key}: {repr(e)[:120]}", flush=True)
                    break
            if T is None:
                continue   # 真失败 (非 OOM 或等满 5 分钟仍 OOM): 不写 json, 续跑时再试

        records.append({
            "sample_id": f"vivi_{key}",
            "dataset_source": "vivi",
            "split": "train",
            "text": "",
            "wav_path": audio,
            "motion_pt_path": pt_path,
            "video_path": video,
            "pt_length": T,
            "fps": 25,
        })

        if (i + 1) % args.flush_every == 0 or (i + 1) == len(entries):
            with open(args.out_json, "w") as w:
                for r in records:
                    w.write(json.dumps(r, ensure_ascii=False) + "\n")
            el = time.time() - t0
            rate = (done + skip) / max(el, 1e-9)
            eta = (len(entries) - (i + 1)) / max(rate, 1e-9)
            print(f"[{i+1}/{len(entries)}] done={done} skip={skip} fail={fail}  "
                  f"{el:.0f}s  {rate:.1f} 条/s  ETA {eta/3600:.1f}h", flush=True)

    # 最终再写一次
    with open(args.out_json, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[vivi] 完成: done={done} skip={skip} fail={fail}  "
          f"json {len(records)} 行 -> {args.out_json}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
