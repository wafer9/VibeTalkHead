"""Offline WAN-VAE latent pre-extraction.

训练时 forward 里每步 `vae.encode([B,3,~97,512,512])` 要 13s (逐 latent 帧 3D 卷积),
是 34s/step 的元凶。VAE 冻结 → latent 确定性 → 离线预存,训练直接读 [16,Tlat,64,64]。

用法 (单卡):
    python -m twinlakes.bin.extract_latents \
        --config conf/run_stage1.yaml \
        --in_list data/vivi/train.list \
        --out_dir /nfs-speech-cfs/wangzhou/data/tts/VividHead/wan_latents \
        --out_list data/vivi/train_latent.list

多机多卡 (由 run.sh stage2 驱动, 每张卡一个全局 shard, 断点续跑靠 .ptz 是否存在):
    全局 shard = node_rank * gpus_per_node + local_gpu, world = num_nodes * gpus_per_node
    # 每台机上:
    for local_gpu in 0..gpus_per_node-1; do
        shard=$(( node_rank * gpus_per_node + local_gpu ))
        CUDA_VISIBLE_DEVICES=$local_gpu python -m twinlakes.bin.extract_latents \
            --config conf/run_stage1.yaml --in_list data/vivi/train.list \
            --out_dir <out_dir> --out_list data/vivi/train_latent.$shard.list \
            --rank $shard --world_size $world &
    done; wait
    # 所有机器都跑完后合并:
    cat data/vivi/train_latent.*.list > data/vivi/train_latent.list
"""
from __future__ import print_function

import argparse
import json
import os

import torch
import yaml
from decord import VideoReader, cpu

from twinlakes.vae.wan import WanVAE
from twinlakes.dataset.latent_io import save_latent_zlib

# 本脚本只解码视频做 VAE encode; audio 仅原样透传到 out_list, 不读波形,
# 故不 import torchaudio (省掉它的 CUDA 扩展依赖)。


def get_args():
    parser = argparse.ArgumentParser(description="pre-extract WAN-VAE latents")
    parser.add_argument("--config", required=True, help="train config (for vae_path/dtype)")
    parser.add_argument("--in_list", required=True, help="input data list (key/audio/video)")
    parser.add_argument("--out_dir", required=True, help="dir to save <key>.ptz latents")
    parser.add_argument("--out_list", required=True, help="output data list (key/audio/latent/fps)")
    parser.add_argument("--rank", type=int, default=0,
                        help="global shard index for this process (0..world_size-1)")
    parser.add_argument("--world_size", type=int, default=1,
                        help="total number of shards across all machines*gpus")
    parser.add_argument("--num_threads", type=int, default=8, help="decord decode threads")
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def decode_video(path, num_threads):
    """mp4 -> [C, Tf, 512, 512] float in [-1, 1] (same as processor.decode_wav_raw)."""
    vr = VideoReader(path, ctx=cpu(0), num_threads=num_threads)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [Tf, H, W, C] uint8 RGB
    fps = float(vr.get_avg_fps())
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [Tf, C, H, W]
    frames = frames / 127.5 - 1.0
    video = frames.permute(1, 0, 2, 3).contiguous()  # [C, Tf, H, W]
    return video, fps


def main():
    args = get_args()
    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    dtype = torch.bfloat16 if configs.get("dtype") == "bf16" else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = WanVAE(vae_path=configs["vae_path"], dtype=dtype, device=device)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.in_list, "r") as fin:
        lines = [ln for ln in fin if ln.strip()]
    # shard by rank: 断点续跑靠目标 .pt 存在与否, 这里只切分工作量
    lines = lines[args.rank::args.world_size]

    n_total = len(lines)
    n_done, n_skip, n_fail = 0, 0, 0
    with open(args.out_list, "w") as fout:
        for i, ln in enumerate(lines):
            obj = json.loads(ln)
            key = obj["key"]
            out_pt = os.path.join(args.out_dir, "{}.ptz".format(key))

            try:
                if not os.path.exists(out_pt):
                    video, fps = decode_video(obj["video"], args.num_threads)
                    video = video.to(device=device, dtype=dtype)
                    with torch.no_grad():
                        # WanVAE.encode 契约: [1,C,Tf,H,W] -> squeeze(0) -> [16,Tlat,64,64]
                        latent = vae.encode(video.unsqueeze(0))
                    # bf16 + zlib 无损: 2.37TB -> ~1.62TB, 解压快不拖 dataloader
                    save_latent_zlib(latent.to(torch.bfloat16), out_pt)
                    n_done += 1
                else:
                    # 已存在: 仍需拿 fps 写 out_list
                    vr = VideoReader(obj["video"], ctx=cpu(0), num_threads=1)
                    fps = float(vr.get_avg_fps())
                    n_skip += 1
            except Exception as ex:  # 单条坏样本不该中断全量提取
                n_fail += 1
                print("[fail] key={} err={}".format(key, ex), flush=True)
                continue

            rec = {"key": key, "audio": obj["audio"], "latent": out_pt, "fps": fps}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            if (i + 1) % args.log_interval == 0:
                print(
                    "[rank {}] {}/{} done={} skip={} fail={}".format(
                        args.rank, i + 1, n_total, n_done, n_skip, n_fail
                    ),
                    flush=True,
                )

    print(
        "[rank {}] FINISHED total={} done={} skip={} fail={}".format(
            args.rank, n_total, n_done, n_skip, n_fail
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
