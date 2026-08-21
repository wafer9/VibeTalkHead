"""分层诊断: 定位"生成差"到底坏在哪一环.

对同一条样本跑 3 个层次, 各自 decode 成 mp4:
  L1  gt_decode      : GT latent 直接 VAE decode          -> VAE/数据链路上界
  L2  teacher_force  : GT video 特征填 video 槽(与训练一致),
                       一次性并行去噪所有帧               -> DiT + condition 质量
  L3  autoregressive : 逐帧自回归(推理真实路径)            -> 叠加自回归漂移

判读:
  L1 糊            -> VAE/latent 链路坏 (先修这个)
  L1 好, L2 也糊   -> DiT 没学好 / condition 坍缩 (大概率 lr 把 LLM 训崩)
  L2 好, L3 糊     -> 自回归漂移 (训练/推理不一致)

用法:
  CUDA_VISIBLE_DEVICES=0 python -m twinlakes.bin.diagnose_video \
      --config exp/s1/train.yaml --checkpoint exp/s1/19.pt \
      --test_data data/vivi/dev_latent.list --result_dir exp/s1/diag \
      --num_steps 20 --cfg_scale 1.5 --limit 1 --max_seconds 4
"""
import argparse
import json
import os

import numpy as np
import torch
import torchaudio
import yaml

from twinlakes.models.rq_transformer import LMModel
from twinlakes.utils.checkpoint import load_checkpoint
from twinlakes.dataset.latent_io import load_latent_zlib
from twinlakes.bin.infer_video import (
    encode_audio_features, generate_one, sample_chunk, save_video_with_audio,
    model_dtype, AUDIO_PER_CHUNK, VIDEO_PER_CHUNK, CHUNK_LEN, SAMPLE_RATE,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--num_steps", type=int, default=20)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max_seconds", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


@torch.no_grad()
def build_condition_teacher_forced(model, wav, gt_latent, device):
    """完全复刻训练 forward 的前半段: 用 GT video 特征填 video 槽,
    返回 (h_video[1,T,H_lm], reference[1,16,1,64,64], num_chunks).
    h_video[t] = LLM 在 video 槽前一位的 hidden (video_loss_mask)。"""
    dtype = model_dtype(model)
    tok = model.tokenizer

    audio_feats = encode_audio_features(model, wav, device)      # (1, T_a, H)
    T_a = audio_feats.shape[1]
    T_v = gt_latent.shape[1]
    num_chunks = min(T_a // AUDIO_PER_CHUNK, T_v // VIDEO_PER_CHUNK)
    assert num_chunks > 0

    prompt = " Reference image:<|image_pad|>\n Video output:\n<|vision_start|>"
    body = ("<|vision_pad|>" * AUDIO_PER_CHUNK + "<|vision_end|>" * VIDEO_PER_CHUNK) * num_chunks
    ids = tok.encode(prompt + body + "<|endoftext|>")
    input_ids = torch.tensor(ids, dtype=torch.int64, device=device).unsqueeze(0)
    s = int(torch.where(input_ids[0] == tok.speech_start_id)[0][0].item())

    x = model.lm.get_input_embeddings()(input_ids)              # (1, S, H)

    # video 特征 = comp(GT latent), 与训练 videos_features 一致
    n_vframes = num_chunks * VIDEO_PER_CHUNK
    vlat = gt_latent[:, :n_vframes].unsqueeze(0).to(device=device, dtype=dtype)  # (1,16,Tv,64,64)
    vfeat = model.comp(vlat)                                    # (1, Tv, H)

    for k in range(num_chunks):
        a0 = s + 1 + k * CHUNK_LEN
        x[0, a0: a0 + AUDIO_PER_CHUNK] = \
            audio_feats[0, k * AUDIO_PER_CHUNK:(k + 1) * AUDIO_PER_CHUNK].to(dtype)
        v0 = a0 + AUDIO_PER_CHUNK
        x[0, v0: v0 + VIDEO_PER_CHUNK] = \
            vfeat[0, k * VIDEO_PER_CHUNK:(k + 1) * VIDEO_PER_CHUNK]

    out = model.lm.model(inputs_embeds=x, use_cache=False, return_dict=True)
    hs = out.last_hidden_state                                  # (1, S, H)

    # video 槽前一位的 hidden, 按 chunk 收集
    cond = []
    for k in range(num_chunks):
        v0 = s + 1 + k * CHUNK_LEN + AUDIO_PER_CHUNK
        for j in range(VIDEO_PER_CHUNK):
            cond.append(hs[:, v0 + j - 1])                      # (1, H)
    h_video = torch.stack(cond, dim=1)                          # (1, Tv, H)
    reference = gt_latent[:, :1].unsqueeze(0).to(device=device, dtype=dtype)
    return h_video, reference, num_chunks, n_vframes


@torch.no_grad()
def main():
    args = get_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs['freeze_diffusion_head'] = False
    model = LMModel.from_audio_text_pretrained(configs)
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    dtype = model_dtype(model)
    os.makedirs(args.result_dir, exist_ok=True)

    lines = [l.strip() for l in open(args.test_data) if l.strip()][:args.limit]
    for line in lines:
        obj = json.loads(line); key = obj['key']
        wav, sr = torchaudio.load(obj['audio'], backend="soundfile"); wav = wav[:1]
        if sr != SAMPLE_RATE:
            wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
        if args.max_seconds > 0:
            wav = wav[:, :int(args.max_seconds * SAMPLE_RATE)]
        gt_latent = load_latent_zlib(obj['latent']).float()

        # ---- L1: GT latent 直接 decode ----
        h_video, reference, nc, n_vframes = build_condition_teacher_forced(
            model, wav, gt_latent, device)
        gt_used = gt_latent[:, :n_vframes].to(device=device, dtype=dtype)  # [16,Tv,64,64]
        v1 = model.vae.decode(gt_used)[0]
        save_video_with_audio(v1, wav, os.path.join(args.result_dir, f"{key}_L1_gt.mp4"), fps=30)

        # ---- L2: teacher-forcing 去噪 (GT 特征做 condition, 一次并行去噪 Tv 帧) ----
        tf_latent = sample_chunk(model, h_video, reference, args.cfg_scale, args.num_steps)  # (1,16,Tv,64,64)
        # 拼上首帧参考, 与自回归输出口径一致
        tf_full = torch.cat([gt_latent[:, :1].unsqueeze(0).to(device, dtype), tf_latent], dim=2)[0]
        v2 = model.vae.decode(tf_full)[0]
        save_video_with_audio(v2, wav, os.path.join(args.result_dir, f"{key}_L2_teacher.mp4"), fps=30)

        # ---- L3: 自回归 (真实推理路径) ----
        ar = generate_one(model, wav, gt_latent[:, :1], args.cfg_scale, args.num_steps, device)
        v3 = model.vae.decode(ar.to(dtype))[0]
        save_video_with_audio(v3, wav, os.path.join(args.result_dir, f"{key}_L3_autoreg.mp4"), fps=30)

        # ---- 数值对比 ----
        gt_cmp = gt_latent[:, :n_vframes].to(device, dtype)
        mse_tf = (tf_latent[0].float() - gt_cmp.float()).square().mean().item()
        ar_cmp = ar[:, 1:1 + n_vframes].to(device, dtype)
        mse_ar = (ar_cmp.float() - gt_cmp.float()).square().mean().item()
        gt_var = gt_cmp.float().var().item()
        print(f"[{key}] nc={nc} Tv={n_vframes} | latent MSE  teacher={mse_tf:.4f}  "
              f"autoreg={mse_ar:.4f}  (GT var={gt_var:.4f})", flush=True)
        print(f"        h_video: mean={h_video.float().mean():.4f} std={h_video.float().std():.4f} "
              f"(std≈const => condition 坍缩)", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
