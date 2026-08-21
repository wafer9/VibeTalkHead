# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Audio -> motion -> talking-head video 推理 (非流式, vibevoice 式 LLM + diffusion head).

当前模型 (twinlakes.models.rq_transformer.LMModel) 结构:
  wav(24k) -> vibevoice acoustic_tokenizer.encode -> (B, T_a, vae_dim) -> acoustic_connector
           -> 拼进 prompt 的 audio pad 区; LLM 自回归产出每帧 hidden state,
  diffusion_head (VibeVoice, v_prediction) 以该 hidden state 为条件, 从高斯噪声去噪出
  一帧 160 维 merged motion (= 4 帧 * 40 维 LIA-X motion, 归一化空间)。

推理流程 (与训练 forward 对偶):
  1. wav(24k) + prompt -> LMModel.generate_cfg 自回归 T_v 帧, 得到归一化 merged motion (T_v,160)
     - T_v 由音频长度确定 (processor.num_merged_motion_frames): 7.5Hz audio -> 6.25Hz motion
  2. unmerge: (T_v,160) -> (T_v*4, 40) @25fps
  3. 反归一化: motion = motion_norm * MOTION_STD + MOTION_MEAN
  4. LIA-X Generator 用一张 reference 人脸图 + motion 渲染每帧, 合成 mp4, 再 mux 音频

注意: 训练里 motion 做了全局 per-dim 标准化 (processor.py: _MOTION_MEAN_T/_MOTION_STD_T),
所以模型输出在归一化空间, 渲染前必须反归一化, 否则脸幅度会缩小 ~1/std 基本不动。

用法:
  CUDA_VISIBLE_DEVICES=0 python -m twinlakes.bin.infer_video \
      --config exp/s3/train.yaml \
      --checkpoint exp/s3/<epoch>.pt \
      --test_data data/talker/dev.json \
      --result_dir exp/s3/infer_out \
      --num_steps 10 --cfg_scale 1.5 --limit 8

test_data 每行一个 json, 至少包含:
  wav_path       相对 shards 根目录的音频路径 (与训练 json 一致)
  ref_image      参考人脸图/视频路径 (读首帧); 缺省时用 video_path 首帧
  video_path     (可选) 若无 ref_image, 从该视频读首帧作 reference
  sample_id/key  输出文件名
"""
import argparse
import json
import math
import os
import subprocess

import cv2
import numpy as np
import torch
import torchaudio
import yaml
from tqdm import tqdm

from twinlakes.models.rq_transformer import LMModel
from twinlakes.utils.checkpoint import load_checkpoint
# NOTE: 照 infer_parallel.py 的 import 风格, 只从 vibevoice 拿 AudioNormalizer,
# 不 import twinlakes.dataset.processor (它会连带触发 vibevoice tokenizer 的
# AutoModel.register 二次注册, 与 rq_transformer 顶层的注册冲突而报
# 'VibeVoiceAcousticTokenizerConfig is already used'). motion 统计/长度换算直接内联。
from vibevoice.processor.vibevoice_processor import AudioNormalizer
audio_normalizer = AudioNormalizer()

_AUDIO_BACKEND = os.environ.get("AUDIO_BACKEND", "soundfile")
ACOUSTIC_SR = 24000       # vibevoice acoustic_tokenizer 采样率 (与训练 decode_wav_raw 一致)
SPEECH_TOK_COMPRESS = 3200  # 24000/3200 = 7.5Hz audio token
MERGE = 4                 # 每 4 帧 40 维 motion 合并成 1 帧 160 维
MOTION_DIM = 40           # 单帧 LIA-X motion 维度
SHARDS_ROOT = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards/"
LIAX_CKPT = os.environ.get(
    "LIAX_CKPT", "/nfs-speech-cfs/wangzhou/s2s/Talker-T2AV/deps/LIA-X/lia-x.pt"
)
FPS = 25                  # motion @25fps

# ---- LIA-X 渲染半精度 / TF32 加速 -------------------------------------------
# LIA-X decoder 卷积换 fp16 后逐帧渲染 ~3x, RTF 5.98->2.0, 再叠 TF32 到 ~2.0,
# torch.compile 可进一步到 RTF~1.1。数值上 fp16 对渲染近乎无损 (MAE 7e-4)。
# 注意: dec.direction() 内部有 QR 正交化 (geqrf), 半精度无 CUDA 实现, 必须留 fp32;
# encoder (get_alpha / enc_2r) 也保持 fp32, 只有 decoder 卷积走 fp16。
RENDER_FP16 = os.environ.get("RENDER_FP16", "1") == "1"     # decoder 卷积用 fp16
RENDER_COMPILE = os.environ.get("RENDER_COMPILE", "0") == "1"  # 额外 torch.compile(dec)
_RENDER_DTYPE = torch.float16 if RENDER_FP16 else torch.float32
RENDER_COMPILE = True
# RENDER_FP16 = False
# _RENDER_DTYPE = torch.float32
# print("="*20, _RENDER_DTYPE)

# TF32: 免费 ~2x, 零精度损失 (仅影响 conv/matmul 内部累加, 张量仍 fp32)。
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# motion 归一化统计 (40,), 与 twinlakes/dataset/processor.py 完全同源 (同一份 npy)。
_MOTION_STATS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset", "stats"
)
MOTION_MEAN = np.load(os.path.join(_MOTION_STATS_DIR, "motion_mean.npy")).astype(np.float32)
MOTION_STD = np.load(os.path.join(_MOTION_STATS_DIR, "motion_std.npy")).astype(np.float32)
MOTION_STD = np.clip(MOTION_STD, a_min=1e-6, a_max=None)
# 反归一化用的 (40,) tensor (CPU, 用时 .to(device))
_MEAN_T = torch.from_numpy(MOTION_MEAN)
_STD_T = torch.from_numpy(MOTION_STD)


def num_merged_motion_frames(wav_samples, speech_tok_compress_ratio=SPEECH_TOK_COMPRESS,
                             merge=MERGE, audio_hz_num=6, motion_hz_num=5):
    """由音频采样数反推合并后 (6.25Hz,160维) 的 motion 帧数 T_v。

    与 processor.num_merged_motion_frames 保持一致的确定性映射:
    A = ceil(wav_samples/3200) 个 7.5Hz audio token; T_v = round(A * 5/6)。
    """
    audio_tok = math.ceil(wav_samples / speech_tok_compress_ratio)
    return (audio_tok * motion_hz_num + audio_hz_num // 2) // audio_hz_num


def get_args():
    p = argparse.ArgumentParser(description="audio -> motion -> talking-head video")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--num_steps", type=int, default=10, help="每帧扩散去噪步数")
    p.add_argument("--cfg_scale", type=float, default=1.5,
                   help="classifier-free guidance 强度; 1.0 关闭")
    p.add_argument("--max_seconds", type=float, default=0.0,
                   help=">0 时截断音频到该秒数, 控制生成长度; 0=整条")
    p.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 条")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--liax_ckpt", default=LIAX_CKPT)
    p.add_argument("--shards_root", default=SHARDS_ROOT)
    return p.parse_args()


def model_dtype(model):
    return next(model.parameters()).dtype


def _resolve(root, path):
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(root, path)


@torch.no_grad()
def load_wav_24k(path, max_seconds=0.0):
    """读音频 -> 单声道 24k float [1, N] (与训练 acoustic_tokenizer 输入一致)."""
    wav, sr = torchaudio.load(path, backend=_AUDIO_BACKEND)
    wav = wav[:1]
    if sr != ACOUSTIC_SR:
        wav = torchaudio.transforms.Resample(sr, ACOUSTIC_SR)(wav)
    if max_seconds > 0:
        wav = wav[:, : int(max_seconds * ACOUSTIC_SR)]
    return wav


def build_prompt_inputs(model, wav_24k, device):
    """构造与 processor.tokenize 一致的 prompt 段 (到第二个 <|vision_start|> 为止)。

    prompt 结构:
      <|image_pad|>\n Voice input:<|vision_start|>{audio_pad*A}<|vision_end|>\n
       Video output:\n<|vision_start|>
    A = ceil(N/3200) 个 audio token (7.5Hz)。返回 (input_ids, audio_pos)。
    """
    tokenizer = model.tokenizer
    vae_audio_tok_len = math.ceil(wav_24k.shape[1] / SPEECH_TOK_COMPRESS)
    prompt = ("<|image_pad|>\n Voice input:<|vision_start|>%s<|vision_end|>\n"
              " Video output:\n<|vision_start|>" % ("<|vision_pad|>" * vae_audio_tok_len))
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64,
                             device=device).unsqueeze(0)  # (1, S)

    start_pos = torch.where(input_ids == tokenizer.speech_start_id)
    end_pos = torch.where(input_ids == tokenizer.speech_end_id)
    # audio_pos = [p_s, p_e, s]: 第一个 vision_start/end + 第二个 vision_start
    audio_pos = torch.stack((start_pos[1][0], end_pos[1][0], start_pos[1][1]), dim=0)
    return input_ids, audio_pos


@torch.no_grad()
def extract_reference_motion(liax_model, ref_path, device, dtype):
    """从参考图片提取 LIA-X 40 维 motion -> 归一化 -> 重复 4 次 -> (1,160)。

    对齐训练端 forward 的参考帧构造:
        reference_features = videos_features[:,:1,:40].repeat(1,1,4)   # (B,1,160)
    训练里 videos_features 是「首帧 40 维 raw alpha 经 MOTION_MEAN/STD 归一化」后取前 40 维,
    推理这里用 liax.get_alpha(ref) 抽同一种 40 维 alpha, 走同一套归一化 + 重复 4 次。
    """
    ref_img = read_ref_image(ref_path).to(next(liax_model.parameters()).device)
    alpha = liax_model.get_alpha(ref_img).float().cpu()[0]   # (40,) raw LIA-X motion
    alpha_norm = (alpha - _MEAN_T) / _STD_T                   # (40,) 归一化 (与训练一致)
    ref160 = alpha_norm.repeat(MERGE)                         # (160,) = 40 维重复 4 次
    return ref160.to(device=device, dtype=dtype).unsqueeze(0)  # (1,160)


@torch.no_grad()
def generate_motion(model, wav_24k, device, num_steps, cfg_scale, reference_motion):
    """wav[1,N]@24k + 参考帧 motion -> 归一化 merged motion (T_v,160), 调 generate_cfg 自回归。"""
    dtype = model_dtype(model)
    input_ids, audio_pos = build_prompt_inputs(model, wav_24k, device)

    # acoustic_tokenizer 期望的输入布局: (B, N, 1) (与训练 forward wavs.transpose(1,2) 对齐)
    # 训练端 wav 过 AudioNormalizer, 这里同样处理保持分布一致。
    wav_norm = torch.from_numpy(audio_normalizer(wav_24k[0].numpy())).unsqueeze(0)  # (1, N)
    wavs = wav_norm.to(device=device, dtype=dtype).unsqueeze(2)  # (1, N, 1)
    wavs_lengths = torch.tensor([wav_norm.shape[1]], dtype=torch.int64, device=device)

    # 显式传 num_frames, 让 generate_cfg 不用回落去 import twinlakes.dataset.processor
    # (那条 import 会触发 vibevoice tokenizer 二次注册). 长度换算用本文件内联的同一函数。
    num_frames = max(int(num_merged_motion_frames(int(wav_norm.shape[1]))), 1)

    motion = model.generate_cfg(
        keys=None,
        wavs=wavs,
        wavs_lengths=wavs_lengths,
        input_ids=input_ids,
        audio_pos=[audio_pos],
        reference_motion=reference_motion,
        cfg_scale=cfg_scale,
        num_frames=num_frames,
        num_inference_steps=num_steps,
    )  # (1, T_v, 160)
    return motion[0]  # (T_v, 160) 归一化空间


def unmerge_motion(merged):
    """(T_v, 160) -> (T_v*4, 40): 把 4 帧合并的 160 维还原成 25fps 的 40 维序列。"""
    T_v, dim = merged.shape
    assert dim == MERGE * MOTION_DIM, (dim, MERGE * MOTION_DIM)
    return merged.reshape(T_v * MERGE, MOTION_DIM).contiguous()


def denormalize(motion_norm):
    """(T,40) 归一化 -> 原始 LIA-X motion."""
    return motion_norm * _STD_T.to(motion_norm.device) + _MEAN_T.to(motion_norm.device)


# ---------------------------------------------------------------------------
# LIA-X 渲染 (复用 Talker-T2AV/infer.py 的渲染逻辑)
# ---------------------------------------------------------------------------
def init_render_model(liax_ckpt):
    import sys
    t2av = "/nfs-speech-cfs/wangzhou/s2s/Talker-T2AV"
    if t2av not in sys.path:
        sys.path.insert(0, t2av)
    from lia_x.networks.generator import Generator

    m = Generator(motion_dim=40, scale=2)
    state_dict = torch.load(liax_ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(state_dict, strict=True)
    m.cuda().eval()

    if RENDER_FP16:
        # 只把 decoder 卷积换 fp16; 但 dec.direction (Direction 子模块) 内部走
        # torch.linalg.qr(geqrf), 半精度无 CUDA 实现 -> 把它单独恢复 fp32。
        # 于是: motion -> direction(fp32) -> r_d, 再在 render 里把 r_d 转 fp16 喂卷积。
        m.dec.half()
        m.dec.direction.float()
    if RENDER_COMPILE:
        m.dec = torch.compile(m.dec, mode="default", fullgraph=False)

    print(f"[render] LIA-X loaded from {liax_ckpt} "
          f"(fp16={RENDER_FP16}, compile={RENDER_COMPILE})", flush=True)
    return m


@torch.no_grad()
def read_ref_image(path):
    """读参考图/视频首帧 -> (1,3,512,512) in [-1,1]."""
    if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        cap = cv2.VideoCapture(path)
        ret, bgr = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"无法读取视频首帧: {path}")
    else:
        bgr = cv2.imread(path)
        if bgr is None:
            raise ValueError(f"无法读取图片: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0)


@torch.no_grad()
def render_motion(liax_model, motion_40, ref_path, out_path, fps=FPS):
    """motion_40: (T,40) 原始 LIA-X motion tensor -> 无声 mp4."""
    device = next(liax_model.enc.parameters()).device
    ref = read_ref_image(ref_path).to(device)          # fp32
    z_s2r, feats = liax_model.enc.enc_2r(ref)           # encoder fp32

    # direction 保持 fp32 (QR); 之后把喂给 decoder 卷积的张量统一转成 dec 的 dtype。
    r_d = liax_model.dec.direction(motion_40.to(device).float())
    dec_dtype = _RENDER_DTYPE
    if dec_dtype != torch.float32:
        z_s2r = z_s2r.to(dec_dtype)
        feats = [f.to(dec_dtype) for f in feats] if isinstance(feats, (list, tuple)) \
            else feats.to(dec_dtype)
        r_d = r_d.to(dec_dtype)
    T = r_d.shape[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (512, 512))
    for t in range(T):
        s_r_d_t = z_s2r + r_d[t].unsqueeze(0)
        img_t = liax_model.dec(s_r_d_t, alpha=None, feats=feats)
        img = img_t[0].clamp(-1, 1).cpu()
        img = ((img + 1) / 2 * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()


def mux_audio(video_path, wav_24k, out_path, fps=FPS):
    """把 wav 合进无声视频; 无 ffmpeg 则保留无声视频。"""
    import shutil
    if shutil.which("ffmpeg") is None:
        os.replace(video_path, out_path)
        return
    tmp_wav = out_path + ".tmp.wav"
    torchaudio.save(tmp_wav, wav_24k.cpu(), sample_rate=ACOUSTIC_SR, backend=_AUDIO_BACKEND)
    # 重新编码视频为 H.264 (yuv420p 保证通用播放器兼容), 而不是 -c:v copy 保留 OpenCV
    # 写出的 mp4v(MPEG-4 Part2)。本机 ffmpeg 无 libx264, 用 libopenh264 软编码;
    # 它不支持 -crf, 用码率控制 (-b:v)。faststart 便于流式播放。
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-i", tmp_wav,
         "-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-b:v", "2M",
         "-movflags", "+faststart", "-c:a", "aac", "-shortest", out_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for f in (video_path, tmp_wav):
        if os.path.exists(f):
            os.remove(f)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    model = LMModel.from_audio_text_pretrained(configs)
    load_checkpoint(model, args.checkpoint)
    device = torch.device("cuda:0")
    model = model.to(device).eval()

    liax_model = init_render_model(args.liax_ckpt)

    os.makedirs(args.result_dir, exist_ok=True)
    with open(args.test_data) as f:
        lines = [l.strip() for l in f if l.strip()]
    if args.limit > 0:
        lines = lines[: args.limit]

    import time
    for line in tqdm(lines):
        t0 = time.time()
        obj = json.loads(line)
        key = obj.get("sample_id", obj.get("key", "sample"))

        wav_path = _resolve(args.shards_root, obj.get("wav_path") or obj.get("audio"))
        ref_path = obj.get("ref_image") or _resolve(
            args.shards_root, obj.get("video_path", "")
        )
        if not ref_path or not os.path.exists(ref_path):
            print(f"[skip] {key}: 无 reference 图/视频 ({ref_path})", flush=True)
            continue

        wav = load_wav_24k(wav_path, args.max_seconds)
        t1 = time.time()
        # 参考帧 motion: 从渲染用的同一张参考图提 LIA-X alpha, 保证条件与渲染外观一致
        reference_motion = extract_reference_motion(liax_model, ref_path, device, model_dtype(model))
        t2 = time.time()
        merged_norm = generate_motion(model, wav, device, args.num_steps, args.cfg_scale,
                                      reference_motion)  # (T_v,160)
        t3 = time.time()
        motion_norm = unmerge_motion(merged_norm)  # (T_v*4, 40) @25fps
        t4 = time.time()
        motion = denormalize(motion_norm)  # (T,40) 原始空间

        tmp_video = os.path.join(args.result_dir, f"{key}.noaudio.mp4")
        out_path = os.path.join(args.result_dir, f"{key}.mp4")
        render_motion(liax_model, motion, ref_path, tmp_video, fps=FPS)
        mux_audio(tmp_video, wav, out_path, fps=FPS)
        print(f"[done] {key}  motion={tuple(motion.shape)} -> {out_path}", flush=True)
        t5 = time.time()
        print("t1:%.4f, t2:%.4f, t3:%.4f, t4:%.4f, t5:%.4f" %(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
