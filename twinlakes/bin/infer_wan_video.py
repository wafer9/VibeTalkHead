"""End-to-end inference for the WAN-latent talking-head model."""

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

from twinlakes.models.rq_transformer import LMModel
from twinlakes.utils.checkpoint import load_checkpoint
from twinlakes.vae.wan import WanVAE
from vibevoice.processor.vibevoice_processor import AudioNormalizer


SAMPLE_RATE = 24000
LATENT_FPS = 6.25
VIDEO_FPS = 25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max_seconds", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def load_audio(path, max_seconds):
    wav, sr = torchaudio.load(path, backend="soundfile")
    wav = wav[:1]
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    if max_seconds > 0:
        wav = wav[:, : int(max_seconds * SAMPLE_RATE)]
    return wav


def load_reference(path, device, dtype):
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot read reference frame: {}".format(path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return x.to(device=device, dtype=dtype).div_(127.5).sub_(1.0)


def build_prompt(model, wav_samples, device):
    audio_tokens = math.ceil(wav_samples / 3200)
    prompt = (
        "<|image_pad|>\n Voice input:<|vision_start|>"
        + "<|vision_pad|>" * audio_tokens
        + "<|vision_end|>\n Video output:\n<|vision_start|>"
    )
    ids = torch.tensor(model.tokenizer.encode(prompt), device=device).unsqueeze(0)
    starts = torch.where(ids == model.tokenizer.speech_start_id)[1]
    ends = torch.where(ids == model.tokenizer.speech_end_id)[1]
    audio_pos = [torch.stack((starts[0], ends[0], starts[1]))]
    return ids, audio_pos, audio_tokens


def save_video(video, wav, output):
    # video is [3,T,H,W], in [-1,1].
    silent = output + ".silent.mp4"
    frames = ((video.float().clamp(-1, 1) + 1) * 127.5).byte()
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    wav_path = output + ".wav"
    torchaudio.save(wav_path, wav.cpu(), SAMPLE_RATE, backend="soundfile")
    subprocess.run(
        ["ffmpeg", "-y", "-i", silent, "-i", wav_path, "-c:v", "copy",
         "-c:a", "aac", "-shortest", output],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    os.remove(silent)
    os.remove(wav_path)


@torch.inference_mode()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dtype = torch.bfloat16 if config.get("dtype") == "bf16" else torch.float32

    model = LMModel.from_audio_text_pretrained(config)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device=device, dtype=dtype).eval()
    vae = WanVAE(z_dim=config.get("z_dim", 16), vae_path=config["vae_path"],
                 dtype=dtype, device=device, parallel=False)
    normalizer = AudioNormalizer()
    os.makedirs(args.result_dir, exist_ok=True)

    with open(args.test_data) as f:
        records = [json.loads(line) for line in f if line.strip()][:args.limit]
    for obj in records:
        key = obj.get("sample_id", obj.get("key", "sample"))
        wav_path = obj.get("wav_path", obj.get("audio"))
        ref_path = obj.get("ref_image") or obj.get("video_path", obj.get("video"))
        wav = load_audio(wav_path, args.max_seconds)
        ref_pixels = load_reference(ref_path, device, dtype)
        reference = vae.encode(ref_pixels).squeeze(1)  # [16,64,64]
        ids, audio_pos, audio_tokens = build_prompt(model, wav.shape[1], device)
        wav_norm = torch.from_numpy(normalizer(wav[0].numpy())).view(1, -1, 1)
        wav_norm = wav_norm.to(device=device, dtype=dtype)
        lengths = torch.tensor([wav_norm.shape[1]], device=device, dtype=torch.int32)
        # round(audio_tokens * 5/6) is the total 6.25-Hz latent length;
        # generate_video_latents prepends the reference, so generate T-1 targets.
        total_latents = (audio_tokens * 5 + 3) // 6
        generated = model.generate_video_latents(
            ids, wav_norm, lengths, audio_pos, reference.unsqueeze(0),
            num_frames=max(total_latents - 1, 1), num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )[0]
        decoded = vae.decode(generated)[0]
        output = os.path.join(args.result_dir, key + ".mp4")
        save_video(decoded, wav, output)
        print("[done] {} latent={} video={} -> {}".format(
            key, tuple(generated.shape), tuple(decoded.shape), output), flush=True)


if __name__ == "__main__":
    main()
