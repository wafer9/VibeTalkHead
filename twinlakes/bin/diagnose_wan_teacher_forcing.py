"""Compare GT decode, teacher-forced DiT, and autoregressive WAN inference."""

import argparse
import json
import math
import os

import cv2
import numpy as np
import torch
import yaml

from twinlakes.bin.infer_wan_video import (
    AudioNormalizer,
    LMModel,
    WanVAE,
    build_prompt,
    load_audio,
    load_checkpoint,
    save_video,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--cfg_scale", type=float, default=1.5)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--seed", type=int, default=777)
    return p.parse_args()


def load_video(path, device, dtype):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError("cannot decode video: {}".format(path))
    x = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    return x.to(device=device, dtype=dtype).div_(127.5).sub_(1.0)


@torch.no_grad()
def teacher_conditions(model, wav_norm, wav_length, gt, audio_tokens):
    """Reproduce the complete teacher-forced sequence used by training forward."""
    device = gt.device
    targets = gt.shape[2] - 1
    prompt = (
        "<|image_pad|>\n Voice input:<|vision_start|>"
        + "<|vision_pad|>" * audio_tokens
        + "<|vision_end|>\n Video output:\n<|vision_start|>"
    )
    label = "<|vision_pad|>" * targets + "<|vision_end|>"
    enc = model.tokenizer(
        text=[prompt], text_pair=[label], add_special_tokens=True,
        padding=True, return_tensors="pt", return_token_type_ids=True,
        return_attention_mask=True,
    )
    ids = enc["input_ids"].to(device)
    starts = torch.where(ids == model.tokenizer.speech_start_id)[1]
    ends = torch.where(ids == model.tokenizer.speech_end_id)[1]
    pos = [torch.stack((starts[0], ends[0], starts[1], ends[1]))]
    audio_mask, video_input_mask, video_loss_mask = model._sequence_masks(ids, pos)

    x = model.lm.get_input_embeddings()(ids)
    audio_features, audio_valid = model.encode_audio(wav_norm, wav_length)
    x[audio_mask] = audio_features[audio_valid]
    target_frames = gt[0, :, 1:].permute(1, 0, 2, 3).contiguous()
    x[video_input_mask] = model.video_connector(target_frames)
    out = model.lm.model(
        inputs_embeds=x, attention_mask=enc["attention_mask"].to(device),
        use_cache=False, return_dict=True,
    )
    return out.last_hidden_state[video_loss_mask]


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
        video_path = obj.get("video_path", obj.get("video"))
        wav = load_audio(wav_path, 0.0)
        pixels = load_video(video_path, device, dtype)
        gt = vae.encode(pixels).unsqueeze(0)  # [1,16,T,64,64]

        audio_tokens = math.ceil(wav.shape[1] / 3200)
        expected_total = (audio_tokens * 5 + 3) // 6
        usable = min(gt.shape[2], expected_total)
        gt = gt[:, :, :usable]
        wav_np = normalizer(wav[0].numpy())
        wav_norm = torch.from_numpy(wav_np).view(1, -1, 1).to(device, dtype)
        wav_length = torch.tensor([wav_norm.shape[1]], device=device, dtype=torch.int32)

        cond = teacher_conditions(model, wav_norm, wav_length, gt, audio_tokens)
        n = gt.shape[2] - 1
        reference = gt[:, :, 0].expand(n, -1, -1, -1)
        teacher = model.sample_video_frame(
            cond, reference, num_steps=args.num_steps,
            cfg_scale=args.cfg_scale,
        )
        teacher_full = torch.cat((gt[:, :, :1], teacher.permute(1, 0, 2, 3).unsqueeze(0)), dim=2)

        ids, audio_pos, _ = build_prompt(model, wav.shape[1], device)
        autoreg = model.generate_video_latents(
            ids, wav_norm, wav_length, audio_pos, gt[:, :, 0], num_frames=n,
            num_steps=args.num_steps, cfg_scale=args.cfg_scale,
        )

        outputs = {
            "L1_gt_decode": gt[0],
            "L2_teacher": teacher_full[0],
            "L3_autoreg": autoreg[0],
        }
        for suffix, latent in outputs.items():
            decoded = vae.decode(latent)[0]
            save_video(decoded, wav, os.path.join(args.result_dir, key + "_" + suffix + ".mp4"))

        tf_mse = (teacher.float() - gt[0, :, 1:].permute(1, 0, 2, 3).float()).square().mean()
        ar_mse = (autoreg[:, :, 1:].float() - gt[:, :, 1:].float()).square().mean()
        print("[done] {} total_latents={} teacher_mse={:.6f} autoreg_mse={:.6f}".format(
            key, usable, tf_mse.item(), ar_mse.item()), flush=True)


if __name__ == "__main__":
    main()
