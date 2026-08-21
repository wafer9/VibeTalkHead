import json
import sys
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

from twinlakes.vae.wan import WanVAE
from tqdm import tqdm

def read_full_video(video_path, device="cuda:7", dtype=torch.bfloat16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [T,H,W,C] uint8
    fps = float(vr.get_avg_fps())

    frames = torch.from_numpy(frames)
    video = frames.permute(3, 0, 1, 2).unsqueeze(0)   # [1,C,T,H,W] uint8
    video = video.to(device=device)
    video = video.to(dtype) / 127.5 - 1.0             # bf16 归一化
    return video.contiguous(), fps


device = "cuda:%d" % int(sys.argv[1])
videos_file = sys.argv[2]

torch.cuda.set_device(int(sys.argv[1])) 

vae_path = "/nfs-speech-cfs/wangzhou/tts/SoulX-FlashHead/models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
root_dir = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards/"

vae = WanVAE(
    vae_path=vae_path,
    dtype=torch.bfloat16,
    device=device,
    parallel=False,
)
vae.model.encoder = torch.compile(vae.model.encoder, mode="default")

root_dir = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards/"
video_path = root_dir + "video/seamless/part63/video/naturalistic_train_0167_0053_V03_S1789_I00000279_P5105__V03_S1789_I00000279_P5105_12.mp4"
video, fps = read_full_video(video_path, device=device)
with torch.inference_mode():
    x = vae.encode(video)

with open(videos_file, 'r') as f:
    for line in tqdm(f.readlines()[:100]):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        video_path = root_dir + obj['video_path']
        latent_path = video_path.replace(".mp4", ".pt")

        # 断点续传：已处理跳过
        import os
        # if os.path.exists(latent_path):
        #     continue

        video = x = None
        try:
            video, fps = read_full_video(video_path, device=device)
            with torch.no_grad():
                x = vae.encode(video)              # 确认返回 tensor

            # 若 x 是 list（有的 VAE 返回 list），取第一个
            if isinstance(x, (list, tuple)):
                x = x[0]

            torch.save(x.detach().cpu(), latent_path)
            print(f"[ok] {video_path} -> {x.shape[1]/6.25}", flush=True)
        except Exception as e:
            print(f"[fail] {video_path}: {e}", flush=True)
        finally:
            del video, x
            torch.cuda.empty_cache()