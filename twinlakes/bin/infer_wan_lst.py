import torch

import torch.nn.functional as F
from decord import VideoReader, cpu
import imageio
from tqdm import tqdm

from twinlakes.vae.wan import WanVAE
import sys
# import json
# import torchaudio
# import math
# import os
# _AUDIO_BACKEND = os.environ.get("AUDIO_BACKEND", "soundfile")




def read_full_video(video_path, device="cuda:7", dtype=torch.bfloat16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [T,H,W,C] uint8
    fps = float(vr.get_avg_fps())

    frames = torch.from_numpy(frames)              # uint8, CPU
    # [T,H,W,C] -> [C,T,H,W]，一次 permute
    video = frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,T,H,W] uint8

    # 先搬 GPU（uint8 传输量最小），再在 GPU 上转 dtype + 归一化
    video = video.to(device=device)                # uint8 on GPU
    video = video.to(dtype) / 127.5 - 1.0          # bf16 归一化
    return video.contiguous(), fps



def save_video_tensor(video, output_path, fps=25):
    """
    video:
        [1, 3, T, H, W] 或 [3, T, H, W]
        数值范围通常为 [-1, 1]
    """
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]

    assert video.ndim == 4
    assert video.shape[0] == 3

    # [-1, 1] -> [0, 255]
    video = (
        (video.float().clamp(-1, 1) + 1.0)
        * 127.5
    ).round().to(torch.uint8)

    # [C, T, H, W] -> [T, H, W, C]
    frames = video.permute(1, 2, 3, 0).cpu().numpy()

    with imageio.get_writer(
        output_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        ffmpeg_params=["-crf", "18"],
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


device = "cuda:4"
torch.cuda.set_device(4) 

vae_path = "/nfs-speech-cfs/wangzhou/tts/SoulX-FlashHead/models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
root_dir = "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards/"
video_path = root_dir + "video/seamless/part63/video/naturalistic_train_0167_0053_V03_S1789_I00000279_P5105__V03_S1789_I00000279_P5105_12.mp4"

vae = WanVAE(
    vae_path=vae_path,
    dtype=torch.bfloat16,
    device=device,
    parallel=False,
)
# vae.model = torch.compile(vae.model, mode="max-autotune")
vae.model.encoder = torch.compile(vae.model.encoder, mode="default")

video, fps = read_full_video(video_path, device=device)
print("d1", flush=True)

import time
t0 = time.time()
with torch.inference_mode():
    x = vae.encode(video)
torch.cuda.synchronize(device)
t1 = time.time()

for i in tqdm(range(1000)):
    with torch.inference_mode():
        # videos = torch.stack([videos, v2, v3]) 
        x = vae.encode(video)
torch.cuda.synchronize(device)
t2 = time.time()

print(t2-t1, t1-t0, flush=True)

torch.save(x.detach().cpu(), "/nfs-speech-cfs/wangzhou/s2s/vibehead/2.pt")

# print(x.shape, x[0,0,0,:], x[-1,0,0,:])
# print("d2", flush=True)
# video_r = vae.decode(x)
# print("d3", flush=True)
# save_video_tensor(video=video_r, output_path="/nfs-speech-cfs/wangzhou/s2s/vibehead/1.mp4")
