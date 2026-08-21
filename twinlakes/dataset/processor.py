# Copyright (c) 2021 Wenet Community. (authors: Binbin Zhang)
#               2023 Wenet Community. (authors: Dinghao Zhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import json
from subprocess import PIPE, Popen
from urllib.parse import urlparse
import logging
import librosa
import random

import torch
from torch.nn.utils.rnn import pad_sequence
try:
    import torchaudio
except ImportError:  # latent-only training can use the lightweight soundfile fallback
    torchaudio = None
import soundfile as sf
from scipy.signal import resample_poly
from twinlakes.utils.mask import make_pad_mask
import math
import base64
import numpy as np
# import whisper

try:
    if torchaudio is None:
        raise AttributeError
    torchaudio.utils.sox_utils.set_buffer_size(16500)
except AttributeError:
    pass
import os
try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = cpu = None
from twinlakes.dataset.latent_io import load_latent_zlib

# Motion normalization stats (40-dim), 复用自 Talker-T2AV 的 LIA-X motion 统计。
# 多来源 (seamless/mead/... ) motion 尺度不一, 不归一化会让 diffusion loss 虚高、
# 且高幅度来源主导梯度、加噪信噪比与 scheduler 假设错配。
_MOTION_STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats")
MOTION_MEAN = np.load(os.path.join(_MOTION_STATS_DIR, "motion_mean.npy")).astype(np.float32)  # (40,)
MOTION_STD = np.load(os.path.join(_MOTION_STATS_DIR, "motion_std.npy")).astype(np.float32)    # (40,)
MOTION_STD = np.clip(MOTION_STD, a_min=1e-6, a_max=None)
_MOTION_MEAN_T = torch.from_numpy(MOTION_MEAN)
_MOTION_STD_T = torch.from_numpy(MOTION_STD)

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])
from vibevoice.processor.vibevoice_processor import AudioNormalizer
audio_normalizer = AudioNormalizer()
SAMPLE_RATE=24000
_AUDIO_BACKEND = os.environ.get("AUDIO_BACKEND", "soundfile")

logging.getLogger('langid').setLevel(logging.INFO)



def get_T_after_pool(L_in, dilation=1):
    conv = "[(1,3,1)] + [(1,3,2)] + [(1,2,2)]"
    for (padding, kernel_size, stride) in eval(conv):
        L_out = L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        L_out = 1 + L_out // stride
        L_in = L_out
    return L_out


def get_fbank_samples_for_T(T, hop=160):
    """反推: 16k 音频需多少采样, 使 fbank 帧数 L 过 get_T_after_pool 后恰好为 T。

    对当前 conv 链 [(1,3,1),(1,3,2),(1,2,2)], L = 4T-2, 采样数 = L * hop。
    逐层反推 (取满足 L_out 的最大 L_in): L_in = stride*L_out - 2*padding + kernel_size - 1。
    """
    conv = "[(1,3,1)] + [(1,3,2)] + [(1,2,2)]"
    L = T
    for padding, kernel_size, stride in reversed(eval(conv)):
        L = stride * L - 2 * padding + kernel_size - 1
    return L * hop


import os
try:
    cpu_info = os.popen("lscpu | grep 'Vendor ID'").read()
    # 0x48 --> HiSilicon
    if (cpu_info.rstrip().split(" ")[-1] == "0x48"):
        # NOTE (MengqingCao): set number of threads in the subprocesses to 1
        # Why? There may be some operators ultilizing multi-threads in processor,
        # causing possibly deadlock in Kunpeng.
        # Similar issue in PyTorch: https://github.com/pytorch/pytorch/issues/45198
        torch.set_num_threads(1)
except Exception as ex:
    logging.warning('Failed to set number of thread in Kunpeng, \
        this may cause segmentfault while dataloading, \
        ignore this warning if you are not using Kunpeng')


class UrlOpenError(Exception):

    def __init__(self, msg: str, *args: object) -> None:
        super().__init__(*args)
        self.err_msg = msg

    def __str__(self) -> str:
        return self.err_msg


def parse_json(elem):
    line = elem['line']
    obj = json.loads(line)
    obj['file_name'] = elem['file_name']
    return dict(obj)


def parse_url(elem):
    assert 'file_name' in elem
    assert 'line' in elem
    assert isinstance(elem, dict)
    url = elem['line']
    try:
        pr = urlparse(url)
        # local file
        if pr.scheme == '' or pr.scheme == 'file':
            stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
        else:
            cmd = f'wget -q -O - {url}'
            process = Popen(cmd, shell=True, stdout=PIPE)
            elem.update(process=process)
            stream = process.stdout
        elem.update(stream=stream)
        return elem
    except Exception as ex:
        err_msg = 'Failed to open {}'.format(url)
        raise UrlOpenError(err_msg) from ex


def decode_wav(sample, max_per_line=None):
    """ Parse a json line holding a list of clips from one speaker and expand
        it into multiple training samples.

        Each chosen clip is used once as the target (wav + text); its voice
        prompt is another randomly picked clip from the same list. Only the
        clips actually used (targets + their prompts) are decoded, once each,
        so per-line IO / json parse is amortized without wasting decode CPU.

        Args:
            sample: dict with 'key' and 'text' (a json line carrying wavs/texts)
            max_per_line: cap on how many target samples to emit per line; None
                means use every clip as a target.

        Yields:
            {key, wav, prompt_wav, sample_rate, text}
    """
    assert 'key' in sample
    assert 'text' in sample

    obj = json.loads(sample['text'])
    wavs, texts = obj['wavs'], obj['texts']
    assert len(wavs) == len(texts) and len(wavs) >= 2
    n = len(wavs)

    # pick which clips serve as targets
    if max_per_line is not None and max_per_line < n:
        tgt_indices = random.sample(range(n), max_per_line)
    else:
        tgt_indices = list(range(n))

    # assign a distinct prompt to each target, then decode only what's needed
    prompt_of = {t: random.choice([j for j in range(n) if j != t])
                 for t in tgt_indices}
    cache = {}
    for idx in set(tgt_indices) | set(prompt_of.values()):
        with io.BytesIO(base64.b64decode(wavs[idx])) as file_obj:
            waveform, sample_rate = torchaudio.load(file_obj)
            if sample_rate != SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
        cache[idx] = waveform

    for tgt_idx in tgt_indices:
        yield {
            'key': '{}_{}'.format(sample['key'], tgt_idx),
            'wav': cache[tgt_idx],
            'prompt_wav': cache[prompt_of[tgt_idx]],
            'sample_rate': SAMPLE_RATE,
            'text': texts[tgt_idx],
        }

def read_full_video(video_path):
    if VideoReader is None:
        raise ImportError("decord is required when decoding video instead of loading WAN latents")
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [T,H,W,C] uint8
    fps = float(vr.get_avg_fps())

    frames = torch.from_numpy(frames)              # uint8, CPU
    # [T,H,W,C] -> [C,T,H,W]，一次 permute
    video = frames.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,T,H,W] uint8

    # 先搬 GPU（uint8 传输量最小），再在 GPU 上转 dtype + 归一化
    video = video / 127.5 - 1.0          # bf16 归一化
    return video.contiguous(), fps


def decode_wav_raw(sample, resample=24000):
    """Load audio and an offline WAN-VAE latent from a raw JSON record.

    ``train.json`` stores int8 per-channel quantized ``.pt`` files with keys
    ``q`` and ``scale``.  Plain tensors and the older ``.ptz`` format remain
    supported so extraction jobs do not need to be repeated.
    """
    sample['key'] = sample.get('sample_id', sample.get('key'))
    root_dir = os.environ.get(
        "VIBEHEAD_DATA_ROOT",
        "/nfs-speech-cfs/wangzhou/s2s/vibehead/data/talker/shards",
    )
    wav_path = sample.get('wav_path', sample.get('audio'))
    resolved_wav = os.path.join(root_dir, wav_path)
    if torchaudio is not None:
        waveform, sample_rate = torchaudio.load(resolved_wav, backend=_AUDIO_BACKEND)
        if sample_rate != resample:
            waveform = torchaudio.transforms.Resample(sample_rate, resample)(waveform)
            sample_rate = resample
    else:
        audio_np, sample_rate = sf.read(resolved_wav, dtype='float32', always_2d=True)
        waveform = torch.from_numpy(audio_np.T.copy())
        if sample_rate != resample:
            gcd = math.gcd(sample_rate, resample)
            audio_np = resample_poly(waveform.numpy(), resample // gcd, sample_rate // gcd, axis=1)
            waveform = torch.from_numpy(audio_np.copy()).float()
            sample_rate = resample
    assert sample_rate == resample
    sample['wav'] = waveform[:1]
    sample['sample_rate'] = sample_rate

    latent_path = sample.get('wanvae', sample.get('latent'))
    if latent_path is None:
        raise KeyError("sample has neither 'wanvae' nor 'latent'")
    if latent_path.endswith('.ptz'):
        latent = load_latent_zlib(latent_path)
    else:
        obj = torch.load(latent_path, map_location='cpu', weights_only=True)
        if isinstance(obj, torch.Tensor):
            latent = obj
        elif isinstance(obj, dict) and 'q' in obj and 'scale' in obj:
            q = obj['q']
            scale = obj['scale'].float().reshape(-1, 1, 1, 1)
            latent = q.float().mul_(scale).to(torch.bfloat16)
        else:
            raise ValueError("unsupported WAN latent payload: {}".format(type(obj)))
    if latent.ndim == 5 and latent.shape[0] == 1:
        latent = latent.squeeze(0)
    if latent.ndim != 4 or latent.shape[0] != 16:
        raise ValueError("expected WAN latent [16,T,64,64], got {}".format(tuple(latent.shape)))
    # Time-major layout lets pad_sequence batch variable-duration clips.
    sample['video_latent'] = latent.to(torch.bfloat16).permute(1, 0, 2, 3).contiguous()

    return sample


def num_merged_motion_frames(wav_samples, speech_tok_compress_ratio=3200,
                             audio_hz_num=6, motion_hz_num=5):
    """由音频采样数反推合并后 (6.25Hz, 160维) 的 motion 帧数 T_v。

    确定性映射: A = ceil(wav_samples / 3200) 个 audio token (7.5Hz),
    merged motion 6.25Hz, 速率比 audio:motion = 6:5 -> T_v = round(A * 5/6)。
    推理端据此直接算出自回归 (denoise) 需要的帧数, 无需读 motion 文件。
    """
    audio_tok = math.ceil(wav_samples / speech_tok_compress_ratio)
    # round(A * 5/6): 用整数运算避免浮点抖动
    return (audio_tok * motion_hz_num + audio_hz_num // 2) // audio_hz_num


def merge_motion_frames_by_audio(video, wav_samples, merge=4):
    """(T,40)@25fps -> (T_v,160)@6.25fps, T_v 由音频长度确定 (裁/补对齐)。

    Args:
        video: (3, T, H, W) 已归一化的 25fps motion。
        wav_samples: 音频采样数 (24k), 用来算目标帧数。
        merge: 合并帧数 (4)。
    Returns:
        (T_v, 40*merge) float32, T_v = num_merged_motion_frames(wav_samples)。
    """
    T_v = num_merged_motion_frames(wav_samples, merge=merge)
    need = T_v * merge  # 合并前需要的 25fps 帧数
    _, T, dim, _ = motion.shape
    if T < need:  # motion 偏短: 尾帧重复补齐 (静止, 优于 0 填充造成的突跳)
        pad = motion[:, -1, :, :].expand(need - T, dim)
        motion = torch.cat([motion, pad], dim=1)
    else:         # motion 偏长: 直接裁到 need
        motion = motion[:,:need, :, :]
    # (3, need, dim, dim) -> (T_v, merge, 40) -> (T_v, merge*40=160)
    return motion.reshape(T_v, merge * dim).contiguous()




def filter(sample, min_frames=10, max_frames = 3000):
    num_frames = sample['wav'].shape[1]/sample['sample_rate'] * 100
    if num_frames < min_frames:
        return False
    if num_frames > max_frames:
        return False
    if sample.get('video_latent') is None or sample['video_latent'].shape[0] < 2:
        return False

    return True


def sort_by_feats(sample):
    return sample['wav'].shape[1]


def tokenize(sample, tokenizer):
    speech_tok_compress_ratio = 3200
    

    # First WAN frame is the fixed identity reference; AR targets start at t=1.
    vae_audio_tok_len = math.ceil(sample['wav'].shape[1] / speech_tok_compress_ratio)
    vae_video_tok_len = sample['video_latent'].shape[0] - 1

    # # 2. 各自能凑出多少个完整 chunk, 取较小值(木桶原理)
    # num_chunks = min(vae_audio_tok_len // 6, vae_video_tok_len // 5)

    # # 3. 根据公共 chunk 数, 反算对齐后的 token 数
    # audio_tok_clipped = num_chunks * 6
    # video_tok_clipped = num_chunks * 5

    # # 4. 裁剪音频和视频 latent (latent 域直接按帧裁)
    # sample['wav'] = sample['wav'][:, :audio_tok_clipped * speech_tok_compress_ratio]
    # sample['video_latent'] = sample['video_latent'][:, :video_tok_clipped]

    # # 5. 校验对齐
    # assert math.ceil(sample['wav'].shape[1] / speech_tok_compress_ratio) == audio_tok_clipped
    # assert sample['video_latent'].shape[1] == video_tok_clipped
    # assert audio_tok_clipped // 6 == video_tok_clipped // 5 == num_chunks

    prompt = "<|image_pad|>\n Voice input:<|vision_start|>%s<|vision_end|>\n Video output:\n<|vision_start|>" % ("<|vision_pad|>" * vae_audio_tok_len)

    
    label = "<|vision_pad|>" * vae_video_tok_len + "<|vision_end|>"

    prompt, label = [prompt], [label]

    encoding = tokenizer(
                        text=prompt,
                        text_pair=label,
                        add_special_tokens=True,
                        padding=True,  # truncation
                        return_tensors="pt",
                        return_token_type_ids=True,
                        return_attention_mask=True)
    token_type_ids = encoding["token_type_ids"]
    sample['labels'] = encoding["input_ids"].clone()
    sample["labels"][token_type_ids == 0] = -100
    prompt, label = prompt[0], label[0]

    sample['input_ids'] = encoding["input_ids"]

    start_pos = torch.where(encoding["input_ids"] == tokenizer.speech_start_id)
    end_pos = torch.where(encoding["input_ids"] == tokenizer.speech_end_id)
    audio_pos = torch.stack((start_pos[1][0], end_pos[1][0], start_pos[1][1], end_pos[1][1]), dim=0)

    wav = torch.from_numpy(audio_normalizer(sample['wav'][0].numpy())).unsqueeze(0)
    batch = {
        "key": sample['key'],
        "prompt": prompt,
        "label": label,
        "input_ids": sample['input_ids'][0],
        "label_ids": sample["labels"][0] if 'labels' in sample else torch.zeros([1, 0]),
        "wav": wav,
        "audio_pos": audio_pos,
        "video_latent": sample['video_latent'],  # [T,16,64,64], includes reference frame
    }
    return batch


def padding(data, ):

    sample = data
    assert isinstance(sample, list)
    input_ids_length = torch.tensor([x['wav'].shape[1] for x in sample], dtype=torch.int32)
    order = torch.argsort(input_ids_length, descending=True)

    keys = [sample[i]['key'] for i in order]
    wavs = [sample[i]['wav'].transpose(0,1) for i in order]
    video_latents = [sample[i]['video_latent'] for i in order]

    padded_wavs = pad_sequence(wavs, batch_first=True, padding_value=0)
    wavs_lengths = torch.tensor([sample[i]['wav'].size(1) for i in order], dtype=torch.int32)

    padded_video_latents = pad_sequence(video_latents, batch_first=True, padding_value=0)
    video_latent_lengths = torch.tensor(
        [sample[i]['video_latent'].shape[0] for i in order], dtype=torch.int32
    )

    input_ids = [sample[i]['input_ids'] for i in order]
    label_ids = [sample[i]['label_ids'] for i in order]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    audio_pos = [sample[i]['audio_pos'] for i in order]


    batch = {
        "keys": keys,
        "input_ids": input_ids,
        "label_ids":label_ids,
        "wavs": padded_wavs,
        "wavs_lengths": wavs_lengths,
        "video_latents": padded_video_latents,
        "video_latent_lengths": video_latent_lengths,
        "audio_pos": audio_pos,
    }
    return batch


class DynamicBatchWindow:

    def __init__(self, max_frames_in_batch=12000):
        self.longest_frames = 0
        self.max_frames_in_batch = max_frames_in_batch

    def __call__(self, sample, buffer_size):
        assert isinstance(sample, dict)
        assert 'input_ids' in sample
        assert isinstance(sample['input_ids'], torch.Tensor)
        # new_sample_frames = sample['feat'].size(0)
        new_sample_frames = sample['input_ids'].size(0)
        self.longest_frames = max(self.longest_frames, new_sample_frames)
        frames_after_padding = self.longest_frames * (buffer_size + 1)
        if frames_after_padding > self.max_frames_in_batch:
            self.longest_frames = new_sample_frames
            return True
        return False
