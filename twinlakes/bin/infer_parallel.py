# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from dataclasses import dataclass
import random
import copy
import yaml
import numpy as np
import torch
import torchaudio
import math
import os
import time
import multiprocessing
import json

from twinlakes.models.rq_transformer import LMModel

from twinlakes.utils.checkpoint import load_checkpoint
from tqdm import tqdm
from vibevoice.processor.vibevoice_processor import AudioNormalizer
audio_normalizer = AudioNormalizer()

SAMPLE_RATE=24000
def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--result_dir', required=True, help='asr result file')

    args = parser.parse_args()
    print(args)
    return args


def asr(chunk, i, gpu_lst, configs, result_dir):
    checkpoint = configs['checkpoint']
    del configs['checkpoint']
    configs['freeze_diffusion_head'] = False

    from twinlakes.models.rq_transformer import LMModel
    model = LMModel.from_audio_text_pretrained(configs)
    infos = load_checkpoint(model, checkpoint)
    # device = int(os.environ.get('LOCAL_RANK', 0))
    device = torch.device("cuda:%d" % (i % len(gpu_lst)))
    model = model.to(device)
    # print(device, model.device)

    model.eval()

    cfg_scale = float(result_dir.split('_')[-1])

    speech_tok_compress_ratio = 3200
    _n_done, _tot_gen, _tot_audio = 0, 0.0, 0.0  # RTF 统计(跳过首条 warmup)
    for line in tqdm(chunk):
        obj = json.loads(line)
        key = obj['key']
        wav = obj['wav_path']
        waveform, sp = torchaudio.load(wav)
        if sp != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sp, SAMPLE_RATE)(waveform)
            sp = SAMPLE_RATE
        assert sp == 24000
        waveform = torch.from_numpy(waveform[0].numpy()).unsqueeze(0) # audio_normalizer

        vae_audio_tok_len = math.ceil(waveform.shape[1] / speech_tok_compress_ratio)
    

        prompt = "<|image_pad|>\n \
            Voice input:<|vision_start|>\%s<|vision_end|>\n \
            Video output:\n<|vision_start|>" % ("<|vision_pad|>" * vae_audio_tok_len)

        input_ids = torch.tensor(model.tokenizer.encode(prompt), dtype=torch.int64).unsqueeze(0)


        start_pos = torch.where(input_ids == model.tokenizer.speech_start_id)
        end_pos = torch.where(input_ids == model.tokenizer.speech_end_id)
        audio_pos = torch.stack((start_pos[1][0], end_pos[1][0], start_pos[1][1]), dim=0)

        wavs = waveform.to(device=device, dtype=torch.bfloat16).unsqueeze(2)
        wavs_lengths = torch.tensor([waveform.shape[1]]).to(device=device)
        input_ids = input_ids.to(device)

        torch.cuda.synchronize(device)
        _t0 = time.time()
        wave = model.generate_cfg(
                            keys=[key],
                            wavs=wavs,
                            wavs_lengths=wavs_lengths,
                            input_ids=input_ids,
                            audio_pos=[audio_pos],
                            cfg_scale=cfg_scale,
                            )
        torch.cuda.synchronize(device)
        _dt = time.time() - _t0
        _audio_s = wave.shape[-1] / 24000.0            # 输出音频时长(24kHz)
        if _n_done >= 1:                                # 跳过第 1 条(CUDA/kernel warmup)
            _tot_gen += _dt
            _tot_audio += _audio_s
        _n_done += 1
        torchaudio.save('%s/%s.wav' % (result_dir, key), wave, sample_rate=24000)

    if _tot_audio > 0:
        rtf = _tot_gen / _tot_audio
        print("[RTF][rank %d] method=%s cfg=%s utts=%d gen=%.1fs audio=%.1fs RTF=%.4f"
              % (i, uncond_type, cfg_scale, _n_done - 1, _tot_gen, _tot_audio, rtf), flush=True)
        



def main():
    args = get_args()
    seed=777
    generator = torch.Generator()
    generator.manual_seed(seed)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    configs['checkpoint'] = args.checkpoint
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))

    data = []
    with open(args.test_data, 'r') as f:
        for line in f.readlines():
            data.append(line.strip())

    num = math.ceil(len(data)/float(local_world_size))
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(args.result_dir, exist_ok=True)

    # Using thread pool to speedup
    gpu_lst = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')]

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    asr(chunks[local_rank], local_rank, gpu_lst, configs, args.result_dir)


with torch.no_grad():
    main()
