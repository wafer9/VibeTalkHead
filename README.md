## LLM + Dit
  输入：image，wan_vae_encoder + vibevoice_vae_encoder
  输出：hidden_state -> audio_dit/video_dit



○ 数字人现状：
    ■ 为了做到实时可用，先从简单的talking head 入手，视频的像素是512*512，消耗资源较少；
    ■ FlashHead，开源了模型以及训练数据vividHead，RTX4090 显卡可以做到fps=96，效果的话头部动作幅度比较小，看着有点呆。
    ■ FlashHead基于扩散模型，连续视频生成，使用chunk的方式，最后几帧作为下一个chunk推理的参考。
    ■ 基于chunk的方式，可以认为是块级别的llm自回归，但没有用到长的历史信息，可以改进下。
○ 计划：
    ■ 基于llm + dit的方式重训一版看看效果，llm+dit 使用vibevoice方案
    ■ audio token使用vibevoice 的VAE，帧率为7.5Hz，video token使用LIA-X，25fps视频，特征维度为25*40；
    ■ 训练了一版audio token作为输入，预测video lia-x token，效果不错，sync-c和sync-d和flashhead接近，但fid和fvd较差。
    ■ 分析原因可能和lia-x能力有关，替换lia-x为wan2.1 VAE,但1s 视频，wanVAE 特征为（6.25,16,64,64），如何作为一帧送到llm是个问题；

## 实时版 WAN-Latent LLM + Video DiT 实现

训练清单使用 `data/talker_vivid/train.json`。其中 `wanvae` 指向离线特征，磁盘格式为
按通道 int8 量化的 `q[16,T,64,64] + scale[16]`；dataset 在 CPU 上反量化并转成
`[T,16,64,64]`。首个 temporal latent `z_0` 作为固定 reference，训练目标为
`z_1 ... z_{T-1}`。

### WanLocEnc：每帧一个 LLM token

每个 6.25Hz Wan temporal slice 独立编码：

```
z_t [16,64,64]
  -> Conv2d patchify, patch=4
  -> 16x16=256 spatial tokens, dim=384
  -> 加 row/column 2D position
  -> prepend CLS
  -> 2 层 non-causal local Transformer
  -> 取 CLS 并投影到 LLM hidden size
```

LocEnc 不负责无损保存画面，而是向因果 LLM 表达嘴型、表情、姿态和运动状态。
训练采用 teacher forcing：`z_t` 的 LocEnc embedding 回填到 video slot，由前一个位置的
LLM hidden state 预测 `z_t`，避免当前帧信息泄漏。

### RealtimeVideoDiT：恢复完整空间 latent

每个自回归时刻生成一张 `[16,64,64]` latent：

```
noisy z_t [16,64,64] -----+
reference z_0 [16,64,64] -+-> channel concat [48,64,64]
previous z_{t-1}          -+-> 4x4 patchify -> 256 tokens
                                      |
LLM hidden h_{t-1} + flow timestep -> AdaLN-Zero
                                      |
                              8-layer / dim-512 DiT
                                      |
                         unpatchify -> velocity [16,64,64]
```

reference 通过空间对齐的 channel concat 保持身份与背景；previous latent 保持局部纹理和
运动连续性；LLM hidden 负责长历史、语音驱动和高层动作。单个 hidden token 不做长度为 1
的 cross-attention，而是与 flow timestep 相加后调制每个 DiT block 的 attention/FFN。

训练目标采用 rectified flow / CFM：

```
noise ~ N(0,I), t ~ U(0,1)
z_t_noisy = (1-t) * noise + t * target
velocity_target = target - noise
loss = MSE(VideoDiT(z_t_noisy, t, conditions), velocity_target)
```

推理从高斯噪声出发，默认 8 步 Euler；CFG 只丢弃 LLM hidden，reference 与 previous
在条件/无条件分支中保持一致。

### 实时与显存策略

- `patch=4` 将全注意力长度从 1024 降到 256，理论 attention 矩阵缩小 16 倍。
- DiT 默认 8 层、hidden 512，不复制 SoulX Pro 的 30 层/1536 维大模型。
- 一个 batch 内所有抽中的视频帧并行执行一次 CFM forward，不按时间逐帧调用 DiT。
- LLM 和 LocEnc 仍 teacher-force 完整序列；DiT 对每条长视频随机抽 8 帧计算 loss，避免
  显存随视频时长线性膨胀，跨 epoch 覆盖全部帧。
- 预训练 LLM 默认学习率 `1e-5`，新建 LocEnc/VideoDiT 默认 `1e-4`，避免同一高学习率
  破坏 LLM 已有的时序表示。
- 推理时 LLM 使用 KV cache 逐帧推进；每生成一个完整 latent，经同一个 WanLocEnc 压成
  下一枚 video token。

核心代码：

- `twinlakes/models/video_dit.py`：WanLocEnc、AdaLN-Zero Video DiT、patchify/unpatchify。
- `twinlakes/models/rq_transformer.py`：音频/视频 embedding 回填、因果 shift、CFM loss 和单帧采样。
- `twinlakes/dataset/processor.py`：`train.json` 音频与量化 Wan latent 加载、变长 batch padding。
- `conf/run_stage1.yaml`：实时版默认模型规模。

训练命令（暂未单独切分验证集时可先用同一清单做链路验证）：

```
torchrun --nproc_per_node=<GPU数> -m twinlakes.bin.train \
  --config conf/run_stage1.yaml \
  --model_dir exp/wan_realtime \
  --data_type raw \
  --train_data data/talker_vivid/train.json \
  --cv_data data/talker_vivid/train.json \
  --num_workers 4 --pin_memory
```

正式训练应从 `train.json` 固定切出独立 CV 清单，不能长期用训练集报告验证 loss。
