• 测完了，严格使用 HDTF test.jsonl 前 60 条：

  - LIA-X GT-motion renderer 上界
  - FID：7.545
  - FVD：50.988
  - 评测帧数：4860 对
  - FVD：60 clips × 16 帧
  - 口径：整帧、不做人脸裁剪

   Noise σ    FID ↓      FVD ↓
  ━━━━━━━━━  ━━━━━━━  ━━━━━━━━━
         0    7.545     50.988
  ─────────  ───────  ─────────
      0.05    7.281     66.126
  ─────────  ───────  ─────────
      0.10    7.243    105.236
  ─────────  ───────  ─────────
      0.20    7.913    203.475

  - σ≤0.10 时 FID 几乎不变，说明小幅 latent 越界不会直接导致单帧明显变糊或崩脸。
  - FVD 对误差极其敏感：σ=0.10 已从 50.99 上升到 105.24。
  - σ=0.20 时 FVD 达到 203.48，逐帧随机误差造成严重抖动。
  - 因此当前 LIA-X 更主要的问题是 motion latent 的时间稳定性，而不只是 AE latent 分布外导致的单帧模糊。



✅ 保留"相对参考"的建模 —— z 编码运动，decoder 从 reference 拿身份/内容。这天然解决"内容稀释 audio 信号"的问题。
✅ 保留自监督 + 不同帧做 reference —— 沿用 LIA 的解耦机制。
✅ 砍掉 LIA-X 的姿态控制/正交分解 —— 你不需要，模型能小很多。
✅ 把正交字典约束换成 KL 约束 —— 这是你的核心创新：从"非正态难预测"变成"近高斯好预测"。
✅ 让 z 尽量 source-independent —— audio→运动的映射更干净。
✅ 加 LPIPS+GAN 提 FID + latent 加噪训练提 FVD 鲁棒性（前面讨论的）。


decoder 带因果时序状态（causal attention 或 RNN/state），解码第 t 帧时能看到过去的帧/latent。

如果坚持 LIA 那种"随机抽两帧"的训练方式，方案 B 的因果时序没法训——因为时序建模需要连续帧序列，而随机抽两帧没有时序上下文。
解法：训练时用"连续帧片段 + 一个跨片段的 reference"

VibeVoice 的 σ-VAE 思路：
对你的 motion latent 场景，我更推荐只保留 mean，不学习 logvar。这样每个视频帧对应唯一 latent，DiT 监督目标稳定，也不会因为逐帧 posterior sampling 引入额外时序噪声。
不过准确地说，VibeVoice 是“encoder 主要预测 mean，方差固定或关闭”，而不是所有分支都完全没有方差：官方代码的 encoder output 是 mean + fixed std，acoustic tokenizer 可以固定 σ 采样，semantic tokenizer 可以配置成只用 mean。

一个重要细节
不要在最终 latent 上使用逐样本 LayerNorm：y = LayerNorm(mu)
因为它会强制每一帧 latent 自身均值为 0、方差为 1，可能消除 motion amplitude，例如张嘴幅度和头部转动幅度。
应该使用数据集级别的逐通道统计：
最终建议就是：
  使用 64 维 deterministic mean-only motion autoencoder；删除 logvar 和逐样本 VAE KL；使用 corpus normalization + aggregate mean/std/cov regularization；DiT 预测确定性 mean latent；通过显式噪声重建和 Causal Motion Adapter 获得 FVD 鲁棒性。


Relative Motion VAE Encoder
    (x1, xt) → μt, logvar_t
    DiT预测标准化 μt

Residual Causal Motion Adapter
    μt + gated causal context → μ't

Source-Anchored Flow-Render Decoder
    reference features缓存一次
    每帧从同一个reference重新warp
    32/64 temporal flow residual可选

Training
    clean + noisy reconstruction
    LPIPS + identity + Sync
    weak KL
    temporal relation + warp
    image GAN + video GAN


---

# 方案整理与最终建议：先训练 Motion Tokenizer，再训练 LLM + Motion DiT

## 1. 先把目标说清楚

当前阶段不再直接使用 WanVAE 的 `[16, 64, 64]` 空间特征训练 LLM + DiT。WanVAE latent
同时包含身份、纹理、背景、光照和运动，一帧需要大量 spatial token；音频真正强相关的却主要是嘴型、
表情、眼神和头部运动。直接学习 audio -> Wan latent，不仅计算重，而且很容易让模型把容量浪费在
不可由音频确定的高频外观上。

新的路线分成两个完全解耦的阶段：

1. 先训练一个简化但更适合生成建模的新版 LIA-X，学习紧凑、连续、尽量身份无关的 motion latent。
2. 冻结 motion encoder、renderer 和归一化统计，用 motion latent 训练 LLM + 小型 Motion DiT。

这里真正要优化的不是“单帧重建最好看”这一个目标，而是同时满足四个条件：

- **Expressive**：64D latent 足以表达嘴型、非唇部表情、眨眼、视线和头部姿态。
- **Identity-agnostic**：人物身份、肤色、发型、背景和纹理应来自 reference，而不是 motion latent。
- **Predictable**：latent 数值尺度稳定、无极端长尾、同一种运动跨人物尽量接近，方便 DiT 学习。
- **Robust and temporal**：小的 latent 预测误差不能被 renderer 放大成逐帧抖动，同时不能靠过度平滑牺牲 lip-sync。

现有噪声实验已经说明：原 LIA-X 在 `sigma <= 0.10` 时单帧 FID 基本不变，但 FVD 从
`50.99` 上升到 `105.24`。因此首要矛盾确实是 **renderer 对逐帧 motion 误差的时间敏感性**，
而不是简单的“latent 一越界，单帧就崩”。`sigma=0.05` 时 FID 略低于 clean 不应解读为噪声有益，
更可能是 FID 方差或轻微平滑效应；最终结论应以更大的固定测试集和置信区间为准。

## 2. 对前面几个想法的最终取舍

| 问题 | 最终选择 | 原因 |
|---|---|---|
| WanVAE 还是 motion latent | motion latent | 音频只需建模运动；身份和高频细节交给 reference renderer |
| 随机 VAE 还是确定性 AE | **deterministic mean-only** | 每帧只有唯一监督目标，避免 posterior sampling 给 DiT 额外制造时间噪声 |
| 是否学习 `logvar` / 逐样本 KL | **不使用** | DiT 并不要求数据本身服从高斯；强 KL 可能损伤运动幅度、细节和时序连续性 |
| 是否保留正交/稀疏 motion dictionary | **删除** | 它主要服务可解释编辑，不是 audio-to-motion 预测的必要条件，还限制表示能力并增加计算 |
| 是否对最终 latent 做 LayerNorm | **不做** | 逐帧 LayerNorm 会抹掉张嘴幅度、姿态幅度等有意义的全局尺度 |
| latent 维度 | **先固定 64D** | 比当前 40D 留出表达余量，仍足够紧凑；只有 64D 上界不足时才试 96D |
| motion encoder 是否带时序上下文 | **encoder 保持逐帧、无状态** | 同一图像永远得到同一 code，方便离线提取、缓存、诊断和后续生成 |
| renderer 是否看历史 | **只让小型 latent adapter 看过去** | 利用短时上下文抑制抖动，同时避免递归使用上一张 RGB 导致漂移 |
| 训练采样 | **连续片段 + 一个固定 reference** | 随机两帧无法训练 temporal loss、causal adapter 和 video discriminator |
| GAN | 先 image GAN，稳定后再加短片段 video GAN | image GAN 提升纹理，video GAN 解决短时闪烁；从第 0 步一起上容易不稳定 |
| Sync loss | 不是第一阶段主损失 | tokenizer 首先应从真实视频完整保存嘴部运动；仅当 GT-latent 重建的 Sync 指标不足时再加 |

需要修正一个概念：**deterministic mean-only + corpus normalization 不是 VAE**。数据集级
mean/std 只能改善数值条件，mean/std/cov 约束也不能保证联合分布是高斯。好消息是 diffusion/flow
模型本来就可以学习非高斯数据分布，因此没有必要为了“看起来像 `N(0,I)`”而破坏 motion manifold。

## 3. 推荐的新版 LIA-X：Strict Two-Branch Motion Autoencoder

### 3.1 数据流

对一个连续目标片段 `x_1 ... x_T` 和一张固定 reference `x_r`：

```text
x_r -----------------> Reference Encoder E_app -> 多尺度外观特征 A_r（缓存一次）
  |                                                        |
  +-----------------> Motion Encoder E_m -----> m_r        |
                                                           v
x_t -----------------> Motion Encoder E_m -----> m_t -> d_t = N(m_t) - N(m_r)
                                                        |
                                                        v
                                  Residual Causal Motion Adapter
                                                        |
                                                        v
                                  Source-Anchored Flow-Render Decoder
                                                        |
                                                        v
                                                     x_hat_t
```

其中：

- `m_t in R^64` 是逐帧、确定性的 canonical motion code。
- `N` 是 motion encoder 基本定型后冻结的数据集级逐通道标准化，不是逐样本 LayerNorm。
- renderer 使用相对运动 `d_t`，但离线保存给 LLM 的目标是绝对 canonical code `N(m_t)`。
- 推理时由 reference 提供初始运动 `N(m_r)`，renderer 内部再计算相对位移。这样 DiT 不需要积分
  逐帧 delta，避免长视频误差累积。

上图表示最终导出接口。clean warm-up 时可以先在 raw `m_t-m_r` 空间训练；motion encoder 定型后统计
`N`，把第一层 motion projection 的线性权重按 std 等价折叠到标准化坐标，再训练 causal adapter 和
noise robustness。这样不会在 encoder 尚未稳定时让移动统计不断改变 renderer 的输入尺度。

### 3.2 为什么必须严格双路

原 LIA-X 先把 source 编成 512D style code，再把 40D motion 通过正交方向加回同一个 style
空间。这个设计适合线性编辑，但 appearance 和 motion 在 renderer 中仍共享同一个 latent 空间。
新版应明确规定：

- `E_app` 的多尺度特征是身份、纹理和背景的唯一来源。
- `E_m` 的 64D code 只允许通过 flow、mask 和轻量 FiLM/AdaLN 调制运动。
- motion 分支不能向 renderer 传递 target 的空间 feature map或 RGB skip feature。
- 每一帧都重新 warp 同一份 `A_r`，不能把上一张生成图当 source。

这是防止 motion latent 偷带身份和纹理的结构性约束；仅靠小维度、KL 或归一化无法保证解耦。

### 3.3 建议的具体模块

- **Motion Encoder**：轻量 2D ResNet/ConvNeXt，输入可先用 256 人脸 crop，多尺度下采样后
  global pooling，最后线性投影到 64D。保持 frame-wise，不加 RNN/temporal attention。
- **Reference Encoder**：512 输入的卷积金字塔，缓存 `32/64/128/256` 等尺度的 feature map。
- **Motion projection**：直接用 MLP 将 64D motion 投影成各尺度调制参数，不再做 QR、正交字典
  或 sparse dictionary。
- **Flow-render decoder**：保留 LIA-X 有效的多尺度 flow + mask + inpainting/render 分支，
  但每层 residual block 数量先减半。warp 负责搬运 reference 细节，render 分支补牙齿、舌头、
  遮挡和 reference 中不存在的区域。
- **Residual Causal Motion Adapter**：在 `d_t` 上使用小型 causal depthwise Conv1D、GRU，
  或 2 层 causal Transformer；上下文先取 8--16 帧。输出采用零初始化门控残差：
  `d'_t = d_t + tanh(g) * f(d_<=t)`，其中 `g` 初始化为 0。它只缓存 motion state，不缓存 RGB。

第一版不要同时加入 spatial temporal state。若 latent adapter、temporal loss 和 video GAN 后 FVD
仍明显受限，再只在 `32/64` 低分辨率 flow 上加 gated causal residual；不要在高分辨率纹理层做递归，
否则容易出现身份漂移、误差积累和 chunk 边界不一致。

## 4. 如何真正得到 source-independent motion latent

“同一视频随机取 reference/target 重建”是必要条件，但不是充分条件。因为 source 和 target 是同一个人，
encoder 仍可能把身份编码进 64D motion。推荐同时使用以下约束：

### 4.1 同身份连续片段重建

- 每个样本读取 16--32 个连续帧，reference 在整个片段中固定。
- reference 一部分取片段首帧，一部分从同一 track 的片段外随机抽取，覆盖不同姿态和时间间隔。
- photometric augmentation 可在 reference 和 target 间独立采样，以削弱肤色/光照捷径；几何增强必须
  保持定义一致，不能把随机 crop 位移误当成人脸运动。

### 4.2 Cross-identity motion cycle

从人物 A 取 reference，从人物 B 取 motion：

```text
x_A<-B = Render(E_app(x_A), E_m(x_B) - E_m(x_A))
```

没有 `x_A` 做出 B 动作的真值图像，因此不做像素监督，而做两类闭环约束：

- 冻结的人脸识别网络要求 `ID(x_A<-B)` 接近 `ID(x_A)`。
- 重新编码后要求 `E_m(x_A<-B)` 接近 stop-gradient 的 `E_m(x_B)`。

这比只加一个 identity adversarial classifier 更贴近最终 cross-identity renderer 的实际用法。
若数据有可靠 identity label，可额外训练一个只用于诊断的 motion-code identity probe；对抗去身份损失只用
很小权重，因为身份与真实说话风格存在统计相关，过强会顺带删掉有用运动。

### 4.3 Reference invariance 检查

同一个 driving 片段用同一人物的多张 reference 渲染，再重新编码 motion。不同 reference 得到的
motion 序列应接近。这个指标既可以做 loss，也必须作为 validation diagnostic，防止模型只在重建指标上好看。

## 5. 训练损失：按作用分组，而不是一次全部堆上

### 5.1 单帧保真

- Charbonnier/L1 reconstruction：保证颜色和低频结构。
- LPIPS 或 VGG perceptual：保证感知结构和局部细节。
- Identity loss：保证 reference 身份，重点用在 cross-identity cycle。
- Mouth/eye region 加权：由固定 face mask 或 landmark 区域给嘴、眼更高权重，避免全脸损失淹没小区域。

### 5.2 时序保真

不要只最小化 `|x_hat_t - x_hat_(t-1)|` 或 `|m_t-m_(t-1)|`，那会直接把真实嘴部运动也抹平。
应该匹配生成视频和 GT 视频的变化量：

```text
L_delta = |Delta phi(x_hat) - Delta phi(x_gt)|
L_accel = |Delta^2 phi(x_hat) - Delta^2 phi(x_gt)|
```

`phi` 可先用多尺度 VGG/LPIPS feature；若后续仍有明显局部滑动，再加入离线 optical-flow warping
consistency。短片段 video discriminator 在 clean reconstruction 稳定后加入，用于补充 feature-delta
loss 难以覆盖的真实时间纹理。

### 5.3 Latent 约束

- 默认不使用逐样本 KL，不预测 `logvar`，不从 posterior 采样。
- 训练早期只做很弱的 batch variance/covariance 防塌缩约束，不能强迫每个样本归一化。
- motion encoder 基本收敛后，在完整训练集统计每维 mean/std 并冻结，使用 std floor 防止小方差维爆炸；
  此后不再大幅更新 encoder。
- 默认先用逐通道标准化；只有发现强相关维明显拖慢 DiT 时，才做带 eigenvalue floor 的 PCA/ZCA
  whitening 对照实验。不要默认全白化低方差维。

### 5.4 GAN 和 Sync 的加入顺序

1. 先用 reconstruction + perceptual + identity + temporal relation 得到稳定 clean 模型。
2. 再加入 image discriminator，恢复牙齿、眼睛和皮肤的高频质量。
3. 再加入 8--16 帧 video discriminator，改善闪烁。
4. 只有当 **GT motion latent 重建** 的 Sync-C/Sync-D 已明显差于输入视频时，才加低权重 Sync loss。

如果 GT latent 上界的 lip-sync 已好，Sync 的主要责任应放在第二阶段 audio-to-motion，而不是 renderer。

## 6. Latent 噪声鲁棒训练：必须做，但要晚一点做

直接从第 0 步给 latent 加噪有两个风险：encoder 可能通过无限放大 code 尺度绕过噪声，renderer 也可能
过早学成平滑器。因此推荐：

1. 先训练 clean tokenizer。
2. 固定 corpus normalizer；robust finetune 前可先冻结 motion encoder，只训练 causal adapter 和 renderer。
3. 在**标准化后的** motion 上加噪，并仍以 clean target frame 为监督。
4. clean/noisy 两个分支同时训练，保留足够比例 clean batch，避免鲁棒性换掉上界质量。

噪声不能只有逐帧 iid Gaussian。真实 DiT 误差通常混合了几种模式：

- clip-level bias：整段轻微偏移；
- low-frequency drift：AR(1) 或低通相关噪声；
- frame-wise jitter：当前实验使用的 iid 噪声；
- 少量 outlier：个别 chunk 预测失败。

初始训练覆盖 normalized RMS `sigma=0.01--0.10`，`0.20` 只作为压力测试，不必一开始要求完全恢复。
同时加入 clean/noisy 输出一致性 loss。第二阶段跑出一个小型 LLM+DiT pilot 后，统计真实预测残差的
per-dim covariance、时间相关性和频谱，再按真实误差分布做最后一轮 renderer robust finetune。这种闭环
比凭经验一直加 iid Gaussian 更有效。

## 7. 推荐训练日程

### Phase 0：数据与评测协议先固定

- 训练输入固定为 25 fps、统一人脸 crop；清理镜头切换、多人、严重遮挡、音画错位和检测框跳变。
- 按 identity 划分 train/CV/test，不能让同一人物跨集合。
- 先在 256 分辨率验证架构和 loss，再迁移到 512；最终 renderer 和评测统一 512。
- 固定一套至少数百条视频的 held-out benchmark。当前 60 条可用于快速迭代，不能作为最终结论。

### Phase 1：clean reconstruction warm-up

- 连续 16 帧起步，固定 reference。
- 只使用 reconstruction、LPIPS/VGG、mouth/eye weight 和基本 identity loss。
- 先证明 64D bottleneck 的 clean upper bound 不弱于原 40D LIA-X。

### Phase 2：解耦和时序

- 片段扩到 16--32 帧，先加入 feature-delta/acceleration loss，不急着加入有状态 renderer。
- 加入 cross-identity cycle、reference invariance 和短片段 video discriminator。
- 定期训练 identity probe，检查 motion code 是否仍能轻易识别人。

### Phase 3：高保真 512 finetune

- motion encoder 基本定型后统计并冻结 64D corpus mean/std，再加入 zero-gated causal motion adapter。
- 加 image GAN，再加 video GAN。
- 保持 reference feature 始终从同一张图缓存，不使用上一帧 RGB。

### Phase 4：robust finetune

- 使用 clean + 多种时序结构的 noisy latent 双分支。
- 画出 `sigma=0/0.02/0.05/0.10/0.20` 的 FID、FVD、Sync、identity 和 jitter 曲线。
- tokenizer 定版后冻结 encoder、normalizer、adapter 和 renderer，再批量离线提取 motion latent。

## 8. 第二阶段：用新 motion latent 训练 LLM + DiT

### 8.1 Token 组织

保留原视频的 25 fps motion，不要让 tokenizer 自身时间降采样。沿用当前工程中已验证的分组方式：

```text
u_t = N(E_m(x_t)) in R^64                         # 25 Hz
Y_c = [u_(4c), u_(4c+1), u_(4c+2), u_(4c+3)]     # 4 x 64, 6.25 Hz
```

每个 LLM video step 对应 160 ms、4 个 motion frame。Motion DiT 在一个 chunk 内联合预测 `4 x 64`，
使用 4 个 temporal token 和双向 intra-chunk attention，而不是把 256 维向量当成互不相关的平面回归。
这样既保留 25 fps 嘴型，又把 LLM 序列长度降到 6.25 Hz。

由于 VibeVoice audio token 是 7.5 Hz，继续使用确定性的 **6 个 audio token : 5 个 video chunk**
时间调度；所有对齐按真实 timestamp 计算，不用简单按数组下标强行一一对应。若目标是在线流式推理，
音频和视频 token 应按该 6:5 节奏交错进入因果 LLM，并明确可接受的 audio lookahead；若是离线生成，
可以让视频段看到完整 audio prefix。

### 8.2 LLM 和 Motion DiT 的职责

- LLM 负责长历史、语义/韵律、说话风格、头动趋势和跨 chunk 连贯性。
- 小型 Motion DiT 负责在 LLM hidden 条件下恢复一个 `4 x 64` 连续 motion chunk 的多模态分布。
- reference image 不进入 motion DiT；只把 reference 的初始 motion `u_ref` 作为序列初态。身份外观继续留在
  renderer 一侧，避免 motion generator 被 reference 表情和身份污染。
- teacher forcing 时，用一个轻量 Motion Connector 把 GT `4 x 64` chunk 映射成下一步 LLM 输入；
  推理时直接把刚生成的 motion chunk 经过同一个 connector 回填，不再需要 WanLocEnc。
- DiT 使用标准化的 deterministic motion target，可以继续采用当前 VibeVoice 的 cosine schedule +
  v-prediction；先复用已有实现，暂时没有必要同时更换为另一套 flow matching 目标。

### 8.3 防止第二阶段重新制造抖动

- DiT 联合生成 4 帧，显式建模 chunk 内速度和加速度。
- 训练时对历史 GT chunk 加与推理误差匹配的噪声，减少纯 teacher forcing 的 exposure bias。
- 稳定后再引入类似 diffusion forcing 的异步历史噪声；不要一开始同时改 tokenizer 和生成器。
- 在低噪声 timestep 对 `x0` 估计加小权重 boundary velocity/acceleration loss，重点约束相邻 chunk 边界。
- CFG 从较小值开始；过强 CFG 往往会放大嘴型和头动，也会放大帧间 jitter。

## 9. 必须设置的阶段门槛

在投入完整 LLM+DiT 训练前，新 tokenizer 至少要通过以下测试：

1. **Clean upper bound**：固定 reference + GT motion 的 FID/FVD、LPIPS、Sync 和 identity；与当前
   `FID=7.545, FVD=50.988` 使用完全相同协议比较。
2. **Noise curve**：不仅看 `sigma=0.1` 单点，还看整个曲线以及 iid/bias/drift/outlier 四种噪声。
3. **Cross-identity**：ArcFace identity 应跟 reference，pose/expression/mouth 应跟 driving。
4. **Leakage probe**：从 motion latent 预测 identity、颜色或数据源的准确率应明显低于旧 40D latent。
5. **Temporal fidelity**：同时报告真实运动幅度和 jitter/acceleration，防止靠静止或过平滑骗过 FVD。
6. **Long-video drift**：至少测试数分钟连续渲染；chunk 边界不能跳，身份不能逐渐变化。
7. **Runtime**：reference encoder 只运行一次；motion adapter + renderer 在目标设备上应稳定超过 25 fps。

建议把“可进入第二阶段”的硬门槛定为：clean 指标至少不劣于旧 LIA-X，cross-identity 明显改善，且
`sigma=0.10` 的 FVD 相对增幅显著小于当前约 `2.06x`。具体阈值应在扩大测试集并计算方差后再锁定，
不要用当前 60 条数据过早定死一个绝对数字。

第二阶段再分两层验收：

- **Oracle renderer**：GT motion -> renderer，代表最终画质上限。
- **Predicted motion**：audio -> LLM+DiT -> renderer；两者差值才是 motion generator 的责任。

否则只看最终 FVD，无法判断应该继续改 tokenizer、DiT 还是 renderer。

## 10. 最小但信息量最大的消融顺序

| 实验 | 改动 | 回答的问题 |
|---|---|---|
| A | 原 40D LIA-X | 固定基线 |
| B | 严格双路 + 64D，仍逐帧训练 | 容量和结构解耦是否改善 clean upper bound |
| C | B + 连续片段 + temporal relation | 时序监督本身贡献多少 |
| D | C + causal latent adapter | renderer 状态是否继续降低 jitter |
| E | D + cross-ID cycle | motion 是否真正去身份、跨人泛化是否改善 |
| F | E + structured-noise finetune | 对 DiT 误差的鲁棒性是否改善 |
| G | F + image/video GAN | 高频质量和 FVD 的收益是否值得训练复杂度 |

不要同时比较 32/64/96D、KL、GAN、temporal block 和多种噪声；那样即使结果变好，也无法知道是哪一项
起作用。先固定 64D 完成 A--F。只有 B 的 clean upper bound 明显受瓶颈限制时才增加到 96D；如果
64D 已够，维度越小越利于后续长序列 motion generation。

## 11. 最终推荐的一句话版本

**训练一个 64D、逐帧确定性、严格 appearance/motion 双路的 motion autoencoder；用连续片段、
cross-identity cycle 和只作用于 latent 历史的零门控 causal adapter 学会时序与去身份；clean 收敛后固定
corpus normalization，再用真实 DiT 风格的结构化 latent 噪声做 renderer 鲁棒微调。随后冻结整套
tokenizer，将 25 fps motion 每 4 帧组成一个 6.25 Hz chunk，让 LLM 建模长历史，让小型 Motion DiT
联合生成 `4 x 64`，最后始终从同一个 reference feature cache 渲染，不递归使用上一帧图像。**

## 12. 相关设计依据

- [LIA-X](https://arxiv.org/abs/2508.09959)：证明了 reference warp-render 和紧凑隐式 motion code
  的有效性；它的正交/稀疏字典主要用于可解释控制，本方案不需要保留。
- [VASA-1](https://arxiv.org/abs/2404.10667)：强调 identity-agnostic facial dynamics latent、
  cross-identity identity loss，以及先学习 face latent、再训练 motion generator 的两阶段路线。
- [X-Actor](https://arxiv.org/abs/2508.02944)：进一步验证了紧凑 identity-agnostic motion latent、
  motion/video 解耦和 noisy-history/diffusion-forcing 对长时生成的价值。
- [VibeVoice tokenizer implementation](https://github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/modular_vibevoice_tokenizer.py)：
  其 acoustic tokenizer 使用 encoder mean 配合固定标准差；本任务为了逐帧确定监督，进一步选择
  `std=0` 的 mean-only motion code，而不是照搬音频采样策略。


---

# 已实现的训练代码

## 代码入口

- `twinlakes/motion_tokenizer/model.py`：64D frame-wise Motion Encoder、Reference Encoder、
  zero-gated Causal Motion Adapter、多尺度 source-anchored flow renderer、corpus normalizer 和
  structured motion noise。
- `twinlakes/motion_tokenizer/data.py`：直接读取 `data/talker_vivid/train.json`，按 byte offset
  随机访问 609 MB JSONL，不将整个 manifest 读入内存；从视频读取连续片段和固定 reference。
- `twinlakes/motion_tokenizer/losses.py`：Charbonnier、Laplacian pyramid、局部 gradient、嘴眼区域、
  temporal velocity/acceleration、flow TV、cross-motion cycle、image/video GAN 等损失。
- `twinlakes/bin/train_motion_tokenizer.py`：单卡/DDP、bf16、梯度累积、分阶段开关、验证、预览、
  checkpoint 和断点续训。
- `twinlakes/bin/extract_motion_latents.py`：冻结模型后批量导出 `[T,64]@25Hz` 的 normalized motion。
- `twinlakes/bin/reconstruct_motion_tokenizer.py`：生成 GT-motion renderer upper bound，支持
  iid/bias/drift/mixed normalized-noise 压力测试。

## 第一次训练：256 分辨率

默认配置直接使用指定的 1305371 条 manifest：

```bash
source /data/joe/anaconda3/etc/profile.d/conda.sh
conda activate vibe

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
CONFIG=conf/motion_tokenizer.yaml \
OUTPUT_DIR=exp/motion_tokenizer_256 \
bash run_motion_tokenizer.sh
```

第一次启动会生成：

```text
data/talker_vivid/train.json.idx.npy
data/talker_vivid/dev.json.idx.npy
```

它们只保存 JSONL 每行的 byte offset，不复制视频或 manifest，之后启动会直接 mmap，已经加入
`.gitignore`。

默认 phase boundary：

```text
0       -- 80k    clean continuous reconstruction
80k     --        cross-identity motion cycle
120k    --        image GAN
180k    --        short-clip video GAN
200k    --        freeze corpus mean/std, motion encoder LR -> 0, ramp causal adapter
240k    --        structured-noise robust finetune
500k              end of 256 phase
```

这些 step 是第一版起点，不是不可修改的常数。是否进入下一 phase 应结合 validation preview、clean
upper bound 和 identity leakage，而不能只看 total loss。

单卡调试可直接运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m twinlakes.bin.train_motion_tokenizer \
  --config conf/motion_tokenizer.yaml \
  --output_dir exp/motion_debug \
  --max_steps 100 --num_workers 0
```

默认每卡 batch size 是 1。cross-ID reference 会从下一个 DDP rank 取得，不会额外打开第二个视频；
如果只用单卡训练 cross-ID，需要把 batch size 调到至少 2。

## 第二次训练：512 分辨率

模型是 fully convolutional；256 phase 完成后保留 optimizer moment、normalizer 和 global step，使用较小
学习率继续训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
CONFIG=conf/motion_tokenizer_512.yaml \
OUTPUT_DIR=exp/motion_tokenizer_512 \
bash run_motion_tokenizer.sh \
  --resume exp/motion_tokenizer_256/step_000050000.pt
```

512 配置将 clip 从 16 帧降到 8 帧、render micro-chunk 从 4 降到 2，并冻结 motion encoder；
它主要提升 renderer 高频细节，不再改变给 LLM 使用的 motion space。

## VGG/ArcFace 权重（已经接入）

使用下面的命令会下载 PyTorch 官方 VGG16，以及 InsightFace 官方 `buffalo_l` 中的
`w600k_r50.onnx`。准备脚本会校验 SHA256，并把 ArcFace R50 一次性转成可微的 TorchScript；
训练时不需要 ONNXRuntime，也不会再联网。转换依赖只装进 `checkpoints/motion_tokenizer/python_deps`，
不会改动 `vibe` conda 环境。

```bash
source /data/joe/anaconda3/etc/profile.d/conda.sh
conda activate vibe
python tools/prepare_motion_tokenizer_weights.py --install-converter-deps
```

两个训练配置已经设置为：

```yaml
loss:
  perceptual: 0.1
  max_perceptual_frames: 4  # 512 阶段为 2，控制显存
  vgg_weights_path: checkpoints/motion_tokenizer/vgg16-397923af.pth
  identity: 0.1
  cross_identity: 0.1
  identity_model_path: checkpoints/motion_tokenizer/arcface_w600k_r50.ts
```

ArcFace 接收 `[-1,1]` RGB、内部 resize 到 `112x112` 并输出 512D embedding。ArcFace 参数被冻结，
但生成帧到 identity loss 的梯度保持连通。这里不使用 ONNXRuntime，是因为它不能把 loss 梯度反传回
renderer。`buffalo_l` 预训练模型受 InsightFace 条款约束，只可用于非商业研究；如果项目将商业化，
需要换成拥有相应授权的 identity backbone。

## 导出 motion latent

normalizer 在 checkpoint 中必须已经 finalized，否则导出工具会直接拒绝，避免把不稳定 raw latent
混入第二阶段训练数据。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
  -m twinlakes.bin.extract_motion_latents \
  --checkpoint exp/motion_tokenizer_512/step_000200000.pt \
  --manifest data/talker_vivid/train.json \
  --output_root /nfs-speech-cfs/wangzhou/data/talkhead/motion64
```

每个输出文件包含：

```text
motion: [T,64] float16, normalized, 25 Hz
fps: 25
motion_dim: 64
normalized: true
source_checkpoint: ...
```

每个 rank 写独立 manifest，全部结束后 rank 0 自动合并为
`/nfs-speech-cfs/wangzhou/data/talkhead/motion64/manifest.jsonl`。

## GT-motion upper bound 与噪声曲线

```bash
CUDA_VISIBLE_DEVICES=0 python -m twinlakes.bin.reconstruct_motion_tokenizer \
  --checkpoint exp/motion_tokenizer_512/step_000200000.pt \
  --manifest data/talker_vivid/dev.json \
  --output_dir outputs/motion64_gt \
  --limit 100 --resolution 512 --noise_sigma 0

CUDA_VISIBLE_DEVICES=0 python -m twinlakes.bin.reconstruct_motion_tokenizer \
  --checkpoint exp/motion_tokenizer_512/step_000200000.pt \
  --manifest data/talker_vivid/dev.json \
  --output_dir outputs/motion64_noise010_iid \
  --limit 100 --resolution 512 --noise_sigma 0.10 --noise_mode iid
```

重建工具先对完整 latent 序列运行 causal adapter，再分块渲染并立即写视频，因此不会把几百帧 512p
视频同时留在 CPU/GPU 内存里，也不会在 render chunk 边界重置 causal state。

---

## 方案更新：原生 512 Large 端到端 tokenizer

此前“256 收敛后冻结 encoder，再做 512 renderer finetune”的方案已被实验结果否定：40k--45k
虽然重建误差略降，但 motion encoder/normalizer 完全没有更新，44k 之后 Sync 也不再提升。新的主线改为
直接从零训练原生 512、约 200M 参数的 clean warp-render autoencoder。

配置：`conf/motion_tokenizer_512_large.yaml`，新的 delta-token 实验输出到
`exp/motion_tokenizer_512_large_delta`，避免与此前 absolute-token 试跑的 TensorBoard 混合。

- generator：198.86M，其中 motion encoder 49.90M、reference encoder 66.51M、renderer 82.41M；
- motion latent 仍为 64D，容量增加集中在空间 encoder/renderer；
- motion/reference/renderer 分别增加残差深度，并加入真正的 full-resolution render stage；
- image discriminator 为 512/256/128 三尺度、5 层 PatchGAN，共 20.87M；
- 0--90k 全部端到端训练，90k 固定 late-window normalizer 和 motion encoder，最后 10k 校准 renderer；
- 主目标只保留 reconstruction、VGG、image GAN、feature matching，以及很小的 flow/latent 正则；
- causal、structured noise、cross-ID、video GAN、landmark/velocity loss 在 clean 主训练中全部关闭。

### 512 Large 的 motion token 定义

绝对 encoder 输出 `E(x_t)` 不再直接作为 LLM/DiT 目标。训练 renderer 时使用

```text
d_t = E(x_t) - E(x_ref)
```

作为唯一进入 renderer 的 motion；normalizer 也只统计 `d_t`。批量导出完整视频时固定第一帧为参考，输出

```text
token_t = (E(x_t) - E(x_0)) / std_delta
```

其中采用只缩放、不减均值的 delta normalization，保证 `token_0` 严格为全零。导出的 `.pt` 额外写入
`representation=first_frame_relative_delta`，防止后续数据管线把它和旧的 absolute motion 混用。
推理 renderer 直接将 token 乘回 `std_delta` 后使用，不再先恢复 absolute motion 再减 reference code。
训练 renderer 时仍保留随机远端 reference，但 70k--90k 的 normalizer 只统计 reference 为视频首帧的样本，
保证训练统计的 delta 原点与正式导出一致。

为避免后续混淆，固定 LIA-X 术语如下。encoder 输出的 `A_{r->t}` 是从共享隐式 reference `r` 到帧 `t`
的 motion-dictionary 系数；它是 canonical coefficient，并不是已经减去 source/首帧的 delta。做 cross-
reenactment 时实际施加的是 `Delta A_t = A_{r->t} - A_{r->1}`，再加到 source baseline `A_{r->s}` 上，
即最终系数为 `A_{r->s} + Delta A_t`。因此，“LIA-X 的动画传递使用相对运动差”是对的，但不能把原始
`A_{r->t}` 和 `Delta A_t` 当成同一个张量。当前旧 s5 数据管线保存和预测的是逐帧 raw `A_{r->t}`，没有
在数据侧执行 `A_{r->t} - A_{r->1}`；self-reenactment 时 source 与 driving 首帧一致，公式中的差值会抵消，
所以可以直接使用 `A_{r->t}`。

cross-ID 暂时不进入本轮 clean reconstruction baseline，但这**不是**因为 delta 可以替代或降低 cross-ID
解耦要求。`E(x_t)-E(x_ref)` 只有在 encoder 的 identity 成分近似为可加、且同一身份不同帧共享同一
offset 时才会抵消 identity；对一般非线性、identity-dependent motion coordinates 并无保证。LIA-X 的关键
是先学习共享 canonical `A` 坐标和 motion dictionary，再在动画应用层计算 `Delta A`；其解耦不能简单归因于
first-frame subtraction。正式接受 motion token 前，必须执行 identity probe 和 cross-reference oracle；若
identity leakage 明显，就需要加入 cross-ID/cycle/invariance 约束，不能继续把 delta 当作替代方案。

reference/target 的几何和 photometric augmentation 必须整段共享。若两边独立做 brightness、contrast、
saturation jitter，而训练目标又要求逐像素匹配 target，网络只能把这组人为制造的颜色差塞进 motion delta，
会直接破坏 token 语义。

另外，image GAN 的 fake/real 抽帧必须共享相同索引。尤其 feature matching 是成对特征距离，如果分别随机
抽帧，会把不同人物或不同时刻当作配对监督；512 Large 代码已改为 paired frame sampling。
训练日志每 20 step 对各 rank 当前 scalar 做一次全局均值，避免只看 rank 0 的最后一个 batch；同时记录
`motion_delta_std_{min,mean,max}`、active dimension 比例、flow magnitude 和 mask mean，用于区分 latent
塌缩、只复制 reference 和正常的 warp-render 学习。

两机 16 卡沿用统一入口：

```bash
# master 节点
bash run.sh 2 0

# worker 节点
bash run.sh 2 1
```

单机 8 卡：

```bash
bash run_motion_tokenizer.sh
```

默认每卡 `batch_size=8`、`clip_length=4`、不做梯度累积；16 卡时每次 optimizer update 等效处理
128 个 clip / 512 个 target frame。activation checkpointing 已启用，16 卡 clean 训练实测单卡峰值约
36.96 GB、约 0.339 optimizer step/s。10k 开启 GAN 后需要重新观察峰值显存和吞吐。

### TODO：将 motion delta 做成 Gaussian-friendly latent

当前主训练先保持 deterministic、reference-relative delta，优先验证 512 Large 的重建上限、嘴型/眨眼
信息量和 identity leakage。后续在 clean autoencoder 收敛、normalizer 固定以后，引入类似 VibeVoice
acoustic tokenizer 的 stochastic latent 训练，使 renderer 能在带预测误差的连续 latent 上保持细节和稳定性。

这里必须明确：做 Gaussian-friendly latent 的主要目的**不是解决 delta 积分误差**。本方案使用的是相对固定
第一帧的 `d_t = E(x_t) - E(x_0)`，不是相邻帧 velocity `E(x_t) - E(x_{t-1})`；每一帧都独立相对
同一个 reference 定义，推理时不需要逐帧积分，因此不会产生传统 delta random walk。Gaussian-friendly
主要解决的是 absolute/relative latent 的各维尺度不均、协方差病态、分布空洞、预测值落入训练分布之外，
以及 renderer 对轻微 latent 预测误差过度敏感的问题。高斯化本身不自动产生容错性，必须配合与真实
LLM/DiT residual 的幅度和时间相关性匹配的 noisy-latent reconstruction，显式训练 decoder 的局部平滑性。

后续 LLM/DiT 是否预测 reference-relative delta 不能仅凭结构直觉预先定论。delta 的确定优势只有严格零起点、
不重复预测 reference baseline，以及相对固定第一帧时没有相邻帧 delta 的积分漂移；它**不天然实现身份解耦**，
也不天然比 absolute code 更高斯或更容易由音频预测。相反，delta 还会让同一嘴型/姿态的数值依赖 reference
初始状态。必须保留同架构下的 absolute-vs-delta ablation，比较 clean/noise oracle、identity probe、
cross-reference 和真实 LLM/DiT 预测误差后再选表示。若两者各有优势，可采用“reference absolute anchor +
relative motion residual”的混合表示，而不是把 delta 当作免除 cross-ID 训练的理由。

计划如下：

1. 保留 `d_t = E(x_t) - E(x_0)`；首帧 `d_0=0` 是特殊原点，禁止加噪，也不参加 Gaussian prior loss。
2. 在 held-out 数据上统计每维 mean/std、covariance eigenvalue、effective rank、skewness、kurtosis 和尾部分位数；
   逐维 scale normalization 只保证二阶尺度一致，不能被误称为严格正态分布。
3. normalizer 固定后，在 normalized delta 上加入固定小方差 Gaussian noise，联合优化 clean/noisy reconstruction；
   从小 `sigma` 开始，根据嘴型、眨眼 oracle 和真实 DiT residual 决定最终噪声强度，不能直接照搬
   VibeVoice 的 `fix_std=0.5`。
4. 若分布明显重尾或多维相关影响 DiT，再比较弱 KL、MMD/Sliced-Wasserstein 和可逆 whitening/flow；
   优先约束 aggregated posterior，避免强 per-sample KL 压低 motion mutual information。
5. LLM/DiT 阶段同时比较 GT/predicted token 的 per-dim variance ratio、低频 drift、velocity、acceleration、
   高频能量和跨 token boundary continuity。Gaussian-friendly 只解决优化与抗噪，不能替代时序分布匹配。

验收标准不是“直方图看起来像高斯”，而是：token 数据集尺度均衡、无塌缩、renderer 在合理 latent noise
下不丢嘴唇/眼睛细节，并且 LLM/DiT 预测分布与 GT 的幅度和时间频谱一致。

---

## 当前重训版本：256 LIA-X-style feature-warp autoencoder

512 Large 的独立 motion/reference encoder 和最终 warped-RGB blend 容易形成“motion 通路弱、reference
复制通路强”的捷径。当前重训版改回更接近 LIA-X 的主干，配置为
`conf/motion_tokenizer_256_liax.yaml`，输出目录为 `exp/motion_tokenizer_256_liax_fused`：

- source/target 使用同一个多尺度 encoder；
- encoder 先产生 512D shared style，再预测 64D canonical motion coefficient `A_t`；
- 64D coefficient 经正交 motion dictionary 映射回 renderer style；
- renderer 内部使用 absolute target coefficient `A_t`，而导给 LLM/DiT 的 token 仍是
  `A_t - A_0` 的 scale-only normalized delta；
- 每一层先用 FiLM residual blocks 充分注入 target motion，再从同一个 motion-refined feature 预测
  flow 和 feature mask；删除 warp 后的 concat/pre convolution，避免 flow/mask 在运动调制不足时提前决定；
- 执行 `warped_feature = mask * grid_sample(source_feature, flow)`、
  `generated_feature = (1 - mask) * h` 和
  `fused_feature = warped_feature + generated_feature`；下一层 renderer 和当前尺度 ToRGB 都接收
  `fused_feature`，使遮挡、口腔内部和眼睑等不能由 reference warp 得到的内容有直接生成通路；
- feature mask 保留，因为它属于 LIA-X 的 ToFlow 特征融合；删除的是最终输出端
  `mask * warped_reference_rgb + (1 - mask) * render_rgb` 这条 raw-RGB shortcut；
- 最终图像只由各尺度 fused feature 的 residual blocks 和 ToRGB skip 累加生成；
- 删除各尺度 warp 后 concat/pre convolution 后，generator 为 200.880M 参数，正好处于目标的约 200M 规模；
- loss 保持简单：L1 reconstruction + VGG perceptual + image GAN + 很小的 absolute coefficient L1 sparsity；
  causal、noise、cross-ID、landmark、velocity 和 video GAN 暂不加入首轮 baseline。

训练按每卡 batch=8、16 卡 global batch=128 设置为 50k step：0--5k 只训练 L1/VGG/sparsity，5k
启动 image GAN，之后一直端到端训练到 50k。本轮不收集或固定 motion normalizer，也不冻结 shared
encoder/motion head；它只验证重建上限，不产出供 LLM/DiT 正式训练的最终 token 分布。后续用 KL 或其他
Gaussian-friendly 约束重新定义 latent 分布。未配置的 causal、noise、cross-ID 和 video GAN stage 均关闭。

本机真实数据单步 smoke test 已通过：batch=8、clip=4、render chunk=1 时峰值显存 22.64GB；双卡 DDP
也已通过。forward/backward 中除显式关闭的 causal adapter 外，shared encoder、motion head、orthogonal
dictionary、FiLM、flow、feature mask、generated branch 和 ToRGB 都有非零梯度。正式训练脚本默认已经
切到此配置：

```bash
# 两机 16 卡
bash run.sh 2 0
bash run.sh 2 1

# 单机 8 卡
bash run_motion_tokenizer.sh
```
