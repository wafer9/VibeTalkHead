"""Realtime-oriented WAN latent connector and local video DiT.

Both modules operate on one WAN temporal slice at a time.  Spatial 4x4
patching keeps the local sequence at 256 tokens for a 64x64 latent, which is
small enough to run the denoiser several times per generated frame.
"""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings for t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float().reshape(-1, 1) * 1000.0 * freqs.reshape(1, -1)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (y.transpose(1, 2) for y in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class LocalEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * ffn_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class WanLocEnc(nn.Module):
    """Compress every [16,64,64] WAN frame into one LLM embedding.

    A learned CLS token participates in a non-causal local transformer, as in
    VoxCPM LocEnc.  Separable row/column embeddings preserve 2-D position.
    """

    def __init__(
        self,
        output_dim: int,
        in_channels: int = 16,
        latent_size: int = 64,
        patch_size: int = 4,
        hidden_dim: int = 384,
        num_layers: int = 2,
        num_heads: int = 6,
        ffn_ratio: float = 3.0,
    ):
        super().__init__()
        assert latent_size % patch_size == 0
        grid = latent_size // patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.row_embed = nn.Parameter(torch.randn(1, grid, 1, hidden_dim) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, grid, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            LocalEncoderBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B,T,16,64,64] or [N,16,64,64]."""
        restore = z.ndim == 5
        if restore:
            b, t, c, h, w = z.shape
            z = z.reshape(b * t, c, h, w)
        x = self.patch_embed(z)
        gh, gw = x.shape[-2:]
        pos = (self.row_embed[:, :gh] + self.col_embed[:, :, :gw]).reshape(1, gh * gw, -1)
        x = x.flatten(2).transpose(1, 2) + pos.to(dtype=x.dtype)
        cls = self.cls_token.expand(x.shape[0], -1, -1).to(dtype=x.dtype)
        x = torch.cat([cls, x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(self.norm(x[:, 0]))
        return x.reshape(b, t, -1) if restore else x


class PrefixVideoDiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, int(dim * ffn_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No causal mask: time/LLM/reference prefix tokens and noisy spatial
        # tokens communicate bidirectionally, as in VoxCPM LocalDiT.
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class RealtimeVideoDiT(nn.Module):
    """Bidirectional prefix-token DiT for one WAN temporal latent slice.

    Token layout is ``[time, LLM, reference patches, noisy patches]``.  Unlike
    AdaLN conditioning, the audio-aware LLM state is an ordinary Transformer
    token visible to every noisy patch from the first training step.
    """

    def __init__(
        self,
        llm_dim: int,
        latent_channels: int = 16,
        latent_size: int = 64,
        patch_size: int = 4,
        reference_patch_size: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_ratio: float = 3.0,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        assert latent_size % patch_size == 0
        assert latent_size % reference_patch_size == 0
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.reference_patch_size = reference_patch_size
        self.cond_dropout = cond_dropout
        grid = latent_size // patch_size
        reference_grid = latent_size // reference_patch_size

        self.noisy_patch_embed = nn.Conv2d(
            latent_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.reference_patch_embed = nn.Conv2d(
            latent_channels,
            hidden_dim,
            kernel_size=reference_patch_size,
            stride=reference_patch_size,
        )
        self.noisy_row_embed = nn.Parameter(torch.randn(1, grid, 1, hidden_dim) * 0.02)
        self.noisy_col_embed = nn.Parameter(torch.randn(1, 1, grid, hidden_dim) * 0.02)
        self.reference_row_embed = nn.Parameter(
            torch.randn(1, reference_grid, 1, hidden_dim) * 0.02
        )
        self.reference_col_embed = nn.Parameter(
            torch.randn(1, 1, reference_grid, hidden_dim) * 0.02
        )
        # time / LLM / reference / noisy segment identifiers.
        self.token_type_embed = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Qwen's final hidden state is already normalized.  A second
        # LayerNorm would remove cond/uncond differences represented by mean
        # or magnitude, so use a learnable MLP directly.
        self.llm_proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.llm_gain = nn.Parameter(torch.tensor(1.0))
        self.blocks = nn.ModuleList(
            [PrefixVideoDiTBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_channels * patch_size * patch_size)
        self._init_output()

    def _init_output(self):
        # Close-to-zero initial v-prediction without blocking gradients to the
        # prefix decoder (strict zero would block them on the first step).
        nn.init.normal_(self.out_proj.weight, std=1e-3)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        hidden: torch.Tensor,
        reference: torch.Tensor,
        drop_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Latent arguments are [N,16,64,64]; hidden is [N,H]."""
        if drop_condition is None and self.training and self.cond_dropout > 0:
            drop_condition = torch.rand(hidden.shape[0], device=hidden.device) < self.cond_dropout
        if drop_condition is not None:
            hidden = hidden.masked_fill(drop_condition[:, None], 0)

        noisy_tokens = self.noisy_patch_embed(noisy)
        gh, gw = noisy_tokens.shape[-2:]
        noisy_pos = (
            self.noisy_row_embed[:, :gh] + self.noisy_col_embed[:, :, :gw]
        ).reshape(1, gh * gw, -1)
        noisy_tokens = noisy_tokens.flatten(2).transpose(1, 2)
        noisy_tokens = noisy_tokens + noisy_pos.to(dtype=noisy_tokens.dtype)
        noisy_tokens = noisy_tokens + self.token_type_embed[3].to(dtype=noisy_tokens.dtype)

        reference_tokens = self.reference_patch_embed(reference)
        rh, rw = reference_tokens.shape[-2:]
        reference_pos = (
            self.reference_row_embed[:, :rh] + self.reference_col_embed[:, :, :rw]
        ).reshape(1, rh * rw, -1)
        reference_tokens = reference_tokens.flatten(2).transpose(1, 2)
        reference_tokens = reference_tokens + reference_pos.to(dtype=reference_tokens.dtype)
        reference_tokens = reference_tokens + self.token_type_embed[2].to(
            dtype=reference_tokens.dtype
        )

        time_token = self.time_mlp(
            timestep_embedding(timestep, noisy_tokens.shape[-1]).to(dtype=noisy_tokens.dtype)
        )
        time_token = time_token + self.token_type_embed[0].to(dtype=time_token.dtype)
        llm_token = self.llm_gain.to(dtype=noisy_tokens.dtype) * self.llm_proj(hidden)
        llm_token = llm_token + self.token_type_embed[1].to(dtype=llm_token.dtype)

        x = torch.cat(
            [time_token.unsqueeze(1), llm_token.unsqueeze(1), reference_tokens, noisy_tokens],
            dim=1,
        )
        for block in self.blocks:
            x = block(x)
        x = self.out_proj(self.final_norm(x[:, -gh * gw:]))
        p, c = self.patch_size, self.latent_channels
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, gh * p, gw * p)
