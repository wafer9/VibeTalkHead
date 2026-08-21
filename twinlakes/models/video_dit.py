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


class VideoDiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = FeedForward(dim, int(dim * ffn_ratio))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = \
            self.modulation(cond).unsqueeze(1).chunk(6, dim=-1)
        y = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn(y)
        y = self.norm2(x) * (1 + scale_f) + shift_f
        return x + gate_f * self.ffn(y)


class RealtimeVideoDiT(nn.Module):
    """Predict a WAN-frame flow velocity from noise, reference, history and LLM state."""

    def __init__(
        self,
        llm_dim: int,
        latent_channels: int = 16,
        latent_size: int = 64,
        patch_size: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ffn_ratio: float = 3.0,
        cond_dropout: float = 0.1,
    ):
        super().__init__()
        assert latent_size % patch_size == 0
        self.latent_channels = latent_channels
        self.patch_size = patch_size
        self.cond_dropout = cond_dropout
        grid = latent_size // patch_size

        # noisy target + fixed identity reference + previous generated frame
        self.patch_embed = nn.Conv2d(
            latent_channels * 3, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        self.row_embed = nn.Parameter(torch.randn(1, grid, 1, hidden_dim) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, grid, hidden_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.llm_proj = nn.Sequential(
            nn.LayerNorm(llm_dim), nn.Linear(llm_dim, hidden_dim)
        )
        self.blocks = nn.ModuleList([
            VideoDiTBlock(hidden_dim, num_heads, ffn_ratio) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, latent_channels * patch_size * patch_size)
        self._init_adaln_zero()

    def _init_adaln_zero(self):
        for block in self.blocks:
            nn.init.zeros_(block.modulation[-1].weight)
            nn.init.zeros_(block.modulation[-1].bias)
        nn.init.zeros_(self.final_mod[-1].weight)
        nn.init.zeros_(self.final_mod[-1].bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        hidden: torch.Tensor,
        reference: torch.Tensor,
        previous: torch.Tensor,
        drop_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """All latent arguments are [N,16,64,64]; hidden is [N,H]."""
        if drop_condition is None and self.training and self.cond_dropout > 0:
            drop_condition = torch.rand(hidden.shape[0], device=hidden.device) < self.cond_dropout
        if drop_condition is not None:
            hidden = hidden.masked_fill(drop_condition[:, None], 0)

        x = self.patch_embed(torch.cat([noisy, reference, previous], dim=1))
        gh, gw = x.shape[-2:]
        pos = (self.row_embed[:, :gh] + self.col_embed[:, :, :gw]).reshape(1, gh * gw, -1)
        x = x.flatten(2).transpose(1, 2) + pos.to(dtype=x.dtype)
        cond = self.time_mlp(timestep_embedding(timestep, x.shape[-1]).to(dtype=x.dtype))
        cond = cond + self.llm_proj(hidden)
        for block in self.blocks:
            x = block(x, cond)
        shift, scale = self.final_mod(cond).unsqueeze(1).chunk(2, dim=-1)
        x = self.out_proj(self.final_norm(x) * (1 + scale) + shift)
        p, c = self.patch_size, self.latent_channels
        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], c, gh * p, gw * p)

