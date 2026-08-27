"""Thin sequence wrapper around the official LIA-X encoder and decoder."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .liax import Decoder, Encoder


class MotionTokenizer(nn.Module):
    """LIA-X reconstruction model with a video-friendly batch/time interface.

    The learnable network is the vendored LIA-X Encoder + Decoder without a
    custom renderer, normalizer, causal adapter, noise branch, or cross cycle.
    Training follows the native LIA-X pair reconstruction path: the decoder is
    conditioned on the target's absolute coefficient.  The public motion token
    remains reference-relative, delta_t = alpha_t - alpha_reference; rendering
    a predicted delta converts it back to alpha_t before decoding.
    """

    def __init__(
        self,
        style_dim: int = 512,
        motion_dim: int = 64,
        scale: int = 1,
        decoder_blocks_per_stage: int = 4,
        channels: Optional[Sequence[int]] = None,
        flow_identity_init: bool = False,
        flow_init_scale: float = 1.0,
    ):
        super().__init__()
        self.style_dim = int(style_dim)
        self.motion_dim = int(motion_dim)
        self.scale = int(scale)
        self.decoder_blocks_per_stage = int(decoder_blocks_per_stage)
        self.flow_identity_init = bool(flow_identity_init)
        self.flow_init_scale = float(flow_init_scale)
        self.channels = (
            None if channels is None else [int(x) for x in channels]
        )
        self.encoder = Encoder(
            self.style_dim,
            self.motion_dim,
            self.scale,
            feature_channels=self.channels,
        )
        self.decoder = Decoder(
            self.style_dim,
            self.motion_dim,
            self.scale,
            blocks_per_stage=self.decoder_blocks_per_stage,
            stage_channels=self.channels,
            flow_identity_init=self.flow_identity_init,
            flow_init_scale=self.flow_init_scale,
        )

    def encode_motion(self, image: torch.Tensor) -> torch.Tensor:
        """Return the absolute LIA-X alpha coefficient for each frame."""
        return self.encoder.enc_motion(image)

    def encode_reference(
        self, reference: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.encoder.enc_2r(reference)

    @staticmethod
    def _repeat_batch(x: torch.Tensor, count: int) -> torch.Tensor:
        return x[:, None].expand(-1, count, *x.shape[1:]).reshape(
            -1, *x.shape[1:]
        )

    def _decode_sequence(
        self,
        reference_style: torch.Tensor,
        reference_features: Sequence[torch.Tensor],
        target_motion: torch.Tensor,
        render_chunk: int,
    ) -> torch.Tensor:
        batch, time, _ = target_motion.shape
        render_chunk = time if render_chunk <= 0 else int(render_chunk)
        images = []
        flow_stats = []
        for start in range(0, time, render_chunk):
            stop = min(start + render_chunk, time)
            count = stop - start
            style = self._repeat_batch(reference_style, count)
            features = [
                self._repeat_batch(feature, count)
                for feature in reference_features
            ]
            alpha = target_motion[:, start:stop].reshape(batch * count, -1)
            image = self.decoder(style, [alpha], features)
            images.append(image.reshape(batch, count, *image.shape[1:]))
            flow_stats.append(self.decoder.last_flow_stats)
        self.last_flow_stats = torch.stack(flow_stats).mean(dim=0)
        return torch.cat(images, dim=1)

    def forward(
        self,
        reference: torch.Tensor,
        frames: torch.Tensor,
        *,
        render_chunk: int = 1,
    ) -> Dict[str, torch.Tensor]:
        batch, time = frames.shape[:2]
        reference_style, reference_features = self.encode_reference(reference)
        reference_motion = self.encoder.enc_r2t(reference_style)

        target_chunks = []
        encode_chunk = time if render_chunk <= 0 else int(render_chunk)
        for start in range(0, time, encode_chunk):
            stop = min(start + encode_chunk, time)
            target = frames[:, start:stop].reshape(
                batch * (stop - start), *frames.shape[2:]
            )
            target_chunks.append(
                self.encode_motion(target).reshape(batch, stop - start, -1)
            )
        target_motion = torch.cat(target_chunks, dim=1)
        motion_delta = target_motion - reference_motion[:, None]
        reconstruction = self._decode_sequence(
            reference_style, reference_features, target_motion, render_chunk
        )
        return {
            "reconstruction": reconstruction,
            "reference_motion": reference_motion,
            "target_motion": target_motion,
            "motion_delta": motion_delta,
            "flow_stats": self.last_flow_stats,
        }

    def render_motion(
        self,
        reference: torch.Tensor,
        motion_delta: torch.Tensor,
        *,
        render_chunk: int = 1,
    ) -> torch.Tensor:
        """Render reference-relative motion deltas through native LIA-X alpha."""
        reference_style, reference_features = self.encode_reference(reference)
        reference_motion = self.encoder.enc_r2t(reference_style)
        target_motion = motion_delta + reference_motion[:, None]
        return self._decode_sequence(
            reference_style, reference_features, target_motion, render_chunk
        )
