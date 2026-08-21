"""Causal audio/visual LM with a realtime local WAN-frame flow head."""

import logging
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
from twinlakes.models.video_dit import RealtimeVideoDiT, WanLocEnc

logger = logging.getLogger(__name__)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) >= lengths.unsqueeze(1)


class LMMConfig(PretrainedConfig):
    model_type = "lam"
    _auto_class = "AutoConfig"
    is_composition = True

    def __init__(
        self,
        audio_config: Optional[Dict] = None,
        text_config: Optional[Dict] = None,
        num_special_tokens_add: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.audio_config = audio_config or {}
        self.text_config = text_config or {}
        self.num_special_tokens_add = num_special_tokens_add


class LMModel(nn.Module):
    """Use an LLM for temporal planning and a small CFM DiT for WAN frames."""

    def __init__(
        self,
        config: VibeVoiceConfig,
        language_model: PreTrainedModel,
        acoustic_tokenizer: PreTrainedModel,
        acoustic_connector: SpeechConnector,
        video_connector: WanLocEnc,
        video_dit: RealtimeVideoDiT,
        tokenizer,
        speech_scaling_factor: torch.Tensor,
        speech_bias_factor: torch.Tensor,
        max_dit_frames_per_sample: int = 8,
    ):
        super().__init__()
        self.config = config
        self.lm = language_model
        self.acoustic_tokenizer = acoustic_tokenizer
        self.acoustic_connector = acoustic_connector
        self.video_connector = video_connector
        self.video_dit = video_dit
        self.tokenizer = tokenizer
        self.max_dit_frames_per_sample = max_dit_frames_per_sample
        self.register_buffer("speech_scaling_factor", speech_scaling_factor)
        self.register_buffer("speech_bias_factor", speech_bias_factor)

    @property
    def comp(self):
        """Backward-compatible name used by existing diagnostic scripts."""
        return self.video_connector

    @property
    def diffusion_head(self):
        return self.video_dit

    @classmethod
    def from_audio_text_pretrained(cls, configs: Optional[Dict] = None):
        configs = configs or {}
        lm_path = configs["temporal_model"]
        lm_config = AutoConfig.from_pretrained(lm_path, trust_remote_code=True)
        language_model = AutoModelForCausalLM.from_pretrained(lm_path, trust_remote_code=True)
        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(lm_path)

        vibevoice = VibeVoiceForConditionalGenerationInference.from_pretrained(
            configs["vibevoice_path"]
        )
        acoustic_tokenizer = vibevoice.model.acoustic_tokenizer
        acoustic_connector = SpeechConnector(
            acoustic_tokenizer.config.vae_dim, lm_config.hidden_size
        )

        loc = configs.get("wan_locenc", {})
        video_connector = WanLocEnc(
            output_dim=lm_config.hidden_size,
            in_channels=loc.get("latent_channels", 16),
            latent_size=loc.get("latent_size", 64),
            patch_size=loc.get("patch_size", 4),
            hidden_dim=loc.get("hidden_dim", 384),
            num_layers=loc.get("num_layers", 2),
            num_heads=loc.get("num_heads", 6),
            ffn_ratio=loc.get("ffn_ratio", 3.0),
        )

        dit = configs.get("video_dit", {})
        video_dit = RealtimeVideoDiT(
            llm_dim=lm_config.hidden_size,
            latent_channels=dit.get("latent_channels", 16),
            latent_size=dit.get("latent_size", 64),
            patch_size=dit.get("patch_size", 4),
            hidden_dim=dit.get("hidden_dim", 512),
            num_layers=dit.get("num_layers", 8),
            num_heads=dit.get("num_heads", 8),
            ffn_ratio=dit.get("ffn_ratio", 3.0),
            cond_dropout=dit.get("cond_dropout", 0.1),
        )
        model = cls(
            config=vibevoice.config,
            language_model=language_model,
            acoustic_tokenizer=acoustic_tokenizer,
            acoustic_connector=acoustic_connector,
            video_connector=video_connector,
            video_dit=video_dit,
            tokenizer=tokenizer,
            speech_scaling_factor=vibevoice.speech_scaling_factor,
            speech_bias_factor=vibevoice.speech_bias_factor,
            max_dit_frames_per_sample=configs.get("max_dit_frames_per_sample", 8),
        )
        dtype = torch.bfloat16 if configs.get("dtype") == "bf16" else torch.float32
        return model.to(dtype=dtype)

    def encode_audio(self, wavs: torch.Tensor, wavs_lengths: torch.Tensor):
        self.acoustic_tokenizer.eval()
        with torch.no_grad():
            features = self.acoustic_tokenizer.encode(wavs.transpose(1, 2)).mean
        features = (features + self.speech_bias_factor) * self.speech_scaling_factor
        features = self.acoustic_connector(features)
        lengths = torch.ceil(wavs_lengths / 3200).to(torch.int64)
        lengths = lengths.clamp_max(features.shape[1])
        return features, ~make_pad_mask(lengths, features.shape[1])

    @staticmethod
    def _sequence_masks(input_ids: torch.Tensor, audio_pos: List[torch.Tensor]):
        audio_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        video_input_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        video_loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for idx, pos in enumerate(audio_pos):
            p_s, p_e, v_s, v_e = (int(v) for v in pos.tolist())
            audio_mask[idx, p_s + 1:p_e] = True
            video_input_mask[idx, v_s + 1:v_e] = True
            # Hidden at vision_start predicts target 0; hidden at target i-1 predicts i.
            video_loss_mask[idx, v_s:v_e - 1] = True
        return audio_mask, video_input_mask, video_loss_mask

    def forward(
        self,
        keys: List[str],
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        wavs: torch.Tensor,
        wavs_lengths: torch.Tensor,
        video_latents: torch.Tensor,
        video_latent_lengths: torch.Tensor,
        audio_pos: List[torch.Tensor],
    ):
        del keys, labels
        valid_ids = (input_ids >= 0) & (input_ids < self.lm.config.vocab_size)
        safe_ids = input_ids.masked_fill(~valid_ids, self.tokenizer.eos_id)
        x = self.lm.get_input_embeddings()(safe_ids)

        audio_features, audio_valid = self.encode_audio(wavs, wavs_lengths)
        audio_mask, video_input_mask, video_loss_mask = self._sequence_masks(input_ids, audio_pos)

        # [B,Tall,C,H,W]: frame 0 is reference; frames 1: are AR targets.
        target_lengths = (video_latent_lengths - 1).clamp_min(0)
        max_targets = video_latents.shape[1] - 1
        target_valid = ~make_pad_mask(target_lengths.to(torch.int64), max_targets)
        targets_padded = video_latents[:, 1:]
        video_embeddings = self.video_connector(targets_padded[target_valid])

        if int(audio_mask.sum()) != int(audio_valid.sum()):
            raise ValueError(
                "audio token mismatch: prompt={} encoder={}".format(
                    int(audio_mask.sum()), int(audio_valid.sum())
                )
            )
        if int(video_input_mask.sum()) != int(target_valid.sum()):
            raise ValueError(
                "video token mismatch: prompt={} latent={}".format(
                    int(video_input_mask.sum()), int(target_valid.sum())
                )
            )

        # Reference occupies the first <|image_pad|>; GT targets are teacher-forced.
        x[:, 0] = self.video_connector(video_latents[:, 0])
        x[audio_mask] = audio_features[audio_valid]
        x[video_input_mask] = video_embeddings
        attention_mask = valid_ids.to(torch.long)
        outputs = self.lm.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        conditions = outputs.last_hidden_state[video_loss_mask]

        # The LLM/LocEnc sees the complete teacher-forced sequence, while the
        # expensive denoiser trains on a random subset of frames from each clip.
        # Over epochs every frame is covered without making memory scale with
        # clip duration.
        selected = target_valid.clone()
        limit = self.max_dit_frames_per_sample
        if self.training and limit > 0:
            for row in range(selected.shape[0]):
                indices = torch.where(selected[row])[0]
                if indices.numel() > limit:
                    keep = indices[torch.randperm(indices.numel(), device=indices.device)[:limit]]
                    selected[row].zero_()
                    selected[row, keep] = True
        selected_among_valid = selected[target_valid]
        conditions = conditions[selected_among_valid]

        targets = targets_padded[selected]
        previous_padded = video_latents[:, :-1]
        previous = previous_padded[selected]
        batch_index = torch.arange(video_latents.shape[0], device=video_latents.device)
        batch_index = batch_index[:, None].expand_as(selected)[selected]
        reference = video_latents[:, 0][batch_index]

        # Rectified-flow/CFM target. All valid frames are trained in one DiT batch.
        noise = torch.randn_like(targets)
        t = torch.rand(targets.shape[0], device=targets.device, dtype=torch.float32)
        noisy = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * targets
        target_velocity = targets - noise
        prediction = self.video_dit(noisy, t, conditions, reference, previous)
        flow_loss = torch.nn.functional.mse_loss(
            prediction.float(), target_velocity.float(), reduction="mean"
        )
        return {"loss": flow_loss, "flow_loss": flow_loss, "diffusion_loss": flow_loss}

    @torch.no_grad()
    def sample_video_frame(
        self,
        hidden: torch.Tensor,
        reference: torch.Tensor,
        previous: torch.Tensor,
        num_steps: int = 8,
        cfg_scale: float = 1.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Euler-sample one [B,16,64,64] frame from CFM noise."""
        b = hidden.shape[0]
        z = torch.randn(
            b, 16, 64, 64, device=hidden.device, dtype=hidden.dtype, generator=generator
        )
        dt = 1.0 / num_steps
        do_cfg = cfg_scale != 1.0
        for step in range(num_steps):
            t = torch.full((b,), (step + 0.5) * dt, device=z.device, dtype=torch.float32)
            if do_cfg:
                z_in = torch.cat([z, z], dim=0)
                h_in = torch.cat([hidden, torch.zeros_like(hidden)], dim=0)
                ref_in = torch.cat([reference, reference], dim=0)
                prev_in = torch.cat([previous, previous], dim=0)
                velocity = self.video_dit(z_in, t.repeat(2), h_in, ref_in, prev_in)
                cond, uncond = velocity.chunk(2)
                velocity = uncond + cfg_scale * (cond - uncond)
            else:
                velocity = self.video_dit(z, t, hidden, reference, previous)
            z = z + dt * velocity
        return z

    @torch.no_grad()
    def generate_video_latents(
        self,
        input_ids: torch.Tensor,
        wavs: torch.Tensor,
        wavs_lengths: torch.Tensor,
        audio_pos: List[torch.Tensor],
        reference: torch.Tensor,
        num_frames: int,
        num_steps: int = 8,
        cfg_scale: float = 1.5,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Autoregressively generate WAN latents using the LLM KV cache.

        ``input_ids`` is the prompt through the video ``vision_start`` token;
        ``reference`` is [B,16,64,64].  The returned tensor includes reference
        at temporal index zero and has shape [B,16,num_frames+1,64,64].
        """
        valid_ids = (input_ids >= 0) & (input_ids < self.lm.config.vocab_size)
        safe_ids = input_ids.masked_fill(~valid_ids, self.tokenizer.eos_id)
        x = self.lm.get_input_embeddings()(safe_ids)
        audio_features, audio_valid = self.encode_audio(wavs, wavs_lengths)
        audio_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        video_start = []
        for row, pos in enumerate(audio_pos):
            values = [int(v) for v in pos.tolist()]
            p_s, p_e, v_s = values[:3]
            audio_mask[row, p_s + 1:p_e] = True
            video_start.append(v_s)
        if int(audio_mask.sum()) != int(audio_valid.sum()):
            raise ValueError("audio prompt/encoder length mismatch during generation")
        x[:, 0] = self.video_connector(reference)
        x[audio_mask] = audio_features[audio_valid]

        out = self.lm.model(
            inputs_embeds=x,
            attention_mask=valid_ids.to(torch.long),
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        rows = torch.arange(x.shape[0], device=x.device)
        h = out.last_hidden_state[rows, torch.tensor(video_start, device=x.device)]

        previous = reference
        generated = [reference]
        for frame_index in range(num_frames):
            frame = self.sample_video_frame(
                h, reference, previous, num_steps, cfg_scale, generator
            )
            generated.append(frame)
            previous = frame
            if frame_index + 1 < num_frames:
                frame_token = self.video_connector(frame).unsqueeze(1)
                out = self.lm.model(
                    inputs_embeds=frame_token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                h = out.last_hidden_state[:, -1]
        return torch.stack(generated, dim=2)
