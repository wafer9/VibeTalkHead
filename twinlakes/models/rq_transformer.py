"""Causal audio/visual LM with a realtime WAN-frame diffusion head."""

import logging
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler
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
    """Use an LLM for temporal planning and a small diffusion DiT for WAN frames."""

    def __init__(
        self,
        config: VibeVoiceConfig,
        language_model: PreTrainedModel,
        acoustic_tokenizer: PreTrainedModel,
        acoustic_connector: SpeechConnector,
        video_connector: WanLocEnc,
        video_dit: RealtimeVideoDiT,
        noise_scheduler: DPMSolverMultistepScheduler,
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
        self.noise_scheduler = noise_scheduler
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
            reference_patch_size=dit.get("reference_patch_size", 8),
            hidden_dim=dit.get("hidden_dim", 512),
            num_layers=dit.get("num_layers", 8),
            num_heads=dit.get("num_heads", 8),
            ffn_ratio=dit.get("ffn_ratio", 3.0),
            cond_dropout=dit.get("cond_dropout", 0.1),
        )
        diffusion_config = vibevoice.config.diffusion_head_config
        noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=diffusion_config.ddpm_num_steps,
            beta_schedule=diffusion_config.ddpm_beta_schedule,
            prediction_type=diffusion_config.prediction_type,
        )
        model = cls(
            config=vibevoice.config,
            language_model=language_model,
            acoustic_tokenizer=acoustic_tokenizer,
            acoustic_connector=acoustic_connector,
            video_connector=video_connector,
            video_dit=video_dit,
            noise_scheduler=noise_scheduler,
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

        # Conditional/unconditional LLM inputs differ only at audio positions.
        # The first-frame WAN reference is a DiT spatial condition and is not
        # injected into either LLM branch.
        uncond_x = x.clone()
        x[audio_mask] = audio_features[audio_valid]
        x[video_input_mask] = video_embeddings
        uncond_x[video_input_mask] = video_embeddings
        attention_mask = valid_ids.to(torch.long)
        outputs = self.lm.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        uncond_outputs = self.lm.model(
            inputs_embeds=uncond_x,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        conditions = outputs.last_hidden_state[video_loss_mask]
        uncond_conditions = uncond_outputs.last_hidden_state[video_loss_mask]

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
        uncond_conditions = uncond_conditions[selected_among_valid]

        targets = targets_padded[selected]
        batch_index = torch.arange(video_latents.shape[0], device=video_latents.device)
        batch_index = batch_index[:, None].expand_as(selected)[selected]
        reference = video_latents[:, 0][batch_index]

        # VibeVoice diffusion objective (v-prediction), applied to WAN frames.
        # Conditional and audio-empty unconditional branches share exactly the
        # same target/noise/timestep so their losses and later CFG are aligned.
        noise = torch.randn_like(targets)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (targets.shape[0],),
            device=targets.device,
            dtype=torch.long,
        )
        noisy = self.noise_scheduler.add_noise(targets, noise, timesteps)
        target_velocity = self.noise_scheduler.get_velocity(targets, noise, timesteps)
        dit_timesteps = timesteps.to(dtype=targets.dtype)
        keep_condition = torch.zeros(
            targets.shape[0], device=targets.device, dtype=torch.bool
        )
        prediction = self.video_dit(
            noisy, dit_timesteps, conditions, reference,
            drop_condition=keep_condition,
        )
        uncond_prediction = self.video_dit(
            noisy, dit_timesteps, uncond_conditions, reference,
            drop_condition=keep_condition,
        )
        diffusion_loss = torch.nn.functional.mse_loss(
            prediction.float(), target_velocity.float(), reduction="mean"
        )
        uncond_diffusion_loss = torch.nn.functional.mse_loss(
            uncond_prediction.float(), target_velocity.float(), reduction="mean"
        )
        loss = diffusion_loss * 0.8 + uncond_diffusion_loss * 0.2
        return {
            "loss": loss,
            "diffusion_loss": diffusion_loss,
            "uncond_diffusion_loss": uncond_diffusion_loss,
        }

    @torch.no_grad()
    def sample_video_frame(
        self,
        hidden: torch.Tensor,
        reference: torch.Tensor,
        num_steps: int = 8,
        cfg_scale: float = 1.5,
        generator: Optional[torch.Generator] = None,
        uncond_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DPM-Solver sample one [B,16,64,64] WAN frame."""
        b = hidden.shape[0]
        z = torch.randn(
            b, 16, 64, 64, device=hidden.device, dtype=hidden.dtype, generator=generator
        )
        self.noise_scheduler.set_timesteps(num_steps, device=hidden.device)
        do_cfg = cfg_scale != 1.0
        if do_cfg and uncond_hidden is None:
            uncond_hidden = torch.zeros_like(hidden)
        keep_condition = torch.zeros(
            b * (2 if do_cfg else 1), device=hidden.device, dtype=torch.bool
        )
        for timestep in self.noise_scheduler.timesteps:
            if do_cfg:
                z_in = torch.cat([z, z], dim=0)
                h_in = torch.cat([hidden, uncond_hidden], dim=0)
                ref_in = torch.cat([reference, reference], dim=0)
                t_in = timestep.expand(2 * b).to(dtype=z.dtype)
                model_output = self.video_dit(
                    z_in, t_in, h_in, ref_in,
                    drop_condition=keep_condition,
                )
                cond_output, uncond_output = model_output.chunk(2)
                model_output = uncond_output + cfg_scale * (cond_output - uncond_output)
            else:
                t_in = timestep.expand(b).to(dtype=z.dtype)
                model_output = self.video_dit(
                    z, t_in, hidden, reference,
                    drop_condition=keep_condition,
                )
            z = self.noise_scheduler.step(model_output, timestep, z).prev_sample
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
        cond_x = x.clone()
        uncond_x = x.clone()
        cond_x[audio_mask] = audio_features[audio_valid]

        do_cfg = cfg_scale != 1.0
        prefill_x = torch.cat([cond_x, uncond_x], dim=0) if do_cfg else cond_x
        prefill_mask = (
            torch.cat([valid_ids, valid_ids], dim=0) if do_cfg else valid_ids
        ).to(torch.long)
        out = self.lm.model(
            inputs_embeds=prefill_x,
            attention_mask=prefill_mask,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        batch_size = x.shape[0]
        rows = torch.arange(prefill_x.shape[0], device=x.device)
        starts = torch.tensor(video_start, device=x.device)
        starts = starts.repeat(2) if do_cfg else starts
        h = out.last_hidden_state[rows, starts]

        generated = [reference]
        for frame_index in range(num_frames):
            cond_h = h[:batch_size]
            uncond_h = h[batch_size:] if do_cfg else None
            frame = self.sample_video_frame(
                cond_h, reference, num_steps, cfg_scale, generator,
                uncond_hidden=uncond_h,
            )
            generated.append(frame)
            if frame_index + 1 < num_frames:
                frame_token = self.video_connector(frame).unsqueeze(1)
                if do_cfg:
                    frame_token = frame_token.repeat(2, 1, 1)
                out = self.lm.model(
                    inputs_embeds=frame_token,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                h = out.last_hidden_state[:, -1]
        return torch.stack(generated, dim=2)
