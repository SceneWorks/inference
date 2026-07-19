//! Snapshot configuration parsing for ACE-Step 1.5 (sc-12842).
//!
//! The pinned snapshot is a diffusers-style directory (`model_index.json` at the root naming the
//! components); each component (`transformer/`, `vae/`, `condition_encoder/`, `text_encoder/`,
//! `scheduler/`) carries its own JSON config. These structs mirror the exact fields the reference
//! `AceStepPipeline` reads, with reference defaults for optional fields, and cross-check the
//! dimensions that must agree across components so a drifted snapshot fails loudly at load rather
//! than silently mis-sampling.

use std::path::Path;

use candle_audio::{AudioError, Result};
use serde::Deserialize;

/// `model_index.json` — pipeline identity.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelIndex {
    #[serde(rename = "_class_name")]
    pub class_name: String,
}

/// `vae/config.json` — the `AutoencoderOobleck` stereo music VAE. The decode path is fully
/// determined by these fields (`decoder_input_channels` latents → `audio_channels` waveform,
/// upsampling by `∏ downsampling_ratios`).
#[derive(Debug, Clone, Deserialize)]
pub struct VaeConfig {
    /// Output waveform channels (2 — stereo).
    pub audio_channels: usize,
    /// Per-stage channel multipliers applied to `decoder_channels` (encoder order; the decoder
    /// walks them in reverse).
    pub channel_multiples: Vec<usize>,
    /// Base decoder width (128).
    pub decoder_channels: usize,
    /// Latent channels the decoder consumes (64 — the DiT `audio_acoustic_hidden_dim`).
    pub decoder_input_channels: usize,
    /// Per-stage temporal ratios (encoder order; the decoder upsamples in reverse). `∏` = 1920,
    /// so 48 kHz / 1920 = 25 latent frames per second.
    pub downsampling_ratios: Vec<usize>,
    /// Output sample rate in Hz (48 000).
    pub sampling_rate: u32,
}

impl VaeConfig {
    /// Samples per latent frame (`∏ downsampling_ratios`).
    pub fn hop_length(&self) -> usize {
        self.downsampling_ratios.iter().product()
    }

    /// Latent frames per second (`sampling_rate / hop_length`).
    pub fn latents_per_second(&self) -> f64 {
        self.sampling_rate as f64 / self.hop_length() as f64
    }

    pub fn validate(&self) -> Result<()> {
        if self.audio_channels != 2 {
            return Err(AudioError::Msg(format!(
                "acestep VAE: audio_channels {} unsupported (the pinned Oobleck VAE is stereo)",
                self.audio_channels
            )));
        }
        if self.channel_multiples.len() != self.downsampling_ratios.len() {
            return Err(AudioError::Msg(format!(
                "acestep VAE: channel_multiples ({}) and downsampling_ratios ({}) length mismatch",
                self.channel_multiples.len(),
                self.downsampling_ratios.len()
            )));
        }
        if self.hop_length() == 0 || !self.sampling_rate.is_multiple_of(self.hop_length() as u32) {
            return Err(AudioError::Msg(format!(
                "acestep VAE: sampling_rate {} not divisible by hop_length {}",
                self.sampling_rate,
                self.hop_length()
            )));
        }
        Ok(())
    }
}

/// `transformer/config.json` — the `AceStepTransformer1DModel` DiT hyperparameters.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformerConfig {
    /// Patchify-conv input channels (192 = acoustic 64 + context 128).
    pub in_channels: usize,
    /// Acoustic latent channels (64 — the VAE latent dim, and the depatchify output width).
    pub audio_acoustic_hidden_dim: usize,
    /// Cross-attention context width (2048 — the condition encoder output).
    pub encoder_hidden_size: usize,
    /// Hidden width (2560).
    pub hidden_size: usize,
    /// FFN width (9728 — SwiGLU).
    pub intermediate_size: usize,
    /// Attention head dim (128; `num_attention_heads · head_dim = 4096 ≠ hidden_size`).
    pub head_dim: usize,
    /// Query heads (32).
    pub num_attention_heads: usize,
    /// Key/value heads (8 — grouped-query attention).
    pub num_key_value_heads: usize,
    /// DiT blocks (32).
    pub num_hidden_layers: usize,
    /// 1-D patch size (2 — the patchify Conv1d kernel/stride and depatchify ConvTranspose1d).
    pub patch_size: usize,
    /// Per-layer attention type: `"sliding_attention"` or `"full_attention"`, one per block.
    pub layer_types: Vec<String>,
    /// Sliding-window width for the `sliding_attention` layers (128).
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// Attention q/k/v projection bias (false for the pinned checkpoint).
    #[serde(default)]
    pub attention_bias: bool,
    /// Guidance-distilled checkpoint (turbo): CFG is baked into the weights, so guidance is
    /// ignored regardless of the request value.
    #[serde(default)]
    pub is_turbo: bool,
}

impl TransformerConfig {
    pub fn validate(&self, vae: &VaeConfig) -> Result<()> {
        if self.audio_acoustic_hidden_dim != vae.decoder_input_channels {
            return Err(AudioError::Msg(format!(
                "acestep: DiT audio_acoustic_hidden_dim {} != VAE decoder_input_channels {}",
                self.audio_acoustic_hidden_dim, vae.decoder_input_channels
            )));
        }
        // Text-to-music context is [src_latents(acoustic) | chunk_mask(acoustic)] concatenated
        // with the noisy latents (acoustic): 3 × acoustic == in_channels.
        if self.in_channels != 3 * self.audio_acoustic_hidden_dim {
            return Err(AudioError::Msg(format!(
                "acestep: DiT in_channels {} != 3 × audio_acoustic_hidden_dim {} (noisy + src + \
                 mask)",
                self.in_channels, self.audio_acoustic_hidden_dim
            )));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(AudioError::Msg(format!(
                "acestep: DiT layer_types ({}) != num_hidden_layers ({})",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        for t in &self.layer_types {
            if t != "sliding_attention" && t != "full_attention" {
                return Err(AudioError::Msg(format!(
                    "acestep: DiT unknown layer_type {t:?} (expected sliding_attention / \
                     full_attention)"
                )));
            }
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(AudioError::Msg(format!(
                "acestep: DiT heads {} not a multiple of kv heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(AudioError::Msg(format!(
                "acestep: DiT head_dim {} must be even for RoPE",
                self.head_dim
            )));
        }
        Ok(())
    }

    pub fn is_sliding(&self, layer: usize) -> bool {
        self.layer_types[layer] == "sliding_attention"
    }
}

/// `condition_encoder/config.json` — the `AceStepConditionEncoder` (lyric + timbre encoders +
/// text/lyric/timbre fusion into the DiT cross-attention context).
#[derive(Debug, Clone, Deserialize)]
pub struct ConditionEncoderConfig {
    /// Fused context width (2048 — must equal the DiT `encoder_hidden_size`).
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Lyric contextual-encoder depth (8).
    pub num_lyric_encoder_hidden_layers: usize,
    /// Timbre encoder depth (4).
    pub num_timbre_encoder_hidden_layers: usize,
    /// Text-encoder hidden width feeding the fusion (1024 — Qwen3-Embedding-0.6B).
    pub text_hidden_dim: usize,
    /// Timbre latent width (64 — the VAE acoustic dim).
    pub timbre_hidden_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub sliding_window: usize,
}

impl ConditionEncoderConfig {
    pub fn validate(
        &self,
        transformer: &TransformerConfig,
        text: &TextEncoderConfig,
    ) -> Result<()> {
        if self.hidden_size != transformer.encoder_hidden_size {
            return Err(AudioError::Msg(format!(
                "acestep: condition encoder hidden_size {} != DiT encoder_hidden_size {}",
                self.hidden_size, transformer.encoder_hidden_size
            )));
        }
        if self.text_hidden_dim != text.hidden_size {
            return Err(AudioError::Msg(format!(
                "acestep: condition encoder text_hidden_dim {} != text encoder hidden_size {}",
                self.text_hidden_dim, text.hidden_size
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(AudioError::Msg(format!(
                "acestep: condition encoder heads {} not a multiple of kv heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        Ok(())
    }
}

/// `text_encoder/config.json` — the Qwen3-Embedding-0.6B causal-LM hyperparameters this port
/// reads (the prompt encoder + the lyric token-embedding lookup).
#[derive(Debug, Clone, Deserialize)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default)]
    pub attention_bias: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
}

impl TextEncoderConfig {
    pub fn validate(&self) -> Result<()> {
        if self.use_sliding_window {
            return Err(AudioError::Msg(
                "acestep text encoder: sliding-window attention is not implemented (the pinned \
                 Qwen3-Embedding-0.6B config has it off)"
                    .into(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(AudioError::Msg(format!(
                "acestep text encoder: heads {} not a multiple of kv heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        Ok(())
    }
}

/// `scheduler/scheduler_config.json` — ACE-Step configures `FlowMatchEulerDiscreteScheduler` with
/// `num_train_timesteps=1` and `shift=1.0`; the pipeline computes its own shifted sigma ladder and
/// feeds it via `set_timesteps(sigmas=…)`. The only field this port reads is the default `shift`
/// (the request's `scheduler_shift` overrides it), falling back to the turbo default 3.0.
#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_shift")]
    pub shift: f64,
}

fn default_shift() -> f64 {
    3.0
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| AudioError::Msg(format!("parse {}: {e}", path.display())))
}

/// All configs of one snapshot directory, parsed and cross-checked.
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    pub index: ModelIndex,
    pub vae: VaeConfig,
    pub transformer: TransformerConfig,
    pub condition: ConditionEncoderConfig,
    pub text_encoder: TextEncoderConfig,
    pub scheduler: SchedulerConfig,
}

impl SnapshotConfig {
    pub fn from_snapshot(root: &Path) -> Result<Self> {
        let index: ModelIndex = read_json(&root.join("model_index.json"))?;
        if index.class_name != "AceStepPipeline" {
            return Err(AudioError::Msg(format!(
                "{} is not an ACE-Step snapshot (_class_name {:?})",
                root.display(),
                index.class_name
            )));
        }
        let vae: VaeConfig = read_json(&root.join("vae/config.json"))?;
        vae.validate()?;
        let transformer: TransformerConfig = read_json(&root.join("transformer/config.json"))?;
        transformer.validate(&vae)?;
        let text_encoder: TextEncoderConfig = read_json(&root.join("text_encoder/config.json"))?;
        text_encoder.validate()?;
        let condition: ConditionEncoderConfig =
            read_json(&root.join("condition_encoder/config.json"))?;
        condition.validate(&transformer, &text_encoder)?;
        let scheduler: SchedulerConfig = read_json(&root.join("scheduler/scheduler_config.json"))
            .unwrap_or(SchedulerConfig {
                shift: default_shift(),
            });
        Ok(Self {
            index,
            vae,
            transformer,
            condition,
            text_encoder,
            scheduler,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_snapshot(dir: &Path) {
        for sub in [
            "scheduler",
            "transformer",
            "text_encoder",
            "condition_encoder",
            "vae",
        ] {
            std::fs::create_dir_all(dir.join(sub)).unwrap();
        }
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "AceStepPipeline"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("vae/config.json"),
            r#"{"_class_name": "AutoencoderOobleck", "audio_channels": 2,
                "channel_multiples": [1,2,4,8,16], "decoder_channels": 128,
                "decoder_input_channels": 64, "downsampling_ratios": [2,4,4,6,10],
                "encoder_hidden_size": 128, "sampling_rate": 48000}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("transformer/config.json"),
            r#"{"in_channels": 192, "audio_acoustic_hidden_dim": 64, "encoder_hidden_size": 2048,
                "hidden_size": 2560, "intermediate_size": 9728, "head_dim": 128,
                "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 4,
                "patch_size": 2, "layer_types": ["sliding_attention","full_attention",
                "sliding_attention","full_attention"], "sliding_window": 128,
                "rms_norm_eps": 1e-6, "rope_theta": 1000000, "is_turbo": true}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("text_encoder/config.json"),
            r#"{"vocab_size": 151669, "hidden_size": 1024, "intermediate_size": 3072,
                "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "attention_bias": false, "rms_norm_eps": 1e-6,
                "rope_theta": 1000000, "use_sliding_window": false}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("condition_encoder/config.json"),
            r#"{"hidden_size": 2048, "intermediate_size": 6144, "head_dim": 128,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "num_lyric_encoder_hidden_layers": 8, "num_timbre_encoder_hidden_layers": 4,
                "text_hidden_dim": 1024, "timbre_hidden_dim": 64, "rms_norm_eps": 1e-6,
                "rope_theta": 1000000, "sliding_window": 128}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("scheduler/scheduler_config.json"),
            r#"{"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 3.0}"#,
        )
        .unwrap();
    }

    #[test]
    fn parses_and_cross_checks_the_pinned_snapshot_shape() {
        let dir = std::env::temp_dir().join("acestep-config-parse");
        let _ = std::fs::remove_dir_all(&dir);
        write_snapshot(&dir);
        let cfg = SnapshotConfig::from_snapshot(&dir).unwrap();
        assert_eq!(cfg.vae.hop_length(), 1920);
        assert_eq!(cfg.vae.latents_per_second(), 25.0);
        assert_eq!(cfg.transformer.hidden_size, 2560);
        assert_eq!(cfg.transformer.in_channels, 192);
        assert!(cfg.transformer.is_turbo);
        assert!(cfg.transformer.is_sliding(0));
        assert!(!cfg.transformer.is_sliding(1));
        assert_eq!(cfg.condition.num_lyric_encoder_hidden_layers, 8);
        assert_eq!(cfg.text_encoder.hidden_size, 1024);
        assert_eq!(cfg.scheduler.shift, 3.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_foreign_or_drifted_snapshots() {
        let dir = std::env::temp_dir().join("acestep-config-reject");
        let _ = std::fs::remove_dir_all(&dir);
        write_snapshot(&dir);
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "SomethingElse"}"#,
        )
        .unwrap();
        assert!(SnapshotConfig::from_snapshot(&dir).is_err());
        // A DiT whose in_channels breaks the 3×acoustic invariant is rejected.
        write_snapshot(&dir);
        std::fs::write(
            dir.join("transformer/config.json"),
            r#"{"in_channels": 128, "audio_acoustic_hidden_dim": 64, "encoder_hidden_size": 2048,
                "hidden_size": 2560, "intermediate_size": 9728, "head_dim": 128,
                "num_attention_heads": 32, "num_key_value_heads": 8, "num_hidden_layers": 2,
                "patch_size": 2, "layer_types": ["sliding_attention","full_attention"],
                "sliding_window": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000}"#,
        )
        .unwrap();
        assert!(SnapshotConfig::from_snapshot(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
