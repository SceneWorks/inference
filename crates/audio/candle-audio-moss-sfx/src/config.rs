//! Snapshot configuration parsing for MOSS-SoundEffect v2.0 (sc-12841).
//!
//! The pinned snapshot is a diffusers-style directory (`model_index.json` at the root naming the
//! components); each component carries its own JSON config. These structs mirror the exact fields
//! the reference pipeline reads (`WanAudioPipeline.from_pretrained` +
//! `MossSoundEffectPipeline.from_pretrained`), with reference defaults for optional fields.

use std::path::Path;

use candle_audio::{AudioError, Result};
use serde::Deserialize;

/// `model_index.json` — pipeline identity + output surface.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelIndex {
    #[serde(rename = "_class_name")]
    pub class_name: String,
    /// Output sample rate in Hz (48 000 for the v2.0 release).
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    /// The longest latent window the model denoises, in whole seconds (30 for v2.0).
    #[serde(default = "default_max_inference_seconds")]
    pub max_inference_seconds: u32,
    /// `"dac"` — the only VAE type this port supports (the legacy `"oobleck"` path uses a
    /// different RoPE table and VAE and is not shipped by the pinned snapshot).
    #[serde(default = "default_vae_type")]
    pub vae_type: String,
}

fn default_sample_rate() -> u32 {
    48_000
}
fn default_max_inference_seconds() -> u32 {
    30
}
fn default_vae_type() -> String {
    "dac".to_string()
}

/// `scheduler/scheduler_config.json` — the flow-match schedule parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    /// The classic flow-match shift (`σ = shift·s / (1 + (shift−1)·s)`); 5.0 for v2.0.
    #[serde(default = "default_shift")]
    pub shift: f64,
    /// Terminal σ of the linear ramp (0.0 for v2.0).
    #[serde(default)]
    pub sigma_min: f64,
    /// `true` ⇒ `linspace(1, σ_min, steps+1)[:-1]` (the v2.0 shape) rather than
    /// `linspace(1, σ_min, steps)`.
    #[serde(default)]
    pub extra_one_step: bool,
    /// Timestep scale: the DiT timestep at step `k` is `σ_k · num_train_timesteps`.
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: u32,
}

fn default_shift() -> f64 {
    5.0
}
fn default_num_train_timesteps() -> u32 {
    1000
}

/// `transformer/config.json` — the Wan-style 1D audio DiT hyperparameters.
#[derive(Debug, Clone, Deserialize)]
pub struct DitConfig {
    /// Latent channels in (128 — the DAC VAE latent dim).
    pub in_dim: usize,
    /// Latent channels out (128).
    pub out_dim: usize,
    /// Text-context width (2048 — Qwen3-1.7B hidden size).
    pub text_dim: usize,
    /// Sinusoidal timestep-embedding width (256).
    pub freq_dim: usize,
    /// LayerNorm/RMSNorm epsilon (1e-6).
    pub eps: f64,
    /// 1-D patch size (`[1]` for v2.0 — the patch conv is k=1/s=1).
    pub patch_size: Vec<usize>,
    /// Hidden width (1536 for the 1.3B variant).
    pub dim: usize,
    /// FFN width (8960).
    pub ffn_dim: usize,
    /// Attention heads (12; head_dim = dim / num_heads = 128).
    pub num_heads: usize,
    /// DiT blocks (30).
    pub num_layers: usize,
    /// Must be false — the audio DiT has no image branch.
    #[serde(default)]
    pub has_image_input: bool,
    /// `"dac"` selects the plain 1-D RoPE table (`precompute_freqs_cis_1d`).
    #[serde(default = "default_vae_type")]
    pub vae_type: String,
}

impl DitConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    /// Reject configurations this port does not implement, so a drifted snapshot fails loudly
    /// at load rather than silently mis-sampling.
    pub fn validate(&self) -> Result<()> {
        if self.has_image_input {
            return Err(AudioError::Msg(
                "moss-sfx DiT: has_image_input=true is not the audio checkpoint shape".into(),
            ));
        }
        if self.vae_type != "dac" {
            return Err(AudioError::Msg(format!(
                "moss-sfx DiT: vae_type {:?} unsupported (this port implements the \"dac\" RoPE \
                 path only)",
                self.vae_type
            )));
        }
        if self.patch_size != [1] {
            return Err(AudioError::Msg(format!(
                "moss-sfx DiT: patch_size {:?} unsupported (the v2.0 checkpoint uses [1])",
                self.patch_size
            )));
        }
        if !self.dim.is_multiple_of(self.num_heads) || !self.head_dim().is_multiple_of(2) {
            return Err(AudioError::Msg(format!(
                "moss-sfx DiT: dim {} not an even-head multiple of num_heads {}",
                self.dim, self.num_heads
            )));
        }
        Ok(())
    }
}

/// `text_encoder/config.json` — the Qwen3-1.7B causal-LM hyperparameters this port reads.
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
                "moss-sfx text encoder: sliding-window attention is not implemented (the pinned \
                 Qwen3-1.7B config has it off)"
                    .into(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(AudioError::Msg(format!(
                "moss-sfx text encoder: heads {} not a multiple of kv heads {}",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        Ok(())
    }
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
    pub scheduler: SchedulerConfig,
    pub dit: DitConfig,
    pub text_encoder: TextEncoderConfig,
}

impl SnapshotConfig {
    pub fn from_snapshot(root: &Path) -> Result<Self> {
        let index: ModelIndex = read_json(&root.join("model_index.json"))?;
        if index.class_name != "MossSoundEffectPipeline" {
            return Err(AudioError::Msg(format!(
                "{} is not a MOSS-SoundEffect snapshot (_class_name {:?})",
                root.display(),
                index.class_name
            )));
        }
        if index.vae_type != "dac" {
            return Err(AudioError::Msg(format!(
                "moss-sfx: vae_type {:?} unsupported (this port implements \"dac\" only)",
                index.vae_type
            )));
        }
        let scheduler: SchedulerConfig = read_json(&root.join("scheduler/scheduler_config.json"))?;
        let dit: DitConfig = read_json(&root.join("transformer/config.json"))?;
        dit.validate()?;
        let text_encoder: TextEncoderConfig = read_json(&root.join("text_encoder/config.json"))?;
        text_encoder.validate()?;
        if text_encoder.hidden_size != dit.text_dim {
            return Err(AudioError::Msg(format!(
                "moss-sfx: text encoder hidden_size {} != DiT text_dim {}",
                text_encoder.hidden_size, dit.text_dim
            )));
        }
        Ok(Self {
            index,
            scheduler,
            dit,
            text_encoder,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_snapshot(dir: &Path) {
        std::fs::create_dir_all(dir.join("scheduler")).unwrap();
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::create_dir_all(dir.join("text_encoder")).unwrap();
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "MossSoundEffectPipeline", "sample_rate": 48000,
                "max_inference_seconds": 30, "vae_type": "dac"}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("scheduler/scheduler_config.json"),
            r#"{"_class_name": "FlowMatchScheduler", "shift": 5.0, "sigma_min": 0.0,
                "extra_one_step": true, "num_train_timesteps": 1000}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("transformer/config.json"),
            r#"{"in_dim": 128, "out_dim": 128, "text_dim": 2048, "freq_dim": 256,
                "eps": 1e-6, "patch_size": [1], "has_image_input": false, "vae_type": "dac",
                "dim": 1536, "ffn_dim": 8960, "num_heads": 12, "num_layers": 30}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("text_encoder/config.json"),
            r#"{"vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144,
                "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "attention_bias": false, "rms_norm_eps": 1e-6,
                "rope_theta": 1000000, "use_sliding_window": false}"#,
        )
        .unwrap();
    }

    #[test]
    fn parses_the_pinned_snapshot_shape() {
        let dir = std::env::temp_dir().join("moss-sfx-config-parse");
        let _ = std::fs::remove_dir_all(&dir);
        write_snapshot(&dir);
        let cfg = SnapshotConfig::from_snapshot(&dir).unwrap();
        assert_eq!(cfg.index.sample_rate, 48_000);
        assert_eq!(cfg.index.max_inference_seconds, 30);
        assert!(cfg.scheduler.extra_one_step);
        assert_eq!(cfg.scheduler.shift, 5.0);
        assert_eq!(cfg.dit.dim, 1536);
        assert_eq!(cfg.dit.head_dim(), 128);
        assert_eq!(cfg.text_encoder.num_hidden_layers, 28);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_foreign_or_drifted_snapshots() {
        let dir = std::env::temp_dir().join("moss-sfx-config-reject");
        let _ = std::fs::remove_dir_all(&dir);
        write_snapshot(&dir);
        // A different pipeline class is rejected up front.
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "SomethingElse"}"#,
        )
        .unwrap();
        assert!(SnapshotConfig::from_snapshot(&dir).is_err());
        // An oobleck-typed DiT config is rejected (unimplemented legacy RoPE path).
        write_snapshot(&dir);
        std::fs::write(
            dir.join("transformer/config.json"),
            r#"{"in_dim": 64, "out_dim": 64, "text_dim": 2048, "freq_dim": 256,
                "eps": 1e-6, "patch_size": [1], "has_image_input": false, "vae_type": "oobleck",
                "dim": 1536, "ffn_dim": 8960, "num_heads": 12, "num_layers": 30}"#,
        )
        .unwrap();
        assert!(SnapshotConfig::from_snapshot(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
