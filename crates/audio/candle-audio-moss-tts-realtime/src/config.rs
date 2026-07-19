//! `config.json` parsing for MOSS-TTS-Realtime-1.7B (sc-13334).
//!
//! The pinned snapshot carries a single top-level `config.json` (the `MossTTSRealtime`
//! architecture) with two nested sub-configs — `language_config` (the Qwen3-1.7B backbone) and
//! `local_config` (the CSM-style local/depth transformer that decodes the RVQ codebooks) — plus
//! the top-level audio-token ids the processor threads through the multi-channel input. These
//! structs mirror exactly the fields the reference `configuration_mossttsrealtime.py` reads, with
//! the reference defaults for the optional fields.

use std::path::Path;

use candle_audio::{AudioError, Result};
use serde::Deserialize;

/// The Qwen3 backbone hyperparameters (`config.json.language_config`) — a standard Qwen3-1.7B
/// causal LM (GQA with per-head q/k RMSNorm, half-split RoPE, SiLU MLP).
#[derive(Debug, Clone, Deserialize)]
pub struct LanguageConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_bos")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
}

/// The local/depth transformer hyperparameters (`config.json.local_config`) — a small
/// (4-layer) causal transformer run **once per audio frame** over the RVQ depth axis
/// (`rvq` = 16 codebooks), seeded by the backbone's last hidden state at depth position 0 and
/// projecting each depth position through its own per-codebook LM head.
#[derive(Debug, Clone, Deserialize)]
pub struct LocalConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub attention_bias: bool,
    /// The number of RVQ codebooks per audio frame (16).
    pub rvq: usize,
    /// The audio codebook vocabulary (1027 = 1024 real codes + pad/bos/eos).
    pub audio_vocab_size: usize,
    /// The in-codebook padding id (1024).
    pub audio_pad_token: usize,
}

/// The assembled MOSS-TTS-Realtime config (`config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct MossTtsRealtimeConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub language_config: LanguageConfig,
    pub local_config: LocalConfig,
    /// The number of RVQ codebooks per frame (16) — mirrors `local_config.rvq`.
    pub rvq: usize,
    /// Audio codebook vocabulary (1027).
    pub audio_vocab_size: usize,
    /// In-codebook padding id (1024).
    pub audio_pad_token: usize,
    /// The text-channel padding id emitted on audio-only continuation steps (151655).
    #[serde(default = "default_text_pad")]
    pub text_pad: u32,
    /// The reference-audio padding id on the text channel (151654).
    #[serde(default = "default_reference_audio_pad")]
    pub reference_audio_pad: u32,
}

fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_bos() -> u32 {
    151_643
}
fn default_eos() -> u32 {
    151_645
}
fn default_text_pad() -> u32 {
    151_655
}
fn default_reference_audio_pad() -> u32 {
    151_654
}

impl MossTtsRealtimeConfig {
    /// Parse `config.json` from a snapshot directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Parse a `config.json` document.
    pub fn from_json(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text)
            .map_err(|e| AudioError::Msg(format!("parse MOSS-TTS-Realtime config.json: {e}")))?;
        if cfg.rvq != cfg.local_config.rvq {
            return Err(AudioError::Msg(format!(
                "config.json: top-level rvq ({}) disagrees with local_config.rvq ({})",
                cfg.rvq, cfg.local_config.rvq
            )));
        }
        if cfg.rvq == 0 {
            return Err(AudioError::Msg("config.json: rvq must be > 0".into()));
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed but structurally faithful `config.json` (real hyperparameters, from the pinned
    /// snapshot) — no weights required.
    pub(crate) const SAMPLE: &str = r#"{
        "architectures": ["MossTTSRealtime"],
        "audio_pad_token": 1024,
        "audio_vocab_size": 1027,
        "rvq": 16,
        "text_pad": 151655,
        "reference_audio_pad": 151654,
        "language_config": {
            "vocab_size": 151936, "hidden_size": 2048, "intermediate_size": 6144,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
            "attention_bias": false, "bos_token_id": 151643, "eos_token_id": 151645
        },
        "local_config": {
            "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 4,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "rms_norm_eps": 1e-6, "rope_theta": 1000000, "attention_bias": false,
            "rvq": 16, "audio_vocab_size": 1027, "audio_pad_token": 1024
        }
    }"#;

    #[test]
    fn parses_the_real_hyperparameters() {
        let cfg = MossTtsRealtimeConfig::from_json(SAMPLE).unwrap();
        assert_eq!(cfg.architectures, ["MossTTSRealtime"]);
        assert_eq!(cfg.rvq, 16);
        assert_eq!(cfg.audio_vocab_size, 1027);
        assert_eq!(cfg.audio_pad_token, 1024);
        assert_eq!(cfg.text_pad, 151655);
        assert_eq!(cfg.language_config.hidden_size, 2048);
        assert_eq!(cfg.language_config.num_hidden_layers, 28);
        assert_eq!(cfg.language_config.num_key_value_heads, 8);
        assert_eq!(cfg.language_config.head_dim, 128);
        assert_eq!(cfg.local_config.num_hidden_layers, 4);
        assert_eq!(cfg.local_config.rvq, 16);
    }

    #[test]
    fn rejects_inconsistent_rvq() {
        let bad = SAMPLE.replace(
            "\"rvq\": 16,\n        \"text_pad\"",
            "\"rvq\": 8,\n        \"text_pad\"",
        );
        assert!(MossTtsRealtimeConfig::from_json(&bad).is_err());
    }
}
