//! `config.json` parsing for MOSS-TTSD-v0.5 (sc-13360).
//!
//! MOSS-TTSD's dialogue model (`model_type: moss_ttsd`, `MossTTSDForCausalLM`) is a **flat** config:
//! a standard Qwen3 causal-LM backbone (`Qwen3Model`) plus the delay-pattern multi-channel head
//! surface. At every sequence position the token is a **`channels`-wide** (8) grid: channel 0 is the
//! text channel (whose vocabulary also carries the first speech codebook, mapped into
//! `speech_token_range`) and channels 1..channels-1 are the remaining audio codebooks. These fields
//! mirror exactly what the reference `configuration_moss_ttsd.MossTTSDConfig` reads.

use std::path::Path;

use candle_audio::{AudioError, Result};
use serde::Deserialize;

use crate::blocks::BlockConfig;

/// The assembled MOSS-TTSD config (`config.json`) — the Qwen3 backbone hyperparameters plus the
/// delay-pattern channel/speech-token surface.
#[derive(Debug, Clone, Deserialize)]
pub struct MossTtsdConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default = "default_model_type")]
    pub model_type: String,

    // --- Qwen3 backbone ---
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

    // --- delay-pattern multi-channel head surface ---
    /// The number of codebook channels per position (8): channel 0 = text + speech codebook 0,
    /// channels 1..channels-1 = the remaining audio codebooks.
    #[serde(default = "default_channels")]
    pub channels: usize,
    /// The audio-codebook vocabulary for channels 1..channels-1 (1025 = 1024 codes + pad 1024).
    #[serde(default = "default_speech_vocab_size")]
    pub speech_vocab_size: usize,
    /// The in-codebook padding id used on the audio channels (1024).
    #[serde(default = "default_speech_pad_token")]
    pub speech_pad_token: u32,
    /// `[start, end)` of the text-channel vocab range that carries the **first** speech codebook
    /// (`[151665, 152689)`): a channel-0 audio token minus `start` is codebook-0 code `0..1024`.
    #[serde(default = "default_speech_token_range")]
    pub speech_token_range: [u32; 2],
    /// The channel-0 end-of-speech token id (152694) — a sampled channel-0 token outside
    /// `speech_token_range` ends the audio stream and starts the delay-tail drain.
    #[serde(default = "default_speech_eos_token")]
    pub speech_eos_token: u32,
    /// The text-channel padding id threaded into the delay tail (`<|endoftext|>` = 151643, the
    /// tokenizer pad). Sourced from `bos_token_id`/`eos_token_id` (both 151643) with a default.
    #[serde(default = "default_text_pad", alias = "bos_token_id")]
    pub text_pad_id: u32,
}

fn default_model_type() -> String {
    "moss_ttsd".into()
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_channels() -> usize {
    8
}
fn default_speech_vocab_size() -> usize {
    1025
}
fn default_speech_pad_token() -> u32 {
    1024
}
fn default_speech_token_range() -> [u32; 2] {
    [151_665, 152_689]
}
fn default_speech_eos_token() -> u32 {
    152_694
}
fn default_text_pad() -> u32 {
    151_643
}

impl MossTtsdConfig {
    /// Parse `config.json` from a snapshot directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| AudioError::Msg(format!("read {}: {e}", path.display())))?;
        Self::from_json(&text)
    }

    /// Parse a `config.json` document, validating the delay-pattern invariants the port relies on.
    pub fn from_json(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text)
            .map_err(|e| AudioError::Msg(format!("parse MOSS-TTSD config.json: {e}")))?;
        if cfg.channels < 2 {
            return Err(AudioError::Msg(format!(
                "config.json: channels ({}) must be >= 2 (text + >=1 audio codebook)",
                cfg.channels
            )));
        }
        if cfg.speech_vocab_size == 0 {
            return Err(AudioError::Msg(
                "config.json: speech_vocab_size must be > 0".into(),
            ));
        }
        if cfg.speech_token_range[1] <= cfg.speech_token_range[0] {
            return Err(AudioError::Msg(format!(
                "config.json: speech_token_range {:?} is not an increasing [start, end)",
                cfg.speech_token_range
            )));
        }
        // The channel-0 speech window must be wide enough for all 1024 codebook-0 codes.
        let window = cfg.speech_token_range[1] - cfg.speech_token_range[0];
        if (window as usize) < cfg.speech_vocab_size - 1 {
            return Err(AudioError::Msg(format!(
                "config.json: speech_token_range width {window} < codebook-0 code count {}",
                cfg.speech_vocab_size - 1
            )));
        }
        Ok(cfg)
    }

    /// The number of audio codebook channels (channels excluding the text channel): 7 for v0.5.
    pub fn audio_channels(&self) -> usize {
        self.channels - 1
    }

    /// Extract the Qwen3 [`BlockConfig`] for the backbone decoder layers.
    pub fn block(&self) -> BlockConfig {
        BlockConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            attention_bias: self.attention_bias,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed but structurally faithful `config.json` (the real MOSS-TTSD-v0.5 hyperparameters).
    pub(crate) const SAMPLE: &str = r#"{
        "model_type": "moss_ttsd",
        "architectures": ["MossTTSDForCausalLM"],
        "vocab_size": 152697,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000,
        "attention_bias": false,
        "bos_token_id": 151643,
        "channels": 8,
        "speech_vocab_size": 1025,
        "speech_pad_token": 1024,
        "speech_token_range": [151665, 152689],
        "speech_eos_token": 152694
    }"#;

    #[test]
    fn parses_the_real_hyperparameters() {
        let cfg = MossTtsdConfig::from_json(SAMPLE).unwrap();
        assert_eq!(cfg.architectures, ["MossTTSDForCausalLM"]);
        assert_eq!(cfg.model_type, "moss_ttsd");
        assert_eq!(cfg.vocab_size, 152697);
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.channels, 8);
        assert_eq!(cfg.audio_channels(), 7);
        assert_eq!(cfg.speech_vocab_size, 1025);
        assert_eq!(cfg.speech_pad_token, 1024);
        assert_eq!(cfg.speech_token_range, [151665, 152689]);
        assert_eq!(cfg.speech_eos_token, 152694);
        assert_eq!(cfg.text_pad_id, 151643);
        let b = cfg.block();
        assert_eq!(b.hidden_size, 2048);
        assert_eq!(b.head_dim, 128);
        assert_eq!(b.rope_theta, 1_000_000.0);
    }

    #[test]
    fn rejects_degenerate_channels() {
        let bad = SAMPLE.replace("\"channels\": 8", "\"channels\": 1");
        assert!(MossTtsdConfig::from_json(&bad).is_err());
    }

    #[test]
    fn rejects_inverted_speech_range() {
        let bad = SAMPLE.replace("[151665, 152689]", "[152689, 151665]");
        assert!(MossTtsdConfig::from_json(&bad).is_err());
    }
}
