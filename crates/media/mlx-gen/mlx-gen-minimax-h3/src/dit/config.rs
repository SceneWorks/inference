//! The published `transformer/config.json` geometry.

use mlx_gen::{Error, Result};

/// MiniMax-H3 tags every row of the packed sequence with its modality and keeps one set of AdaLN
/// modulation parameters per `(timestep, modality)` pair.
///
/// `MINIMAX_H3_MODALITY_NUM` in `transformer_minimax_h3.py`. The tag values are `0` video, `1`
/// text, `2` audio (epic 17137 activity 18718 §6), and the AdaLN table row of a sequence row is
/// `timestep_index · MODALITY_NUM + tag`.
pub const MODALITY_NUM: i32 = 3;

/// Modulation vectors an AdaLN projection emits per `(timestep, modality)`: `shift_msa`,
/// `scale_msa`, `gate_msa`, `shift_mlp`, `scale_mlp`, `gate_mlp`, in that order.
pub const MODULATION_PARAMS: i32 = 6;

/// Geometry of `MiniMaxH3Transformer3DModel`, read from `transformer/config.json`.
///
/// The defaults are the shipped values. Every field is dimension-parametric so the parity fixture
/// can run the same code at committable dims.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxH3DitConfig {
    /// Attention heads. Shipped: 56.
    pub num_attention_heads: i32,
    /// Channels per head. Shipped: 128. Note `heads · head_dim` (7168) is **larger** than
    /// `hidden_size` (5376) — an unusual shape that a `hidden_size / heads` derivation gets wrong.
    pub attention_head_dim: i32,
    /// Residual-stream width. Shipped: 5376.
    pub hidden_size: i32,
    /// Blocks in the main stack. Shipped: 50.
    pub num_layers: i32,
    /// Blocks in the text token refiner. Shipped: 2.
    pub num_refiner_layers: i32,
    /// SwiGLU inner width. Shipped: 14336, so `ff.net.0.proj` emits `2 · 14336`.
    pub ffn_dim: i32,
    /// Video latent channels. Shipped: 24.
    pub in_channels: i32,
    /// Audio latent channels. Shipped: 32.
    pub audio_in_channels: i32,
    /// `(t, h, w)` patch. Shipped: `[1, 2, 2]`.
    pub patch_size: [i32; 3],
    /// Width of the text-encoder context. Shipped: 5120 — and there is **no projection inside the
    /// text encoder**; `context_embedder` here is what maps 5120 → `hidden_size`.
    pub text_dim: i32,
    /// Sinusoidal timestep-embedding width. Shipped: 256.
    pub freq_dim: i32,
    /// Timestep MLP inner width. Shipped: 5376.
    pub time_embed_hidden_dim: i32,
    /// Timestep MLP output width — the input of every AdaLN projection. Shipped: 2688.
    pub time_embed_dim: i32,
    /// Rotary frequencies **per axis**. Shipped: 16, so `2 · 3 · 16 = 96` of the 128 head channels
    /// rotate and 32 pass through.
    pub rope_freq_dim: i32,
    /// Rotary frequency base. Shipped: 10000.0.
    pub rope_theta: f32,
    /// Epsilon of `norm1` / `norm2`. Shipped: 1e-5.
    pub norm_eps: f32,
    /// Epsilon of `norm_q` / `norm_k`. Shipped: 1e-5.
    pub qk_norm_eps: f32,
    /// Epsilon of the refiner's `final_norm` and of `norm_out`. Shipped: 1e-5.
    pub final_norm_eps: f32,
}

impl Default for MiniMaxH3DitConfig {
    fn default() -> Self {
        Self {
            num_attention_heads: 56,
            attention_head_dim: 128,
            hidden_size: 5376,
            num_layers: 50,
            num_refiner_layers: 2,
            ffn_dim: 14336,
            in_channels: 24,
            audio_in_channels: 32,
            patch_size: [1, 2, 2],
            text_dim: 5120,
            freq_dim: 256,
            time_embed_hidden_dim: 5376,
            time_embed_dim: 2688,
            rope_freq_dim: 16,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
        }
    }
}

impl MiniMaxH3DitConfig {
    /// `heads · head_dim` — the attention inner width, 7168 in the shipped model.
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Rotated head channels: `2 · 3 · rope_freq_dim`. The remaining
    /// `attention_head_dim - rotary_dim` pass through unrotated.
    pub fn rotary_dim(&self) -> i32 {
        2 * 3 * self.rope_freq_dim
    }

    /// Columns one AdaLN projection emits: `6 · MODALITY_NUM · hidden_size` (96768 shipped).
    pub fn adaln_out_features(&self) -> i32 {
        MODULATION_PARAMS * MODALITY_NUM * self.hidden_size
    }

    /// Video rows carry `in_channels · prod(patch_size)` features (96 shipped).
    pub fn video_patch_dim(&self) -> i32 {
        self.in_channels * self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
    }

    /// Reject a geometry the block math cannot express.
    pub fn validate(&self) -> Result<()> {
        if self.num_attention_heads <= 0 || self.attention_head_dim <= 0 || self.hidden_size <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: heads/head_dim/hidden_size must be positive, got {}/{}/{}",
                self.num_attention_heads, self.attention_head_dim, self.hidden_size
            )));
        }
        if self.rope_freq_dim <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: rope_freq_dim must be positive, got {}",
                self.rope_freq_dim
            )));
        }
        if self.rotary_dim() > self.attention_head_dim {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: rotary_dim {} exceeds head dim {}; 2·3·rope_freq_dim must fit in \
                 a head",
                self.rotary_dim(),
                self.attention_head_dim
            )));
        }
        if self.ffn_dim <= 0 || self.time_embed_dim <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: ffn_dim {} and time_embed_dim {} must be positive",
                self.ffn_dim, self.time_embed_dim
            )));
        }
        Ok(())
    }

    /// Parse the published `transformer/config.json`.
    pub fn from_diffusers_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Msg(format!("minimax-h3 dit config: {e}")))?;
        let d = Self::default();
        let int = |k: &str, fallback: i32| -> i32 {
            v.get(k).and_then(serde_json::Value::as_i64).map_or(fallback, |x| x as i32)
        };
        let float = |k: &str, fallback: f32| -> f32 {
            v.get(k).and_then(serde_json::Value::as_f64).map_or(fallback, |x| x as f32)
        };
        let patch = v
            .get("patch_size")
            .and_then(serde_json::Value::as_array)
            .and_then(|a| {
                let mut out = [0i32; 3];
                if a.len() != 3 {
                    return None;
                }
                for (slot, item) in out.iter_mut().zip(a) {
                    *slot = item.as_i64()? as i32;
                }
                Some(out)
            })
            .unwrap_or(d.patch_size);

        let cfg = Self {
            num_attention_heads: int("num_attention_heads", d.num_attention_heads),
            attention_head_dim: int("attention_head_dim", d.attention_head_dim),
            hidden_size: int("hidden_size", d.hidden_size),
            num_layers: int("num_layers", d.num_layers),
            num_refiner_layers: int("num_refiner_layers", d.num_refiner_layers),
            ffn_dim: int("ffn_dim", d.ffn_dim),
            in_channels: int("in_channels", d.in_channels),
            audio_in_channels: int("audio_in_channels", d.audio_in_channels),
            patch_size: patch,
            text_dim: int("text_dim", d.text_dim),
            freq_dim: int("freq_dim", d.freq_dim),
            time_embed_hidden_dim: int("time_embed_hidden_dim", d.time_embed_hidden_dim),
            time_embed_dim: int("time_embed_dim", d.time_embed_dim),
            rope_freq_dim: int("rope_freq_dim", d.rope_freq_dim),
            rope_theta: float("rope_theta", d.rope_theta),
            norm_eps: float("norm_eps", d.norm_eps),
            qk_norm_eps: float("qk_norm_eps", d.qk_norm_eps),
            final_norm_eps: float("final_norm_eps", d.final_norm_eps),
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped geometry, including the two facts a `hidden_size / heads` derivation would get
    /// wrong: the attention inner width is WIDER than the residual stream, and the rotary is
    /// partial.
    #[test]
    fn shipped_geometry_is_wider_than_the_residual_stream() {
        let cfg = MiniMaxH3DitConfig::default();
        assert_eq!(cfg.inner_dim(), 7168);
        assert!(cfg.inner_dim() > cfg.hidden_size);
        assert_ne!(cfg.hidden_size / cfg.num_attention_heads, cfg.attention_head_dim);
        assert_eq!(cfg.rotary_dim(), 96);
        assert!(cfg.rotary_dim() < cfg.attention_head_dim);
        assert_eq!(cfg.adaln_out_features(), 96_768);
        assert_eq!(cfg.video_patch_dim(), 96);
        cfg.validate().unwrap();
    }

    #[test]
    fn parses_the_published_config() {
        let text = r#"{
            "_class_name": "MiniMaxH3Transformer3DModel",
            "num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 5376,
            "num_layers": 50, "num_refiner_layers": 2, "ffn_dim": 14336,
            "in_channels": 24, "audio_in_channels": 32, "patch_size": [1, 2, 2],
            "text_dim": 5120, "freq_dim": 256, "time_embed_hidden_dim": 5376,
            "time_embed_dim": 2688, "rope_freq_dim": 16, "rope_theta": 10000.0,
            "norm_eps": 1e-05, "qk_norm_eps": 1e-05, "final_norm_eps": 1e-05
        }"#;
        assert_eq!(
            MiniMaxH3DitConfig::from_diffusers_json(text).unwrap(),
            MiniMaxH3DitConfig::default()
        );
    }

    /// A rotary wider than a head is a geometry the block cannot express, not a silent truncation.
    #[test]
    fn rejects_a_rotary_wider_than_a_head() {
        let mut cfg = MiniMaxH3DitConfig::default();
        cfg.rope_freq_dim = 32;
        assert_eq!(cfg.rotary_dim(), 192);
        let e = cfg.validate().unwrap_err().to_string();
        assert!(e.contains("exceeds head dim"), "unexpected error: {e}");
    }
}
