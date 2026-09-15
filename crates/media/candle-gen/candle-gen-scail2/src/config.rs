//! SCAIL-2 model configuration: the Wan2.1-14B-I2V dimensions plus the SCAIL-2-specific conditioning
//! knobs the base Wan DiT does not carry (the 28-channel mask stem, the i2v binary-mask channels, and
//! the per-source RoPE shifts).
//!
//! Unlike `candle-gen-wan`'s [`candle_gen_wan::config::TransformerConfig`] (which models the Wan2.2
//! TI2V-5B / A14B presets with channel-concat I2V), SCAIL-2 is a distinct DiT — packed-token
//! conditioning + CLIP image cross-attn — so its config is standalone here. The shared z16 VAE / UMT5
//! components are still configured through candle-gen-wan ([`candle_gen_wan::config::Vae16Config`],
//! [`candle_gen_wan::config::TextEncoderConfig`]).

use std::path::Path;

use candle_gen::gen_core::Quant;
use candle_gen::quant::MLX_GROUP_SIZE;
use candle_gen::{CandleError, Result as CResult};
use serde_json::Value;

/// SCAIL-2 conditioning + Wan2.1-14B-I2V dimensions (zai-org/SCAIL-2 `wan/modules/model_scail2.py`,
/// `configs/config-14b.json`).
#[derive(Clone, Debug)]
pub struct Scail2Config {
    /// Hidden dim (5120).
    pub dim: usize,
    /// FFN inner dim (13824).
    pub ffn_dim: usize,
    /// Attention heads (40).
    pub num_heads: usize,
    /// Transformer blocks (40).
    pub num_layers: usize,
    /// Patch-embed input channels = `vae_z_dim` (16) + `i2v_mask_dim` (4) = 20.
    pub in_dim: usize,
    /// Velocity-prediction output channels (16).
    pub out_dim: usize,
    /// Channel count of the color-coded semantic-mask latent fed to `patch_embedding_mask` (28 =
    /// 7 color classes × temporal-pack 4; see [`crate::preprocess::extract_and_compress_mask_to_latent`]).
    pub mask_dim: usize,
    /// Binary i2v-mask channels concatenated onto each latent before patch-embed (4); `in_dim` (20) =
    /// `vae_z_dim` (16) + 4.
    pub i2v_mask_dim: usize,
    /// Sinusoidal timestep-embedding dim (256).
    pub freq_dim: usize,
    /// UMT5 context length the DiT cross-attends over (512), zero-padded.
    pub text_len: usize,
    /// UMT5 text-embedding width (4096).
    pub text_dim: usize,
    /// `WanLayerNorm` / qk-RMSNorm epsilon (1e-6).
    pub eps: f64,
    /// `(p_t, p_h, p_w)` patch ((1, 2, 2)).
    pub patch: (usize, usize, usize),
    /// z16 VAE latent channels (16).
    pub vae_z_dim: usize,
    /// RoPE H-shift applied to the reference chunk in REPLACEMENT mode (`replace_flag = true`); 0 in
    /// animation mode.
    pub replace_h_shift: usize,
    /// RoPE W-shift applied to the spatially-downsampled pose chunk (120).
    pub pose_w_shift: usize,
    /// The hosted MLX-affine DiT tier, when `config.json` declares one.  Only the block
    /// attention/FFN projections are packed; the surrounding SCAIL2 tensors remain dense.
    pub packed_quant: Option<Quant>,
}

impl Default for Scail2Config {
    fn default() -> Self {
        Self::scail2_14b()
    }
}

impl Scail2Config {
    /// The shipped SCAIL-2 14B config (zai-org/SCAIL-2, `configs/config-14b.json`).
    pub fn scail2_14b() -> Self {
        Self {
            dim: 5120,
            ffn_dim: 13824,
            num_heads: 40,
            num_layers: 40,
            in_dim: 20,
            out_dim: 16,
            mask_dim: 28,
            i2v_mask_dim: 4,
            freq_dim: 256,
            text_len: 512,
            text_dim: 4096,
            eps: 1e-6,
            patch: (1, 2, 2),
            vae_z_dim: 16,
            replace_h_shift: 120,
            pose_w_shift: 120,
            packed_quant: None,
        }
    }

    /// Attention head dimension (`dim / num_heads` = 128).
    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    /// Load from a snapshot dir's `config.json` (the upstream `config-14b.json` layout: `in_dim`,
    /// `mask_dim`, `dim`, `ffn_dim`, `num_heads`, `num_layers`, `out_dim`). Any field absent from the
    /// JSON keeps the shipped 14B default.
    pub fn from_model_dir(root: &Path) -> CResult<Self> {
        let mut cfg = Self::scail2_14b();
        let path = root.join("config.json");
        if path.exists() {
            let text = std::fs::read_to_string(&path)
                .map_err(|e| CandleError::Msg(format!("scail2: read config.json: {e}")))?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| CandleError::Msg(format!("scail2: parse config.json: {e}")))?;
            set_usize(&v, "in_dim", &mut cfg.in_dim);
            set_usize(&v, "out_dim", &mut cfg.out_dim);
            set_usize(&v, "dim", &mut cfg.dim);
            set_usize(&v, "ffn_dim", &mut cfg.ffn_dim);
            set_usize(&v, "num_heads", &mut cfg.num_heads);
            set_usize(&v, "num_layers", &mut cfg.num_layers);
            set_usize(&v, "mask_dim", &mut cfg.mask_dim);
            cfg.packed_quant = packed_quantization(&v)?;
        }
        Ok(cfg)
    }
}

/// Parse the SCAIL2 hosted-tier marker.  This is intentionally stricter than generic packed
/// detection: this provider consumes the published group-64 affine Q4/Q8 layout only.  A malformed
/// marker must never let U32 code words fall through the dense loader.
fn packed_quantization(v: &Value) -> CResult<Option<Quant>> {
    let Some(q) = v.get("quantization") else {
        return Ok(None);
    };
    let Some(q) = q.as_object() else {
        return Err(CandleError::Msg(
            "scail2: config.json quantization must be an object".into(),
        ));
    };
    let bits = q.get("bits").and_then(Value::as_u64).ok_or_else(|| {
        CandleError::Msg("scail2: packed quantization must declare bits 4 or 8".into())
    })?;
    let group_size = q
        .get("group_size")
        .and_then(Value::as_u64)
        .unwrap_or(MLX_GROUP_SIZE as u64);
    if group_size != MLX_GROUP_SIZE as u64 {
        return Err(CandleError::Msg(format!(
            "scail2: packed quantization group_size must be {MLX_GROUP_SIZE}, got {group_size}"
        )));
    }
    match bits {
        4 => Ok(Some(Quant::Q4)),
        8 => Ok(Some(Quant::Q8)),
        _ => Err(CandleError::Msg(format!(
            "scail2: packed quantization bits must be 4 or 8, got {bits}"
        ))),
    }
}

fn set_usize(v: &Value, key: &str, slot: &mut usize) {
    if let Some(n) = v.get(key).and_then(Value::as_u64) {
        *slot = n as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_14b_dims() {
        let c = Scail2Config::scail2_14b();
        assert_eq!(c.dim, 5120);
        assert_eq!(c.num_layers, 40);
        assert_eq!(c.num_heads, 40);
        assert_eq!(c.in_dim, 20);
        assert_eq!(c.out_dim, 16);
        assert_eq!(c.mask_dim, 28);
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.vae_z_dim, 16);
        assert_eq!(c.packed_quant, None);
    }

    #[test]
    fn hosted_affine_marker_accepts_q4_q8_and_rejects_other_formats() {
        for (bits, want) in [(4, Quant::Q4), (8, Quant::Q8)] {
            assert_eq!(
                packed_quantization(&serde_json::json!({
                    "quantization": { "bits": bits, "group_size": 64 }
                }))
                .unwrap(),
                Some(want)
            );
        }
        for marker in [
            serde_json::json!({ "quantization": { "bits": 3 } }),
            serde_json::json!({ "quantization": { "bits": 4, "group_size": 32 } }),
            serde_json::json!({ "quantization": true }),
        ] {
            assert!(packed_quantization(&marker).is_err(), "{marker}");
        }
    }
}
