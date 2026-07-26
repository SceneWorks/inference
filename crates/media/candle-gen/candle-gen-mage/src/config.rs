//! Mage-Flow RL architecture constants.
//!
//! These values mirror the nine fields consumed by the frozen Torch constructor and the
//! code-hardcoded values audited by `mlx-gen-mage`. Published-but-ignored JSON fields are verified,
//! never treated as runtime switches.

use candle_core::{Error, Result};

pub const FAMILY: &str = "mage_flow";
pub const MODEL_ID: &str = "mage_flow";
pub const EDIT_MODEL_ID: &str = "mage_flow_edit";
pub const EDIT_BASE_MODEL_ID: &str = "mage_flow_edit_base";
pub const EDIT_TURBO_MODEL_ID: &str = "mage_flow_edit_turbo";
pub const LATENT_CHANNELS: usize = 128;
pub const VAE_DOWNSAMPLE: usize = 16;
pub const HIDDEN_SIZE: usize = 3072;
pub const CONTEXT_DIM: usize = 2560;
pub const DEPTH: usize = 12;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const AXES_DIM: [usize; 3] = [16, 56, 56];
pub const ROPE_THETA: f64 = 10_000.0;
pub const NORM_EPS: f64 = 1e-6;
pub const STATIC_SHIFT: f64 = 6.0;
pub const SIZE_MULTIPLE: u32 = 16;
pub const MIN_SIZE: u32 = 512;
pub const MAX_SIZE: u32 = 2048;
pub const TXT_MAX_LENGTH: usize = 2048;
pub const DROP_IDX_GEN: usize = 34;
pub const DROP_IDX_EDIT: usize = 64;
pub const VL_COND_LONG_EDGE: u32 = 384;
pub const TE_LAYERS: usize = 36;
pub const TE_HEADS: usize = 32;
pub const TE_KV_HEADS: usize = 8;
pub const TE_HEAD_DIM: usize = 128;
pub const TE_ROPE_THETA: f32 = 5_000_000.0;

pub const PROMPT_TEMPLATE: &str = "<|im_start|>system\nDescribe the image by detailing the \
color, shape, size, texture, quantity, text, spatial relationships of the objects and \
background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

pub const EDIT_PROMPT_TEMPLATE: &str =
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, \
texture, objects, background), then explain how the user's text instruction should alter or \
modify the image. Generate a new image that meets the user's requirements while maintaining \
consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}\
<|im_end|>\n<|im_start|>assistant\n";

pub const EDIT_IMAGE_PLACEHOLDER: &str = "<|vision_start|><|image_pad|><|vision_end|>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MageConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub axes_dim: [usize; 3],
    pub checkpoint: bool,
    pub patch_size: usize,
}

impl Default for MageConfig {
    fn default() -> Self {
        Self {
            in_channels: LATENT_CHANNELS,
            out_channels: LATENT_CHANNELS,
            context_in_dim: CONTEXT_DIM,
            hidden_size: HIDDEN_SIZE,
            num_heads: HEADS,
            depth: DEPTH,
            axes_dim: AXES_DIM,
            checkpoint: false,
            patch_size: 1,
        }
    }
}

impl MageConfig {
    pub fn from_json(text: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_str(text).map_err(|e| Error::Msg(format!("mage config: {e}")))?;
        let n = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| Error::Msg(format!("mage config: missing integer `{k}`")))
        };
        let axes = v
            .get("axes_dim")
            .and_then(|x| x.as_array())
            .ok_or_else(|| Error::Msg("mage config: missing `axes_dim`".into()))?;
        if axes.len() != 3 {
            return Err(Error::Msg(
                "mage config: axes_dim must have 3 entries".into(),
            ));
        }
        let cfg = Self {
            in_channels: n("in_channels")?,
            out_channels: n("out_channels")?,
            context_in_dim: n("context_in_dim")?,
            hidden_size: n("hidden_size")?,
            num_heads: n("num_heads")?,
            depth: n("depth")?,
            axes_dim: [
                axes[0].as_u64().unwrap_or(0) as usize,
                axes[1].as_u64().unwrap_or(0) as usize,
                axes[2].as_u64().unwrap_or(0) as usize,
            ],
            checkpoint: v
                .get("checkpoint")
                .and_then(|x| x.as_bool())
                .ok_or_else(|| Error::Msg("mage config: missing bool `checkpoint`".into()))?,
            patch_size: n("patch_size")?,
        };
        cfg.validate()?;
        // These keys are stripped by Torch before construction. Reject drift instead of silently
        // selecting different math.
        verify(v.get("theta"), ROPE_THETA, "theta")?;
        verify(v.get("static_shift"), STATIC_SHIFT, "static_shift")?;
        for (key, expected) in [
            ("schedule_mode", "z-image"),
            ("rope_type", "msrope"),
            ("time_type", "qwen_proj"),
            ("double_block_type", "double_stream"),
            ("activation_fn", "gelu-approximate"),
        ] {
            if let Some(got) = v.get(key).and_then(|x| x.as_str()) {
                if got != expected {
                    return Err(Error::Msg(format!(
                        "mage config: `{key}` is `{got}`, frozen Torch requires `{expected}`"
                    )));
                }
            }
        }
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let expected = Self::default();
        if self != &expected {
            return Err(Error::Msg(format!(
                "mage config: checkpoint geometry {self:?} differs from frozen RL geometry \
                 {expected:?}"
            )));
        }
        Ok(())
    }
}

fn verify(value: Option<&serde_json::Value>, expected: f64, key: &str) -> Result<()> {
    if let Some(got) = value.and_then(|x| x.as_f64()) {
        if (got - expected).abs() > f64::EPSILON {
            return Err(Error::Msg(format!(
                "mage config: `{key}` is {got}, frozen Torch requires {expected}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_geometry_is_exact() {
        let c = MageConfig::default();
        assert_eq!((c.hidden_size, c.depth, c.num_heads), (3072, 12, 24));
        assert_eq!(c.axes_dim, [16, 56, 56]);
        assert_eq!(c.in_channels, 128);
        c.validate().unwrap();
    }

    #[test]
    fn z_image_swiglu_mutation_is_rejected() {
        let s = r#"{"in_channels":128,"out_channels":128,"context_in_dim":2560,
          "hidden_size":3072,"num_heads":24,"depth":12,"axes_dim":[16,56,56],
          "checkpoint":false,"patch_size":1,"activation_fn":"silu"}"#;
        assert!(MageConfig::from_json(s).is_err());
    }
}
