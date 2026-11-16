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
        // A non-positive or non-finite theta makes every `inv_freq` NaN, and every attention score
        // with it, which would propagate silently rather than fail.
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::Msg(format!(
                "minimax-h3 dit: rope_theta must be finite and positive, got {}",
                self.rope_theta
            )));
        }
        Ok(())
    }

    /// Parse the published `transformer/config.json`.
    ///
    /// **Every key is required.** Silently defaulting an absent or wrongly-typed key would let a
    /// variant checkpoint — a future H3 revision, or `transformer_ref` (sc-17149) — load and run
    /// against *this* model\'s numbers. Nothing downstream shape-checks `rope_theta`, `freq_dim`,
    /// `patch_size` or the three epsilons, so a wrong value in any of them is silent.
    pub fn from_diffusers_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Msg(format!("minimax-h3 dit config: {e}")))?;
        let missing = |k: &str| {
            Error::Msg(format!(
                "minimax-h3 dit config: missing or non-numeric `{k}`"
            ))
        };
        let int = |k: &str| -> Result<i32> {
            v.get(k)
                .and_then(serde_json::Value::as_i64)
                .map(|x| x as i32)
                .ok_or_else(|| missing(k))
        };
        let float = |k: &str| -> Result<f32> {
            v.get(k)
                .and_then(serde_json::Value::as_f64)
                .map(|x| x as f32)
                .ok_or_else(|| missing(k))
        };
        let patch_size = v
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
            .ok_or_else(|| {
                Error::Msg("minimax-h3 dit config: `patch_size` must be a 3-element array".into())
            })?;

        let cfg = Self {
            num_attention_heads: int("num_attention_heads")?,
            attention_head_dim: int("attention_head_dim")?,
            hidden_size: int("hidden_size")?,
            num_layers: int("num_layers")?,
            num_refiner_layers: int("num_refiner_layers")?,
            ffn_dim: int("ffn_dim")?,
            in_channels: int("in_channels")?,
            audio_in_channels: int("audio_in_channels")?,
            patch_size,
            text_dim: int("text_dim")?,
            freq_dim: int("freq_dim")?,
            time_embed_hidden_dim: int("time_embed_hidden_dim")?,
            time_embed_dim: int("time_embed_dim")?,
            rope_freq_dim: int("rope_freq_dim")?,
            rope_theta: float("rope_theta")?,
            norm_eps: float("norm_eps")?,
            qk_norm_eps: float("qk_norm_eps")?,
            final_norm_eps: float("final_norm_eps")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

/// The `_class_name` of MiniMax-H3-**Image**'s `transformer_diffusers/config.json`.
pub const IMAGE_DIT_CLASS_NAME: &str = "MiniMaxH3ImageTransformer2DModel";

/// Geometry of `MiniMaxH3ImageTransformer2DModel` (`transformer_diffusers/config.json`).
///
/// The image DiT is the H3 block stack without its audio stream, so its geometry is H3's
/// [`MiniMaxH3DitConfig`] with `audio_in_channels` 0 (there are no audio projections to size) plus
/// the number of per-step output heads the checkpoint was distilled with. Every shared layer
/// ([`crate::dit::DitBlock`], [`crate::dit::TokenRefiner`], [`crate::dit::MmRope`]) reads `base`.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxH3ImageDitConfig {
    /// The shared block-stack geometry. `audio_in_channels` is always 0.
    pub base: MiniMaxH3DitConfig,
    /// `num_pdd_heads`: one output head per distilled sampling step. Shipped: 6.
    pub num_pdd_heads: i32,
}

impl Default for MiniMaxH3ImageDitConfig {
    fn default() -> Self {
        Self {
            base: MiniMaxH3DitConfig {
                audio_in_channels: 0,
                ..MiniMaxH3DitConfig::default()
            },
            num_pdd_heads: 6,
        }
    }
}

impl MiniMaxH3ImageDitConfig {
    /// Reject a geometry the image forward cannot express.
    pub fn validate(&self) -> Result<()> {
        self.base.validate()?;
        if self.base.audio_in_channels != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3-image dit: the image DiT has no audio stream, got audio_in_channels {}",
                self.base.audio_in_channels
            )));
        }
        if self.num_pdd_heads <= 0 {
            return Err(Error::Msg(format!(
                "minimax-h3-image dit: num_pdd_heads must be positive, got {}",
                self.num_pdd_heads
            )));
        }
        Ok(())
    }

    /// Parse `transformer_diffusers/config.json`.
    ///
    /// Every H3 key except `audio_in_channels` is required, plus `num_pdd_heads`. A config naming
    /// any other `_class_name`, or carrying `audio_in_channels`, is refused: the H3 video config
    /// would otherwise parse here and run the image forward over an audio-bearing checkpoint.
    pub fn from_diffusers_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Msg(format!("minimax-h3-image dit config: {e}")))?;
        if let Some(class) = v.get("_class_name").and_then(serde_json::Value::as_str) {
            if class != IMAGE_DIT_CLASS_NAME {
                return Err(Error::Msg(format!(
                    "minimax-h3-image dit config: _class_name is {class:?}, expected \
                     {IMAGE_DIT_CLASS_NAME:?}"
                )));
            }
        }
        let mut obj = v
            .as_object()
            .cloned()
            .ok_or_else(|| Error::Msg("minimax-h3-image dit config: not a JSON object".into()))?;
        if obj.contains_key("audio_in_channels") {
            return Err(Error::Msg(
                "minimax-h3-image dit config: unexpected audio_in_channels; the image DiT has no \
                 audio stream"
                    .into(),
            ));
        }
        let num_pdd_heads = obj
            .get("num_pdd_heads")
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| {
                Error::Msg(
                    "minimax-h3-image dit config: missing or non-numeric `num_pdd_heads`".into(),
                )
            })? as i32;
        obj.insert("audio_in_channels".into(), 0.into());
        let base =
            MiniMaxH3DitConfig::from_diffusers_json(&serde_json::Value::Object(obj).to_string())?;
        let cfg = Self {
            base,
            num_pdd_heads,
        };
        cfg.validate()?;
        Ok(cfg)
    }
}

#[cfg(test)]
pub(crate) mod image_tests {
    use super::*;

    /// The published `transformer_diffusers/config.json` at `492fd8c4`, verbatim.
    pub(crate) const IMAGE_DIT_JSON: &str = r#"{
        "_class_name": "MiniMaxH3ImageTransformer2DModel", "_diffusers_version": "0.42.0.dev0",
        "num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 5376,
        "num_layers": 50, "num_refiner_layers": 2, "ffn_dim": 14336, "in_channels": 24,
        "patch_size": [1, 2, 2], "text_dim": 5120, "freq_dim": 256,
        "time_embed_hidden_dim": 5376, "time_embed_dim": 2688, "rope_freq_dim": 16,
        "rope_theta": 10000.0, "norm_eps": 1e-05, "qk_norm_eps": 1e-05,
        "final_norm_eps": 1e-05, "num_pdd_heads": 6
    }"#;

    #[test]
    fn parses_the_published_image_config() {
        let cfg = MiniMaxH3ImageDitConfig::from_diffusers_json(IMAGE_DIT_JSON).unwrap();
        assert_eq!(cfg, MiniMaxH3ImageDitConfig::default());
        // AdaLN width 6 · 3 · 5376: H3's three-modality table, with image rows tagged as video.
        assert_eq!(cfg.base.adaln_out_features(), 96_768);
        assert_eq!(cfg.base.video_patch_dim(), 96);
        assert_eq!(cfg.num_pdd_heads, 6);
    }

    #[test]
    fn refuses_the_video_config_and_audio_keys() {
        let video = IMAGE_DIT_JSON.replace(IMAGE_DIT_CLASS_NAME, "MiniMaxH3Transformer3DModel");
        let e = MiniMaxH3ImageDitConfig::from_diffusers_json(&video).unwrap_err();
        assert!(e.to_string().contains("_class_name"), "{e}");
        let audio = IMAGE_DIT_JSON.replace(
            "\"in_channels\": 24,",
            "\"in_channels\": 24, \"audio_in_channels\": 32,",
        );
        let e = MiniMaxH3ImageDitConfig::from_diffusers_json(&audio).unwrap_err();
        assert!(e.to_string().contains("audio_in_channels"), "{e}");
        let no_heads = IMAGE_DIT_JSON.replace(", \"num_pdd_heads\": 6", "");
        let e = MiniMaxH3ImageDitConfig::from_diffusers_json(&no_heads).unwrap_err();
        assert!(e.to_string().contains("num_pdd_heads"), "{e}");
        // The shared keys stay required.
        let no_theta = IMAGE_DIT_JSON.replace("\"rope_theta\": 10000.0,", "");
        assert!(MiniMaxH3ImageDitConfig::from_diffusers_json(&no_theta).is_err());
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
        assert_ne!(
            cfg.hidden_size / cfg.num_attention_heads,
            cfg.attention_head_dim
        );
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

    /// **A missing key is an error, not a silent fall back to this model's numbers.** A variant
    /// checkpoint that omitted `rope_theta` would otherwise load and run at 10000.0 with nothing
    /// downstream able to notice.
    #[test]
    fn a_missing_or_malformed_key_is_rejected() {
        assert!(
            MiniMaxH3DitConfig::from_diffusers_json("{}").is_err(),
            "an empty config must not resolve to the shipped defaults"
        );

        let full = r#"{
            "num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 5376,
            "num_layers": 50, "num_refiner_layers": 2, "ffn_dim": 14336,
            "in_channels": 24, "audio_in_channels": 32, "patch_size": [1, 2, 2],
            "text_dim": 5120, "freq_dim": 256, "time_embed_hidden_dim": 5376,
            "time_embed_dim": 2688, "rope_freq_dim": 16, "rope_theta": 10000.0,
            "norm_eps": 1e-05, "qk_norm_eps": 1e-05, "final_norm_eps": 1e-05
        }"#;
        MiniMaxH3DitConfig::from_diffusers_json(full).unwrap();

        // Drop each key in turn; every one must be required.
        for key in [
            "num_attention_heads",
            "attention_head_dim",
            "hidden_size",
            "num_layers",
            "num_refiner_layers",
            "ffn_dim",
            "in_channels",
            "audio_in_channels",
            "patch_size",
            "text_dim",
            "freq_dim",
            "time_embed_hidden_dim",
            "time_embed_dim",
            "rope_freq_dim",
            "rope_theta",
            "norm_eps",
            "qk_norm_eps",
            "final_norm_eps",
        ] {
            let mut v: serde_json::Value = serde_json::from_str(full).unwrap();
            v.as_object_mut().unwrap().remove(key);
            assert!(
                MiniMaxH3DitConfig::from_diffusers_json(&v.to_string()).is_err(),
                "`{key}` must be required"
            );
        }

        // A 2-element patch is malformed, not a reason to assume [1, 2, 2].
        let mut v: serde_json::Value = serde_json::from_str(full).unwrap();
        v["patch_size"] = serde_json::json!([2, 2]);
        assert!(MiniMaxH3DitConfig::from_diffusers_json(&v.to_string()).is_err());
    }

    /// A non-positive rotary base yields NaN frequencies, which would propagate through every
    /// attention score instead of failing.
    #[test]
    fn rejects_a_non_positive_rope_theta() {
        for theta in [0.0f32, -1.0, f32::NAN] {
            let cfg = MiniMaxH3DitConfig {
                rope_theta: theta,
                ..MiniMaxH3DitConfig::default()
            };
            let e = cfg.validate().unwrap_err().to_string();
            assert!(e.contains("rope_theta"), "theta {theta}: {e}");
        }
    }

    /// A rotary wider than a head is a geometry the block cannot express, not a silent truncation.
    #[test]
    fn rejects_a_rotary_wider_than_a_head() {
        let cfg = MiniMaxH3DitConfig {
            rope_freq_dim: 32,
            ..MiniMaxH3DitConfig::default()
        };
        assert_eq!(cfg.rotary_dim(), 192);
        let e = cfg.validate().unwrap_err().to_string();
        assert!(e.contains("exceeds head dim"), "unexpected error: {e}");
    }
}
