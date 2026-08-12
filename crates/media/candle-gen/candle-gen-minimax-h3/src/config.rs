//! MiniMax-H3 video-VAE configuration (`vae/config.json`, `AutoencoderKLMiniMaxH3`).
//!
//! Two published configs describe the same model and **neither alone is sufficient**:
//!
//! - the diffusers `vae/config.json` at the snapshot root carries the geometry, the temporal
//!   knobs and the per-channel `latents_mean`/`latents_std`; but it does **not** carry
//!   `qk_norm_type`, `norm_type`, `ffn_activation_fn` or `ffn_use_gated`;
//! - `FL2VA/video_vae/source/config.json` (`AutoencoderKLLegacy`) carries those four under
//!   `vit_decoder_kwargs`, and is the config the Apache-2.0 reference implementation actually
//!   instantiates from.
//!
//! Reconstructing the decoder from the diffusers config alone would silently pick the reference
//! module's *defaults* — `norm_type="layer_norm"`, `qk_norm_type=None`,
//! `ffn_activation_fn="gelu"`, `ffn_use_gated=false` — every one of which is wrong for this
//! checkpoint. Those four are therefore pinned as constants in [`crate::blocks`] and asserted
//! against the shipped tensor shapes by [`crate::vae`]'s loader (a LayerNorm would need a
//! `norm1.bias` the checkpoint does not have; a non-gated FFN would need a `[inner, dim]` rather
//! than `[2·inner, dim]` `w1`).
//!
//! **`qk_norm_type` leaves no tensor behind at all** (`qk_norm_affine = false`), so a port that
//! misses it loads every published tensor and passes an exhaustive key-mapping proof while being
//! silently wrong.

use candle_gen::{CandleError, Result};

/// Latent channel count (`latent_channels` / `z_channels`).
pub const LATENT_CHANNELS: usize = 24;
/// Spatial compression factor — ∏`spatial_downsample_factors` = `[2,2,2,2,1,1]` → 16×.
pub const VAE_RATIO: usize = 16;
/// Temporal compression factor — ∏`temporal_downsample_factors` = `[1,2,2,1,1,1]` → 4×.
pub const VAE_RATIO_T: usize = 4;
/// Frames per encoded clip. Drives the whole temporal chunk plan ([`crate::chunking`]).
pub const CLIP_LENGTH: i32 = 17;
/// Latent tokens dropped from the tail of an encode — and, at decode, the reason each clip is
/// decoded twice (`split_count = 2`) with an overlap-blended seam. **Not** a training-only knob.
pub const TOKEN_DROP: i32 = 3;
/// Transformer decoder depth.
pub const DECODER_NUM_LAYERS: usize = 36;
/// Transformer decoder attention heads.
pub const DECODER_NUM_HEADS: usize = 32;
/// Transformer decoder per-head dim; `heads · dim_head` = 2048 is the model dim.
pub const DECODER_HEAD_DIM: usize = 64;
/// Learned register tokens appended after the patch tokens (before the zero CLS token).
pub const DECODER_NUM_REGISTER_TOKENS: usize = 4;
/// FFN inner-dim multiplier; the FFN is SwiGLU-gated, so `w1` is `[2·mult·dim, dim]`.
pub const DECODER_FFN_MULT: usize = 4;
/// RoPE base.
pub const DECODER_ROPE_THETA: f32 = 100.0;
/// **Partial** rotary: only the first `int(head_dim · ratio)` = 48 of 64 head dims are rotated;
/// the remaining 16 pass through untouched.
pub const DECODER_ROPE_DIM_RATIO: f32 = 0.75;
/// `eps` for the decoder's RMSNorms and its output LayerNorm (`decoder_norm_eps`).
pub const DECODER_NORM_EPS: f64 = 1e-5;

/// Per-channel latent de-normalization mean (`latents_mean`, 24 entries).
pub const LATENTS_MEAN: [f32; LATENT_CHANNELS] = [
    0.858_090_34,
    -0.960_659_15,
    1.066_164,
    -0.509_032_55,
    -0.272_758_2,
    -1.367_541_4,
    -0.255_325_5,
    -0.269_075_54,
    -0.537_684_1,
    -0.046_409_73,
    0.665_737_03,
    0.196_901_28,
    -0.546_060_8,
    -0.403_534_2,
    -0.236_830_25,
    0.259_284_53,
    -0.301_339_45,
    0.211_341_99,
    -1.120_684_9,
    0.358_193_34,
    -0.042_251_438,
    0.260_483,
    0.228_640_93,
    0.705_603_2,
];

/// Per-channel latent de-normalization standard deviation (`latents_std`, 24 entries).
pub const LATENTS_STD: [f32; LATENT_CHANNELS] = [
    1.222_377_4,
    1.276_726_4,
    1.683_177_5,
    1.754_945_5,
    1.563_621_6,
    2.194_143_5,
    0.965_313_8,
    1.056_988_6,
    0.841_948_9,
    0.772_995_3,
    1.895_593_8,
    0.946_841_84,
    0.799_680_95,
    0.449_889,
    0.719_74,
    0.693_629_3,
    2.961_095,
    2.769_42,
    3.049_618_5,
    2.108_805_4,
    3.276_226_3,
    3.162_735_7,
    2.281_681_3,
    2.612_784_4,
];

/// Video-VAE config. Dimension-parametric so the same code runs the shipped 36-layer / 2048-dim
/// decoder and the tiny committed parity fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxH3VaeConfig {
    /// Latent channels fed to the decoder's `proj_in`.
    pub latent_channels: usize,
    /// Decoded pixel channels.
    pub out_channels: usize,
    /// Transformer depth.
    pub num_layers: usize,
    /// Attention heads.
    pub num_heads: usize,
    /// Per-head dim.
    pub head_dim: usize,
    /// Learned register tokens.
    pub num_register_tokens: usize,
    /// FFN inner-dim multiplier (gated, so `w1` emits `2 · mult · dim`).
    pub ffn_mult: usize,
    /// RoPE base.
    pub rope_theta: f32,
    /// Fraction of each head's dims that are rotated.
    pub rope_dim_ratio: f32,
    /// Norm epsilon.
    pub norm_eps: f64,
    /// Spatial patch size — equals the spatial compression factor.
    pub patch_size: usize,
    /// Temporal patch size — equals the temporal compression factor.
    pub patch_size_t: usize,
    /// Frames per clip.
    pub clip_length: i32,
    /// Latent tokens dropped at the encode tail.
    pub token_drop: i32,
    /// Per-channel de-normalization mean.
    pub latents_mean: Vec<f32>,
    /// Per-channel de-normalization standard deviation.
    pub latents_std: Vec<f32>,
}

impl Default for MiniMaxH3VaeConfig {
    /// The shipped `MiniMaxAI/MiniMax-H3` video VAE.
    fn default() -> Self {
        Self {
            latent_channels: LATENT_CHANNELS,
            out_channels: 3,
            num_layers: DECODER_NUM_LAYERS,
            num_heads: DECODER_NUM_HEADS,
            head_dim: DECODER_HEAD_DIM,
            num_register_tokens: DECODER_NUM_REGISTER_TOKENS,
            ffn_mult: DECODER_FFN_MULT,
            rope_theta: DECODER_ROPE_THETA,
            rope_dim_ratio: DECODER_ROPE_DIM_RATIO,
            norm_eps: DECODER_NORM_EPS,
            patch_size: VAE_RATIO,
            patch_size_t: VAE_RATIO_T,
            clip_length: CLIP_LENGTH,
            token_drop: TOKEN_DROP,
            latents_mean: LATENTS_MEAN.to_vec(),
            latents_std: LATENTS_STD.to_vec(),
        }
    }
}

impl MiniMaxH3VaeConfig {
    /// Model dim — `num_heads · head_dim`.
    pub fn dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Rotated head dims — `int(head_dim · rope_dim_ratio)`, matching the reference's `int()`
    /// truncation. Must be divisible by `2 · 3` (the 3 position axes, half-split rotary).
    pub fn rope_apply_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope_dim_ratio) as usize
    }

    /// Suffix tokens appended after the patch tokens: the register tokens plus one zero CLS token.
    pub fn num_suffix_tokens(&self) -> usize {
        1 + self.num_register_tokens
    }

    /// Output features of `proj_out` — `out_channels · patch_size_t · patch_size²`.
    pub fn patch_dim(&self) -> usize {
        self.out_channels * self.patch_size_t * self.patch_size * self.patch_size
    }

    /// Parse a diffusers `vae/config.json` body. The four decoder knobs the diffusers config omits
    /// are taken from the module constants (see the module docs).
    pub fn from_diffusers_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| CandleError::Msg(format!("minimax-h3 vae/config.json: {e}")))?;
        let num = |key: &str| -> Result<f64> {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    CandleError::Msg(format!("minimax-h3 vae/config.json: missing {key}"))
                })
        };
        let usize_of = |key: &str| -> Result<usize> {
            let raw = num(key)?;
            if raw < 0.0 {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 vae/config.json: {key} is negative ({raw})"
                )));
            }
            Ok(raw as usize)
        };
        let vec_f32 = |key: &str| -> Result<Vec<f32>> {
            let arr = v
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    CandleError::Msg(format!("minimax-h3 vae/config.json: missing {key}"))
                })?;
            arr.iter()
                .map(|e| {
                    e.as_f64().map(|f| f as f32).ok_or_else(|| {
                        CandleError::Msg(format!(
                            "minimax-h3 vae/config.json: non-numeric in {key}"
                        ))
                    })
                })
                .collect()
        };
        let prod = |key: &str| -> Result<usize> {
            let arr = v
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    CandleError::Msg(format!("minimax-h3 vae/config.json: missing {key}"))
                })?;
            let mut acc: usize = 1;
            for e in arr {
                let f = e.as_u64().ok_or_else(|| {
                    CandleError::Msg(format!(
                        "minimax-h3 vae/config.json: non-positive-integer in {key}"
                    ))
                })?;
                acc *= f as usize;
            }
            Ok(acc)
        };

        let cfg = Self {
            latent_channels: usize_of("latent_channels")?,
            out_channels: usize_of("out_channels")?,
            num_layers: usize_of("decoder_num_layers")?,
            num_heads: usize_of("decoder_num_attention_heads")?,
            head_dim: usize_of("decoder_attention_head_dim")?,
            num_register_tokens: usize_of("decoder_num_register_tokens")?,
            ffn_mult: usize_of("decoder_ffn_mult")?,
            rope_theta: num("decoder_rope_theta")? as f32,
            rope_dim_ratio: num("decoder_rope_dim_ratio")? as f32,
            norm_eps: num("decoder_norm_eps")?,
            patch_size: prod("spatial_downsample_factors")?,
            patch_size_t: prod("temporal_downsample_factors")?,
            clip_length: num("clip_length")? as i32,
            token_drop: num("token_drop")? as i32,
            latents_mean: vec_f32("latents_mean")?,
            latents_std: vec_f32("latents_std")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject configs the decoder cannot express.
    pub fn validate(&self) -> Result<()> {
        if self.latents_mean.len() != self.latent_channels
            || self.latents_std.len() != self.latent_channels
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: latents_mean/std must have {} entries, got {}/{}",
                self.latent_channels,
                self.latents_mean.len(),
                self.latents_std.len()
            )));
        }
        let rope = self.rope_apply_dim();
        if rope == 0 || !rope.is_multiple_of(6) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: rope_apply_dim {rope} must be a positive multiple of 6 \
                 (3 position axes, half-split rotary)"
            )));
        }
        if rope > self.head_dim {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: rope_apply_dim {rope} exceeds head_dim {}",
                self.head_dim
            )));
        }
        if self.clip_length <= 0 || self.patch_size_t == 0 || self.patch_size == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 vae: clip_length and patch sizes must be positive".into(),
            ));
        }
        if self.token_drop < 0 {
            return Err(CandleError::Msg(
                "minimax-h3 vae: token_drop must be >= 0".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_defaults_match_the_published_config() {
        let cfg = MiniMaxH3VaeConfig::default();
        cfg.validate().unwrap();
        assert_eq!(cfg.dim(), 2048);
        assert_eq!(cfg.num_layers, 36);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.patch_size_t, 4);
        assert_eq!(cfg.clip_length, 17);
        assert_eq!(cfg.token_drop, 3);
        // proj_out emits out_channels · patch_size_t · patch_size² = 3·4·16·16.
        assert_eq!(cfg.patch_dim(), 3072);
    }

    /// AC: partial rotary is 0.75, i.e. 48 of 64 head dims rotate and 16 pass through. A ratio of
    /// 1.0 (the reference module's default) would rotate all 64 and is a different model.
    #[test]
    fn rope_is_partial_at_three_quarters() {
        let cfg = MiniMaxH3VaeConfig::default();
        assert_eq!(cfg.rope_dim_ratio, 0.75);
        assert_eq!(cfg.rope_apply_dim(), 48);
        assert!(
            cfg.rope_apply_dim() < cfg.head_dim,
            "rotary must be PARTIAL"
        );
        assert_eq!(cfg.head_dim - cfg.rope_apply_dim(), 16, "pass-through dims");
        // 48 / (2·3) = 8 frequencies per position axis.
        assert_eq!(cfg.rope_apply_dim() % 6, 0);
    }

    /// AC: register tokens are 4, and the token suffix is 4 registers + 1 zero CLS token.
    #[test]
    fn register_token_count_is_pinned() {
        let cfg = MiniMaxH3VaeConfig::default();
        assert_eq!(cfg.num_register_tokens, 4);
        assert_eq!(cfg.num_suffix_tokens(), 5);
    }

    /// AC: the 24-entry de-normalization statistics are present and are NOT unit — a port that
    /// silently assumed `mean=0, std=1` would decode a differently-scaled latent.
    #[test]
    fn latent_statistics_have_24_non_unit_entries() {
        let cfg = MiniMaxH3VaeConfig::default();
        assert_eq!(cfg.latents_mean.len(), 24);
        assert_eq!(cfg.latents_std.len(), 24);
        assert!(cfg.latents_mean.iter().any(|m| m.abs() > 0.5));
        assert!(cfg.latents_std.iter().any(|s| (s - 1.0).abs() > 0.5));
        assert!(cfg.latents_std.iter().all(|s| *s > 0.0));
        // Spot-check both ends against `vae/config.json` verbatim.
        assert!((cfg.latents_mean[0] - 0.858_090_34).abs() < 1e-6);
        assert!((cfg.latents_mean[23] - 0.705_603_2).abs() < 1e-6);
        assert!((cfg.latents_std[0] - 1.222_377_4).abs() < 1e-6);
        assert!((cfg.latents_std[23] - 2.612_784_4).abs() < 1e-6);
    }

    /// The same `vae/config.json` body the MLX lane parses must produce the same config here — the
    /// two backends must not disagree about the model before a single tensor is read.
    #[test]
    fn diffusers_json_round_trips_to_the_defaults() {
        let json = r#"{
            "latent_channels": 24, "out_channels": 3,
            "spatial_downsample_factors": [2,2,2,2,1,1],
            "temporal_downsample_factors": [1,2,2,1,1,1],
            "decoder_num_layers": 36, "decoder_num_attention_heads": 32,
            "decoder_attention_head_dim": 64, "decoder_num_register_tokens": 4,
            "decoder_ffn_mult": 4, "decoder_rope_theta": 100.0,
            "decoder_rope_dim_ratio": 0.75, "decoder_norm_eps": 1e-05,
            "clip_length": 17, "token_drop": 3,
            "latents_mean": [0.858090341091156,-0.9606591463088989,1.0661640167236328,-0.5090325474739075,-0.2727581858634949,-1.3675414323806763,-0.2553254961967468,-0.26907554268836975,-0.5376840829849243,-0.0464097298681736,0.6657370328903198,0.19690127670764923,-0.5460608005523682,-0.4035342037677765,-0.23683024942874908,0.25928452610969543,-0.30133944749832153,0.211341992020607,-1.1206848621368408,0.3581933379173279,-0.04225143790245056,0.2604829967021942,0.22864092886447906,0.7056031823158264],
            "latents_std": [1.2223774194717407,1.2767263650894165,1.6831774711608887,1.7549455165863037,1.5636216402053833,2.194143533706665,0.9653137922286987,1.0569885969161987,0.841948926448822,0.7729952931404114,1.8955937623977661,0.946841835975647,0.7996809482574463,0.44988900423049927,0.7197399735450745,0.6936293244361877,2.961095094680786,2.7694199085235596,3.0496184825897217,2.1088054180145264,3.276226282119751,3.1627357006073,2.2816812992095947,2.6127843856811523]
        }"#;
        let parsed = MiniMaxH3VaeConfig::from_diffusers_json(json).unwrap();
        assert_eq!(parsed, MiniMaxH3VaeConfig::default());
    }

    #[test]
    fn validate_rejects_a_rope_ratio_the_rotary_cannot_express() {
        // 64 · 0.1 = 6.4 -> 6, divisible by 6 and legal.
        let legal = MiniMaxH3VaeConfig {
            rope_dim_ratio: 0.1,
            ..Default::default()
        };
        assert!(legal.validate().is_ok());
        // 64 · 0.5 = 32, not a multiple of 6 -> cannot split across 3 axes.
        let illegal = MiniMaxH3VaeConfig {
            rope_dim_ratio: 0.5,
            ..Default::default()
        };
        assert!(illegal.validate().is_err());
    }

    #[test]
    fn validate_rejects_mismatched_latent_statistics() {
        let mut cfg = MiniMaxH3VaeConfig::default();
        cfg.latents_mean.pop();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn a_malformed_config_is_an_error_not_a_default() {
        assert!(MiniMaxH3VaeConfig::from_diffusers_json("{").is_err());
        assert!(MiniMaxH3VaeConfig::from_diffusers_json("{}").is_err());
    }
}
