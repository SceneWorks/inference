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
//! checkpoint. Those four are therefore pinned as constants here and asserted against the shipped
//! tensor shapes by [`crate::vae`]'s loader (a LayerNorm would need a `norm1.bias` the checkpoint
//! does not have; a non-gated FFN would need a `[inner, dim]` rather than `[2·inner, dim]` `w1`).

use mlx_gen::{Error, Result};

/// Latent channel count (`latent_channels` / `z_channels`).
pub const LATENT_CHANNELS: usize = 24;
/// Spatial compression factor — ∏`spatial_downsample_factors` = `[2,2,2,2,1,1]` → 16×.
pub const VAE_RATIO: i32 = 16;
/// Temporal compression factor — ∏`temporal_downsample_factors` = `[1,2,2,1,1,1]` → 4×.
pub const VAE_RATIO_T: i32 = 4;
/// Frames per encoded clip. Drives the whole temporal chunk plan ([`crate::chunking`]).
pub const CLIP_LENGTH: i32 = 17;
/// Latent tokens dropped from the tail of an encode — and, at decode, the reason each clip is
/// decoded twice (`split_count = 2`) with an overlap-blended seam. **Not** a training-only knob.
pub const TOKEN_DROP: i32 = 3;
/// Transformer decoder depth.
pub const DECODER_NUM_LAYERS: usize = 36;
/// Transformer decoder attention heads.
pub const DECODER_NUM_HEADS: i32 = 32;
/// Transformer decoder per-head dim; `heads · dim_head` = 2048 is the model dim.
pub const DECODER_HEAD_DIM: i32 = 64;
/// Learned register tokens appended after the patch tokens (before the zero CLS token).
pub const DECODER_NUM_REGISTER_TOKENS: i32 = 4;
/// FFN inner-dim multiplier; the FFN is SwiGLU-gated, so `w1` is `[2·mult·dim, dim]`.
pub const DECODER_FFN_MULT: i32 = 4;
/// RoPE base.
pub const DECODER_ROPE_THETA: f32 = 100.0;
/// **Partial** rotary: only the first `round(head_dim · ratio)` = 48 of 64 head dims are rotated;
/// the remaining 16 pass through untouched.
pub const DECODER_ROPE_DIM_RATIO: f32 = 0.75;
/// `eps` for the decoder's RMSNorms and its output LayerNorm (`decoder_norm_eps`).
pub const DECODER_NORM_EPS: f32 = 1e-5;

/// Encoded pixel channels.
pub const ENCODER_IN_CHANNELS: i32 = 3;
/// Channels each encoder down-block emits (`block_out_channels`).
pub const ENCODER_BLOCK_OUT_CHANNELS: [i32; 6] = [128, 256, 256, 512, 512, 1024];
/// Residual blocks per encoder down-block (`layers_per_block`).
pub const ENCODER_LAYERS_PER_BLOCK: usize = 2;
/// Per-level spatial stride. Their product is [`VAE_RATIO`]; levels 4 and 5 have **no**
/// downsampler, which is why the product is not `2^len`.
pub const ENCODER_SPATIAL_DOWNSAMPLE_FACTORS: [i32; 6] = [2, 2, 2, 2, 1, 1];
/// Per-level temporal stride. Their product is [`VAE_RATIO_T`].
pub const ENCODER_TEMPORAL_DOWNSAMPLE_FACTORS: [i32; 6] = [1, 2, 2, 1, 1, 1];
/// Groups in every encoder `GroupNorm`.
pub const ENCODER_NORM_NUM_GROUPS: i32 = 32;
/// The encoder's norm epsilon — **1e-6**, not the decoder's 1e-5.
pub const ENCODER_NORM_EPS: f32 = 1e-6;

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
    pub latent_channels: i32,
    /// Decoded pixel channels.
    pub out_channels: i32,
    /// Transformer depth.
    pub num_layers: usize,
    /// Attention heads.
    pub num_heads: i32,
    /// Per-head dim.
    pub head_dim: i32,
    /// Learned register tokens.
    pub num_register_tokens: i32,
    /// FFN inner-dim multiplier (gated, so `w1` emits `2 · mult · dim`).
    pub ffn_mult: i32,
    /// RoPE base.
    pub rope_theta: f32,
    /// Fraction of each head's dims that are rotated.
    pub rope_dim_ratio: f32,
    /// Norm epsilon.
    pub norm_eps: f32,
    /// Spatial patch size — equals the spatial compression factor.
    pub patch_size: i32,
    /// Temporal patch size — equals the temporal compression factor.
    pub patch_size_t: i32,
    /// Frames per clip.
    pub clip_length: i32,
    /// Latent tokens dropped at the encode tail.
    pub token_drop: i32,
    /// Per-channel de-normalization mean.
    pub latents_mean: Vec<f32>,
    /// Per-channel de-normalization standard deviation.
    pub latents_std: Vec<f32>,

    // --- the CNN encoder half (sc-17148) -----------------------------------------------------
    // The decoder is a 36-layer ViT and needs none of these; the encoder is a 3-D causal CNN and
    // needs all of them. `patch_size` / `patch_size_t` above are the *products* of the two factor
    // lists, which is all the decoder ever wanted — the per-level lists are kept separately
    // because the encoder's shape at level `i` depends on the individual factors, not the product.
    /// Encoded pixel channels — 3.
    pub in_channels: i32,
    /// Channels each down-block emits. `len()` is the number of levels.
    pub block_out_channels: Vec<i32>,
    /// Residual blocks per down-block.
    pub layers_per_block: usize,
    /// Per-level spatial stride. A level with factor 1 has **no** downsampler at all.
    pub spatial_downsample_factors: Vec<i32>,
    /// Per-level temporal stride, applied by the same downsampler conv.
    pub temporal_downsample_factors: Vec<i32>,
    /// Groups in every encoder `GroupNorm` — 32.
    pub norm_num_groups: i32,
    /// The **encoder's** norm epsilon, 1e-6. Distinct from [`Self::norm_eps`], which is the
    /// decoder's 1e-5; `vae/config.json` ships both (`norm_eps` and `decoder_norm_eps`) and using
    /// one for the other is a silent numeric change.
    pub encoder_norm_eps: f32,
}

impl Default for MiniMaxH3VaeConfig {
    /// The shipped `MiniMaxAI/MiniMax-H3` video VAE.
    fn default() -> Self {
        Self {
            latent_channels: LATENT_CHANNELS as i32,
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
            in_channels: ENCODER_IN_CHANNELS,
            block_out_channels: ENCODER_BLOCK_OUT_CHANNELS.to_vec(),
            layers_per_block: ENCODER_LAYERS_PER_BLOCK,
            spatial_downsample_factors: ENCODER_SPATIAL_DOWNSAMPLE_FACTORS.to_vec(),
            temporal_downsample_factors: ENCODER_TEMPORAL_DOWNSAMPLE_FACTORS.to_vec(),
            norm_num_groups: ENCODER_NORM_NUM_GROUPS,
            encoder_norm_eps: ENCODER_NORM_EPS,
        }
    }
}

impl MiniMaxH3VaeConfig {
    /// Model dim — `num_heads · head_dim`.
    pub fn dim(&self) -> i32 {
        self.num_heads * self.head_dim
    }

    /// Rotated head dims — `int(head_dim · rope_dim_ratio)`, matching the reference's
    /// `int()` truncation. Must be divisible by `2 · 3` (the 3 position axes, half-split rotary).
    pub fn rope_apply_dim(&self) -> i32 {
        (self.head_dim as f32 * self.rope_dim_ratio) as i32
    }

    /// Suffix tokens appended after the patch tokens: the register tokens plus one zero CLS token.
    pub fn num_suffix_tokens(&self) -> i32 {
        1 + self.num_register_tokens
    }

    /// Output features of `proj_out` — `out_channels · patch_size_t · patch_size²`.
    pub fn patch_dim(&self) -> i32 {
        self.out_channels * self.patch_size_t * self.patch_size * self.patch_size
    }

    /// Parse a diffusers `vae/config.json` body. The four decoder knobs the diffusers config omits
    /// are taken from the module constants (see the module docs).
    pub fn from_diffusers_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Msg(format!("minimax-h3 vae/config.json: {e}")))?;
        let num = |key: &str| -> Result<f64> {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| Error::Msg(format!("minimax-h3 vae/config.json: missing {key}")))
        };
        let vec_f32 = |key: &str| -> Result<Vec<f32>> {
            let arr = v
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| Error::Msg(format!("minimax-h3 vae/config.json: missing {key}")))?;
            arr.iter()
                .map(|e| {
                    e.as_f64().map(|f| f as f32).ok_or_else(|| {
                        Error::Msg(format!("minimax-h3 vae/config.json: non-numeric in {key}"))
                    })
                })
                .collect()
        };
        let vec_i32 = |key: &str| -> Result<Vec<i32>> {
            let arr = v
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| Error::Msg(format!("minimax-h3 vae/config.json: missing {key}")))?;
            arr.iter()
                .map(|e| {
                    e.as_i64().map(|f| f as i32).ok_or_else(|| {
                        Error::Msg(format!("minimax-h3 vae/config.json: non-integer in {key}"))
                    })
                })
                .collect()
        };
        let prod = |key: &str| -> Result<i32> { Ok(vec_i32(key)?.iter().product()) };

        let cfg = Self {
            latent_channels: num("latent_channels")? as i32,
            out_channels: num("out_channels")? as i32,
            num_layers: num("decoder_num_layers")? as usize,
            num_heads: num("decoder_num_attention_heads")? as i32,
            head_dim: num("decoder_attention_head_dim")? as i32,
            num_register_tokens: num("decoder_num_register_tokens")? as i32,
            ffn_mult: num("decoder_ffn_mult")? as i32,
            rope_theta: num("decoder_rope_theta")? as f32,
            rope_dim_ratio: num("decoder_rope_dim_ratio")? as f32,
            norm_eps: num("decoder_norm_eps")? as f32,
            patch_size: prod("spatial_downsample_factors")?,
            patch_size_t: prod("temporal_downsample_factors")?,
            clip_length: num("clip_length")? as i32,
            token_drop: num("token_drop")? as i32,
            latents_mean: vec_f32("latents_mean")?,
            latents_std: vec_f32("latents_std")?,
            in_channels: num("in_channels")? as i32,
            block_out_channels: vec_i32("block_out_channels")?,
            layers_per_block: num("layers_per_block")? as usize,
            spatial_downsample_factors: vec_i32("spatial_downsample_factors")?,
            temporal_downsample_factors: vec_i32("temporal_downsample_factors")?,
            norm_num_groups: num("norm_num_groups")? as i32,
            encoder_norm_eps: num("norm_eps")? as f32,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Reject configs the decoder cannot express.
    pub fn validate(&self) -> Result<()> {
        if self.latents_mean.len() != self.latent_channels as usize
            || self.latents_std.len() != self.latent_channels as usize
        {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: latents_mean/std must have {} entries, got {}/{}",
                self.latent_channels,
                self.latents_mean.len(),
                self.latents_std.len()
            )));
        }
        let rope = self.rope_apply_dim();
        if rope == 0 || rope % 6 != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: rope_apply_dim {rope} must be a positive multiple of 6 \
                 (3 position axes, half-split rotary)"
            )));
        }
        if rope > self.head_dim {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: rope_apply_dim {rope} exceeds head_dim {}",
                self.head_dim
            )));
        }
        if self.clip_length <= 0 || self.patch_size_t <= 0 || self.patch_size <= 0 {
            return Err(Error::Msg(
                "minimax-h3 vae: clip_length and patch sizes must be positive".into(),
            ));
        }
        if self.token_drop < 0 {
            return Err(Error::Msg("minimax-h3 vae: token_drop must be >= 0".into()));
        }
        self.validate_encoder()
    }

    /// Levels in the CNN encoder.
    pub fn num_encoder_levels(&self) -> usize {
        self.block_out_channels.len()
    }

    /// Input channels of encoder level `i` — `block_out_channels[i - 1]`, and
    /// `block_out_channels[0]` at level 0 (`conv_in` already lifts 3 → 128).
    pub fn encoder_level_in_channels(&self, level: usize) -> i32 {
        if level == 0 {
            self.block_out_channels[0]
        } else {
            self.block_out_channels[level - 1]
        }
    }

    /// Whether level `i` carries a `downsamplers.0`.
    ///
    /// The reference's condition is `temporal_factor · spatial_factor > 1` — **the product**, not
    /// the spatial factor alone. Levels 4 and 5 of the shipped config are `1 · 1` and have no
    /// downsampler module at all, so a loader that emits `downsamplers.0.conv.*` names for them
    /// asks for tensors the checkpoint does not have. A hypothetical temporal-only level
    /// (`spatial 1`, `temporal 2`) does carry one, which is why the spatial factor alone is the
    /// wrong predicate.
    pub fn encoder_level_has_downsampler(&self, level: usize) -> bool {
        self.temporal_downsample_factors[level] * self.spatial_downsample_factors[level] > 1
    }

    /// Reject encoder configs the CNN half cannot express.
    ///
    /// The two factor lists' products **must** equal [`Self::patch_size`] / [`Self::patch_size_t`],
    /// which the decoder already depends on: the encoder's compression and the decoder's patch are
    /// the same two numbers, and a config in which they disagree produces latents the decoder
    /// silently reshapes into the wrong geometry.
    pub fn validate_encoder(&self) -> Result<()> {
        let levels = self.num_encoder_levels();
        if levels == 0 {
            return Err(Error::Msg(
                "minimax-h3 vae: the encoder needs at least one level".into(),
            ));
        }
        if self.spatial_downsample_factors.len() != levels
            || self.temporal_downsample_factors.len() != levels
        {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: {levels} block_out_channels but {}/{} spatial/temporal \
                 downsample factors — the three lists are per-level and must agree",
                self.spatial_downsample_factors.len(),
                self.temporal_downsample_factors.len()
            )));
        }
        if self.in_channels <= 0 || self.layers_per_block == 0 {
            return Err(Error::Msg(
                "minimax-h3 vae: in_channels and layers_per_block must be positive".into(),
            ));
        }
        if let Some(bad) = self
            .block_out_channels
            .iter()
            .find(|&&c| c <= 0 || c % self.norm_num_groups != 0)
        {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: encoder channel width {bad} must be a positive multiple of the \
                 {} GroupNorm groups",
                self.norm_num_groups
            )));
        }
        if self
            .spatial_downsample_factors
            .iter()
            .chain(&self.temporal_downsample_factors)
            .any(|&f| f <= 0)
        {
            return Err(Error::Msg(
                "minimax-h3 vae: downsample factors must be positive".into(),
            ));
        }
        let spatial: i32 = self.spatial_downsample_factors.iter().product();
        let temporal: i32 = self.temporal_downsample_factors.iter().product();
        if spatial != self.patch_size || temporal != self.patch_size_t {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: the encoder compresses {spatial}x spatially and {temporal}x \
                 temporally, but the decoder patch is {}x / {}x — encode and decode must be \
                 inverse geometries",
                self.patch_size, self.patch_size_t
            )));
        }
        if self.encoder_norm_eps <= 0.0 {
            return Err(Error::Msg(
                "minimax-h3 vae: encoder_norm_eps must be positive".into(),
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
            "in_channels": 3, "block_out_channels": [128,256,256,512,512,1024],
            "layers_per_block": 2, "norm_num_groups": 32, "norm_eps": 1e-06,
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
}
