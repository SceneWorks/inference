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

/// Encoded pixel channels.
pub const ENCODER_IN_CHANNELS: usize = 3;
/// Channels each encoder down-block emits (`block_out_channels`).
pub const ENCODER_BLOCK_OUT_CHANNELS: [usize; 6] = [128, 256, 256, 512, 512, 1024];
/// Residual blocks per encoder down-block (`layers_per_block`).
pub const ENCODER_LAYERS_PER_BLOCK: usize = 2;
/// Per-level spatial stride. Their product is [`VAE_RATIO`]; levels 4 and 5 have **no**
/// downsampler, which is why the product is not `2^len`.
pub const ENCODER_SPATIAL_DOWNSAMPLE_FACTORS: [usize; 6] = [2, 2, 2, 2, 1, 1];
/// Per-level temporal stride. Their product is [`VAE_RATIO_T`].
pub const ENCODER_TEMPORAL_DOWNSAMPLE_FACTORS: [usize; 6] = [1, 2, 2, 1, 1, 1];
/// Groups in every encoder `GroupNorm`.
pub const ENCODER_NORM_NUM_GROUPS: usize = 32;
/// The encoder's norm epsilon — **1e-6**, not the decoder's 1e-5.
pub const ENCODER_NORM_EPS: f64 = 1e-6;

/// MiniMax-H3-**Image** decoder depth (`vae_diffusers/config.json` `decoder_num_layers`).
pub const IMAGE_DECODER_NUM_LAYERS: usize = 24;
/// MiniMax-H3-Image decoder attention heads; `heads · dim_head` = 1024 is its model dim.
pub const IMAGE_DECODER_NUM_HEADS: usize = 16;
/// MiniMax-H3-Image partial rotary ratio: `int(64 · 0.5)` = 32 rotated head dims, split over the
/// **two** `(h, w)` position axes (8 frequencies per axis). 32 is not a multiple of 6, which is
/// why the video decoder's 3-axis validation refused this config.
pub const IMAGE_DECODER_ROPE_DIM_RATIO: f32 = 0.5;

/// Which ViT decoder the VAE carries.
///
/// The two MiniMax checkpoints share every decoder layer (block, attention, gated FFN, registers,
/// zero CLS token, partial rotary at theta 100) and differ only in how the latent grid is laid out
/// as tokens and how `proj_out` unpacks them. One decoder type covers both; this is the switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecoderLayout {
    /// MiniMax-H3 video (`MiniMaxH3VideoViTDecoder3d`): `(t, h, w)` rotary coordinates and an
    /// output patch of `patch_size_t × patch_size²` pixels per token.
    #[default]
    Video3d,
    /// MiniMax-H3-Image (`MiniMaxH3ImageViTDecoder2d`): `(h, w)` rotary coordinates only and a
    /// spatial-only `patch_size²` output patch. There is no time axis, so no temporal chunking and
    /// no `token_drop` seam either.
    Image2d,
}

impl DecoderLayout {
    /// Rotary position axes — 3 for `(t, h, w)`, 2 for `(h, w)`.
    pub fn rope_axes(self) -> usize {
        match self {
            Self::Video3d => 3,
            Self::Image2d => 2,
        }
    }
}

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

    // --- the CNN encoder half (sc-19008) -----------------------------------------------------
    // The decoder is a 36-layer ViT and needs none of these; the encoder is a 3-D causal CNN and
    // needs all of them. `patch_size` / `patch_size_t` above are the *products* of the two factor
    // lists, which is all the decoder ever wanted — the per-level lists are kept separately
    // because the encoder's shape at level `i` depends on the individual factors, not the product.
    /// Encoded pixel channels — 3.
    pub in_channels: usize,
    /// Channels each down-block emits. `len()` is the number of levels.
    pub block_out_channels: Vec<usize>,
    /// Residual blocks per down-block.
    pub layers_per_block: usize,
    /// Per-level spatial stride. A level with factor 1 has **no** downsampler at all.
    pub spatial_downsample_factors: Vec<usize>,
    /// Per-level temporal stride, applied by the same downsampler conv.
    pub temporal_downsample_factors: Vec<usize>,
    /// Groups in every encoder `GroupNorm` — 32.
    pub norm_num_groups: usize,
    /// The **encoder's** norm epsilon, 1e-6. Distinct from [`Self::norm_eps`], which is the
    /// decoder's 1e-5; `vae/config.json` ships both (`norm_eps` and `decoder_norm_eps`) and using
    /// one for the other is a silent numeric change.
    pub encoder_norm_eps: f64,
    /// Which decoder token layout this VAE uses. [`DecoderLayout::Video3d`] for MiniMax-H3.
    pub decoder_layout: DecoderLayout,
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
            in_channels: ENCODER_IN_CHANNELS,
            block_out_channels: ENCODER_BLOCK_OUT_CHANNELS.to_vec(),
            layers_per_block: ENCODER_LAYERS_PER_BLOCK,
            spatial_downsample_factors: ENCODER_SPATIAL_DOWNSAMPLE_FACTORS.to_vec(),
            temporal_downsample_factors: ENCODER_TEMPORAL_DOWNSAMPLE_FACTORS.to_vec(),
            norm_num_groups: ENCODER_NORM_NUM_GROUPS,
            encoder_norm_eps: ENCODER_NORM_EPS,
            decoder_layout: DecoderLayout::Video3d,
        }
    }
}

impl MiniMaxH3VaeConfig {
    /// Model dim — `num_heads · head_dim`.
    pub fn dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// The shipped `MiniMaxAI/MiniMax-H3-Image` VAE (`vae_diffusers/`,
    /// `AutoencoderKLMiniMaxH3Image`): the H3 encoder geometry and latent statistics with the
    /// 24-layer / 1024-dim 2-D decoder.
    ///
    /// `clip_length` is 1 and `token_drop` 0 because an image is one frame: the image VAE has no
    /// temporal chunk plan, and those two fields only drive the video decode's chunking.
    pub fn image() -> Self {
        Self {
            num_layers: IMAGE_DECODER_NUM_LAYERS,
            num_heads: IMAGE_DECODER_NUM_HEADS,
            rope_dim_ratio: IMAGE_DECODER_ROPE_DIM_RATIO,
            clip_length: 1,
            token_drop: 0,
            decoder_layout: DecoderLayout::Image2d,
            ..Self::default()
        }
    }

    /// Rotated head dims — `int(head_dim · rope_dim_ratio)`, matching the reference's `int()`
    /// truncation. Must be divisible by `2 · axes` (half-split rotary over
    /// [`DecoderLayout::rope_axes`] position axes).
    pub fn rope_apply_dim(&self) -> usize {
        (self.head_dim as f32 * self.rope_dim_ratio) as usize
    }

    /// Suffix tokens appended after the patch tokens: the register tokens plus one zero CLS token.
    pub fn num_suffix_tokens(&self) -> usize {
        1 + self.num_register_tokens
    }

    /// Frames each decoded token expands to: `patch_size_t` for the video decoder, 1 for the image
    /// decoder (its `proj_out` emits a spatial-only patch even though the shared encoder half
    /// still compresses time by `patch_size_t`).
    pub fn decoder_patch_size_t(&self) -> usize {
        match self.decoder_layout {
            DecoderLayout::Video3d => self.patch_size_t,
            DecoderLayout::Image2d => 1,
        }
    }

    /// Output features of `proj_out` — `out_channels · decoder_patch_size_t · patch_size²`.
    pub fn patch_dim(&self) -> usize {
        self.out_channels * self.decoder_patch_size_t() * self.patch_size * self.patch_size
    }

    /// Parse a diffusers `vae/config.json` body. The four decoder knobs the diffusers config omits
    /// are taken from the module constants (see the module docs).
    pub fn from_diffusers_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| CandleError::Msg(format!("minimax-h3 vae/config.json: {e}")))?;
        let cfg = Self::parse_diffusers(&v)?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Parse a diffusers `vae_diffusers/config.json` body for MiniMax-H3-Image
    /// (`AutoencoderKLMiniMaxH3Image`).
    ///
    /// The image config carries no `clip_length` / `token_drop` (there is no temporal chunk plan);
    /// everything else is read exactly as [`Self::from_diffusers_json`] reads it, and the decoder
    /// layout is [`DecoderLayout::Image2d`]. A `_class_name` naming any other class is refused, so
    /// the video config cannot be fed through this path and silently decoded with 2 rotary axes.
    pub fn from_image_diffusers_json(text: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| CandleError::Msg(format!("minimax-h3-image vae config.json: {e}")))?;
        if let Some(class) = v.get("_class_name").and_then(serde_json::Value::as_str) {
            if class != "AutoencoderKLMiniMaxH3Image" {
                return Err(CandleError::Msg(format!(
                    "minimax-h3-image vae config.json: _class_name is {class:?}, expected \
                     \"AutoencoderKLMiniMaxH3Image\" (the image VAE's diffusers config)"
                )));
            }
        }
        let mut obj = v.as_object().cloned().ok_or_else(|| {
            CandleError::Msg("minimax-h3-image vae config.json: not an object".into())
        })?;
        for key in ["clip_length", "token_drop"] {
            if obj.contains_key(key) {
                return Err(CandleError::Msg(format!(
                    "minimax-h3-image vae config.json: unexpected {key}; the image VAE has no \
                     temporal chunk plan"
                )));
            }
        }
        obj.insert("clip_length".into(), 1.into());
        obj.insert("token_drop".into(), 0.into());
        let mut cfg = Self::parse_diffusers(&serde_json::Value::Object(obj))?;
        cfg.decoder_layout = DecoderLayout::Image2d;
        cfg.validate()?;
        Ok(cfg)
    }

    /// The field mapping both diffusers configs share; the caller sets the layout and validates.
    fn parse_diffusers(v: &serde_json::Value) -> Result<Self> {
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
        let vec_usize = |key: &str| -> Result<Vec<usize>> {
            let arr = v
                .get(key)
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| {
                    CandleError::Msg(format!("minimax-h3 vae/config.json: missing {key}"))
                })?;
            arr.iter()
                .map(|e| {
                    e.as_u64().map(|f| f as usize).ok_or_else(|| {
                        CandleError::Msg(format!(
                            "minimax-h3 vae/config.json: non-positive-integer in {key}"
                        ))
                    })
                })
                .collect()
        };
        let prod = |key: &str| -> Result<usize> { Ok(vec_usize(key)?.iter().product()) };

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
            in_channels: usize_of("in_channels")?,
            block_out_channels: vec_usize("block_out_channels")?,
            layers_per_block: usize_of("layers_per_block")?,
            spatial_downsample_factors: vec_usize("spatial_downsample_factors")?,
            temporal_downsample_factors: vec_usize("temporal_downsample_factors")?,
            norm_num_groups: usize_of("norm_num_groups")?,
            // `norm_eps` is the ENCODER's; the decoder's ships separately as `decoder_norm_eps`.
            encoder_norm_eps: num("norm_eps")?,
            decoder_layout: DecoderLayout::Video3d,
        };
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
        // Mirrors the audio config's guard: decode multiplies by `latents_std` and encode divides
        // by it, so a zero produces a dead channel on one path and an inf/NaN on the other, and a
        // non-finite entry poisons every frame with no shape symptom anywhere.
        if let Some((i, &s)) = self
            .latents_std
            .iter()
            .enumerate()
            .find(|(_, s)| !s.is_finite() || **s <= 0.0)
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: latents_std[{i}] = {s} must be a positive finite number"
            )));
        }
        let rope = self.rope_apply_dim();
        let axes = self.decoder_layout.rope_axes();
        if rope == 0 || !rope.is_multiple_of(2 * axes) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: rope_apply_dim {rope} must be a positive multiple of {} \
                 ({axes} position axes, half-split rotary)",
                2 * axes
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
        self.validate_encoder()
    }

    /// Levels in the CNN encoder.
    pub fn num_encoder_levels(&self) -> usize {
        self.block_out_channels.len()
    }

    /// Input channels of encoder level `i` — `block_out_channels[i - 1]`, and
    /// `block_out_channels[0]` at level 0 (`conv_in` already lifts 3 → 128).
    pub fn encoder_level_in_channels(&self, level: usize) -> usize {
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
            return Err(CandleError::Msg(
                "minimax-h3 vae: the encoder needs at least one level".into(),
            ));
        }
        if self.spatial_downsample_factors.len() != levels
            || self.temporal_downsample_factors.len() != levels
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: {levels} block_out_channels but {}/{} spatial/temporal \
                 downsample factors — the three lists are per-level and must agree",
                self.spatial_downsample_factors.len(),
                self.temporal_downsample_factors.len()
            )));
        }
        if self.in_channels == 0 || self.layers_per_block == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 vae: in_channels and layers_per_block must be positive".into(),
            ));
        }
        if self.norm_num_groups == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 vae: norm_num_groups must be positive".into(),
            ));
        }
        if let Some(bad) = self
            .block_out_channels
            .iter()
            .find(|&&c| c == 0 || !c.is_multiple_of(self.norm_num_groups))
        {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: encoder channel width {bad} must be a positive multiple of the \
                 {} GroupNorm groups",
                self.norm_num_groups
            )));
        }
        if self
            .spatial_downsample_factors
            .iter()
            .chain(&self.temporal_downsample_factors)
            .any(|&f| f == 0)
        {
            return Err(CandleError::Msg(
                "minimax-h3 vae: downsample factors must be positive".into(),
            ));
        }
        // **Only 1 and 2 are expressible spatially.** The strided downsampler is a kernel-3 conv
        // with no spatial padding, preceded by an asymmetric bottom/right pad of exactly 1 — which
        // makes the output `ceil(size / 2)` for a stride of 2 and is applied for **that stride
        // alone** (`DownBlock3d::pads_bottom_right`, mirroring the reference's own
        // `if self.spatial_stride == 2`). A stride of 3 or more needs a different pad, and without
        // it the conv silently CROPS instead of compressing. A validator that admits a factor the
        // forward pass mishandles is how silent wrongness gets in, so reject it here rather than
        // let it load (sc-19008 review). The temporal factor has no such cap: its front-only pad
        // of 2 against a kernel of 3 yields `ceil(frames / stride)` at any stride.
        if let Some(&bad) = self.spatial_downsample_factors.iter().find(|&&f| f > 2) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: spatial downsample factor {bad} is not expressible — the \
                 encoder's asymmetric bottom/right pad of 1 only produces ceil(size / 2), so only \
                 factors 1 and 2 compress rather than crop"
            )));
        }
        let spatial: usize = self.spatial_downsample_factors.iter().product();
        let temporal: usize = self.temporal_downsample_factors.iter().product();
        if spatial != self.patch_size || temporal != self.patch_size_t {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: the encoder compresses {spatial}x spatially and {temporal}x \
                 temporally, but the decoder patch is {}x / {}x — encode and decode must be \
                 inverse geometries",
                self.patch_size, self.patch_size_t
            )));
        }
        if self.encoder_norm_eps <= 0.0 {
            return Err(CandleError::Msg(
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

    /// The published `MiniMaxAI/MiniMax-H3-Image` `vae_diffusers/config.json`, verbatim in every
    /// field the parser reads.
    const IMAGE_DIFFUSERS_JSON: &str = r#"{
        "_class_name": "AutoencoderKLMiniMaxH3Image", "_diffusers_version": "0.42.0.dev0",
        "in_channels": 3, "out_channels": 3, "latent_channels": 24,
        "block_out_channels": [128,256,256,512,512,1024], "layers_per_block": 2,
        "spatial_downsample_factors": [2,2,2,2,1,1],
        "temporal_downsample_factors": [1,2,2,1,1,1],
        "norm_num_groups": 32, "norm_eps": 1e-06, "spatial_padding_mode": "reflect",
        "decoder_num_layers": 24, "decoder_num_attention_heads": 16,
        "decoder_attention_head_dim": 64, "decoder_num_register_tokens": 4,
        "decoder_ffn_mult": 4, "decoder_rope_theta": 100.0, "decoder_rope_dim_ratio": 0.5,
        "decoder_norm_eps": 1e-05,
        "latents_mean": [0.858090341091156,-0.9606591463088989,1.0661640167236328,-0.5090325474739075,-0.2727581858634949,-1.3675414323806763,-0.2553254961967468,-0.26907554268836975,-0.5376840829849243,-0.0464097298681736,0.6657370328903198,0.19690127670764923,-0.5460608005523682,-0.4035342037677765,-0.23683024942874908,0.25928452610969543,-0.30133944749832153,0.211341992020607,-1.1206848621368408,0.3581933379173279,-0.04225143790245056,0.2604829967021942,0.22864092886447906,0.7056031823158264],
        "latents_std": [1.2223774194717407,1.2767263650894165,1.6831774711608887,1.7549455165863037,1.5636216402053833,2.194143533706665,0.9653137922286987,1.0569885969161987,0.841948926448822,0.7729952931404114,1.8955937623977661,0.946841835975647,0.7996809482574463,0.44988900423049927,0.7197399735450745,0.6936293244361877,2.961095094680786,2.7694199085235596,3.0496184825897217,2.1088054180145264,3.276226282119751,3.1627357006073,2.2816812992095947,2.6127843856811523]
    }"#;

    #[test]
    fn image_diffusers_json_parses_to_the_image_defaults() {
        let parsed = MiniMaxH3VaeConfig::from_image_diffusers_json(IMAGE_DIFFUSERS_JSON).unwrap();
        assert_eq!(parsed, MiniMaxH3VaeConfig::image());
        assert_eq!(parsed.decoder_layout, DecoderLayout::Image2d);
        assert_eq!(parsed.dim(), 1024);
        assert_eq!(parsed.num_layers, 24);
        // 64 · 0.5 = 32 rotated dims over 2 axes -> 8 frequencies per axis.
        assert_eq!(parsed.rope_apply_dim(), 32);
        assert_eq!(
            parsed.rope_apply_dim() % (2 * parsed.decoder_layout.rope_axes()),
            0
        );
        // proj_out is the published [768, 1024]: 3 · 16 · 16, no temporal factor.
        assert_eq!(parsed.patch_dim(), 768);
        // The shared encoder half still compresses time 4x; only the decoder patch is 2-D.
        assert_eq!(parsed.patch_size_t, 4);
        assert_eq!(parsed.decoder_patch_size_t(), 1);
        // The statistics are H3's own, verified identical on the published configs.
        let video = MiniMaxH3VaeConfig::default();
        assert_eq!(parsed.latents_mean, video.latents_mean);
        assert_eq!(parsed.latents_std, video.latents_std);
    }

    /// The 3-axis validation is what refused the H3-Image config before the 2-D layout existed;
    /// it still refuses that ratio for the video layout.
    #[test]
    fn the_image_rope_ratio_is_legal_only_on_the_two_axis_layout() {
        assert!(MiniMaxH3VaeConfig::image().validate().is_ok());
        let msg = MiniMaxH3VaeConfig {
            decoder_layout: DecoderLayout::Video3d,
            ..MiniMaxH3VaeConfig::image()
        }
        .validate()
        .expect_err("32 rotated dims cannot split over 3 axes")
        .to_string();
        assert!(msg.contains("multiple of 6"), "{msg}");
        // A ratio legal for 3 axes but not 2: 64 · 0.375 = 24 -> 24 % 4 == 0, legal; 64 · 0.1875
        // = 12 -> legal on both; 64 · 0.09375 = 6 -> 6 % 4 != 0, refused on the image layout.
        let msg = MiniMaxH3VaeConfig {
            rope_dim_ratio: 0.093_75,
            ..MiniMaxH3VaeConfig::image()
        }
        .validate()
        .expect_err("6 rotated dims cannot split over 2 axes")
        .to_string();
        assert!(msg.contains("multiple of 4"), "{msg}");
    }

    #[test]
    fn the_image_parser_refuses_the_video_config() {
        let video = IMAGE_DIFFUSERS_JSON.replace(
            "\"_class_name\": \"AutoencoderKLMiniMaxH3Image\"",
            "\"_class_name\": \"AutoencoderKLMiniMaxH3\"",
        );
        let msg = MiniMaxH3VaeConfig::from_image_diffusers_json(&video)
            .expect_err("the video class is not the image VAE")
            .to_string();
        assert!(msg.contains("_class_name"), "{msg}");
        let chunked = IMAGE_DIFFUSERS_JSON.replace(
            "\"latent_channels\": 24,",
            "\"latent_channels\": 24, \"clip_length\": 17,",
        );
        let msg = MiniMaxH3VaeConfig::from_image_diffusers_json(&chunked)
            .expect_err("an image config carries no temporal chunk plan")
            .to_string();
        assert!(msg.contains("clip_length"), "{msg}");
    }

    #[test]
    fn validate_rejects_mismatched_latent_statistics() {
        let mut cfg = MiniMaxH3VaeConfig::default();
        cfg.latents_mean.pop();
        assert!(cfg.validate().is_err());
    }

    /// A zero or non-finite `latents_std` entry is refused with a diagnostic naming the channel —
    /// decode multiplies by it and encode divides by it, so the failure would otherwise be a dead
    /// channel or a NaN-poisoned latent with no shape symptom.
    #[test]
    fn validate_rejects_degenerate_latent_std_entries() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut cfg = MiniMaxH3VaeConfig::default();
            cfg.latents_std[7] = bad;
            let e = cfg
                .validate()
                .expect_err(&format!("latents_std entry {bad} must be rejected"))
                .to_string();
            assert!(e.contains("latents_std[7]"), "{e}");
        }
    }

    #[test]
    fn a_malformed_config_is_an_error_not_a_default() {
        assert!(MiniMaxH3VaeConfig::from_diffusers_json("{").is_err());
        assert!(MiniMaxH3VaeConfig::from_diffusers_json("{}").is_err());
    }

    /// AC (sc-19008): the shipped encoder geometry, and the two facts about it a port is most
    /// likely to get wrong — the encoder's epsilon is **not** the decoder's, and the downsampler
    /// predicate is the **product** of the two factors.
    #[test]
    fn shipped_encoder_geometry_is_pinned() {
        let cfg = MiniMaxH3VaeConfig::default();
        assert_eq!(cfg.num_encoder_levels(), 6);
        assert_eq!(cfg.block_out_channels, vec![128, 256, 256, 512, 512, 1024]);
        assert_eq!(cfg.layers_per_block, 2);
        assert_eq!(cfg.in_channels, 3);
        assert_eq!(cfg.norm_num_groups, 32);
        // The encoder's eps is 1e-6 and the decoder's is 1e-5. Using one for the other is a silent
        // numeric change that no shape or key-mapping proof can see.
        assert_eq!(cfg.encoder_norm_eps, 1e-6);
        assert_eq!(cfg.norm_eps, 1e-5);
        assert_ne!(cfg.encoder_norm_eps, cfg.norm_eps);

        // Level 0 takes `block_out_channels[0]` because `conv_in` already lifted 3 -> 128.
        assert_eq!(cfg.encoder_level_in_channels(0), 128);
        for level in 1..6 {
            assert_eq!(
                cfg.encoder_level_in_channels(level),
                cfg.block_out_channels[level - 1]
            );
        }
        // Downsamplers on 0..4, none on 4 and 5 (both factors 1).
        for level in 0..4 {
            assert!(cfg.encoder_level_has_downsampler(level), "level {level}");
        }
        for level in [4, 5] {
            assert!(!cfg.encoder_level_has_downsampler(level), "level {level}");
        }
    }

    /// The downsampler predicate is `temporal · spatial > 1`, **not** `spatial > 1`. A
    /// temporal-only level really does carry a downsampler, so the two predicates are genuinely
    /// different and only one of them matches the checkpoint.
    #[test]
    fn a_temporal_only_level_still_carries_a_downsampler() {
        let cfg = MiniMaxH3VaeConfig {
            block_out_channels: vec![32, 32],
            spatial_downsample_factors: vec![1, 1],
            temporal_downsample_factors: vec![2, 1],
            patch_size: 1,
            patch_size_t: 2,
            norm_num_groups: 32,
            layers_per_block: 1,
            ..Default::default()
        };
        cfg.validate_encoder().unwrap();
        assert!(
            cfg.encoder_level_has_downsampler(0),
            "spatial 1 / temporal 2 downsamples in time and MUST declare its conv"
        );
        assert!(!cfg.encoder_level_has_downsampler(1));
    }

    /// Each encoder guard, mutated **individually** — mutating the set would only prove the set.
    #[test]
    fn validate_encoder_rejects_each_inconsistency_on_its_own() {
        let base = MiniMaxH3VaeConfig::default();
        base.validate_encoder().unwrap();

        // Per-level lists that disagree in length.
        let mut c = base.clone();
        c.spatial_downsample_factors.pop();
        assert!(c.validate_encoder().is_err(), "ragged spatial factors");

        let mut c = base.clone();
        c.temporal_downsample_factors.pop();
        assert!(c.validate_encoder().is_err(), "ragged temporal factors");

        // A width that is not a multiple of the GroupNorm groups cannot be normalized.
        let mut c = base.clone();
        c.block_out_channels[2] = 130;
        assert!(c.validate_encoder().is_err(), "width vs groups");

        // The encoder's compression must be the decoder's patch, or the two halves are not
        // inverse geometries.
        let mut c = base.clone();
        c.spatial_downsample_factors[4] = 2;
        assert!(c.validate_encoder().is_err(), "spatial cumprod vs patch");

        let mut c = base.clone();
        c.temporal_downsample_factors[4] = 2;
        assert!(c.validate_encoder().is_err(), "temporal cumprod vs patch");

        let mut c = base.clone();
        c.encoder_norm_eps = 0.0;
        assert!(c.validate_encoder().is_err(), "non-positive eps");

        let mut c = base.clone();
        c.layers_per_block = 0;
        assert!(c.validate_encoder().is_err(), "zero layers per block");

        let mut c = base.clone();
        c.in_channels = 0;
        assert!(c.validate_encoder().is_err(), "zero input channels");

        let mut c = base.clone();
        c.block_out_channels.clear();
        c.spatial_downsample_factors.clear();
        c.temporal_downsample_factors.clear();
        assert!(c.validate_encoder().is_err(), "no levels at all");

        let mut c = base.clone();
        c.spatial_downsample_factors[0] = 0;
        assert!(c.validate_encoder().is_err(), "zero downsample factor");

        let mut c = base;
        c.norm_num_groups = 0;
        assert!(c.validate_encoder().is_err(), "zero groups");
    }

    /// A spatial factor above 2 must be **rejected**, not loaded and silently cropped.
    ///
    /// `DownBlock3d`'s asymmetric bottom/right pad is applied for `spatial == 2` alone, so a level
    /// with a spatial factor of 4 would load, skip the pad, and have its kernel-3 stride-4 conv
    /// crop instead of compress. Before sc-19008's review the validator accepted **any** positive
    /// factor, which is an asymmetry between what it admits and what the forward pass handles.
    ///
    /// The mutation deliberately keeps the spatial cumprod at the shipped 16 (`[4, 2, 2, 1, 1, 1]`
    /// rather than the shipped `[2, 2, 2, 2, 1, 1]`) so it still satisfies the compression-vs-patch
    /// guard: a config that tripped *that* check instead would prove nothing about this one. The
    /// error text is asserted for the same reason.
    #[test]
    fn validate_encoder_rejects_a_spatial_factor_the_pad_cannot_express() {
        let base = MiniMaxH3VaeConfig::default();
        base.validate_encoder().unwrap();
        let shipped: usize = base.spatial_downsample_factors.iter().product();

        let mut c = base.clone();
        c.spatial_downsample_factors = vec![4, 2, 2, 1, 1, 1];
        let mutated: usize = c.spatial_downsample_factors.iter().product();
        assert_eq!(
            mutated, shipped,
            "the mutation must leave the cumprod at the shipped {shipped}x, or the \
             compression-vs-patch guard is what rejects it"
        );
        assert_eq!(
            c.spatial_downsample_factors.len(),
            base.spatial_downsample_factors.len(),
            "the mutation must not also make the per-level lists ragged"
        );
        let err = c
            .validate_encoder()
            .expect_err("a spatial factor of 4 is not expressible")
            .to_string();
        assert!(
            err.contains("spatial downsample factor 4"),
            "the rejection must name the offending factor, got: {err}"
        );
        assert!(
            err.contains("crop"),
            "the rejection must say what goes wrong, got: {err}"
        );

        // 3 is rejected for the same reason — the cap is "> 2", not "!= 4".
        let mut c = base.clone();
        c.spatial_downsample_factors = vec![3, 2, 2, 1, 1, 1];
        c.patch_size = 12;
        assert!(
            c.validate_encoder().is_err(),
            "an odd spatial factor above 2 is equally inexpressible"
        );

        // And the cap is SPATIAL only: the temporal front-pad of 2 against a kernel of 3 gives
        // `ceil(frames / stride)` at any stride, so a temporal 4 must still be accepted. Without
        // this arm, a validator that rejected every factor above 2 on both axes would pass the
        // assertions above while forbidding a config the forward pass handles correctly.
        let mut c = base;
        c.temporal_downsample_factors = vec![4, 2, 1, 1, 1, 1];
        c.patch_size_t = 8;
        c.validate_encoder()
            .expect("a temporal factor of 4 is expressible and must not be rejected");
    }

    /// A `vae/config.json` that omits an encoder key is an error, not a silently-defaulted
    /// encoder. This is the class the epic has been bitten by: a config key with no tensor behind
    /// it loads cleanly and is wrong.
    #[test]
    fn a_config_missing_an_encoder_key_is_rejected() {
        let full = r#"{
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
        assert!(MiniMaxH3VaeConfig::from_diffusers_json(full).is_ok());
        for key in [
            "\"in_channels\": 3,",
            "\"layers_per_block\": 2,",
            "\"norm_num_groups\": 32,",
            "\"norm_eps\": 1e-06,",
        ] {
            let without = full.replace(key, "");
            assert_ne!(without, full, "the probe must actually remove `{key}`");
            let err = MiniMaxH3VaeConfig::from_diffusers_json(&without)
                .expect_err(&format!("dropping {key} must be an error"))
                .to_string();
            assert!(err.contains("missing"), "unexpected error for {key}: {err}");
        }
        // `block_out_channels` is a list, so drop it separately.
        let without = full.replace("\"block_out_channels\": [128,256,256,512,512,1024],", "");
        assert!(MiniMaxH3VaeConfig::from_diffusers_json(&without).is_err());
    }
}
