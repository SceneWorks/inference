//! The MiniMax-H3 video VAE decode path (`klvae.py::AutoencoderKLLegacy`).
//!
//! `decode` is three stages:
//!
//! 1. **de-normalize** the latent per channel — `z · latents_std + latents_mean` (24-vectors, not
//!    a scalar scale/shift);
//! 2. **chunk** it temporally ([`crate::chunking`]) and decode each chunk;
//! 3. **cross-fade** the `token_drop`-induced seams and trim the repeat padding.
//!
//! `encode` (sc-19008, [`crate::vae_encoder`]) is the mirror image and is needed by `fl2va`: a
//! keyframe is conditioned through both the text encoder's vision tower **and** this VAE. The two
//! halves are structurally unrelated — the decoder is a 36-layer ViT, the encoder a causal conv
//! stack — and only `encode`'s single-frame short circuit is subtle enough to restate here: a
//! keyframe is one frame, so it skips the 17-frame temporal chunking entirely.
//!
//! The pipeline (sc-17156) and Ref2VA (sc-17157) are separate slices.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::loader::sorted_safetensors;
use candle_gen::{CandleError, Result, Weights};

use crate::blocks::blend;
use crate::chunking::{TemporalGeometry, TemporalPlan};
use crate::config::MiniMaxH3VaeConfig;
use crate::decoder::ViT3dDecoder;
use crate::nn::linear;
use crate::spatial_tiling::{BoundedStitch, SpatialTiling, TilePlan};
use crate::vae_encoder::{DiagonalGaussian, VideoEncoder3d};

/// Split a fused `to_qkv` tensor into the published checkpoint's `to_q`/`to_k`/`to_v`.
///
/// The reference reads its fused projection as `qkv.view(B, S, -1, 3·dim_head)` followed by
/// `chunk(3, dim=-1)`, so the output features are ordered **per head** as `[q_h, k_h, v_h]` — NOT
/// as three contiguous `heads·dim_head` blocks. Output row `h·(3·dim_head) + j·dim_head + d`
/// therefore belongs to projection `j` of head `h`.
///
/// This is [`crate::layout`]'s Rule 2, VAE form. The published `vae/` shards ship the split form
/// already, so nothing calls this on the load path; it exists to pin the equivalence, and the
/// parity fixture carries the reference's fused tensors alongside the split ones so the rule is
/// asserted against real reference output rather than against this comment. A naive
/// `chunk(3, dim=0)` is a different partition of the same rows and is shape-identical.
pub fn split_fused_qkv(fused: &Tensor, heads: usize, head_dim: usize) -> Result<[Tensor; 3]> {
    let s = fused.dims();
    if s.is_empty() || s[0] != 3 * heads * head_dim {
        return Err(CandleError::Msg(format!(
            "minimax-h3 qkv split: expected leading dim {}, got {s:?}",
            3 * heads * head_dim
        )));
    }
    let mut lead = vec![heads, 3, head_dim];
    lead.extend_from_slice(&s[1..]);
    let view = fused.reshape(lead)?;
    let mut shape = vec![heads * head_dim];
    shape.extend_from_slice(&s[1..]);
    let mut out = Vec::with_capacity(3);
    for j in 0..3 {
        let part = view.narrow(1, j, 1)?.contiguous()?;
        out.push(part.reshape(shape.clone())?);
    }
    let mut it = out.into_iter();
    Ok([
        it.next().expect("3 splits"),
        it.next().expect("3 splits"),
        it.next().expect("3 splits"),
    ])
}

/// The encode half — `quant_conv` and the CNN encoder, loaded together or not at all.
#[derive(Debug, Clone)]
struct EncodeHalf {
    quant_w: Tensor,
    quant_b: Tensor,
    encoder: VideoEncoder3d,
}

/// The video VAE — both halves: the ViT decoder (sc-17154) and the causal-CNN encoder (sc-19008).
#[derive(Debug, Clone)]
pub struct MiniMaxH3VideoVae {
    post_quant_w: Tensor,
    post_quant_b: Tensor,
    decoder: ViT3dDecoder,
    /// The encode half, present only when the weight map carried `encoder.*` / `quant_conv.*`.
    /// See [`MiniMaxH3VideoVae::from_weights`].
    encode: Option<Box<EncodeHalf>>,
    geometry: TemporalGeometry,
    latents_mean: Tensor,
    latents_std: Tensor,
    cfg: MiniMaxH3VaeConfig,
    dtype: DType,
    /// `use_tiling` plus the four `tile_sample_min_*` knobs, shared by encode and decode exactly as
    /// upstream shares them. Defaults to the shipped 256/64 with tiling **on**, exactly as
    /// `AutoencoderKLMiniMaxH3.__init__` does.
    tiling: SpatialTiling,
}

impl MiniMaxH3VideoVae {
    /// Build from an already-populated [`Weights`] map (the parity fixtures take this path).
    pub fn from_weights(
        w: &Weights,
        cfg: &MiniMaxH3VaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        cfg.validate()?;
        // `post_quant_conv` is an `nn.Conv3d(embed_dim, z_channels, 1)` — a 1×1×1 kernel, i.e. a
        // pointwise channel map. Reshaping `[C_out, C_in, 1, 1, 1]` to `[C_out, C_in]` and
        // applying it as a linear over the channel axis is exactly equivalent and avoids a conv3d
        // candle does not ship.
        let raw = w.require("post_quant_conv.weight")?.to_dtype(dtype)?;
        let s = raw.dims();
        let c = cfg.latent_channels;
        if s != [c, c, 1, 1, 1] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: post_quant_conv.weight is {s:?}, expected [{c}, {c}, 1, 1, 1]"
            )));
        }
        let post_quant_w = raw.reshape((c, c))?;
        let post_quant_b = w.require("post_quant_conv.bias")?.to_dtype(dtype)?;

        // The encode half is loaded **only when the map carries it**. The published `vae/` always
        // does — [`Self::load`] enforces that — but the committed decode parity fixture does not,
        // because its bytes are shared verbatim with `mlx-gen-minimax-h3` and adding tensors to it
        // would break both crates' cross-backend digests. `encode` then reports a typed error
        // naming the missing half rather than panicking or silently returning noise.
        let encode = if w.contains("encoder.conv_in.weight") {
            // `quant_conv` is the mirror of `post_quant_conv`: `nn.Conv3d(2·z, 2·z, 1)` over the
            // POSTERIOR PARAMETERS, so it is twice as wide — it maps mean and logvar together.
            let raw_q = w.require("quant_conv.weight")?.to_dtype(dtype)?;
            let sq = raw_q.dims();
            let c2 = 2 * c;
            if sq != [c2, c2, 1, 1, 1] {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 vae: quant_conv.weight is {sq:?}, expected [{c2}, {c2}, 1, 1, 1] \
                     — it maps the posterior's mean AND logvar, so it is twice the latent width"
                )));
            }
            Some(Box::new(EncodeHalf {
                quant_w: raw_q.reshape((c2, c2))?,
                quant_b: w.require("quant_conv.bias")?.to_dtype(dtype)?,
                encoder: VideoEncoder3d::from_weights(w, "encoder", cfg, dtype)?,
            }))
        } else {
            None
        };

        let decoder = ViT3dDecoder::from_weights(w, "decoder", cfg, dtype)?;
        let geometry = TemporalGeometry::new(cfg)?;

        Ok(Self {
            post_quant_w,
            post_quant_b,
            decoder,
            encode,
            geometry,
            latents_mean: Tensor::from_vec(cfg.latents_mean.clone(), (1, c, 1, 1, 1), device)?
                .to_dtype(dtype)?,
            latents_std: Tensor::from_vec(cfg.latents_std.clone(), (1, c, 1, 1, 1), device)?
                .to_dtype(dtype)?,
            cfg: cfg.clone(),
            dtype,
            tiling: SpatialTiling::default(),
        })
    }

    /// The tiling knobs in force, for both halves.
    pub fn tiling(&self) -> SpatialTiling {
        self.tiling
    }

    /// The reference's `enable_tiling(...)`: turn spatial tiling on, overriding only what is given.
    ///
    /// Tiling is already on by default, so this is only needed to change the geometry or to undo a
    /// [`Self::disable_tiling`].
    pub fn enable_tiling(
        &mut self,
        tile_height: Option<usize>,
        tile_width: Option<usize>,
        overlap_height: Option<usize>,
        overlap_width: Option<usize>,
    ) {
        self.tiling
            .enable(tile_height, tile_width, overlap_height, overlap_width);
    }

    /// The reference's `disable_tiling()`.
    ///
    /// **This changes the output.** MiniMax-H3 was released with tiling enabled and the released
    /// frames are the blended-tile ones, so an untiled decode is a different — not merely a slower
    /// or larger — result at any canvas above one tile. It exists because the reference exposes it
    /// and the parity tests need to demonstrate the difference.
    pub fn disable_tiling(&mut self) {
        self.tiling.disable();
    }

    /// Builder form of [`Self::enable_tiling`] / [`Self::disable_tiling`], for the fixtures.
    pub fn with_tiling(mut self, tiling: SpatialTiling) -> Self {
        self.tiling = tiling;
        self
    }

    /// Every tensor name the VAE consumes — the exhaustive mapping over **both** halves. Any
    /// published `vae/` tensor outside this set would be silently ignored, which is the failure
    /// mode this list exists to make testable.
    pub fn tensor_names(cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        let mut v = vec![
            "post_quant_conv.weight".to_string(),
            "post_quant_conv.bias".to_string(),
            "quant_conv.weight".to_string(),
            "quant_conv.bias".to_string(),
        ];
        v.extend(ViT3dDecoder::names("decoder", cfg));
        v.extend(VideoEncoder3d::names("encoder", cfg));
        v
    }

    /// Every tensor name the **decode** half alone consumes — [`Self::tensor_names`] minus
    /// `encoder.*` / `quant_conv.*`.
    ///
    /// Kept as its own entry point because a decode-only caller (and the committed decode fixture)
    /// legitimately holds a weight map without the encode half, and because the two counts are
    /// what the exhaustive real-weight proof partitions the published index into.
    pub fn decode_tensor_names(cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        Self::tensor_names(cfg)
            .into_iter()
            .filter(|n| !n.starts_with("encoder.") && !n.starts_with("quant_conv."))
            .collect()
    }

    /// Every tensor name the **encode** half alone consumes — the 118 `encoder.*` /
    /// `quant_conv.*` keys the published index carries.
    pub fn encode_tensor_names(cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        Self::tensor_names(cfg)
            .into_iter()
            .filter(|n| n.starts_with("encoder.") || n.starts_with("quant_conv."))
            .collect()
    }

    /// Load from a snapshot root — reads `vae/config.json` and the `vae/` shards, **both halves**.
    pub fn load(root: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = root.as_ref().join("vae");
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            CandleError::Msg(format!(
                "minimax-h3 vae: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3VaeConfig::from_diffusers_json(&text)?;
        let files = sorted_safetensors(&dir, "minimax-h3 vae")?;
        let w = Weights::from_files_filtered(
            &files,
            device,
            dtype,
            &["decoder.", "post_quant_conv.", "encoder.", "quant_conv."],
        )?;
        let vae = Self::from_weights(&w, &cfg, device, dtype)?;
        // **Production is strict.** `from_weights` tolerates a decode-only weight map because the
        // committed parity fixture is one; a real `vae/` component always ships all 703 tensors,
        // so a snapshot missing the encode half is a broken download, not a supported shape. Fail
        // here rather than at the first `fl2va` request.
        if !vae.can_encode() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: {} has no `encoder.*` / `quant_conv.*` tensors; the published \
                 vae/ component carries all 703 and fl2va keyframe conditioning needs the encode \
                 half",
                dir.display()
            )));
        }
        Ok(vae)
    }

    /// Load **only the decode half** from a snapshot root.
    ///
    /// The 116-tensor conv encoder never lands on the device, which is what a decode-only caller
    /// (video output, the sc-17154 smokes) wants — the encode half is **180,337,888 parameters,
    /// 0.67 GiB / 0.72 GB at f32**, and pure waste there. That is the sum over the published shard
    /// headers of the 116 `encoder.*` plus 2 `quant_conv.*` tensors, all of which ship F32; the
    /// remaining 585 decode tensors are 9.03 GiB of the component's 9.70 GiB. `encode` on the
    /// result reports a typed error naming the missing half.
    pub fn load_decode_only(root: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = root.as_ref().join("vae");
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            CandleError::Msg(format!(
                "minimax-h3 vae: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3VaeConfig::from_diffusers_json(&text)?;
        let files = sorted_safetensors(&dir, "minimax-h3 vae")?;
        let w =
            Weights::from_files_filtered(&files, device, dtype, &["decoder.", "post_quant_conv."])?;
        Self::from_weights(&w, &cfg, device, dtype)
    }

    /// The config in force.
    pub fn config(&self) -> &MiniMaxH3VaeConfig {
        &self.cfg
    }

    /// The derived temporal chunk constants.
    pub fn geometry(&self) -> TemporalGeometry {
        self.geometry
    }

    /// Per-channel latent de-normalization: `z · latents_std + latents_mean`.
    pub fn denormalize(&self, latents: &Tensor) -> Result<Tensor> {
        let z = latents.to_dtype(self.dtype)?;
        Ok(z.broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?)
    }

    /// Per-channel latent normalization: `(z - latents_mean) / latents_std` — the exact inverse of
    /// [`Self::denormalize`], and what the reference applies to a freshly-encoded conditioning
    /// latent.
    ///
    /// There is **no `scaling_factor` and no `shift_factor`** on this VAE; normalization is the
    /// per-channel 24-vector pair and lives in the pipeline rather than the model. A port that
    /// reaches for the usual scalar `scaling_factor` finds none and is liable to invent one.
    pub fn normalize(&self, latents: &Tensor) -> Result<Tensor> {
        let z = latents.to_dtype(self.dtype)?;
        Ok(z.broadcast_sub(&self.latents_mean)?
            .broadcast_div(&self.latents_std)?)
    }

    /// Whether the encode half is loaded.
    pub fn can_encode(&self) -> bool {
        self.encode.is_some()
    }

    /// The encode half, or a typed error naming what is missing.
    fn encode_half(&self) -> Result<&EncodeHalf> {
        self.encode.as_deref().ok_or_else(|| {
            CandleError::Msg(
                "minimax-h3 vae: this VAE was built without its encode half (no `encoder.*` / \
                 `quant_conv.*` in the weight map), so it can decode but not encode"
                    .into(),
            )
        })
    }

    /// The pointwise `quant_conv` channel map over the posterior parameters.
    fn quant_conv(&self, half: &EncodeHalf, x: &Tensor) -> Result<Tensor> {
        let nhwc = x.permute((0, 2, 3, 4, 1))?.contiguous()?;
        let mapped = linear(&nhwc, &half.quant_w, &half.quant_b)?;
        Ok(mapped.permute((0, 4, 1, 2, 3))?.contiguous()?)
    }

    /// `encoder` then `quant_conv` on ONE temporal clip, **spatially tiled** — the reference's
    /// `_encode_clip`.
    ///
    /// Tiling is on by default in the shipped VAE
    /// ([`crate::vae_encoder::ENCODER_TILING_IS_ON_BY_DEFAULT`]), so this is the released
    /// behaviour, not an opt-in memory optimization. At a canvas no larger than one tile the plan
    /// degenerates to a single span and the result is bit-identical to
    /// [`Self::encode_clip_untiled`].
    pub fn encode_clip(&self, pixels: &Tensor) -> Result<Tensor> {
        if !self.tiling.enabled {
            return self.encode_clip_untiled(pixels);
        }
        // Encode and decode read the SAME knobs, because upstream has one `use_tiling` and one set
        // of `tile_sample_min_*` shared by both halves. `fl2va` keyframe conditioning encodes
        // through this path, so a caller that retunes tiling for decode retunes it for the
        // conditioning encode too — which is upstream's behaviour, not an accident of this port.
        self.encode_clip_tiled_2d(
            pixels,
            self.tiling.tile_height,
            self.tiling.tile_width,
            self.tiling.overlap_height,
            self.tiling.overlap_width,
        )
    }

    /// The encoder and `quant_conv` in ONE pass — the reference's `use_tiling = False` branch.
    pub fn encode_clip_untiled(&self, pixels: &Tensor) -> Result<Tensor> {
        let half = self.encode_half()?;
        let pixels = pixels.to_dtype(self.dtype)?;
        self.quant_conv(half, &half.encoder.forward(&pixels)?)
    }

    /// [`Self::encode_clip`] at an explicit tile geometry.
    ///
    /// Production always uses the shipped 256/64. This exists because a fixture canvas large
    /// enough to tile at 256 px would not be committable, so the parity fixture shrinks the tile
    /// geometry to exercise **this same code path** at fixture scale — the alternative being a
    /// tiling implementation that nothing gates.
    pub fn encode_clip_tiled(
        &self,
        pixels: &Tensor,
        tile_size: usize,
        min_overlap: usize,
    ) -> Result<Tensor> {
        self.encode_clip_tiled_2d(pixels, tile_size, tile_size, min_overlap, min_overlap)
    }

    /// [`Self::encode_clip_tiled`] with the height and width geometry given separately, as the
    /// reference's four `tile_sample_min_*` fields allow.
    pub fn encode_clip_tiled_2d(
        &self,
        pixels: &Tensor,
        tile_height: usize,
        tile_width: usize,
        overlap_height: usize,
        overlap_width: usize,
    ) -> Result<Tensor> {
        let half = self.encode_half()?;
        let s = pixels.dims().to_vec();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae encode: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        let (height, width) = (s[3], s[4]);
        let ratio = self.cfg.patch_size;
        let rows = TilePlan::split(height, tile_height, overlap_height, ratio)?;
        let cols = TilePlan::split(width, tile_width, overlap_width, ratio)?;
        let pixels = pixels.to_dtype(self.dtype)?;

        // The overlaps blend in LATENT units — the pixel overlaps are exact multiples of the
        // spatial compression ratio precisely so this division is exact.
        let latent = |o: &[usize]| o.iter().map(|v| v / ratio).collect::<Vec<usize>>();
        // Streamed through the same [`BoundedStitch`] the decode half uses, so the two halves of
        // this pair do not drift apart — the failure sc-19008 already found on exactly this pair,
        // where `encode_clip` hardcoded its tile constants while `decode_clip` read `self.tiling`.
        let mut stitch = BoundedStitch::new(
            rows.len(),
            cols.len(),
            &latent(&rows.overlaps),
            &latent(&cols.overlaps),
        )?;
        for (i, &y) in rows.starts.iter().enumerate() {
            for (j, &x) in cols.starts.iter().enumerate() {
                let tile = pixels.narrow(3, y, rows.lengths[i])?;
                let tile = tile.narrow(4, x, cols.lengths[j])?.contiguous()?;
                stitch.push(self.quant_conv(half, &half.encoder.forward(&tile)?)?)?;
            }
        }
        stitch.finish()
    }

    /// Encode normalized pixels `[B, 3, T, H, W]` into a posterior — the reference's `encode`.
    ///
    /// **A single frame takes a different path.** `T == 1` short-circuits straight to
    /// [`Self::encode_clip`]: no padding up to `clip_length`, no temporal chunking, no
    /// `token_drop`, and exactly one latent frame comes out. Padding a keyframe up to 17 frames by
    /// repetition instead would run the temporal path over 17 copies of the same image and return
    /// several latent frames rather than one — a plausible tensor of the wrong shape carrying
    /// conditioning MiniMax-H3 was never trained on. This is the path `fl2va` takes for every
    /// keyframe.
    ///
    /// Longer inputs are padded up to a multiple of `clip_length` by **repeating the last frame**,
    /// encoded clip by clip, and have `token_drop` trailing latent frames removed — once, at the
    /// tail of the concatenation, not per clip.
    ///
    /// The pad and the multi-clip concatenation are gated by the `multiclip` probe of
    /// `real_weight_encode_matches_the_official_diffusers_vae` (20 frames, deliberately ragged; 25
    /// was measured inert — the pad lands inside the dropped tail, see the note in
    /// `real_weights.rs`).
    /// Every committed fixture is 1, 5 or exactly 17 frames and reaches neither, and a front pad, a
    /// zero pad or a per-clip `token_drop` all give the identical output *shape* — so only a
    /// reference from the official implementation separates them (sc-19008 review).
    pub fn encode(&self, pixels: &Tensor) -> Result<DiagonalGaussian> {
        self.encode_half()?;
        let s = pixels.dims().to_vec();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae encode: expected [B, 3, T, H, W], got {s:?}"
            )));
        }
        if s[1] != self.cfg.in_channels {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae encode: expected {} input channels, got {}",
                self.cfg.in_channels, s[1]
            )));
        }
        let num_frames = s[2];
        if num_frames == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 vae encode: need at least one frame".into(),
            ));
        }
        let x = pixels.to_dtype(self.dtype)?;
        if num_frames == 1 {
            return DiagonalGaussian::from_parameters(&self.encode_clip(&x)?);
        }

        let clip = usize::try_from(self.cfg.clip_length).map_err(|_| {
            CandleError::Msg("minimax-h3 vae encode: clip_length must be positive".into())
        })?;
        let x = if num_frames.is_multiple_of(clip) {
            x
        } else {
            let pad = clip - num_frames % clip;
            let last = x.narrow(2, num_frames - 1, 1)?;
            let mut parts = vec![x.clone()];
            for _ in 0..pad {
                parts.push(last.clone());
            }
            Tensor::cat(&parts, 2)?
        };
        let padded = x.dims()[2];
        let mut moments = Vec::with_capacity(padded / clip);
        for i in 0..(padded / clip) {
            moments.push(self.encode_clip(&x.narrow(2, i * clip, clip)?.contiguous()?)?);
        }
        let refs: Vec<&Tensor> = moments.iter().collect();
        let mut out = Tensor::cat(&refs, 2)?;
        let drop = usize::try_from(self.cfg.token_drop).map_err(|_| {
            CandleError::Msg("minimax-h3 vae encode: token_drop must be >= 0".into())
        })?;
        if drop > 0 {
            let t = out.dims()[2];
            if t <= drop {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 vae encode: {t} latent frames is not more than the {drop} dropped \
                     at the tail"
                )));
            }
            out = out.narrow(2, 0, t - drop)?.contiguous()?;
        }
        DiagonalGaussian::from_parameters(&out)
    }

    /// `post_quant_conv` then the ViT decoder on ONE temporal clip, **spatially tiled** — the
    /// reference's `_decode_clip`.
    ///
    /// Tiling is on by default in the shipped VAE
    /// ([`crate::spatial_tiling::TILING_IS_ON_BY_DEFAULT`]), and upstream is explicit that **the
    /// released frames are the blended-tile ones, so disabling tiling changes the output**. This is
    /// therefore released behaviour, not an opt-in memory optimization: decoding the shipped
    /// 1344×768 canvas in one pass differs from the reference over a 7×4 grid of tiles.
    ///
    /// At a canvas no larger than one tile the plan degenerates to a single full-length span and
    /// the result is bit-identical to [`Self::decode_clip_untiled`] — which is why the sub-tile
    /// fixtures committed for sc-17154 stayed valid across this change.
    pub fn decode_clip(&self, z: &Tensor) -> Result<Tensor> {
        if !self.tiling.enabled {
            return self.decode_clip_untiled(z);
        }
        self.decode_clip_tiled(
            z,
            self.tiling.tile_height,
            self.tiling.tile_width,
            self.tiling.overlap_height,
            self.tiling.overlap_width,
        )
    }

    /// [`Self::decode_clip`] at an explicit tile geometry, ignoring `use_tiling`.
    ///
    /// Production always uses the shipped 256/64. This exists because a fixture canvas large enough
    /// to tile at 256 px would not be committable, so the parity fixtures shrink the tile geometry
    /// to exercise **this same code path** at fixture scale — the alternative being a tiling
    /// implementation that nothing gates.
    pub fn decode_clip_tiled(
        &self,
        z: &Tensor,
        tile_height: usize,
        tile_width: usize,
        overlap_height: usize,
        overlap_width: usize,
    ) -> Result<Tensor> {
        let s = z.dims();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        // **Tiles are laid out in PIXEL space and then mapped back onto the latent grid** — the
        // tile size is a pixel quantity, so planning on the latent extent directly would tile at
        // 16× the intended granularity.
        let ratio = self.cfg.patch_size;
        let rows = TilePlan::split(s[3] * ratio, tile_height, overlap_height, ratio)?;
        let cols = TilePlan::split(s[4] * ratio, tile_width, overlap_width, ratio)?;

        // **Rung 2 (sc-18660).** Tiles stream into a [`BoundedStitch`] rather than accumulating
        // into a `rows x cols` grid. The tile *geometry* is pinned by output correctness
        // (sc-18786) and cannot be a memory lever, so what rung 2 bounds is the number of decoded
        // tiles held live: `O(cols)` strips instead of the whole grid. Bit-identical to the
        // full-grid stitch, asserted at `max|Δ| == 0.0`.
        let mut stitch =
            BoundedStitch::new(rows.len(), cols.len(), &rows.overlaps, &cols.overlaps)?;
        for (i, &y) in rows.starts.iter().enumerate() {
            for (j, &x) in cols.starts.iter().enumerate() {
                let tile = z
                    .narrow(3, y / ratio, rows.lengths[i] / ratio)?
                    .narrow(4, x / ratio, cols.lengths[j] / ratio)?
                    .contiguous()?;
                stitch.push(self.decode_clip_untiled(&tile)?)?;
            }
        }
        // **The overlaps are used UNDIVIDED here.** The reference's `_encode_clip` divides them by
        // `spatial_compression_ratio` before stitching and `_decode_clip` does not: the plan is in
        // pixels either way, but a decoded tile comes back out in pixel space while an encoded one
        // comes back in latent space. Dividing here would blend over a 16×-too-narrow seam AND trim
        // 16× too little, so the stitched tensor comes out the wrong size: mutating the divide back
        // in at the 512×320 parity canvas returns [1, 3, 17, 752, 500] instead of
        // [1, 3, 17, 512, 320].
        stitch.finish()
    }

    /// `post_quant_conv` then the ViT decoder in ONE pass over the whole canvas.
    ///
    /// This is the reference's `use_tiling = False` branch. It does **not** reproduce the released
    /// frames above one tile; see [`Self::decode_clip`].
    pub fn decode_clip_untiled(&self, z: &Tensor) -> Result<Tensor> {
        let s = z.dims();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        let z = z.to_dtype(self.dtype)?;
        // Pointwise channel map in NDHWC, then back to NCTHW for the decoder's own packing.
        let nhwc = z.permute((0, 2, 3, 4, 1))?.contiguous()?;
        let mapped = linear(&nhwc, &self.post_quant_w, &self.post_quant_b)?;
        let z2 = mapped.permute((0, 4, 1, 2, 3))?.contiguous()?;
        self.decoder.forward(&z2)
    }

    /// Decode an already de-normalized latent with temporal chunking
    /// (`klvae.py::decode_temporal`).
    pub fn decode_temporal(&self, z: &Tensor) -> Result<Tensor> {
        let s = z.dims().to_vec();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 vae: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        let input_tokens = i32::try_from(s[2]).map_err(|_| {
            CandleError::Msg(format!(
                "minimax-h3 vae: {} temporal tokens is absurd",
                s[2]
            ))
        })?;
        let plan = TemporalPlan::new(self.geometry, input_tokens)?;
        let z = z.to_dtype(self.dtype)?;

        // Pad by REPEATING the last token, not with zeros.
        let z = if plan.pad_tokens > 0 {
            let last = z.narrow(2, s[2] - 1, 1)?;
            let mut parts = vec![z.clone()];
            for _ in 0..plan.pad_tokens {
                parts.push(last.clone());
            }
            Tensor::cat(&parts, 2)?
        } else {
            z
        };

        let split_count = self.geometry.split_count();
        let mut parts: Vec<Tensor> = Vec::new();
        let mut overlap: Option<Tensor> = None;

        for span in &plan.chunks {
            let clip_z = z.narrow(2, span.start as usize, span.tokens() as usize)?;
            let clip_dec = self.decode_clip(&clip_z)?;
            let clip_frames = clip_dec.dims()[2] as i32;

            for j in 0..split_count {
                let (start, end) = plan.split_span(clip_frames, j);
                // Defensive: an empty split cannot arise from a `TemporalGeometry::from_parts`
                // geometry any more — a `token_drop` of a whole chunk (the one case that zeroed
                // `token_overlap` with two splits) is now refused there — but `TemporalGeometry`'s
                // fields are public, so a literal-built geometry could still produce one. The plan
                // counts such a split as 0 frames via its `max(0, ..)`; skipping keeps the tensor
                // path and the plan in agreement rather than narrowing to a negative length.
                if start >= end {
                    continue;
                }
                let part = clip_dec.narrow(2, start as usize, (end - start) as usize)?;
                if j == 0 {
                    let part = match overlap.take() {
                        Some(prev) => blend(&prev, &part, self.geometry.frame_overlap, 2)?,
                        None => part,
                    };
                    parts.push(part);
                } else {
                    overlap = Some(part);
                }
            }
        }
        if let Some(tail) = overlap.take() {
            parts.push(tail);
        }

        let dec = Tensor::cat(&parts, 2)?;
        let frames = dec.dims()[2] as i32;
        // Mirror the reference's own frame-plan assertion rather than trusting the arithmetic.
        if frames != plan.total_frames {
            return Err(CandleError::Msg(format!(
                "minimax-h3 decode: produced {frames} frames, plan said {}",
                plan.total_frames
            )));
        }
        if plan.pad_frames > 0 {
            Ok(dec
                .narrow(2, 0, (frames - plan.pad_frames) as usize)?
                .contiguous()?)
        } else {
            Ok(dec)
        }
    }

    /// Full decode: de-normalize, then chunked temporal decode.
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let z = self.denormalize(latents)?;
        self.decode_temporal(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// **Both halves, exhaustively.** The published `vae/` index has 703 tensors: 116 encoder +
    /// 2 `quant_conv` + 583 decoder + 2 `post_quant_conv`. sc-17154 claimed only the 585 decode
    /// tensors; sc-19008 adds the encode half, so the mapping is now the whole file and any
    /// published tensor outside it is a loader gap rather than a documented omission.
    #[test]
    fn tensor_names_match_the_whole_published_vae() {
        let cfg = MiniMaxH3VaeConfig::default();
        let names = MiniMaxH3VideoVae::tensor_names(&cfg);
        assert_eq!(names.len(), 703, "the published vae/ index has 703 tensors");
        assert_eq!(583 + 2 + 116 + 2, names.len());
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "no name is claimed twice");

        // The two halves partition the whole, with no name in both and none in neither.
        let decode = MiniMaxH3VideoVae::decode_tensor_names(&cfg);
        let encode = MiniMaxH3VideoVae::encode_tensor_names(&cfg);
        assert_eq!(decode.len(), 585);
        assert_eq!(encode.len(), 118, "the encode half is 116 + 2 quant_conv");
        assert_eq!(decode.len() + encode.len(), names.len());
        let decode_set: std::collections::BTreeSet<_> = decode.iter().collect();
        assert!(
            encode.iter().all(|n| !decode_set.contains(n)),
            "the two halves must be disjoint"
        );

        // Levels 4 and 5 have NO downsampler (`temporal · spatial == 1`); emitting names for one
        // would ask for tensors the checkpoint does not have.
        assert!(
            !names
                .iter()
                .any(|n| n.contains("down_blocks.4.downsamplers")
                    || n.contains("down_blocks.5.downsamplers")),
            "the last two levels carry no downsampler"
        );
        for level in 0..4 {
            assert!(
                names.iter().any(
                    |n| n == &format!("encoder.down_blocks.{level}.downsamplers.0.conv.weight")
                ),
                "level {level} downsamples and must declare its conv"
            );
        }
        // `conv_shortcut` exists exactly where the width changes: levels 1, 3 and 5.
        for level in [1, 3, 5] {
            assert!(
                names
                    .iter()
                    .any(|n| n
                        == &format!("encoder.down_blocks.{level}.resnets.0.conv_shortcut.weight")),
                "level {level} changes width and needs a residual projection"
            );
        }
        for level in [0, 2, 4] {
            assert!(
                !names
                    .iter()
                    .any(|n| n.contains(&format!("down_blocks.{level}.resnets.0.conv_shortcut"))),
                "level {level} keeps its width and must not declare a shortcut"
            );
        }
        // Only `resnets.0` can ever change width — `resnets.1` always sees `out_channels` in.
        assert!(
            !names.iter().any(|n| n.contains("resnets.1.conv_shortcut")),
            "the second resnet of a level never changes width"
        );
    }

    /// The interleaved split is an exact partition: reassembling `[q|k|v]` per head must give the
    /// fused tensor back, byte for byte.
    #[test]
    fn fused_qkv_split_is_an_exact_partition() {
        let (heads, dim_head, cols) = (3usize, 4usize, 2usize);
        let n = 3 * heads * dim_head * cols;
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let fused =
            Tensor::from_vec(vals.clone(), (3 * heads * dim_head, cols), &Device::Cpu).unwrap();
        let [q, k, v] = split_fused_qkv(&fused, heads, dim_head).unwrap();
        for part in [&q, &k, &v] {
            assert_eq!(part.dims(), &[heads * dim_head, cols]);
        }
        // Rebuild by interleaving per head and compare with the original.
        let mut rebuilt = Vec::with_capacity(n);
        let (q, k, v) = (flat(&q), flat(&k), flat(&v));
        for h in 0..heads {
            for part in [&q, &k, &v] {
                let base = h * dim_head * cols;
                rebuilt.extend_from_slice(&part[base..base + dim_head * cols]);
            }
        }
        assert_eq!(rebuilt, vals);
    }

    /// A NAIVE `chunk(3, dim=0)` split would produce a different partition. Pin the difference so
    /// a future "simplification" to the contiguous form fails loudly.
    #[test]
    fn interleaved_split_differs_from_a_contiguous_split() {
        let (heads, dim_head) = (2usize, 2usize);
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let fused = Tensor::from_vec(vals, (12, 1), &Device::Cpu).unwrap();
        let [q, _, _] = split_fused_qkv(&fused, heads, dim_head).unwrap();
        // Interleaved: head 0 rows 0..2, head 1 rows 6..8.
        assert_eq!(flat(&q), vec![0.0, 1.0, 6.0, 7.0]);
        // A contiguous split would have produced rows 0..4.
        assert_ne!(flat(&q), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_rejects_a_mismatched_leading_dim() {
        let fused = Tensor::zeros((10, 1), DType::F32, &Device::Cpu).unwrap();
        assert!(split_fused_qkv(&fused, 2, 2).is_err());
    }

    #[test]
    fn load_errors_cleanly_on_a_missing_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("no-such-snapshot");
        match MiniMaxH3VideoVae::load(&missing, &Device::Cpu, DType::F32) {
            Err(e) => assert!(e.to_string().contains("config.json"), "{e}"),
            Ok(_) => panic!("expected a load error for a missing snapshot"),
        }
    }
}
