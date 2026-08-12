//! The MiniMax-H3 video VAE decode path (`klvae.py::AutoencoderKLLegacy`).
//!
//! `decode` is three stages:
//!
//! 1. **de-normalize** the latent per channel — `z · latents_std + latents_mean` (24-vectors, not
//!    a scalar scale/shift);
//! 2. **chunk** it temporally ([`crate::chunking`]) and decode each chunk;
//! 3. **cross-fade** the `token_drop`-induced seams and trim the repeat padding.
//!
//! `encode` (sc-17148, [`crate::vae_encoder`]) is the mirror image and is needed by `fl2va`: a
//! keyframe is conditioned through both the text encoder's vision tower **and** this VAE. The two
//! halves are structurally unrelated — the decoder is a 36-layer ViT, the encoder a causal conv
//! stack — and only `encode`'s single-frame short circuit is subtle enough to restate here: a
//! keyframe is one frame, so it skips the 17-frame temporal chunking entirely.
//!
//! The audio VAE (sc-17141), the DiT (sc-17144) and the pipeline (sc-17146/17147) are separate
//! slices.

use std::path::Path;

use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::blocks::blend;
use crate::chunking::{TemporalGeometry, TemporalPlan};
use crate::config::MiniMaxH3VaeConfig;
use crate::decoder::ViT3dDecoder;
use crate::tensor::slice_axis;
use crate::vae_encoder::{
    stitch_tiles, DiagonalGaussian, TilePlan, VideoEncoder3d, TILE_SAMPLE_MIN_OVERLAP,
    TILE_SAMPLE_MIN_SIZE,
};

/// Split a fused `to_qkv` tensor into the published checkpoint's `to_q`/`to_k`/`to_v`.
///
/// The reference reads its fused projection as `qkv.view(B, S, -1, 3·dim_head)` followed by
/// `chunk(3, dim=-1)`, so the output features are ordered **per head** as `[q_h, k_h, v_h]` — NOT
/// as three contiguous `heads·dim_head` blocks. Output row `h·(3·dim_head) + j·dim_head + d`
/// therefore belongs to projection `j` of head `h`.
///
/// The published `vae/` shards ship the split form already, so nothing calls this on the load
/// path; it exists to pin the equivalence, and the parity fixture carries the reference's fused
/// tensors alongside the split ones so the rule is asserted against real reference output rather
/// than against this comment.
pub fn split_fused_qkv(fused: &Array, heads: i32, head_dim: i32) -> Result<[Array; 3]> {
    let s = fused.shape();
    if s.is_empty() || s[0] != 3 * heads * head_dim {
        return Err(Error::Msg(format!(
            "minimax-h3 qkv split: expected leading dim {}, got {s:?}",
            3 * heads * head_dim
        )));
    }
    let mut lead = vec![heads, 3, head_dim];
    lead.extend_from_slice(&s[1..]);
    let view = fused.reshape(&lead)?;
    let mut out = Vec::with_capacity(3);
    for j in 0..3 {
        let idx = Array::from_slice(&[j], &[1]);
        let part = view.take_axis(&idx, 1)?;
        let mut shape = vec![heads * head_dim];
        shape.extend_from_slice(&s[1..]);
        out.push(part.reshape(&shape)?);
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
    quant_w: Array,
    quant_b: Array,
    encoder: VideoEncoder3d,
}

/// The video VAE — both halves: the ViT decoder (sc-17140) and the causal-CNN encoder (sc-17148).
#[derive(Debug, Clone)]
pub struct MiniMaxH3VideoVae {
    post_quant_w: Array,
    post_quant_b: Array,
    decoder: ViT3dDecoder,
    /// The encode half, present only when the weight map carried `encoder.*` / `quant_conv.*`.
    /// See [`MiniMaxH3VideoVae::from_weights`].
    encode: Option<Box<EncodeHalf>>,
    geometry: TemporalGeometry,
    latents_mean: Array,
    latents_std: Array,
    cfg: MiniMaxH3VaeConfig,
    dtype: Dtype,
}

impl MiniMaxH3VideoVae {
    /// Build from an already-populated [`Weights`] map (the parity fixtures take this path).
    pub fn from_weights(w: &mut Weights, cfg: &MiniMaxH3VaeConfig, dtype: Dtype) -> Result<Self> {
        cfg.validate()?;
        // `post_quant_conv` is an `nn.Conv3d(embed_dim, z_channels, 1)` — a 1×1×1 kernel, i.e. a
        // pointwise channel map. Reshaping `[C_out, C_in, 1, 1, 1]` to `[C_out, C_in]` and
        // applying it as a linear over the channel axis is exactly equivalent and avoids a conv.
        let raw = w.require("post_quant_conv.weight")?.as_dtype(dtype)?;
        let s = raw.shape();
        let c = cfg.latent_channels;
        if s != [c, c, 1, 1, 1] {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: post_quant_conv.weight is {s:?}, expected [{c}, {c}, 1, 1, 1]"
            )));
        }
        let post_quant_w = raw.reshape(&[c, c])?;
        let post_quant_b = w.require("post_quant_conv.bias")?.as_dtype(dtype)?;

        // The encode half is loaded **only when the map carries it**. The published `vae/` always
        // does — [`Self::load`] enforces that — but the committed decode parity fixture does not,
        // because its bytes are shared verbatim with `candle-gen-minimax-h3` (sc-17154) and adding
        // tensors to it would break that crate's cross-backend digest. `encode` then reports a
        // typed error naming the missing half rather than panicking or silently returning noise.
        let encode = if w.get("encoder.conv_in.weight").is_some() {
            // `quant_conv` is the mirror of `post_quant_conv`: `nn.Conv3d(2·z, 2·z, 1)` over the
            // POSTERIOR PARAMETERS, so it is twice as wide — it maps mean and logvar together.
            let raw_q = w.require("quant_conv.weight")?.as_dtype(dtype)?;
            let sq = raw_q.shape();
            let c2 = 2 * c;
            if sq != [c2, c2, 1, 1, 1] {
                return Err(Error::Msg(format!(
                    "minimax-h3 vae: quant_conv.weight is {sq:?}, expected [{c2}, {c2}, 1, 1, 1] \
                     — it maps the posterior's mean AND logvar, so it is twice the latent width"
                )));
            }
            Some(Box::new(EncodeHalf {
                quant_w: raw_q.reshape(&[c2, c2])?,
                quant_b: w.require("quant_conv.bias")?.as_dtype(dtype)?,
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
            latents_mean: Array::from_slice(&cfg.latents_mean, &[1, c, 1, 1, 1]).as_dtype(dtype)?,
            latents_std: Array::from_slice(&cfg.latents_std, &[1, c, 1, 1, 1]).as_dtype(dtype)?,
            cfg: cfg.clone(),
            dtype,
        })
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

    /// Load from a snapshot root — reads `vae/config.json` and the `vae/` shards.
    pub fn load(root: impl AsRef<Path>, dtype: Dtype) -> Result<Self> {
        let dir = root.as_ref().join("vae");
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| {
            Error::Msg(format!(
                "minimax-h3 vae: reading {}: {e}",
                config_path.display()
            ))
        })?;
        let cfg = MiniMaxH3VaeConfig::from_diffusers_json(&text)?;
        let mut w = Weights::from_dir(&dir)?;
        let vae = Self::from_weights(&mut w, &cfg, dtype)?;
        // **Production is strict.** `from_weights` tolerates a decode-only weight map because the
        // committed parity fixture is one; a real `vae/` component always ships all 703 tensors,
        // so a snapshot missing the encode half is a broken download, not a supported shape. Fail
        // here rather than at the first `fl2va` request.
        if !vae.can_encode() {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: {} has no `encoder.*` / `quant_conv.*` tensors; the published \
                 vae/ component carries all 703 and fl2va keyframe conditioning needs the encode \
                 half",
                dir.display()
            )));
        }
        Ok(vae)
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
    pub fn denormalize(&self, latents: &Array) -> Result<Array> {
        let z = latents.as_dtype(self.dtype)?;
        Ok(add(&multiply(&z, &self.latents_std)?, &self.latents_mean)?)
    }

    /// Per-channel latent normalization: `(z - latents_mean) / latents_std` — the exact inverse of
    /// [`Self::denormalize`], and what the reference applies to a freshly-encoded conditioning
    /// latent.
    ///
    /// There is **no `scaling_factor` and no `shift_factor`** on this VAE; normalization is the
    /// per-channel 24-vector pair and lives in the pipeline rather than the model. A port that
    /// reaches for the usual scalar `scaling_factor` finds none and is liable to invent one.
    pub fn normalize(&self, latents: &Array) -> Result<Array> {
        let z = latents.as_dtype(self.dtype)?;
        Ok(z.subtract(&self.latents_mean)?.divide(&self.latents_std)?)
    }

    /// `encoder` then `quant_conv` on ONE temporal clip, **spatially tiled** — the reference's
    /// `_encode_clip`.
    ///
    /// Tiling is on by default in the shipped VAE
    /// ([`crate::vae_encoder::ENCODER_TILING_IS_ON_BY_DEFAULT`]), so this is the released
    /// behaviour, not an opt-in memory optimization. At a canvas no larger than one tile the plan
    /// degenerates to a single span and the result is bit-identical to an untiled encode.
    pub fn encode_clip(&self, pixels: &Array) -> Result<Array> {
        self.encode_clip_tiled(pixels, TILE_SAMPLE_MIN_SIZE, TILE_SAMPLE_MIN_OVERLAP)
    }

    /// [`Self::encode_clip`] at an explicit tile geometry.
    ///
    /// Production always uses the shipped 256/64. This exists because a fixture canvas large
    /// enough to tile at 256 px would not be committable, so the parity fixture shrinks the tile
    /// geometry to exercise **this same code path** at fixture scale — the alternative being a
    /// tiling implementation that nothing gates.
    pub fn encode_clip_tiled(
        &self,
        pixels: &Array,
        tile_size: i32,
        min_overlap: i32,
    ) -> Result<Array> {
        let half = self.encode_half()?;
        let s = pixels.shape();
        if s.len() != 5 {
            return Err(Error::Msg(format!(
                "minimax-h3 vae encode: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        let (height, width) = (s[3], s[4]);
        let ratio = self.cfg.patch_size;
        let rows = TilePlan::split(height, tile_size, min_overlap, ratio)?;
        let cols = TilePlan::split(width, tile_size, min_overlap, ratio)?;

        let mut grid = Vec::with_capacity(rows.len());
        for (i, &y) in rows.starts.iter().enumerate() {
            let mut row = Vec::with_capacity(cols.len());
            for (j, &x) in cols.starts.iter().enumerate() {
                let tile = slice_axis(pixels, 3, y, y + rows.lengths[i])?;
                let tile = slice_axis(&tile, 4, x, x + cols.lengths[j])?;
                row.push(self.quant_conv(half, &half.encoder.forward(&tile)?)?);
            }
            grid.push(row);
        }
        // The overlaps blend in LATENT units — the pixel overlaps are exact multiples of the
        // spatial compression ratio precisely so this division is exact.
        let latent = |o: &[i32]| o.iter().map(|v| v / ratio).collect::<Vec<i32>>();
        stitch_tiles(&grid, &latent(&rows.overlaps), &latent(&cols.overlaps))
    }

    /// The encode half, or a typed error naming what is missing.
    fn encode_half(&self) -> Result<&EncodeHalf> {
        self.encode.as_deref().ok_or_else(|| {
            Error::Msg(
                "minimax-h3 vae: this VAE was built without its encode half (no `encoder.*` / \
                 `quant_conv.*` in the weight map), so it can decode but not encode"
                    .into(),
            )
        })
    }

    /// Whether the encode half is loaded.
    pub fn can_encode(&self) -> bool {
        self.encode.is_some()
    }

    /// The pointwise `quant_conv` channel map over the posterior parameters.
    fn quant_conv(&self, half: &EncodeHalf, x: &Array) -> Result<Array> {
        let nhwc = x.transpose_axes(&[0, 2, 3, 4, 1])?;
        let mapped = linear(&nhwc, &half.quant_w, &half.quant_b)?;
        Ok(mapped.transpose_axes(&[0, 4, 1, 2, 3])?)
    }

    /// Encode normalized pixels `[B, 3, T, H, W]` into a posterior — the reference's `encode`.
    ///
    /// **A single frame takes a different path.** `T == 1` short-circuits straight to
    /// [`Self::encode_clip`]: no padding up to `clip_length`, no temporal chunking, no
    /// `token_drop`, and exactly one latent frame comes out. Padding a keyframe up to 17 frames by
    /// repetition instead would run the temporal path over 17 copies of the same image and return
    /// `17 / 4 - 3` latent frames rather than one — a plausible tensor of the wrong shape carrying
    /// conditioning MiniMax-H3 was never trained on. This is the path `fl2va` takes for every
    /// keyframe.
    ///
    /// Longer inputs are padded up to a multiple of `clip_length` by **repeating the last frame**,
    /// encoded clip by clip, and have `token_drop` trailing latent frames removed.
    pub fn encode(&self, pixels: &Array) -> Result<DiagonalGaussian> {
        self.encode_half()?;
        let s = pixels.shape();
        if s.len() != 5 {
            return Err(Error::Msg(format!(
                "minimax-h3 vae encode: expected [B, 3, T, H, W], got {s:?}"
            )));
        }
        if s[1] != self.cfg.in_channels {
            return Err(Error::Msg(format!(
                "minimax-h3 vae encode: expected {} input channels, got {}",
                self.cfg.in_channels, s[1]
            )));
        }
        let num_frames = s[2];
        if num_frames < 1 {
            return Err(Error::Msg(
                "minimax-h3 vae encode: need at least one frame".into(),
            ));
        }
        let x = pixels.as_dtype(self.dtype)?;
        if num_frames == 1 {
            return DiagonalGaussian::from_parameters(&self.encode_clip(&x)?);
        }

        let clip = self.cfg.clip_length;
        let x = if num_frames % clip == 0 {
            x
        } else {
            let pad = (clip - num_frames % clip) % clip;
            let last = slice_axis(&x, 2, num_frames - 1, num_frames)?;
            let mut parts = vec![x.clone()];
            for _ in 0..pad {
                parts.push(last.clone());
            }
            concatenate_axis(&parts, 2)?
        };
        let padded = x.shape()[2];
        let mut moments = Vec::with_capacity((padded / clip) as usize);
        for i in 0..(padded / clip) {
            moments.push(self.encode_clip(&slice_axis(&x, 2, i * clip, (i + 1) * clip)?)?);
        }
        let mut out = concatenate_axis(&moments, 2)?;
        if self.cfg.token_drop > 0 {
            let t = out.shape()[2];
            if t <= self.cfg.token_drop {
                return Err(Error::Msg(format!(
                    "minimax-h3 vae encode: {t} latent frames is not more than the {} dropped at \
                     the tail",
                    self.cfg.token_drop
                )));
            }
            out = slice_axis(&out, 2, 0, t - self.cfg.token_drop)?;
        }
        DiagonalGaussian::from_parameters(&out)
    }

    /// `post_quant_conv` then the ViT decoder — the reference's `decode`, on ONE chunk.
    pub fn decode_clip(&self, z: &Array) -> Result<Array> {
        let s = z.shape();
        if s.len() != 5 {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        let z = z.as_dtype(self.dtype)?;
        // Pointwise channel map in NDHWC, then back to NCTHW for the decoder's own packing.
        let nhwc = z.transpose_axes(&[0, 2, 3, 4, 1])?;
        let mapped = linear(&nhwc, &self.post_quant_w, &self.post_quant_b)?;
        let z2 = mapped.transpose_axes(&[0, 4, 1, 2, 3])?;
        self.decoder.forward(&z2)
    }

    /// Decode an already de-normalized latent with temporal chunking
    /// (`klvae.py::decode_temporal`).
    pub fn decode_temporal(&self, z: &Array) -> Result<Array> {
        let s = z.shape();
        if s.len() != 5 {
            return Err(Error::Msg(format!(
                "minimax-h3 vae: expected [B, C, T, H, W], got {s:?}"
            )));
        }
        let plan = TemporalPlan::new(self.geometry, s[2])?;
        let z = z.as_dtype(self.dtype)?;

        // Pad by REPEATING the last token, not with zeros.
        let z = if plan.pad_tokens > 0 {
            let last = slice_axis(&z, 2, s[2] - 1, s[2])?;
            let mut parts = vec![z];
            for _ in 0..plan.pad_tokens {
                parts.push(last.clone());
            }
            concatenate_axis(&parts, 2)?
        } else {
            z
        };

        let split_count = self.geometry.split_count();
        let mut parts: Vec<Array> = Vec::new();
        let mut overlap: Option<Array> = None;

        for span in &plan.chunks {
            let clip_z = slice_axis(&z, 2, span.start, span.end)?;
            let clip_dec = self.decode_clip(&clip_z)?;
            let clip_frames = clip_dec.shape()[2];

            for j in 0..split_count {
                let (start, end) = plan.split_span(clip_frames, j);
                // A split can legitimately be empty when `token_overlap` is 0 but `token_drop` is
                // not (i.e. `token_drop` is a multiple of `tokens_chunk_size`): the chunk decodes
                // to exactly `chunk_dec` frames, so split 1 has nothing left. The reference
                // produces an empty tensor there and its blend degenerates to the next chunk; the
                // plan already counts such a split as 0 frames via its `max(0, ..)`, so skipping
                // keeps the tensor path and the plan in agreement. The shipped config never hits
                // this (overlap 2), but a caller-supplied `token_drop` can.
                if start >= end {
                    continue;
                }
                let part = slice_axis(&clip_dec, 2, start, end)?;
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

        let dec = concatenate_axis(&parts, 2)?;
        let frames = dec.shape()[2];
        // Mirror the reference's own frame-plan assertion rather than trusting the arithmetic.
        if frames != plan.total_frames {
            return Err(Error::Msg(format!(
                "minimax-h3 decode: produced {frames} frames, plan said {}",
                plan.total_frames
            )));
        }
        if plan.pad_frames > 0 {
            slice_axis(&dec, 2, 0, frames - plan.pad_frames)
        } else {
            Ok(dec)
        }
    }

    /// Full decode: de-normalize, then chunked temporal decode.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let z = self.denormalize(latents)?;
        self.decode_temporal(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Both halves, exhaustively.** The published `vae/` index has 703 tensors: 116 encoder +
    /// 2 `quant_conv` + 583 decoder + 2 `post_quant_conv`. sc-17140 claimed only the 585 decode
    /// tensors; sc-17148 added the encode half, so the mapping is now the whole file and any
    /// published tensor outside it is a loader gap rather than a documented omission.
    #[test]
    fn tensor_names_match_the_whole_published_vae() {
        let cfg = MiniMaxH3VaeConfig::default();
        let names = MiniMaxH3VideoVae::tensor_names(&cfg);
        assert_eq!(names.len(), 703, "the published vae/ index has 703 tensors");
        assert_eq!(583 + 2 + 116 + 2, names.len());
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "no name is claimed twice");

        // The encode half on its own is the 118 the spike counted.
        let encoder: Vec<&String> = names
            .iter()
            .filter(|n| n.starts_with("encoder.") || n.starts_with("quant_conv."))
            .collect();
        assert_eq!(encoder.len(), 118);
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
    }

    /// The interleaved split is an exact partition: reassembling `[q|k|v]` per head must give the
    /// fused tensor back, byte for byte.
    #[test]
    fn fused_qkv_split_is_an_exact_partition() {
        let (heads, dim_head, cols) = (3, 4, 2);
        let n = 3 * heads * dim_head * cols;
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let fused = Array::from_slice(&vals, &[3 * heads * dim_head, cols]);
        let [q, k, v] = split_fused_qkv(&fused, heads, dim_head).unwrap();
        for part in [&q, &k, &v] {
            assert_eq!(part.shape(), &[heads * dim_head, cols]);
        }
        // Rebuild by interleaving per head and compare with the original.
        let mut rebuilt = Vec::with_capacity(n as usize);
        let (q, k, v) = (
            q.as_slice::<f32>().to_vec(),
            k.as_slice::<f32>().to_vec(),
            v.as_slice::<f32>().to_vec(),
        );
        for h in 0..heads {
            for part in [&q, &k, &v] {
                let base = (h * dim_head * cols) as usize;
                rebuilt.extend_from_slice(&part[base..base + (dim_head * cols) as usize]);
            }
        }
        assert_eq!(rebuilt, vals);
    }

    /// A NAIVE `chunk(3, dim=0)` split would produce a different partition. Pin the difference so
    /// a future "simplification" to the contiguous form fails loudly.
    #[test]
    fn interleaved_split_differs_from_a_contiguous_split() {
        let (heads, dim_head) = (2, 2);
        let vals: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let fused = Array::from_slice(&vals, &[12, 1]);
        let [q, _, _] = split_fused_qkv(&fused, heads, dim_head).unwrap();
        // Interleaved: head 0 rows 0..2, head 1 rows 6..8.
        assert_eq!(q.as_slice::<f32>(), &[0.0, 1.0, 6.0, 7.0]);
        // A contiguous split would have produced rows 0..4.
        assert_ne!(q.as_slice::<f32>(), &[0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn split_rejects_a_mismatched_leading_dim() {
        let fused = Array::from_slice(&[0.0f32; 10], &[10, 1]);
        assert!(split_fused_qkv(&fused, 2, 2).is_err());
    }

    #[test]
    fn load_errors_cleanly_on_a_missing_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("no-such-snapshot");
        match MiniMaxH3VideoVae::load(&missing, Dtype::Float32) {
            Err(e) => assert!(e.to_string().contains("config.json")),
            Ok(_) => panic!("expected a load error for a missing snapshot"),
        }
    }
}
