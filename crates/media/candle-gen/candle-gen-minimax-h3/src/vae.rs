//! The MiniMax-H3 video VAE decode path (`klvae.py::AutoencoderKLLegacy`).
//!
//! `decode` is three stages:
//!
//! 1. **de-normalize** the latent per channel — `z · latents_std + latents_mean` (24-vectors, not
//!    a scalar scale/shift);
//! 2. **chunk** it temporally ([`crate::chunking`]) and decode each chunk;
//! 3. **cross-fade** the `token_drop`-induced seams and trim the repeat padding.
//!
//! Only the decoder is ported (sc-17154). The 3-D causal CNN encoder, the DiT (sc-17155) and the
//! pipeline (sc-17156) are separate slices.

use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::loader::sorted_safetensors;
use candle_gen::{CandleError, Result, Weights};

use crate::blocks::blend;
use crate::chunking::{TemporalGeometry, TemporalPlan};
use crate::config::MiniMaxH3VaeConfig;
use crate::decoder::ViT3dDecoder;
use crate::nn::linear;
use crate::spatial_tiling::{stitch_tiles, SpatialTiling, TilePlan};

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

/// The video VAE's decode half.
#[derive(Debug, Clone)]
pub struct MiniMaxH3VideoVae {
    post_quant_w: Tensor,
    post_quant_b: Tensor,
    decoder: ViT3dDecoder,
    geometry: TemporalGeometry,
    latents_mean: Tensor,
    latents_std: Tensor,
    cfg: MiniMaxH3VaeConfig,
    dtype: DType,
    /// `use_tiling` plus the four `tile_sample_min_*` knobs. Defaults to the shipped 256/64 with
    /// tiling **on**, exactly as `AutoencoderKLMiniMaxH3.__init__` does.
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

        let decoder = ViT3dDecoder::from_weights(w, "decoder", cfg, dtype)?;
        let geometry = TemporalGeometry::new(cfg)?;

        Ok(Self {
            post_quant_w,
            post_quant_b,
            decoder,
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

    /// The tiling knobs in force.
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

    /// Every tensor name the decode path consumes — the exhaustive mapping. Any published `vae/`
    /// tensor outside this set plus the `encoder.*` / `quant_conv.*` encoder half would be silently
    /// ignored, which is the failure mode this list exists to make testable.
    pub fn tensor_names(cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        let mut v = vec![
            "post_quant_conv.weight".to_string(),
            "post_quant_conv.bias".to_string(),
        ];
        v.extend(ViT3dDecoder::names("decoder", cfg));
        v
    }

    /// Load from a snapshot root — reads `vae/config.json` and the `vae/` shards.
    ///
    /// Only the decode-half prefixes are materialized, so the 116-tensor conv encoder never lands
    /// on the device.
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

        let mut grid = Vec::with_capacity(rows.len());
        for (i, &y) in rows.starts.iter().enumerate() {
            let mut row = Vec::with_capacity(cols.len());
            for (j, &x) in cols.starts.iter().enumerate() {
                let tile = z
                    .narrow(3, y / ratio, rows.lengths[i] / ratio)?
                    .narrow(4, x / ratio, cols.lengths[j] / ratio)?
                    .contiguous()?;
                row.push(self.decode_clip_untiled(&tile)?);
            }
            grid.push(row);
        }
        // **The overlaps are used UNDIVIDED here.** The reference's `_encode_clip` divides them by
        // `spatial_compression_ratio` before stitching and `_decode_clip` does not: the plan is in
        // pixels either way, but a decoded tile comes back out in pixel space while an encoded one
        // comes back in latent space. Dividing here would blend over a 16×-too-narrow seam AND trim
        // 16× too little, so the stitched tensor comes out the wrong size: mutating the divide back
        // in at the 512×320 parity canvas returns [1, 3, 17, 752, 500] instead of
        // [1, 3, 17, 512, 320].
        stitch_tiles(&grid, &rows.overlaps, &cols.overlaps)
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

    #[test]
    fn tensor_names_match_the_published_vae_decode_half() {
        let cfg = MiniMaxH3VaeConfig::default();
        let names = MiniMaxH3VideoVae::tensor_names(&cfg);
        // The published `vae/` index has 703 tensors: 116 encoder + 2 quant_conv (the encode half,
        // not ported here) + 583 decoder + 2 post_quant_conv.
        assert_eq!(names.len(), 583 + 2);
        assert_eq!(703 - 116 - 2, names.len());
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len());
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
