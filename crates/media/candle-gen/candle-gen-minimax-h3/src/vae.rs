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
        })
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

    /// `post_quant_conv` then the ViT decoder — the reference's `decode`, on ONE chunk.
    pub fn decode_clip(&self, z: &Tensor) -> Result<Tensor> {
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
