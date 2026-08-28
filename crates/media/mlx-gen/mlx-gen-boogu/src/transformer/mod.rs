//! The Boogu mixed single/double-stream DiT (`BooguImageTransformer2DModel`) forward.
//!
//! Two entry points share one inner path: [`BooguTransformer::forward`] (text-to-image) and
//! [`BooguTransformer::forward_edit_multi`] (text+image-to-image, `N ∈ [1, 5]` references). Edit
//! VAE-encodes each reference image, patch-embeds it through `ref_image_patch_embedder` +
//! `image_index_embedding`, refines it in `ref_image_refiner`, and prepends those tokens —
//! `[ref₀; …; noise]` — to the image sequence (with the noise positions shifted by `max(ref_h, ref_w)`
//! in the unified RoPE).
//!
//! Text-to-image flow (the reference-image blocks stay dormant):
//! ```text
//!   time_caption_embed:  temb = TimestepEmbedder(sinusoid(t·scale));  caption = Linear(RMSNorm(instr))
//!   patchify(p=2, 16→64) → x_embedder                                 → img tokens  [1, Li, 3360]
//!   context_refiner ×2  (no modulation)        on instruct tokens     [1, Lt, 3360]
//!   noise_refiner   ×2  (modulated)            on img tokens
//!   double_stream   ×8  (joint instruct↔img attn + img self-attn)
//!   fuse → [instruct; img]                                            [1, Lt+Li, 3360]
//!   single_stream   ×32 (modulated)            on the joint sequence
//!   norm_out (LuminaLayerNormContinuous + temb) → Linear(3360→64)
//!   unpatchify(img tokens)                                            → velocity [1, 16, H, W]
//! ```
//!
//! Per-sample `B = 1`: true-CFG runs this twice (cond/uncond) rather than padding a batch, so every
//! attention is full/unmasked and numerically identical to the reference's per-sample slice.

mod block;
pub mod rope;

use std::cell::RefCell;

use mlx_rs::fast::{layer_norm, rms_norm};
use mlx_rs::ops::{concatenate_axis, cos, exp, multiply, sin, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::BooguConfig;
use crate::quant::lin;
use block::{DoubleBlock, ModBlock, PlainBlock};
use rope::RopeTables;

/// The Boogu DiT. Carries the text-to-image modules plus the reference-image conditioning path
/// (`ref_image_patch_embedder` + `ref_image_refiner` + `image_index_embedding`) the Edit (E7) forward
/// exercises; the T2I forward simply leaves those dormant.
pub struct BooguTransformer {
    cfg: BooguConfig,
    x_embedder: AdaptableLinear,
    ref_image_patch_embedder: AdaptableLinear,
    image_index_embedding: Array,
    caption_norm: Array,
    caption_linear: AdaptableLinear,
    time_lin1: AdaptableLinear,
    time_lin2: AdaptableLinear,
    context_refiner: Vec<PlainBlock>,
    noise_refiner: Vec<ModBlock>,
    ref_image_refiner: Vec<ModBlock>,
    double_stream: Vec<DoubleBlock>,
    single_stream: Vec<ModBlock>,
    norm_out_lin1: AdaptableLinear,
    norm_out_lin2: AdaptableLinear,
    /// Per-render RoPE-table cache. The tables depend only on the ordered request geometry, not on
    /// flow time or latent values, so the cond/uncond legs can reuse them across every denoise step.
    /// Four entries keep the usual two CFG geometries resident with bounded incidental headroom.
    rope_cache: RopeCache<RopeGeom, RopeTables>,
}

/// Every input that can change a Boogu RoPE table. Reference grids remain ordered because the
/// OmniGen2 `pe_shift` advances after each reference, so swapping two differently shaped sources is
/// a distinct layout even when their combined token count is unchanged.
type RopeGeom = (usize, usize, usize, Vec<(usize, usize)>);

/// Capacity shared with the Candle Boogu implementation. One slot thrashed between the two
/// true-CFG caption lengths; four keeps both legs plus a small number of incidental geometries.
const ROPE_CACHE_CAP: usize = 4;

/// A small FIFO cache for refcounted MLX table handles. FIFO is deliberate and deterministic: hits
/// do not reorder entries, and inserting past capacity evicts the oldest geometry. A `RefCell` is
/// sufficient because MLX generation is synchronous on one worker thread, matching the existing
/// Qwen-Image RoPE cache's interior-mutability contract.
struct RopeCache<K, V> {
    entries: RefCell<Vec<(K, V)>>,
    cap: usize,
}

impl<K: PartialEq, V: Clone> RopeCache<K, V> {
    fn new(cap: usize) -> Self {
        assert!(cap > 0, "RoPE cache capacity must be non-zero");
        Self {
            entries: RefCell::new(Vec::with_capacity(cap)),
            cap,
        }
    }

    /// Return the cached value for `key`, building and inserting it only on a miss.
    fn get_or_build(&self, key: K, build: impl FnOnce() -> V) -> V {
        let mut entries = self.entries.borrow_mut();
        if let Some((_, value)) = entries.iter().find(|(candidate, _)| *candidate == key) {
            mlx_gen::diagnostics::record_cache(
                "boogu::rope_tables",
                mlx_gen::diagnostics::CacheDisposition::Hit,
            );
            return value.clone();
        }

        mlx_gen::diagnostics::record_cache(
            "boogu::rope_tables",
            mlx_gen::diagnostics::CacheDisposition::Miss,
        );
        let value = build();
        if entries.len() == self.cap {
            entries.remove(0);
        }
        entries.push((key, value.clone()));
        value
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.borrow().len()
    }
}

impl BooguTransformer {
    /// Build from a loaded `transformer/` weight set (already validated by [`crate::convert`]).
    pub fn from_weights(w: &Weights, cfg: &BooguConfig) -> Result<Self> {
        let (heads, kv, hd) = (
            cfg.num_attention_heads as i32,
            cfg.num_kv_heads as i32,
            cfg.head_dim() as i32,
        );
        let eps = cfg.norm_eps;

        let plain = |name: String| PlainBlock::from_weights(w, &name, heads, kv, hd, eps);
        let mod_ = |name: String| ModBlock::from_weights(w, &name, heads, kv, hd, eps);
        let dbl = |name: String| DoubleBlock::from_weights(w, &name, heads, kv, hd, eps);

        Ok(Self {
            cfg: cfg.clone(),
            x_embedder: lin(w, "x_embedder", true)?,
            ref_image_patch_embedder: lin(w, "ref_image_patch_embedder", true)?,
            image_index_embedding: w.require("image_index_embedding")?.clone(),
            caption_norm: w
                .require("time_caption_embed.caption_embedder.0.weight")?
                .clone(),
            caption_linear: lin(w, "time_caption_embed.caption_embedder.1", true)?,
            time_lin1: lin(w, "time_caption_embed.timestep_embedder.linear_1", true)?,
            time_lin2: lin(w, "time_caption_embed.timestep_embedder.linear_2", true)?,
            context_refiner: (0..cfg.num_refiner_layers)
                .map(|i| plain(format!("context_refiner.{i}")))
                .collect::<Result<_>>()?,
            noise_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("noise_refiner.{i}")))
                .collect::<Result<_>>()?,
            ref_image_refiner: (0..cfg.num_refiner_layers)
                .map(|i| mod_(format!("ref_image_refiner.{i}")))
                .collect::<Result<_>>()?,
            double_stream: (0..cfg.num_double_stream_layers)
                .map(|i| dbl(format!("double_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            single_stream: (0..cfg.num_single_stream_layers())
                .map(|i| mod_(format!("single_stream_layers.{i}")))
                .collect::<Result<_>>()?,
            norm_out_lin1: lin(w, "norm_out.linear_1", true)?,
            norm_out_lin2: lin(w, "norm_out.linear_2", true)?,
            rope_cache: RopeCache::new(ROPE_CACHE_CAP),
        })
    }

    /// Build or reuse tables for the complete ordered T2I/edit geometry. Configuration fields are
    /// immutable for the transformer's lifetime, so the key only needs request-varying dimensions.
    fn rope_tables(
        &self,
        cap_len: usize,
        ht: usize,
        wt: usize,
        ref_grids: &[(usize, usize)],
    ) -> RopeTables {
        self.rope_cache
            .get_or_build((cap_len, ht, wt, ref_grids.to_vec()), || {
                if ref_grids.is_empty() {
                    RopeTables::build_t2i(
                        cap_len,
                        ht,
                        wt,
                        self.cfg.axes_dim_rope[0],
                        self.cfg.rope_theta,
                    )
                } else {
                    RopeTables::build_edit_multi(
                        cap_len,
                        ref_grids,
                        ht,
                        wt,
                        self.cfg.axes_dim_rope[0],
                        self.cfg.rope_theta,
                    )
                }
            })
    }

    /// Text-to-image velocity prediction.
    ///
    /// - `latent`: `[1, 16, H, W]` (H, W multiples of `patch_size`),
    /// - `timestep`: `[1]` f32 (raw, pre-scale),
    /// - `instruction_hidden`: `[1, L, 4096]` raw Qwen3-VL `last_hidden_state`,
    /// - `instruction_mask`: `[1, L]` (counts the valid leading tokens).
    ///
    /// Returns the velocity `[1, 16, H, W]`.
    pub fn forward(
        &self,
        latent: &Array,
        timestep: &Array,
        instruction_hidden: &Array,
        instruction_mask: &Array,
    ) -> Result<Array> {
        self.forward_inner(latent, &[], timestep, instruction_hidden, instruction_mask)
    }

    /// Edit velocity prediction with **multiple** reference latents (`N ∈ [1, 5]`, OmniGen2 lineage).
    /// Identical to [`Self::forward`] but with clean reference latents packed — each after its own
    /// `ref_image_patch_embedder` + `image_index_embedding[j]` + `ref_image_refiner` — *before* the
    /// noise tokens: `[ref₀; …; ref_{N-1}; noise]` (`[1, 16, rHⱼ, rWⱼ]` each). `N = 1` is the
    /// single-reference edit case.
    pub fn forward_edit_multi(
        &self,
        latent: &Array,
        ref_latents: &[Array],
        timestep: &Array,
        instruction_hidden: &Array,
        instruction_mask: &Array,
    ) -> Result<Array> {
        self.forward_inner(
            latent,
            ref_latents,
            timestep,
            instruction_hidden,
            instruction_mask,
        )
    }

    /// Shared T2I / edit forward. With an empty `ref_latents` this is the exact text-to-image path
    /// (no reference block, `combined_image == image`); with one or more it prepends each refined
    /// reference-image block and shifts the noise positions per the OmniGen2 unified RoPE.
    fn forward_inner(
        &self,
        latent: &Array,
        ref_latents: &[Array],
        timestep: &Array,
        instruction_hidden: &Array,
        instruction_mask: &Array,
    ) -> Result<Array> {
        let p = self.cfg.patch_size as i32;
        let (h, w) = (latent.shape()[2], latent.shape()[3]);
        let (ht, wt) = (h / p, w / p);
        let img_len = ht * wt;

        // Run in the model (weight) dtype — typically bf16 — to match the reference's compute path;
        // the dense Linear feeds activations to matmul as-is (no upcast).
        let dt = self.caption_norm.dtype();
        let latent = latent.as_dtype(dt)?;

        // Valid instruction length (B = 1): slice off any padding.
        let cap_len = sum(&instruction_mask.as_dtype(Dtype::Float32)?, false)?.item::<f32>() as i32;
        let instruct = slice_axis1(&instruction_hidden.as_dtype(dt)?, 0, cap_len)?;

        // Timestep + caption embedding.
        let temb = self.timestep_embed(timestep)?; // [1, 1, 1024]
        let caption = self.caption_linear.forward(&rms_norm(
            &instruct,
            &self.caption_norm,
            self.cfg.norm_eps,
        )?)?; // [1, cap, 3360]

        // Patchify the noise latent → target image tokens.
        let img = self.x_embedder.forward(&patchify(&latent, p)?)?; // [1, img_len, 3360]

        // Reference images (Edit): build the RoPE for the full `[instruct; ref₀; …; ref_{N-1}; noise]`
        // packing (no references ⇒ the text-to-image layout), then patch-embed each reference and add
        // its per-image index embedding (row `j` for the `j`-th reference).
        let ref_grids: Vec<(usize, usize)> = ref_latents
            .iter()
            .map(|rl| ((rl.shape()[2] / p) as usize, (rl.shape()[3] / p) as usize))
            .collect();
        let rope = self.rope_tables(cap_len as usize, ht as usize, wt as usize, &ref_grids);

        let mut ref_tokens: Vec<Array> = Vec::with_capacity(ref_latents.len());
        for (j, rl) in ref_latents.iter().enumerate() {
            let rl = rl.as_dtype(dt)?;
            let ref_t = self.ref_image_patch_embedder.forward(&patchify(&rl, p)?)?; // [1, ref_lenⱼ, 3360]
            let idx = self
                .image_index_embedding
                .take_axis(Array::from_slice(&[j as i32], &[1]), 0)?
                .as_dtype(dt)?
                .reshape(&[1, 1, self.cfg.hidden_size as i32])?;
            ref_tokens.push(mlx_rs::ops::add(&ref_t, &idx)?);
        }

        let (text_cos, text_sin) = rope.text()?;
        let (noise_cos, noise_sin) = rope.image()?; // target (noise) tokens only
        let (comb_cos, comb_sin) = rope.combined_image()?; // [ref₀…; noise] for img self-attn
        let (joint_cos, joint_sin) = rope.joint();

        // Context refinement (instruction stream).
        let mut instruct_h = caption;
        for blk in &self.context_refiner {
            instruct_h = blk.forward(&instruct_h, &text_cos, &text_sin)?;
        }

        // Noise refinement (target image stream).
        let mut img = img;
        for blk in &self.noise_refiner {
            img = blk.forward(&img, &noise_cos, &noise_sin, &temb)?;
        }

        // Reference refinement: each reference is refined **independently** (its own positions, no
        // cross-reference attention — mirroring the reference's per-image refiner rows), then all the
        // refined references are prepended to the noise tokens to form the combined image sequence
        // `[ref₀; …; ref_{N-1}; noise]` (Edit). T2I leaves the sequence as the noise tokens.
        let mut img = if ref_tokens.is_empty() {
            img
        } else {
            let mut seq: Vec<Array> = Vec::with_capacity(ref_tokens.len() + 1);
            for (j, mut ref_t) in ref_tokens.into_iter().enumerate() {
                let (ref_cos, ref_sin) = rope.ref_image_at(j)?;
                for blk in &self.ref_image_refiner {
                    ref_t = blk.forward(&ref_t, &ref_cos, &ref_sin, &temb)?;
                }
                seq.push(ref_t);
            }
            seq.push(img);
            let refs: Vec<&Array> = seq.iter().collect();
            concatenate_axis(&refs, 1)?
        };

        // Dual-stream blocks (joint instruct↔combined-image attn + combined-image self-attn).
        for blk in &self.double_stream {
            let (ni, nt) = blk.forward(
                &img,
                &instruct_h,
                &joint_cos,
                &joint_sin,
                &comb_cos,
                &comb_sin,
                &temb,
            )?;
            img = ni;
            instruct_h = nt;
        }

        // Fuse to the joint sequence, then single-stream blocks.
        let mut joint = concatenate_axis(&[&instruct_h, &img], 1)?; // [1, cap+ref+img, 3360]
        for blk in &self.single_stream {
            joint = blk.forward(&joint, &joint_cos, &joint_sin, &temb)?;
        }

        // Continuous-AdaLN output projection (LuminaLayerNormContinuous, eps 1e-6, no affine).
        let scale = self.norm_out_lin1.forward(&silu(&temb)?)?; // [1, 1, 3360]
        let normed = layer_norm(&joint, None, None, 1e-6)?;
        let normed = multiply(&normed, &mlx_rs::ops::add(&scale, Array::from_f32(1.0))?)?;
        let out = self.norm_out_lin2.forward(&normed)?; // [1, cap+ref+img, 64]

        // Unpatchify the trailing target-image tokens into the velocity (the reference tokens, when
        // present, are dropped — only the noise/target slice is the prediction).
        let total = out.shape()[1];
        let img_tokens = slice_axis1(&out, total - img_len, total)?;
        unpatchify(&img_tokens, ht, wt, p, self.cfg.out_channels as i32)
    }

    /// `Lumina2CombinedTimestepCaptionEmbedding` timestep branch:
    /// `sinusoid(timestep · timestep_scale, 256) → Linear → SiLU → Linear` → `[1, 1, 1024]`.
    fn timestep_embed(&self, timestep: &Array) -> Result<Array> {
        let scaled = multiply(
            &timestep.as_dtype(Dtype::Float32)?,
            scalar(self.cfg.timestep_scale),
        )?;
        // Sinusoid in f32 (the reference builds it in f32), then cast to the model dtype like the
        // reference's `timestep_proj.to(dtype)` before the embedder MLP.
        let proj = sinusoidal_timestep(&scaled, 256)?.as_dtype(self.caption_norm.dtype())?; // [1, 256]
        let t = self.time_lin1.forward(&proj)?;
        let t = silu(&t)?;
        let t = self.time_lin2.forward(&t)?; // [1, 1024]
        Ok(t.expand_dims(1)?) // [1, 1, 1024]
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.x_embedder
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.ref_image_patch_embedder
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.caption_linear
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.time_lin1
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.time_lin2
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        for b in &mut self.context_refiner {
            b.quantize(bits)?;
        }
        for b in &mut self.noise_refiner {
            b.quantize(bits)?;
        }
        for b in &mut self.ref_image_refiner {
            b.quantize(bits)?;
        }
        for b in &mut self.double_stream {
            b.quantize(bits)?;
        }
        for b in &mut self.single_stream {
            b.quantize(bits)?;
        }
        self.norm_out_lin1
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        self.norm_out_lin2
            .quantize(bits, Some(crate::quant::GROUP_SIZE))?;
        Ok(())
    }

    /// Drop every attention block back to three separate q/k/v projections — SC-18319's
    /// **fused-off baseline arm**, which the P6 matrix measures the packed path against.
    ///
    /// Bit-exact by construction (the packed matrix is the row-wise concatenation of the three
    /// bases, and `mlx_gen::qkv::FusedQkvProjection::unfuse` slices it back by the same row ranges);
    /// asserted at boogu's real widths in `transformer::block`'s tests. A quantized boogu is already
    /// split — the group-32 kv `out` refuses the pack — so this is a no-op there.
    pub fn unfuse_qkv(&mut self) -> Result<()> {
        for b in &mut self.context_refiner {
            b.unfuse_qkv()?;
        }
        for b in &mut self.noise_refiner {
            b.unfuse_qkv()?;
        }
        for b in &mut self.ref_image_refiner {
            b.unfuse_qkv()?;
        }
        for b in &mut self.double_stream {
            b.unfuse_qkv()?;
        }
        for b in &mut self.single_stream {
            b.unfuse_qkv()?;
        }
        Ok(())
    }

    /// SC-18319 — whether **every** attention block currently holds one packed q/k/v matrix.
    ///
    /// The P6 matrix reads this as the *activation receipt* for the fused arm rather than inferring
    /// it from a flag or from timing: a dense boogu **loaded under an opt-in** reports `true`, and a
    /// Q4/Q8 boogu reports `false` because the group-32 kv `out = 840` refuses the pack (see
    /// `mlx_gen::qkv::NoFusion::OutFeaturesNotGroupAligned`). `false` after
    /// [`unfuse_qkv`](Self::unfuse_qkv) is the baseline arm having actually taken effect.
    ///
    /// Fusion is opt-in (`mlx_gen::qkv::set_fused_qkv` defaults to off), so a transformer loaded
    /// without an explicit `FusedQkvGuard::set(true)` in scope reports `false` here for that reason
    /// alone — `mlx_gen::qkv::NoFusion::Disabled` rather than the group-32 rule. The block's own
    /// `FusedQkvProjection::refusal` is what distinguishes the two causes, and `transformer::block`'s
    /// tests assert both.
    pub fn qkv_fusion_engaged(&self) -> bool {
        self.context_refiner.iter().all(|b| b.qkv_fusion_engaged())
            && self.noise_refiner.iter().all(|b| b.qkv_fusion_engaged())
            && self
                .ref_image_refiner
                .iter()
                .all(|b| b.qkv_fusion_engaged())
            && self.double_stream.iter().all(|b| b.qkv_fusion_engaged())
            && self.single_stream.iter().all(|b| b.qkv_fusion_engaged())
    }
}

// ── Shared helpers ──────────────────────────────────────────────────────────────────────────

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Slice `[b, L, ...]` along the sequence axis (axis 1) to `[start, end)`.
pub(crate) fn slice_axis1(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), 1)?)
}

// The GQA `repeat_interleave` that used to live here is now `mlx_gen::qkv::repeat_kv` (SC-18319,
// knob 10) — the identical `expand_dims → broadcast_to → reshape`, parameterized by which of the two
// layouts the stream is in.

/// diffusers `get_timestep_embedding(x, dim, flip_sin_to_cos=True, downscale_freq_shift=0,
/// max_period=10000)`: `freq_i = 10000^(−i/half)`, `emb = x·freq`, `concat([cos, sin], -1)` (cos
/// first). `x`: `[N]` → `[N, dim]`. `ln(10000)` in f64 to match `math.log` rounding.
fn sinusoidal_timestep(x: &Array, dim: i32) -> Result<Array> {
    let half = dim / 2;
    let arange: Vec<f32> = (0..half).map(|i| i as f32).collect();
    let arange = Array::from_slice(&arange, &[half]);
    let neg_ln = -(10000f64.ln()) as f32;
    let exponent = mlx_rs::ops::divide(&multiply(&arange, scalar(neg_ln))?, scalar(half as f32))?;
    let freqs = exp(&exponent)?; // [half]
    let axis = x.shape().len() as i32;
    let emb = multiply(&x.expand_dims(axis)?, &freqs)?; // [N, half]
    Ok(concatenate_axis(&[&cos(&emb)?, &sin(&emb)?], -1)?)
}

/// `c (h p1) (w p2) -> (h w) (p1 p2 c)` with batch: `[1, C, H, W] → [1, (H/p)(W/p), p·p·C]`.
fn patchify(latent: &Array, p: i32) -> Result<Array> {
    let sh = latent.shape();
    let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
    let (ht, wt) = (h / p, w / p);
    let x = latent.reshape(&[b, c, ht, p, wt, p])?; // B, C, h, p1, w, p2
    let x = x.transpose_axes(&[0, 2, 4, 3, 5, 1])?; // B, h, w, p1, p2, C
    Ok(x.reshape(&[b, ht * wt, p * p * c])?)
}

/// `(h w) (p1 p2 c) -> c (h p1) (w p2)` with batch: `[1, (h)(w), p·p·C] → [1, C, h·p, w·p]`.
fn unpatchify(tokens: &Array, ht: i32, wt: i32, p: i32, c: i32) -> Result<Array> {
    let b = tokens.shape()[0];
    let x = tokens.reshape(&[b, ht, wt, p, p, c])?; // B, h, w, p1, p2, C
    let x = x.transpose_axes(&[0, 5, 1, 3, 2, 4])?; // B, C, h, p1, w, p2
    Ok(x.reshape(&[b, c, ht * p, wt * p])?)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use mlx_gen::diagnostics::{self, CacheDisposition, DiagnosticCounter};

    use super::*;

    const AXES_DIM: usize = 40;
    const THETA: f32 = 10_000.0;

    fn flat_joint(tables: &RopeTables) -> Vec<f32> {
        let (cos, sin) = tables.joint();
        let mut flat = cos.as_slice::<f32>().to_vec();
        flat.extend_from_slice(sin.as_slice::<f32>());
        flat
    }

    /// sc-21687: same-geometry forwards reuse one table set, changed target geometry and changed
    /// ordered reference layouts miss, and every cached value remains byte-identical to a fresh
    /// construction with the expected sequence shapes.
    #[test]
    fn rope_cache_hits_misses_and_preserves_table_parity() {
        let cache: RopeCache<RopeGeom, RopeTables> = RopeCache::new(ROPE_CACHE_CAP);
        let builds = Cell::new(0usize);
        let scope = diagnostics::begin_observed_request("sc-21687", "boogu").unwrap();

        let t2i_key = (3, 2, 4, Vec::new());
        let first = cache.get_or_build(t2i_key.clone(), || {
            builds.set(builds.get() + 1);
            RopeTables::build_t2i(3, 2, 4, AXES_DIM, THETA)
        });
        let hit = cache.get_or_build(t2i_key, || {
            builds.set(builds.get() + 1);
            RopeTables::build_t2i(3, 2, 4, AXES_DIM, THETA)
        });
        assert_eq!(builds.get(), 1, "repeated geometry must not rebuild");
        assert_eq!(first.joint().0.shape(), &[1, 11, 60]);
        assert_eq!(flat_joint(&first), flat_joint(&hit));

        let changed_target = cache.get_or_build((3, 2, 5, Vec::new()), || {
            builds.set(builds.get() + 1);
            RopeTables::build_t2i(3, 2, 5, AXES_DIM, THETA)
        });
        assert_eq!(changed_target.joint().0.shape(), &[1, 13, 60]);

        // Same combined reference-token count and target, but a different ordered layout. The
        // `pe_shift` walk makes these distinct tables and the cache key must not collapse them.
        let refs_a = vec![(2, 3), (1, 4)];
        let refs_b = vec![(1, 4), (2, 3)];
        let edit_a = cache.get_or_build((3, 4, 4, refs_a.clone()), || {
            builds.set(builds.get() + 1);
            RopeTables::build_edit_multi(3, &refs_a, 4, 4, AXES_DIM, THETA)
        });
        let edit_b = cache.get_or_build((3, 4, 4, refs_b.clone()), || {
            builds.set(builds.get() + 1);
            RopeTables::build_edit_multi(3, &refs_b, 4, 4, AXES_DIM, THETA)
        });
        assert_eq!(
            builds.get(),
            4,
            "every distinct geometry builds exactly once"
        );
        assert_eq!(edit_a.joint().0.shape(), &[1, 29, 60]);
        assert_eq!(edit_b.joint().0.shape(), &[1, 29, 60]);
        assert_eq!(edit_a.ref_image_at(0).unwrap().0.shape(), &[1, 6, 60]);
        assert_eq!(edit_a.ref_image_at(1).unwrap().0.shape(), &[1, 4, 60]);
        assert_ne!(flat_joint(&edit_a), flat_joint(&edit_b));

        let report = scope.finish();
        assert!(report.counters.contains(&DiagnosticCounter::Cache {
            site: "boogu::rope_tables",
            disposition: CacheDisposition::Hit,
            count: 1,
        }));
        assert!(report.counters.contains(&DiagnosticCounter::Cache {
            site: "boogu::rope_tables",
            disposition: CacheDisposition::Miss,
            count: 4,
        }));
    }

    /// sc-21687: capacity is a hard bound and overflow evicts insertion order, not access order.
    #[test]
    fn rope_cache_capacity_four_evicts_oldest_deterministically() {
        let cache: RopeCache<usize, usize> = RopeCache::new(ROPE_CACHE_CAP);
        let builds = Cell::new(0usize);
        let lookup = |key| {
            cache.get_or_build(key, || {
                builds.set(builds.get() + 1);
                key * 10
            })
        };

        for key in 1..=4 {
            assert_eq!(lookup(key), key * 10);
        }
        assert_eq!(cache.len(), ROPE_CACHE_CAP);
        assert_eq!(builds.get(), 4);

        assert_eq!(lookup(2), 20); // hit does not reorder FIFO entries
        assert_eq!(lookup(5), 50); // evicts oldest key 1
        assert_eq!(cache.len(), ROPE_CACHE_CAP);
        assert_eq!(builds.get(), 5);
        assert_eq!(lookup(2), 20, "key 2 must remain cached");
        assert_eq!(lookup(1), 10, "evicted oldest key must rebuild");
        assert_eq!(cache.len(), ROPE_CACHE_CAP);
        assert_eq!(builds.get(), 6);
    }
}
