//! The NEO-Unify **vision embedder** — the candle port of `mlx-gen-sensenova`'s `vision.rs`
//! (`modeling_neo_vit.py`). The T2I slice only ever builds the **generation-path** instance
//! (`fm_modules.vision_model_mot_gen.embeddings`); the understanding-path tower (it2i/VQA) is Phase 6.
//!
//! For the 8B-MoT checkpoint the "vision tower" has no transformer blocks: a full-kernel
//! `patch_embedding` (Conv2d 3→`hidden_size`, kernel=stride=`patch_size`) + erf-GELU, an
//! **interleaved** 2D RoPE over the patch grid, then a `factor×factor`-strided `dense_embedding`
//! (Conv2d `hidden_size`→`llm_hidden_size`) that merges each 2×2 block of patches into one LLM token.
//!
//! The full-kernel `patch_embedding` is computed as a Linear over the flattened patch (an exact
//! equivalent). candle conv2d is **NCHW** (mlx is NHWC), so `dense_embedding` keeps the torch
//! `[llm, embed, f, f]` kernel layout unchanged and the roped patch grid is fed as `[1, embed, h, w]`.

use candle_gen::candle_core::{Result as CResult, Tensor};
use candle_gen::candle_nn::{Linear, Module, VarBuilder};
use candle_gen::Result;

use crate::config::NeoChatConfig;
use candle_gen::rope::{RopeLayout, RopeMath, RopeSpec};

/// The vision patch embedder's 1-D RoPE: complex-interleaved pairs, half tables `[n, part/2]` (f32,
/// the embedder's dtype), rotated in the activation dtype.
const VISION_ROPE: RopeSpec = RopeSpec::interleaved(RopeLayout::Bshd).with_math(RopeMath::Native);

/// A NEO vision embedder (Conv patch-embed → 2D RoPE → Conv patch-merge).
pub struct NeoVisionEmbedder {
    /// `patch_embedding` as a Linear over the flattened `[ch·ps·ps]` patch.
    patch: Linear,
    /// `dense_embedding` conv kernel `[llm, embed, factor, factor]` (torch/NCHW layout) + bias.
    dense_w: Tensor,
    dense_b: Tensor,
    embed_dim: usize,
    downsample_factor: usize,
    rope_theta: f32,
}

impl NeoVisionEmbedder {
    /// Build from a checkpoint. `prefix` = the embeddings namespace, e.g.
    /// `"fm_modules.vision_model_mot_gen.embeddings"` (generation path).
    pub fn from_weights(vb: &VarBuilder, cfg: &NeoChatConfig, prefix: &str) -> Result<Self> {
        let factor = (1.0 / cfg.vision.downsample_ratio).round() as usize;
        // All four load at **f32** regardless of the backbone's store dtype (sc-14249): the two conv
        // kernels are ~17.6 M parameters (~0.1% of the checkpoint) and stay dense in every shipped
        // tier, and the interleaved 2D RoPE below builds its tables in f32 — so widening here keeps
        // the whole vision path at one dtype instead of adding a boundary for no footprint gain.
        // patch_embedding: torch Conv2d weight [embed, ch, ps, ps] -> flat [embed, ch·ps·ps].
        let patch_w = crate::quant::get_f32(vb, &format!("{prefix}.patch_embedding.weight"))?;
        let (embed_dim, ch, ps, _) = patch_w.dims4()?;
        let patch_w = patch_w.reshape((embed_dim, ch * ps * ps))?;
        let patch_b = crate::quant::get_f32(vb, &format!("{prefix}.patch_embedding.bias"))?;
        let dense_w = crate::quant::get_f32(vb, &format!("{prefix}.dense_embedding.weight"))?; // [llm, embed, f, f]
        let dense_b = crate::quant::get_f32(vb, &format!("{prefix}.dense_embedding.bias"))?; // [llm]
        Ok(Self {
            patch: Linear::new(patch_w, Some(patch_b)),
            dense_w,
            dense_b,
            embed_dim,
            downsample_factor: factor,
            rope_theta: cfg.vision.rope_theta_vision,
        })
    }

    /// Embed `pixel_values` `[N, ch·ps·ps]` (row-major patch list, channel-first patch layout) for the
    /// images described by `grid` (each `(h, w)` patch-grid). Returns `[Σ (h/f)·(w/f), llm_hidden]`
    /// tokens in row-major order, concatenated across images.
    pub fn forward(&self, pixel_values: &Tensor, grid: &[(usize, usize)]) -> CResult<Tensor> {
        let embed = self.embed_dim;
        // patch_embedding (full-kernel conv == linear over the flat patch) + erf-GELU.
        let pe = __model_proof_output!(self.patch.forward(pixel_values))?.gelu_erf()?; // [N, embed]

        // Interleaved 2D RoPE (f32). Split the embedding in half: first half rotates by abs-x, second
        // by abs-y. The narrowed halves are made contiguous so the per-pair reshape inside the RoPE
        // helper is valid.
        let (abs_x, abs_y) = abs_positions(grid);
        let half = embed / 2;
        let h0 = pe.narrow(1, 0, half)?.contiguous()?;
        let h1 = pe.narrow(1, half, half)?.contiguous()?;
        let p1 = rope_1d_interleaved(&h0, &abs_x, self.rope_theta)?;
        let p2 = rope_1d_interleaved(&h1, &abs_y, self.rope_theta)?;
        let roped = Tensor::cat(&[&p1, &p2], 1)?; // [N, embed]

        // dense_embedding (factor×factor patch merge) per image, as an NCHW strided conv.
        let f = self.downsample_factor;
        let llm_dim = self.dense_b.dim(0)?;
        let dense_b = self.dense_b.reshape((1, llm_dim, 1, 1))?; // [1, llm, 1, 1]
        let mut outs: Vec<Tensor> = Vec::with_capacity(grid.len());
        let mut cur = 0usize;
        for &(h, w) in grid {
            let n = h * w;
            let block = roped
                .narrow(0, cur, n)? // [n, embed]
                .reshape((h, w, embed))?
                .permute((2, 0, 1))? // [embed, h, w]
                .unsqueeze(0)? // [1, embed, h, w]
                .contiguous()?;
            let merged = __model_proof_output!(block.conv2d(&self.dense_w, 0, f, 1, 1))?; // [1, llm, h/f, w/f]
            let merged = __model_proof_output!(merged.broadcast_add(&dense_b))?;
            let (_, llm, hf, wf) = merged.dims4()?;
            // [1, llm, h/f, w/f] -> [(h/f)·(w/f), llm], row-major.
            let tokens = merged
                .squeeze(0)?
                .permute((1, 2, 0))? // [h/f, w/f, llm]
                .contiguous()?
                .reshape((hf * wf, llm))?;
            outs.push(tokens);
            cur += n;
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        Tensor::cat(&refs, 0)
    }
}

/// Row-major patch coordinates `(abs_x, abs_y)` for the concatenated images: `abs_x = i % w`,
/// `abs_y = i / w` within each image.
fn abs_positions(grid: &[(usize, usize)]) -> (Vec<i32>, Vec<i32>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for &(h, w) in grid {
        for i in 0..(h * w) {
            xs.push((i % w) as i32);
            ys.push((i / w) as i32);
        }
    }
    (xs, ys)
}

/// Interleaved 1D RoPE on `x` `[N, part]` at integer `positions` (length N), base `theta`. Pairs
/// `(x[2j], x[2j+1])` rotate by `positions · theta^(-2j/part)`. Mirrors `apply_rotary_emb_1d`.
fn rope_1d_interleaved(x: &Tensor, positions: &[i32], theta: f32) -> CResult<Tensor> {
    let (n, part) = x.dims2()?;
    let half = part / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / part as f32))
        .collect();
    let mut freqs = Vec::with_capacity(n * half);
    for &p in positions {
        for &f in &inv_freq {
            freqs.push(p as f32 * f);
        }
    }
    let freqs = Tensor::from_vec(freqs, (n, half), x.device())?;
    // SW-6129: the rotation is the shared `candle_gen::rope::apply_rope` with `VISION_ROPE` over a
    // `[1, n, 1, part]` view (fused on CUDA; elsewhere the shared reference chain, op for op the former
    // `[n, half, 2]` chain — `rope_route_tests` pins the bit identity).
    let rotated = candle_gen::rope::apply_rope(
        &x.reshape((1, n, 1, part))?,
        &freqs.cos()?,
        &freqs.sin()?,
        &VISION_ROPE,
    )?;
    rotated.reshape((n, part))
}

#[cfg(test)]
mod rope_route_tests {
    use super::*;
    use candle_gen::candle_core::{DType, Device};

    /// The pre-SW-6129 `rope_1d_interleaved`, verbatim (the oracle).
    fn former(x: &Tensor, positions: &[i32], theta: f32) -> CResult<Tensor> {
        let (n, part) = x.dims2()?;
        let half = part / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| 1.0f32 / theta.powf((2 * j) as f32 / part as f32))
            .collect();
        let mut freqs = Vec::with_capacity(n * half);
        for &p in positions {
            for &f in &inv_freq {
                freqs.push(p as f32 * f);
            }
        }
        let freqs = Tensor::from_vec(freqs, (n, half), x.device())?;
        let cos = freqs.cos()?.reshape((n, half, 1))?;
        let sin = freqs.sin()?.reshape((n, half, 1))?;
        let xr = x.reshape((n, half, 2))?;
        let x1 = xr.narrow(2, 0, 1)?; // even lane [n, half, 1]
        let x2 = xr.narrow(2, 1, 1)?; // odd lane
        let rot1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rot2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        let out = Tensor::cat(&[&rot1, &rot2], 2)?; // [n, half, 2] interleaved
        out.reshape((n, part))
    }
    fn bits(t: &Tensor) -> Vec<u32> {
        let v = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        v.to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// SW-6129: the shared route is bit-identical to the former private chain on CPU (f32, the
    /// embedder's dtype) at the real abs-position layout of a 2-image grid.
    #[test]
    fn shared_rope_matches_the_former_private_chain() {
        let dev = Device::Cpu;
        let (xs, ys) = abs_positions(&[(4, 6), (3, 5)]);
        for (pos, part) in [(&xs, 64usize), (&ys, 32)] {
            let x = Tensor::randn(0f32, 1.0, (pos.len(), part), &dev).unwrap();
            let got = rope_1d_interleaved(&x, pos, 10000.0).unwrap();
            assert_eq!(
                bits(&got),
                bits(&former(&x, pos, 10000.0).unwrap()),
                "part {part}"
            );
        }
    }

    /// CUDA smoke (f32, the embedder's dtype; the bf16 cast is exercised by the kernel suite): the
    /// vision RoPE route takes the fused kernel, bit-identical to the shared reference (switch
    /// off) and to the former private chain.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_bf16_rope_route_is_fused_and_bit_identical() {
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::rope::ROPE_PATH;
        let dev = Device::new_cuda(0).unwrap();
        let _media = media_fused_policy_guard(Some(true));
        let _rope = ROPE_PATH.guard(Some(true));
        const CALLS: u64 = 1;
        let (xs, _) = abs_positions(&[(64, 64)]);
        let x = Tensor::randn(0f32, 1.0, (xs.len(), 64), &dev).unwrap();
        ROPE_PATH.set_enabled(Some(true));
        let start = ROPE_PATH.thread_counts();
        let on = rope_1d_interleaved(&x, &xs, 10000.0).unwrap();
        let c = ROPE_PATH.thread_counts().since(&start);
        assert_eq!((c.fused, c.reference_total()), (CALLS, 0), "{c}");
        ROPE_PATH.set_enabled(Some(false));
        let off = rope_1d_interleaved(&x, &xs, 10000.0).unwrap();
        ROPE_PATH.set_enabled(Some(true));
        assert_eq!(bits(&on), bits(&off));
        assert_eq!(bits(&on), bits(&former(&x, &xs, 10000.0).unwrap()));
    }
}
