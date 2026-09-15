//! EVA sub-LN `Attention`. Candle port of `eva_vit_model.py Attention(subln=True, rope=…)`.
//!
//! subln layout: separate `q_proj`/`k_proj`/`v_proj` (Linear, bias=False) plus standalone `q_bias` and
//! `v_bias` params (k has **no** bias), an `inner_attn_ln` (LayerNorm over all-head-dim) before `proj`.
//! RoPE is applied to the **patch** tokens of q/k only (the CLS token at index 0 is left unrotated).
//! Attention: scale q by `head_dim**-0.5`, softmax in f32 (the reference's explicit path; xformers
//! absent ⇒ `xattn=False`).

use candle_core::{Tensor, D};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{LayerNorm, Linear, Module};

use candle_gen::weights::Weights;
use candle_gen::Result as GenResult;

use crate::eva_clip::rope::VisionRope;
use crate::eva_clip::{join, layer_norm};

pub struct Attention {
    q_proj: Linear, // q_proj.weight + standalone q_bias
    k_proj: Linear, // k_proj.weight, NO bias
    v_proj: Linear, // v_proj.weight + standalone v_bias
    inner_ln: LayerNorm,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: usize,
        head_dim: usize,
    ) -> GenResult<Self> {
        let q_proj = Linear::new(
            w.require(&join(prefix, "q_proj.weight"))?,
            Some(w.require(&join(prefix, "q_bias"))?),
        );
        let k_proj = Linear::new(w.require(&join(prefix, "k_proj.weight"))?, None);
        let v_proj = Linear::new(
            w.require(&join(prefix, "v_proj.weight"))?,
            Some(w.require(&join(prefix, "v_bias"))?),
        );
        let proj = Linear::new(
            w.require(&join(prefix, "proj.weight"))?,
            Some(w.require(&join(prefix, "proj.bias"))?),
        );
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            inner_ln: layer_norm(w, &join(prefix, "inner_attn_ln"))?,
            proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// `x`: `[B, N, C]` (N = 1 CLS + grid² patches). `rope` is the shared block-invariant table.
    pub fn forward(&self, x: &Tensor, rope: &VisionRope) -> candle_core::Result<Tensor> {
        let (b, n, _c) = x.dims3()?;
        let (h, hd) = (self.num_heads, self.head_dim);

        // subln projections: q/v biased, k unbiased.
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // [B, N, C] → [B, heads, N, hd]
        let to_heads = |t: &Tensor| -> candle_core::Result<Tensor> {
            t.reshape((b, n, h, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = self.rope_patch_tokens(&to_heads(&q)?, rope)?;
        let k = self.rope_patch_tokens(&to_heads(&k)?, rope)?;
        let v = to_heads(&v)?;

        // SDPA (softmax in f32; the head dim is small, the tower is f32 anyway).
        let scores = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * self.scale)?;
        let probs = softmax_last_dim(&scores)?;
        let attn = probs.matmul(&v)?;

        let out = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, h * hd))?;
        let out = self.inner_ln.forward(&out)?;
        self.proj.forward(&out)
    }

    /// Apply RoPE to `x[:, :, 1:, :]` (patch tokens) only; the CLS token at index 0 is untouched.
    ///
    /// The result is forced **contiguous** (sc-16956). `Tensor::cat` on a non-zero axis takes a
    /// transposing slow path when its inputs are not all contiguous — and `rope.apply` returns a
    /// strided view — so the joined `[B, heads, N, hd]` tensor comes back with a transposed layout.
    /// Candle's CPU gemm accepts that; its **CUDA** matmul does not (`matmul is only supported for
    /// contiguous tensors`), so the whole EVA tower failed at the first attention on the one platform
    /// this crate exists for, while every CPU unit test passed. Materializing here fixes q and k at
    /// once and is numerically a no-op.
    fn rope_patch_tokens(&self, x: &Tensor, rope: &VisionRope) -> candle_core::Result<Tensor> {
        let n = x.dim(2)?;
        let cls = x.narrow(2, 0, 1)?;
        let pat = rope.apply(&x.narrow(2, 1, n - 1)?)?;
        Tensor::cat(&[&cls, &pat], 2)?.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    /// An `Attention` whose projections are never exercised. `rope_patch_tokens` reads no field of
    /// `self`, so building one this way keeps the guard below off the weight loader and lets it run
    /// on CPU in microseconds.
    fn bare_attention(dev: &Device, num_heads: usize, head_dim: usize) -> Attention {
        let c = num_heads * head_dim;
        let weight = Tensor::zeros((c, c), DType::F32, dev).unwrap();
        let vector = Tensor::zeros(c, DType::F32, dev).unwrap();
        let linear = || Linear::new(weight.clone(), None);
        Attention {
            q_proj: linear(),
            k_proj: linear(),
            v_proj: linear(),
            inner_ln: LayerNorm::new(vector.clone(), vector.clone(), 1e-6),
            proj: linear(),
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        }
    }

    /// `rope_patch_tokens` must hand back a **contiguous** `[B, heads, N, hd]` tensor (sc-16956).
    ///
    /// This is the CPU-runnable guard on a CUDA-only fix. `forward` feeds the result straight into
    /// `matmul`, and candle's CUDA matmul rejects a non-contiguous operand outright (`matmul is only
    /// supported for contiguous tensors`) where its CPU gemm silently accepts one — so without the
    /// `.contiguous()` the whole EVA tower fails at the first attention on the one platform this crate
    /// exists for, while every CPU test still passes. The real-weight row that would otherwise catch
    /// it is `#[ignore]`d behind five env vars, a GPU and `--features cuda --release`.
    #[test]
    fn rope_patch_tokens_returns_a_contiguous_tensor() {
        let dev = Device::Cpu;
        // grid² patch tokens + the unrotated CLS token at index 0.
        let (grid, heads, head_dim) = (2usize, 2usize, 4usize);
        let n = 1 + grid * grid;
        let rope = VisionRope::build(head_dim, grid, grid, 10_000.0, &dev).unwrap();
        let attn = bare_attention(&dev, heads, head_dim);

        // Contiguous going in — exactly what `forward`'s `to_heads` produces.
        let x = Tensor::arange(0f32, (heads * n * head_dim) as f32, &dev)
            .unwrap()
            .reshape((1, heads, n, head_dim))
            .unwrap();
        assert!(
            x.is_contiguous(),
            "the input must start contiguous or this row proves nothing about the join"
        );

        // Non-vacuity: the join itself really does come back strided, so the `.contiguous()` inside
        // `rope_patch_tokens` is load-bearing rather than defensive. `narrow` on axis 2 yields a view,
        // and `Tensor::cat` on a non-zero axis with a non-contiguous input takes a transposing path
        // (`cat0` then `transpose(0, dim)`). If candle ever makes that path contiguous this assertion
        // fires — which is the correct failure: the guard below would have gone vacuous.
        let cls = x.narrow(2, 0, 1).unwrap();
        let pat = rope.apply(&x.narrow(2, 1, n - 1).unwrap()).unwrap();
        assert!(
            !Tensor::cat(&[&cls, &pat], 2).unwrap().is_contiguous(),
            "the unmaterialized join is expected to be strided; this guard is vacuous if it is not"
        );

        let out = attn.rope_patch_tokens(&x, &rope).unwrap();
        assert_eq!(out.dims(), &[1, heads, n, head_dim]);
        assert!(
            out.is_contiguous(),
            "rope_patch_tokens must materialize its join — candle's CUDA matmul rejects a strided \
             operand, so a view here breaks the EVA tower on GPU while CPU tests stay green"
        );
    }
}
