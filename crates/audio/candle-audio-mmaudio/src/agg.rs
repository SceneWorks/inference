//! Spatial aggregation head — MMAudio's `SpatialTransformerEncoderLayer` (a `BaseEncoderLayer`,
//! itself a pre-norm `nn.TransformerEncoderLayer` with a prepended learnable CLS token whose output
//! is returned). It pools the 14×14 spatial grid of every temporal frame into one 768-d vector.
//!
//! PyTorch stores the MHA projection as fused `in_proj_weight (3D, D)` / `in_proj_bias (3D)` plus an
//! `out_proj` Linear; we slice the fused tensors into per-Q/K/V weights. `norm_first=True` and
//! `activation=nn.GELU()` (erf) are transcribed from MMAudio's `transf_enc_layer_kwargs`.

use candle_audio::candle_core::{Result as CResult, Tensor, D};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, Module, VarBuilder};

use crate::config;
use crate::preprocess::softmax_last;

/// A pre-norm transformer encoder layer with a CLS token, applied per frame to pool spatial tokens.
pub struct SpatialAggLayer {
    cls_token: Tensor,      // (1, 1, D)
    in_proj_weight: Tensor, // (3D, D)
    in_proj_bias: Tensor,   // (3D,)
    out_proj: Linear,       // D → D
    linear1: Linear,        // D → 4D
    linear2: Linear,        // 4D → D
    norm1: LayerNorm,       // pre-attn norm
    norm2: LayerNorm,       // pre-ff norm
}

impl SpatialAggLayer {
    pub fn load(vb: VarBuilder) -> CResult<Self> {
        let d = config::EMBED_DIM;
        Ok(Self {
            cls_token: vb.get((1, 1, d), "cls_token")?,
            in_proj_weight: vb.get((3 * d, d), "self_attn.in_proj_weight")?,
            in_proj_bias: vb.get(3 * d, "self_attn.in_proj_bias")?,
            out_proj: linear(d, d, vb.pp("self_attn").pp("out_proj"))?,
            linear1: linear(d, config::MLP_HIDDEN, vb.pp("linear1"))?,
            linear2: linear(config::MLP_HIDDEN, d, vb.pp("linear2"))?,
            norm1: layer_norm(d, config::LN_EPS, vb.pp("norm1"))?,
            norm2: layer_norm(d, config::LN_EPS, vb.pp("norm2"))?,
        })
    }

    /// Multi-head self-attention over `(B, N, D)` using the fused in-projection.
    fn self_attn(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, n, d) = x.dims3()?;
        let h = config::NUM_HEADS;
        let hd = config::HEAD_DIM;
        // Fused projection then split rows into Q/K/V.
        let proj = x
            .broadcast_matmul(&self.in_proj_weight.t()?)?
            .broadcast_add(&self.in_proj_bias)?; // (B, N, 3D)
        let q = proj.narrow(2, 0, d)?;
        let k = proj.narrow(2, d, d)?;
        let v = proj.narrow(2, 2 * d, d)?;
        let shape = |t: &Tensor| -> CResult<Tensor> {
            t.reshape((b, n, h, hd))?.transpose(1, 2)?.contiguous()
        };
        let q = shape(&q)?;
        let k = shape(&k)?;
        let v = shape(&v)?;
        let scale = (hd as f64).powf(-0.5);
        let sim = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?.contiguous()?)? * scale)?;
        let attn =
            softmax_last(&sim).map_err(|e| candle_audio::candle_core::Error::Msg(e.to_string()))?;
        let ctx = attn.matmul(&v)?; // (B, h, N, hd)
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, n, d))?;
        self.out_proj.forward(&ctx)
    }

    fn ff(&self, x: &Tensor) -> CResult<Tensor> {
        let x = self.linear1.forward(x)?.gelu_erf()?;
        self.linear2.forward(&x)
    }

    /// `x`: `(B, seq, D)` spatial tokens. Prepends CLS, runs the pre-norm encoder layer, returns the
    /// CLS output `(B, D)`.
    pub fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, _seq, d) = x.dims3()?;
        let cls = self.cls_token.broadcast_as((b, 1, d))?;
        let x = Tensor::cat(&[&cls, x], 1)?; // (B, 1+seq, D)
                                             // norm_first: x = x + sa(norm1(x)); x = x + ff(norm2(x))
        let x = (&x + self.self_attn(&self.norm1.forward(&x)?)?)?;
        let x = (&x + self.ff(&self.norm2.forward(&x)?)?)?;
        // Return the CLS token representation.
        x.narrow(1, 0, 1)?.squeeze(1)
    }
}
