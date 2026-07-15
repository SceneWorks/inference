//! Small candle neural-net helpers shared by the AsymmDiT ([`crate::transformer`]) and the AsymmVAE
//! ([`crate::vae`]) — the candle twins of the `mlx_gen::nn` ops the MLX Mochi port uses. Kept
//! composable (no fused-kernel dispatch) so they match the MLX numerics on CPU and CUDA alike. Every
//! norm computes in **f32** (upcasting a bf16 input) then casts back, mirroring the reference's f32
//! norm/attention islands over the bf16 DiT.

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::Result;

/// SiLU / swish: `x · sigmoid(x)`, in the input dtype.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok((x * sigmoid(x)?)?)
}

/// `y = x · Wᵀ` for a stored `[out, in]` weight, no bias, over the **last** axis of `x` (any leading
/// dims). Flattens the leading dims to one matmul then restores them — candle's `Linear::forward` only
/// special-cases ranks ≤ 4, so this covers the rank-5 NCTHW→channel-last projections the VAE uses.
///
/// The weight is upcast to `x`'s dtype at use. The AsymmDiT stores weights at bf16 but runs **f32**
/// activations (the MLX parity regime), so this lets an f32 activation stream flow through bf16-stored
/// projections with a transient per-matmul f32 weight view (freed after the matmul); when weight and
/// activation already share a dtype (the f32 VAE), it is a no-op.
pub fn linear_nb(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let in_dim = *dims.last().expect("linear_nb: x has no axes");
    let out_dim = w.dim(0)?;
    let rows = x.elem_count() / in_dim;
    let flat = x.reshape((rows, in_dim))?;
    let w = w.to_dtype(x.dtype())?;
    let y = flat.matmul(&w.t()?)?;
    let mut out_dims = dims;
    *out_dims.last_mut().unwrap() = out_dim;
    Ok(y.reshape(out_dims)?)
}

/// `y = x · Wᵀ + b` over the last axis (see [`linear_nb`]). `b` is `[out]`, upcast to `x`'s dtype.
pub fn linear_b(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(linear_nb(x, w)?.broadcast_add(&b.to_dtype(x.dtype())?)?)
}

/// **Weightless** RMS norm over the last axis, computed in f32 and returned in f32
/// (`RMSNorm(0, eps, False)` — Mochi's `MochiRMSNormZero.norm` / `MochiModulatedRMSNorm.norm`):
/// `x / sqrt(mean(x²) + eps)`.
pub fn rms_weightless(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let ms = xf.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xf.broadcast_div(&(ms + eps)?.sqrt()?)?)
}

/// **Weighted** RMS norm over the last axis in f32 (`MochiRMSNorm(dim_head, eps, True)` — the per-head
/// `qk_norm`). `weight` is `[head_dim]` (or any shape broadcastable over the leading dims), applied in
/// f32; the result stays f32.
pub fn rms_weighted(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let normed = rms_weightless(x, eps)?;
    Ok(normed.broadcast_mul(&weight.to_dtype(DType::F32)?)?)
}

/// LayerNorm with **no** affine params (`elementwise_affine=false`) over the last axis:
/// `(x − mean) / sqrt(var + eps)` (population variance) — the candle twin of MLX
/// `layer_norm(x, None, None, eps)` used by the DiT's final `AdaLayerNormContinuous`. Computed in f32.
pub fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let mean = xf.mean_keepdim(D::Minus1)?;
    let xc = xf.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xc.broadcast_div(&(var + eps)?.sqrt()?)?)
}

/// Sinusoidal timestep embedding (diffusers `Timesteps`, `flip_sin_to_cos=True`,
/// `downscale_freq_shift=0`) → `[B, dim]` f32: `emb = t · exp(arange · −ln(max_period)/half)`, result
/// `cat([cos(emb), sin(emb)], -1)`. The candle twin of `mlx_gen::nn::timestep_sincos`. `t` is `[B]`.
pub fn timestep_sincos(t: &Tensor, dim: usize, max_period: f64, device: &Device) -> Result<Tensor> {
    let b = t.dim(0)?;
    let half = dim / 2;
    let neg_log = -(max_period.ln());
    let denom = half as f64; // downscale_freq_shift = 0
    let freqs: Vec<f32> = (0..half)
        .map(|i| ((i as f64) * neg_log / denom).exp() as f32)
        .collect();
    let f = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let t = t.to_dtype(DType::F32)?.reshape((b, 1))?; // [B, 1]
    let emb = t.broadcast_mul(&f)?; // [B, half]
    Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1).map_err(Into::into) // [B, dim]
}

/// Scaled-dot-product attention over `[B, H, S, D]` q/k/v with an optional additive mask, routed
/// through the shared i32-overflow-safe budgeted helper (sc-9116) with the fused `softmax_last_dim`.
/// `scale` multiplies the scores before softmax. The additive `mask` broadcasts over `[B, H, Sq, Sk]`.
pub fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, mask: Option<&Tensor>) -> Result<Tensor> {
    Ok(candle_gen::sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        mask,
        candle_gen::candle_nn::ops::softmax_last_dim,
        candle_gen::ATTN_SCORES_BUDGET,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn weightless_rms_normalizes_to_unit_ms() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 0.0], (1, 4), &dev).unwrap();
        let y = rms_weightless(&x, 1e-6).unwrap();
        // rms = sqrt(mean(9,16,0,0)) = sqrt(6.25) = 2.5; y = x/2.5 → [1.2, 1.6, 0, 0].
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!((v[0] - 1.2).abs() < 1e-4 && (v[1] - 1.6).abs() < 1e-4);
    }

    #[test]
    fn timestep_sincos_pos_zero_is_one_then_zero() {
        // At t=0 every freq·t = 0 → cos=1, sin=0: first half all 1, second half all 0.
        let dev = Device::Cpu;
        let t = Tensor::from_vec(vec![0.0f32], 1, &dev).unwrap();
        let e = timestep_sincos(&t, 8, 10000.0, &dev).unwrap();
        let v = e.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(&v[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&v[4..], &[0.0, 0.0, 0.0, 0.0]);
    }
}
