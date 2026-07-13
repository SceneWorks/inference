//! Small candle neural-net helpers shared by the Cosmos DiT and the `AnimaTextConditioner` — the
//! candle twins of the `mlx_gen::nn` ops the MLX port uses. Kept composable (no fused-kernel
//! dispatch) so they match the MLX numerics on CPU and CUDA alike.

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::Linear;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::Result;

/// Bias-less dense `Linear` loaded from `{name}.weight` (shape read from disk via `get_unchecked`, so a
/// packed/dense weight of any `[out, in]` loads unchanged) — the candle twin of the MLX `lin` helper.
pub fn lin(vb: &VarBuilder, name: &str) -> Result<Linear> {
    Ok(Linear::new(
        vb.get_unchecked(&format!("{name}.weight"))?,
        None,
    ))
}

/// `Linear` **with** bias loaded from `{name}.weight` + `{name}.bias` (the conditioner MLP + out_proj).
pub fn lin_bias(vb: &VarBuilder, name: &str) -> Result<Linear> {
    Ok(Linear::new(
        vb.get_unchecked(&format!("{name}.weight"))?,
        Some(vb.get_unchecked(&format!("{name}.bias"))?),
    ))
}

/// Plain RMSNorm `w · x / sqrt(mean(x²) + eps)` over the last dim (matches `mlx_rs::fast::rms_norm`).
/// `w` broadcasts over the leading dims — used both for the per-head q/k norm (over `head_dim`) and
/// the feature-axis norms. Computed in the input dtype (upcast the caller's tensor to f32 first if
/// bf16 precision matters).
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.sqr()?.mean_keepdim(D::Minus1)?; // [.., 1]
    let denom = (mean + eps)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?.broadcast_mul(w)?)
}

/// LayerNorm with **no** affine params (`elementwise_affine=false`) over the last dim:
/// `(x − mean) / sqrt(var + eps)`, population variance — the candle twin of MLX
/// `layer_norm(x, None, None, eps)` used by the Cosmos adaLN norms.
pub fn layer_norm_no_affine(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = (var + eps)?.sqrt()?;
    Ok(xc.broadcast_div(&denom)?)
}

/// `(1 + scale) · norm + shift` (the diffusers AdaLN modulate, `one_matches_scale=true`). `scale`/
/// `shift` broadcast over the sequence axis.
pub fn modulate(norm: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    let scaled = norm.broadcast_mul(&(scale + 1.0)?)?;
    Ok(scaled.broadcast_add(shift)?)
}

/// HF **half-split** RoPE apply: `x·cos + rotate_half(x)·sin`, `rotate_half([x1,x2]) = [-x2, x1]`.
/// `x` is `[b, s, heads, head_dim]`; `cos`/`sin` are `[1, s, head_dim]` and broadcast over the head
/// axis (2). Cast the tables to `x`'s dtype so a bf16 DiT rotates bf16 q/k. Matches
/// `mlx_gen::nn::apply_text_rope` exactly (the Cosmos DiT self-attn + the conditioner both use it).
pub fn apply_rope_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.to_dtype(x.dtype())?.unsqueeze(2)?; // [1,s,1,hd]
    let sin = sin.to_dtype(x.dtype())?.unsqueeze(2)?;
    let d = x.dim(3)?;
    let half = d / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], 3)?; // rotate_half = cat(-x2, x1)
    Ok((x.broadcast_mul(&cos)? + rot.broadcast_mul(&sin)?)?)
}

/// Sinusoidal timestep embedding (diffusers `Timesteps`, `flip_sin_to_cos=True`,
/// `downscale_freq_shift=0`) → `[B, dim]` at `dtype`: `emb = t · exp(arange · −ln(max_period)/half)`,
/// result `cat([cos(emb), sin(emb)], -1)`. The candle twin of `mlx_gen::nn::timestep_sincos`. `sigma`
/// is `[B]`.
pub fn timestep_sincos(
    sigma: &Tensor,
    dim: usize,
    max_period: f64,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let b = sigma.dim(0)?;
    let half = dim / 2;
    let neg_log = -(max_period.ln());
    let denom = half as f64; // downscale_freq_shift = 0
    let freqs: Vec<f32> = (0..half)
        .map(|i| ((i as f64) * neg_log / denom).exp() as f32)
        .collect();
    let f = Tensor::from_vec(freqs, (1, half), device)?; // [1, half]
    let t = sigma.to_dtype(DType::F32)?.reshape((b, 1))?; // [B, 1]
    let emb = t.broadcast_mul(&f)?; // [B, half]
    let out = Tensor::cat(&[&emb.cos()?, &emb.sin()?], 1)?; // [B, dim]
    Ok(out.to_dtype(dtype)?)
}

/// Scaled-dot-product attention over `[B, H, S, D]` q/k/v with an optional additive mask, routed
/// through the shared i32-overflow-safe budgeted helper (sc-9116) with the fused `softmax_last_dim`
/// (inference-only — no backward needed here). `scale` multiplies the scores before softmax.
pub fn sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
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
