//! Shared rotary / sinusoidal position helpers (sc-27286): the byte-identical bodies the Lens,
//! Qwen-Image, FLUX.2 and Wan transformers each carried. Hoisting them changes no numerics — every
//! caller keeps its own axis layout, table build and per-render cache.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::rotary_emb::rope_i;

/// Apply **interleaved** RoPE (lanes `2i`/`2i+1` are the real/imag pair) to `x` `[B, H, S, head_dim]`
/// with `cos`/`sin` `[S, head_dim/2]`, via candle's fused `rope_i`. The rotation runs in f32 (the
/// tables are f32) and the result is cast back to `x`'s dtype.
pub fn apply_interleaved_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let dtype = x.dtype();
    let xf = x.to_dtype(DType::F32)?.contiguous()?;
    rope_i(&xf, cos, sin)?.to_dtype(dtype)
}

/// Sinusoidal timestep embedding `[1, dim]`, `[cos(args) | sin(args)]` (diffusers
/// `flip_sin_to_cos = True`, `downscale_freq_shift = 0`), base 10000:
/// `args[k] = t · scale · 10000^{-k/half}`, computed in f32 in exactly that order. `scale` is `1000.0`
/// for the providers that feed a raw sigma (diffusers `Timesteps(scale=1000)`) and `1.0` for a caller
/// that pre-scales its timestep (multiplying by `1.0` is exact, so that case is bit-identical to
/// `t · freq`).
pub fn sinusoidal_timestep_embedding(
    t: f32,
    dim: usize,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let half = dim / 2;
    let ln = 10000f32.ln();
    let mut cos = vec![0f32; half];
    let mut sin = vec![0f32; half];
    for k in 0..half {
        let freq = (-ln * k as f32 / half as f32).exp();
        let arg = t * scale * freq;
        cos[k] = arg.cos();
        sin[k] = arg.sin();
    }
    let cos = Tensor::from_vec(cos, (1, half), device)?;
    let sin = Tensor::from_vec(sin, (1, half), device)?;
    Tensor::cat(&[&cos, &sin], D::Minus1)
}
