//! Small candle neural-net helpers shared by the Cosmos DiT and the `AnimaTextConditioner` — the
//! candle twins of the `mlx_gen::nn` ops the MLX port uses. Kept composable (no fused-kernel
//! dispatch) so they match the MLX numerics on CPU and CUDA alike.

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::Linear;
use candle_gen::candle_nn::VarBuilder;
use candle_gen::rope::{RopeLayout, RopeMath, RopeSpec};
use candle_gen::Result;

/// Bias-less dense `Linear` loaded from `{name}.weight` (shape read from disk via `get_unchecked`, so a
/// packed/dense weight of any `[out, in]` loads unchanged) — the candle twin of the MLX `lin` helper.
pub fn lin(vb: &VarBuilder, name: &str) -> Result<Linear> {
    Ok(Linear::from_var_builder(&vb.pp(name), false)?)
}

/// `Linear` **with** bias loaded from `{name}.weight` + `{name}.bias` (the conditioner MLP + out_proj).
pub fn lin_bias(vb: &VarBuilder, name: &str) -> Result<Linear> {
    Ok(Linear::from_var_builder(&vb.pp(name), true)?)
}

/// Plain RMSNorm `w · x / sqrt(mean(x²) + eps)` over the last dim (matches `mlx_rs::fast::rms_norm`).
/// `w` broadcasts over the leading dims — used both for the per-head q/k norm (over `head_dim`) and
/// the feature-axis norms. Reductions use f32; normalized values round before the weight.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    candle_gen::train::profile::cohort_reuse("anima:shared-differentiable-RMS");
    Ok(candle_gen::train::norm::rms_norm_cast_before_weight(
        x, w, eps,
    )?)
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
///
/// The shared composable [`candle_gen::modulate::modulate`] (SW-6129) — the exact ops this copy ran.
pub fn modulate(norm: &Tensor, scale: &Tensor, shift: &Tensor) -> Result<Tensor> {
    Ok(candle_gen::modulate::modulate(norm, scale, shift)?)
}

/// Anima's RoPE convention: HF half-split pairs on `[b, s, heads, head_dim]`, full-width tables, the
/// rotation in `x`'s dtype (the tables cast down).
pub const ANIMA_ROPE: RopeSpec = RopeSpec::split_half(RopeLayout::Bshd).with_math(RopeMath::Native);

/// HF **half-split** RoPE apply: `x·cos + rotate_half(x)·sin`, `rotate_half([x1,x2]) = [-x2, x1]`.
/// `x` is `[b, s, heads, head_dim]`; `cos`/`sin` are `[1, s, head_dim]` and broadcast over the head
/// axis (2). Cast the tables to `x`'s dtype so a bf16 DiT rotates bf16 q/k. Matches
/// `mlx_gen::nn::apply_text_rope` exactly (the Cosmos DiT self-attn + the conditioner both use it).
///
/// SW-6129: the shared [`candle_gen::rope::apply_rope`] with [`ANIMA_ROPE`] — the fused strided kernel
/// on CUDA, and everywhere else the shared reference chain, op for op this function's former body
/// (`rope_route_tests` pins the bit identity).
pub fn apply_rope_half(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    Ok(candle_gen::rope::apply_rope(x, cos, sin, &ANIMA_ROPE)?)
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
    Ok(candle_gen::fused_attention::training_sdpa_budgeted_bhsd(
        q,
        k,
        v,
        scale,
        mask,
        |scores| {
            if scores.track_op() {
                candle_gen::candle_nn::ops::softmax(scores, D::Minus1)
            } else {
                candle_gen::candle_nn::ops::softmax_last_dim(scores)
            }
        },
        candle_gen::ATTN_SCORES_BUDGET,
    )?)
}

#[cfg(test)]
mod rope_route_tests {
    use super::*;

    /// The pre-SW-6129 `apply_rope_half`, verbatim (the oracle).
    fn former(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let cos = cos.to_dtype(x.dtype()).unwrap().unsqueeze(2).unwrap();
        let sin = sin.to_dtype(x.dtype()).unwrap().unsqueeze(2).unwrap();
        let d = x.dim(3).unwrap();
        let half = d / 2;
        let x1 = x.narrow(3, 0, half).unwrap();
        let x2 = x.narrow(3, half, half).unwrap();
        let rot = Tensor::cat(&[&x2.neg().unwrap(), &x1], 3).unwrap();
        (x.broadcast_mul(&cos).unwrap() + rot.broadcast_mul(&sin).unwrap()).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        let v = t.to_dtype(DType::F32).unwrap().flatten_all().unwrap();
        v.to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    /// The conditioner's `text_rope` table (`[1, s, 64]`) and a 3-axis-like random full-width table.
    fn tables(s: usize, hd: usize, dev: &Device) -> Vec<(Tensor, Tensor)> {
        let (c, sn) = crate::rope::text_rope(s, hd, 10_000.0, dev).unwrap();
        let a = (Tensor::randn(0f32, 1.0, (1, s, hd), dev).unwrap() * 40.0).unwrap();
        vec![(c, sn), (a.cos().unwrap(), a.sin().unwrap())]
    }

    /// The shared route is bit-identical to the former private chain on CPU, f32 / bf16 / f16.
    #[test]
    fn shared_rope_matches_the_former_private_chain() {
        let dev = Device::Cpu;
        let (s, h, hd) = (19, 3, 64);
        for (cos, sin) in tables(s, hd, &dev) {
            let x32 = Tensor::randn(0f32, 1.0, (2, s, h, hd), &dev).unwrap();
            for dt in [DType::F32, DType::BF16, DType::F16] {
                let x = x32.to_dtype(dt).unwrap();
                let got = apply_rope_half(&x, &cos, &sin).unwrap();
                assert_eq!((got.dims(), got.dtype()), (x.dims(), dt));
                assert_eq!(bits(&got), bits(&former(&x, &cos, &sin)), "{dt:?}");
            }
        }
    }

    /// CUDA bf16 smoke: the Anima RoPE route takes the fused kernel and equals both the shared
    /// reference (switch off) and the former private chain, bit for bit.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_bf16_rope_route_is_fused_and_bit_identical() {
        use candle_gen::fused_path::media_fused_policy_guard;
        use candle_gen::rope::ROPE_PATH;
        let dev = Device::new_cuda(0).unwrap();
        let _media = media_fused_policy_guard(Some(true));
        let _rope = ROPE_PATH.guard(Some(true));
        // Anima DiT 1024^2: 64x64 post-patch tokens, 16 heads x 128.
        let (s, h, hd) = (4096, 16, 128);
        for (cos, sin) in tables(s, hd, &dev) {
            let x = Tensor::randn(0f32, 1.0, (1, s, h, hd), &dev).unwrap();
            let x = x.to_dtype(DType::BF16).unwrap();
            ROPE_PATH.set_enabled(Some(true));
            let start = ROPE_PATH.thread_counts();
            let on = apply_rope_half(&x, &cos, &sin).unwrap();
            let c = ROPE_PATH.thread_counts().since(&start);
            assert_eq!((c.fused, c.reference_total()), (1, 0), "{c}");
            ROPE_PATH.set_enabled(Some(false));
            let off = apply_rope_half(&x, &cos, &sin).unwrap();
            ROPE_PATH.set_enabled(Some(true));
            assert_eq!(bits(&on), bits(&off));
            assert_eq!(bits(&on), bits(&former(&x, &cos, &sin)));
        }
    }
}
