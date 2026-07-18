//! Small candle neural-net helpers shared by the Bernini planner modules (the Qwen2.5-VL backbone +
//! the connector) — the candle twins of the `mlx_gen::nn` ops the MLX port uses. Kept composable (no
//! fused-kernel dispatch) so they match the MLX numerics on CPU and CUDA alike (mirrors
//! `candle-gen-anima/src/nn.rs`).

use candle_gen::candle_core::{DType, Tensor, D};
use candle_gen::candle_nn::{Linear, VarBuilder};
use candle_gen::Result;

/// `Linear` **with** bias loaded from `{name}.weight` + `{name}.bias`.
pub fn lin_bias(vb: &VarBuilder, name: &str) -> Result<Linear> {
    Ok(Linear::new(
        vb.get_unchecked(&format!("{name}.weight"))?,
        Some(vb.get_unchecked(&format!("{name}.bias"))?),
    ))
}

/// Plain RMSNorm `w · x / sqrt(mean(x²) + eps)` over the last dim (matches `mlx_rs::fast::rms_norm`).
/// `w` broadcasts over the leading dims. The reduction runs in f32, matching Qwen2RMSNorm and torch's
/// mixed-precision norm kernels, then the normalized values cast back to the input dtype before the
/// affine multiply.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.sqr()?.mean_keepdim(D::Minus1)?; // [.., 1]
    let denom = (mean + eps)?.sqrt()?;
    Ok(x.broadcast_div(&denom)?.to_dtype(dtype)?.broadcast_mul(w)?)
}

/// Torch `nn.LayerNorm` over the last dim: `(x − mean)/sqrt(var + eps)`, with **biased** variance
/// (divide by N, no Bessel) matching torch, then optional affine `·w + b`. The candle twin of
/// `mlx_rs::fast::layer_norm`; `w`/`b` `None` is the `elementwise_affine=False` variant (the clip-diff
/// `FinalLayer`'s `norm_final`). Reductions and affine arithmetic run in f32, matching torch's
/// mixed-precision LayerNorm kernel, and the result casts back to the input dtype.
pub fn layer_norm(x: &Tensor, w: Option<&Tensor>, b: Option<&Tensor>, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let mut y = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    if let Some(w) = w {
        y = y.broadcast_mul(&w.to_dtype(DType::F32)?)?;
    }
    if let Some(b) = b {
        y = y.broadcast_add(&b.to_dtype(DType::F32)?)?;
    }
    Ok(y.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
        (a.to_dtype(DType::F32).unwrap() - b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn rms_norm_bf16_reduces_in_f32_and_casts_back() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 5.0], (1, 4), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(vec![0.75f32, 1.25, 0.5, 1.5], (4,), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let actual = rms_norm(&x, &w, 1e-6).unwrap();
        let xf = x.to_dtype(DType::F32).unwrap();
        let denom = (xf.sqr().unwrap().mean_keepdim(D::Minus1).unwrap() + 1e-6)
            .unwrap()
            .sqrt()
            .unwrap();
        let expected = xf
            .broadcast_div(&denom)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap()
            .broadcast_mul(&w)
            .unwrap();
        let legacy = x
            .broadcast_div(
                &(x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap() + 1e-6)
                    .unwrap()
                    .sqrt()
                    .unwrap(),
            )
            .unwrap()
            .broadcast_mul(&w)
            .unwrap();

        assert_eq!(actual.dtype(), DType::BF16);
        assert_eq!(max_abs(&actual, &expected), 0.0);
        assert!(max_abs(&actual, &legacy) > 0.0);
    }

    #[test]
    fn layer_norm_bf16_reduces_and_applies_affine_in_f32_then_casts_back() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 5.0], (1, 4), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::from_vec(vec![0.75f32, 1.25, 0.5, 1.5], (4,), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b = Tensor::from_vec(vec![0.1f32, -0.2, 0.3, -0.4], (4,), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        let actual = layer_norm(&x, Some(&w), Some(&b), 1e-6).unwrap();
        let xf = x.to_dtype(DType::F32).unwrap();
        let mean = xf.mean_keepdim(D::Minus1).unwrap();
        let centered = xf.broadcast_sub(&mean).unwrap();
        let denom = (centered.sqr().unwrap().mean_keepdim(D::Minus1).unwrap() + 1e-6)
            .unwrap()
            .sqrt()
            .unwrap();
        let expected = centered
            .broadcast_div(&denom)
            .unwrap()
            .broadcast_mul(&w.to_dtype(DType::F32).unwrap())
            .unwrap()
            .broadcast_add(&b.to_dtype(DType::F32).unwrap())
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        assert_eq!(actual.dtype(), DType::BF16);
        assert_eq!(max_abs(&actual, &expected), 0.0);
    }
}
