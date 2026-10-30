//! Composable RMS normalization with explicit f32 reduction and rounding boundaries (sc-26854).

use candle_core::{DType, Tensor, D};

/// Normalize in f32 with a differentiable reciprocal standard deviation. The mean reduction
/// remains f32 even when the activation stream is BF16/F16. No learned module is called here.
pub fn rms_normalize_f32(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let xf = x.to_dtype(DType::F32)?;
    let width = xf.dim(D::Minus1)?;
    if width == 0 || !eps.is_finite() || eps < 0.0 {
        candle_core::bail!(
            "RMS norm requires a nonempty final dimension and finite nonnegative epsilon"
        );
    }
    let mean = (xf.sqr()?.sum_keepdim(D::Minus1)? / width as f64)?;
    // Multiply by reciprocal rather than broadcast-dividing every full-width element.
    if super::profile::reference_math() {
        xf.broadcast_div(&(mean + eps)?.sqrt()?)
    } else {
        xf.broadcast_mul(&(mean + eps)?.sqrt()?.recip()?)
    }
}

/// Apply f32 weight before the final activation cast (Krea's pre-folded scale+1 contract).
pub fn rms_norm_diff(x: &Tensor, weight_f32: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    rms_normalize_f32(x, eps)?
        .broadcast_mul(weight_f32)?
        .to_dtype(x.dtype())
}

/// Cast normalized values to activation dtype before multiplying the weight. Some families
/// deliberately retain this earlier rounding point; do not replace them with `rms_norm_diff`.
pub fn rms_norm_cast_before_weight(
    x: &Tensor,
    weight: &Tensor,
    eps: f64,
) -> candle_core::Result<Tensor> {
    rms_normalize_f32(x, eps)?
        .to_dtype(x.dtype())?
        .broadcast_mul(weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    #[test]
    fn reciprocal_norm_matches_reference_values_and_gradients() -> candle_core::Result<()> {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let x = Var::from_tensor(
                &Tensor::from_vec(
                    vec![0.01f32, -0.4, 2.1, 0.7, 1.2, -3.0, 0.5, 0.1],
                    (2, 4),
                    &Device::Cpu,
                )?
                .to_dtype(dtype)?,
            )?;
            let weight = Tensor::from_vec(vec![1.1f32, 0.9, 0.8, 1.2], 4, &Device::Cpu)?;
            let xf = x.to_dtype(DType::F32)?;
            let mean = (xf.sqr()?.sum_keepdim(D::Minus1)? / 4.0)?;
            let reference = xf
                .broadcast_div(&(mean + 1e-6)?.sqrt()?)?
                .broadcast_mul(&weight)?
                .to_dtype(dtype)?;
            let actual = rms_norm_diff(&x, &weight, 1e-6)?;
            let tol = if dtype == DType::F32 { 1e-6 } else { 0.02 };
            let error = (&actual.to_dtype(DType::F32)? - &reference.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert!(error <= tol, "{dtype:?} error {error}");
            let ga = actual.to_dtype(DType::F32)?.sqr()?.sum_all()?.backward()?;
            let gr = reference
                .to_dtype(DType::F32)?
                .sqr()?
                .sum_all()?
                .backward()?;
            let error = (ga.get(&x).unwrap().to_dtype(DType::F32)?
                - gr.get(&x).unwrap().to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
            assert!(error <= tol, "{dtype:?} gradient error {error}");
        }
        Ok(())
    }
}
