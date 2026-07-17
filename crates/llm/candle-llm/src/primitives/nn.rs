//! Neural-net leaves the decoders compose: linear projection, RMSNorm, the SiLU activation, the
//! embedding gather, and the host token-id → `Tensor` lift.
//!
//! Weights follow the HF convention — stored `[out, in]` — so [`linear`] is `x @ wᵀ` via
//! [`candle_nn::Linear`] (which broadcasts the weight over batch dims; no bias on Llama/Qwen).
//! [`rms_norm`] is hand-rolled and upcasts to f32 internally (matching `candle-transformers`'
//! `RmsNorm` and `mlx_rs::fast::rms_norm`) so bf16 decoders stay numerically stable.

use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};

use crate::error::{Error, Result};

/// `x @ weightᵀ (+ bias)`. `weight` is `[out, in]` (HF layout); `x` is `[..., in]`.
pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let lin = Linear::new(weight.clone(), bias.cloned());
    Ok(lin.forward(x)?)
}

/// RMSNorm: `x / sqrt(mean(x²) + eps) * weight`, computed in f32 and cast back to `x`'s dtype.
/// `weight` is `[d]` and broadcasts over the leading dims; the norm is over the last axis.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let orig = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let last = xf.rank() - 1;
    let mean = xf.sqr()?.mean_keepdim(last)?;
    let denom = (mean + eps)?.sqrt()?;
    let normed = xf.broadcast_div(&denom)?;
    let wf = weight.to_dtype(DType::F32)?;
    Ok(normed.broadcast_mul(&wf)?.to_dtype(orig)?)
}

/// SiLU / swish activation.
pub fn silu(x: &Tensor) -> Result<Tensor> {
    Ok(candle_nn::ops::silu(x)?)
}

/// GeLU, tanh approximation (`gelu_pytorch_tanh` — the Gemma GeGLU and SigLIP MLP activation).
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu()?)
}

/// GeLU, exact erf form (HF `nn.GELU()` default — the LLaVA multi-modal projector's activation).
pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    Ok(x.gelu_erf()?)
}

/// LayerNorm over the last axis: `(x - mean) / sqrt(var + eps) * weight + bias`. Computed in f32 and
/// cast back to `x`'s dtype (matching `candle_nn::LayerNorm`); `weight`/`bias` are `[d]` and
/// broadcast over the leading dims. The SigLIP tower's norms (unlike Llama's RMSNorm) are full
/// mean/variance LayerNorms with a learned bias.
pub fn layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    let orig = x.dtype();
    let xf = x.to_dtype(DType::F32)?;
    let last = xf.rank() - 1;
    let mean = xf.mean_keepdim(last)?;
    let centered = xf.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(last)?;
    let denom = (var + eps)?.sqrt()?;
    let normed = centered.broadcast_div(&denom)?;
    let wf = weight.to_dtype(DType::F32)?;
    let bf = bias.to_dtype(DType::F32)?;
    Ok(normed
        .broadcast_mul(&wf)?
        .broadcast_add(&bf)?
        .to_dtype(orig)?)
}

/// 2-D convolution of an NCHW input by an `[out_c, in_c, kH, kW]` (PyTorch-layout) `weight`, with an
/// optional per-output-channel `bias` (`[out_c]`). `stride`/`padding` are symmetric; dilation 1,
/// groups 1. Returns NCHW `[b, out_c, h_out, w_out]`. The SigLIP patch embedding is a stride-`patch`
/// `patch × patch` conv (a non-overlapping patchifier).
pub fn conv2d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let out = x.conv2d(weight, padding, stride, 1, 1)?;
    match bias {
        None => Ok(out),
        Some(b) => {
            let oc = b.dims1()?;
            Ok(out.broadcast_add(&b.reshape((1, oc, 1, 1))?)?)
        }
    }
}

/// Logit soft-cap `cap · tanh(x / cap)` (Gemma-2 caps attention scores and final logits). A no-op as
/// `cap → ∞`; it squashes extremes toward `±cap` while staying ~linear near 0.
pub fn soft_cap(x: &Tensor, cap: f32) -> Result<Tensor> {
    let c = cap as f64;
    Ok(x.affine(1.0 / c, 0.0)?.tanh()?.affine(c, 0.0)?)
}

/// Embedding gather: rows of `weight` (`[vocab, hidden]`) selected by `ids` (`[batch, seq]`, u32),
/// returning `[batch, seq, hidden]`. The result keeps `weight`'s dtype.
pub fn embed(weight: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (b, s) = ids.dims2()?;
    let hidden = weight.dim(1)?;
    let flat = ids.flatten_all()?;
    let gathered = weight.index_select(&flat, 0)?;
    Ok(gathered.reshape((b, s, hidden))?)
}

/// Lift a host token-id slice into a batch-1 `[1, len]` u32 `Tensor` on `device`.
pub fn input_ids(ids: &[i32], device: &Device) -> Result<Tensor> {
    let data: Vec<u32> = ids.iter().map(|&i| i as u32).collect();
    Ok(Tensor::from_vec(data, (1, ids.len()), device)?)
}

/// Lift equal-length host token-id rows into a `[batch, len]` u32 `Tensor` on `device`.
pub fn input_ids_batch(rows: &[&[i32]], device: &Device) -> Result<Tensor> {
    let batch = rows.len();
    if batch == 0 {
        return Err(Error::Msg("input_ids_batch: no rows".into()));
    }
    let len = rows[0].len();
    let mut flat: Vec<u32> = Vec::with_capacity(batch * len);
    for (i, r) in rows.iter().enumerate() {
        if r.len() != len {
            return Err(Error::Msg(format!(
                "input_ids_batch: row {i} has length {} != {len}",
                r.len()
            )));
        }
        flat.extend(r.iter().map(|&i| i as u32));
    }
    Ok(Tensor::from_vec(flat, (batch, len), device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_matches_manual_matmul() {
        // x: [1,2], w: [3,2] (out=3, in=2). y = x @ w.t() -> [1,3].
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();
        let w =
            Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], (3, 2), &Device::Cpu).unwrap();
        let y = linear(&x, &w, None).unwrap();
        assert_eq!(y.dims(), &[1, 3]);
        let h = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(h, vec![1.0, 2.0, 3.0]); // [x0, x1, x0+x1]
    }

    #[test]
    fn embed_gathers_rows() {
        // vocab 3, hidden 2: row0=[0,1] row1=[2,3] row2=[4,5]
        let w =
            Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], (3, 2), &Device::Cpu).unwrap();
        let ids = input_ids(&[2, 0], &Device::Cpu).unwrap();
        let e = embed(&w, &ids).unwrap();
        assert_eq!(e.dims(), &[1, 2, 2]);
        let h = e.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(h, vec![4.0, 5.0, 0.0, 1.0]);
    }

    #[test]
    fn input_ids_batch_shapes() {
        let a = [1, 2, 3];
        let b = [4, 5, 6];
        let arr = input_ids_batch(&[&a, &b], &Device::Cpu).unwrap();
        assert_eq!(arr.dims(), &[2, 3]);
    }

    #[test]
    fn input_ids_batch_rejects_ragged() {
        let a = [1, 2, 3];
        let b = [4, 5];
        assert!(input_ids_batch(&[&a, &b], &Device::Cpu).is_err());
    }

    #[test]
    fn layer_norm_zero_mean_unit_var() {
        // [1,2,3,4]: mean 2.5, var 1.25, eps tiny ⇒ normalized has mean ~0, then *1 + 0.
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu).unwrap();
        let w = Tensor::ones((4,), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::zeros((4,), DType::F32, &Device::Cpu).unwrap();
        let y = layer_norm(&x, &w, &b, 1e-6).unwrap();
        let h = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mean: f32 = h.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-4, "mean {mean}");
        // Symmetric about center: first and last are negatives of each other.
        assert!((h[0] + h[3]).abs() < 1e-4);
        assert!(h[0] < 0.0 && h[3] > 0.0);
    }

    #[test]
    fn conv2d_patchifies_with_stride() {
        // 1x1x4x4 input, a 1x1x2x2 averaging kernel at stride 2 ⇒ 1x1x2x2 of 2x2-block sums.
        let x = Tensor::from_vec(
            (0..16).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 1, 4, 4),
            &Device::Cpu,
        )
        .unwrap();
        let k = Tensor::ones((1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let y = conv2d(&x, &k, None, 2, 0).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 2]);
        let h = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // top-left 2x2 block = 0+1+4+5 = 10; top-right = 2+3+6+7 = 18; etc.
        assert_eq!(h, vec![10.0, 18.0, 42.0, 50.0]);
    }

    #[test]
    fn conv2d_adds_bias_per_channel() {
        let x = Tensor::zeros((1, 1, 2, 2), DType::F32, &Device::Cpu).unwrap();
        let k = Tensor::ones((3, 1, 1, 1), DType::F32, &Device::Cpu).unwrap();
        let b = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &Device::Cpu).unwrap();
        let y = conv2d(&x, &k, Some(&b), 1, 0).unwrap();
        assert_eq!(y.dims(), &[1, 3, 2, 2]);
        let h = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(
            h,
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn rms_norm_unit_weight_normalizes() {
        // A constant vector normalizes to ~1 per element (rms == value).
        let x = Tensor::from_vec(vec![2.0f32, 2.0, 2.0, 2.0], (1, 4), &Device::Cpu).unwrap();
        let w = Tensor::ones((4,), DType::F32, &Device::Cpu).unwrap();
        let y = rms_norm(&x, &w, 1e-6).unwrap();
        let h = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for v in h {
            assert!((v - 1.0).abs() < 1e-3, "{v}");
        }
    }
}
