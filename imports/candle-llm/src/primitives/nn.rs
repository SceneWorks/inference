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

/// Convert a logits/last-position `Tensor` to a host `f32` vector (e.g. for host-side sampling).
pub fn to_f32_host(x: &Tensor) -> Result<Vec<f32>> {
    Ok(x.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?)
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
