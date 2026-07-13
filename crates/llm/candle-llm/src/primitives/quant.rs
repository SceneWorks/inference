//! Group-wise quantization (Q4 / Q8) for linear projections, via Candle's native quantized tensors.
//!
//! Per the story's decision, this uses **Candle's** quantization (`candle_core::quantized`):
//! [`QTensor::quantize`] packs a dense `[out, in]` weight into a GGML block-quantized tensor, and
//! [`QMatMul`] computes `x @ wᵀ` against it (the quantized analogue of [`super::nn::linear`]). This
//! backs quantize-on-load. Note GGML block quant requires the input dimension to be a multiple of
//! the block size (Q4K: 256, Q8_0: 32); real model dims satisfy this, but tiny synthetic weights may
//! not — those load dense.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Tensor};
use candle_nn::Module;

use crate::error::Result;

/// A linear projection whose weight is stored GGML block-quantized.
pub struct QuantizedLinear {
    inner: QMatMul,
    /// Optional additive bias applied after the matmul.
    bias: Option<Tensor>,
}

impl QuantizedLinear {
    /// Quantize a dense `[out, in]` weight (the input dim must be a multiple of `dtype`'s block
    /// size). `bias`, if present, is added after the matmul.
    pub fn quantize(weight: &Tensor, dtype: GgmlDType, bias: Option<Tensor>) -> Result<Self> {
        let qt = QTensor::quantize(&weight.to_dtype(DType::F32)?, dtype)?;
        Ok(Self {
            inner: QMatMul::from_qtensor(qt)?,
            bias,
        })
    }

    /// Forward pass: `x @ dequant(weight)ᵀ (+ bias)`. The quantized matmul runs in f32; the result is
    /// cast back to `x`'s dtype so it composes with a bf16 decoder.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xf = x.to_dtype(DType::F32)?;
        let y = self.inner.forward(&xf)?.to_dtype(x.dtype())?;
        match &self.bias {
            Some(b) => Ok(y.broadcast_add(b)?),
            None => Ok(y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::nn::linear;
    use candle_core::Device;

    /// Quantized matmul should approximate the dense linear it replaces (Q8_0, in=256 = 8 blocks).
    #[test]
    fn quantized_matmul_approximates_linear_q8() {
        let (out, inn) = (4usize, 256usize);
        let wdata: Vec<f32> = (0..out * inn)
            .map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5)
            .collect();
        let w = Tensor::from_vec(wdata, (out, inn), &Device::Cpu).unwrap();
        let xdata: Vec<f32> = (0..inn).map(|i| (i as f32 / inn as f32) - 0.5).collect();
        let x = Tensor::from_vec(xdata, (1, inn), &Device::Cpu).unwrap();

        let dense = linear(&x, &w, None)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let q = QuantizedLinear::quantize(&w, GgmlDType::Q8_0, None).unwrap();
        let quant = q
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (a, b) in dense.iter().zip(&quant) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }
}
