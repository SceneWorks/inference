//! Group-wise quantization (Q4 / Q8) for linear projections, via Candle's native quantized tensors.
//!
//! Per the story's decision, this uses **Candle's** quantization (`candle_core::quantized`):
//! [`QTensor::quantize`] packs a dense `[out, in]` weight into a GGML block-quantized tensor. Dense
//! quantize-on-load uses [`QMatMul`]; pre-quantized MLX-affine tiers dequantize the resident weight
//! per forward so activation outliers remain full-precision, matching the shared Candle packed-tier
//! contract. GGML block quant requires the input dimension to be a multiple of the block size
//! (Q4K: 256, Q8_0: 32); real model dims satisfy this, but tiny synthetic weights may not.

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};

use crate::error::Result;

/// A linear projection whose weight is stored GGML block-quantized.
pub struct QuantizedLinear {
    inner: QuantizedWeight,
    /// Optional additive bias applied after the matmul.
    bias: Option<Tensor>,
}

enum QuantizedWeight {
    /// Existing dense load-time quantization path.
    Matmul(QMatMul),
    /// MLX affine packed tiers keep activations full precision and dequantize the resident weight
    /// per forward, matching the shared Candle packed-tier policy.
    Dequant(std::sync::Arc<QTensor>),
}

impl QuantizedLinear {
    /// Quantize a dense `[out, in]` weight (the input dim must be a multiple of `dtype`'s block
    /// size). `bias`, if present, is added after the matmul.
    pub fn quantize(weight: &Tensor, dtype: GgmlDType, bias: Option<Tensor>) -> Result<Self> {
        let qt = QTensor::quantize(&weight.to_dtype(DType::F32)?, dtype)?;
        Ok(Self {
            inner: QuantizedWeight::Matmul(QMatMul::from_qtensor(qt)?),
            bias,
        })
    }

    /// Convert an MLX affine Q8 triple (`U32` byte-packed codes plus per-group scale/bias) into the
    /// resident Q8_0 tensor used by this primitive. This matches the established Candle packed-tier
    /// policy: reconstruct the source affine grid exactly, then perform the accepted Q8_0 re-pack.
    pub fn from_mlx_affine_q8(
        weight: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
        device: &Device,
    ) -> Result<Self> {
        let (out_dim, packed_cols) = weight.dims2()?;
        let (scale_rows, scale_cols) = scales.dims2()?;
        let in_dim = scale_cols.checked_mul(group_size).ok_or_else(|| {
            crate::error::Error::Config("MLX affine Q8 input width overflow".into())
        })?;
        if weight.dtype() != DType::U32
            || group_size == 0
            || scale_rows != out_dim
            || biases.dims2()? != (scale_rows, scale_cols)
            || packed_cols.checked_mul(4) != Some(in_dim)
        {
            return Err(crate::error::Error::Config(format!(
                "invalid MLX affine Q8 triple: weight {:?} {:?}, scales {:?}, biases {:?}, group {group_size}",
                weight.dtype(),
                weight.shape(),
                scales.shape(),
                biases.shape()
            )));
        }

        let cpu = Device::Cpu;
        let words = weight.to_device(&cpu)?.flatten_all()?.to_vec1::<u32>()?;
        let scales = scales
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let biases = biases
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut grid = vec![0f32; out_dim * in_dim];
        for row in 0..out_dim {
            let word_row = row * packed_cols;
            let group_row = row * scale_cols;
            let value_row = row * in_dim;
            for col in 0..in_dim {
                let word = words[word_row + col / 4];
                let code = ((word >> (8 * (col % 4))) & 0xff) as f32;
                let group = group_row + col / group_size;
                grid[value_row + col] = scales[group] * code + biases[group];
            }
        }
        let dense = Tensor::from_vec(grid, (out_dim, in_dim), &cpu)?;
        let qt = QTensor::quantize_onto(&dense, GgmlDType::Q8_0, device)?;
        Ok(Self {
            inner: QuantizedWeight::Dequant(std::sync::Arc::new(qt)),
            bias,
        })
    }

    /// Forward pass: `x @ dequant(weight)ᵀ (+ bias)`. The quantized matmul runs in f32; the result is
    /// cast back to `x`'s dtype so it composes with a bf16 decoder.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = match &self.inner {
            QuantizedWeight::Matmul(inner) => inner
                .forward(&x.to_dtype(DType::F32)?)?
                .to_dtype(x.dtype())?,
            QuantizedWeight::Dequant(weight) => {
                let dense = weight.dequantize(x.device())?.to_dtype(x.dtype())?;
                Linear::new(dense, None).forward(x)?
            }
        };
        match &self.bias {
            Some(b) => Ok(y.broadcast_add(&b.to_dtype(y.dtype())?)?),
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

    #[test]
    fn mlx_affine_q8_reconstructs_lsb_bytes_bias_and_input_groups() {
        let dev = Device::Cpu;
        let (out, inn, group) = (2usize, 64usize, 16usize);
        let codes: Vec<u8> = (0..out * inn)
            .map(|i| [0, 255, 17, 193, 91, 7, 241, 63][i % 8])
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(4)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u32, |word, (index, code)| {
                    word | ((*code as u32) << (index * 8))
                })
            })
            .collect();
        let scale_values = vec![0.01f32, 0.02, 0.03, 0.04, 0.015, 0.025, 0.035, 0.045];
        let bias_values = vec![-1.0f32, 0.5, -0.25, 1.25, 0.75, -1.5, 2.0, -0.75];
        let weight = Tensor::from_vec(words, (out, inn / 4), &dev).unwrap();
        let scales = Tensor::from_vec(scale_values.clone(), (out, inn / group), &dev).unwrap();
        let biases = Tensor::from_vec(bias_values.clone(), (out, inn / group), &dev).unwrap();
        let x = Tensor::ones((2, inn), DType::F32, &dev).unwrap();

        let packed =
            QuantizedLinear::from_mlx_affine_q8(&weight, &scales, &biases, None, group, &dev)
                .unwrap();
        let resident = match &packed.inner {
            QuantizedWeight::Dequant(weight) => weight.dequantize(&dev).unwrap(),
            QuantizedWeight::Matmul(_) => panic!("MLX affine Q8 must use the packed-source path"),
        };
        let got = resident.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected: Vec<f32> = codes
            .iter()
            .enumerate()
            .map(|(index, &code)| {
                let row = index / inn;
                let col = index % inn;
                let source_group = row * (inn / group) + col / group;
                scale_values[source_group] * code as f32 + bias_values[source_group]
            })
            .collect();
        for (index, (actual, expected)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 0.06,
                "index {index}: {actual} vs affine source {expected}"
            );
        }

        let y = packed.forward(&x).unwrap();
        assert_eq!(y.dims(), &[2, out]);
    }
}
