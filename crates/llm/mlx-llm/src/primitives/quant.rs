//! Group-wise affine quantization (Q4 / Q8) for linear projections.
//!
//! This is greenfield for the engine — the mlx-gen LLM stacks (prompt-refine, JoyCaption) reject
//! quantization at load. We build on MLX's native group-wise affine quantization
//! (`ops::quantize` / `ops::quantized_matmul`), which packs an `[out, in]` weight into a quantized
//! tensor plus per-group `scales`/`biases`. This backs quantize-on-load (story 7163) and the GGUF
//! path (7165).

use mlx_rs::ops::{add, dequantize, quantize, quantized_matmul};
use mlx_rs::{Array, Dtype};

use crate::error::Result;

/// A token embedding table stored in MLX's group-wise packed affine format.
///
/// Unlike a linear projection, embedding lookup selects packed rows first and dequantizes only the
/// requested tokens. This preserves the snapshot's physical representation instead of expanding a
/// vocabulary-sized dense table at load time.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    /// Packed `[vocab, hidden * bits / 32]` words.
    pub weight: Array,
    /// Per-row, per-group scales.
    pub scales: Array,
    /// Per-row, per-group affine biases.
    pub biases: Array,
    /// Elements per quantization group.
    pub group_size: i32,
    /// Bits per weight.
    pub bits: i32,
    hidden_size: i32,
}

impl QuantizedEmbedding {
    /// Validate and retain already-quantized embedding parts from a snapshot, holding the affine
    /// `scales`/`biases` in the model's `compute` dtype (sc-20671: a stored F16 scale must not
    /// promote BF16 activations to F32).
    pub fn from_quantized(
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
        compute: Dtype,
    ) -> Result<Self> {
        let weight_shape = weight.shape();
        let scales_shape = scales.shape();
        let biases_shape = biases.shape();
        if weight_shape.len() != 2 || scales_shape.len() != 2 || biases_shape != scales_shape {
            return Err(crate::error::Error::Config(format!(
                "quantized embedding parts must be rank-2 with matching scale/bias shapes: weight={weight_shape:?}, scales={scales_shape:?}, biases={biases_shape:?}"
            )));
        }
        if weight_shape[0] != scales_shape[0] || group_size <= 0 || !matches!(bits, 4 | 8) {
            return Err(crate::error::Error::Config(format!(
                "invalid quantized embedding geometry: weight={weight_shape:?}, scales={scales_shape:?}, group_size={group_size}, bits={bits}"
            )));
        }
        let values_per_word = 32 / bits;
        let hidden_size = weight_shape[1]
            .checked_mul(values_per_word)
            .ok_or_else(|| {
                crate::error::Error::Config("quantized embedding width overflow".into())
            })?;
        if hidden_size % group_size != 0 || scales_shape[1] != hidden_size / group_size {
            return Err(crate::error::Error::Config(format!(
                "quantized embedding metadata does not reconstruct hidden size {hidden_size}: scales={scales_shape:?}, group_size={group_size}"
            )));
        }
        Ok(Self {
            weight,
            scales: in_compute_dtype(scales, compute)?,
            biases: in_compute_dtype(biases, compute)?,
            group_size,
            bits,
            hidden_size,
        })
    }

    /// Gather packed vocabulary rows and dequantize only the requested `[batch, sequence]` tokens.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let ids_shape = ids.shape();
        if ids_shape.len() != 2 {
            return Err(crate::error::Error::Msg(format!(
                "quantized embedding ids must be [batch, sequence], got {ids_shape:?}"
            )));
        }
        let flat = ids.reshape(&[-1])?;
        let weight = self.weight.take_axis(&flat, 0)?;
        let scales = self.scales.take_axis(&flat, 0)?;
        let biases = self.biases.take_axis(&flat, 0)?;
        let rows = dequantize(&weight, &scales, Some(&biases), self.group_size, self.bits)?;
        Ok(rows.reshape(&[ids_shape[0], ids_shape[1], self.hidden_size])?)
    }

    /// Reuse a tied quantized embedding as the vocabulary projection without materializing it.
    pub fn tied_linear(&self) -> QuantizedLinear {
        QuantizedLinear {
            weight: self.weight.clone(),
            scales: self.scales.clone(),
            biases: self.biases.clone(),
            group_size: self.group_size,
            bits: self.bits,
            bias: None,
        }
    }
}

/// Cast a stored affine quantization parameter to the model's compute dtype.
///
/// MLX's affine `quantized_matmul` produces `promote_types(x, scales)` (and `dequantize` the scale
/// dtype), so a stored scale dtype that differs from the activations silently widens every
/// downstream tensor: BF16
/// activations against the F16 scales mlx-community checkpoints ship promote to **F32**, and the
/// whole decoder — including the K/V handed to the cache — then runs 4-byte (sc-20671). Holding the
/// parameters in the compute dtype at load keeps activations, and therefore the KV cache, in the
/// compute dtype whatever dtype the snapshot stored them in.
pub(crate) fn in_compute_dtype(parameter: Array, compute: Dtype) -> Result<Array> {
    if parameter.dtype() == compute {
        Ok(parameter)
    } else {
        Ok(parameter.as_dtype(compute)?)
    }
}

/// A linear projection whose weight is stored group-wise quantized.
///
/// Forward is `quantized_matmul(x, weight, scales, biases, transpose = true, ...)`, which computes
/// `x @ weight.t()` against the dequantized weight — the quantized analogue of
/// [`super::nn::linear`]. `transpose = true` matches the HF `[out, in]` weight layout.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Packed quantized weight.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group biases (zero-points).
    pub biases: Array,
    /// Elements per quantization group (e.g. 64).
    pub group_size: i32,
    /// Bits per weight (4 or 8).
    pub bits: i32,
    /// Optional additive bias applied after the matmul.
    pub bias: Option<Array>,
}

impl QuantizedLinear {
    /// Retain already-quantized projection parts from a snapshot (the packed `weight` plus its
    /// per-group `scales`/`biases`), holding the affine parameters — and the optional additive
    /// `bias` — in the model's `compute` dtype (sc-20671: a stored F16 scale must not promote BF16
    /// activations to F32). The packed weight is used as-is.
    pub fn from_stored(
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
        bias: Option<Array>,
        compute: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            weight,
            scales: in_compute_dtype(scales, compute)?,
            biases: in_compute_dtype(biases, compute)?,
            group_size,
            bits,
            bias: bias.map(|b| in_compute_dtype(b, compute)).transpose()?,
        })
    }

    /// Quantize a dense `[out, in]` weight into a `QuantizedLinear`. `group_size` must divide the
    /// input dimension; `bits` is typically 4 or 8.
    pub fn quantize(
        weight: &Array,
        group_size: i32,
        bits: i32,
        bias: Option<Array>,
    ) -> Result<Self> {
        let (w, scales, biases) = quantize(weight, group_size, bits)?;
        Ok(Self {
            weight: w,
            scales,
            biases,
            group_size,
            bits,
            bias,
        })
    }

    /// Forward pass: `x @ dequant(weight).t() (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true, // transpose: weight is [out, in]
            self.group_size,
            self.bits,
        )?;
        match &self.bias {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }

    /// Reconstruct the dense weight (mostly for parity tests / inspection).
    pub fn dequantize_weight(&self) -> Result<Array> {
        Ok(dequantize(
            &self.weight,
            &self.scales,
            Some(&self.biases),
            self.group_size,
            self.bits,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::nn::{embed, input_ids, linear};

    #[test]
    fn quantized_embedding_gathers_rows_without_expanding_the_table() {
        let dense = Array::from_slice(
            &(0..4 * 64)
                .map(|i| ((i * 7 % 29) as f32 / 29.0) - 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        );
        let linear = QuantizedLinear::quantize(&dense, 64, 4, None).unwrap();
        let embedding = QuantizedEmbedding::from_quantized(
            linear.weight.clone(),
            linear.scales.clone(),
            linear.biases.clone(),
            linear.group_size,
            linear.bits,
            Dtype::Float32,
        )
        .unwrap();
        let ids = input_ids(&[3, 1]);
        let actual = embedding.forward(&ids).unwrap();
        let expected = embed(&linear.dequantize_weight().unwrap(), &ids).unwrap();
        assert_eq!(actual.shape(), &[1, 2, 64]);
        let actual = actual.as_slice::<f32>();
        let expected = expected.as_slice::<f32>();
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(left, right)| (*left - *right).abs() <= f32::EPSILON));
    }

    /// Quantize→dequantize should round-trip within the affine grid's tolerance.
    #[test]
    fn quantize_dequantize_roundtrip_q8() {
        // [out=2, in=64] so a single group of 64 covers the input dim.
        let n = 2 * 64;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) - 0.5).collect();
        let w = Array::from_slice(&data, &[2, 64]);
        let q = QuantizedLinear::quantize(&w, 64, 8, None).unwrap();
        let recon = q.dequantize_weight().unwrap();
        let orig = w.as_slice::<f32>().to_vec();
        let back = recon.as_slice::<f32>().to_vec();
        let max_err = orig
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // 8-bit affine over a ~1.0 range: error well under 1%.
        assert!(max_err < 0.01, "max_err = {max_err}");
    }

    /// Quantized matmul should approximate the dense linear it replaces.
    #[test]
    fn quantized_matmul_approximates_linear_q8() {
        let n = 4 * 64;
        let wdata: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5).collect();
        let w = Array::from_slice(&wdata, &[4, 64]); // [out=4, in=64]
        let x = Array::from_slice(
            &(0..64).map(|i| (i as f32 / 64.0) - 0.5).collect::<Vec<_>>(),
            &[1, 64],
        );
        let dense = linear(&x, &w, None).unwrap().as_slice::<f32>().to_vec();
        let q = QuantizedLinear::quantize(&w, 64, 8, None).unwrap();
        let quant = q.forward(&x).unwrap().as_slice::<f32>().to_vec();
        for (a, b) in dense.iter().zip(&quant) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }
}
