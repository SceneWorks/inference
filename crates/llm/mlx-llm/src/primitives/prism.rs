//! Prism ternary affine operators for Ternary Bonsai checkpoints.
//!
//! The packed MLX artifact stores 16 two-bit affine codes per `u32`, with one scale and bias for
//! each 128 input values. Bonsai folds a normalized blockwise Walsh-Hadamard rotation into every
//! packed matrix: linear activations apply `signs` then `H`, while embedding rows apply the inverse
//! `H` then `signs`. These operators keep the packed weights resident and use MLX's native
//! two-bit matmul/dequantize kernels; they never materialize a full dense model.

use mlx_rs::ops::{dequantize, multiply, quantized_matmul};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};

const GROUP_SIZE: i32 = 128;
const BITS: i32 = 2;

fn validate_parts(
    label: &str,
    weight: &Array,
    scales: &Array,
    biases: &Array,
    signs: &Array,
    block: i32,
) -> Result<(i32, i32)> {
    let ws = weight.shape();
    let ss = scales.shape();
    let bs = biases.shape();
    if weight.dtype() != Dtype::Uint32 {
        return Err(Error::Config(format!(
            "Prism packed `{label}` weight must be U32, got {:?}",
            weight.dtype()
        )));
    }
    if ws.len() != 2 || ss.len() != 2 || bs != ss {
        return Err(Error::Config(format!(
            "Prism packed `{label}` has invalid part shapes: weight {ws:?}, scales {ss:?}, biases {bs:?}"
        )));
    }
    if !matches!(
        scales.dtype(),
        Dtype::Float16 | Dtype::Float32 | Dtype::Bfloat16
    ) || biases.dtype() != scales.dtype()
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` affine parameters must share an F16/F32/BF16 dtype"
        )));
    }
    let rows = ss[0];
    let width = ss[1]
        .checked_mul(GROUP_SIZE)
        .ok_or_else(|| Error::Config(format!("Prism packed `{label}` width overflow")))?;
    if rows <= 0
        || width <= 0
        || ws[0] != rows
        || ws[1] != width / 16
        || block <= 0
        || width % block != 0
        || signs.shape() != [width]
        || signs.dtype() != Dtype::Float32
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` geometry mismatch: weight {ws:?}, scales {ss:?}, signs {:?}, block {block}",
            signs.shape()
        )));
    }
    let sign_values = signs.as_slice::<f32>();
    if sign_values
        .iter()
        .any(|&value| value != -1.0 && value != 1.0)
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` signs must contain only -1 or +1"
        )));
    }
    let scale_values = scales.as_dtype(Dtype::Float32)?;
    let bias_values = biases.as_dtype(Dtype::Float32)?;
    if scale_values
        .as_slice::<f32>()
        .iter()
        .chain(bias_values.as_slice::<f32>())
        .any(|value| !value.is_finite())
    {
        return Err(Error::Config(format!(
            "Prism packed `{label}` affine parameters must be finite"
        )));
    }
    Ok((rows, width))
}

fn block_hadamard(x: &Array, signs: &Array, block: i32, inverse: bool) -> Result<Array> {
    let shape = x.shape().to_vec();
    let width = *shape
        .last()
        .ok_or_else(|| Error::Config("Prism Hadamard input must have a final axis".into()))?;
    if width % block != 0 || signs.shape() != [width] {
        return Err(Error::Config(format!(
            "Prism Hadamard width {width} is incompatible with block {block} and signs {:?}",
            signs.shape()
        )));
    }
    let dtype = x.dtype();
    let mut transformed = x.as_dtype(Dtype::Float32)?;
    if !inverse {
        transformed = multiply(&transformed, signs)?;
    }
    transformed = transformed
        .reshape(&[-1, block])?
        .hadamard_transform(Some(1.0 / (block as f32).sqrt()))?
        .reshape(&shape)?;
    if inverse {
        transformed = multiply(&transformed, signs)?;
    }
    Ok(transformed.as_dtype(dtype)?)
}

/// A packed Prism affine matrix with its required forward activation rotation.
#[derive(Debug)]
pub struct PrismLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    signs: Array,
    block: i32,
}

impl PrismLinear {
    /// Validate and retain a packed `[out, in]` matrix without dequantizing it.
    pub fn new(
        label: &str,
        weight: Array,
        scales: Array,
        biases: Array,
        signs: Array,
        block: i32,
    ) -> Result<Self> {
        validate_parts(label, &weight, &scales, &biases, &signs, block)?;
        Ok(Self {
            weight,
            scales,
            biases,
            signs,
            block,
        })
    }

    /// Apply `signs`, normalized blockwise Hadamard, then packed affine matmul.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let rotated = block_hadamard(x, &self.signs, self.block, false)?;
        Ok(quantized_matmul(
            &rotated,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true,
            GROUP_SIZE,
            BITS,
        )?)
    }
}

/// Packed Prism token embeddings with the inverse rotation applied after row dequantization.
#[derive(Debug)]
pub struct PrismEmbedding {
    weight: Array,
    scales: Array,
    biases: Array,
    signs: Array,
    block: i32,
    width: i32,
}

impl PrismEmbedding {
    /// Validate and retain a packed `[vocab, hidden]` embedding table.
    pub fn new(
        label: &str,
        weight: Array,
        scales: Array,
        biases: Array,
        signs: Array,
        block: i32,
    ) -> Result<Self> {
        let (_, width) = validate_parts(label, &weight, &scales, &biases, &signs, block)?;
        Ok(Self {
            weight,
            scales,
            biases,
            signs,
            block,
            width,
        })
    }

    /// Gather only requested packed rows, dequantize them, then apply `H` and `signs`.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let id_shape = ids.shape();
        if id_shape.len() != 2 {
            return Err(Error::Msg(format!(
                "Prism embedding expects [batch, sequence] ids, got {id_shape:?}"
            )));
        }
        let flat = ids.reshape(&[-1])?;
        let weight = self.weight.take_axis(&flat, 0)?;
        let scales = self.scales.take_axis(&flat, 0)?;
        let biases = self.biases.take_axis(&flat, 0)?;
        let rows = dequantize(&weight, &scales, Some(&biases), GROUP_SIZE, BITS)?;
        let rows = rows.reshape(&[id_shape[0], id_shape[1], self.width])?;
        block_hadamard(&rows, &self.signs, self.block, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::nn::linear;

    #[test]
    fn packed_linear_and_inverse_embedding_match_dense_operator_oracles() {
        let width = 128i32;
        let rows = 2i32;
        let signs = Array::from_slice(
            &(0..width)
                .map(|i| if i % 3 == 0 { -1.0f32 } else { 1.0 })
                .collect::<Vec<_>>(),
            &[width],
        );
        let mut words = vec![0u32; (rows * width / 16) as usize];
        for row in 0..rows as usize {
            for col in 0..width as usize {
                let code = ((row + col) % 3) as u32;
                words[row * (width as usize / 16) + col / 16] |= code << (2 * (col % 16));
            }
        }
        let scales = Array::from_slice(&[0.25f32, 0.5], &[rows, 1]);
        let biases = Array::from_slice(&[-0.25f32, -0.5], &[rows, 1]);
        let weight = Array::from_slice(&words, &[rows, width / 16]);
        let dense_rotated = dequantize(&weight, &scales, Some(&biases), GROUP_SIZE, BITS).unwrap();
        let x = Array::from_slice(
            &(0..width)
                .map(|i| (i as f32 - 50.0) / 64.0)
                .collect::<Vec<_>>(),
            &[1, width],
        );
        let rotated = block_hadamard(&x, &signs, width, false).unwrap();
        let expected = linear(&rotated, &dense_rotated, None).unwrap();
        let packed = PrismLinear::new(
            "test",
            weight.clone(),
            scales.clone(),
            biases.clone(),
            signs.clone(),
            width,
        )
        .unwrap()
        .forward(&x)
        .unwrap();
        assert!(packed
            .all_close(&expected, 1e-4, 1e-4, None)
            .unwrap()
            .item::<bool>());

        let ids = Array::from_slice(&[1i32, 0], &[1, 2]);
        let embedding =
            PrismEmbedding::new("embedding", weight, scales, biases, signs.clone(), width)
                .unwrap()
                .forward(&ids)
                .unwrap();
        let gathered = dense_rotated
            .take_axis(Array::from_slice(&[1i32, 0], &[2]), 0)
            .unwrap()
            .reshape(&[1, 2, width])
            .unwrap();
        let expected = block_hadamard(&gathered, &signs, width, true).unwrap();
        assert!(embedding
            .all_close(&expected, 1e-5, 1e-5, None)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn malformed_packed_metadata_fails_closed() {
        let weight = Array::from_slice(&[0u32; 8], &[1, 8]);
        let scales = Array::from_slice(&[1.0f32], &[1, 1]);
        let biases = Array::from_slice(&[-1.0f32], &[1, 1]);
        let bad_signs = Array::from_slice(&vec![1.0f32; 127], &[127]);
        let err = PrismLinear::new("bad", weight, scales, biases, bad_signs, 128).unwrap_err();
        assert!(err.to_string().contains("geometry mismatch"), "{err}");
    }
}
