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

/// The input width `in` of an MLX affine **8-bit** triple in the layout
/// [`QuantizedLinear::from_mlx_affine_q8`] accepts — a `U32` code matrix `[out, in / 4]` (four
/// 8-bit codes per word) beside `[out, in / group_size]` scales and biases — or `None` for any
/// other layout, which it refuses (a 4-bit pack, eight codes per word, among them). Load admission
/// prices the Q8_0 repack by this same test (sc-24140), so it prices no copy for a triple the
/// loader refuses.
pub(crate) fn mlx_affine_q8_in_dim(
    weight_is_u32: bool,
    weight: (usize, usize),
    scales: (usize, usize),
    biases: (usize, usize),
    group_size: usize,
) -> Option<usize> {
    let (out_dim, packed_cols) = weight;
    let in_dim = scales.1.checked_mul(group_size)?;
    (weight_is_u32
        && group_size != 0
        && scales.0 == out_dim
        && biases == scales
        && packed_cols.checked_mul(4) == Some(in_dim))
    .then_some(in_dim)
}

/// The GGML block types a prepared snapshot persists (sc-19375). Their block byte sizes are
/// pairwise distinct — Q4_0 18 B / 32 weights, Q8_0 34 B / 32, Q4_K 144 B / 256 — which is what
/// makes a stored block tensor self-describing.
const STORED_GGML: [GgmlDType; 3] = [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K];

/// The GGML block type and logical `(rows, cols)` of a **stored GGML block tensor** — a prepared
/// Q4 / Q8 snapshot's on-disk projection form (sc-19375): a `U8` tensor `[rows, blocks_per_row,
/// block_bytes]` holding the raw GGML blocks of a `[rows, cols]` weight, row-major (GGML blocks run
/// along the input dimension, so each row is `blocks_per_row` whole blocks). A stacked weight (the
/// qwen3_5 MoE experts, `[experts, rows, cols]`) is stored `[experts, rows, blocks_per_row,
/// block_bytes]`, and its `rows` here count every expert's. The block type is the one of
/// [`STORED_GGML`] whose block size is `block_bytes`. `None` for any other tensor. The loader,
/// load admission, the weight reader and the NVFP4 gate classify a tensor by this one rule, from
/// its dtype and shape alone.
pub(crate) fn ggml_block_storage(
    is_u8: bool,
    shape: &[usize],
) -> Option<(GgmlDType, usize, usize)> {
    let [lead @ .., rows, blocks, block_bytes] = shape else {
        return None;
    };
    if !is_u8 {
        return None;
    }
    let dtype = STORED_GGML
        .into_iter()
        .find(|d| d.type_size() == *block_bytes)?;
    let rows = lead.iter().try_fold(*rows, |n, &d| n.checked_mul(d))?;
    Some((dtype, rows, blocks.checked_mul(dtype.block_size())?))
}

/// Whether `t` is a stored GGML block tensor ([`ggml_block_storage`]).
pub(crate) fn is_ggml_block_tensor(t: &Tensor) -> bool {
    ggml_block_storage(t.dtype() == DType::U8, t.dims()).is_some()
}

/// Serialize a 2-D [`QTensor`] of a persisted block type (Q4_0 / Q8_0 / Q4_K) into its stored
/// GGML block tensor on the CPU: a `U8` `[rows, blocks_per_row, block_bytes]` tensor of the raw
/// GGML blocks, row-major, which [`from_ggml_block_tensor`] reads back. GGML quantizes each block
/// on its own, so the blocks of any row slice are byte-for-byte those of quantizing that slice
/// alone — which is what lets a loader carve fused parts and stacked experts out of one stored
/// tensor.
pub fn to_ggml_block_tensor(qt: &QTensor) -> Result<Tensor> {
    let dtype = qt.dtype();
    let (rows, cols) = qt.shape().dims2()?;
    if !STORED_GGML.contains(&dtype) || !cols.is_multiple_of(dtype.block_size()) {
        return Err(crate::error::Error::Unsupported(format!(
            "ggml block storage: cannot persist a {dtype:?} [{rows}, {cols}] tensor"
        )));
    }
    let bytes = qt.data()?.into_owned();
    let blocks = cols / dtype.block_size();
    Ok(Tensor::from_vec(
        bytes,
        (rows, blocks, dtype.type_size()),
        &Device::Cpu,
    )?)
}

/// Rebuild the [`QTensor`] a rank-3 stored GGML block tensor holds, directly on `device`: the
/// blocks are used exactly as stored — never dequantized and re-quantized. A stacked (rank-4)
/// tensor is sliced to one expert first.
pub fn from_ggml_block_tensor(stored: &Tensor, device: &Device) -> Result<QTensor> {
    let storage = ggml_block_storage(stored.dtype() == DType::U8, stored.dims());
    let (Some((dtype, rows, cols)), 3) = (storage, stored.rank()) else {
        return Err(crate::error::Error::Config(format!(
            "ggml block storage: {:?} {:?} is not a rank-3 stored GGML block tensor",
            stored.dtype(),
            stored.shape()
        )));
    };
    let bytes = stored
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    // candle reinterprets the byte slice as a block slice (2-byte aligned `f16` fields), so hand
    // it u64-backed storage rather than rely on the byte vector's allocation alignment.
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    // SAFETY: `words` owns `words.len() * 8 >= bytes.len()` initialized bytes, and u8 has no
    // alignment requirement; the byte view does not outlive `words`.
    let aligned =
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) };
    aligned.copy_from_slice(&bytes);
    Ok(candle_core::quantized::ggml_file::qtensor_from_ggml(
        dtype,
        aligned,
        vec![rows, cols],
        device,
    )?)
}

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
    /// Wrap a tensor which was already stored in a GGUF quantized representation. This preserves
    /// the compact resident payload and lets [`QMatMul`] dispatch the matching CPU/CUDA kernel;
    /// loading a Q8 projector must not materialize a second dense copy of the matrix.
    pub fn from_qtensor(weight: QTensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            inner: QuantizedWeight::Matmul(QMatMul::from_qtensor(weight)?),
            bias,
        })
    }

    /// Quantize a dense `[out, in]` weight (the input dim must be a multiple of `dtype`'s block
    /// size). `bias`, if present, is added after the matmul.
    ///
    /// `weight` may be a view — an expert sliced out of a stacked qwen3_5 MoE tensor, or one part
    /// of a fused Phi-3 `qkv_proj` / `gate_up_proj`. Candle's quantizer reads its source's storage
    /// from the start, ignoring the view's offset and extent, so the f32 source it is handed is
    /// always compacted first: a cast to f32 builds a fresh tensor, but an f32 weight (every
    /// weight on a host device, whose compute dtype is f32) would otherwise be passed through as
    /// the view itself and quantize the wrong rows — or trip candle's size check (sc-24140).
    pub fn quantize(weight: &Tensor, dtype: GgmlDType, bias: Option<Tensor>) -> Result<Self> {
        let source = match weight.dtype() {
            DType::F32 => weight.force_contiguous()?,
            _ => weight.to_dtype(DType::F32)?,
        };
        let qt = QTensor::quantize(&source, dtype)?;
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
        let (_, scale_cols) = scales.dims2()?;
        let Some(in_dim) = mlx_affine_q8_in_dim(
            weight.dtype() == DType::U32,
            (out_dim, packed_cols),
            scales.dims2()?,
            biases.dims2()?,
            group_size,
        ) else {
            return Err(crate::error::Error::Config(format!(
                "invalid MLX affine Q8 triple: weight {:?} {:?}, scales {:?}, biases {:?}, group {group_size}",
                weight.dtype(),
                weight.shape(),
                scales.shape(),
                biases.shape()
            )));
        };

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

    /// Resident bytes of the weight as stored (the GGML block payload, or the dense tensor
    /// `QMatMul` expanded a float-typed GGUF matrix into) plus the bias — the load telemetry's
    /// per-projection footprint (sc-24135).
    pub fn resident_bytes(&self) -> usize {
        let dense = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let weight = match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) | QuantizedWeight::Dequant(q) => {
                q.storage_size_in_bytes()
            }
            QuantizedWeight::Matmul(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => dense(t),
        };
        weight + self.bias.as_ref().map_or(0, dense)
    }

    /// Logical weight elements (`out · in`).
    pub fn weight_elems(&self) -> usize {
        match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) | QuantizedWeight::Dequant(q) => {
                q.shape().elem_count()
            }
            QuantizedWeight::Matmul(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => t.elem_count(),
        }
    }

    /// The GGML block dtype the weight is stored in, when it is block-quantized.
    pub fn ggml_dtype(&self) -> Option<GgmlDType> {
        match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) | QuantizedWeight::Dequant(q) => {
                Some(q.dtype())
            }
            QuantizedWeight::Matmul(_) => None,
        }
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

    /// sc-19375: a stored block is used exactly as stored. The hand-built Q8_0 blocks are **not**
    /// what candle's quantizer would write for their values (`d = 1.0` over codes no wider than
    /// ±10, where a requantize picks `d = 10 / 127` and rescales every code), so a loader that
    /// dequantized and re-quantized — which round-trips a canonical block to identical bytes —
    /// changes these bytes and fails here.
    #[test]
    fn stored_blocks_rebuild_byte_exact_even_when_non_canonical() {
        let one = half::f16::from_f32(1.0).to_le_bytes();
        let mut bytes = Vec::new();
        for row in 0..2i32 {
            for block in 0..2i32 {
                bytes.extend_from_slice(&one);
                bytes.extend((0..32i32).map(|i| (((i + row + block) % 21) - 10) as i8 as u8));
            }
        }
        let stored = Tensor::from_vec(bytes.clone(), (2, 2, 34), &Device::Cpu).unwrap();
        assert_eq!(
            ggml_block_storage(true, stored.dims()),
            Some((GgmlDType::Q8_0, 2, 64))
        );
        let qt = from_ggml_block_tensor(&stored, &Device::Cpu).unwrap();
        assert_eq!(qt.dtype(), GgmlDType::Q8_0);
        assert_eq!(qt.shape().dims(), &[2, 64]);
        assert_eq!(qt.data().unwrap().as_ref(), bytes.as_slice());
        // The blocks really are non-canonical: requantizing their values rewrites them.
        let requantized = QTensor::quantize(&qt.dequantize(&Device::Cpu).unwrap(), GgmlDType::Q8_0)
            .unwrap()
            .data()
            .unwrap()
            .into_owned();
        assert_ne!(requantized, bytes);
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
