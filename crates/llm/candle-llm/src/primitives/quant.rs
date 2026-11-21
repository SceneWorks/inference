//! Group-wise quantization (Q4 / Q8) for linear projections, via Candle's native quantized tensors.
//!
//! Moved to [`candle_layers::llm_projection`] (`SW-6957` / `SW-F335`) and re-exported here so all
//! internal decoders and external consumers remain 100% compatible.

pub use candle_layers::llm_projection::{
    from_ggml_block_tensor, ggml_block_storage, is_ggml_block_tensor, mlx_affine_in_dim,
    mlx_affine_q8_in_dim, to_ggml_block_tensor, Q8RowEmbedding, QuantizedLinear, QuantizedWeight,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::nn::linear;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Tensor};

    /// SW-7201: a Q8 row embedding gathers exactly the rows GGML's own Q8_0 dequantization
    /// produces from the stored blocks, sized as admission prices it; a non-Q8_0 table is
    /// refused.
    #[test]
    fn a_q8_row_embedding_dequantizes_the_stored_blocks_exactly() {
        let data: Vec<f32> = (0..6 * 64)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 61.0)
            .collect();
        let table = Tensor::from_vec(data, (6, 64), &Device::Cpu).unwrap();
        let qt = QTensor::quantize(&table, GgmlDType::Q8_0).unwrap();
        let stored = to_ggml_block_tensor(&qt).unwrap();
        let embedding = Q8RowEmbedding::from_stored("t", &stored, &Device::Cpu).unwrap();
        let ids = Tensor::from_vec(vec![4u32, 0, 5, 4], (2, 2), &Device::Cpu).unwrap();
        let rows = embedding.forward(&ids).unwrap();
        assert_eq!(rows.dims(), &[2, 2, 64]);
        let expected = qt
            .dequantize(&Device::Cpu)
            .unwrap()
            .index_select(&ids.flatten_all().unwrap(), 0)
            .unwrap()
            .reshape((2, 2, 64))
            .unwrap();
        let diff = (rows - expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0);
        assert_eq!(
            Some(embedding.resident_bytes()),
            Q8RowEmbedding::device_bytes(6, 64)
        );

        let q4 =
            to_ggml_block_tensor(&QTensor::quantize(&table, GgmlDType::Q4_0).unwrap()).unwrap();
        assert!(Q8RowEmbedding::from_stored("t", &q4, &Device::Cpu).is_err());
    }

    /// Quantized matmul should approximate the dense linear it replaces (Q8_0, in=256 = 8 blocks).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::quant::tests::quantized_matmul_approximates_linear_q8");
    }

    /// sc-26266: a weight-only projection holds the same GGML blocks as the `QMatMul` one (same
    /// resident bytes) but multiplies the activation densely against the dequantized weight — so
    /// its forward is exactly `x · deq(W)ᵀ`, even beside a 1e4 outlier channel, where the
    /// activation-quantizing path visibly is not.
    ///
    /// *Mutation that reds this:* `quantize_dequantized` building a `QuantizedWeight::Matmul` (for
    /// any one of Q8_0 / Q4_0 / Q4_K), or `dequantizes_forward` not reporting it.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn a_dequantized_forward_projection_never_quantizes_the_activation() {
        // Q4_K is the shipped q4 tier on 256-aligned widths (the real Qwen3-VL tower); its
        // super-block needs `in` = 256.
        for (dtype, inn) in [
            (GgmlDType::Q8_0, 64usize),
            (GgmlDType::Q4_0, 64),
            (GgmlDType::Q4K, 256),
        ] {
            weight_only_forward_is_exact(dtype, inn);
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::quant::tests::a_dequantized_forward_projection_never_quantizes_the_activation");
    }

    fn weight_only_forward_is_exact(dtype: GgmlDType, inn: usize) {
        let out = 8usize;
        let wdata: Vec<f32> = (0..out * inn)
            .map(|i| ((i * 7 % 13) as f32 / 13.0) - 0.5)
            .collect();
        let w = Tensor::from_vec(wdata, (out, inn), &Device::Cpu).unwrap();
        let mut xdata: Vec<f32> = (0..2 * inn)
            .map(|i| ((i * 5 % 11) as f32 / 11.0) - 0.5)
            .collect();
        xdata[3] = 1.0e4;
        let x = Tensor::from_vec(xdata, (2, inn), &Device::Cpu).unwrap();
        let rows = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let weight_only = QuantizedLinear::quantize_dequantized(&w, dtype, None).unwrap();
        let matmul = QuantizedLinear::quantize(&w, dtype, None).unwrap();
        assert_eq!(weight_only.resident_bytes(), matmul.resident_bytes());
        assert_eq!(weight_only.ggml_dtype(), Some(dtype));
        assert!(weight_only.dequantizes_forward() && !matmul.dequantizes_forward());
        let deq = QTensor::quantize(&w, dtype)
            .unwrap()
            .dequantize(&Device::Cpu)
            .unwrap();
        let exact = rows(linear(&x, &deq, None).unwrap());
        assert_eq!(rows(weight_only.forward(&x).unwrap()), exact, "{dtype:?}");
        let fast = rows(matmul.forward(&x).unwrap());
        let drift = exact
            .iter()
            .zip(&fast)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(drift > 0.0, "{dtype:?}: the QMatMul path matched exactly");
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
    #[ignore = "model inference; explicit model workflow only"]
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

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::quant::tests::mlx_affine_q8_reconstructs_lsb_bytes_bias_and_input_groups");
    }

    /// sc-24444: a 4-bit MLX affine triple (eight little-endian nibbles per word — a published
    /// companion MTP head's layout) reconstructs its exact affine grid before the Q8_0 repack, and
    /// the 8-bit reading of the same words is refused by geometry.
    #[test]
    fn mlx_affine_q4_reconstructs_lsb_nibbles_bias_and_input_groups() {
        let dev = Device::Cpu;
        let (out, inn, group) = (2usize, 64usize, 32usize);
        let codes: Vec<u8> = (0..out * inn)
            .map(|i| [0, 15, 1, 14, 7, 8, 3, 12][i % 8] ^ ((i / 8) % 2) as u8)
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|chunk| {
                chunk.iter().enumerate().fold(0u32, |word, (index, code)| {
                    word | ((*code as u32) << (index * 4))
                })
            })
            .collect();
        let scale_values = vec![0.1f32, 0.05, 0.02, 0.08];
        let bias_values = vec![-0.75f32, 0.25, -0.1, 0.4];
        let weight = Tensor::from_vec(words, (out, inn / 8), &dev).unwrap();
        let scales = Tensor::from_vec(scale_values.clone(), (out, inn / group), &dev).unwrap();
        let biases = Tensor::from_vec(bias_values.clone(), (out, inn / group), &dev).unwrap();

        let packed =
            QuantizedLinear::from_mlx_affine(&weight, &scales, &biases, None, group, 4, &dev)
                .unwrap();
        let resident = match &packed.inner {
            QuantizedWeight::Dequant(weight) => weight.dequantize(&dev).unwrap(),
            QuantizedWeight::Matmul(_) => panic!("MLX affine Q4 must use the packed-source path"),
        };
        let got = resident.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (index, (actual, &code)) in got.iter().zip(&codes).enumerate() {
            let (row, col) = (index / inn, index % inn);
            let g = row * (inn / group) + col / group;
            let expected = scale_values[g] * f32::from(code) + bias_values[g];
            assert!(
                (actual - expected).abs() < 0.01,
                "index {index}: {actual} vs affine source {expected}"
            );
        }
        assert!(
            QuantizedLinear::from_mlx_affine_q8(&weight, &scales, &biases, None, group, &dev)
                .is_err()
        );
    }
}

/// sc-24446: a CUDA quantize-on-load leaves no garbage in the quantized storage's row padding.
/// candle's MMQ kernels (any quantized matmul over more than 8 rows) load `MMQ_ITER_K` = 256
/// elements of every row, so a row narrower than that — the stage-2 synthetic fixture's 32/64,
/// a real 5504-wide `down_proj`'s ragged last tile — reads blocks out of the padding after the
/// last row. Their quants meet zero activations, but each block's f16 scale is still multiplied
/// in: an inf/NaN there made the YuE stage-2 test's logits NaN, intermittently, as whatever the
/// pool last held there changed.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::cuda_backend::cudarc;
    use candle_core::quantized::{GgmlDType, QMatMul};
    use candle_core::{Device, Tensor};

    /// The quantizer's storage is `data` plus 512 elements' worth of blocks of padding.
    fn padded_bytes(dtype: GgmlDType, elems: usize) -> (usize, usize) {
        let data = elems / dtype.block_size() * dtype.type_size();
        (data, data + 512 / dtype.block_size() * dtype.type_size())
    }

    fn uniform(seed: u64, len: usize) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Fill the device's free pool with all-ones bytes (an f16 `0xFFFF` is a NaN) in exactly the
    /// size the quantizer is about to allocate, then free it, so its allocation is handed poisoned
    /// memory rather than fresh (zeroed) pages.
    fn poison_pool(dev: &candle_core::CudaDevice, bytes: usize) {
        let poison = vec![0xFFu8; bytes];
        let slices: Vec<_> = (0..256)
            .map(|_| dev.clone_htod(&poison).expect("poison buffer"))
            .collect();
        drop(slices);
    }

    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn quantize_on_load_zeroes_the_padding_mmq_reads() {
        let dev = crate::device::new_cuda_for_test().expect("a CUDA device");
        let Device::Cuda(cuda) = &dev else {
            unreachable!("new_cuda_for_test opens a CUDA device")
        };
        // 64 output rows of 32 inputs (one Q8_0 / Q4_0 block per row), 16 activation rows: the
        // MMQ path, whose tile reads 7 blocks past the end of the last weight row.
        let (out, inn, rows) = (64usize, 32usize, 16usize);
        for (seed, dtype) in [(1u64, GgmlDType::Q8_0), (2, GgmlDType::Q4_0)] {
            let w_host =
                Tensor::from_vec(uniform(seed, out * inn), (out, inn), &Device::Cpu).unwrap();
            let x_host =
                Tensor::from_vec(uniform(seed + 10, rows * inn), (rows, inn), &Device::Cpu)
                    .unwrap();
            let w = w_host.to_device(&dev).unwrap();
            let (data, padded) = padded_bytes(dtype, out * inn);
            // `quantize` compacts an f32 weight first (one `out·in·4`-byte copy), then allocates
            // the padded storage: poison both sizes.
            poison_pool(cuda, out * inn * 4);
            poison_pool(cuda, padded);
            let q = QuantizedLinear::quantize(&w, dtype, None).unwrap();

            // The padding itself is zero.
            let QuantizedWeight::Matmul(QMatMul::QTensor(qt)) = &q.inner else {
                panic!("{dtype:?} quantizes to a GGML block tensor");
            };
            assert_eq!(qt.storage_size_in_bytes(), data);
            let base = qt.device_ptr().unwrap() as u64;
            let mut tail = vec![0xAAu8; padded - data];
            cuda.cuda_stream().context().bind_to_thread().unwrap();
            // SAFETY: `[base, base + padded)` is the live allocation `qt` owns; the legacy-stream
            // device orders this synchronous copy after the quantizer's upload.
            unsafe { cudarc::driver::result::memcpy_dtoh_sync(&mut tail, base + data as u64) }
                .unwrap();
            assert!(
                tail.iter().all(|&b| b == 0),
                "{dtype:?}: {} non-zero padding bytes",
                tail.iter().filter(|&&b| b != 0).count()
            );

            // And the MMQ forward over it is finite and matches the dequantized weight's matmul.
            let y = q
                .forward(&x_host.to_device(&dev).unwrap())
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            let deq = qt.dequantize(&Device::Cpu).unwrap();
            let want = x_host
                .matmul(&deq.t().unwrap())
                .unwrap()
                .to_vec2::<f32>()
                .unwrap();
            for (r, (got, want)) in y.iter().zip(&want).enumerate() {
                for (c, (g, w)) in got.iter().zip(want).enumerate() {
                    assert!(g.is_finite(), "{dtype:?}: output [{r}, {c}] is {g}");
                    assert!((g - w).abs() < 0.05, "{dtype:?}: [{r}, {c}] {g} vs {w}");
                }
            }
        }

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED primitives::quant::cuda_tests::quantize_on_load_zeroes_the_padding_mmq_reads");
    }
}
