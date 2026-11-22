//! Quantization primitives re-exported from [`candle_layers::quant`] plus `candle-gen` dotted-adapter
//! and sidecar loaders (`SW-6955`).

pub mod act_probe;
pub mod adapt;
pub mod adapters;
pub mod convrot;
pub mod convrot_int8;
#[cfg(feature = "cuda")]
pub mod eight_bit_linear;
pub mod nvfp4_kitchen;
pub mod repack;
pub mod sidecar;
pub mod w4a8_int8;

pub use adapters::{install_dotted_adapters, AdditiveAdapterReport};
pub use candle_layers::quant::*;
pub use sidecar::PackedWeightSidecars;

/// Decode the affine group-64 grid used by diffusers VAE attention projections.
pub fn dequant_packed_to_dense(
    vb: &candle_nn::VarBuilder,
    base: &str,
    device: &candle_core::Device,
    dtype: candle_core::DType,
) -> candle_core::Result<candle_core::Tensor> {
    let vb = vb.pp(base);
    let codes = vb.get_unchecked_dtype("weight", candle_core::DType::U32)?;
    let scales = vb.get_unchecked_dtype("scales", candle_core::DType::F32)?;
    let biases = vb.get_unchecked_dtype("biases", candle_core::DType::F32)?;
    dequant_vae_grid(&codes, &scales, &biases)?
        .to_device(device)?
        .to_dtype(dtype)
}

pub(crate) fn dequant_vae_grid(
    codes: &candle_core::Tensor,
    scales: &candle_core::Tensor,
    biases: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    let cols = scales.dims2()?.1;
    if cols == 0 {
        candle_core::bail!("VAE projection: affine scales must have a positive group count");
    }
    let bits = mlx_packed_bits(codes.dims2()?.1, cols);
    match bits {
        4 => dequant_mlx_q4_reference(codes, scales, biases),
        8 => dequant_mlx_q8(codes, scales, biases),
        _ => candle_core::bail!("VAE projection: unsupported affine bit-width {bits}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::tensor_cosine;
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::safetensors::MmapedSafetensors;
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{Embedding, Linear as CandleNnLinear, Module, VarBuilder};
    use gen_core::Quant;

    // ---- helpers -----------------------------------------------------------------------------

    /// Test-side MLX packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles).
    fn pack_mlx_q4(codes: &[u8]) -> Vec<u32> {
        codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect()
    }

    /// Build a real MLX Q4 packed triple for an `[out, in]` weight (group 64) with f16-exact,
    /// per-group-distinct scales/biases and position-dependent codes — the packed-parts fixtures the
    /// `from_packed` / detect loaders consume, plus the exact affine grid they must reproduce.
    fn q4_fixture(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / MLX_GROUP_SIZE;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / MLX_GROUP_SIZE;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / MLX_GROUP_SIZE;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let wq = Tensor::from_vec(pack_mlx_q4(&codes), (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    // ---- packed-vs-dense forward parity ------------------------------------------------------

    /// A packed-loaded `QLinear` forward matches a *dense* linear built from the SAME affine grid the
    /// pack represents — bit-exact (the repack is lossless, and both forwards dequant-to-dense-matmul).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_qlinear_forward_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, grid) = q4_fixture(out_dim, in_dim);

        let packed = QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        assert!(packed.is_quantized());
        let dense = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        )));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = tensor_cosine(&packed.forward(&x)?, &dense.forward(&x)?) as f32;
        assert!(cos > 0.99999, "packed vs dense-grid cosine {cos:.6}");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::packed_qlinear_forward_matches_dense_grid");
        Ok(())
    }

    /// [`QLinear::from_qtensor_dequant`] (sc-12735) ingests a **native GGUF k-quant** `QTensor` and keeps
    /// it **quantized-resident**: the block type survives ([`QLinear::quant_dtype`] is `Some(Q4K)`, NOT
    /// dequantized to a dense weight at load), the forward is the sc-7702-safe
    /// [`MatmulStrategy::DequantDense`] path (dequant-on-matmul, not int8-fast), and it approximates the
    /// dense weight it was quantized from (Q4_K tolerance). The resident-not-dense guarantee at the core.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn from_qtensor_dequant_keeps_kquant_resident_and_forwards() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 256usize); // Q4_K needs `in` a multiple of 256.
        let w = Tensor::randn(0f32, 0.1f32, (out_dim, in_dim), &dev)?;
        let bias = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;
        // A resident k-quant QTensor — the shape a native GGUF loader hands over WITHOUT dequantizing.
        let qt = std::sync::Arc::new(QTensor::quantize(&w, GgmlDType::Q4K)?);

        let ql = QLinear::from_qtensor_dequant(qt, Some(bias.clone()));
        assert!(
            ql.is_quantized(),
            "the ingested QTensor must load quantized, not dense"
        );
        assert_eq!(
            ql.quant_dtype(),
            Some(GgmlDType::Q4K),
            "the resident weight must stay Q4_K (never dequantized to a dense [out,in] at load)"
        );
        assert_eq!(
            ql.matmul_strategy(),
            Some(MatmulStrategy::DequantDense),
            "the forward must dequant-on-matmul (sc-7702-safe), not candle's int8 fast path"
        );

        // The dequant-on-forward output approximates the dense weight it was quantized from + bias.
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let dense = CandleNnLinear::new(w, Some(bias));
        let cos =
            tensor_cosine(&ql.forward(&x)?, &__model_proof_output!(dense.forward(&x))?) as f32;
        assert!(cos > 0.99, "Q4_K dequant-forward vs dense cosine {cos:.5}");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::from_qtensor_dequant_keeps_kquant_resident_and_forwards");
        Ok(())
    }

    /// A packed-loaded `QEmbedding` reproduces the affine grid rows exactly (index-select over the
    /// dequantized table == the dense grid table).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_qembedding_matches_dense_grid() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32, 128);
        let (wq, s, b, grid) = q4_fixture(vocab, hidden);

        let packed = QEmbedding::from_packed(&wq, &s, &b, &dev)?;
        assert!(packed.is_quantized());
        let dense = QEmbedding::Dense(Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        ));

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let (p, d) = (packed.forward(&idx)?, dense.forward(&idx)?);
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "packed embedding deviates from dense grid");
        #[cfg(feature = "model-inference-proof")]
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::packed_qembedding_matches_dense_grid"
        );
        Ok(())
    }

    /// Both packed installation routes retain exactly one full-vocabulary representation: the host
    /// byte table used for future row reads. The temporary device `QTensor` built by a lookup is not
    /// retained after installation.
    #[test]
    fn packed_qembedding_resident_storage_is_host_rows_only() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32, 128);
        let (wq, s, b, _) = q4_fixture(vocab, hidden);
        let packed =
            QEmbedding::from_packed_dtype_gs(&wq, &s, &b, &dev, DType::BF16, MLX_GROUP_SIZE)?;
        assert_eq!(
            packed.resident_storage(),
            Some(QEmbeddingStorage {
                host_packed_bytes: vocab * hidden / GgmlDType::Q4_1.block_size()
                    * GgmlDType::Q4_1.type_size(),
                device_table_bytes: 0,
            }),
            "repacked MLX rows must not leave a duplicate device table resident"
        );

        let table = Tensor::randn(0f32, 1f32, (vocab, hidden), &dev)?;
        let qtensor = QTensor::quantize(&table, GgmlDType::Q4_0)?;
        let host_packed_bytes = qtensor.data()?.len();
        let packed = QEmbedding::from_qtensor(qtensor, DType::F32)?;
        assert_eq!(
            packed.resident_storage(),
            Some(QEmbeddingStorage {
                host_packed_bytes,
                device_table_bytes: 0,
            }),
            "already-device-format rows must also release their source table after installation"
        );
        Ok(())
    }

    /// Packed lookup dequantizes only the distinct selected rows, then uses its inverse map to
    /// restore repeated tokens and the dense embedding's arbitrary index shape exactly.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_qembedding_deduplicates_rows_and_restores_duplicates() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (256, 128);
        let (wq, s, b, grid) = q4_fixture(vocab, hidden);
        let packed = QEmbedding::from_packed(&wq, &s, &b, &dev)?;
        let dense = QEmbedding::Dense(Embedding::new(
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
            hidden,
        ));
        let indexes = Tensor::from_vec(vec![7u32, 99, 7, 255, 99, 7], (2, 3), &dev)?;

        let work = packed.lookup_work(&indexes)?;
        assert_eq!(work.selected_rows, 3, "each duplicate must share one row");
        assert_eq!(
            work.dequantized_bytes,
            3 * hidden * DType::F32.size_in_bytes(),
            "the bounded dense allocation is exactly the selected rows"
        );
        assert!(
            work.dequantized_bytes < vocab * hidden * DType::F32.size_in_bytes(),
            "the lookup must not allocate a dense vocabulary table"
        );

        let (got, expected) = (packed.forward(&indexes)?, dense.forward(&indexes)?);
        assert_eq!(got.dims(), expected.dims());
        assert_eq!(got.dtype(), expected.dtype());
        assert_eq!(
            (got.sub(&expected)?).abs()?.max_all()?.to_scalar::<f32>()?,
            0.0,
            "row selection must restore every duplicate at its original position"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::packed_qembedding_deduplicates_rows_and_restores_duplicates");
        Ok(())
    }

    /// The allocation probe is intentionally independent of vocabulary size: same request, same
    /// number of distinct rows, same transient dense bytes.
    #[test]
    fn packed_qembedding_work_scales_with_selected_rows_not_vocabulary() -> Result<()> {
        let dev = Device::Cpu;
        let hidden = 128;
        let indexes = Tensor::from_vec(vec![3u32, 17, 3, 17, 42], (5,), &dev)?;
        let make = |vocab| -> Result<QEmbedding> {
            let (wq, s, b, _) = q4_fixture(vocab, hidden);
            QEmbedding::from_packed(&wq, &s, &b, &dev)
        };
        let small = make(64)?.lookup_work(&indexes)?;
        let large = make(1024)?.lookup_work(&indexes)?;
        assert_eq!(small, large);
        assert_eq!(small.selected_rows, 3);
        Ok(())
    }

    /// The already-device-format sidecar route (used by Krea) takes the same row-selective path as
    /// an MLX affine triple, rather than silently retaining the old whole-table dequantization.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn qtensor_qembedding_deduplicates_rows_with_dense_parity() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (128, 128);
        let dense_table = Tensor::from_vec(
            (0..vocab * hidden)
                .map(|i| (i as f32 * 0.001 - 3.0).sin())
                .collect::<Vec<_>>(),
            (vocab, hidden),
            &dev,
        )?;
        let qtensor = QTensor::quantize(&dense_table, GgmlDType::Q4_0)?;
        let reference = qtensor.dequantize(&dev)?;
        let packed = QEmbedding::from_qtensor(qtensor, DType::F32)?;
        let dense = QEmbedding::Dense(Embedding::new(reference, hidden));
        let indexes = Tensor::from_vec(vec![4u32, 4, 77, 12, 77], (5,), &dev)?;

        assert_eq!(packed.lookup_work(&indexes)?.selected_rows, 3);
        let (got, expected) = (packed.forward(&indexes)?, dense.forward(&indexes)?);
        assert_eq!(got.dims(), expected.dims());
        assert_eq!(got.dtype(), expected.dtype());
        assert_eq!(
            (got.sub(&expected)?).abs()?.max_all()?.to_scalar::<f32>()?,
            0.0
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::qtensor_qembedding_deduplicates_rows_with_dense_parity");
        Ok(())
    }

    /// The packed `QEmbedding` forward output dtype matches the dense embedding path's — a bf16
    /// dense table yields bf16 rows, so the packed path (loaded with the same `vb.dtype()`) must too.
    /// The dense path here goes through the same VarBuilder-detect loader (`embedding`), so it holds
    /// the `[base].weight` in the vb's bf16 dtype exactly as a real bf16 text-encoder would.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn packed_qembedding_forward_dtype_matches_dense() -> Result<()> {
        let dev = Device::Cpu;
        let (vocab, hidden) = (32, 128);
        let (wq, s, b, grid) = q4_fixture(vocab, hidden);

        // A packed table and a dense table, both written to safetensors and loaded through the
        // `embedding` detect-loader at bf16 vb dtype — the dense table is the reference dtype path.
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        map.insert("emb.weight".into(), wq);
        map.insert("emb.scales".into(), s);
        map.insert("emb.biases".into(), b);
        map.insert(
            "dense.weight".into(),
            Tensor::from_vec(grid, (vocab, hidden), &dev)?,
        );

        let tmp_guard = tempfile::tempdir().unwrap();
        let tmp = tmp_guard.path().join("sc9086_emb_dtype.safetensors");
        candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::BF16, dev.clone());

        let packed = embedding(&vb, "emb", vocab, hidden)?;
        assert!(
            packed.is_quantized(),
            "`.scales` present ⇒ packed embedding"
        );
        let dense = embedding(&vb, "dense", vocab, hidden)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense embedding");

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let p = __model_proof_output!(packed.forward(&idx))?;
        let d = __model_proof_output!(dense.forward(&idx))?;
        assert_eq!(d.dtype(), DType::BF16, "dense bf16 embedding yields bf16");
        assert_eq!(
            p.dtype(),
            d.dtype(),
            "packed embedding forward dtype must match the dense path (dtype parity)"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::packed_qembedding_forward_dtype_matches_dense");
        Ok(())
    }

    // ---- idempotent quantize ------------------------------------------------------------------

    /// `quantize` is a no-op on an already-packed `QLinear` — it must NOT re-quantize (double-quantize)
    /// a weight that loaded packed. The stored `Q4_1` weight and the forward stay unchanged.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn quantize_is_noop_on_packed() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, _grid) = q4_fixture(out_dim, in_dim);

        let mut packed = QLinear::from_packed(&wq, &s, &b, None, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let before = packed.forward(&x)?;

        // The weight is Q4_1 (packed), not the Q4_0 a load-time quantize would emit.
        match &packed {
            QLinear::Quantized { weight, .. } => assert_eq!(weight.dtype(), GgmlDType::Q4_1),
            _ => panic!("expected packed Quantized"),
        }

        packed.quantize(Quant::Q4)?; // must no-op, not re-quantize to Q4_0
        match &packed {
            QLinear::Quantized { weight, .. } => assert_eq!(
                weight.dtype(),
                GgmlDType::Q4_1,
                "quantize re-quantized a packed weight (Q4_1 → {:?})",
                weight.dtype()
            ),
            _ => panic!("quantize turned a packed linear dense"),
        }
        let after = packed.forward(&x)?;
        let dev_max = (before.sub(&after)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "no-op quantize changed the forward");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::quantize_is_noop_on_packed");
        Ok(())
    }

    /// **sc-11028 regression.** `QTensor::quantize{,_onto}` reads the tensor's RAW backing storage
    /// (its `flatten_all` is a zero-copy reshape on a strides-contiguous layout), ignoring a view's
    /// start offset. Folding a dim-0 narrow (a fused-qkv row chunk — `.contiguous()` is a no-op on
    /// it) therefore used to quantize the WRONG rows: flux2's ComfyUI in-place DiT loaded every
    /// double block's to_k/to_v as a copy of to_q — an incoherent render while per-layer weight
    /// parity read ~1.0 (the view API is offset-aware; the fold was not). The fold must
    /// materialize offset views into owned storage first: each folded chunk's forward must track
    /// the matching chunk of the dense fused weight (Q8, near-lossless) — chunks 1/2 fail without
    /// the guard, chunk 0 (offset 0) vacuously passes.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn fold_materializes_offset_views() -> Result<()> {
        let dev = Device::Cpu;
        let (d, in_dim) = (64usize, 32usize);
        let fused = Tensor::randn(0f32, 1f32, (3 * d, in_dim), &dev)?;
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        for (i, name) in ["q", "k", "v"].iter().enumerate() {
            // The trap shape: a dim-0 narrow keeps contiguous strides, so `.contiguous()` no-ops
            // and the chunk stays a VIEW into `fused` at storage offset i·d·in_dim.
            let view = fused.narrow(0, i * d, d)?.contiguous()?;
            assert!(
                view.is_contiguous(),
                "narrow must look contiguous (the trap)"
            );
            let dense_ref =
                __model_proof_output!(CandleNnLinear::new(view.copy()?, None).forward(&x))?;
            for (label, onto) in [("quantize", false), ("quantize_onto", true)] {
                let mut lin =
                    QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(view.clone(), None)));
                if onto {
                    lin.quantize_onto(Quant::Q8, &dev)?;
                } else {
                    lin.quantize(Quant::Q8)?;
                }
                let cos = tensor_cosine(&lin.forward(&x)?, &dense_ref) as f32;
                assert!(
                    cos > 0.999,
                    "chunk {name} via {label}: cos {cos:.6} — an offset view quantized the wrong \
                     rows (sc-11028)"
                );
            }
        }
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::fold_materializes_offset_views");
        Ok(())
    }

    /// The dense→quantize path stays idempotent too (a second `quantize` is a no-op, not a panic).
    #[test]
    fn quantize_dense_then_idempotent() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(w, None)));
        lin.quantize(Quant::Q8)?;
        assert!(lin.is_quantized());
        lin.quantize(Quant::Q8)?; // no-op, must not error
        assert!(matches!(lin, QLinear::Quantized { bias: None, .. }));
        Ok(())
    }

    /// `quantize_dequant_onto` folds a dense projection to the **dequant-dense** arm (sc-7702-safe)
    /// onto an explicit device — the SD3.5 CPU-stage path (sc-8504). It must NOT take the int8-fast
    /// arm `quantize_onto` uses, and its `Q4_0`/`Q8_0` blocks (and thus forward) must be bit-identical
    /// to the in-place [`QLinear::quantize`] fold of the same weight.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn quantize_dequant_onto_is_dequant_dense_and_matches_in_place() -> Result<()> {
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 32);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, (out_dim,), &dev)?;

        for quant in [Quant::Q4, Quant::Q8] {
            // In-place dequant-dense fold (the reference).
            let mut in_place = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(
                w.clone(),
                Some(b.clone()),
            )));
            in_place.quantize(quant)?;

            // CPU-stage → dequant-dense onto the SAME device.
            let mut staged = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(
                w.clone(),
                Some(b.clone()),
            )));
            staged.quantize_dequant_onto(quant, &dev)?;
            assert_eq!(
                staged.matmul_strategy(),
                Some(MatmulStrategy::DequantDense),
                "quantize_dequant_onto must use the sc-7702-safe dequant-dense arm, not int8-fast"
            );

            // The forwards are bit-identical (same f32 CPU source → same GGUF blocks → same dequant).
            let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
            let a = in_place.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            let c = staged.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
            for (p, r) in a.iter().zip(c.iter()) {
                assert_eq!(
                    p.to_bits(),
                    r.to_bits(),
                    "{quant:?} quantize_dequant_onto forward differs from in-place quantize"
                );
            }
        }
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::quantize_dequant_onto_is_dequant_dense_and_matches_in_place");
        Ok(())
    }

    // ---- PackedConfig detect ------------------------------------------------------------------

    #[test]
    fn packed_config_detects_quantization_block() {
        let packed = serde_json::json!({ "quantization": { "bits": 4, "group_size": 64 } });
        assert_eq!(
            PackedConfig::from_config(&packed),
            Some(PackedConfig {
                bits: 4,
                group_size: 64
            })
        );
        let dense = serde_json::json!({ "hidden_size": 2048 });
        assert_eq!(PackedConfig::from_config(&dense), None);
    }

    /// **`quantization.bits` present but `group_size` absent ⇒ default to 64, NOT silent dense
    /// (sc-9410).** A packed component missing `group_size` still stores u32 codes; degrading it to a
    /// dense read would load garbage. The existing group-64 / group-32 behavior stays byte-identical —
    /// only the *absent* case changed (it used to return `None`).
    #[test]
    fn packed_config_defaults_absent_group_size_to_64() {
        let no_gs = serde_json::json!({ "quantization": { "bits": 4 } });
        assert_eq!(
            PackedConfig::from_config(&no_gs),
            Some(PackedConfig {
                bits: 4,
                group_size: MLX_GROUP_SIZE as i32
            }),
            "absent group_size must default to the MLX group size (64), not degrade to dense"
        );
        // Explicit group sizes are unchanged.
        assert_eq!(
            PackedConfig::from_config(
                &serde_json::json!({ "quantization": { "bits": 4, "group_size": 32 } })
            ),
            Some(PackedConfig {
                bits: 4,
                group_size: 32
            })
        );
        // Still `None` when there is nothing marking it packed (no `bits`).
        assert_eq!(
            PackedConfig::from_config(&serde_json::json!({ "quantization": { "group_size": 64 } })),
            None
        );
    }

    /// NVFP4 detect/select seam (sc-11041): a `quantization.format == "nvfp4"` config routes to
    /// [`MatmulStrategy::Nvfp4`]; any other packed tier to `DequantDense`; a dense component to `None`.
    #[test]
    fn detect_strategy_routes_nvfp4_vs_affine_vs_dense() {
        let nvfp4 = serde_json::json!({ "quantization": { "bits": 4, "format": "nvfp4" } });
        assert!(PackedConfig::is_nvfp4(&nvfp4));
        assert_eq!(
            PackedConfig::detect_strategy(&nvfp4),
            Some(MatmulStrategy::Nvfp4),
            "an nvfp4-format packed tier must route to the FP4 strategy"
        );
        // Case-insensitive marker.
        assert!(PackedConfig::is_nvfp4(
            &serde_json::json!({ "quantization": { "format": "NVFP4" } })
        ));

        // A plain MLX affine packed tier (bits/group_size, no format) → dequant-dense QLinear path.
        let affine = serde_json::json!({ "quantization": { "bits": 4, "group_size": 64 } });
        assert!(!PackedConfig::is_nvfp4(&affine));
        assert_eq!(
            PackedConfig::detect_strategy(&affine),
            Some(MatmulStrategy::DequantDense)
        );

        // A dense component (no quantization block) selects no strategy.
        let dense = serde_json::json!({ "hidden_size": 2048 });
        assert_eq!(PackedConfig::detect_strategy(&dense), None);
    }

    /// The in-place GGUF fold refuses [`MatmulStrategy::Nvfp4`] — NVFP4 weights come from the offline
    /// packer, not a `Q4_0`/`Q8_0` fold (sc-11041). A dense fold call that somehow selected it errors
    /// rather than silently mis-quantizing.
    #[test]
    fn fold_rejects_nvfp4_strategy() {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev).unwrap();
        let mut lin = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(w, None)));
        let err = lin
            .fold(
                Quant::Q4,
                None,
                MatmulStrategy::Nvfp4,
                QuantForward {
                    flatten_leading: false,
                    cast_back: true,
                },
                false,
            )
            .unwrap_err();
        assert!(
            format!("{err}").contains("Nvfp4"),
            "fold must reject MatmulStrategy::Nvfp4 with a clear message, got: {err}"
        );
    }

    // ---- packed-detect over a VarBuilder ------------------------------------------------------

    /// `lin` / `embedding` route to the packed path when the `.scales` sibling is present in the
    /// VarBuilder, and to the dense path otherwise — the one-call packed-detect contract. Uses an
    /// in-memory safetensors so no weights are needed.
    #[test]
    fn lin_and_embedding_packed_detect() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64, 128);
        let (wq, s, b, _grid) = q4_fixture(out_dim, in_dim);
        let mut map: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();
        map.insert("proj.weight".into(), wq);
        map.insert("proj.scales".into(), s);
        map.insert("proj.biases".into(), b);
        // A dense sibling with no `.scales`.
        map.insert(
            "dense.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let tmp = tmp.path().join("sc9086_detect.safetensors");
        candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let packed = lin(&vb, "proj", in_dim, out_dim, false)?;
        assert!(packed.is_quantized(), "`.scales` present ⇒ packed path");
        let dense = lin(&vb, "dense", in_dim, out_dim, false)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense path unchanged");

        let emb = embedding(&vb, "proj", out_dim, in_dim)?;
        assert!(emb.is_quantized(), "`.scales` present ⇒ packed embedding");

        Ok(())
    }

    // ---- sc-7702 outlier / no-fast_mmq regression --------------------------------------------

    /// **sc-7702 regression (CUDA only).** The shared `QLinear` forward must stay accurate for
    /// activations with massive outliers — the gpt-oss text features (±10⁴) that made a Q4 DiT render
    /// solid black through candle's int8 `QMatMul` fast path (`fast_mmq`, batch > 8): that path
    /// quantizes the *activation* to per-32-element `q8_1`, so one outlier sets a block scale that
    /// rounds every co-located channel to **zero**. The packed-load `QLinear` must NOT take that path;
    /// it dequantizes the weight to a *dense* matmul, keeping the activation full-precision. Here each
    /// block's outlier sits on a **zero-weight** channel (carries no signal — the reference is built
    /// purely from its block-mates): the dequant path tracks the f32 reference, the raw int8 path
    /// collapses to ~0. A revert to `QMatMul::forward` inside `QLinear::forward` fails the `> 0.99`
    /// assert. This exercises the shared **dequant-on-forward** compute path in `QLinear::forward` —
    /// the byte-identical path a packed `Q4_1` load and a load-time `Q4_0` quantize both feed (the
    /// weight is dequantized to a dense matmul; the int8 activation fast path stays off), so a
    /// `Q4_0` `QLinear` built via the load-time `quantize` covers the packed forward too. Skips on
    /// CPU (the int8 MMVQ/MMQ path is CUDA-only).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn q4_packed_forward_survives_outlier_activations() -> Result<()> {
        use candle_core::quantized::QMatMul;
        let dev = match crate::default_device() {
            Ok(d) if !matches!(d, Device::Cpu) => d,
            _ => {
                eprintln!(
                    "SKIP q4_packed_forward_survives_outlier_activations: needs a CUDA device"
                );
                #[cfg(feature = "model-inference-proof")]
                crate::model_proof::missing_required_work("required work omitted at crates/media/candle-gen/candle-gen/src/quant/mod.rs:2030 (q4_packed_forward_survives_outlier_activations): Selected learned GPU/real-weight case exits normally when its required device or model artifact is unavailable; downstream learned forward/matrix comparison is omitted.");
                return Ok(());
            }
        };
        let (in_dim, out_dim, m, blk) = (256usize, 256usize, 64usize, 32usize); // m>8 ⇒ MMQ path

        // A dense weight whose per-32-block channel-0 is ZEROED (the outlier channel carries no
        // signal). Built on the CPU (the `quantize_onto` source requirement) then moved to `dev`.
        let mut w = Tensor::randn(0f32, 0.1f32, (out_dim, in_dim), &Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for o in 0..out_dim {
            for b in 0..(in_dim / blk) {
                w[o * in_dim + b * blk] = 0.0;
            }
        }
        let w_cpu = Tensor::from_vec(w, (out_dim, in_dim), &Device::Cpu)?;
        let w = w_cpu.to_device(&dev)?;

        // Activation: ~N(0,1) with a +30000 outlier on each block's channel-0.
        let mut x = Tensor::randn(0f32, 1f32, (m, in_dim), &dev)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for r in 0..m {
            for b in 0..(in_dim / blk) {
                x[r * in_dim + b * blk] = 30000.0;
            }
        }
        let x = Tensor::from_vec(x, (m, in_dim), &dev)?;
        let reference = x.matmul(&w.t()?)?; // f32 dense ground truth on `dev`

        // Dequant-on-forward path: a Q4 `QLinear` built via the load-time `quantize` — the shortest
        // way to a real QTensor, and the byte-identical `QLinear::forward` compute path a packed
        // `from_packed` load also feeds (both dequant the weight to a dense matmul).
        let qt = QTensor::quantize_onto(&w_cpu, GgmlDType::Q4_0, &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(w, None)));
        lin.quantize(Quant::Q4)?;
        let dequant = lin.forward(&x)?;

        // Raw int8 `QMatMul` over the same Q4 weight — the path the black-render bug took.
        let int8 = QMatMul::from_qtensor(qt)?.forward(&x)?;

        let dq = tensor_cosine(&dequant, &reference) as f32;
        let q8 = tensor_cosine(&int8, &reference) as f32;
        eprintln!("sc-7702 packed outlier cosine: dequant(QLinear)={dq:.4}  int8(QMatMul)={q8:.4}");
        assert!(
            dq > 0.99,
            "packed QLinear Q4 forward must track f32 under outlier activations (cosine {dq:.4}) — \
             the int8 activation-quant path must not return inside QLinear::forward (sc-7702)"
        );
        assert!(
            dq > q8 + 0.05,
            "vacuous test: the raw int8 path ({q8:.4}) did not degrade vs dequant ({dq:.4})"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::q4_packed_forward_survives_outlier_activations");
        Ok(())
    }

    // ---- F-025 / sc-9005 unified-seam drifts --------------------------------------------------

    /// A dense [`DenseLinear::Linear`] forward equals a plain `candle_nn::Linear` exactly (the Lens /
    /// FLUX.2 dense layout is a pass-through).
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn dense_linear_layout_equals_plain_linear() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 64);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev)?;
        let plain = CandleNnLinear::new(w.clone(), Some(b.clone()));
        let ql = QLinear::Dense(DenseLinear::Linear(plain.clone()));

        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let dev_max = (ql
            .forward(&x)?
            .sub(&__model_proof_output!(plain.forward(&x))?)?)
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "Dense(Linear) must equal plain Linear");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::dense_linear_layout_equals_plain_linear");
        Ok(())
    }

    /// [`QLinear::forward_upcast`] is **byte-identical** to [`QLinear::forward`] when the weight
    /// already matches the activation dtype — the inertness guarantee every existing call site relies
    /// on (FLUX / Chroma / SDXL / Lens / SAM3 …). Covers both dense layouts, biased and bias-less.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn forward_upcast_is_inert_at_matching_dtype() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 48);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev)?;
        let x = Tensor::randn(0f32, 1f32, (2, 5, in_dim), &dev)?;

        let cases = vec![
            (
                "Linear+bias",
                QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(
                    w.clone(),
                    Some(b.clone()),
                ))),
            ),
            (
                "Linear no-bias",
                QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(w.clone(), None))),
            ),
            (
                "Transposed+bias",
                QLinear::Dense(DenseLinear::Transposed {
                    weight_t: w.t()?.contiguous()?,
                    bias: Some(b.clone()),
                }),
            ),
        ];
        for (name, ql) in cases {
            let dev_max = (ql.forward_upcast(&x)?.sub(&ql.forward(&x)?)?)
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(
                dev_max, 0.0,
                "{name}: forward_upcast must be inert at matching dtype"
            );
        }
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::forward_upcast_is_inert_at_matching_dtype");
        Ok(())
    }

    /// [`QLinear::forward_upcast`] runs f32 activations against **bf16-resident** weights (the Mochi
    /// T5 regime) — where plain `forward` would fail on the dtype mismatch — and agrees with the
    /// all-f32 result to bf16's own rounding.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn forward_upcast_runs_f32_acts_over_bf16_weights() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 48);
        let w32 = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let x = Tensor::randn(0f32, 1f32, (2, 5, in_dim), &dev)?;

        let bf16 = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(
            w32.to_dtype(DType::BF16)?,
            None,
        )));
        // Plain forward can't mix f32 activations with bf16 weights; upcast can.
        assert!(bf16.forward(&x).is_err(), "sanity: mismatch must fail");
        let got = bf16.forward_upcast(&x)?;
        assert_eq!(
            got.dtype(),
            DType::F32,
            "output follows the activation dtype"
        );

        // Reference: the same bf16-rounded weight, materialized at f32.
        let ref_lin = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(
            w32.to_dtype(DType::BF16)?.to_dtype(DType::F32)?,
            None,
        )));
        let dev_max = (got.sub(&ref_lin.forward(&x)?)?)
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "bf16-resident + f32-compute must equal the bf16-rounded weight materialized at f32"
        );
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::forward_upcast_runs_f32_acts_over_bf16_weights");
        Ok(())
    }

    /// A pre-transposed [`DenseLinear::Transposed`] forward (SAM3 / SeedVR2 layout) equals the
    /// `[out,in]` `candle_nn::Linear` result over a rank-3 activation — the flatten-GEMM path is
    /// numerically the same `x·Wᵀ + b`.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn dense_transposed_layout_matches_linear_on_nd() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 48);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?; // [out, in]
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev)?;
        let transposed = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: w.t()?.contiguous()?,
            bias: Some(b.clone()),
        });
        let plain = CandleNnLinear::new(w, Some(b));

        // rank-3 activation (the SeedVR2/SAM3 case that goes through the leading-dim flatten).
        let x = Tensor::randn(0f32, 1f32, (2usize, 5usize, in_dim), &dev)?;
        let got = transposed.forward(&x)?;
        let want = __model_proof_output!(plain.forward(&x))?;
        assert_eq!(got.dims(), want.dims());
        let cos = tensor_cosine(&got, &want) as f32;
        assert!(cos > 0.99999, "transposed vs linear cosine {cos:.6}");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::dense_transposed_layout_matches_linear_on_nd");
        Ok(())
    }

    /// The int8-fast fold builds a [`QuantWeight::Matmul`] (candle's `QMatMul` path), quantizes/forwards
    /// near-losslessly at Q8, and reports [`MatmulStrategy::Int8Fast`] — the FLUX.2 / SAM3 / SeedVR2
    /// strategy, kept distinct from the dequant-dense (Lens) path.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn int8_fast_fold_quantizes_and_reports_strategy() -> Result<()> {
        let dev = Device::Cpu;
        let (in_dim, out_dim) = (32, 64);
        let w = Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?;
        let b = Tensor::randn(0f32, 1f32, out_dim, &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(w, Some(b))));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let dense = lin.forward(&x)?;

        lin.quantize_int8_fast(Quant::Q8, false, false, true)?;
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::Int8Fast));
        assert!(matches!(
            lin,
            QLinear::Quantized {
                weight: QuantWeight::Matmul(_),
                ..
            }
        ));
        let q = lin.forward(&x)?;
        assert_eq!(q.dims(), dense.dims());
        let cos = tensor_cosine(&dense, &q) as f32;
        assert!(cos > 0.999, "int8-fast Q8 cosine {cos:.5}");
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::int8_fast_fold_quantizes_and_reports_strategy");
        Ok(())
    }

    /// `quantize` (the Lens fold) yields the [`MatmulStrategy::DequantDense`] (sc-7702-safe) path.
    #[test]
    fn dequant_dense_fold_reports_strategy() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(w, None)));
        lin.quantize(Quant::Q4)?;
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::DequantDense));
        Ok(())
    }

    /// The SAM3/SeedVR2 skip predicate: an `in_features` not a multiple of 32 stays dense; a divisible
    /// one folds. Idempotent (a second fold is a no-op).
    #[test]
    fn int8_fast_skip_predicate_and_idempotent() -> Result<()> {
        let dev = Device::Cpu;
        // in=20, not a multiple of 32 → stays dense under the skip predicate.
        let odd = Tensor::randn(0f32, 1f32, (64, 20), &dev)?;
        let mut lin = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: odd.t()?.contiguous()?,
            bias: None,
        });
        lin.quantize_int8_fast(Quant::Q8, true, true, false)?;
        assert!(matches!(lin, QLinear::Dense(_)), "in=20 must stay dense");

        // in=32 → folds; a second call is a no-op.
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: w.t()?.contiguous()?,
            bias: None,
        });
        lin.quantize_int8_fast(Quant::Q8, true, true, false)?;
        assert!(lin.is_quantized(), "in=32 must fold");
        lin.quantize_int8_fast(Quant::Q8, true, true, false)?; // idempotent
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::Int8Fast));
        Ok(())
    }

    /// `quantize_onto` lands the folded weight on an explicit device (here the CPU) with the int8-fast
    /// strategy and an f32-promoted bias — FLUX.2's CPU-stage → quantize-onto path.
    #[test]
    fn quantize_onto_explicit_device_int8_fast() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let b = Tensor::randn(0f32, 1f32, 64, &dev)?;
        let mut lin = QLinear::Dense(DenseLinear::Linear(CandleNnLinear::new(w, Some(b))));
        lin.quantize_onto(Quant::Q8, &dev)?;
        assert_eq!(lin.matmul_strategy(), Some(MatmulStrategy::Int8Fast));
        match &lin {
            QLinear::Quantized { bias: Some(b), .. } => {
                assert_eq!(b.dtype(), DType::F32, "bias promoted to f32")
            }
            _ => panic!("expected quantized with bias"),
        }
        // Idempotent.
        lin.quantize_onto(Quant::Q8, &dev)?;
        assert!(lin.is_quantized());
        Ok(())
    }

    /// `to_device` moves a dense `Transposed` projection and is a no-op on a quantized one.
    #[test]
    fn to_device_moves_dense_only() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::randn(0f32, 1f32, (64, 32), &dev)?;
        let mut lin = QLinear::from_dense(DenseLinear::Transposed {
            weight_t: w.t()?.contiguous()?,
            bias: None,
        });
        lin.to_device(&dev)?; // dense: rebuilds in place
        assert!(matches!(lin, QLinear::Dense(_)));
        lin.quantize(Quant::Q8)?;
        lin.to_device(&dev)?; // quantized: no-op, must not error
        assert!(lin.is_quantized());
        Ok(())
    }
}
