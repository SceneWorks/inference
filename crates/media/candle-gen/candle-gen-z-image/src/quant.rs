//! Regression tests for the shared quantized projection and VAE loaders.

#[cfg(test)]
mod tests {
    use candle_gen::candle_core::safetensors::MmapedSafetensors;
    use candle_gen::candle_core::{DType, Device, Result, Tensor};
    use candle_gen::candle_nn::{Linear, VarBuilder};
    use candle_gen::quant as shared;
    use candle_gen::quant::{DenseLinear, MatmulStrategy, QLinear};
    use candle_gen::testkit::tensor_cosine;
    use std::collections::HashMap;

    /// Test-side MLX Q4 packer: per-element 4-bit codes → MLX u32 words (LSB-first nibbles), group 64.
    /// Returns `(wq [out, in/8] u32, scales [out, in/64], biases [out, in/64], affine grid [out, in])` —
    /// the exact packed-parts fixture the detect loaders consume, plus the affine grid they reproduce.
    fn q4_packed(out_dim: usize, in_dim: usize) -> (Tensor, Tensor, Tensor, Vec<f32>) {
        let dev = Device::Cpu;
        const G: usize = 64;
        let codes: Vec<u8> = (0..out_dim * in_dim)
            .map(|i| ((i * 7 + i / 13) % 16) as u8)
            .collect();
        let groups = out_dim * in_dim / G;
        let scales: Vec<f32> = (0..groups).map(|g| 0.0625 * (g as f32 + 1.0)).collect();
        let biases: Vec<f32> = (0..groups).map(|g| -0.5 - 0.25 * g as f32).collect();
        let gpr = in_dim / G;
        let grid: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| {
                let (row, col) = (i / in_dim, i % in_dim);
                let g = row * gpr + col / G;
                scales[g] * codes[i] as f32 + biases[g]
            })
            .collect();
        let words: Vec<u32> = codes
            .chunks_exact(8)
            .map(|c| {
                c.iter()
                    .enumerate()
                    .fold(0u32, |acc, (i, &q)| acc | ((q as u32 & 0xF) << (4 * i)))
            })
            .collect();
        let wq = Tensor::from_vec(words, (out_dim, in_dim / 8), &dev).unwrap();
        let s = Tensor::from_vec(scales, (out_dim, gpr), &dev).unwrap();
        let b = Tensor::from_vec(biases, (out_dim, gpr), &dev).unwrap();
        (wq, s, b, grid)
    }

    /// **Packed-detect fires on the z-image key layout (incl. the `attention.to_out.0` nesting).**
    /// Writes a safetensors mimicking the real z-image DiT packed layout — a `to_out.0` triple (the key
    /// remap that would silently fall back to dense if the loader `.pp()`'d past the `.scales` sibling)
    /// and a dense `to_q` sibling — and loads both through `linear_detect`. The `.scales`/`.biases`
    /// siblings must survive the `to_out.0` base string, `to_out.0` must load quantized via the
    /// sc-7702-safe **dequant-dense** arm, the dense sibling stays `Dense`, and the packed forward must
    /// reproduce the affine grid bit-exactly.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (64usize, 128usize);
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        // The `attention.to_out.0` packed triple (the nested key the z-image DiT threads as one base).
        map.insert("attention.to_out.0.weight".into(), wq);
        map.insert("attention.to_out.0.scales".into(), s);
        map.insert("attention.to_out.0.biases".into(), b);
        // A dense sibling (`to_q`) with no `.scales` → the dense path must stay unchanged.
        map.insert(
            "attention.to_q.weight".into(),
            Tensor::randn(0f32, 1f32, (out_dim, in_dim), &dev)?,
        );

        let tmp = tmp.path().join("sc9408_detect.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: we just wrote this file and nothing else touches it during the test.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());
        let attn = vb.pp("attention");

        // `to_out.0` — packed-detected through the remapped base (never `.pp("0")` past the sibling).
        // The packed load must take the sc-7702-safe dequant-dense arm, NOT the int8-fast fold arm.
        let packed = QLinear::linear_detect(in_dim, out_dim, &attn, "to_out.0", false)?;
        assert_eq!(
            packed.matmul_strategy(),
            Some(MatmulStrategy::DequantDense),
            "`.scales` under to_out.0 ⇒ packed dequant-dense load"
        );

        // `to_q` — dense (no `.scales`), path unchanged.
        let dense = QLinear::linear_detect(in_dim, out_dim, &attn, "to_q", false)?;
        assert!(!dense.is_quantized(), "no `.scales` ⇒ dense path unchanged");
        assert!(matches!(dense, QLinear::Dense(_)));

        // The packed forward reproduces the affine grid (bit-exact repack + dequant-on-forward).
        let grid_lin = QLinear::from_dense(DenseLinear::Linear(Linear::new(
            Tensor::from_vec(grid, (out_dim, in_dim), &dev)?,
            None,
        )));
        let x = Tensor::randn(0f32, 1f32, (4, in_dim), &dev)?;
        let cos = tensor_cosine(&packed.forward(&x)?, &grid_lin.forward(&x)?) as f32;
        assert!(cos > 0.99999, "packed vs affine-grid cosine {cos:.6}");

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::linear_detect_fires_on_to_out_remap_and_leaves_dense_unchanged");
        Ok(())
    }

    /// The packed-detecting embedding loader fires on `{base}.scales` and reproduces the affine grid
    /// rows (the Qwen3 TE `model.embed_tokens` is packed in the q4/q8 tiers), while a `.scales`-less
    /// table stays dense.
    #[test]
    #[ignore = "model inference; explicit model workflow only"]
    fn embedding_detect_fires_and_matches_grid() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (vocab, hidden) = (32usize, 128usize);
        let (wq, s, b, grid) = q4_packed(vocab, hidden);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("embed_tokens.weight".into(), wq);
        map.insert("embed_tokens.scales".into(), s);
        map.insert("embed_tokens.biases".into(), b);
        map.insert(
            "dense.weight".into(),
            Tensor::from_vec(grid.clone(), (vocab, hidden), &dev)?,
        );
        let tmp = tmp.path().join("sc9408_emb.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let emb = shared::embedding(&vb, "embed_tokens", vocab, hidden)?;
        assert!(
            emb.resident_storage().is_some(),
            "`.scales` ⇒ packed embedding"
        );
        let dense = shared::embedding(&vb, "dense", vocab, hidden)?;
        assert!(
            dense.resident_storage().is_none(),
            "no `.scales` ⇒ dense embedding"
        );

        let idx = Tensor::from_vec(vec![0u32, 5, 31, 12, 5], (5,), &dev)?;
        let p = emb.forward(&idx)?;
        let d = dense.forward(&idx)?;
        let dev_max = (p.sub(&d)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(
            dev_max, 0.0,
            "packed embedding deviates from the dense grid"
        );

        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED quant::tests::embedding_detect_fires_and_matches_grid");
        Ok(())
    }

    /// The VAE dequant-to-dense helper reproduces the affine grid a packed VAE attention weight
    /// represents (the path the stock VAE consumes for the 8 mid-block attention projections).
    #[test]
    fn dequant_packed_to_dense_matches_grid() -> Result<()> {
        let tmp = tempfile::tempdir().unwrap();
        let dev = Device::Cpu;
        let (out_dim, in_dim) = (512usize, 64usize); // the real VAE mid-block attn shape
        let (wq, s, b, grid) = q4_packed(out_dim, in_dim);

        let mut map: HashMap<String, Tensor> = HashMap::new();
        map.insert("to_q.weight".into(), wq);
        map.insert("to_q.scales".into(), s);
        map.insert("to_q.biases".into(), b);
        let tmp = tmp.path().join("sc9408_vae.safetensors");
        candle_gen::candle_core::safetensors::save(&map, &tmp)?;
        // SAFETY: freshly written, single-reader.
        let st = unsafe { MmapedSafetensors::new(&tmp)? };
        let vb = VarBuilder::from_backend(Box::new(st), DType::F32, dev.clone());

        let dense = shared::dequant_packed_to_dense(&vb, "to_q", &dev, DType::F32)?;
        assert_eq!(dense.dims(), &[out_dim, in_dim]);
        let grid_t = Tensor::from_vec(grid, (out_dim, in_dim), &dev)?;
        let dev_max = (dense.sub(&grid_t)?).abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(dev_max, 0.0, "vae dequant deviates from the affine grid");

        Ok(())
    }
}
