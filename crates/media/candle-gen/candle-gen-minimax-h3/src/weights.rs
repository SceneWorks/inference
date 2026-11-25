//! H3's shared component reader: packed tiers reuse the common runtime-format sidecars.
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::{Result, Weights};

pub(crate) fn load_component_weights(
    dir: &Path,
    shards: &[PathBuf],
    device: &Device,
    dtype: Option<DType>,
    prefixes: &[&str],
) -> Result<Weights> {
    if let Some(packed) = crate::tier::packed_config(dir)? {
        // Native FP8/INT8/W4A8 imports retain their validated readers. Only MLX affine triples
        // need this conversion; dense tensors in mixed tiers retain their existing precision.
        let headers = candle_gen::loader::mmap_safetensors_multi(shards)?;
        let has_affine = headers.tensors().iter().any(|(name, _)| {
            name.ends_with(".scales") && prefixes.iter().any(|prefix| name.starts_with(prefix))
        });
        // Published tiers already store their affine scales at the encoder's store precision.
        // A legacy custom tier may rely on a lossy load-time cast. Keep that behavior rather
        // than silently changing its quantization grid when preparing source-format sidecars.
        let scale_precision_matches = headers.tensors().iter().all(|(name, view)| {
            if !(name.ends_with(".scales") || name.ends_with(".biases"))
                || !prefixes.iter().any(|prefix| name.starts_with(prefix))
            {
                return true;
            }
            let stored = format!("{:?}", view.dtype());
            dtype.is_none_or(|dtype| {
                stored == format!("{dtype:?}")
                    || (dtype == DType::F32 && matches!(stored.as_str(), "F16" | "BF16"))
            })
        });
        drop(headers);
        if has_affine && scale_precision_matches {
            let cancel = crate::block_stream::current()
                .map(|policy| policy.cancel.clone())
                .unwrap_or_default();
            let weights = Weights::from_files_cached_packed(
                shards, dir, packed, device, dtype, prefixes, &cancel,
            )
            .map_err(|error| {
                if cancel.is_cancelled() {
                    candle_gen::CandleError::Canceled
                } else {
                    error
                }
            })?;
            if let Some(cache) = weights.packed_sidecars() {
                tracing::info!(event = "h3_runtime_weight_cache_prepared", component = %dir.display(),
                    created = cache.created_count(), reused = cache.reused_count(),
                    source_bytes_hashed = cache.source_bytes_hashed(),
                    payload_bytes_hashed = cache.payload_bytes_hashed(),
                    "Prepared direct runtime-format weight loading");
            }
            return Ok(weights);
        }
    }
    match dtype {
        Some(dtype) => Weights::from_files_filtered_host_packed(shards, device, dtype, prefixes),
        None => Weights::from_files_native_int8(shards, device),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type TestResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    use candle_gen::candle_core::{safetensors, Tensor};
    use candle_gen::quant::{pack_mlx_affine, QEmbedding, QLinear, QuantWeight};
    use std::collections::HashMap;

    const TE: &str = "model.language_model.layers.0.self_attn.q_proj";
    const DIT: &str = "transformer_blocks.0.attn.to_q";
    const EMBED: &str = "model.language_model.embed_tokens";

    fn fixture(dir: &Path, bits: usize, delta: f32) -> TestResult<PathBuf> {
        let mut tensors = HashMap::new();
        for (index, base) in [
            TE,
            DIT,
            EMBED,
            "model.language_model.layers.50.self_attn.q_proj",
        ]
        .iter()
        .enumerate()
        {
            let dense = Tensor::from_vec(
                (0..128)
                    .map(|i| ((i * 17 + index) % 41) as f32 / 16.0 + delta)
                    .collect::<Vec<_>>(),
                (2, 64),
                &Device::Cpu,
            )?;
            let (weight, scales, biases) = pack_mlx_affine(&dense, bits, 64)?;
            tensors.insert(format!("{base}.weight"), weight);
            tensors.insert(format!("{base}.scales"), scales);
            tensors.insert(format!("{base}.biases"), biases);
        }
        tensors.insert(
            "head.weight".into(),
            Tensor::new(&[1.0001f32], &Device::Cpu)?,
        );
        tensors.insert(
            "head.bias".into(),
            Tensor::new(&[0.125f32], &Device::Cpu)?.to_dtype(DType::F16)?,
        );
        let file = dir.join("model.safetensors");
        safetensors::save(&tensors, &file)?;
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({"quantization":{"bits":bits,"group_size":64}}).to_string(),
        )?;
        Ok(file)
    }

    fn bytes(linear: &candle_gen::quant::AdaptLinear) -> Vec<u8> {
        match linear.base_qlinear().unwrap() {
            QLinear::Quantized {
                weight: QuantWeight::Dequant(weight),
                ..
            } => weight.data().unwrap().into_owned(),
            _ => panic!("expected the existing dequant-dense packed execution strategy"),
        }
    }

    fn assert_reuse(bits: usize) -> TestResult<()> {
        let dir = tempfile::tempdir()?;
        let file = fixture(dir.path(), bits, 0.0)?;
        let selected = [
            "model.language_model.layers.0.",
            "model.language_model.embed_tokens.",
            "transformer_blocks.",
            "head.",
        ];
        let first = load_component_weights(
            dir.path(),
            std::slice::from_ref(&file),
            &Device::Cpu,
            None,
            &selected,
        )?;
        let cache = first.packed_sidecars().unwrap();
        assert_eq!(cache.created_count(), 3);
        assert!(cache.source_bytes_hashed() > 0);
        assert!(!first.contains("model.language_model.layers.50.self_attn.q_proj.weight"));
        assert_eq!(first.require("head.weight")?.dtype(), DType::F32);
        assert_eq!(first.require("head.bias")?.dtype(), DType::F16);
        let raw = Weights::from_files_native_int8(std::slice::from_ref(&file), &Device::Cpu)?;
        for (base, component) in [(TE, crate::quant::TEXT_ENCODER), (DIT, crate::quant::DIT)] {
            let expected = crate::quant::lin(&raw, component, base, false, DType::F32)?;
            let actual = crate::quant::lin(&first, component, base, false, DType::F32)?;
            assert_eq!(bytes(&actual.linear), bytes(&expected.linear));
            assert_eq!(actual.base_bytes, expected.base_bytes);
        }
        let expected_embed = crate::quant::embed(&raw, EMBED, DType::F32)?;
        let actual_embed = crate::quant::embed(&first, EMBED, DType::F32)?;
        match (&expected_embed.embedding, &actual_embed.embedding) {
            (
                QEmbedding::Quantized { row_data: a, .. },
                QEmbedding::Quantized { row_data: b, .. },
            ) => assert_eq!(a, b),
            _ => panic!("embedding must retain its compact host rows"),
        }
        drop(raw);
        drop(first);
        let warm = load_component_weights(dir.path(), &[file], &Device::Cpu, None, &selected)?;
        let cache = warm.packed_sidecars().unwrap();
        assert_eq!(cache.created_count(), 0);
        assert_eq!(cache.reused_count(), 3);
        assert_eq!(cache.source_bytes_hashed(), 0);
        assert_eq!(cache.payload_bytes_hashed(), 0);
        for (base, component) in [(TE, crate::quant::TEXT_ENCODER), (DIT, crate::quant::DIT)] {
            assert!(
                crate::quant::lin(&warm, component, base, false, DType::F32)?
                    .linear
                    .is_quantized()
            );
        }
        // Header-only profiling must not materialize packed operands as a side effect.
        assert!(warm.stored_tensor_bytes(&format!("{TE}.weight"))? > 0);
        assert_eq!(cache.source_bytes_hashed(), 0);
        Ok(())
    }

    #[test]
    fn runtime_cache_q8_reuses_text_transformer_and_embedding_without_rehashing() -> TestResult<()>
    {
        assert_reuse(8)
    }

    #[test]
    fn runtime_cache_q4_preserves_existing_packed_tier() -> TestResult<()> {
        assert_reuse(4)
    }

    #[test]
    fn runtime_cache_source_changes_and_corruption_rebuild_before_loading() -> TestResult<()> {
        let dir = tempfile::tempdir()?;
        let file = fixture(dir.path(), 8, 0.0)?;
        let first = load_component_weights(
            dir.path(),
            std::slice::from_ref(&file),
            &Device::Cpu,
            None,
            &[""],
        )?;
        let old_path = first
            .packed_sidecars()
            .unwrap()
            .path_for(DIT)
            .unwrap()
            .to_owned();
        drop(first);
        fixture(dir.path(), 8, 0.25)?;
        let changed = load_component_weights(
            dir.path(),
            std::slice::from_ref(&file),
            &Device::Cpu,
            None,
            &[""],
        )?;
        let cache = changed.packed_sidecars().unwrap();
        assert!(cache.source_bytes_hashed() > 0);
        let path = cache.path_for(DIT).unwrap().to_owned();
        assert_ne!(path, old_path);
        let expected =
            bytes(&crate::quant::lin(&changed, crate::quant::DIT, DIT, false, DType::F32)?.linear);
        drop(changed);
        std::fs::write(&path, b"corrupt")?;
        let repaired = load_component_weights(dir.path(), &[file], &Device::Cpu, None, &[""])?;
        assert_eq!(repaired.packed_sidecars().unwrap().created_count(), 1);
        assert_eq!(
            bytes(&crate::quant::lin(&repaired, crate::quant::DIT, DIT, false, DType::F32)?.linear),
            expected
        );
        Ok(())
    }

    #[test]
    #[ignore = "complete synthetic DiT construction; explicit H3 model-proof workflow"]
    fn runtime_cache_transformer_constructor_reuses_both_video_partitions() -> TestResult<()> {
        use crate::dit::MiniMaxH3Dit;
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/dit_block.safetensors");
        let mut tensors = safetensors::load(fixture, &Device::Cpu)?;
        let (weight, scales, biases) = pack_mlx_affine(&tensors[&format!("{DIT}.weight")], 8, 64)?;
        tensors.insert(format!("{DIT}.weight"), weight);
        tensors.insert(format!("{DIT}.scales"), scales);
        tensors.insert(format!("{DIT}.biases"), biases);
        let config = serde_json::json!({
            "num_attention_heads":4, "attention_head_dim":24, "hidden_size":64,
            "num_layers":2, "num_refiner_layers":2, "ffn_dim":32, "in_channels":24,
            "audio_in_channels":32, "patch_size":[1,2,2], "text_dim":40, "freq_dim":16,
            "time_embed_hidden_dim":64, "time_embed_dim":48, "rope_freq_dim":2,
            "rope_theta":10000.0, "norm_eps":1e-5, "qk_norm_eps":1e-5,
            "final_norm_eps":1e-5, "quantization":{"bits":8,"group_size":64}
        });
        let dir = tempfile::tempdir()?;
        for partition in ["transformer", "transformer_ref"] {
            let root = dir.path().join(partition);
            std::fs::create_dir(&root)?;
            let file = root.join("model.safetensors");
            safetensors::save(&tensors, &file)?;
            std::fs::write(root.join("config.json"), config.to_string())?;
            let first = MiniMaxH3Dit::load_from_dir(&root, &Device::Cpu, DType::F32)?;
            let second = MiniMaxH3Dit::load_from_dir(&root, &Device::Cpu, DType::F32)?;
            assert_eq!(first.config().num_layers, 2);
            assert_eq!(second.adaln_nbytes(), first.adaln_nbytes());
            let weights = load_component_weights(&root, &[file], &Device::Cpu, None, &[""])?;
            let cache = weights.packed_sidecars().unwrap();
            assert_eq!(cache.created_count(), 0);
            assert_eq!(cache.reused_count(), 1);
            assert_eq!(cache.source_bytes_hashed(), 0);
            assert_eq!(cache.payload_bytes_hashed(), 0);
        }
        Ok(())
    }

    #[test]
    fn runtime_cache_preserves_legacy_encoder_scale_rounding() -> TestResult<()> {
        let dir = tempfile::tempdir()?;
        let file = fixture(dir.path(), 8, 0.0)?;
        let weights = load_component_weights(
            dir.path(),
            std::slice::from_ref(&file),
            &Device::Cpu,
            Some(DType::BF16),
            &[""],
        )?;
        // Fixture scales are F32; the old BF16 encoder deliberately rounds them on load.
        assert!(weights.packed_sidecars().is_none());
        let old =
            Weights::from_files_filtered_host_packed(&[file], &Device::Cpu, DType::BF16, &[""])?;
        assert_eq!(
            bytes(
                &crate::quant::lin(&weights, crate::quant::TEXT_ENCODER, TE, false, DType::BF16)?
                    .linear
            ),
            bytes(
                &crate::quant::lin(&old, crate::quant::TEXT_ENCODER, TE, false, DType::BF16)?
                    .linear
            )
        );
        Ok(())
    }

    #[test]
    fn runtime_cache_dense_components_keep_their_stored_precision() -> TestResult<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("model.safetensors");
        let value = Tensor::new(&[1.0001f32], &Device::Cpu)?;
        safetensors::save(&HashMap::from([("head.weight".to_owned(), value)]), &file)?;
        std::fs::write(dir.path().join("config.json"), "{}")?;
        let weights = load_component_weights(dir.path(), &[file], &Device::Cpu, None, &[""])?;
        assert!(weights.packed_sidecars().is_none());
        assert_eq!(
            weights.require("head.weight")?.to_vec1::<f32>()?,
            vec![1.0001]
        );
        Ok(())
    }
}
