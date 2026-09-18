//! Strict loader for published Prism/Bonsai MLX affine-2 snapshots.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use candle_core::{DType, Device};
use core_llm::PrismHadamardMetadata;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::primitives::{PrismPackedWeight, PrismRegistry, Weights};

pub(crate) struct PrismMlxCheckpoint {
    pub weights: Weights,
    pub registry: PrismRegistry,
}

impl PrismMlxCheckpoint {
    pub fn open(dir: &Path, device: &Device, config: &Value) -> Result<Self> {
        validate_config(config)?;
        let metadata = read_metadata(dir)?;
        let modules = config
            .get("modules")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Config("Prism config missing modules array".into()))?;
        let loaded = Weights::from_dir(dir, device)?;
        let mut tensors = loaded.into_map();
        let mut packed = HashMap::with_capacity(modules.len());
        let mut declared = BTreeSet::new();
        for module in modules {
            let path = module
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Config("Prism module missing path".into()))?;
            if !declared.insert(path.to_string()) {
                return Err(Error::Config(format!("duplicate Prism module {path}")));
            }
            if module.get("block").and_then(Value::as_u64) != Some(metadata.block_size as u64)
                || module.get("dtype").and_then(Value::as_str) != Some("float16")
            {
                return Err(Error::Config(format!(
                    "Prism module {path} does not match frozen block/dtype contract"
                )));
            }
            let base = format!("language_model.{path}");
            let weight_key = format!("{base}.weight");
            let scales_key = format!("{base}.scales");
            let biases_key = format!("{base}.biases");
            let words = tensors
                .remove(&weight_key)
                .ok_or_else(|| Error::MissingTensor(weight_key.clone()))?;
            let scales = tensors
                .remove(&scales_key)
                .ok_or_else(|| Error::MissingTensor(scales_key.clone()))?;
            let biases = tensors
                .remove(&biases_key)
                .ok_or_else(|| Error::MissingTensor(biases_key.clone()))?;
            let weight = PrismPackedWeight::from_mlx_affine2(
                weight_key.clone(),
                words,
                scales,
                &biases,
                &metadata,
                None,
                None,
            )?;
            packed.insert(weight_key, Arc::new(weight));
        }
        if packed.len() != metadata.forward_weight_names.len() + metadata.inverse_weight_names.len()
        {
            return Err(Error::Config(format!(
                "Prism module count {} != Hadamard tensor count {}",
                packed.len(),
                metadata.forward_weight_names.len() + metadata.inverse_weight_names.len()
            )));
        }
        if let Some(name) = tensors.keys().find(|name| {
            name.ends_with(".scales")
                || name.ends_with(".biases")
                || tensors
                    .get(*name)
                    .is_some_and(|tensor| tensor.dtype() == DType::U32)
        }) {
            return Err(Error::Config(format!(
                "undeclared Prism packed tensor {name}; refusing a partial load"
            )));
        }
        Ok(Self {
            weights: Weights::from_map(tensors, device.clone()),
            registry: PrismRegistry::new(packed)?,
        })
    }
}

fn validate_config(config: &Value) -> Result<()> {
    let q = config
        .get("quantization")
        .ok_or_else(|| Error::Config("Prism config missing quantization contract".into()))?;
    if config.get("schema_version").and_then(Value::as_u64) != Some(2)
        || config.get("model_type").and_then(Value::as_str) != Some("prism_hadamard_qwen35")
        || q.get("bits").and_then(Value::as_u64) != Some(2)
        || q.get("group_size").and_then(Value::as_u64) != Some(128)
        || q.get("mode").and_then(Value::as_str) != Some("affine")
    {
        return Err(Error::Config(
            "unsupported Prism config (requires schema 2 qwen35 affine-2 group-128)".into(),
        ));
    }
    Ok(())
}

fn read_metadata(dir: &Path) -> Result<PrismHadamardMetadata> {
    let path = dir.join("hadamard.json");
    let value: Value = serde_json::from_slice(&std::fs::read(&path)?)
        .map_err(|error| Error::Config(format!("parse {}: {error}", path.display())))?;
    let exact = |key: &str, expected: &str| -> Result<()> {
        if value.get(key).and_then(Value::as_str) == Some(expected) {
            Ok(())
        } else {
            Err(Error::Config(format!("invalid Prism metadata {key}")))
        }
    };
    if value.get("prism.hadamard.version").and_then(Value::as_u64) != Some(1) {
        return Err(Error::Config(
            "unsupported Prism Hadamard metadata version".into(),
        ));
    }
    exact(
        "prism.hadamard.transform",
        "normalized-sylvester-walsh-hadamard",
    )?;
    exact("prism.hadamard.axis", "input-last-dimension")?;
    exact("prism.hadamard.sign_mode", "explicit")?;
    let block_size = value
        .get("prism.hadamard.block_size")
        .and_then(Value::as_u64)
        .ok_or_else(|| Error::Config("Prism metadata missing block size".into()))?
        as usize;
    let names = |key: &str| -> Result<BTreeSet<String>> {
        value
            .get(key)
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Config(format!("Prism metadata missing {key}")))?
            .iter()
            .map(|item| {
                item.as_str().map(str::to_string).ok_or_else(|| {
                    Error::Config(format!("Prism metadata {key} contains a non-string"))
                })
            })
            .collect()
    };
    let widths = value
        .get("prism.hadamard.sign_widths")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("Prism metadata missing sign widths".into()))?;
    let values = value
        .get("prism.hadamard.sign_values")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Config("Prism metadata missing sign values".into()))?;
    let mut offset = 0usize;
    let mut signs_by_width = BTreeMap::new();
    for width in widths {
        let width = width
            .as_u64()
            .ok_or_else(|| Error::Config("invalid Prism sign width".into()))?
            as usize;
        let end = offset
            .checked_add(width)
            .ok_or_else(|| Error::Config("Prism sign width overflow".into()))?;
        let slice = values
            .get(offset..end)
            .ok_or_else(|| Error::Config("truncated Prism sign values".into()))?;
        let signs = slice
            .iter()
            .map(|sign| {
                sign.as_i64()
                    .and_then(|v| i8::try_from(v).ok())
                    .ok_or_else(|| Error::Config("invalid Prism sign value".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        if signs_by_width.insert(width, signs).is_some() {
            return Err(Error::Config(format!("duplicate Prism sign width {width}")));
        }
        offset = end;
    }
    if offset != values.len() {
        return Err(Error::Config("extra Prism sign values".into()));
    }
    let metadata = PrismHadamardMetadata {
        block_size,
        signs_by_width,
        forward_weight_names: names("prism.hadamard.weight_names")?,
        inverse_weight_names: names("prism.hadamard.inverse_weight_names")?,
        gdn_v_grouped: value
            .get("prism.hadamard.gdn_v_grouped")
            .and_then(Value::as_bool)
            .ok_or_else(|| Error::Config("Prism metadata missing gdn_v_grouped".into()))?,
    };
    metadata
        .validate()
        .map_err(|error| Error::Config(format!("Prism Hadamard metadata: {error}")))?;
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::Tensor;
    use serde_json::json;

    use super::*;

    fn config() -> Value {
        json!({
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "modules": [{"path": "model.embed_tokens", "block": 128, "embedding": true, "dtype": "float16"}]
        })
    }

    fn write_fixture(dir: &Path, bad_bias: bool) {
        let device = Device::Cpu;
        let base = "language_model.model.embed_tokens";
        let mut tensors = HashMap::new();
        tensors.insert(
            format!("{base}.weight"),
            Tensor::zeros((2, 8), DType::U32, &device).unwrap(),
        );
        tensors.insert(
            format!("{base}.scales"),
            Tensor::ones((2, 1), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{base}.biases"),
            Tensor::full(if bad_bias { 0f32 } else { -1f32 }, (2, 1), &device).unwrap(),
        );
        candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
        std::fs::write(
            dir.join("hadamard.json"),
            serde_json::to_vec(&json!({
                "prism.hadamard.version": 1,
                "prism.hadamard.block_size": 128,
                "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
                "prism.hadamard.axis": "input-last-dimension",
                "prism.hadamard.sign_mode": "explicit",
                "prism.hadamard.weight_names": [],
                "prism.hadamard.inverse_weight_names": [format!("{base}.weight")],
                "prism.hadamard.sign_widths": [128],
                "prism.hadamard.sign_values": vec![1; 128],
                "prism.hadamard.gdn_v_grouped": true
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn strict_mlx_checkpoint_splits_packed_from_auxiliary_weights() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), false);
        let checkpoint = PrismMlxCheckpoint::open(dir.path(), &Device::Cpu, &config()).unwrap();
        assert_eq!(checkpoint.registry.len(), 1);
        assert!(checkpoint
            .registry
            .contains("language_model.model.embed_tokens.weight"));
        assert!(checkpoint.weights.is_empty());
    }

    #[test]
    fn strict_mlx_checkpoint_rejects_malformed_affine_metadata() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(dir.path(), true);
        let error = match PrismMlxCheckpoint::open(dir.path(), &Device::Cpu, &config()) {
            Ok(_) => panic!("malformed affine metadata was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("biases exactly equal"));
    }

    #[test]
    fn compact_checkpoint_executes_qwen_prefill_and_cached_decode() {
        use crate::models::{Qwen35Config, Qwen35Model};

        let dir = tempfile::tempdir().unwrap();
        let device = Device::Cpu;
        let projection_paths = [
            "lm_head",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        ];
        let mut modules = projection_paths
            .iter()
            .map(|path| json!({"path": path, "block": 128, "embedding": false, "dtype": "float16"}))
            .collect::<Vec<_>>();
        modules.push(json!({"path": "model.embed_tokens", "block": 128, "embedding": true, "dtype": "float16"}));
        let config = json!({
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "modules": modules,
            "text_config": {
                "model_type": "qwen3_5_text", "hidden_size": 128, "intermediate_size": 128,
                "num_hidden_layers": 1, "num_attention_heads": 1, "num_key_value_heads": 1,
                "head_dim": 128, "vocab_size": 128, "rms_norm_eps": 1e-6,
                "full_attention_interval": 1, "linear_num_value_heads": 1,
                "linear_num_key_heads": 1, "linear_key_head_dim": 128,
                "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "partial_rotary_factor": 0.25, "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 0, "mtp_use_dedicated_embeddings": false
            }
        });
        let mut tensors = HashMap::new();
        for path in projection_paths {
            let rows = if path.ends_with("q_proj") { 256 } else { 128 };
            let base = format!("language_model.{path}");
            tensors.insert(
                format!("{base}.weight"),
                Tensor::zeros((rows, 8), DType::U32, &device).unwrap(),
            );
            tensors.insert(
                format!("{base}.scales"),
                Tensor::ones((rows, 1), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{base}.biases"),
                Tensor::full(-1f32, (rows, 1), &device).unwrap(),
            );
        }
        let embed = "language_model.model.embed_tokens";
        tensors.insert(
            format!("{embed}.weight"),
            Tensor::zeros((128, 8), DType::U32, &device).unwrap(),
        );
        tensors.insert(
            format!("{embed}.scales"),
            Tensor::ones((128, 1), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{embed}.biases"),
            Tensor::full(-1f32, (128, 1), &device).unwrap(),
        );
        for name in [
            "language_model.model.norm.weight",
            "language_model.model.layers.0.input_layernorm.weight",
            "language_model.model.layers.0.post_attention_layernorm.weight",
            "language_model.model.layers.0.self_attn.q_norm.weight",
            "language_model.model.layers.0.self_attn.k_norm.weight",
        ] {
            tensors.insert(
                name.into(),
                Tensor::zeros((128,), DType::F32, &device).unwrap(),
            );
        }
        candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors")).unwrap();
        let forward = projection_paths
            .iter()
            .map(|path| format!("language_model.{path}.weight"))
            .collect::<Vec<_>>();
        std::fs::write(dir.path().join("hadamard.json"), serde_json::to_vec(&json!({
            "prism.hadamard.version": 1, "prism.hadamard.block_size": 128,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension", "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": forward,
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [128], "prism.hadamard.sign_values": vec![1; 128],
            "prism.hadamard.gdn_v_grouped": true
        })).unwrap()).unwrap();
        let checkpoint = PrismMlxCheckpoint::open(dir.path(), &device, &config).unwrap();
        let qcfg = Qwen35Config::from_json(&config).unwrap();
        let model = Qwen35Model::from_prism_weights(
            &checkpoint.weights,
            "language_model.model",
            qcfg,
            &checkpoint.registry,
            DType::F32,
        )
        .unwrap();
        let mut cache = model.new_cache();
        let prefill = Tensor::from_vec(vec![1i64, 2], (1, 2), &device).unwrap();
        let logits = model.decode_logits(&prefill, &mut cache, 0).unwrap();
        assert_eq!(logits.dims(), &[1, 128]);
        assert!(logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|x| x.is_finite()));
        let next = Tensor::from_vec(vec![3i64], (1, 1), &device).unwrap();
        assert_eq!(
            model.decode_logits(&next, &mut cache, 2).unwrap().dims(),
            &[1, 128]
        );
    }
}
