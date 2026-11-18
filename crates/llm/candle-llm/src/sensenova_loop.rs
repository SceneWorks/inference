//! Validate the published snapshot before placing weights on the device. The
//! logical tensor inventory is identical after unpacking a native Q4/Q8 tier.
use crate::error::{Error, Result};
use crate::models::qwen35::quantizes_tensor;
use crate::primitives::projection::{ProjectionFormat, QuantSpec};
use crate::primitives::quant::ggml_block_storage;
use core_llm::sensenova_loop::{validate_sidecars, MODEL_REVISION};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub(crate) fn shards(dir: &Path) -> Vec<PathBuf> {
    core_llm::sensenova_loop::TARGET_SHARDS
        .iter()
        .map(|(name, _)| dir.join(name))
        .collect()
}

pub(crate) fn format_name(q: QuantSpec) -> &'static str {
    if q.bits() == 4 {
        "Q4_K"
    } else {
        "Q8_0"
    }
}

pub(crate) fn tier(
    config: &Value,
    requested: Option<&ProjectionFormat>,
) -> Result<Option<QuantSpec>> {
    let q = crate::prepare::persisted_ggml_blocks(config);
    if config.get("quantization_config").is_some()
        || config["text_config"].get("quantization").is_some()
    {
        return Err(Error::Config(
            "SenseNova-Loop: unsupported or ambiguous quantization metadata".into(),
        ));
    }
    if let Some(meta) = config.get("quantization") {
        let q = q.ok_or_else(|| {
            Error::Config("SenseNova-Loop: expected native GGML Q4_K or Q8_0 snapshot".into())
        })?;
        if meta["format"].as_str() != Some(format_name(q))
            || meta["source_revision"].as_str() != Some(MODEL_REVISION)
        {
            return Err(Error::Config(format!(
                "SenseNova-Loop: expected {} blocks from revision {MODEL_REVISION}",
                format_name(q)
            )));
        }
    }
    if let Some(requested) = requested {
        if requested
            .ggml()
            .zip(q)
            .is_none_or(|(a, b)| a.bits() != b.bits())
        {
            return Err(Error::Config("SenseNova-Loop: requested quantization differs from snapshot tier; prepare the requested native tier first".into()));
        }
    }
    Ok(q)
}

pub(crate) fn validate_snapshot(dir: &Path, q: Option<QuantSpec>) -> Result<()> {
    validate_sidecars(dir).map_err(Error::Config)?;
    validate_indexed_weights(dir, q)
}

fn validate_indexed_weights(dir: &Path, q: Option<QuantSpec>) -> Result<()> {
    let index = dir.join("model.safetensors.index.json");
    let index: Value = serde_json::from_reader(std::fs::File::open(&index).map_err(|e| {
        Error::Config(format!(
            "SenseNova-Loop: missing shard index {}: {e}",
            index.display()
        ))
    })?)
    .map_err(|e| Error::Config(format!("SenseNova-Loop: invalid shard index: {e}")))?;
    let map = index["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Config("SenseNova-Loop: missing index weight_map".into()))?;
    let mut expected: BTreeMap<_, _> = core_llm::sensenova_loop::target_tensors()
        .map(|(key, shape, dtype)| (key, (shape, dtype)))
        .collect();
    if map.len() != expected.len() {
        return Err(Error::Config(format!(
            "SenseNova-Loop: index lists {} tensors, expected {}",
            map.len(),
            expected.len()
        )));
    }
    for path in shards(dir) {
        let filename = path.file_name().and_then(|s| s.to_str());
        let header = crate::provider::read_safetensors_header(&path).map_err(|e| {
            Error::Config(format!("SenseNova-Loop: reading {}: {e}", path.display()))
        })?;
        for (key, tensor) in header {
            if key == "__metadata__" {
                continue;
            }
            let (dims, original_dtype) = expected.remove(&key).ok_or_else(|| {
                Error::Config(format!(
                    "SenseNova-Loop: unexpected or duplicate tensor {key}"
                ))
            })?;
            if map.get(&key).and_then(Value::as_str) != filename {
                return Err(Error::Config(format!(
                    "SenseNova-Loop: tensor {key} is not indexed to {}",
                    path.display()
                )));
            }
            let dtype = tensor["dtype"].as_str().unwrap_or("");
            let shape: Vec<usize> = serde_json::from_value(tensor["shape"].clone())
                .map_err(|e| Error::Config(format!("SenseNova-Loop: {key} shape: {e}")))?;
            validate_tensor(&key, dtype, &shape, dims, original_dtype, q)?;
        }
    }
    if let Some((key, _)) = expected.first_key_value() {
        return Err(Error::Config(format!(
            "SenseNova-Loop: missing indexed tensor {key}"
        )));
    }
    Ok(())
}

fn validate_tensor(
    key: &str,
    dtype: &str,
    shape: &[usize],
    dims: &[usize],
    original_dtype: &str,
    q: Option<QuantSpec>,
) -> Result<()> {
    if let Some(q) = q.filter(|_| quantizes_tensor(key) || key == "lm_head.weight") {
        if shape.len() != 3
            || ggml_block_storage(dtype == "U8", shape)
                .is_none_or(|(format, rows, cols)| format != q.dtype || [rows, cols] != dims)
        {
            return Err(Error::Config(format!("SenseNova-Loop: {key} must hold native {} blocks for {dims:?}, found {dtype} {shape:?}", format_name(q))));
        }
    } else if shape != dims || dtype != original_dtype {
        return Err(Error::Config(format!(
            "SenseNova-Loop: {key} requires {original_dtype} {dims:?}, found {dtype} {shape:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sensenova_native_tier_requires_exact_format_revision_and_projection_storage() {
        let dense = json!({});
        assert!(tier(&dense, None).unwrap().is_none());
        let q8 = QuantSpec::q8();
        let q4 = QuantSpec::q4();
        for q in [q4, q8] {
            let format = ProjectionFormat::from(q);
            let packed = json!({"quantization":{"bits":q.bits(),"storage":"ggml","format":format_name(q),"source_revision":MODEL_REVISION}});
            assert_eq!(
                tier(&packed, Some(&format)).unwrap().unwrap().bits(),
                q.bits()
            );
            assert!(tier(&dense, Some(&format)).is_err());
            let other = if q.bits() == 4 { q8 } else { q4 };
            assert!(tier(&packed, Some(&other.into())).is_err());
            let mut bad = packed.clone();
            bad["quantization"]["source_revision"] = json!("wrong");
            assert!(tier(&bad, None).is_err());
            let mut eligible = 0;
            for (name, dims, dtype) in core_llm::sensenova_loop::target_tensors() {
                validate_tensor(&name, dtype, dims, dims, dtype, None).unwrap();
                if quantizes_tensor(&name) || name == "lm_head.weight" {
                    eligible += 1;
                    assert!(validate_tensor(&name, dtype, dims, dims, dtype, Some(q)).is_err());
                    let packed_dims =
                        [dims[0], dims[1] / q.dtype.block_size(), q.dtype.type_size()];
                    validate_tensor(&name, "U8", &packed_dims, dims, dtype, Some(q)).unwrap();
                    assert!(
                        validate_tensor(&name, "U8", &packed_dims, dims, dtype, Some(other))
                            .is_err()
                    );
                } else {
                    validate_tensor(&name, dtype, dims, dims, dtype, Some(q)).unwrap();
                    if dtype == "F32" {
                        assert!(validate_tensor(&name, "BF16", dims, dims, dtype, Some(q)).is_err());
                    }
                }
            }
            assert_eq!(eligible, 201);
        }
    }

    #[test]
    fn sensenova_native_incomplete_snapshot_names_the_missing_shard() {
        let dir = tempfile::tempdir().unwrap();
        let names: serde_json::Map<_, _> = core_llm::sensenova_loop::target_tensors()
            .map(|(name, _, _)| (name, json!("model-00001.safetensors")))
            .collect();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            json!({"weight_map":names}).to_string(),
        )
        .unwrap();
        let error = validate_indexed_weights(dir.path(), None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("model-00001.safetensors"), "{error}");
    }

    #[test]
    #[ignore = "requires original SenseNova-Loop-9B shard headers and sidecars; metadata only"]
    fn sensenova_native_frozen_snapshot_metadata() {
        let dir = PathBuf::from(std::env::var_os("SENSENOVA_LOOP_BF16_SNAPSHOT").unwrap());
        let config: Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap()).unwrap();
        let cfg = crate::models::Qwen35Config::from_json(&config).unwrap();
        let plan = cfg.sensenova_loop.unwrap();
        assert_eq!(plan.visits().filter(|&i| cfg.is_linear(i)).count(), 42);
        validate_snapshot(&dir, tier(&config, None).unwrap()).unwrap();
        let spec = core_llm::LoadSpec::dense(dir.to_string_lossy());
        assert!(crate::provider::can_load(&spec));
        assert!(crate::provider::can_load_vision(&spec));
        assert_eq!(crate::provider::eos_token_ids(&dir), [248044, 248046]);
        use core_llm::{ChatTemplate, JinjaChatTemplate, Message, RenderOptions, Tokenizer};
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
        let template = JinjaChatTemplate::with_tokens(
            std::fs::read_to_string(dir.join("chat_template.jinja")).unwrap(),
            "",
            "<|im_end|>",
        );
        for thinking in [None, Some(true), Some(false)] {
            let rendered = template
                .render_with(
                    &[Message::user(
                        "<|vision_start|><|image_pad|><|vision_end|>Describe it.",
                    )],
                    &RenderOptions {
                        add_generation_prompt: true,
                        enable_thinking: thinking,
                        ..Default::default()
                    },
                )
                .unwrap();
            let ending = if thinking == Some(false) {
                "<think>\n\n</think>\n\n"
            } else {
                "<think>\n"
            };
            assert!(rendered.ends_with(ending));
            let ids = tokenizer.encode(&rendered, false).unwrap();
            assert!(ids.windows(3).any(|ids| ids == [248053, 248056, 248054]));
        }
    }
}
