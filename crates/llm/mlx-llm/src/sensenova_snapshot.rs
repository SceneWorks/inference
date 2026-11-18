//! Exact source and native affine tier layouts. MLX loads the target index of the
//! original release; a separately published MTP file is not part of that index.
use std::path::Path;

use core_llm::sensenova_loop::{self, SenseNovaLoop, MODEL_REVISION, TARGET_SHARDS};
use mlx_rs::Dtype;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::primitives::{projection::QuantSpec, Weights};
use crate::snapshot::is_projection;

pub(crate) fn manifest(bits: i32) -> Value {
    json!({"format":"mlx-affine-v1","source":"sensenova/SenseNova-Loop-9B",
        "revision":MODEL_REVISION,"bits":bits,"group_size":64,
        "packing":"uint32","scales":"bfloat16","biases":"bfloat16","routers":"float32"})
}

#[derive(Clone, Copy)]
pub(crate) enum Tier {
    Bf16,
    Affine(QuantSpec),
}

impl Tier {
    fn shards(self, dir: &Path) -> Vec<std::path::PathBuf> {
        match self {
            Self::Bf16 => TARGET_SHARDS
                .iter()
                .map(|(name, _)| dir.join(name))
                .collect(),
            Self::Affine(_) => vec![dir.join("model.safetensors")],
        }
    }
}

pub(crate) fn tier(config: &Value) -> Result<Option<Tier>> {
    if SenseNovaLoop::from_config(config)
        .map_err(Error::Config)?
        .is_none()
    {
        return Ok(None);
    }
    for owner in [config, &config["text_config"]] {
        for field in ["quantization_config", "compression_config", "quant_method"] {
            if owner.get(field).is_some() {
                return Err(Error::Config(format!(
                    "SenseNova-Loop-9B: unsupported {field} metadata"
                )));
            }
        }
    }
    if config["text_config"].get("quantization").is_some() {
        return Err(Error::Config(
            "SenseNova-Loop-9B: unexpected nested quantization".into(),
        ));
    }
    if let Some(block) = config.get("quantization") {
        for spec in [QuantSpec::q4(), QuantSpec::q8()] {
            if *block == json!({"group_size":64,"bits":spec.bits})
                && config.get("sensenova_mlx") == Some(&manifest(spec.bits))
            {
                return Ok(Some(Tier::Affine(spec)));
            }
        }
        return Err(Error::Config("SenseNova-Loop-9B requires native MLX affine Q4/Q8 tier metadata with the exact source revision".into()));
    }
    if config.get("sensenova_mlx").is_some() {
        return Err(Error::Config(
            "SenseNova-Loop-9B: incompatible stored quantization".into(),
        ));
    }
    Ok(Some(Tier::Bf16))
}

pub(crate) fn release_shards(
    dir: &Path,
    config: &Value,
) -> Result<Option<Vec<std::path::PathBuf>>> {
    Ok(tier(config)?.map(|tier| tier.shards(dir)))
}

impl Weights {
    /// Load an HF directory with its wrapper configuration. Preserve the target
    /// shard set and check the native tier before materializing its tensors.
    pub fn from_configured_dir(dir: &Path, config: &Value) -> Result<Self> {
        let Some(tier) = tier(config)? else {
            return Weights::from_dir(dir);
        };
        sensenova_loop::validate_sidecars(dir).map_err(Error::Config)?;
        let stored = match tier {
            Tier::Bf16 => {
                sensenova_loop::validate_source_shards(dir).map_err(Error::Config)?;
                None
            }
            Tier::Affine(spec) => Some(spec),
        };
        let weights = Weights::from_shards(tier.shards(dir))?;
        validate_layout(&weights, stored)?;
        Ok(weights)
    }
}

fn validate_layout(weights: &Weights, stored: Option<QuantSpec>) -> Result<()> {
    let mut count = 0;
    let mut check = |key: &str, shape: &[usize], dtype: Dtype| -> Result<()> {
        let tensor = weights
            .metadata(key)
            .ok_or_else(|| Error::MissingTensor(key.into()))?;
        if tensor.dtype() != dtype
            || !tensor
                .shape()
                .iter()
                .map(|&d| d as usize)
                .eq(shape.iter().copied())
        {
            return Err(Error::Config(format!(
                "SenseNova-Loop-9B {key}: expected {dtype:?} {shape:?}, found {:?} {:?}",
                tensor.dtype(),
                tensor.shape()
            )));
        }
        count += 1;
        Ok(())
    };
    for (name, shape, kind) in sensenova_loop::target_tensors() {
        if let Some(spec) = stored.filter(|_| is_projection(&name)) {
            let base = name.strip_suffix(".weight").expect("projection key");
            let rows = shape[0];
            check(
                &name,
                &[rows, shape[1] * spec.bits as usize / 32],
                Dtype::Uint32,
            )?;
            let grid = [rows, shape[1] / spec.group_size as usize];
            for suffix in ["scales", "biases"] {
                check(&format!("{base}.{suffix}"), &grid, Dtype::Bfloat16)?;
            }
        } else {
            check(
                &name,
                shape,
                if kind == "F32" {
                    Dtype::Float32
                } else {
                    Dtype::Bfloat16
                },
            )?;
        }
    }
    if weights.len() != count {
        return Err(Error::Config(format!(
            "SenseNova-Loop-9B expected {count} target tensors, found {}",
            weights.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn sensenova_packed_tiers_keep_routers_and_vision_dense() {
        let keys: HashSet<_> = sensenova_loop::target_tensors()
            .map(|(key, _, _)| key)
            .collect();
        assert_eq!(keys.len(), 857);
        assert_eq!(keys.iter().filter(|k| is_projection(k)).count(), 200);
        for (k, shape, dtype) in sensenova_loop::target_tensors() {
            if k.contains(".mhc_") {
                assert_eq!(dtype, "F32");
                assert_eq!(
                    shape,
                    if k.ends_with(".bias") {
                        &[8][..]
                    } else {
                        &[8, 8192][..]
                    }
                );
                assert!(!is_projection(&k));
            }
            if k.starts_with("model.visual.") {
                assert!(!is_projection(&k));
            }
        }
    }

    #[test]
    fn sensenova_resolver_requires_one_native_packing_scheme() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("config.json");
        let spec = core_llm::LoadSpec::dense(root.path().display().to_string());
        let mut config: Value = serde_json::from_str(include_str!(
            "../../../contracts/core-llm/src/sensenova_loop.json"
        ))
        .unwrap();
        config["sensenova_mlx"] = manifest(4);
        config["quantization"] = json!({"bits":4,"group_size":64});
        std::fs::write(&path, config.to_string()).unwrap();
        assert!(crate::provider::can_load(&spec));
        assert!(crate::provider::weightless_vision(&spec));
        config["quantization_config"] = json!({"quant_method":"awq"});
        std::fs::write(&path, config.to_string()).unwrap();
        assert!(!crate::provider::can_load(&spec));
        assert!(!crate::provider::weightless_vision(&spec));
        config
            .as_object_mut()
            .unwrap()
            .remove("quantization_config");
        config["quantization"]["storage"] = json!("ggml");
        std::fs::write(&path, config.to_string()).unwrap();
        assert!(!crate::provider::can_load(&spec));
    }

    #[test]
    fn sensenova_rejects_foreign_pack_and_selects_target_shards() {
        let mut config: Value = serde_json::from_str(include_str!(
            "../../../contracts/core-llm/src/sensenova_loop.json"
        ))
        .unwrap();
        let root = Path::new("snapshot");
        let shards = release_shards(root, &config).unwrap().unwrap();
        assert_eq!(shards.len(), 11);
        assert!(!shards.contains(&root.join("mtp.safetensors")));
        for bits in [4, 8] {
            config["quantization"] = json!({"bits":bits,"group_size":64});
            config["sensenova_mlx"] = manifest(bits);
            assert_eq!(
                release_shards(root, &config).unwrap().unwrap(),
                [root.join("model.safetensors")]
            );
            for (path, field) in [
                ("", "quantization_config"),
                ("/text_config", "quantization"),
                ("/quantization", "storage"),
                ("/quantization", "mode"),
            ] {
                let mut ambiguous = config.clone();
                ambiguous.pointer_mut(path).unwrap()[field] = json!("foreign");
                assert!(release_shards(root, &ambiguous).is_err(), "{path}/{field}");
            }
            config["sensenova_mlx"]["format"] = json!("foreign");
            assert!(release_shards(root, &config).is_err());
        }
    }
}

#[cfg(test)]
mod incomplete {
    use super::*;

    #[test]
    fn sensenova_incomplete_target_and_sidecars_name_missing_input() {
        let root = tempfile::tempdir().unwrap();
        let config: Value = serde_json::from_str(include_str!(
            "../../../contracts/core-llm/src/sensenova_loop.json"
        ))
        .unwrap();
        let sidecar = Weights::from_configured_dir(root.path(), &config)
            .err()
            .unwrap()
            .to_string();
        assert!(sidecar.contains("tokenizer.json"));
        let empty = Weights::from_map(Default::default());
        let missing = validate_layout(&empty, None).unwrap_err().to_string();
        assert!(missing.contains("lm_head.weight"));
    }
}
