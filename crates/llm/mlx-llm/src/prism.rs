//! Strict loader for Prism/Bonsai Hadamard-packed MLX snapshots.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use core_llm::{PrismHadamardMetadata, PrismTransformRole};
use mlx_rs::{Array, Dtype};

use crate::error::{Error, Result};
use crate::primitives::prism::{PrismEmbedding, PrismLinear};
use crate::primitives::Weights;

#[derive(Clone, Debug)]
struct ModuleEntry {
    path: String,
    block: usize,
    embedding: bool,
    dtype: String,
}

/// Validated metadata needed to construct packed MLX operators without expanding their weights.
#[derive(Clone, Debug)]
pub struct PrismMlxPack {
    modules: BTreeMap<String, ModuleEntry>,
    hadamard: PrismHadamardMetadata,
    /// The published VLM checkpoint carries a dense `vision_tower.*` sidecar. It is retained in the
    /// loaded weight map for the multimodal integration and is never mistaken for text weights.
    pub has_vision_tower: bool,
}

impl PrismMlxPack {
    pub(crate) fn from_gguf(
        hadamard: PrismHadamardMetadata,
        module_specs: impl IntoIterator<Item = (String, bool)>,
    ) -> Result<Self> {
        hadamard
            .validate()
            .map_err(|e| Error::Config(e.to_string()))?;
        let modules: BTreeMap<_, _> = module_specs
            .into_iter()
            .map(|(path, embedding)| {
                let entry = ModuleEntry {
                    path: path.clone(),
                    block: hadamard.block_size,
                    embedding,
                    dtype: "float16".into(),
                };
                (path, entry)
            })
            .collect();
        let expected: BTreeSet<_> = modules
            .values()
            .map(|m| format!("language_model.{}.weight", m.path))
            .collect();
        let declared: BTreeSet<_> = hadamard
            .forward_weight_names
            .union(&hadamard.inverse_weight_names)
            .cloned()
            .collect();
        if expected != declared {
            return Err(Error::Config(
                "GGUF Prism packed tensors and canonical Hadamard manifest differ".into(),
            ));
        }
        for module in modules.values() {
            let role = hadamard.role(&format!("language_model.{}.weight", module.path));
            if module.embedding != (role == PrismTransformRole::Inverse) {
                return Err(Error::Config(format!(
                    "GGUF Prism module `{}` has the wrong transform role",
                    module.path
                )));
            }
        }
        Ok(Self {
            modules,
            hadamard,
            has_vision_tower: false,
        })
    }

    /// Read and validate the frozen Prism schema plus its explicit Hadamard contract.
    pub fn from_dir(dir: &Path, config: &serde_json::Value, weights: &Weights) -> Result<Self> {
        require_eq(config, "schema_version", &serde_json::json!(2))?;
        require_eq(
            config,
            "model_type",
            &serde_json::json!("prism_hadamard_qwen35"),
        )?;
        require_eq(config, "base_model_type", &serde_json::json!("qwen3_5"))?;
        require_eq(
            config,
            "tensor_namespace",
            &serde_json::json!("mlx-vlm-qwen3_5"),
        )?;
        require_eq(
            config,
            "requires_runtime",
            &serde_json::json!("runtime/artifact.py"),
        )?;
        require_eq(
            config,
            "hadamard_config",
            &serde_json::json!("hadamard.json"),
        )?;
        require_eq(
            config,
            "gdn_activation_layout",
            &serde_json::json!("grouped"),
        )?;
        let quant = config
            .get("quantization")
            .ok_or_else(|| Error::Config("Prism config missing `quantization`".into()))?;
        for (key, expected) in [
            ("bits", serde_json::json!(2)),
            ("group_size", serde_json::json!(128)),
            ("mode", serde_json::json!("affine")),
        ] {
            require_eq(quant, key, &expected)?;
        }
        let components = config
            .get("components")
            .ok_or_else(|| Error::Config("Prism config missing `components`".into()))?;
        require_eq(components, "text", &serde_json::json!(true))?;
        require_eq(components, "mtp", &serde_json::json!(false))?;

        let entries = config
            .get("modules")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::Config("Prism config missing `modules`".into()))?;
        let mut modules = BTreeMap::new();
        for value in entries {
            let entry = ModuleEntry {
                path: value
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned(),
                block: value
                    .get("block")
                    .and_then(|v| v.as_u64())
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(0),
                embedding: value
                    .get("embedding")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
                dtype: value
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned(),
            };
            if entry.path.is_empty()
                || entry.block == 0
                || !entry.block.is_power_of_two()
                || entry.dtype != "float16"
                || modules.insert(entry.path.clone(), entry).is_some()
            {
                return Err(Error::Config(
                    "Prism modules contain an invalid or duplicate entry".into(),
                ));
            }
        }

        let hpath = dir.join("hadamard.json");
        let h: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&hpath)
                .map_err(|e| Error::Config(format!("read {}: {e}", hpath.display())))?,
        )
        .map_err(|e| Error::Config(format!("parse {}: {e}", hpath.display())))?;
        for (key, expected) in [
            ("prism.hadamard.version", serde_json::json!(1)),
            (
                "prism.hadamard.transform",
                serde_json::json!("normalized-sylvester-walsh-hadamard"),
            ),
            (
                "prism.hadamard.axis",
                serde_json::json!("input-last-dimension"),
            ),
            ("prism.hadamard.sign_mode", serde_json::json!("explicit")),
            ("prism.hadamard.gdn_v_grouped", serde_json::json!(true)),
        ] {
            require_eq(&h, key, &expected)?;
        }
        let block_size = h["prism.hadamard.block_size"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::Config("invalid Prism Hadamard block size".into()))?;
        let forward_weight_names = string_set(&h, "prism.hadamard.weight_names")?;
        let inverse_weight_names = string_set(&h, "prism.hadamard.inverse_weight_names")?;
        let widths = usize_array(&h, "prism.hadamard.sign_widths")?;
        let values = h["prism.hadamard.sign_values"]
            .as_array()
            .ok_or_else(|| Error::Config("invalid Prism Hadamard sign values".into()))?;
        let mut signs_by_width = BTreeMap::new();
        let mut at = 0usize;
        for width in widths {
            let end = at
                .checked_add(width)
                .ok_or_else(|| Error::Config("Prism Hadamard sign length overflow".into()))?;
            let slice = values
                .get(at..end)
                .ok_or_else(|| Error::Config("Prism Hadamard signs are truncated".into()))?;
            let signs = slice
                .iter()
                .map(|v| match v.as_f64() {
                    Some(-1.0) => Ok(-1),
                    Some(1.0) => Ok(1),
                    _ => Err(Error::Config("Prism signs must be -1 or +1".into())),
                })
                .collect::<Result<Vec<_>>>()?;
            if signs_by_width.insert(width, signs).is_some() {
                return Err(Error::Config("duplicate Prism sign width".into()));
            }
            at = end;
        }
        if at != values.len() {
            return Err(Error::Config(
                "Prism Hadamard signs have trailing values".into(),
            ));
        }
        let hadamard = PrismHadamardMetadata {
            block_size,
            signs_by_width,
            forward_weight_names,
            inverse_weight_names,
            gdn_v_grouped: true,
        };
        hadamard
            .validate()
            .map_err(|e| Error::Config(e.to_string()))?;

        let expected: BTreeSet<_> = modules
            .values()
            .map(|m| format!("language_model.{}.weight", m.path))
            .collect();
        let declared: BTreeSet<_> = hadamard
            .forward_weight_names
            .union(&hadamard.inverse_weight_names)
            .cloned()
            .collect();
        if expected != declared {
            return Err(Error::Config(
                "Prism module manifest and Hadamard weight sets differ".into(),
            ));
        }
        for module in modules.values() {
            let name = format!("language_model.{}.weight", module.path);
            let role = hadamard.role(&name);
            if module.block != block_size
                || (module.embedding && role != PrismTransformRole::Inverse)
                || (!module.embedding && role != PrismTransformRole::Forward)
            {
                return Err(Error::Config(format!(
                    "Prism module `{}` disagrees with Hadamard metadata",
                    module.path
                )));
            }
            for suffix in ["weight", "scales", "biases", "signs"] {
                let key = format!("language_model.{}.{}", module.path, suffix);
                if !weights.contains(&key) {
                    return Err(Error::MissingTensor(key));
                }
            }
        }
        for key in weights.keys().filter(|key| {
            key.starts_with("language_model.")
                && (key.ends_with(".signs") || key.ends_with(".scales") || key.ends_with(".biases"))
        }) {
            let base = key
                .strip_prefix("language_model.")
                .and_then(|key| key.rsplit_once('.').map(|(base, _)| base))
                .unwrap_or("");
            if !modules.contains_key(base) {
                return Err(Error::Config(format!(
                    "Prism packed part `{key}` is not declared in the module manifest"
                )));
            }
        }
        let has_vision_tower = weights.keys().any(|k| k.starts_with("vision_tower."));
        if components
            .get("vision")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
            != has_vision_tower
        {
            return Err(Error::Config(
                "Prism `components.vision` disagrees with `vision_tower.*` tensors".into(),
            ));
        }
        Ok(Self {
            modules,
            hadamard,
            has_vision_tower,
        })
    }

    pub(crate) fn linear(&self, weights: &Weights, weight_key: &str) -> Result<PrismLinear> {
        let module = self.module(weight_key, false)?;
        let (weight, scales, biases, signs) = parts(weights, weight_key)?;
        validate_affine_parts(weight_key, &scales, &biases)?;
        let width = affine_width(weight_key, &scales)?;
        self.validate_sign_tensor(weight_key, width, &signs)?;
        PrismLinear::new(
            weight_key,
            weight,
            scales,
            biases,
            signs,
            module.block as i32,
        )
    }

    pub(crate) fn embedding(&self, weights: &Weights, weight_key: &str) -> Result<PrismEmbedding> {
        let module = self.module(weight_key, true)?;
        let (weight, scales, biases, signs) = parts(weights, weight_key)?;
        validate_affine_parts(weight_key, &scales, &biases)?;
        let width = affine_width(weight_key, &scales)?;
        self.validate_sign_tensor(weight_key, width, &signs)?;
        PrismEmbedding::new(
            weight_key,
            weight,
            scales,
            biases,
            signs,
            module.block as i32,
        )
    }

    fn module(&self, weight_key: &str, embedding: bool) -> Result<&ModuleEntry> {
        let path = weight_key
            .strip_prefix("language_model.")
            .and_then(|v| v.strip_suffix(".weight"))
            .ok_or_else(|| Error::Config(format!("invalid Prism tensor name `{weight_key}`")))?;
        let module = self
            .modules
            .get(path)
            .ok_or_else(|| Error::Config(format!("packed tensor `{weight_key}` is undeclared")))?;
        if module.embedding != embedding {
            return Err(Error::Config(format!(
                "Prism tensor `{weight_key}` has the wrong operator kind"
            )));
        }
        Ok(module)
    }

    fn validate_sign_tensor(&self, name: &str, width: usize, signs: &Array) -> Result<()> {
        let expected = self
            .hadamard
            .classify_weight(name, width)
            .map_err(|e| Error::Config(e.to_string()))?
            .signs
            .ok_or_else(|| Error::Config(format!("Prism tensor `{name}` has no transform")))?;
        if signs.dtype() != Dtype::Float32 || signs.shape() != [width as i32] {
            return Err(Error::Config(format!(
                "Prism tensor `{name}` has invalid signs"
            )));
        }
        let actual = signs.as_slice::<f32>();
        if actual.iter().zip(expected).any(|(&a, &b)| a != b as f32) {
            return Err(Error::Config(format!(
                "Prism tensor `{name}` signs disagree with hadamard.json"
            )));
        }
        Ok(())
    }
}

fn validate_affine_parts(name: &str, scales: &Array, biases: &Array) -> Result<()> {
    if scales.dtype() != Dtype::Float16
        || biases.dtype() != Dtype::Float16
        || scales.shape().len() != 2
        || scales.shape() != biases.shape()
    {
        return Err(Error::Config(format!(
            "Prism tensor `{name}` scales and biases must be F16"
        )));
    }
    let scale32 = scales.as_dtype(Dtype::Float32)?;
    let bias32 = biases.as_dtype(Dtype::Float32)?;
    if scale32
        .as_slice::<f32>()
        .iter()
        .zip(bias32.as_slice::<f32>())
        .any(|(&scale, &bias)| !scale.is_finite() || bias != -scale)
    {
        return Err(Error::Config(format!(
            "Prism tensor `{name}` requires finite affine bias == -scale"
        )));
    }
    Ok(())
}

fn affine_width(name: &str, scales: &Array) -> Result<usize> {
    let groups = scales.shape().get(1).copied().unwrap_or(0);
    if groups <= 0 {
        return Err(Error::Config(format!(
            "Prism tensor `{name}` has invalid affine geometry"
        )));
    }
    usize::try_from(groups)
        .ok()
        .and_then(|n| n.checked_mul(128))
        .ok_or_else(|| Error::Config(format!("Prism tensor `{name}` width overflow")))
}

fn parts(weights: &Weights, weight_key: &str) -> Result<(Array, Array, Array, Array)> {
    let base = weight_key
        .strip_suffix(".weight")
        .expect("validated weight suffix");
    Ok((
        weights.require(weight_key)?.clone(),
        weights.require(&format!("{base}.scales"))?.clone(),
        weights.require(&format!("{base}.biases"))?.clone(),
        weights.require(&format!("{base}.signs"))?.clone(),
    ))
}

fn require_eq(value: &serde_json::Value, key: &str, expected: &serde_json::Value) -> Result<()> {
    if value.get(key) != Some(expected) {
        return Err(Error::Config(format!(
            "Prism `{key}` must be {expected}, got {}",
            value.get(key).unwrap_or(&serde_json::Value::Null)
        )));
    }
    Ok(())
}

fn string_set(value: &serde_json::Value, key: &str) -> Result<BTreeSet<String>> {
    value[key]
        .as_array()
        .ok_or_else(|| Error::Config(format!("Prism `{key}` must be an array")))?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| Error::Config(format!("Prism `{key}` contains a non-string")))
        })
        .collect()
}

fn usize_array(value: &serde_json::Value, key: &str) -> Result<Vec<usize>> {
    value[key]
        .as_array()
        .ok_or_else(|| Error::Config(format!("Prism `{key}` must be an array")))?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| Error::Config(format!("Prism `{key}` contains an invalid width")))
        })
        .collect()
}
