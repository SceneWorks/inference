//! ModelOpt NVFP4 checkpoints on the Candle lane (SW-7099).
//!
//! Two consumers of one byte-level import ([`Nvfp4Tensor::from_modelopt_parts`]):
//!
//! - **Native load** ([`modelopt_projection`]): on a device that serves NVFP4 (CUDA sm_120), a
//!   ModelOpt projection is staged as the resident NVFP4 operand exactly as published — never
//!   dequantized and requantized. Any other format request over a ModelOpt projection is refused
//!   by name ([`core_llm::modelopt::dense_runtime_refusal`]): its U8 codes are not dense weights.
//! - **Tier conversion** ([`convert_snapshot_to_bf16`]): a CPU pass that writes the dequantized
//!   bf16 snapshot every other runtime starts from (MLX and non-Blackwell CUDA; `prepare_snapshot`
//!   then writes their Q8 / Q4 tiers). Example: `modelopt_nvfp4_to_bf16`.
//!
//! The bf16 tier is not bit-exact: an element is `e2m1 · e4m3 · weight_scale_2`, and the f32
//! per-tensor multiplier makes the product's mantissa wider than bf16's 8 bits, so each element
//! carries a relative rounding error of at most 2⁻⁹ — far below the NVFP4 quantization step it was
//! decoded from.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_quant_kernels::{Nvfp4Tensor, Nvfp4Weight};
use core_llm::modelopt::{self, INPUT_SCALE, WEIGHT_SCALE, WEIGHT_SCALE_2};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::primitives::projection::{Projection, ProjectionFormat};
use crate::primitives::Weights;

/// Read the ModelOpt triple of projection `stem` from host tensors into the canonical container,
/// or `None` when `stem` is not a ModelOpt projection (no `.weight_scale` sibling).
pub fn modelopt_packed(
    get: impl Fn(&str) -> Option<Tensor>,
    stem: &str,
) -> Result<Option<Nvfp4Tensor>> {
    let Some(scales) = get(&format!("{stem}{WEIGHT_SCALE}")) else {
        return Ok(None);
    };
    let missing = |what: &str| Error::Config(format!("ModelOpt projection `{stem}` has no {what}"));
    let weight = get(&format!("{stem}.weight")).ok_or_else(|| missing(".weight"))?;
    let scale_2 = get(&format!("{stem}{WEIGHT_SCALE_2}")).ok_or_else(|| missing(WEIGHT_SCALE_2))?;
    let (rows, half) = weight.dims2()?;
    let cols = half * 2;
    if weight.dtype() != DType::U8 || scales.dtype() != DType::U8 {
        return Err(Error::Config(format!(
            "ModelOpt projection `{stem}`: expected U8 codes and E4M3 scale bytes, got {:?} / {:?}",
            weight.dtype(),
            scales.dtype()
        )));
    }
    if scales.dims() != [rows, cols / modelopt::GROUP_SIZE] {
        return Err(Error::Config(format!(
            "ModelOpt projection `{stem}`: scales {:?} do not match codes [{rows}, {half}]",
            scales.dims()
        )));
    }
    let global = scale_2
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let [global] = global[..] else {
        return Err(Error::Config(format!(
            "ModelOpt projection `{stem}`: {WEIGHT_SCALE_2} is not a scalar"
        )));
    };
    let packed = weight.flatten_all()?.to_vec1::<u8>()?;
    let block_scales = scales.flatten_all()?.to_vec1::<u8>()?;
    Ok(Some(Nvfp4Tensor::from_modelopt_parts(
        &packed,
        &block_scales,
        global,
        rows,
        cols,
    )?))
}

/// The projection `<stem>.weight` as a ModelOpt NVFP4 load builds it, or `None` when it is a
/// dense weight. Under an NVFP4 `format` the published codes are staged on `device` unchanged;
/// any other format is refused by name rather than reading U8 codes as a dense weight.
pub(crate) fn modelopt_projection(
    w: &Weights,
    key: &str,
    format: Option<&ProjectionFormat>,
    device: &Device,
) -> Result<Option<Projection>> {
    let Some(stem) = key.strip_suffix(".weight") else {
        return Ok(None);
    };
    if !w.contains(&format!("{stem}{WEIGHT_SCALE}")) {
        return Ok(None);
    }
    let Some(ProjectionFormat::Nvfp4(ctx)) = format else {
        return Err(Error::Unsupported(format!(
            "{} (projection `{stem}` requested as {format:?})",
            modelopt::dense_runtime_refusal("a non-NVFP4 Candle load")
        )));
    };
    let packed = modelopt_packed(|k| w.get(k).cloned(), stem)?.expect("scale sibling checked");
    let weight = Nvfp4Weight::from_packed(&packed, None, device, ctx)?;
    Ok(Some(Projection::Nvfp4(std::sync::Arc::new(weight))))
}

/// What [`convert_snapshot_to_bf16`] wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvertReport {
    /// Projections decoded from NVFP4 to bf16.
    pub dequantized: usize,
    /// Tensors copied through unchanged.
    pub passthrough: usize,
    /// Safetensors shards written.
    pub shards: usize,
}

fn copy_tree(src: &Path, out: &Path, skip: &dyn Fn(&Path) -> bool) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let path = entry?.path();
        if skip(&path) {
            continue;
        }
        let target = out.join(path.file_name().expect("read_dir entries are named"));
        if path.is_dir() {
            std::fs::create_dir_all(&target)?;
            copy_tree(&path, &target, &|_| false)?;
        } else {
            std::fs::copy(&path, &target)?;
        }
    }
    Ok(())
}

/// Dequantize a ModelOpt NVFP4 snapshot at `source` into a dense bf16 snapshot at `out_dir`
/// (created; must be empty or absent), on the CPU, one shard at a time.
///
/// Every ModelOpt projection becomes a bf16 `<stem>.weight`; its `.weight_scale`,
/// `.weight_scale_2` and `.input_scale` are dropped. Every other tensor is copied byte for byte
/// (bf16 stays bf16). `config.json` loses its `quantization_config`; `hf_quant_config.json` and the
/// source index are not copied (a new index is written); every other file and directory — the
/// tokenizer, chat template, `generation_config.json`, `processor_config.json` and the model's
/// own `prompt/` files — is copied unchanged.
pub fn convert_snapshot_to_bf16(source: &Path, out_dir: &Path) -> Result<ConvertReport> {
    let config: Value = serde_json::from_str(&std::fs::read_to_string(source.join("config.json"))?)
        .map_err(|e| Error::Config(format!("config.json: {e}")))?;
    if modelopt::modelopt_nvfp4(&config)
        .map_err(Error::Unsupported)?
        .is_none()
    {
        return Err(Error::Config(format!(
            "{} is not a ModelOpt NVFP4 snapshot",
            source.display()
        )));
    }
    if out_dir.exists() && std::fs::read_dir(out_dir)?.next().is_some() {
        return Err(Error::Config(format!(
            "output directory {} is not empty",
            out_dir.display()
        )));
    }
    std::fs::create_dir_all(out_dir)?;

    let mut shards: Vec<_> = std::fs::read_dir(source)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(Error::Config(format!(
            "no .safetensors shards in {}",
            source.display()
        )));
    }
    // Which stems are quantized, across every shard (a triple never splits in practice, but the
    // index allows it, so look the companions up snapshot-wide).
    let mut all: HashMap<String, Tensor> = HashMap::new();
    let mut report = ConvertReport {
        dequantized: 0,
        passthrough: 0,
        shards: shards.len(),
    };
    let mut weight_map = serde_json::Map::new();
    let mut total_bytes = 0u64;
    let shard_count = shards.len();
    for (index, shard) in shards.iter().enumerate() {
        let weights = Weights::from_file_host_raw(shard)?;
        all.extend(weights.into_map());
        // Emit every tensor whose triple is now complete; keep incomplete triples for the next
        // shard. With the published layout every shard is self-contained.
        let ready: Vec<String> = all
            .keys()
            .filter(|k| !modelopt::is_companion(k))
            .filter(
                |k| match modelopt::quantized_stem(k, |n| all.contains_key(n)) {
                    Some(stem) => all.contains_key(&format!("{stem}{WEIGHT_SCALE_2}")),
                    None => {
                        // A `.weight` whose scale may still arrive in a later shard stays pending
                        // only if it is U8 codes.
                        !(all[*k].dtype() == DType::U8 && index + 1 < shard_count)
                    }
                },
            )
            .cloned()
            .collect();
        let mut out: HashMap<String, Tensor> = HashMap::new();
        for key in ready {
            let tensor = all.remove(&key).expect("listed above");
            let stem = key.strip_suffix(".weight");
            let packed = match stem {
                Some(stem) => modelopt_packed(
                    |k| {
                        if k == key {
                            Some(tensor.clone())
                        } else {
                            all.get(k).cloned()
                        }
                    },
                    stem,
                )?,
                None => None,
            };
            let converted = match (packed, stem) {
                (Some(packed), Some(stem)) => {
                    for suffix in [WEIGHT_SCALE, WEIGHT_SCALE_2, INPUT_SCALE] {
                        all.remove(&format!("{stem}{suffix}"));
                    }
                    report.dequantized += 1;
                    let (rows, cols) = (packed.rows, packed.cols);
                    Tensor::from_vec(packed.dequantize_to_vec(), (rows, cols), &Device::Cpu)?
                        .to_dtype(DType::BF16)?
                }
                _ => {
                    if tensor.dtype() == DType::U8 && key.ends_with(".weight") {
                        return Err(Error::Config(format!(
                            "U8 tensor `{key}` has no ModelOpt scales; refusing to copy codes \
                             as a dense weight"
                        )));
                    }
                    report.passthrough += 1;
                    tensor
                }
            };
            total_bytes += (converted.elem_count() * converted.dtype().size_in_bytes()) as u64;
            out.insert(key, converted);
        }
        let name = format!("model-{:05}-of-{:05}.safetensors", index + 1, shard_count);
        for key in out.keys() {
            weight_map.insert(key.clone(), Value::String(name.clone()));
        }
        candle_core::safetensors::save(&out, out_dir.join(&name))?;
    }
    let leftovers: Vec<&String> = all.keys().filter(|k| !k.ends_with(INPUT_SCALE)).collect();
    if !leftovers.is_empty() {
        return Err(Error::Config(format!(
            "ModelOpt tensors without a complete triple: {leftovers:?}"
        )));
    }

    let mut dense_config = config;
    if let Some(obj) = dense_config.as_object_mut() {
        obj.remove("quantization_config");
        obj.insert("dtype".into(), Value::String("bfloat16".into()));
    }
    if let Some(text) = dense_config
        .get_mut("text_config")
        .and_then(Value::as_object_mut)
    {
        text.remove("quantization_config");
    }
    let skip = |p: &Path| {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        name == "config.json"
            || name == "hf_quant_config.json"
            || name == "model.safetensors.index.json"
            || name.ends_with(".safetensors")
            || name.starts_with('.')
    };
    copy_tree(source, out_dir, &skip)?;
    std::fs::write(
        out_dir.join("config.json"),
        serde_json::to_string_pretty(&dense_config).map_err(|e| Error::Msg(e.to_string()))?,
    )?;
    let index = serde_json::json!({
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    });
    std::fs::write(
        out_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).map_err(|e| Error::Msg(e.to_string()))?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-projection ModelOpt snapshot: `[rows, 32]` codes, linear scales, a scalar
    /// multiplier, an activation scale, and a dense bf16 norm.
    fn tiny_snapshot(dir: &Path) {
        let (rows, cols) = (2usize, 32usize);
        let mut codes = vec![0u8; rows * cols / 2];
        codes[0] = 0xB2; // row 0: +1.0, -1.5
        codes[16 + 8] = 0x17; // row 1, cols 16/17: +6.0, +0.5
        let mut tensors = HashMap::new();
        tensors.insert(
            "layer.proj.weight".to_string(),
            Tensor::from_vec(codes, (rows, cols / 2), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "layer.proj.weight_scale".to_string(),
            Tensor::from_vec(vec![0x40u8, 0x38, 0x38, 0x30], (rows, 2), &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "layer.proj.weight_scale_2".to_string(),
            Tensor::new(0.25f32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "layer.proj.input_scale".to_string(),
            Tensor::new(1.0f32, &Device::Cpu).unwrap(),
        );
        tensors.insert(
            "layer.norm.weight".to_string(),
            Tensor::new(&[1.0f32, 2.0], &Device::Cpu)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap(),
        );
        candle_core::safetensors::save(&tensors, dir.join("model-00001-of-00001.safetensors"))
            .unwrap();
        let config = serde_json::json!({
            "model_type": "qwen3_5",
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                                    "group_size": 16},
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("hf_quant_config.json"), "{}").unwrap();
        std::fs::create_dir_all(dir.join("prompt/t2i")).unwrap();
        std::fs::write(dir.join("prompt/t2i/x.txt"), "kept").unwrap();
    }

    #[test]
    fn a_modelopt_snapshot_converts_to_dequantized_bf16() {
        let src = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        tiny_snapshot(src.path());
        let report = convert_snapshot_to_bf16(src.path(), out.path()).unwrap();
        assert_eq!(
            report,
            ConvertReport {
                dequantized: 1,
                passthrough: 1,
                shards: 1
            }
        );
        let back = candle_core::safetensors::load(
            out.path().join("model-00001-of-00001.safetensors"),
            &Device::Cpu,
        )
        .unwrap();
        let mut keys: Vec<_> = back.keys().cloned().collect();
        keys.sort();
        assert_eq!(keys, ["layer.norm.weight", "layer.proj.weight"]);
        let w = &back["layer.proj.weight"];
        assert_eq!((w.dtype(), w.dims()), (DType::BF16, &[2usize, 32][..]));
        let v = w.to_dtype(DType::F32).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(v[0][0], 0.5);
        assert_eq!(v[0][1], -0.75);
        assert_eq!(v[1][16], 0.75);
        assert_eq!(v[1][17], 0.0625);
        let config: Value =
            serde_json::from_str(&std::fs::read_to_string(out.path().join("config.json")).unwrap())
                .unwrap();
        assert!(config.get("quantization_config").is_none());
        assert!(!out.path().join("hf_quant_config.json").exists());
        assert_eq!(
            std::fs::read_to_string(out.path().join("prompt/t2i/x.txt")).unwrap(),
            "kept"
        );
        let index: Value = serde_json::from_str(
            &std::fs::read_to_string(out.path().join("model.safetensors.index.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            index["weight_map"]["layer.proj.weight"],
            "model-00001-of-00001.safetensors"
        );
    }

    #[test]
    fn a_dense_snapshot_is_not_converted() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(
            src.path().join("config.json"),
            r#"{"model_type":"qwen3_5"}"#,
        )
        .unwrap();
        let out = tempfile::tempdir().unwrap();
        let err = convert_snapshot_to_bf16(src.path(), &out.path().join("o")).unwrap_err();
        assert!(err.to_string().contains("not a ModelOpt NVFP4"), "{err}");
    }

    #[test]
    fn host_loading_keeps_the_triple_as_raw_bytes() {
        let src = tempfile::tempdir().unwrap();
        tiny_snapshot(src.path());
        let w = Weights::from_dir(src.path(), &Device::Cpu).unwrap();
        let packed = modelopt_packed(|k| w.get(k).cloned(), "layer.proj")
            .unwrap()
            .unwrap();
        assert_eq!(
            (packed.rows, packed.cols, packed.global_scale),
            (2, 32, 0.25)
        );
        assert!(modelopt_packed(|k| w.get(k).cloned(), "layer.norm")
            .unwrap()
            .is_none());
    }

    #[test]
    fn a_non_nvfp4_load_of_a_modelopt_projection_is_refused_by_name() {
        let src = tempfile::tempdir().unwrap();
        tiny_snapshot(src.path());
        let w = Weights::from_dir(src.path(), &Device::Cpu).unwrap();
        let err = match modelopt_projection(&w, "layer.proj.weight", None, &Device::Cpu) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a dense load of NVFP4 codes must refuse"),
        };
        assert!(err.contains("nvfp4: ") && err.contains("ModelOpt"), "{err}");
        assert!(
            modelopt_projection(&w, "layer.norm.weight", None, &Device::Cpu)
                .unwrap()
                .is_none()
        );
    }
}
