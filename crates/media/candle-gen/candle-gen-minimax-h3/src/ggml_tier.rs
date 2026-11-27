//! Offline conversion of a published MiniMax-H3 **MLX-affine** tier into a **Candle-native GGML**
//! tier — the artifact `SceneWorks/minimax-h3-candle` publishes, so a Windows/Linux q4/q8 install
//! loads its projections as-is instead of repacking ~130 GB of MLX triples on first load.
//!
//! # What changes, and what is byte-for-byte unchanged
//!
//! A component directory (`transformer`, `transformer_ref` or `text_encoder`) is converted shard by
//! shard, keeping every shard's file name:
//!
//! * every MLX-packed triple — `{base}.weight` (U32 codes) + `{base}.scales` + `{base}.biases` — is
//!   replaced by ONE `{base}.weight` **stored GGML block tensor**, `U8 [out, in / 32,
//!   block_bytes]` ([`candle_gen::candle_layers::llm_projection::to_ggml_block_tensor`]). Its blocks
//!   are produced by [`candle_gen::quant::repack_packed_weight_uncached`] — the **same** function the
//!   load-time repack calls — so the resident `QTensor` a GGML tier loads is the one an MLX tier
//!   would have repacked into: `Q4_1` for q4 (lossless up to the f16 scale/min cast), `Q8_0` for q8
//!   (the accepted sc-9085 affine-grid re-quant). The rendered model does not change.
//! * every other tensor — the dense-by-policy heads, norms, biases and the vision tower's unpacked
//!   `linear_fc2` — is copied at its stored dtype and shape;
//! * `*.index.json` drops the `.scales` / `.biases` keys of converted bases and restates
//!   `metadata.total_size`;
//! * `config.json`'s `quantization` block becomes `{"bits": N, "storage": "ggml"}` (the
//!   `candle-llm` sc-19375 marker). `bits` keeps the tier's width so the tier reconcile
//!   ([`crate::tier`]) and the group-size check still read it; the GGML block size is intrinsic to
//!   the stored blocks, so no `group_size` is written;
//! * every other sidecar (tokenizer files, preprocessor configs, chat template) is copied verbatim.
//!
//! The output is written to `<dst>.partial` and renamed into place only after every file is
//! complete, so an interrupted run never leaves a directory that looks finished.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_layers::llm_projection::to_ggml_block_tensor;
use candle_gen::quant::{repack_packed_weight_uncached, MLX_GROUP_SIZE};
use candle_gen::{CandleError, Result};
use serde_json::Value;

use crate::quant::GGML_STORAGE;

/// The three tiered component directories a Candle-native tier carries, matching the MLX rehost's
/// `{q4,q8}/{transformer,transformer_ref,text_encoder}` layout.
pub const COMPONENTS: [&str; 3] = ["transformer", "transformer_ref", "text_encoder"];

/// What one component conversion did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComponentReport {
    /// Shards written.
    pub shards: usize,
    /// MLX triples replaced by stored GGML blocks.
    pub converted: usize,
    /// Tensors copied unchanged.
    pub passthrough: usize,
    /// Total tensor payload bytes written (the restated `metadata.total_size`).
    pub tensor_bytes: u64,
}

fn io<T>(r: std::io::Result<T>, what: impl std::fmt::Display) -> Result<T> {
    r.map_err(|e| CandleError::Msg(format!("minimax-h3 ggml tier: {what}: {e}")))
}

fn msg(text: impl Into<String>) -> CandleError {
    CandleError::Msg(format!("minimax-h3 ggml tier: {}", text.into()))
}

/// Rewrite every `quantization` object that carries `bits` into the GGML marker, recursively (the
/// text encoder's block is top level; a nested `text_config` block would be rewritten the same way).
/// Returns how many blocks were rewritten.
fn mark_ggml(value: &mut Value) -> usize {
    let mut n = 0;
    match value {
        Value::Object(map) => {
            if let Some(Value::Object(q)) = map.get_mut("quantization") {
                if let Some(bits) = q.get("bits").cloned() {
                    q.clear();
                    q.insert("bits".into(), bits);
                    q.insert("storage".into(), Value::from(GGML_STORAGE));
                    n += 1;
                }
            }
            for (key, child) in map.iter_mut() {
                if key != "quantization" {
                    n += mark_ggml(child);
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|v| n += mark_ggml(v)),
        _ => {}
    }
    n
}

/// The rewritten `config.json` text for a GGML tier. Refuses a config with no packed marker: a
/// dense component has nothing to convert and must not be relabelled.
pub fn ggml_config(text: &str) -> Result<String> {
    let mut config: Value =
        serde_json::from_str(text).map_err(|e| msg(format!("config.json: {e}")))?;
    if mark_ggml(&mut config) == 0 {
        return Err(msg(
            "config.json carries no `quantization.bits` block — only a packed MLX tier converts",
        ));
    }
    serde_json::to_string_pretty(&config).map_err(|e| msg(format!("config.json: {e}")))
}

/// Convert one shard's tensors: each MLX triple becomes stored GGML blocks, everything else passes
/// through unchanged. `report` is updated in place.
pub fn convert_tensors(
    tensors: HashMap<String, Tensor>,
    report: &mut ComponentReport,
) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(tensors.len());
    for (name, t) in &tensors {
        if let Some(base) = name
            .strip_suffix(".scales")
            .or_else(|| name.strip_suffix(".biases"))
        {
            if !tensors.contains_key(&format!("{base}.weight")) {
                return Err(msg(format!("`{name}` has no `{base}.weight` in its shard")));
            }
            continue;
        }
        let packed = name.strip_suffix(".weight").and_then(|base| {
            let scales = tensors.get(&format!("{base}.scales"))?;
            Some((base, scales))
        });
        let Some((base, scales)) = packed else {
            out.insert(name.clone(), t.clone());
            report.passthrough += 1;
            continue;
        };
        let biases = tensors
            .get(&format!("{base}.biases"))
            .ok_or_else(|| msg(format!("`{base}.scales` has no `{base}.biases`")))?;
        if t.dtype() != DType::U32 {
            return Err(msg(format!(
                "`{name}` is {:?}, expected the U32 code stream of an MLX pack",
                t.dtype()
            )));
        }
        // The load-time repack receives scales/biases at the compute dtype; widening bf16 to f32 is
        // exact, and the repack rounds them to the block's f16 fields either way.
        let scales = scales.to_dtype(DType::F32)?;
        let biases = biases.to_dtype(DType::F32)?;
        let qt = repack_packed_weight_uncached(t, &scales, &biases, MLX_GROUP_SIZE, &Device::Cpu)?;
        let blocks = to_ggml_block_tensor(&qt).map_err(|e| msg(format!("{base}: {e}")))?;
        out.insert(name.clone(), blocks);
        report.converted += 1;
    }
    Ok(out)
}

fn component_files(src: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = io(std::fs::read_dir(src), src.display())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .filter(|p| {
            !p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|n| n.starts_with('.'))
        })
        .collect();
    files.sort();
    Ok(files)
}

fn is_shard(p: &Path) -> bool {
    p.extension().and_then(|s| s.to_str()) == Some("safetensors")
}

/// Convert the MLX-tier component directory `src` into a GGML-tier component at `dst`.
///
/// `dst` must not exist. `progress` is called after each shard with that shard's output path and
/// the running report. See the module docs for the exact transformation.
pub fn convert_component(
    src: &Path,
    dst: &Path,
    progress: &mut dyn FnMut(&Path, &ComponentReport) -> Result<()>,
) -> Result<ComponentReport> {
    if dst.exists() {
        return Err(msg(format!(
            "{} already exists — refusing to overwrite a tier",
            dst.display()
        )));
    }
    let partial = dst.with_extension("partial");
    if partial.exists() {
        io(std::fs::remove_dir_all(&partial), partial.display())?;
    }
    io(std::fs::create_dir_all(&partial), partial.display())?;

    let files = component_files(src)?;
    let shards: Vec<&PathBuf> = files.iter().filter(|p| is_shard(p)).collect();
    if shards.is_empty() {
        return Err(msg(format!("no .safetensors in {}", src.display())));
    }
    let mut report = ComponentReport::default();
    let mut weight_map = serde_json::Map::new();
    for shard in &shards {
        let name = shard.file_name().expect("shard file name");
        let tensors = candle_gen::candle_core::safetensors::load(shard, &Device::Cpu)?;
        let out = convert_tensors(tensors, &mut report)?;
        for (key, t) in &out {
            report.tensor_bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
            weight_map.insert(
                key.clone(),
                Value::from(name.to_string_lossy().into_owned()),
            );
        }
        let target = partial.join(name);
        candle_gen::candle_core::safetensors::save(&out, &target)?;
        report.shards += 1;
        progress(&target, &report)?;
    }

    for file in files.iter().filter(|p| !is_shard(p)) {
        let name = file.file_name().expect("sidecar file name");
        let target = partial.join(name);
        let text_name = name.to_string_lossy();
        if text_name == "config.json" {
            let text = io(std::fs::read_to_string(file), file.display())?;
            io(
                std::fs::write(&target, ggml_config(&text)?),
                target.display(),
            )?;
        } else if text_name.ends_with(".index.json") {
            let text = io(std::fs::read_to_string(file), file.display())?;
            let mut index: Value =
                serde_json::from_str(&text).map_err(|e| msg(format!("{text_name}: {e}")))?;
            let mut keys: Vec<_> = weight_map.keys().cloned().collect();
            keys.sort();
            let sorted: serde_json::Map<String, Value> = keys
                .into_iter()
                .map(|k| {
                    let v = weight_map[&k].clone();
                    (k, v)
                })
                .collect();
            index["weight_map"] = Value::Object(sorted);
            index["metadata"] = serde_json::json!({ "total_size": report.tensor_bytes });
            let text = serde_json::to_string_pretty(&index)
                .map_err(|e| msg(format!("{text_name}: {e}")))?;
            io(std::fs::write(&target, text), target.display())?;
        } else {
            io(std::fs::copy(file, &target), file.display())?;
        }
    }
    if !partial.join("config.json").is_file() {
        return Err(msg(format!("{} has no config.json", src.display())));
    }
    io(std::fs::rename(&partial, dst), dst.display())?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::testkit::q4_pack;

    #[test]
    fn the_config_marker_becomes_ggml_storage_and_keeps_the_width() {
        let out =
            ggml_config(r#"{"hidden_size": 8, "quantization": {"bits": 4, "group_size": 64}}"#)
                .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            v["quantization"],
            serde_json::json!({"bits": 4, "storage": "ggml"})
        );
        assert_eq!(v["hidden_size"], 8);
        assert!(
            ggml_config(r#"{"hidden_size": 8}"#).is_err(),
            "a dense config must refuse"
        );
    }

    #[test]
    fn a_triple_becomes_the_exact_blocks_the_load_time_repack_builds() {
        let (wq, scales, biases, _grid) = q4_pack(8, 128, 3);
        let mut tensors = HashMap::new();
        tensors.insert("p.weight".to_string(), wq.clone());
        tensors.insert("p.scales".to_string(), scales.clone());
        tensors.insert("p.biases".to_string(), biases.clone());
        tensors.insert(
            "p.bias".to_string(),
            Tensor::zeros(8, DType::F32, &Device::Cpu).unwrap(),
        );
        let mut report = ComponentReport::default();
        let out = convert_tensors(tensors, &mut report).unwrap();
        assert_eq!(report.converted, 1);
        assert_eq!(report.passthrough, 1);
        assert!(!out.contains_key("p.scales") && !out.contains_key("p.biases"));
        let stored = &out["p.weight"];
        assert_eq!(stored.dtype(), DType::U8);
        assert_eq!(
            stored.dims(),
            [8, 4, 20],
            "Q4_1: 128 / 32 blocks of 20 B per row"
        );
        let expected =
            repack_packed_weight_uncached(&wq, &scales, &biases, MLX_GROUP_SIZE, &Device::Cpu)
                .unwrap();
        let stored_bytes = stored.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(stored_bytes, expected.data().unwrap().into_owned());
    }
}
