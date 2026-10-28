//! Weightless snapshot inspection for [`LlamaProvider`](super::LlamaProvider)'s load gates
//! (sc-27302): the tensor names a safetensors snapshot holds — read from its index or each shard's
//! header, never from tensor data — and the NVFP4 refusals they and `config.json` decide (a
//! packed-affine or GGML-block snapshot, a family that cannot hold NVFP4 projections).

use super::*;

/// Why a snapshot's projections are already packed MLX-affine, or `None` (the packed half of
/// [`nvfp4_model_gate`], sc-24140 feature-end review). Packed means both: a `quantization` block
/// in `config.json` (top level or `text_config`, where [`ModelConfig`] reads it) **and** a
/// `<stem>.weight` whose `<stem>.scales` sidecar the snapshot names. A prepared Q4/Q8 snapshot
/// ([`crate::prepare`]) whose block says `storage: "ggml"` stores its projections as GGML blocks
/// (sc-19375) and is packed too; an older prepared snapshot carries the block over *dense*
/// weights, so it is not packed: NVFP4 quantizes its projections like any dense snapshot's. The
/// tensor names come from `model.safetensors.index.json` when it exists, else from each shard's
/// header — never from tensor data. An unreadable index or header is left to the loader's own
/// error.
pub(super) fn packed_affine_refusal(dir: &Path, config: &Value) -> Option<String> {
    let block = config.get("quantization").or_else(|| {
        config
            .get("text_config")
            .and_then(|text| text.get("quantization"))
    });
    let block = block?;
    if block.get("storage").and_then(Value::as_str) == Some(crate::prepare::GGML_STORAGE) {
        return Some(
            "this snapshot's projections are stored as GGML blocks (a prepared Q4 / Q8 tier); \
             NVFP4 projections are quantized from a dense snapshot"
                .into(),
        );
    }
    let names = snapshot_tensor_names(dir)?;
    let mut stems: Vec<&str> = names
        .iter()
        .filter_map(|name| {
            let stem = name.strip_suffix(".scales")?;
            names.contains(&format!("{stem}.weight")).then_some(stem)
        })
        .collect();
    // The first packed projection by name, so the refusal reads the same on every run.
    stems.sort_unstable();
    let stem = stems.first()?;
    Some(format!(
        "this snapshot's projections are packed MLX-affine (a `quantization` block over affine \
         sidecars such as `{stem}.scales`); NVFP4 projections are quantized from a dense snapshot"
    ))
}

/// The tensor names a safetensors snapshot holds: the index's `weight_map` keys, else every
/// `*.safetensors` shard's header keys (the 8-byte length prefix and the JSON header only).
/// `None` when neither can be read.
pub(super) fn snapshot_tensor_names(dir: &Path) -> Option<std::collections::HashSet<String>> {
    if let Some(index) = read_json(dir, "model.safetensors.index.json") {
        let map = index.get("weight_map")?.as_object()?;
        return Some(map.keys().cloned().collect());
    }
    let mut names = std::collections::HashSet::new();
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            names.extend(safetensors_header_names(&path)?);
        }
    }
    Some(names)
}

/// The tensor names in one safetensors file's header, without reading its data.
pub(super) fn safetensors_header_names(path: &Path) -> Option<Vec<String>> {
    Some(
        safetensors_header(path)?
            .into_iter()
            .map(|(key, _)| key)
            .filter(|key| key.as_str() != "__metadata__")
            .collect(),
    )
}

/// Test seam for [`packed_affine_refusal`]: write tiny tensors named `names` into `dir` — as a
/// `model.safetensors` shard, or (`index`) only a `model.safetensors.index.json` naming them.
#[cfg(test)]
pub(crate) fn write_test_snapshot_tensors(dir: &Path, names: &[&str], index: bool) {
    if index {
        let map: serde_json::Map<String, Value> = names
            .iter()
            .map(|name| {
                (
                    name.to_string(),
                    Value::from("model-00001-of-00001.safetensors"),
                )
            })
            .collect();
        let index = serde_json::json!({"metadata": {}, "weight_map": map});
        std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();
        return;
    }
    let tensors: std::collections::HashMap<String, Tensor> = names
        .iter()
        .map(|name| {
            let dtype = if name.ends_with(".weight") {
                DType::U32
            } else {
                DType::BF16
            };
            let tensor = Tensor::zeros((2, 2), dtype, &Device::Cpu).unwrap();
            (name.to_string(), tensor)
        })
        .collect();
    candle_core::safetensors::save(&tensors, dir.join("model.safetensors")).unwrap();
}

/// Why a snapshot's family cannot hold NVFP4 projections, or `None` when it can (the family half of
/// [`nvfp4_format_with`]). Prism/Bonsai snapshots are already packed affine-2; the qwen3_5 hybrid
/// and every llama-family (`CausalLm`) architecture quantize their dense projections through the
/// shared loader ([`Projection::load_as`](crate::primitives::Projection::load_as) /
/// [`load_eligible`](crate::primitives::Projection::load_eligible)).
pub(super) fn nvfp4_family_refusal(config: &Value) -> Option<String> {
    if config.get("model_type").and_then(Value::as_str) == Some("prism_hadamard_qwen35") {
        return Some(
            "a Prism/Bonsai snapshot is already packed affine-2; NVFP4 projections are quantized \
             from a dense snapshot"
                .into(),
        );
    }
    match Architecture::from_config(config) {
        // Every architecture the provider dispatches decodes through `Qwen35Model` or `CausalLm`,
        // and both load NVFP4 projections.
        Ok(
            Architecture::Qwen35
            | Architecture::Llama
            | Architecture::Qwen3
            | Architecture::Phi3
            | Architecture::Qwen2Moe
            | Architecture::Gemma2
            | Architecture::Glm4
            | Architecture::DeepseekV2
            | Architecture::Qwen3Vl
            | Architecture::Gemma4Unified
            | Architecture::Gemma4,
        )
        | Err(_) => None,
    }
}
