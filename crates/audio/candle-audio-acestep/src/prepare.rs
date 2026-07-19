//! Audio-lane snapshot-preparation accommodation for ACE-Step 1.5 (sc-12842), following the
//! Kokoro / MOSS-SFX pattern.
//!
//! An ACE-Step snapshot is a diffusers-style directory whose root carries `model_index.json` but
//! no top-level `config.json`/`tokenizer.json`, so `candle-llm`'s HF probe rejects it. The
//! accommodation (composed by `candle-audio-catalog` into the lane's single `candle` preparer
//! registration): recognize an ACE-Step snapshot by its `model_index.json` pipeline class + the
//! sharded DiT index, and prepare it as a validated **passthrough** (the snapshot is already in its
//! loadable safetensors form). A requested quantization is a typed `Unsupported`.

use std::path::Path;

use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

/// Relative path of the DiT shard index inside a snapshot.
pub const DIT_INDEX: &str = "transformer/diffusion_pytorch_model.safetensors.index.json";

/// Weightless probe: is `dir` an ACE-Step snapshot (a `model_index.json` naming the
/// `AceStepPipeline` + the DiT shard index)? Reads only JSON, never a weight shard.
pub fn is_acestep_snapshot(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join(DIT_INDEX).is_file() {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(dir.join("model_index.json")) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| {
            v.get("_class_name")
                .and_then(|c| c.as_str())
                .map(|c| c == "AceStepPipeline")
        })
        .unwrap_or(false)
}

/// [`is_acestep_snapshot`] over a [`PrepareSpec`].
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_acestep_snapshot(&spec.source)
}

/// Count the tensors of one safetensors file from its 8-byte header length + JSON header only.
fn safetensors_tensor_count(path: &Path) -> CoreResult<usize> {
    let bytes = std::fs::read(path)
        .map_err(|e| CoreError::Msg(format!("prepare: read {}: {e}", path.display())))?;
    if bytes.len() < 8 {
        return Err(CoreError::Msg(format!(
            "prepare: {} is not a safetensors file (short header)",
            path.display()
        )));
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = bytes.get(8..8 + n).ok_or_else(|| {
        CoreError::Msg(format!(
            "prepare: {} safetensors header truncated",
            path.display()
        ))
    })?;
    let v: serde_json::Value = serde_json::from_slice(header)
        .map_err(|e| CoreError::Msg(format!("prepare: {} header: {e}", path.display())))?;
    let obj = v.as_object().ok_or_else(|| {
        CoreError::Msg(format!(
            "prepare: {} header is not an object",
            path.display()
        ))
    })?;
    Ok(obj.keys().filter(|k| *k != "__metadata__").count())
}

fn count_component(dir: &Path, stem: &str, total: &mut usize) -> CoreResult<()> {
    let index = dir.join(format!("{stem}.safetensors.index.json"));
    if index.is_file() {
        let text = std::fs::read_to_string(&index)
            .map_err(|e| CoreError::Msg(format!("prepare: read {}: {e}", index.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| CoreError::Msg(format!("prepare: {}: {e}", index.display())))?;
        let mut shards: Vec<String> = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|m| {
                m.values()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        shards.sort();
        shards.dedup();
        for shard in shards {
            *total += safetensors_tensor_count(&dir.join(shard))?;
        }
    } else {
        let single = dir.join(format!("{stem}.safetensors"));
        if single.is_file() {
            *total += safetensors_tensor_count(&single)?;
        }
    }
    Ok(())
}

/// Prepare (verify + passthrough) an ACE-Step snapshot: counts the DiT, condition-encoder,
/// text-encoder, and VAE tensors from their safetensors headers (no storage reads).
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !is_acestep_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not an ACE-Step audio snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: ACE-Step snapshots have no {q:?} form — the pinned checkpoint ships dense-only"
        )));
    }
    let mut num_tensors = 0usize;
    count_component(
        &spec.source.join("transformer"),
        "diffusion_pytorch_model",
        &mut num_tensors,
    )?;
    count_component(
        &spec.source.join("condition_encoder"),
        "diffusion_pytorch_model",
        &mut num_tensors,
    )?;
    count_component(&spec.source.join("text_encoder"), "model", &mut num_tensors)?;
    count_component(
        &spec.source.join("vae"),
        "diffusion_pytorch_model",
        &mut num_tensors,
    )?;
    if num_tensors == 0 {
        return Err(CoreError::Msg(format!(
            "prepare: {} carries no component safetensors tensors",
            spec.source.display()
        )));
    }

    Ok(PrepareReport {
        input_format: ModelFormat::Safetensors,
        quantized: None,
        out_dir: spec.source.clone(),
        num_tensors,
        passthrough: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_safetensors(names: &[&str]) -> Vec<u8> {
        let entries: Vec<String> = names
            .iter()
            .map(|n| format!(r#""{n}": {{"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}"#))
            .collect();
        let header = format!("{{{}}}", entries.join(", "));
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(header.as_bytes());
        out
    }

    fn spec(dir: &Path) -> PrepareSpec {
        PrepareSpec {
            source: dir.to_path_buf(),
            out_dir: dir.join("out"),
            quantize: None,
        }
    }

    #[test]
    fn probe_recognizes_acestep_and_rejects_others() {
        let dir = std::env::temp_dir().join("acestep-prepare-probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(
            dir.join(DIT_INDEX),
            r#"{"weight_map": {"a": "diffusion_pytorch_model-00001-of-00002.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "StableDiffusionPipeline"}"#,
        )
        .unwrap();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "AceStepPipeline"}"#,
        )
        .unwrap();
        assert!(can_prepare(&spec(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir = std::env::temp_dir().join("acestep-prepare-quant");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::write(
            dir.join(DIT_INDEX),
            r#"{"weight_map": {"a": "diffusion_pytorch_model.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("transformer/diffusion_pytorch_model.safetensors"),
            tiny_safetensors(&["w"]),
        )
        .unwrap();
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "AceStepPipeline"}"#,
        )
        .unwrap();
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
