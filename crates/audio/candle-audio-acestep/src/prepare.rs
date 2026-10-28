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

use candle_audio::prepare::optional_component_tensor_count;
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
    for (component, stem) in [
        ("transformer", "diffusion_pytorch_model"),
        ("condition_encoder", "diffusion_pytorch_model"),
        ("text_encoder", "model"),
        ("vae", "diffusion_pytorch_model"),
    ] {
        num_tensors += optional_component_tensor_count(&spec.source.join(component), stem)
            .map_err(|e| CoreError::Msg(format!("prepare: {e}")))?;
    }
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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
    }
}
