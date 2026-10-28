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

use candle_audio::prepare::{
    json_str_is, optional_component_tensor_count, passthrough_report, probe_json_config,
};
use core_llm::{Error as CoreError, PrepareReport, PrepareSpec, Result as CoreResult};

/// Relative path of the DiT shard index inside a snapshot.
pub const DIT_INDEX: &str = "transformer/diffusion_pytorch_model.safetensors.index.json";

/// Weightless probe: is `dir` an ACE-Step snapshot (a `model_index.json` naming the
/// `AceStepPipeline` + the DiT shard index)? Reads only JSON, never a weight shard.
pub fn is_acestep_snapshot(dir: &Path) -> bool {
    probe_json_config(dir, DIT_INDEX, "model_index.json", |v| {
        json_str_is(v, "_class_name", "AceStepPipeline")
    })
}

/// [`is_acestep_snapshot`] over a [`PrepareSpec`].
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_acestep_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) an ACE-Step snapshot: counts the DiT, condition-encoder,
/// text-encoder, and VAE tensors from their safetensors headers (no storage reads).
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    passthrough_report(spec, "ACE-Step", is_acestep_snapshot, |source| {
        let mut num_tensors = 0usize;
        for (component, stem) in [
            ("transformer", "diffusion_pytorch_model"),
            ("condition_encoder", "diffusion_pytorch_model"),
            ("text_encoder", "model"),
            ("vae", "diffusion_pytorch_model"),
        ] {
            num_tensors += optional_component_tensor_count(&source.join(component), stem)?;
        }
        if num_tensors == 0 {
            return Err(CoreError::Msg(format!(
                "prepare: {} carries no component safetensors tensors",
                source.display()
            )));
        }
        Ok(num_tensors)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::testkit::tiny_safetensors;

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
