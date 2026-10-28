//! Audio-lane snapshot-preparation accommodation for MOSS-SoundEffect (sc-12841), following
//! the Kokoro pattern (sc-12836).
//!
//! A MOSS snapshot is a diffusers-style directory whose root carries `model_index.json` but no
//! top-level `config.json`/`tokenizer.json`, so `candle-llm`'s HF probe rejects it outright —
//! without an audio-aware path the lane could not prepare the snapshot at all. The
//! accommodation (composed by `candle-audio-catalog` into the lane's single `candle`
//! registration): recognize a MOSS-SoundEffect snapshot by its `model_index.json` pipeline
//! class + the DiT checkpoint, and prepare it as a validated **passthrough** — the snapshot is
//! already in its loadable form (safetensors + torch VAE pickle; there is no quantized/
//! converted variant to materialize), so preparation verifies and returns it. A requested
//! quantization is a typed `Unsupported`, never a silent dense fallback.

use std::path::Path;

use candle_audio::candle_core::pickle::read_pth_tensor_info;
use candle_audio::prepare::{
    component_tensor_count, json_str_is, passthrough_report, probe_json_config,
    safetensors_tensor_count,
};
use core_llm::{Error as CoreError, PrepareReport, PrepareSpec, Result as CoreResult};

/// Relative path of the DiT checkpoint inside a snapshot.
pub const DIT_WEIGHTS: &str = "transformer/diffusion_pytorch_model.safetensors";

/// Weightless probe: is `dir` a MOSS-SoundEffect snapshot (a `model_index.json` naming the
/// `MossSoundEffectPipeline` + the DiT safetensors)? Reads only `model_index.json`, never a
/// weight shard.
pub fn is_moss_sfx_snapshot(dir: &Path) -> bool {
    probe_json_config(dir, DIT_WEIGHTS, "model_index.json", |v| {
        json_str_is(v, "_class_name", "MossSoundEffectPipeline")
    })
}

/// [`is_moss_sfx_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane
/// registration consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_moss_sfx_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) a MOSS-SoundEffect snapshot. Counts the DiT and text-encoder
/// safetensors tensors from their headers and the VAE checkpoint's tensors from pickle
/// metadata (no storage reads) so the report is honest about what the snapshot holds.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    passthrough_report(spec, "MOSS-SoundEffect", is_moss_sfx_snapshot, |source| {
        let mut num_tensors = safetensors_tensor_count(&source.join(DIT_WEIGHTS))?;

        // Resolve the full text-encoder set before reading any shard headers.
        num_tensors += component_tensor_count(&source.join("text_encoder"), "model")?;

        // The VAE torch checkpoint (state_dict section, metadata only).
        let vae = source.join("vae").join(crate::vae::VAE_FILE);
        if !vae.is_file() {
            return Err(CoreError::Msg(format!(
                "prepare: {} is missing the VAE checkpoint {}",
                source.display(),
                crate::vae::VAE_FILE
            )));
        }
        let infos = read_pth_tensor_info(&vae, false, Some("state_dict"))
            .map_err(|e| CoreError::Msg(format!("prepare: {}: {e}", vae.display())))?;
        if infos.is_empty() {
            return Err(CoreError::Msg(format!(
                "prepare: {} carries no state_dict tensors",
                vae.display()
            )));
        }
        Ok(num_tensors + infos.len())
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

    /// A minimal valid safetensors byte blob with `n` zero-size F32 tensors.

    #[test]
    fn probe_rejects_non_moss_layouts() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        // Empty dir → no.
        assert!(!can_prepare(&spec(&dir)));
        // DiT weights but a foreign model_index → still no.
        std::fs::write(
            dir.join(DIT_WEIGHTS),
            tiny_safetensors(&["patch_embedding.weight"]),
        )
        .unwrap();
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "StableDiffusionPipeline"}"#,
        )
        .unwrap();
        assert!(!can_prepare(&spec(&dir)));
        // The MOSS pipeline class → yes (probe reads no weights).
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "MossSoundEffectPipeline"}"#,
        )
        .unwrap();
        assert!(can_prepare(&spec(&dir)));
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::write(dir.join(DIT_WEIGHTS), tiny_safetensors(&["w"])).unwrap();
        std::fs::write(
            dir.join("model_index.json"),
            r#"{"_class_name": "MossSoundEffectPipeline"}"#,
        )
        .unwrap();
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
    }
}
