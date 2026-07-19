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
use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

/// Relative path of the DiT checkpoint inside a snapshot.
pub const DIT_WEIGHTS: &str = "transformer/diffusion_pytorch_model.safetensors";

/// Weightless probe: is `dir` a MOSS-SoundEffect snapshot (a `model_index.json` naming the
/// `MossSoundEffectPipeline` + the DiT safetensors)? Reads only `model_index.json`, never a
/// weight shard.
pub fn is_moss_sfx_snapshot(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join(DIT_WEIGHTS).is_file() {
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
                .map(|c| c == "MossSoundEffectPipeline")
        })
        .unwrap_or(false)
}

/// [`is_moss_sfx_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane
/// registration consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_moss_sfx_snapshot(&spec.source)
}

/// Count the tensors of one safetensors file from its header only (8-byte little-endian length
/// + JSON; the `__metadata__` entry is not a tensor). No tensor storage is read.
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
            "prepare: {} header is not a JSON object",
            path.display()
        ))
    })?;
    Ok(obj.keys().filter(|k| *k != "__metadata__").count())
}

/// Prepare (verify + passthrough) a MOSS-SoundEffect snapshot. Counts the DiT and text-encoder
/// safetensors tensors from their headers and the VAE checkpoint's tensors from pickle
/// metadata (no storage reads) so the report is honest about what the snapshot holds.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !is_moss_sfx_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not a MOSS-SoundEffect audio snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: MOSS-SoundEffect snapshots have no {q:?} form — the pinned checkpoint \
             ships dense-only"
        )));
    }
    let mut num_tensors = safetensors_tensor_count(&spec.source.join(DIT_WEIGHTS))?;

    // Text-encoder shards (whatever the index lists).
    let te_dir = spec.source.join("text_encoder");
    let index_path = te_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let text = std::fs::read_to_string(&index_path)
            .map_err(|e| CoreError::Msg(format!("prepare: read {}: {e}", index_path.display())))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| CoreError::Msg(format!("prepare: {}: {e}", index_path.display())))?;
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
            num_tensors += safetensors_tensor_count(&te_dir.join(shard))?;
        }
    }

    // The VAE torch checkpoint (state_dict section, metadata only).
    let vae = spec.source.join("vae").join(crate::vae::VAE_FILE);
    if vae.is_file() {
        let infos = read_pth_tensor_info(&vae, false, Some("state_dict"))
            .map_err(|e| CoreError::Msg(format!("prepare: {}: {e}", vae.display())))?;
        if infos.is_empty() {
            return Err(CoreError::Msg(format!(
                "prepare: {} carries no state_dict tensors",
                vae.display()
            )));
        }
        num_tensors += infos.len();
    } else {
        return Err(CoreError::Msg(format!(
            "prepare: {} is missing the VAE checkpoint {}",
            spec.source.display(),
            crate::vae::VAE_FILE
        )));
    }

    Ok(PrepareReport {
        // core-llm's own detect_format calls a safetensors-bearing snapshot dir Safetensors;
        // the VAE pickle rides inside that shape (same convention as the Kokoro report).
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

    fn spec(dir: &Path) -> PrepareSpec {
        PrepareSpec {
            source: dir.to_path_buf(),
            out_dir: dir.join("out"),
            quantize: None,
        }
    }

    /// A minimal valid safetensors byte blob with `n` zero-size F32 tensors.
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

    #[test]
    fn probe_rejects_non_moss_layouts() {
        let dir = std::env::temp_dir().join("moss-sfx-prepare-probe");
        let _ = std::fs::remove_dir_all(&dir);
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir = std::env::temp_dir().join("moss-sfx-prepare-quant");
        let _ = std::fs::remove_dir_all(&dir);
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safetensors_header_count_is_metadata_free() {
        let dir = std::env::temp_dir().join("moss-sfx-prepare-count");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.safetensors");
        std::fs::write(&p, tiny_safetensors(&["a", "b", "__metadata__"])).unwrap();
        assert_eq!(safetensors_tensor_count(&p).unwrap(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
