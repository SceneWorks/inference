//! Audio-lane snapshot-preparation accommodation for MOSS-TTS-Realtime (sc-13334), following the
//! Kokoro / MOSS-SoundEffect pattern.
//!
//! A MOSS-TTS-Realtime snapshot is a single-model directory: `config.json` (the `MossTTSRealtime`
//! architecture) + `model.safetensors` + the Qwen tokenizer. It carries no separate LLM-shaped
//! top-level layout the `candle-llm` HF probe recognizes, so the audio lane provides a probe +
//! validated **passthrough** preparer (the snapshot is already loadable; there is no quantized or
//! converted variant to materialize). A requested quantization is a typed `Unsupported`, never a
//! silent dense fallback. It is wired into `candle-audio-catalog`'s audio-lane preparer chain
//! alongside the provider registration (sc-13392).

use std::path::Path;

use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

/// The single-file checkpoint inside a snapshot.
pub const MODEL_WEIGHTS: &str = "model.safetensors";

/// Weightless probe: is `dir` a MOSS-TTS-Realtime snapshot (a `config.json` whose `architectures`
/// names `MossTTSRealtime` + the safetensors checkpoint)? Reads only `config.json`.
pub fn is_moss_tts_realtime_snapshot(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join(MODEL_WEIGHTS).is_file() {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| {
            v.get("architectures")
                .and_then(|a| a.as_array())
                .map(|a| a.iter().any(|s| s.as_str() == Some("MossTTSRealtime")))
        })
        .unwrap_or(false)
}

/// [`is_moss_tts_realtime_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane
/// registration consults in the audio-lane preparer chain.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_moss_tts_realtime_snapshot(&spec.source)
}

/// Count the tensors of one safetensors file from its header only (8-byte length + JSON;
/// `__metadata__` is not a tensor). No tensor storage is read.
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

/// Prepare (verify + passthrough) a MOSS-TTS-Realtime snapshot.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !is_moss_tts_realtime_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not a MOSS-TTS-Realtime audio snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: MOSS-TTS-Realtime snapshots have no {q:?} form — the pinned checkpoint ships \
             dense-only"
        )));
    }
    let num_tensors = safetensors_tensor_count(&spec.source.join(MODEL_WEIGHTS))?;
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

    fn spec(dir: &Path) -> PrepareSpec {
        PrepareSpec {
            source: dir.to_path_buf(),
            out_dir: dir.join("out"),
            quantize: None,
        }
    }

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
        let dir = std::env::temp_dir().join("moss-tts-rt-prepare-probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(dir.join(MODEL_WEIGHTS), tiny_safetensors(&["w"])).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures": ["Qwen3ForCausalLM"]}"#,
        )
        .unwrap();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures": ["MossTTSRealtime"]}"#,
        )
        .unwrap();
        assert!(can_prepare(&spec(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir = std::env::temp_dir().join("moss-tts-rt-prepare-quant");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(MODEL_WEIGHTS), tiny_safetensors(&["w"])).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures": ["MossTTSRealtime"]}"#,
        )
        .unwrap();
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
