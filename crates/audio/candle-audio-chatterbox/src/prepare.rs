//! Audio-lane snapshot-preparation accommodation for Chatterbox (sc-13222), following the
//! Kokoro/MOSS pattern (sc-12836/sc-12841).
//!
//! A Chatterbox snapshot is a flat directory holding `t3_cfg.safetensors`, `s3gen.safetensors`, and
//! `tokenizer.json` but no top-level `config.json` (the hyperparameters are hard-coded in the
//! reference / mirrored in [`crate::config`]), so `candle-llm`'s HF probe would demand a
//! tokenizer/config layout it does not have. The accommodation (composed by `candle-audio-catalog`
//! into the lane's single `candle` registration): recognize a Chatterbox snapshot by its
//! checkpoints, and prepare it as a validated **passthrough** — the safetensors are already
//! loadable; preparation verifies and returns them. A requested quantization is a typed
//! `Unsupported`, never a silent dense fallback.

use std::path::Path;

use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

use crate::model::{T3_WEIGHTS_FILE, TOKENIZER_FILE};
use crate::s3gen::S3GEN_WEIGHTS_FILE;

/// Weightless probe: is `dir` a Chatterbox snapshot (the T3 + S3Gen checkpoints + the tokenizer)?
/// Reads no weight bytes.
pub fn is_chatterbox_snapshot(dir: &Path) -> bool {
    dir.is_dir()
        && dir.join(T3_WEIGHTS_FILE).is_file()
        && dir.join(S3GEN_WEIGHTS_FILE).is_file()
        && dir.join(TOKENIZER_FILE).is_file()
}

/// [`is_chatterbox_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane
/// registration consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_chatterbox_snapshot(&spec.source)
}

/// Count the tensors of one safetensors file from its header only (8-byte little-endian length +
/// JSON; `__metadata__` is not a tensor). No tensor storage is read.
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

/// Prepare (verify + passthrough) a Chatterbox snapshot. Counts the T3 and S3Gen safetensors
/// tensors from their headers so the report is honest about what the snapshot holds.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !is_chatterbox_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not a Chatterbox audio snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: Chatterbox snapshots have no {q:?} form — the pinned checkpoints ship \
             dense-only"
        )));
    }
    let num_tensors = safetensors_tensor_count(&spec.source.join(T3_WEIGHTS_FILE))?
        + safetensors_tensor_count(&spec.source.join(S3GEN_WEIGHTS_FILE))?;

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

    fn make_snapshot(dir: &Path) {
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join(T3_WEIGHTS_FILE),
            tiny_safetensors(&["text_emb.weight"]),
        )
        .unwrap();
        std::fs::write(
            dir.join(S3GEN_WEIGHTS_FILE),
            tiny_safetensors(&["flow.w", "mel2wav.w"]),
        )
        .unwrap();
        std::fs::write(dir.join(TOKENIZER_FILE), r#"{"model":{"type":"BPE"}}"#).unwrap();
    }

    #[test]
    fn probe_recognizes_a_chatterbox_snapshot_only() {
        let dir = std::env::temp_dir().join("chatterbox-prepare-probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!can_prepare(&spec(&dir))); // empty
        make_snapshot(&dir);
        assert!(can_prepare(&spec(&dir)));
        // Missing the S3Gen checkpoint → not recognized.
        std::fs::remove_file(dir.join(S3GEN_WEIGHTS_FILE)).unwrap();
        assert!(!can_prepare(&spec(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_passthrough_counts_tensors() {
        let dir = std::env::temp_dir().join("chatterbox-prepare-pass");
        make_snapshot(&dir);
        let report = prepare(&spec(&dir)).unwrap();
        assert!(report.passthrough);
        assert_eq!(report.num_tensors, 3); // 1 (T3) + 2 (S3Gen)
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir = std::env::temp_dir().join("chatterbox-prepare-quant");
        make_snapshot(&dir);
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
