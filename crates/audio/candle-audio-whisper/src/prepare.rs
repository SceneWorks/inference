//! Audio-lane snapshot-preparation accommodation for Whisper (sc-12850).
//!
//! A Whisper snapshot (`config.json` + `tokenizer.json` + `model.safetensors`) looks like an HF LLM
//! snapshot, so `candle-llm`'s `can_prepare` would accept it — but its `config.json` describes an
//! encoder-decoder ASR architecture (`model_type: "whisper"`), not a causal LM, so the LLM preparer
//! is the wrong owner. The accommodation (composed by `candle-audio-catalog` into the lane's single
//! `candle` registration, WITHOUT weakening the LLM preparer): recognize a Whisper snapshot by its
//! `model_type` and prepare it as a validated **passthrough** — the safetensors checkpoint is
//! already loadable dense; there is no quantized/converted variant to materialize. A requested
//! quantization is a typed `Unsupported`, never a silent dense fallback.

use std::path::Path;

use candle_audio::prepare::{json_str_is, passthrough_report, probe_json_config};
use core_llm::{PrepareReport, PrepareSpec, Result as CoreResult};

use crate::model::{CONFIG_FILE, WEIGHTS_FILE};

/// Weightless probe: is `dir` a Whisper snapshot (a `config.json` whose `model_type` is `"whisper"`
/// alongside `model.safetensors`)? Reads only `config.json` metadata, never a weight shard.
pub fn is_whisper_snapshot(dir: &Path) -> bool {
    probe_json_config(dir, WEIGHTS_FILE, CONFIG_FILE, |v| {
        json_str_is(v, "model_type", "whisper")
    })
}

/// [`is_whisper_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane registration
/// consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_whisper_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) a Whisper snapshot; the report carries no tensor count.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    passthrough_report(spec, "Whisper", is_whisper_snapshot, |_| Ok(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::Error as CoreError;

    fn spec(dir: &Path) -> PrepareSpec {
        PrepareSpec {
            source: dir.to_path_buf(),
            out_dir: dir.join("out"),
            quantize: None,
        }
    }

    #[test]
    fn probe_rejects_non_whisper_layouts() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        // Empty dir → no.
        assert!(!can_prepare(&spec(&dir)));
        // An LLM-shaped config.json (no whisper model_type) + safetensors → still no.
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "qwen2"}"#).unwrap();
        std::fs::write(dir.join(WEIGHTS_FILE), b"stub").unwrap();
        assert!(!can_prepare(&spec(&dir)));
        // A whisper config.json → yes (probe reads no weights).
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "whisper"}"#).unwrap();
        assert!(can_prepare(&spec(&dir)));
    }

    #[test]
    fn prepare_passthrough_and_refuses_quantization_typed() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "whisper"}"#).unwrap();
        std::fs::write(dir.join(WEIGHTS_FILE), b"stub").unwrap();
        let report = prepare(&spec(&dir)).unwrap();
        assert!(report.passthrough);
        assert_eq!(report.out_dir, dir);
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
    }
}
