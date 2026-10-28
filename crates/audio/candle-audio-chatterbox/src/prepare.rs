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

use candle_audio::prepare::{passthrough_report, safetensors_tensor_count};
use core_llm::{PrepareReport, PrepareSpec, Result as CoreResult};

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

/// Prepare (verify + passthrough) a Chatterbox snapshot. Counts the T3 and S3Gen safetensors
/// tensors from their headers so the report is honest about what the snapshot holds.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    passthrough_report(spec, "Chatterbox", is_chatterbox_snapshot, |dir| {
        Ok(safetensors_tensor_count(&dir.join(T3_WEIGHTS_FILE))?
            + safetensors_tensor_count(&dir.join(S3GEN_WEIGHTS_FILE))?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::testkit::tiny_safetensors;
    use core_llm::Error as CoreError;

    fn spec(dir: &Path) -> PrepareSpec {
        PrepareSpec {
            source: dir.to_path_buf(),
            out_dir: dir.join("out"),
            quantize: None,
        }
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
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        assert!(!can_prepare(&spec(&dir))); // empty
        make_snapshot(&dir);
        assert!(can_prepare(&spec(&dir)));
        // Missing the S3Gen checkpoint → not recognized.
        std::fs::remove_file(dir.join(S3GEN_WEIGHTS_FILE)).unwrap();
        assert!(!can_prepare(&spec(&dir)));
    }

    #[test]
    fn prepare_passthrough_counts_tensors() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        make_snapshot(&dir);
        let report = prepare(&spec(&dir)).unwrap();
        assert!(report.passthrough);
        assert_eq!(report.num_tensors, 3); // 1 (T3) + 2 (S3Gen)
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        make_snapshot(&dir);
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
    }
}
