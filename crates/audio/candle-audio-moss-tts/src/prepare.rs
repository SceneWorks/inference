//! Audio-lane snapshot-preparation accommodation for MOSS-TTSD (sc-13360), following the Kokoro /
//! MOSS-SoundEffect / MOSS-TTS-Realtime pattern.
//!
//! A MOSS-TTSD snapshot is a single-model directory: `config.json` (the `MossTTSDForCausalLM`
//! architecture) + `model.safetensors` + the Qwen tokenizer. It carries no LLM-shaped top-level
//! layout the `candle-llm` HF probe recognizes, so the audio lane provides a probe + validated
//! **passthrough** preparer (the snapshot is already loadable; there is no quantized/converted
//! variant to materialize). A requested quantization is a typed `Unsupported`, never a silent dense
//! fallback.
//!
//! This preparer is wired into `candle-audio-catalog` (its `can_prepare` sits on the audio-lane
//! preparer chain), and the MOSS-TTSD provider is registered there: the XY_Tokenizer codec was
//! ported (sc-13518) and is staged as the passed-in `codec` component (sc-13662). The
//! probe/passthrough is additionally exercised by this crate's own unit tests.

use std::path::Path;

use candle_audio::prepare::{
    lists_architecture, passthrough_report, probe_json_config, safetensors_tensor_count,
};
use core_llm::{PrepareReport, PrepareSpec, Result as CoreResult};

/// The single-file checkpoint inside a snapshot.
pub const MODEL_WEIGHTS: &str = "model.safetensors";

/// Weightless probe: is `dir` a MOSS-TTSD snapshot (a `config.json` whose `architectures` names
/// `MossTTSDForCausalLM` + the safetensors checkpoint)? Reads only `config.json`.
pub fn is_moss_ttsd_snapshot(dir: &Path) -> bool {
    probe_json_config(dir, MODEL_WEIGHTS, "config.json", |v| {
        lists_architecture(v, "MossTTSDForCausalLM")
    })
}

/// [`is_moss_ttsd_snapshot`] over a [`PrepareSpec`].
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_moss_ttsd_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) a MOSS-TTSD snapshot, counting the checkpoint's tensors from its
/// safetensors header.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    passthrough_report(spec, "MOSS-TTSD", is_moss_ttsd_snapshot, |dir| {
        safetensors_tensor_count(&dir.join(MODEL_WEIGHTS))
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

    #[test]
    fn probe_rejects_non_moss_layouts() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
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
            r#"{"architectures": ["MossTTSDForCausalLM"]}"#,
        )
        .unwrap();
        assert!(can_prepare(&spec(&dir)));
    }

    #[test]
    fn prepare_refuses_quantization_typed() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        std::fs::write(dir.join(MODEL_WEIGHTS), tiny_safetensors(&["w"])).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures": ["MossTTSDForCausalLM"]}"#,
        )
        .unwrap();
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
    }
}
