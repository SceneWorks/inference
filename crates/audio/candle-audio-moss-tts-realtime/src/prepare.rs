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

use candle_audio::prepare::{
    lists_architecture, passthrough_report, probe_json_config, safetensors_tensor_count,
};
use core_llm::{PrepareReport, PrepareSpec, Result as CoreResult};

/// The single-file checkpoint inside a snapshot.
pub const MODEL_WEIGHTS: &str = "model.safetensors";

/// Weightless probe: is `dir` a MOSS-TTS-Realtime snapshot (a `config.json` whose `architectures`
/// names `MossTTSRealtime` + the safetensors checkpoint)? Reads only `config.json`.
pub fn is_moss_tts_realtime_snapshot(dir: &Path) -> bool {
    probe_json_config(dir, MODEL_WEIGHTS, "config.json", |v| {
        lists_architecture(v, "MossTTSRealtime")
    })
}

/// [`is_moss_tts_realtime_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane
/// registration consults in the audio-lane preparer chain.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_moss_tts_realtime_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) a MOSS-TTS-Realtime snapshot, counting the checkpoint's tensors
/// from its safetensors header.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    passthrough_report(
        spec,
        "MOSS-TTS-Realtime",
        is_moss_tts_realtime_snapshot,
        |dir| safetensors_tensor_count(&dir.join(MODEL_WEIGHTS)),
    )
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
            r#"{"architectures": ["MossTTSRealtime"]}"#,
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
            r#"{"architectures": ["MossTTSRealtime"]}"#,
        )
        .unwrap();
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
    }
}
