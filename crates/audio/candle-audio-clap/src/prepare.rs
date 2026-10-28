//! Audio-lane snapshot-preparation accommodation for CLAP (sc-12851).
//!
//! A CLAP snapshot (`config.json` + `tokenizer.json` + `pytorch_model.bin`) is an HF-shaped
//! directory, so `candle-llm`'s `can_prepare` might accept it — but its `config.json` describes a
//! dual audio-text encoder (`model_type: "clap"`), not a causal LM. The accommodation (composed by
//! `candle-audio-catalog` into the lane's single `candle` registration, WITHOUT weakening the LLM
//! preparer): recognize a CLAP snapshot by its `model_type` and prepare it as a validated
//! **passthrough** — the checkpoint is already loadable dense; there is no quantized/converted
//! variant to materialize. A requested quantization is a typed `Unsupported`, never a silent dense
//! fallback.

use std::path::Path;

use candle_audio::prepare::{json_str_is, passthrough_report, probe_json_config};
use core_llm::{PrepareReport, PrepareSpec, Result as CoreResult};

use crate::model::{CONFIG_FILE, WEIGHTS_FILE};

/// Weightless probe: is `dir` a CLAP snapshot (a `config.json` whose `model_type` is `"clap"`
/// alongside `pytorch_model.bin`)? Reads only `config.json` metadata, never a weight shard.
pub fn is_clap_snapshot(dir: &Path) -> bool {
    probe_json_config(dir, WEIGHTS_FILE, CONFIG_FILE, |v| {
        json_str_is(v, "model_type", "clap")
    })
}

/// [`is_clap_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane registration
/// consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_clap_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) a CLAP snapshot. The weights are a pytorch pickle loaded via
/// `VarBuilder::from_pth`; the report carries no tensor count.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    passthrough_report(spec, "CLAP", is_clap_snapshot, |_| Ok(0))
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
    fn probe_rejects_non_clap_layouts() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "qwen2"}"#).unwrap();
        std::fs::write(dir.join(WEIGHTS_FILE), b"stub").unwrap();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "clap"}"#).unwrap();
        assert!(can_prepare(&spec(&dir)));
    }

    #[test]
    fn prepare_passthrough_and_refuses_quantization_typed() {
        let dir_tmp = tempfile::tempdir().unwrap();
        let dir = dir_tmp.path().to_path_buf();
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "clap"}"#).unwrap();
        std::fs::write(dir.join(WEIGHTS_FILE), b"stub").unwrap();
        let report = prepare(&spec(&dir)).unwrap();
        assert!(report.passthrough);
        assert_eq!(report.out_dir, dir);
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
    }
}
