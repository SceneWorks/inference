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

use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

use crate::model::{CONFIG_FILE, WEIGHTS_FILE};

/// Weightless probe: is `dir` a CLAP snapshot (a `config.json` whose `model_type` is `"clap"`
/// alongside `pytorch_model.bin`)? Reads only `config.json` metadata, never a weight shard.
pub fn is_clap_snapshot(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join(WEIGHTS_FILE).is_file() {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(dir.join(CONFIG_FILE)) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| {
            v.get("model_type")
                .and_then(|m| m.as_str())
                .map(|s| s == "clap")
        })
        .unwrap_or(false)
}

/// [`is_clap_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane registration
/// consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_clap_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) a CLAP snapshot.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !is_clap_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not a CLAP snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: CLAP snapshots have no {q:?} form — the checkpoint ships dense-only"
        )));
    }
    Ok(PrepareReport {
        // The snapshot is an HF-shaped directory (the closest `ModelFormat` tag); the weights are a
        // pytorch pickle loaded via `VarBuilder::from_pth`, but `prepare` is a pure passthrough.
        input_format: ModelFormat::Safetensors,
        quantized: None,
        out_dir: spec.source.clone(),
        num_tensors: 0,
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

    #[test]
    fn probe_rejects_non_clap_layouts() {
        let dir = std::env::temp_dir().join("clap-prepare-probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "qwen2"}"#).unwrap();
        std::fs::write(dir.join(WEIGHTS_FILE), b"stub").unwrap();
        assert!(!can_prepare(&spec(&dir)));
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "clap"}"#).unwrap();
        assert!(can_prepare(&spec(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_passthrough_and_refuses_quantization_typed() {
        let dir = std::env::temp_dir().join("clap-prepare-quant");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(CONFIG_FILE), r#"{"model_type": "clap"}"#).unwrap();
        std::fs::write(dir.join(WEIGHTS_FILE), b"stub").unwrap();
        let report = prepare(&spec(&dir)).unwrap();
        assert!(report.passthrough);
        assert_eq!(report.out_dir, dir);
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
