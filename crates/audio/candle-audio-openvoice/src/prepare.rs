//! Audio-lane snapshot-preparation accommodation for OpenVoice V2 (sc-13223).
//!
//! Like Kokoro (sc-12836), an OpenVoice converter snapshot has a `config.json` but no
//! `tokenizer.json`, and pickle (`checkpoint.pth`) rather than safetensors — so `candle-llm`'s
//! `can_prepare` would accept it (config.json present) and `prepare` would then fail on the missing
//! tokenizer, a lying probe. The accommodation (composed by `candle-audio-catalog` into the lane's
//! single `candle` registration, WITHOUT weakening the LLM preparer): recognize an OpenVoice
//! converter snapshot by its VITS `config.json` block + `checkpoint.pth`, and prepare it as a
//! validated **passthrough** — the snapshot is already loadable (dense pickle; there is no
//! quantized/converted variant to materialize). A requested quantization is a typed `Unsupported`,
//! never a silent dense fallback.

use std::path::Path;

use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

use crate::pipeline::{CHECKPOINT_FILE, CONFIG_FILE};

/// Weightless probe: is `dir` an OpenVoice V2 converter snapshot (a `config.json` carrying the VITS
/// `data.filter_length` + `model.gin_channels` blocks alongside `checkpoint.pth`)? Reads only
/// `config.json` metadata, never a weight shard.
pub fn is_openvoice_snapshot(dir: &Path) -> bool {
    if !dir.is_dir() || !dir.join(CHECKPOINT_FILE).is_file() {
        return false;
    }
    let Ok(text) = std::fs::read_to_string(dir.join(CONFIG_FILE)) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .map(|v| {
            v.get("data").and_then(|d| d.get("filter_length")).is_some()
                && v.get("model").and_then(|m| m.get("gin_channels")).is_some()
        })
        .unwrap_or(false)
}

/// [`is_openvoice_snapshot`] over a [`PrepareSpec`] — the probe the composed audio-lane registration
/// consults before delegating to the LLM preparer.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    is_openvoice_snapshot(&spec.source)
}

/// Prepare (verify + passthrough) an OpenVoice V2 converter snapshot.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !is_openvoice_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not an OpenVoice V2 converter snapshot",
            spec.source.display()
        )));
    }
    if let Some(q) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: OpenVoice V2 converter snapshots have no {q:?} form — the checkpoint ships \
             dense-only"
        )));
    }
    Ok(PrepareReport {
        // core-llm's detect_format calls a config.json-bearing snapshot dir Safetensors (the
        // HF-snapshot shape); the pickle checkpoint rides inside that shape.
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
    fn probe_rejects_non_openvoice_layouts() {
        let dir = std::env::temp_dir().join("openvoice-prepare-probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Empty dir → no.
        assert!(!can_prepare(&spec(&dir)));
        // LLM-shaped config.json (no VITS blocks) + a pth → still no.
        std::fs::write(dir.join(CONFIG_FILE), r#"{"hidden_size": 8}"#).unwrap();
        std::fs::write(dir.join(CHECKPOINT_FILE), b"not a checkpoint").unwrap();
        assert!(!can_prepare(&spec(&dir)));
        // OpenVoice-shaped config.json → yes (probe reads no weights).
        std::fs::write(
            dir.join(CONFIG_FILE),
            r#"{"data":{"filter_length":1024},"model":{"gin_channels":256}}"#,
        )
        .unwrap();
        assert!(can_prepare(&spec(&dir)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn prepare_passthrough_and_refuses_quantization_typed() {
        let dir = std::env::temp_dir().join("openvoice-prepare-quant");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(CONFIG_FILE),
            r#"{"data":{"filter_length":1024},"model":{"gin_channels":256}}"#,
        )
        .unwrap();
        std::fs::write(dir.join(CHECKPOINT_FILE), b"stub").unwrap();
        let report = prepare(&spec(&dir)).unwrap();
        assert!(report.passthrough);
        assert_eq!(report.out_dir, dir);
        let mut s = spec(&dir);
        s.quantize = Some(core_llm::Quantize::Q4);
        assert!(matches!(prepare(&s), Err(CoreError::Unsupported(_))));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
