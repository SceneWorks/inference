//! The audio-lane snapshot preparer for YuE2 (sc-22995): how an application derives a `q8` / `q4`
//! tier snapshot ([`crate::tier`]) from the YuE2-3B original it acquired, through the same
//! `core_llm` preparer seam every audio snapshot uses (composed into `candle-audio-catalog`'s
//! audio-lane preparer).
//!
//! * `quantize: None` — the `bf16` tier is the released checkpoint itself: the source is verified
//!   against its pins and returned as is (`passthrough`).
//! * `quantize: Q8 / Q4` — [`crate::tier::convert`] writes the verified, deterministic tier
//!   snapshot to `out_dir` (local; nothing is uploaded — rehosting stays gated, see
//!   [`crate::license`]).
//! * `quantize: Nvfp4` — refused: not a YuE2 tier.
//!
//! The probe ([`can_prepare`]) reads only `config.json` (`model_type: yue2`), never a weight.

use std::path::Path;

use candle_llm::core_llm::{
    Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Quantize, Result as CoreResult,
};

use crate::inventory::ComponentId;
use crate::precision::Tier;
use crate::snapshot::SnapshotDirs;

/// Whether `dir` is a YuE2-3B snapshot — the original or a derived tier (a `config.json` with
/// `model_type: yue2`). Reads only `config.json`.
pub fn is_yue2_snapshot(dir: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(dir.join("config.json")) else {
        return false;
    };
    serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("model_type").and_then(|m| m.as_str()).map(|m| m == "yue2"))
        .unwrap_or(false)
}

/// [`is_yue2_snapshot`] over a [`PrepareSpec`].
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    spec.source.is_dir() && is_yue2_snapshot(&spec.source)
}

fn tier_of(quantize: Option<Quantize>) -> CoreResult<Tier> {
    match quantize {
        None => Ok(Tier::Bf16),
        Some(Quantize::Q8) => Ok(Tier::Q8),
        Some(Quantize::Q4) => Ok(Tier::Q4),
        Some(other) => Err(CoreError::Unsupported(format!(
            "prepare: {other:?} is not a YuE2 tier; YuE2 has bf16 (the released checkpoint), q8 \
             and q4"
        ))),
    }
}

/// Prepare a YuE2 tier (see the [module docs](self)).
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if !can_prepare(spec) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is not a YuE2-3B snapshot",
            spec.source.display()
        )));
    }
    let tier = tier_of(spec.quantize)?;
    if crate::tier::is_tier_snapshot(&spec.source) {
        return Err(CoreError::Unsupported(format!(
            "prepare: {} is already a derived tier; tiers are derived from the released BF16 \
             original only (a quantized weight is never re-quantized)",
            spec.source.display()
        )));
    }
    let lm = ComponentId::Lm.component();
    let dirs = SnapshotDirs::new().with(lm.repo.id, spec.source.clone());
    let num_tensors = lm
        .conversion_manifest()
        .map_err(|e| CoreError::Msg(format!("prepare: {e}")))?
        .tensors
        .len();
    if tier == Tier::Bf16 {
        crate::snapshot::resolve_component(ComponentId::Lm, &dirs)
            .and_then(|_| crate::snapshot::resolve_component(ComponentId::QwenTiktoken, &dirs))
            .map_err(|e| CoreError::Msg(format!("prepare: {e}")))?;
        return Ok(PrepareReport {
            input_format: ModelFormat::Safetensors,
            quantized: None,
            out_dir: spec.source.clone(),
            num_tensors,
            passthrough: true,
        });
    }
    crate::tier::convert(&dirs, tier, &spec.out_dir)
        .map_err(|e| CoreError::Msg(format!("prepare: {e}")))?;
    Ok(PrepareReport {
        input_format: ModelFormat::Safetensors,
        quantized: spec.quantize,
        out_dir: spec.out_dir.clone(),
        num_tensors,
        passthrough: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reads_the_model_type_only_and_refuses_other_tiers() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = |q| PrepareSpec {
            source: tmp.path().to_path_buf(),
            out_dir: tmp.path().join("out"),
            quantize: q,
        };
        assert!(!can_prepare(&spec(None)));
        std::fs::write(tmp.path().join("config.json"), r#"{"model_type":"qwen3"}"#).unwrap();
        assert!(!can_prepare(&spec(None)));
        std::fs::write(tmp.path().join("config.json"), r#"{"model_type":"yue2"}"#).unwrap();
        assert!(can_prepare(&spec(None)));
        assert!(matches!(
            prepare(&spec(Some(Quantize::Nvfp4))),
            Err(CoreError::Unsupported(_))
        ));
        // Not the pinned original: bf16 passthrough verification fails explicitly.
        assert!(prepare(&spec(None)).is_err());
        assert!(prepare(&spec(Some(Quantize::Q8))).is_err());
        assert!(!tmp.path().join("out").exists(), "a failed conversion leaves nothing");
    }
}
