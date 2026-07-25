//! Unregistered audio-lane snapshot preparation for Stable Audio 3.
//!
//! Both supported snapshot shapes are already dense safetensors, so preparation is a validated
//! passthrough. This module is intentionally not composed into `candle-audio-catalog` in sc-14535.

use core_llm::{Error as CoreError, ModelFormat, PrepareReport, PrepareSpec, Result as CoreResult};

use crate::weights::{safetensors_keys, SnapshotLayout};

/// Header-only recognition of a complete full or standalone snapshot.
pub fn can_prepare(spec: &PrepareSpec) -> bool {
    SnapshotLayout::from_dir(&spec.source).is_ok()
}

/// Validate and report an already-loadable dense Stable Audio 3 snapshot.
pub fn prepare(spec: &PrepareSpec) -> CoreResult<PrepareReport> {
    if let Some(quantize) = spec.quantize {
        return Err(CoreError::Unsupported(format!(
            "prepare: Stable Audio 3 snapshots have no {quantize:?} form; only dense passthrough is supported"
        )));
    }
    let layout = SnapshotLayout::from_dir(&spec.source)
        .map_err(|e| CoreError::Msg(format!("prepare: {e}")))?;
    let mut num_tensors = layout.keys.total;
    if let Some(text_weights) = &layout.text_weights_path {
        num_tensors += safetensors_keys(text_weights)
            .map_err(|e| CoreError::Msg(format!("prepare: {e}")))?
            .len();
    }
    Ok(PrepareReport {
        input_format: ModelFormat::Safetensors,
        quantized: None,
        out_dir: spec.source.clone(),
        num_tensors,
        passthrough: true,
    })
}
