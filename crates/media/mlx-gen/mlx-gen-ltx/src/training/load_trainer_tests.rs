#![cfg(test)]

use super::*;

/// sc-9989: the trainer must honor `LoadSpec::text_encoder` (the bundled Gemma-3 dir a
/// self-contained LTX install ships beside the tier weights) exactly like inference does. A
/// nonexistent override is rejected up front with the spec-side message — proving the override is
/// threaded through `load_trainer` → `load_trainer_from_dir` → `resolve_gemma_dir` rather than
/// silently falling back to `$LTX_GEMMA_DIR` / the HF-cache scan. Path-only resolution runs before
/// the split-weight load, so this needs no base snapshot and is deterministic (env-independent).
#[test]
fn load_trainer_forwards_text_encoder_override() {
    let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent/ltx_root".into()));
    spec.text_encoder = Some(WeightsSource::Dir("/nonexistent/ltx_gemma".into()));
    // `Box<dyn Trainer>` isn't `Debug`, so match rather than `unwrap_err`.
    let err = match load_trainer(&spec) {
        Ok(_) => panic!("expected a LoadSpec::text_encoder override error"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("LoadSpec text_encoder"), "got: {err}");
    assert!(err.contains("does not exist"), "got: {err}");
}
