//! Qwen-Image adapter consumption (sc-2528). The model-specific piece is the key→module map (the
//! `AdaptableHost for QwenTransformer` + block/attention/feed-forward hosts, the Rust analog of the
//! fork's `QwenLoRAMapping`); per-file LoKr/LoRA dispatch, LoRA-prefix detection, stacking + mixed,
//! and the strict no-silent-drop policy are the shared core seam (sc-2534). Both T2I (`model.rs`)
//! and Edit (`model_edit.rs`) share the `QwenTransformer`, so this serves both.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a Qwen transformer `host` (stacked, mixed LoRA/LoKr), via
/// the core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target.
pub fn apply_qwen_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "qwen_image")
}
