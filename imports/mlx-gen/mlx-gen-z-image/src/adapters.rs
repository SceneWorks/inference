//! Z-Image adapter consumption (sc-2602). The model-specific piece is the key→module map (the
//! top-level `AdaptableHost for ZImageTransformer`, the Rust analog of the fork's `ZImageLoRAMapping`,
//! in `transformer.rs`); everything else — per-file LoKr/LoRA dispatch, LoRA-prefix detection,
//! stacking + mixed, and the strict no-silent-drop policy — is the shared core seam (sc-2534).

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a Z-Image transformer `host` (stacked, mixed LoRA/LoKr),
/// via the core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target.
pub fn apply_z_image_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "z_image")
}
