//! Mage-Flow adapter consumption (sc-14055): install a trained (or imported) LoRA/LoKr adapter onto
//! the NR-MMDiT for inference.
//!
//! The model-specific piece is the key→module map — the top-level [`AdaptableHost`] for
//! [`MageTransformer`](crate::transformer::MageTransformer) plus the per-module hosts on the joint
//! attention, FFN, and block, all in the DiT source. Everything else — per-file LoKr/LoRA dispatch,
//! LoRA-prefix detection, stacking + mixed, and the strict no-silent-drop policy — is the shared
//! core seam ([`mlx_gen::adapters::loader`]).
//!
//! This is deliberately the minimal PEFT dotted-key reload path the LoRA-training story needs to
//! prove its round-trip: a trainer writes `{path}.lora_A.weight` / `.lora_B.weight` / `.alpha` (or
//! the LoKr `{path}.lokr_w1` + factors) at exactly the [`AdaptableHost::adaptable_paths`] the
//! reload resolves against, so a Mage adapter trained here reloads bit-for-bit. Community
//! family-detection/import (kohya/BFL spellings, mixed stacks) is sc-14057's scope — the shared
//! loader already supports those spellings once the host exposes its kohya/BFL surfaces.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a Mage-Flow NR-MMDiT `host` (stacked, mixed LoRA/LoKr), via
/// the shared strict installer — it errors, never silently drops, on an unmatched target.
pub fn apply_mage_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "mage_flow")
}
