//! Install-time DiT pre-quantization for the Anima convert-at-install tiers (sc-10517).
//!
//! Anima ships **convert-at-install**: SceneWorks never redistributes converted weights, so each tier
//! (bf16 / q8 / q4) is produced on-device from the ungated `circlestone-labs/Anima` source. The bulk
//! is the single-file Cosmos-Predict2 DiT (`split_files/diffusion_models/anima-{variant}-v1.0.safetensors`),
//! which BUNDLES both the DiT and the `AnimaTextConditioner` (the `…llm_adapter.*` sub-tree).
//!
//! ## Quant policy — dense conditioner, transformer-only pack
//! This packs ONLY the Cosmos DiT's group-aligned 2-D Linear leaves to Q`bits` and keeps the
//! 134.7M-param `AnimaTextConditioner` DENSE bf16. That is the "dense-TE, transformer-only" policy the
//! sibling converters use (`sd3_5_*_quant` / `qwen_image` pack the DiT and keep their text encoders
//! dense — a Q4 text-conditioning path degrades semantics). The conditioner sits on the text→DiT
//! cross-attention path, so it is treated like a text encoder, not part of the packed backbone.
//!
//! The Qwen3-0.6B text encoder and the Qwen-Image VAE are SEPARATE files this converter never touches;
//! they stay dense bf16 in every tier. The SceneWorks worker's install-time converter runs the same
//! transformation via the shared [`mlx_gen::quant`] primitives — this crate-owned entry point (mirrors
//! `mlx_gen_flux2::quantize_flux2_dit`) is what the rev-bump lockstep (sc-10523) lets the worker call
//! directly, so the two never drift.

use std::collections::HashMap;
use std::path::Path;

use mlx_gen::quant::{quantize_map, save_map};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::Array;

/// The conditioner sub-tree marker: any key CONTAINING `llm_adapter.` is the bundled
/// `AnimaTextConditioner` (kept dense) — everything else is the Cosmos DiT (packed). Prefix-agnostic,
/// so it handles the `net`-rooted base cut AND the `model.diffusion_model`-rooted turbo/aesthetic cuts
/// identically, exactly like [`crate::loader::split_anima_keys`].
const ADAPTER_MARKER: &str = "llm_adapter.";

/// Whether a `{base}` (a tensor key minus its `.weight` suffix) is a Cosmos-DiT quantization target —
/// true UNLESS it lives inside the bundled `AnimaTextConditioner`. Shared by [`quantize_anima_dit`] and
/// its tests so the "keep the conditioner dense" invariant is one predicate, and so a prefix-agnostic
/// test can prove it holds for every checkpoint root (`net` vs `model.diffusion_model`).
pub fn is_dit_quant_target(base: &str) -> bool {
    !base.contains(ADAPTER_MARKER)
}

/// Pre-quantize an Anima DiT checkpoint to Q`bits` (group size `group_size`), writing a single packed
/// `.safetensors` to `dst_file`.
///
/// Only the Cosmos DiT's group-aligned 2-D Linear leaves are packed (the `{base}.weight` →
/// `{base}.weight`+`.scales`+`.biases` triple, via [`is_dit_quant_target`]); the bundled
/// `AnimaTextConditioner` (`…llm_adapter.*`), every norm / 1-D tensor, the 17-channel patch-embed
/// (in-dim 68, not group-divisible), and any other non-group-aligned projection pass through unchanged
/// (dense bf16). The pack is byte-identical to the load-time
/// [`mlx_gen::adapters::AdaptableLinear::quantize`] (the shared [`quantize_map`] casts to bf16 first),
/// so the loader's [`mlx_gen::quant::lin`] packed-detects each tier off the on-disk `{base}.scales`
/// with no side manifest. `bits` must be 4 or 8; `group_size` is the codebase default 64.
pub fn quantize_anima_dit(
    src_file: &Path,
    dst_file: &Path,
    bits: i32,
    group_size: i32,
) -> Result<()> {
    let w = Weights::from_file(src_file)?;
    let map: HashMap<String, Array> = w
        .keys()
        .map(|k| (k.to_string(), w.get(k).expect("listed key").clone()))
        .collect();
    let packed = quantize_map(map, bits, group_size, is_dit_quant_target)?;
    save_map(dst_file, &packed)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditioner_keys_are_never_quant_targets() {
        // The bundled conditioner (`…llm_adapter.*`) must stay dense on EVERY checkpoint root — the
        // base cut roots the DiT at `net`, turbo/aesthetic at `model.diffusion_model`. A converter that
        // hardcoded `net.llm_adapter.` (the story's original, WRONG instruction) would treat the
        // `model.diffusion_model.llm_adapter.*` conditioner as a DiT target and pack it to Q4 — this
        // predicate is prefix-agnostic and catches that.
        for root in ["net", "model.diffusion_model"] {
            // conditioner projections (dense) — for both roots.
            assert!(!is_dit_quant_target(&format!(
                "{root}.llm_adapter.blocks.0.self_attn.q_proj"
            )));
            assert!(!is_dit_quant_target(&format!(
                "{root}.llm_adapter.out_proj"
            )));
            assert!(!is_dit_quant_target(&format!("{root}.llm_adapter.embed")));
            // Cosmos DiT projections (quant targets) — for both roots.
            assert!(is_dit_quant_target(&format!(
                "{root}.blocks.0.self_attn.q_proj"
            )));
            assert!(is_dit_quant_target(&format!("{root}.final_layer.linear")));
            assert!(is_dit_quant_target(&format!(
                "{root}.t_embedder.1.linear_1"
            )));
        }
    }
}
