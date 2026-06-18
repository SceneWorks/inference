//! Ideogram 4 adapter (LoRA) consumption. The model-specific piece is the key→module map (the
//! `AdaptableHost for Ideogram4Transformer` + the per-block attention / feed-forward / modulation
//! hosts in `transformer/`, the Rust analog of the fork's LoRA mapping); per-file LoRA/LoKr
//! dispatch, prefix detection (`diffusion_model.` / `transformer.` / bare), stacking, and the
//! strict no-silent-drop policy are the shared core seam (sc-2534), exactly as FLUX.2 (sc-2646),
//! Qwen (sc-2528), and Z-Image (sc-2602) use it.
//!
//! The first consumer is the **ostris TurboTime** few-step LoRA (issue #488): a single rank-128
//! adapter over the per-layer `attention.qkv`/`o`, `feed_forward.w{1,2,3}`, and `adaln_modulation`
//! modules, applied at scale 1.0 onto the conditional DiT to drive the CFG-free single-DiT turbo
//! path. LoRA targets the **transformer only** (the Qwen3-VL text encoder and the FLUX.2 VAE are
//! not adapter targets).

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto an Ideogram 4 transformer `host` (stacked, mixed LoRA/LoKr),
/// via the core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target.
/// The core `Adapter::residual` runs in the natural dtype (f32 — Ideogram's DiT activations are
/// f32), so this is dtype-invariant for the crate. Used both for the model-defining TurboTime LoRA
/// (turbo load) and any user Ideogram LoRA.
pub fn apply_ideogram_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "ideogram_4")
}
