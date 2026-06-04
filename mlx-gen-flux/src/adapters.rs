//! FLUX.1 (schnell + dev) adapter consumption (sc-2657). The model-specific piece is the key→module
//! map (the `AdaptableHost for FluxTransformer` + block/attention/feed-forward hosts in
//! `transformer.rs`, the Rust analog of the fork's `FluxLoRAMapping`); per-file LoKr/LoRA dispatch,
//! LoRA-prefix detection, kohya flattening (sc-2618), BFL/ComfyUI fused→split (sc-2743), stacking +
//! mixed, and the strict no-silent-drop policy are the shared core seam (sc-2534), exactly as Z-Image
//! (sc-2602), Qwen (sc-2528), and FLUX.2 (sc-2646) use it. LoRA/LoKr are **transformer-only** for
//! FLUX.1 (the VAE + T5/CLIP text encoders are not adapter targets); the same `FluxTransformer` serves
//! both schnell and dev, so this serves both variants.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::AdaptableHost;
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;

/// Apply every adapter in `specs` onto a FLUX.1 transformer `host` (stacked, mixed LoRA/LoKr), via the
/// core [`apply_adapters_strict`] — errors, never silently drops, on an unmatched target. The adapter
/// residuals run f32 (LoRA's `K = rank ≤ 512` second matmul is the dense 16-bit Metal GEMM shape on the
/// bf16 conditioning path; the core `Adapter::residual` already computes f32 and casts back).
pub fn apply_flux_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    apply_adapters_strict(host, specs, "flux1")
}
