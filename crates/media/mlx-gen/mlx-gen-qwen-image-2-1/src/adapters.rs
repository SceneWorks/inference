//! Qwen-Image 2.1 adapter consumption (sc-24156). The model-specific piece is the key→module map —
//! `AdaptableHost for QwenImage21Transformer` (+ its block / attention / feed-forward hosts) in
//! [`crate::transformer`]. Per-file format dispatch (diffusers/peft LoRA, kohya `lora_unet_` LoRA,
//! peft and third-party LyCORIS LoKr), LoRA-prefix detection, stacking of several files with their
//! own per-file strengths, and the strict no-silent-drop policy are the shared core seam
//! ([`apply_adapters_strict`]); nothing here reimplements them.
//!
//! Every route behind the `qwen_image_2_1` descriptor — text-to-image and the 1–10-reference
//! edit/RGBA path — renders through the one DiT built by `model::load_heavy`, which is where this
//! install runs, so every route gets the same adapters.

use mlx_gen::adapters::loader::{apply_adapters_strict, ApplyReport};
use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::runtime::AdapterSpec;
use mlx_gen::Result;
use mlx_rs::Dtype;

use crate::model::MODEL_ID;

/// Apply every adapter in `specs` onto a Qwen-Image 2.1 DiT `host` (stacked, mixed LoRA/LoKr, each
/// at its own `scale`), via the core [`apply_adapters_strict`]. Errors — never silently drops — when
/// any adapter target resolves to no module (the error names every unmatched key) or when a
/// non-empty spec list matches nothing at all.
///
/// Call it **after** any quantization: the adapters are forward-time residuals over the base, so
/// they apply identically over a dense bf16 DiT and a packed Q4/Q8 one, whereas a quantize run after
/// the install would have to re-pack a base that already carries them.
pub fn apply_qwen_image_2_1_adapters(
    host: &mut impl AdaptableHost,
    specs: &[AdapterSpec],
) -> Result<ApplyReport> {
    let report = apply_adapters_strict(host, specs, MODEL_ID)?;
    // The trainer uses f32 master factors but computes with the folded factors cast to the
    // DiT dtype. A saved master must follow that same path on reload: narrowing only the
    // final f32 residual changes the trained velocity. Packed DiTs compute in bf16.
    for path in host.adaptable_paths() {
        let parts: Vec<_> = path.split('.').collect();
        if let Some(linear) = host.adaptable_mut(&parts) {
            let dtype = linear.weight_dtype().unwrap_or(Dtype::Bfloat16);
            let adapters = linear
                .adapters()
                .iter()
                .map(|adapter| match adapter {
                    Adapter::Lora { a, b, scale } => Ok(Adapter::Lora {
                        a: a.as_dtype(dtype)?,
                        b: b.as_dtype(dtype)?,
                        scale: *scale,
                    }),
                    other => Ok(other.clone()),
                })
                .collect::<mlx_gen::Result<Vec<_>>>()?;
            linear.set_adapters(adapters);
        }
    }
    Ok(report)
}
