//! # mlx-gen-mage
//!
//! The **Mage-Flow** (`microsoft/Mage`, MIT) provider crate for [`mlx-gen`](mlx_gen): a compact 4B
//! text→image + instruction-editing stack made of a one-step **Mage-VAE** codec (128-channel,
//! 16×-downsampled latents), a 12-block **NR-MMDiT** rectified-flow denoiser, and a **Qwen3-VL-4B**
//! text encoder.
//!
//! ## Status — scaffold (sc-14037)
//!
//! This crate is **structure only**. [`config`] carries the real, verified shapes; every model
//! module below is an empty, documented seam, and [`model::load`] returns
//! [`mlx_gen::Error::Unsupported`] rather than a half-working model. The crate is therefore
//! deliberately **not** registered in [`mlx-gen-catalog`]'s shipped provider registry yet — see
//! [`model`] for the registration constants and the story that flips them on.
//!
//! ## Reuse lineage
//!
//! Mage's own `transformer/config.json` declares `schedule_mode: "z-image"`, `rope_type: "msrope"`,
//! `time_type: "qwen_proj"` and `double_block_type: "double_stream"`: the NR-MMDiT is a
//! reparameterised Z-Image (Tongyi) S3-DiT, so `mlx-gen-z-image`-shaped module boundaries are the
//! right template. **Two inherited assumptions are wrong for Mage and are pinned as constants in
//! [`config`] so they cannot leak back in:** the DiT FFN is `gelu-approximate`, *not* SwiGLU
//! ([`config::FFN_ACTIVATION`]), and the text conditioning is the **final** post-RMSNorm hidden
//! state, *not* the penultimate layer.
//!
//! ## Ground truth
//!
//! The frozen PyTorch reference is vendored byte-identically at
//! `crates/media/mlx-gen/_vendor/mage_flow/` (upstream `microsoft/Mage @ df7f84d9`); the six
//! architectural questions it answers are written up in `_vendor/MAGE_FLOW_GAPS.md`, and CPU
//! boundary goldens plus their hardened checker live under `crates/media/mlx-gen/tools/`. Read
//! those before porting anything — several of them correct the original epic description.
//!
//! ## Deliberate divergence
//!
//! The reference runs a **mandatory, fail-closed content-moderation gate** over every prompt
//! (`TextEncoder.screen_text` / `screen_edit`). It is **not** ported (decision recorded on
//! sc-14105): SceneWorks ships no content classifier for any family, by documented product
//! posture. The text-encoder port needs the *embedding* forward only — no `lm_head`, no KV-cache
//! decode loop, no `.generate()`. The Gaussian-Shading watermark ([`latent`]) is kept: provenance
//! marking stays, blocking does not.
//!
//! [`mlx-gen-catalog`]: https://docs.rs/mlx-gen-catalog

// ---------------------------------------------------------------------------------------------
// DECISION (sc-14037): there is deliberately **no `loader.rs`**.
//
// The sibling convention (`mlx-gen-z-image/src/loader.rs`) puts `load_transformer` /
// `load_text_encoder` / `load_vae` in one shared file. That is a three-way collision here, because
// sc-14038, sc-14039 and sc-14040 land concurrently and would each create the same file *and* each
// add the same `pub mod loader;` line. **Each component's weight loading lives inside the module
// that owns it** — `text_encoder::load`, `vae::load`, `transformer::load` — so a story touches only
// its own files. sc-14041, which assembles them, may add a thin `loader.rs` that re-exports the
// three (restoring the sibling's public shape) once all three exist; it must not move the bodies.
// ---------------------------------------------------------------------------------------------

pub mod config;
pub mod model;

// --- NR-MMDiT (sc-14040) -------------------------------------------------------------------
pub mod attention;
pub mod feed_forward;
pub mod final_layer;
pub mod rope_embedder;
pub mod timestep_embedder;
pub mod transformer;
pub mod transformer_block;

// --- Qwen3-VL text encoder (sc-14038; vision tower sc-14048) ---------------------------------
pub mod text_encoder;

// --- Mage-VAE one-step codec (sc-14039) ------------------------------------------------------
pub mod vae;

// --- Gaussian-Shading watermarked initial noise (sc-14104) -----------------------------------
pub mod latent;

// --- Rectified-flow sampler + native-resolution packing (sc-14041) ---------------------------
pub mod pipeline;

// ---------------------------------------------------------------------------------------------
// Re-export surface.
//
// PARALLEL-EXECUTION CONTRACT (sc-14037): the four P1 ports land concurrently, so each owns ONE
// pre-seeded line below. **Replace your own placeholder in place — do not append, reorder, or
// touch a neighbouring line** and the four diffs stay in separate hunks. `lib.rs` already declares
// every module above, so no story needs to add a `mod` line either.
// ---------------------------------------------------------------------------------------------
pub use config::{MageFlowConfig, QwenVlTextConfig, FAMILY};
pub use model::{descriptor_for, MageVariant, MODEL_IDS};
// sc-14038 (text encoder) re-exports here:

pub use vae::{MageVae, VaePart};

// sc-14040 (NR-MMDiT) re-exports here:

// sc-14104 (Gaussian-Shading noise) re-exports here:

// sc-14041 (pipeline + the loaded model) re-exports here:

// Later phases add their own modules rather than growing these: `quant` (Q4/Q8 tiers, sc-14046),
// `convert` (offline pre-quantisation, sc-14046), `adapters` (LoRA/LoKr routing, sc-14057) and
// `training` (LoRA + base fine-tune, sc-14055/sc-14056). They are not stubbed here because their
// shape is decided by those stories, not by this one.

/// Add every Mage-Flow MLX provider to an explicit media registry builder.
///
/// **Not yet called by [`mlx_gen_catalog`](https://docs.rs/mlx-gen-catalog)** — see
/// [`model::REGISTRATIONS`] for why, and which story turns it on.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    model::REGISTRATIONS
        .iter()
        .fold(registry, |registry, registration| {
            registry.register_generator(*registration)
        })
}

/// Build the explicit Mage-Flow MLX provider catalog (this crate only).
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    use super::*;

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(
            generators,
            [
                "mage_flow",
                "mage_flow_base",
                "mage_flow_turbo",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo",
            ]
        );
        // The scaffold ships no trainer (sc-14055/sc-14056) and no captioner/embedder surface.
        assert_eq!(registry.trainers().count(), 0);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }

    /// Every id is prefixed with the family id, matching the image-family convention
    /// (`flux2_*`, `krea_2_*`, `z_image_*`). SceneWorks routes and groups on this prefix.
    #[test]
    fn every_model_id_is_family_prefixed() {
        for id in MODEL_IDS {
            assert!(
                id.starts_with(FAMILY),
                "{id} does not carry the '{FAMILY}' family prefix"
            );
        }
    }
}
