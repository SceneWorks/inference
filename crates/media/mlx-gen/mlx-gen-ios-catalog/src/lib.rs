//! The iOS media composition root: a **narrow** image-generation catalog for a phone.
//!
//! This exists instead of reusing [`mlx_gen_catalog`], and the difference is the whole point.
//! That catalog composes 32 provider crates — including the video families (`wan`, `ltx`,
//! `mochi`, `svd`) — none of which is validated on iOS or shaped for a device with a hard per-app
//! memory cap. Depending on it to obtain one small generator would compile a graph the platform
//! cannot run.
//!
//! So this catalog is a deliberate, reviewable subset: the generators that actually fit. Adding
//! one is an edit here *and* to this crate's ordered surface test, exactly as the platform
//! catalogs work — a provider crate existing in the workspace does not mean it ships.
//!
//! # What ships
//!
//! **SANA** (`mlx-gen-sana`), base and sprint — the smallest capable image generator here, with
//! text conditioning reused from `mlx-gen-pid`'s Gemma-2-2b-it caption encoder (epic 8485 /
//! sc-8488) rather than duplicated.
//!
//! **It does not currently fit an 8 GB device.** Measured with this crate's `image_budget`
//! example against a 4096 MiB budget: 8340 MiB peak at 1024px under `OffloadPolicy::Sequential`,
//! and still 4363 MiB at 256px. Resolution is not the lever — most of the footprint is weights
//! (Q4: encoder 2.3 GB + DiT 2.0 GB + DC-AE 1.25 GB), not activations. An earlier "~2 GB" estimate
//! in these docs came from prose rather than measurement and was wrong by ~4×.
//!
//! Shipping G6 needs a smaller text encoder (the 2-bit SANA quant the crate docs mention is not
//! ported), DC-AE tiling, or a decision to target 12 GB devices only. See `docs/ios-epics.md` E5.
//!
//! # What does not, yet
//!
//! - **Video** — wrong shape for the memory cap and the thermal envelope, full stop.
//! - **`mlx-gen-sensenova`** (the unified AR-LLM-plus-image model) — a separate epic (E6), because
//!   its dual-path runtime shares `mlx-llm`'s KV cache and is the riskiest piece of the iOS
//!   initiative. When it lands it registers here beside SANA.
//! - **Larger diffusion families** (`flux`, `qwen-image`, `z-image`) — they may fit a 12 GB
//!   device, but the guardrail is to generalize *downward* to an 8 GB one
//!   (`docs/architecture/ios-project-spec.md` §0.1). Measure with
//!   `mlx-llm`'s `memory_budget` example before proposing one.

use mlx_gen::gen_core::{ProviderRegistry, ProviderRegistryBuilder, Result};

/// Add the iOS image generators to an explicit media registry builder, in catalog order.
///
/// Order is part of the surface: [`provider_registry`]'s ids are asserted in this order by the
/// crate's surface test, so a reordering is a deliberate edit rather than an accident.
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    mlx_gen_sana::register_providers(registry)
}

/// Build the complete, explicit iOS media provider catalog.
pub fn provider_registry() -> Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    /// The exact, ordered surface this catalog ships on iOS.
    ///
    /// This is the reviewable source of truth for what a phone compiles. The negative assertions
    /// matter as much as the positive one: an incidental `mlx-gen-catalog` dependency, or a video
    /// family added by reflex, fails here rather than being discovered as a jetsam kill on a
    /// device.
    #[test]
    fn catalog_surface_is_explicit_and_narrow() {
        let registry = super::provider_registry().expect("iOS media catalog builds");
        let ids: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            ids,
            ["sana_1600m", "sana_sprint_1600m"],
            "the iOS media surface changed -- if deliberate, update this test and confirm the new \
             provider fits an 8 GB device's cap with mlx-llm's memory_budget example"
        );

        // No video: the families in the full catalog (wan, ltx, mochi, svd) are the wrong shape
        // for a phone, and this catalog exists precisely to exclude them.
        for id in &ids {
            for video in ["wan", "ltx", "mochi", "svd", "seedvr"] {
                assert!(
                    !id.contains(video),
                    "{id} looks like a video provider; iOS ships image generation only"
                );
            }
        }

        // Nothing but generators. A trainer or a captioner arriving here would mean this catalog
        // had drifted toward the full media graph.
        assert_eq!(registry.trainers().len(), 0, "iOS ships no trainers");
        assert_eq!(registry.captioners().len(), 0, "iOS ships no captioners");
    }

    /// Every descriptor is on the `mlx` backend, which `runtime-catalog` will enforce again when
    /// this registry is composed into the bundle. Asserted here too so the failure names *this*
    /// catalog rather than surfacing as a bundle-validation error.
    #[test]
    fn every_generator_is_mlx() {
        let registry = super::provider_registry().expect("iOS media catalog builds");
        for registration in registry.generators() {
            let descriptor = (registration.descriptor)();
            assert_eq!(
                descriptor.backend, "mlx",
                "generator {} is on backend {}",
                descriptor.id, descriptor.backend
            );
        }
    }
}
