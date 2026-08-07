//! ByteDance **Bernini renderer** — native MLX provider (epic 4699, sc-4706).
//!
//! The Bernini renderer is **Wan2.2-T2V-A14B verbatim** (dual-expert MoE, z16 `AutoencoderKLWan`,
//! UMT5-XXL, UniPC flow scheduler) with three Bernini-specific deltas layered on top:
//!
//!   1. **source-id rotary** ([`rope`]) — each conditioning source (and the noisy target) is
//!      patch-embedded with a per-source constant rotary phase composed onto the standard 3-axis Wan
//!      RoPE, so the model can tell sources apart.
//!   2. **token-axis packed conditioning** — VAE-encoded source media are patch-embedded and
//!      concatenated with the noisy target on the token axis; at inference (batch 1) the reference's
//!      varlen attention is a single `cu_seqlens` segment, i.e. plain full self-attention over the
//!      packed sequence (so [`mlx_gen_wan::WanTransformer::forward_packed`] suffices). Only the target
//!      tokens are kept from the prediction.
//!   3. **APG guidance** — the 7 renderer guidance modes (`t2v`, `t2v_apg`, `v2v`, `v2v_chain`,
//!      `v2v_apg`, `r2v_apg`, `rv2v`) with the adaptive-projected-guidance families and the
//!      one-time omega rescale on the low-noise expert switch.
//!
//! The renderer reuses the [`mlx_gen_wan`] foundation directly; the converter
//! ([`mlx_gen_wan::convert::assemble_bernini_renderer_snapshot`], sc-4705) emits a `wan2_2_t2v_14b`
//! snapshot plus a `bernini_renderer.json` knob sidecar that this crate consumes.

pub mod assembly;
pub mod bernini;
pub mod clip_diff;
pub mod config;
pub mod connector;
pub mod convert;
pub mod forward;
pub mod guidance;
pub mod mar;
pub mod memory_strategy;
pub mod pipeline;
pub mod preprocess;
pub mod process;
pub mod qwen2_5_vl;
pub mod rope;
pub mod template;
pub mod vae_features;
pub mod vae_preprocess;
pub mod vision;
pub mod vit_guidance;
pub mod vit_preprocess;

/// Add all MLX Bernini providers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(pipeline::RENDERER_REGISTRATION)
        .register_memory_strategy(memory_strategy::RENDERER_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: memory_strategy::RENDERER_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(
                    memory_strategy::RENDERER_ID,
                    spec,
                )
            },
        })
        .register_memory_behavior(memory_strategy::RENDERER_MEMORY_BEHAVIOR)
        .register_generator(bernini::FULL_REGISTRATION)
        .register_memory_strategy(memory_strategy::FULL_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            provider_id: memory_strategy::FULL_ID,
            contract: |spec| {
                memory_strategy::weights_free_memory_strategy_contract(
                    memory_strategy::FULL_ID,
                    spec,
                )
            },
        })
        .register_memory_behavior(memory_strategy::FULL_MEMORY_BEHAVIOR)
}

/// Build the complete explicit MLX Bernini provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(explicit, ["bernini_renderer", "bernini"]);
    }

    /// The shared weights-free behavior oracle: every registered contract, safety check, fixture and
    /// request scope has to agree with the ladder each provider declares — including that a rung the
    /// contract calls `Missing` cannot be begun, and that an admitted scope's lifecycle hooks accept
    /// exactly the parameters it advertised.
    ///
    /// Both load shapes are walked because rung 4 is declared **per load**: the eager pass is what
    /// proves the `Missing` declaration is enforced rather than merely written.
    #[test]
    fn shared_ladder_registrations_pass_the_weights_free_behavior_oracle() {
        let registry = super::provider_registry().unwrap();
        for load_shape in [
            mlx_gen::LoadShape::DeferredMaterialization,
            mlx_gen::LoadShape::EagerMaterialization,
        ] {
            let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir("/nonexistent".into()))
                .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
                .with_load_shape(load_shape);
            gen_core_testkit::memory_strategy::memory_strategy_registry_conformance(
                &registry, &spec,
            );
        }
    }
}
