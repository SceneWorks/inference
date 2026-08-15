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

/// The single VAE implementation used by both Bernini generator ids.
pub type ProviderVae = mlx_gen_wan::WanVae;
/// Bernini's provider-facing geometry, derived from its concrete VAE assignment.
pub const VAE_TILING: mlx_gen::tiling::VaeTiling = ProviderVae::VAE_TILING;

/// `(temporal, spatial, spatial)` stride used by Bernini's runtime latent construction.
pub(crate) const PROVIDER_VAE_STRIDE: (usize, usize, usize) = (
    VAE_TILING.temporal_scale as usize,
    VAE_TILING.spatial_scale as usize,
    VAE_TILING.spatial_scale as usize,
);

/// Resolve the runtime decode dimensions from a Bernini latent through its assigned VAE geometry.
pub(crate) fn decoded_output_geometry(
    latent_frames: i32,
    latent_height: i32,
    latent_width: i32,
) -> mlx_gen::Result<(i32, i32, i32)> {
    if latent_frames <= 0 || latent_height <= 0 || latent_width <= 0 {
        return Err(mlx_gen::Error::Msg(format!(
            "bernini: invalid latent decode geometry {latent_width}x{latent_height}x{latent_frames}"
        )));
    }
    let out_frames = if VAE_TILING.causal_temporal {
        latent_frames
            .checked_sub(1)
            .and_then(|frames| frames.checked_mul(VAE_TILING.temporal_scale))
            .and_then(|frames| frames.checked_add(1))
    } else {
        latent_frames.checked_mul(VAE_TILING.temporal_scale)
    };
    let out_height = latent_height.checked_mul(VAE_TILING.spatial_scale);
    let out_width = latent_width.checked_mul(VAE_TILING.spatial_scale);
    match (out_frames, out_height, out_width) {
        (Some(frames), Some(height), Some(width)) => Ok((frames, height, width)),
        _ => Err(mlx_gen::Error::Msg(
            "bernini: VAE output geometry overflows i32".into(),
        )),
    }
}

/// Resolve Bernini VAE geometry by registered generator id.
pub fn vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    matches!(provider_id, pipeline::MODEL_ID | bernini::MODEL_ID).then_some(VAE_TILING)
}

/// Resolve Bernini's provider-owned conservative VAE decode working-set peak.
///
/// Bernini deliberately retains its established `TilingConfig::auto` / explicit rung-2 selection;
/// this value is therefore not its selected-plan estimate. It is the calibrated Wan-z16
/// **single-pass upper bound**: every valid Bernini tile is capped at the full output on each axis and
/// the shared accumulator-plus-tile cost is monotone. See the typed upper-bound regression below.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<mlx_gen::VideoDecodeMemoryProfile> {
    vae_tiling(provider_id)?;
    mlx_gen_wan::conservative_video_decode_memory_profile_for_vae(VAE_TILING, width, height, frames)
}

/// Add all MLX Bernini providers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(pipeline::RENDERER_REGISTRATION)
        .register_memory_strategy(memory_strategy::RENDERER_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(mlx_gen::gen_core::MemoryContractFixtureRegistration {
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
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
            surface_specs: mlx_gen::gen_core::mlx_memory_contract_surface_specs,
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

    #[test]
    fn provider_ids_are_bound_to_the_wan_z16_geometry() {
        assert_eq!(super::VAE_TILING, super::ProviderVae::VAE_TILING);
        let geometry = super::decoded_output_geometry(3, 8, 10).unwrap();
        let (frames, height, width) = geometry;
        let reported_profile = super::conservative_video_decode_memory_profile(
            super::pipeline::MODEL_ID,
            width as u32,
            height as u32,
            frames as u32,
        );
        assert_eq!(
            (
                super::VAE_TILING.full_res_channels,
                super::PROVIDER_VAE_STRIDE,
                geometry,
                reported_profile.map(|profile| profile.working_set_bytes())
            ),
            (96, (4, 8, 8), (12, 64, 80), Some(403_292_160))
        );
        for id in [super::pipeline::MODEL_ID, super::bernini::MODEL_ID] {
            let mapped = super::vae_tiling(id).unwrap();
            assert!(!mapped.causal_temporal);
            assert_eq!(mapped, super::VAE_TILING);
            assert_eq!(
                super::conservative_video_decode_memory_profile(
                    id,
                    width as u32,
                    height as u32,
                    frames as u32,
                ),
                mlx_gen::VideoDecodeMemoryProfile::new(403_292_160, 0)
            );
        }
        assert_eq!(super::vae_tiling("not_bernini"), None);
    }

    #[test]
    fn single_pass_peak_bounds_every_bernini_auto_and_explicit_decode_plan() {
        use mlx_gen::tiling::TilingConfig;

        for (width, height, frames) in [
            (512, 512, 12),
            (768, 512, 84),
            (1280, 720, 124),
            (1536, 1024, 204),
        ] {
            let upper = super::conservative_video_decode_memory_profile(
                super::pipeline::MODEL_ID,
                width,
                height,
                frames,
            )
            .unwrap()
            .working_set_bytes();
            let auto = TilingConfig::auto(height as i32, width as i32, frames as i32);
            let auto_peak = mlx_gen_wan::pipeline::video_decode_peak_bytes_for_vae_and_tiling(
                super::VAE_TILING,
                width,
                height,
                frames,
                auto.as_ref(),
            )
            .unwrap();
            assert!(auto_peak <= upper, "auto {width}x{height}x{frames}");

            for &edge in crate::memory_strategy::DECODE_TILE_EDGES {
                let explicit = TilingConfig::spatial_only(
                    edge as i32,
                    crate::memory_strategy::DECODE_OVERLAP as i32,
                );
                let explicit_peak =
                    mlx_gen_wan::pipeline::video_decode_peak_bytes_for_vae_and_tiling(
                        super::VAE_TILING,
                        width,
                        height,
                        frames,
                        Some(&explicit),
                    )
                    .unwrap();
                assert!(
                    explicit_peak <= upper,
                    "explicit {edge} for {width}x{height}x{frames}"
                );
            }
        }
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
