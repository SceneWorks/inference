//! # candle-gen-sana
//!
//! SANA (NVlabs) provider crate for `candle-gen` — the Windows/CUDA + Linux sibling of
//! `mlx-gen-sana` (mlx-gen #612), epic 11776.
//!
//! **Gating spike sc-11777** delivers the two hard primitives whose candle/CUDA feasibility was the
//! GO/NO-GO question for a native SANA port:
//!
//!  - the **DC-AE** (deep-compression autoencoder) **f32 image decoder** — 6-stage conv decoder,
//!    `ResBlock`s, `EfficientViTBlock`s, `ConvPixelShuffle` up-sampling, trimmed-RMS norm (`trms2d`),
//!    SiLU — a faithful component port of diffusers `AutoencoderDC`
//!    (`mit-han-lab/dc-ae-f32c32-sana-1.0`, the autoencoder behind SANA-1.6B 1024px); and
//!  - the **EfficientViT GLU** ReLU-**linear**-attention block (O(N), softmax-free) — the *shared hard
//!    primitive* the SANA Linear-DiT trunk (story 2) reuses, so it is written once here
//!    (`dc_ae::relu_linear_attention` + the `LinearAttn` block).
//!
//! A compact symmetric **encoder** ([`dc_ae::DcAeEncoder`]) rides along only far enough for a
//! round-trip reconstruction check; the decoder is the parity deliverable. See [`dc_ae`] for the
//! block-by-block port and the port notes (NCHW-native, f32).
//!
//! **sc-11778** adds the **Linear-DiT trunk** ([`transformer::SanaTransformer`]) — the ReLU
//! linear-attention DiT blocks (reusing `dc_ae::relu_linear_attention`), the `GLUMBConv` Mix-FFN
//! (3×3 depthwise conv, reusing `dc_ae::glu_mbconv_core`), NoPE, and the adaLN-single timestep /
//! caption conditioning (base SANA-1.6B + the SANA-Sprint guidance-embed / qk-norm superset). Its
//! `[B, 32, H, W]` noise prediction feeds [`dc_ae::DcAeDecoder::decode`] directly.
//!
//! **sc-11779** adds the **text conditioning** ([`text_encoder::SanaTextEncoder`]) — a thin wrapper
//! that REUSES PiD's native gemma-2-2b-it CHI caption encoder ([`candle_gen_pid::CaptionEncoder`])
//! via the shared [`candle_gen_pid::CaptionEncoder::with_chi_prompt`] seam, differing from PiD only
//! in the CHI template's quoting around `Enhanced prompt`. Prompt → `[1, 300, 2304]` gemma
//! last-hidden caption embedding feeding the trunk's `attn2` cross-attention. Mirrors mlx-gen-sana's
//! sc-8488 (mlx-gen #614).
//!
//! **sc-11780** assembles the end-to-end base txt2img [`pipeline`] (TE → trunk → DC-AE, driven by
//! candle's unified flow scheduler, static shift 3.0, true CFG) and the gen-core [`model`] adapter
//! (exposed under `sana_1600m` through the explicit family catalog), mirroring mlx-gen-sana's sc-8489.
//!
//! **sc-11781** adds the **SANA-Sprint** CFG-free few-step variant (epic 11776; the candle sibling of
//! mlx sc-8490): the SCM / TrigFlow continuous-time-consistency sampler
//! ([`candle_gen::run_scm_sampler`] + [`candle_gen::ScmScheduler`]), a SEPARATE
//! [`pipeline::SanaSprintPipeline`] (embedded-guidance trunk forward via
//! [`transformer::SanaTransformer::forward_with_guidance`], 1–4 steps, no CFG uncond pass), and the
//! gen-core [`model`] adapter registered under `sana_sprint_1600m`. The base `sana_1600m` pipeline /
//! trunk `forward` / example are byte-unchanged — Sprint is purely additive.
//!
//! **sc-16959** (epic 16948) wires per-step latent [`preview`]s into **both** routes. This is the only
//! candle family driving two shared samplers, and the only one carrying two committed fits: base
//! previews through [`candle_gen::run_flow_sampler`] with the epic-16624 base DC-AE fit, Sprint through
//! [`candle_gen::run_scm_sampler`] with the Sprint fit and a `1/σ_data` correction. The two
//! autoencoders differ in their tensor bytes at an identical container size, so the two fits are not
//! interchangeable — see [`preview`] for the enumeration, the provenance and the guards.

pub mod config;
pub mod dc_ae;
pub mod memory_strategy;
pub mod model;
pub mod nvfp4_dit;
pub mod pipeline;
pub mod preview;
pub mod text_encoder;
pub mod transformer;

pub use candle_gen::gen_core;
pub use config::{BlockType, DcAeConfig, SanaTransformerConfig};
pub use dc_ae::{DcAeDecoder, DcAeEncoder};
pub use memory_strategy::SanaVariant;
pub use model::{
    descriptor, load, load_sprint, sprint_descriptor, MODEL_ID, RES_MULTIPLE, SPRINT_MODEL_ID,
};
pub use nvfp4_dit::{
    summarize, ActProbe, ActRecord, DitPlan, LayerRole, LayerSparsitySummary, Nvfp4Quant,
    Nvfp4Report,
};
pub use pipeline::{
    denoise_sprint, SanaGenerateRequest, SanaPipeline, SanaSprintPipeline, SPRINT_DEFAULT_GUIDANCE,
    SPRINT_DEFAULT_STEPS,
};
pub use text_encoder::{SanaTextEncoder, MAX_SEQUENCE_LENGTH, SANA_CHI_PROMPT};
pub use transformer::SanaTransformer;

/// Add the Candle SANA base and Sprint generators to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(model::REGISTRATION)
        .register_generator(model::SPRINT_REGISTRATION);
    register_memory_contract_surfaces(registry)
        .register_memory_behavior(memory_strategy::BASE_MEMORY_BEHAVIOR)
        .register_memory_behavior(memory_strategy::SPRINT_MEMORY_BEHAVIOR)
}

/// Register SANA's pre-load memory contract and its weights-free contract fixture for both routes.
///
/// Called unconditionally from [`register_providers`] — nothing here requires a CUDA device, so a
/// selector can price SANA on a host with no GPU and before any weight file is opened. Composition
/// roots that assemble their own catalog must NOT call this a second time; `register_providers`
/// already covers it.
pub fn register_memory_contract_surfaces(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::CANDLE_BASE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: memory_strategy::surface_specs,
            provider_id: MODEL_ID,
            contract: |spec| memory_strategy::weights_free_contract(SanaVariant::Base, spec),
        })
        .register_memory_strategy(memory_strategy::CANDLE_SPRINT_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: memory_strategy::surface_specs,
            provider_id: SPRINT_MODEL_ID,
            contract: |spec| memory_strategy::weights_free_contract(SanaVariant::Sprint, spec),
        })
}

/// Build the complete explicit Candle SANA provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    use super::gen_core;

    /// sc-19753 feature review, BLOCKER 4 — before this, `register_providers` registered two
    /// generators and NOTHING for the memory registry, so a selector could not price either SANA
    /// route pre-load. The registrations must come from `register_providers` alone (no catalog
    /// help) and must be constructible with no CUDA device and no weight files.
    #[test]
    fn register_providers_alone_publishes_both_memory_routes() {
        let registry = super::provider_registry().unwrap();

        let generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(generators, [super::MODEL_ID, super::SPRINT_MODEL_ID]);

        let strategies: Vec<&str> = registry
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(
            strategies,
            [super::MODEL_ID, super::SPRINT_MODEL_ID],
            "a selector cannot price a route with no memory-strategy registration"
        );

        let fixtures: Vec<&str> = registry
            .memory_contract_fixture_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(fixtures, [super::MODEL_ID, super::SPRINT_MODEL_ID]);

        let behaviors: Vec<&str> = registry
            .memory_behavior_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        assert_eq!(behaviors, [super::MODEL_ID, super::SPRINT_MODEL_ID]);

        // The fixture contract really is weights-free: resolve it through the registered seam over
        // every published surface spec, against a path that does not exist.
        for registration in registry.memory_contract_fixture_registrations() {
            let specs = (registration.surface_specs)();
            assert!(!specs.is_empty());
            for surface in specs {
                assert_eq!(
                    surface.resolved_artifact_tier(),
                    gen_core::MemoryContractSurfaceTier::Bf16,
                    "Candle SANA is dense-only; a packed witness names a route it cannot load"
                );
                // Point the witness at a path that does not exist: resolving must not touch disk.
                let mut spec = surface.spec.clone();
                spec.weights = gen_core::WeightsSource::Dir("/nonexistent/sana/snapshot".into());
                let contract = (registration.contract)(&spec).unwrap();
                assert_eq!(contract.provider_id, registration.provider_id);
                assert!(
                    contract.conformance_errors().is_empty(),
                    "{}: {:?}",
                    registration.provider_id,
                    contract.conformance_errors()
                );
            }
        }
    }
}
