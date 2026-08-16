//! # mlx-gen-wan
//!
//! Wan2.2 **video** provider crate for [`mlx_gen`]. Port of the `mlx-video-with-audio` package's
//! Wan path (`generate_wan.py`, `models/wan/*`) onto Rust + `mlx-rs`.
//!
//! **First-class target:** the **Wan2.2 TI2V-5B** — the dense 5B (dim 3072, 30 layers, in/out 48)
//! with its own z48 VAE (`vae22`), delivering text-to-video (T2V) plus native image-conditioned
//! (TI2V) video (sc-2680). The shared infra here (UMT5-XXL TE, the Wan DiT, 3-axis RoPE, 3-D
//! patchify, the flow-match solvers, the T2V pipeline) is the Wan core (sc-2678); the dense/MoE
//! 14B variants reuse it via additional configs + dual-expert routing.
//!
//! This crate publishes three models through its explicit provider registry:
//! **`wan2_2_t2v_14b`** (the dual-expert MoE T2V, fully wired — the catalog loader runs the complete
//! pipeline), **`wan2_2_i2v_14b`** (the dual-expert MoE channel-concat image→video, fully wired —
//! shares the T2V pipeline with the 20-channel `y` conditioning + in_dim-36 patch-embed, sc-2681) and
//! **`wan2_2_ti2v_5b`** (the dense 5B, fully wired — sc-2680: text→video plus native image-conditioned
//! (TI2V) mask-blend video, with its own z48 [`Wan22Vae`]; Q4/Q8 + LoRA/LoKr supported).
//!
//! ## Status (S0–S6)
//! S0 — foundation: registry + config (`config.json`-driven, all Wan presets) + the three
//! flow-match solvers (Euler / DPM++2M / UniPC default) with the shifted-sigma schedule + integer
//! timesteps + 3-axis factorized 3-D RoPE (θ=10000) + 3-D patchify/unpatchify.
//! S1 — the [`Umt5Encoder`] UMT5-XXL text encoder (f32) + `_clean_text`-faithful prompt cleaning,
//! parity-gated against the `mlx_video` reference (bit-exact).
//! S2 — the [`WanVae`] Wan **2.1** VAE (z16, stride 4×8×8): 3-D causal-conv decoder + chunked
//! encoder, channel-L2 norm, per-frame spatial attention, temporal up/down `time_conv`. f32,
//! parity-gated against the reference. (The 5B's distinct z48 `vae22` is sc-2680.)
//! S3 — the [`WanTransformer`] Wan DiT (5B: 30 blocks, qk-RMSNorm self-attn + 3-axis RoPE,
//! text cross-attn, adaLN-6vec modulation, gated-GELU FFN, modulated head). f32 activations,
//! parity-gated f32-against-f32 vs the reference (patch-embed bit-exact).
//! S4 — the [`pipeline`] dense **T2V** machinery: resolution/seq-len math + the CFG denoise loop
//! (`pipeline::denoise`) + VAE decode → uint8 frames (`pipeline::decode_to_frames`). Parity-gated
//! e2e against the reference on a tiny seeded model (injected noise+context).
//! S5 — dual-expert **MoE** routing ([`pipeline::denoise_moe`] + [`Expert`]): a per-step boundary
//! swap (`t ≥ boundary·num_train`) between the high/low-noise experts, each with its own contexts,
//! cross-KV, and guidance. Parity-gated e2e on two tiny seeded experts (both fired across the
//! boundary).
//! S6 — the live `wan2_2_t2v_14b` [`mlx_gen::Generator::generate`] ([`model::Wan14b`]): the staged product
//! pipeline (UMT5 encode → two real 40-layer experts → `denoise_moe` → z16 VAE decode → RGB8
//! frames), verified end-to-end on the **real converted Wan2.2-T2V-A14B checkpoint** against a
//! `mlx_video`-reference golden on matched injected noise (`tests/s6_real_parity.rs`, `#[ignore]` —
//! the 54 GB weights live outside CI; the tiny S1–S5 gates carry CI).
//!
//! sc-2680 — the dense **TI2V-5B** ([`model::Wan`]): the z48 [`Wan22Vae`] (`vae22`: channels-last
//! causal-conv decoder/encoder, spatial 2×2 patchify, `DupUp3D`/`AvgDown3D`, RMS-L2 norm; gated by
//! `tests/vae22_parity.rs`) + the dense [`mlx_gen::Generator::generate`] — **T2V** ([`denoise`]) and
//! **image-conditioned TI2V** mask-blend ([`denoise_ti2v`] + the DiT's per-token-timestep
//! [`WanTransformer::forward_tokens`], gated by `tests/ti2v_parity.rs`). Q4/Q8 (`spec.quantize`) +
//! LoRA/LoKr merge onto the single dense model. The full e2e on the real converted 5B checkpoint is
//! `tests/ti2v_real_parity.rs` (`#[ignore]` — heavy weights outside CI).

pub mod adapters;
pub mod block_stream;
pub mod chunk;
pub mod config;
pub mod convert;
pub mod memory_strategy;
pub mod model;
pub mod model_vace;
pub mod patchify;
pub mod pipeline;
pub mod pth;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod training;
pub mod transformer;
pub mod vace;
pub mod vae;
pub mod vae22;
mod vae_common;

/// Operational Wan video ceiling: `1 + 4 * 256` pixel frames.
pub(crate) const MAX_WAN_FRAMES: usize = 1025;
/// Matching z16/z48 temporal-conditioning budget after 4x causal compression.
pub(crate) const MAX_WAN_CONDITIONING_LATENTS: usize = 257;

pub(crate) fn combined_conditioning_latents(
    control_frames: usize,
    reference_images: usize,
) -> Option<usize> {
    let control_latents = control_frames
        .checked_sub(1)?
        .checked_div(4)?
        .checked_add(1)?;
    control_latents.checked_add(reference_images)
}

pub use adapters::{
    apply_wan_adapters_additive, merge_vace_adapters, merge_vace_adapters_expert,
    merge_wan_adapters, normalize_wan_key, WanLoraReport,
};
pub use block_stream::WanBlockStream;
pub use chunk::{map_seq_chunks, slice_axis0, DitMemoryConfig};
pub use config::{GuideScale, WanModelConfig, WanQuant, WanVaceConfig, SAMPLE_NEG_PROMPT};
pub use model::{
    descriptor, descriptor_i2v_14b, descriptor_t2v_14b, load, Wan, Wan14b, MODEL_ID,
    MODEL_ID_I2V_14B, MODEL_ID_T2V_14B,
};
pub use model_vace::{
    descriptor_vace, descriptor_vace_fun, WanVace, WanVaceFun, MODEL_ID_VACE, MODEL_ID_VACE_FUN,
};
pub use pipeline::{
    build_i2v_y, build_ti2v_keyframe_z, build_ti2v_mask, decode_to_frames, decode_to_frames_22,
    denoise, denoise_moe, denoise_ti2v, frames_to_images, preprocess_i2v_image,
    preprocess_ti2v_image, ti2v_blend_init, Expert,
};
pub use rope::{rope_apply, RopeTable};
pub use scheduler::{
    compute_sigmas, make_scheduler, FlowDpmpp2m, FlowMatchEuler, FlowUniPC, SolverKind,
    WanScheduler,
};

#[cfg(test)]
mod conditioning_budget_tests {
    #[test]
    fn combined_conditioning_latents_is_checked() {
        assert_eq!(super::combined_conditioning_latents(1025, 0), Some(257));
        assert_eq!(super::combined_conditioning_latents(5, 255), Some(257));
        assert_eq!(super::combined_conditioning_latents(5, usize::MAX), None);
        assert_eq!(super::combined_conditioning_latents(0, 0), None);
    }
}
#[doc(hidden)]
pub use text_encoder::encode_text_staged_for_tier;
pub use text_encoder::{clean_text, load_tokenizer, umt5_tokenizer_config, Umt5Encoder};
pub use training::{load_trainer, WanMoeTrainer};
pub use transformer::{
    WanTransformer, WAN_BLOCK_NORM_DIFF_PATCH_TARGETS, WAN_GLOBAL_ADAPTABLE_PATHS,
};
pub const WAN_Z16_VAE_TILING: mlx_gen::tiling::VaeTiling = model::A14bProviderVae::VAE_TILING;
pub const WAN_Z48_VAE_TILING: mlx_gen::tiling::VaeTiling = model::Ti2vProviderVae::VAE_TILING;

#[cfg(test)]
mod vae_tiling_assignment_tests {
    #[test]
    fn every_wan_generator_id_resolves_to_its_concrete_decoder() {
        assert_eq!(super::WAN_Z16_VAE_TILING, mlx_gen::tiling::VaeTiling::WAN);
        assert_eq!(super::WAN_Z48_VAE_TILING, mlx_gen::tiling::VaeTiling::WAN22);
        assert_eq!(super::WAN_Z16_VAE_TILING, super::WanVae::VAE_TILING);
        assert_eq!(super::WAN_Z48_VAE_TILING, super::Wan22Vae::VAE_TILING);
        assert_eq!(
            super::model::ti2v_vae_tiling(super::MODEL_ID),
            Some(super::model::Ti2vProviderVae::VAE_TILING)
        );
        for id in [super::MODEL_ID_T2V_14B, super::MODEL_ID_I2V_14B] {
            assert_eq!(
                super::model::a14b_vae_tiling(id),
                Some(super::model::A14bProviderVae::VAE_TILING)
            );
        }
        for id in [super::MODEL_ID_VACE, super::MODEL_ID_VACE_FUN] {
            assert_eq!(
                super::model_vace::vae_tiling(id),
                Some(super::model_vace::ProviderVae::VAE_TILING)
            );
        }
        assert_eq!(
            super::vae_tiling(super::MODEL_ID),
            Some(super::WAN_Z48_VAE_TILING)
        );
        for id in [
            super::MODEL_ID_T2V_14B,
            super::MODEL_ID_I2V_14B,
            super::MODEL_ID_VACE,
            super::MODEL_ID_VACE_FUN,
        ] {
            assert_eq!(super::vae_tiling(id), Some(super::WAN_Z16_VAE_TILING));
        }
        assert_eq!(super::vae_tiling("not_wan"), None);
    }
}
pub use vace::{
    binarize_mask, build_vace_control, denoise_vace_moe, prepare_masks, prepare_video_latents,
    WanVaceTransformer,
};
pub use vae::WanVae;
pub use vae22::Wan22Vae;

/// Add all MLX Wan generators and trainers to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::TI2V_REGISTRATION)
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_FIXTURE)
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR)
        .register_generator(model::T2V_14B_REGISTRATION)
        .register_generator(model::I2V_14B_REGISTRATION)
        .register_generator(model_vace::VACE_REGISTRATION)
        .register_generator(model_vace::VACE_FUN_REGISTRATION)
        .register_trainer(training::T2V_14B_TRAINER_REGISTRATION)
        .register_trainer(training::I2V_14B_TRAINER_REGISTRATION)
        .register_trainer(training::TI2V_5B_TRAINER_REGISTRATION)
}

/// Build the complete explicit MLX Wan provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

/// Resolve the load-exact numeric tier for the one calibrated MLX Wan route. Other Wan generator
/// ids return `None` explicitly; callers must not infer their tier from `LoadSpec::quantize`.
pub fn resolved_video_memory_numeric_tier(
    provider_id: &str,
    spec: &mlx_gen::LoadSpec,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::MemoryNumericTier>> {
    if provider_id != model::MODEL_ID {
        return Ok(None);
    }
    memory_strategy::resolved_numeric_tier(spec).map(Some)
}

/// Provider-owned profile for the actual selected TI2V-5B z48 decode plan. The returned profile and
/// generation carrier share the same checked byte formula, temporal selector, and live safe budget.
pub fn selected_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
    tile_edge: u32,
    overlap: u32,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::VideoDecodeMemoryProfile>> {
    memory_strategy::selected_video_decode_memory_profile(
        provider_id,
        width,
        height,
        frames,
        tile_edge,
        overlap,
    )
}

/// Resolve the concrete MLX Wan VAE geometry used by a registered generator id.
pub fn vae_tiling(provider_id: &str) -> Option<mlx_gen::tiling::VaeTiling> {
    model::ti2v_vae_tiling(provider_id)
        .or_else(|| model::a14b_vae_tiling(provider_id))
        .or_else(|| model_vace::vae_tiling(provider_id))
}

/// Build a conservative VAE decode profile for one concrete MLX Wan geometry.
pub fn conservative_video_decode_memory_profile_for_vae(
    vae: mlx_gen::tiling::VaeTiling,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<mlx_gen::VideoDecodeMemoryProfile> {
    // The MLX z16 decode is non-causal and materializes four output frames per latent frame. A
    // request need not itself lie on that lattice, so price the full decoded allocation:
    // `4 * ceil(requested_frames / 4)`. The z48 path is causal and already prices the requested
    // output count directly.
    let decoded_frames = if vae == model::A14bProviderVae::VAE_TILING {
        let temporal_scale = vae.temporal_scale as u32;
        frames
            .checked_add(temporal_scale - 1)?
            .checked_div(temporal_scale)?
            .checked_mul(temporal_scale)?
    } else {
        frames
    };
    mlx_gen::VideoDecodeMemoryProfile::new(
        pipeline::conservative_video_decode_peak_bytes_for_vae(vae, width, height, decoded_frames)?,
        0,
    )
}

/// Resolve the provider-owned conservative VAE decode working-set peak for a Wan generator id.
/// Non-causal z16 routes price the full four-frame decode allocation enclosing the request.
pub fn conservative_video_decode_memory_profile(
    provider_id: &str,
    width: u32,
    height: u32,
    frames: u32,
) -> Option<mlx_gen::VideoDecodeMemoryProfile> {
    conservative_video_decode_memory_profile_for_vae(
        vae_tiling(provider_id)?,
        width,
        height,
        frames,
    )
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit_generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let explicit_trainers: Vec<String> = registry
            .trainers()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            explicit_generators,
            [
                "wan2_2_ti2v_5b",
                "wan2_2_t2v_14b",
                "wan2_2_i2v_14b",
                "wan_vace",
                "wan2_2_vace_fun_14b",
            ]
        );
        assert_eq!(
            explicit_trainers,
            ["wan2_2_t2v_14b", "wan2_2_i2v_14b", "wan2_2_ti2v_5b",]
        );
    }

    #[test]
    fn noncausal_z16_profiles_price_the_full_four_frame_allocation() {
        let peak = |provider_id, frames| {
            super::conservative_video_decode_memory_profile(provider_id, 64, 64, frames)
                .map(|profile| profile.working_set_bytes())
        };
        let z16 = super::MODEL_ID_T2V_14B;
        assert_eq!(peak(z16, 1), Some(107_544_576));
        assert_eq!(peak(z16, 9), Some(322_633_728));
        assert_eq!(peak(z16, 81), Some(2_258_436_096));
        assert_eq!(peak(z16, u32::MAX), None);

        // Causal z48 pricing is unchanged: requested output frames are used directly.
        assert_eq!(peak(super::MODEL_ID, 1), Some(14_090_240));
        assert_eq!(peak(super::MODEL_ID, 9), Some(126_812_160));
    }
}
