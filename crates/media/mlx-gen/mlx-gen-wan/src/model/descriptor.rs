//! The TI2V-5B, T2V-A14B and I2V-A14B model descriptors.

use super::*;

/// Stable identity + advertised capabilities for the Wan2.2 TI2V-5B (dense text+image→video).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z48_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // 5B uses real CFG (guide 5.0) with the Chinese anti-artifact negative prompt, and
            // accepts a single image as the TI2V mask-blend conditioning reference. Keyframe =
            // Wan-native first_last_frame / multi-keyframe (epic 3040, sc-3357) via the same
            // mask-blend, pinning the listed latent frames instead of only frame 0.
            supports_negative_prompt: true,
            supports_guidance: true,
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Keyframe],
            // Q4/Q8 (sc-2682) loads via `spec.quantize` (transformer-only); LoRA/LoKr merge onto the
            // single dense model at generate time (the reference `_loras_single` path — shared
            // untagged specs only, reusing the sc-2683/sc-2393 `merge_wan_adapters` seam).
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            // H/W align to patch×vae_stride = 32 (`reject_off_grid`); floor each side at MIN_SIZE = 480
            // (= 15·32 — the z48 vae22 renders garbage below a 15×15 latent grid, sc-10306/sc-12636),
            // matching candle; cap the long edge at 1280 (max_area 704×1280).
            supported_guidance_methods: vec![],
            min_size: MIN_SIZE,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            component_precision_floors: crate::memory_strategy::COMPONENT_PRECISION_FLOORS,
            // Cross-attention text K/V is cached across denoise steps.
            supports_kv_cache: true,
            // Wan pins a static `sample_shift` from config (not the empirical per-resolution mu).
            requires_sigma_shift: false,
            // Honors `OffloadPolicy::Sequential` (epic 12732, sc-12796): the dense path is already
            // staged (TE → DiT → z48 VAE, each loaded → used → dropped in turn), so Sequential adds no
            // expert swap (there are none) — it `clear_cache`-flushes the dead UMT5 TE / VAE-encoder
            // buffers off-GPU after each stage, so they don't linger in MLX's buffer cache (RSS /
            // wired-memory pressure) through denoise + decode. Advertised so the worker's fit-gate can
            // tell "bounds footprint here" from a no-op fallback.
            supports_sequential_offload: true,
            // TE → DiT → z48 VAE is phase-staged on every request; Sequential additionally flushes
            // dead allocator cache between those already-staged phases.
            unconditionally_engages_staged_residency: true,
            // The sc-18322 denoise feature cache, declared on the dense 5B alone: it is the one Wan
            // provider with a single transformer for the whole trajectory (no MoE expert swap to
            // invalidate a retained residual across) and a native one-forward-per-step loop with a step
            // index. Implemented in `crate::feature_cache`, wired into `pipeline::denoise_approx`.
            //
            // Declared is NOT selectable. No characterization artifact family exists yet, so the shared
            // floor refuses every selection against this surface — see `gen_core::approximation`. The
            // declaration is what makes the mechanism discoverable and what the terminal measurement
            // campaign narrows from these candidate intervals to the ones it can vouch for.
            approximation: approximation_surface(),
            ..Default::default()
        },
    }
}

/// Stable identity + advertised capabilities for the Wan2.2 T2V-A14B (dual-expert MoE text→video).
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID_T2V_14B,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG with the per-expert (low, high) guidance pair + the Chinese anti-artifact negative
            // prompt. Pure text→video: no image conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Cross-attention text K/V is cached across denoise steps (per expert).
            supports_kv_cache: true,
            // A14B honors `OffloadPolicy::Sequential` (epic 12732, sc-12736): the staged expert swap
            // holds only the ACTIVE MoE expert resident (never both) and frees the UMT5 TE / VAE
            // off-GPU during denoise, dropping the unified-memory peak to ~one expert. Advertised so
            // the worker's fit-gate can tell "bounds peak here" from a no-op fallback.
            supports_sequential_offload: true,
            // TE/VAE and the active expert are phase-staged even under Resident; Sequential adds the
            // stronger cache-flush/expert-residency controls described above.
            unconditionally_engages_staged_residency: true,
            ..Default::default()
        },
    }
}

/// Stable identity + advertised capabilities for the Wan2.2 I2V-A14B (dual-expert MoE image→video).
/// Identical to the T2V-A14B but advertises a single `Reference` conditioning image (the channel-
/// concat first frame) and the (3.5, 3.5) per-expert guidance.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID_I2V_14B,
        family: "wan",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            // A single image is channel-concatenated as the first-frame conditioning (in_dim 36).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: wan_samplers(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            supported_guidance_methods: vec![],
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: true,
            // A14B honors `OffloadPolicy::Sequential` (epic 12732, sc-12736): the staged expert swap
            // holds only the ACTIVE MoE expert resident (never both) and frees the UMT5 TE / VAE
            // off-GPU during denoise, dropping the unified-memory peak to ~one expert. Advertised so
            // the worker's fit-gate can tell "bounds peak here" from a no-op fallback.
            supports_sequential_offload: true,
            // TE/VAE and the active expert are phase-staged even under Resident; Sequential adds the
            // stronger cache-flush/expert-residency controls described above.
            unconditionally_engages_staged_residency: true,
            ..Default::default()
        },
    }
}
