//! The LTX-2.3 and LTX-2.5 model descriptors.

use super::*;

/// Stable identity + advertised capabilities for the LTX-2.3 AudioVideo model (produces video frames
/// + a synchronized audio track).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&mlx_gen::gen_core::LTX_VIDEO_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Distilled 2-stage path: CFG is forced to 1.0, so no guidance / negative prompt.
            // I2V single-image conditioning (sc-2685) is wired via `Reference`; audio is always
            // produced (sc-2684). Q4/Q8-of-everything is a sibling slice.
            supports_negative_prompt: false,
            // Reference = single-image I2V (sc-2685); Keyframe = first_last_frame / multi-keyframe
            // (replace-latent, epic 3040); `MultiReference` is the ordered 1–4 identity carrier
            // for replace_person (composited to the provider-native one-latent IC-LoRA input);
            // VideoClip = extend_clip / video_bridge (IC-LoRA keyframe-append — requires an
            // IC-LoRA adapter via `spec.adapters`).
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
                ConditioningKind::Keyframe,
                ConditioningKind::VideoClip,
                ConditioningKind::ControlClip,
            ],
            // LoRA (sc-2687) + LoKr (sc-2393) in generate: forward-time residuals + per-pass
            // strength over the full video+audio+cross-modal surface.
            supports_lora: true,
            supports_lokr: true,
            // Quantization is checkpoint-driven (split_model.json); load() rejects on-the-fly
            // spec.quantize that disagrees with the manifest. No on-the-fly re-quant available.
            supported_quants: &[],
            // Curated unified solvers (epic 7114, sc-7122): LTX exposes the SAMPLER axis but NO scheduler
            // (matching ComfyUI) — it keeps its baked distilled σ schedule (8+3 steps) and only swaps the
            // integrator (over the two-stream `MlxAvLatentOps`, joint video+audio). LTX is distilled, so
            // `euler` is the recommended integrator and stays the byte-exact default (unset sampler); the
            // others are exposed for parity with ComfyUI's menu. T2V only — the I2V/keyframe/clip paths
            // (per-token σ + post-step blend) stay native.
            samplers: curated_sampler_names(),
            // height/width must be divisible by SIZE_MULTIPLE (= 2×SPATIAL_SCALE; stage-1 runs at //2//32).
            supported_guidance_methods: vec![],
            min_size: MIN_SIZE,
            max_size: MAX_SIZE,
            max_count: 1,
            // sc-19502 — THE FIX for this lane. `req.steps` was never read here at all: a
            // `steps: 30` request was accepted, the knob did nothing, and the baked 8-step stage-1
            // schedule rendered anyway, while the candle lane refused the same request outright.
            // A control that is binding on one backend and silently inert on the other is the
            // sc-11993 silent-coercion class, and the silent side is the worse one.
            //
            // The distilled schedule genuinely cannot be resampled to an arbitrary count without
            // going out-of-distribution, so the honest resolution is an explicit refusal on BOTH
            // lanes — not a knob that pretends to work here. Same derived constant, same shared
            // floor, same message as candle.
            supported_steps: StepSupport::Exact(vec![crate::pipeline::NATIVE_STEPS]),
            mac_only: true,
            // Not wired onto the shared `Residency` seam (F-176); Sequential is a no-op fallback.
            supports_sequential_offload: false,
            // sc-18816: every generate builds/evaluates/drops Gemma before materializing the AvDiT,
            // then drops the AvDiT before VAE/audio decode. This is physical default behavior, not a
            // selectable Sequential control and not an evidence-composition edge.
            unconditionally_engages_staged_residency: true,
            ..Default::default()
        },
    }
}

/// Stable identity and capability surface of the LTX-2.5 MLX route.  The ordinary 2.3
/// conditioning paths are shared by the two execution shells; the advanced axes below are open
/// only because the split route binds the duration head, temporal upsampler, and alternate decoder
/// into its request execution path.
pub fn descriptor_25() -> ModelDescriptor {
    let mut out = descriptor();
    out.id = MODEL_25_ID;
    out.capabilities.supports_lora = true;
    out.capabilities.supports_lokr = false;
    out.capabilities.supports_prompt_enhancement = true;
    out.capabilities.supports_auto_duration = true;
    out.capabilities.supports_generated_keyframes = true;
    out.capabilities.max_temporal_upsample_rounds =
        mlx_gen::gen_core::ltx_dfr::MAX_TEMPORAL_UPSAMPLE_ROUNDS;
    out.capabilities.supports_diffusion_decoder = true;
    // The catalog entry spans both packed transformer variants.  The loaded provider narrows this
    // union to the exact checkpoint identity in `descriptor_25_for_variant`, so a distilled load
    // still rejects dev-only negative conditioning and a 30-step request.
    out.capabilities.supports_negative_prompt = true;
    out.capabilities.supported_steps = StepSupport::Exact(vec![crate::pipeline::NATIVE_STEPS, 30]);
    out
}

pub(super) fn descriptor_25_for_variant(variant: TransformerVariant) -> ModelDescriptor {
    let mut out = descriptor_25();
    match variant {
        TransformerVariant::Distilled => {
            out.capabilities.supports_negative_prompt = false;
            out.capabilities.supported_steps =
                StepSupport::Exact(vec![crate::pipeline::NATIVE_STEPS]);
        }
        TransformerVariant::Dev => {
            out.capabilities.supports_negative_prompt = true;
            out.capabilities.supported_steps = StepSupport::Exact(vec![30]);
            // The non-distilled checkpoint has one native 30-step guided Euler pipeline.  The
            // curated menu swaps integrators over the baked distilled waypoints and is therefore
            // not a valid dev-checkpoint control.
            out.capabilities.samplers.clear();
        }
    }
    out
}
