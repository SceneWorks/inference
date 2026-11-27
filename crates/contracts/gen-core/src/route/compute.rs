//! Pure capability, descriptor, `HOOK-4` applicability structure, and JSON computation from
//! primary [`RouteRecord`] declarations (`REG-2`, `HOOK-4`, `AP5`).

use serde_json::{json, Value};

use crate::feature::{FeatureRouteStructure, Pass0ConditioningRequirement};
use crate::generator::{
    Capabilities, ConditioningKind, FamilyExtensionKind, Modality, ModelDescriptor, SizeFloor,
    StepSupport,
};
use crate::runtime::Quant;

use super::types::{RenderPlanScope, RouteRecord};

impl RouteRecord {
    /// Compute [`Capabilities`] purely from this route's primary declarations (`REG-2`, `AP5`).
    pub fn compute_capabilities(&self) -> Capabilities {
        let s = &self.structure;
        let supports_low_res_start = self.modality == Modality::Image
            && s.denoiser.supports_low_res_transition()
            && s.latent_codec.supports_spatial_unpack_repack()
            && s.conditioning.pass0_requirement == Pass0ConditioningRequirement::NoiseStartSupported;
        let allows_low_res_start_single_ref_img2img = supports_low_res_start
            && s.conditioning.pass0_requirement == Pass0ConditioningRequirement::NoiseStartSupported
            && s.conditioning.kinds.contains(&ConditioningKind::Reference)
            && !s.conditioning.low_res_start_forbids_reference_conditioning;
        let mut supported_family_extensions = s.supported_family_extensions.clone();
        if s.sampling.supports_h3_image_sampler
            && !supported_family_extensions.contains(&FamilyExtensionKind::H3ImageSampler)
        {
            supported_family_extensions.push(FamilyExtensionKind::H3ImageSampler);
        }

        Capabilities {
            supports_negative_prompt: s.conditioning.supports_negative_prompt,
            supports_guidance: s.sampling.supports_guidance,
            supports_true_cfg: s.sampling.supports_true_cfg,
            conditioning: s.conditioning.kinds.clone(),
            supports_lora: s.adapters.supports_lora,
            supports_lokr: s.adapters.supports_lokr,
            samplers: s.sampling.samplers.clone(),
            schedulers: s.sampling.schedulers.clone(),
            supported_guidance_methods: s.sampling.guidance_methods.clone(),
            min_size: self.limits.min_size,
            max_size: self.limits.max_size,
            max_count: self.seed_and_batch.max_count,
            supported_steps: self.limits.supported_steps.clone(),
            mac_only: self.mac_only,
            size_floor: self.limits.size_floor,
            audio_sample_rates: s.audio.sample_rates.clone(),
            max_audio_duration_secs: self.limits.max_audio_duration_secs,
            audio_voices: s.audio.voices.clone(),
            audio_languages: s.audio.languages.clone(),
            audio_edit_modes: s.audio.edit_modes.clone(),
            supports_preview: s.output.preview.is_supported(),
            supports_hdr: s.output.supports_hdr,
            supports_cfg_interval: s.sampling.supports_cfg_interval,
            supports_prompt_batch: self.seed_and_batch.supports_prompt_batch,
            supports_caption_overflow_policy: s.text_encoders.supports_caption_overflow_policy,
            supports_h3_image_sampler: s.sampling.supports_h3_image_sampler,
            max_prompt_tokens: s.text_encoders.max_prompt_tokens,
            max_enhance_tokens: s.text_encoders.max_enhance_tokens,
            supports_alpha_output: s.output.supports_alpha_output,
            supports_prompt_enhancement: s.text_encoders.supports_prompt_enhancement,
            supports_streaming: s.audio.supports_streaming,
            supports_multi_speaker: s.audio.supports_multi_speaker,
            max_speakers: s.audio.max_speakers,
            supports_segmented_lyrics: s.audio.supports_segmented_lyrics,
            supports_repetition_penalty: s.audio.supports_repetition_penalty,
            supports_reference_region: s.audio.supports_reference_region,
            supports_output_limiter: s.audio.supports_output_limiter,
            supports_symbolic_song: s.audio.supports_symbolic_song,
            supports_audio_artifacts: s.audio.supports_audio_artifacts,
            supports_song_plan_only: s.audio.supports_song_plan_only,
            supports_song_cover: s.audio.supports_song_cover,
            request_memory_strategies: s.residency.request_memory_strategies,
            supports_auto_duration: s.video.supports_auto_duration,
            supports_generated_keyframes: s.video.supports_generated_keyframes,
            max_temporal_upsample_rounds: s.upsamplers.max_temporal_upsample_rounds,
            supports_diffusion_decoder: s.latent_codec.supports_diffusion_decoder,
            supports_conversation_history: s.audio.supports_conversation_history,
            supports_conversation_session: s.audio.supports_conversation_session,
            supported_quants: s.weight_plan.supported_quants,
            component_precision_floors: s.weight_plan.component_precision_floors,
            supports_ar_load_controls: s.weight_plan.supports_ar_load_controls,
            supports_kv_cache: s.residency.supports_kv_cache,
            requires_sigma_shift: s.sampling.requires_sigma_shift,
            supports_sequential_offload: s.residency.supports_sequential_offload,
            unconditionally_engages_staged_residency: s
                .residency
                .unconditionally_engages_staged_residency,
            execution: s.execution.clone(),
            approximation: s.approximation.clone(),
            pag: s.denoiser.pag().cloned(),
            supports_phases: s.denoiser.supports_multiphase_denoise(),
            supports_low_res_start,
            pass0_requirement: s.conditioning.pass0_requirement,
            allows_low_res_start_single_ref_img2img,
            supported_family_extensions,
        }
    }

    /// Compute [`ModelDescriptor`] purely from this route's primary declarations (`REG-2`, `AP5`).
    pub fn compute_descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            encoder_contract: self.structure.text_encoders.encoder_contract,
            denoiser_output_latent_space: self.structure.latent_codec.output_latent_space,
            control_kinds: self.structure.conditioning.control_kinds.clone(),
            required_components: self.structure.weight_plan.required_components,
            id: self.route_id,
            family: self.family,
            backend: self.backend,
            modality: self.modality,
            capabilities: self.compute_capabilities(),
        }
    }

    /// Derive [`FeatureRouteStructure`] purely from this route's primary declarations (`HOOK-4`, `AP5`).
    ///
    /// Zero hardcoded model-ID lookup tables and zero string-matching heuristics on `route_id`:
    /// removing any structural precondition from this [`RouteRecord`] immediately flips Feature
    /// applicability and refusal reasons.
    pub fn to_feature_route_structure(&self) -> FeatureRouteStructure {
        let s = &self.structure;
        let has_render_plan = self.bindings.render_plan.is_some();
        let compiles_single_and_multi_pass = has_render_plan
            && self.bindings.render_plan_scope == RenderPlanScope::SingleAndMultiPass;
        let pixel_stride = s
            .latent_codec
            .effective_pixel_stride(self.limits.size_floor);
        let token_width = s.latent_codec.effective_token_width();
        let has_packed_image_trajectory = self.modality == Modality::Image
            && compiles_single_and_multi_pass
            && pixel_stride.is_some_and(|stride| stride > 0)
            && token_width.is_some_and(|width| width > 0);
        let supports_spatial_unpack_repack = self.modality == Modality::Image
            && has_packed_image_trajectory
            && s.latent_codec.supports_spatial_unpack_repack();
        let default_executed_steps = self.defaults.steps;
        let supports_clean_estimate_renoise_transition = s.denoiser.supports_low_res_transition()
            && compiles_single_and_multi_pass
            && supports_spatial_unpack_repack
            && default_executed_steps.is_some_and(|steps| steps >= 2);
        let supports_multiphase_denoise = s.denoiser.supports_multiphase_denoise();
        let supports_produced_reference_multipass = compiles_single_and_multi_pass
            && has_packed_image_trajectory
            && s.conditioning.kinds.contains(&ConditioningKind::Reference);
        let supports_expert_switching = s.denoiser.supports_expert_switching();
        let supports_pag = s.denoiser.pag().is_some();
        let min_size = (self.limits.min_size > 0).then_some(self.limits.min_size);

        FeatureRouteStructure {
            route_id: self.route_id.to_string(),
            family: self.family.to_string(),
            backend: self.backend.to_string(),
            modality: self.modality,
            has_render_plan,
            has_packed_image_trajectory,
            supports_spatial_unpack_repack,
            supports_clean_estimate_renoise_transition,
            pass0_conditioning: s.conditioning.pass0_requirement,
            supports_multiphase_denoise,
            supports_produced_reference_multipass,
            supports_expert_switching,
            supports_pag,
            default_executed_steps,
            pixel_stride,
            token_width,
            min_size,
        }
    }

    /// Machine-readable JSON summary of this route's primary declarations, defaults, and limits.
    pub fn to_json(&self) -> Value {
        json!({
            "route_id": self.route_id,
            "family": self.family,
            "backend": self.backend,
            "modality": modality_name(self.modality),
            "mac_only": self.mac_only,
            "binding": self.structure.denoiser,
            "defaults": self.defaults,
            "limits": {
                "min_size": self.limits.min_size,
                "max_size": self.limits.max_size,
                "size_floor": size_floor_json(self.limits.size_floor),
                "supported_steps": step_support_json(&self.limits.supported_steps),
                "max_audio_duration_secs": self.limits.max_audio_duration_secs,
            },
            "seed_and_batch": self.seed_and_batch,
            "text_encoders": self.structure.text_encoders,
            "conditioning": {
                "pass0_requirement": self.structure.conditioning.pass0_requirement,
                "supports_negative_prompt": self.structure.conditioning.supports_negative_prompt,
                "modules": self.structure.conditioning.modules,
            },
            "latent_codec": self.structure.latent_codec,
            "upsamplers": self.structure.upsamplers,
            "adapters": self.structure.adapters,
            "residency": self.structure.residency,
            "weight_plan": {
                "source_kind": self.structure.weight_plan.source_kind,
                "supported_quants": self.structure.weight_plan.supported_quants.iter().copied().map(quant_name).collect::<Vec<_>>(),
                "required_components": self.structure.weight_plan.required_components,
                "requires_control_overlay": self.structure.weight_plan.requires_control_overlay,
                "requires_ip_adapter_overlay": self.structure.weight_plan.requires_ip_adapter_overlay,
                "requires_identity_overlay": self.structure.weight_plan.requires_identity_overlay,
            },
            "output": self.structure.output,
            "has_generator": self.bindings.generator.is_some(),
            "has_render_plan": self.bindings.render_plan.is_some(),
            "render_plan_scope": self.bindings.render_plan.map(|_| self.bindings.render_plan_scope),
            "has_video_plan": self.bindings.video_plan.is_some(),
            "has_memory_contract": self.bindings.memory.contract.is_some(),
        })
    }
}

fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Image => "image",
        Modality::Video => "video",
        Modality::Both => "both",
        Modality::Audio => "audio",
        Modality::Vector => "vector",
    }
}

fn quant_name(quant: Quant) -> &'static str {
    match quant {
        Quant::Q4 => "q4",
        Quant::Q8 => "q8",
        Quant::Nvfp4 => "nvfp4",
        Quant::Fp8Rowwise => "fp8r",
    }
}

fn size_floor_json(floor: SizeFloor) -> Value {
    match floor {
        SizeFloor::RangeChecked => json!({ "kind": "range_checked" }),
        SizeFloor::RangeCheckedOnGrid { multiple } => {
            json!({ "kind": "range_checked_on_grid", "multiple": multiple })
        }
        SizeFloor::RangeCheckedOnGridArea {
            multiple,
            max_cells,
        } => json!({
            "kind": "range_checked_on_grid_area",
            "multiple": multiple,
            "max_cells": max_cells,
        }),
        SizeFloor::ResolvedDownstream => json!({ "kind": "resolved_downstream" }),
        SizeFloor::ResolvedDownstreamExplicitGrid { multiple } => json!({
            "kind": "resolved_downstream_explicit_grid",
            "multiple": multiple,
        }),
    }
}

fn step_support_json(steps: &StepSupport) -> Value {
    match steps {
        StepSupport::Unconstrained => json!({ "kind": "unconstrained" }),
        StepSupport::Exact(counts) => json!({ "kind": "exact", "values": counts }),
        StepSupport::Range { min, max } => json!({ "kind": "range", "min": min, "max": max }),
    }
}
