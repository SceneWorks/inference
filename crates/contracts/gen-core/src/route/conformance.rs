//! Weights-free Tier 0 conformance checks for primary [`RouteRecord`] declarations (`REG-2`, `AP5`).

use crate::feature::Pass0ConditioningRequirement;
use crate::generator::{ConditioningKind, Modality, SizeFloor};
use crate::registry::model_descriptor_errors;
use crate::{Error, Result};

use super::types::{
    AdapterInstallPolicy, ModelBinding, PreviewSupport, RenderPlanScope, RouteRecord,
};

fn is_registry_ident(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(|c| {
            c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.' | '/')
        })
}

/// Verify all Tier 0 weights-free invariants on a [`RouteRecord`], returning one message per
/// violation (empty = conformant).
///
/// Covers:
/// - route identity and every computed [`crate::ModelDescriptor`] invariant,
/// - exact parity between `compute_descriptor()` and any attached static `descriptor` fn pointer,
/// - [`crate::RouteDefaults`] coherence against [`crate::ProductLimits`] and declared menus,
/// - [`ModelBinding`] (`FamilyModel` / `FamilyDriver` / `NonDiffusion`) behavioural consistency
///   (`DriverSupport`, `geometry_change`, `boundary`, `pag`, modality compatibility),
/// - [`crate::AdapterSurface`] install-policy consistency,
/// - [`crate::PreviewSupport`], [`crate::ConditioningSurface`], [`crate::AudioSurface`],
///   [`crate::ResidencySurface`], and [`crate::RouteBindings`] `provider_id` consistency.
pub fn route_record_conformance_errors(record: &RouteRecord) -> Vec<String> {
    let mut errs = Vec::new();
    let ctx = format!("route '{}'", record.route_id);

    // 1. Computed descriptor conformance (identity, bounds, menus, conditioning, latent space, PAG).
    let computed = record.compute_descriptor();
    errs.extend(model_descriptor_errors(&computed));

    // 2. Static descriptor fn parity (prevents drift between RouteRecord and exported descriptor fns).
    if let Some(gen) = record.bindings.generator {
        let static_desc = (gen.descriptor)();
        if static_desc != computed {
            errs.push(format!(
                "{ctx}: generator binding descriptor() diverges from RouteRecord::compute_descriptor()"
            ));
        }
    }
    if let Some(feature) = record.bindings.feature_route {
        let mut static_desc = (feature.descriptor)();
        static_desc.id = record.route_id;
        if static_desc != computed {
            errs.push(format!(
                "{ctx}: feature-route binding descriptor() diverges from RouteRecord::compute_descriptor()"
            ));
        }
    }

    // 3. ModelBinding behavioural declarations (`DriverSupport`, geometry change, boundary, PAG).
    let s = &record.structure;
    match &s.denoiser {
        ModelBinding::FamilyModel(spec) => {
            if !is_registry_ident(spec.model_id) {
                errs.push(format!(
                    "{ctx}: FamilyModel model_id {:?} is not a valid registry identifier",
                    spec.model_id
                ));
            }
            if spec.supports_geometry_change && !spec.supports_renoise_from_clean {
                errs.push(format!(
                    "{ctx}: FamilyModel declares supports_geometry_change without supports_renoise_from_clean (PIPE-4 requires clean-estimate re-noising across geometry changes)"
                ));
            }
            if matches!(record.modality, Modality::Audio | Modality::Vector) {
                if spec.supports_geometry_change {
                    errs.push(format!(
                        "{ctx}: {:?} route cannot declare supports_geometry_change",
                        record.modality
                    ));
                }
                if spec.supports_multiphase_denoise {
                    errs.push(format!(
                        "{ctx}: {:?} route cannot declare supports_multiphase_denoise",
                        record.modality
                    ));
                }
                if spec.pag.is_some() {
                    errs.push(format!(
                        "{ctx}: {:?} route cannot declare PAG capability",
                        record.modality
                    ));
                }
            }
        }
        ModelBinding::FamilyDriver(spec) => {
            let support = &spec.support;
            if !is_registry_ident(support.driver_id) {
                errs.push(format!(
                    "{ctx}: FamilyDriver driver_id {:?} is not a valid registry identifier",
                    support.driver_id
                ));
            }
            if support.carry_history && !support.arbitrary_slice {
                errs.push(format!(
                    "{ctx}: FamilyDriver '{}' declares carry_history without arbitrary_slice",
                    support.driver_id
                ));
            }
            if support.geometry_change && !support.renoise_from_clean {
                errs.push(format!(
                    "{ctx}: FamilyDriver '{}' declares geometry_change without renoise_from_clean",
                    support.driver_id
                ));
            }
            let reasons = &support.refusal_reasons;
            for (name, msg) in [
                ("arbitrary_slice", reasons.arbitrary_slice),
                ("carry_history", reasons.carry_history),
                ("reset_history", reasons.reset_history),
                ("renoise_from_clean", reasons.renoise_from_clean),
                ("geometry_change", reasons.geometry_change),
                (
                    "read_only_boundary_estimate",
                    reasons.read_only_boundary_estimate,
                ),
            ] {
                if msg.trim().is_empty() {
                    errs.push(format!(
                        "{ctx}: FamilyDriver '{}' refusal_reasons.{name} must not be empty",
                        support.driver_id
                    ));
                }
            }
            if matches!(record.modality, Modality::Audio | Modality::Vector) {
                if support.geometry_change {
                    errs.push(format!(
                        "{ctx}: {:?} route cannot declare DriverSupport::geometry_change",
                        record.modality
                    ));
                }
                if spec.pag.is_some() {
                    errs.push(format!(
                        "{ctx}: {:?} route cannot declare PAG capability",
                        record.modality
                    ));
                }
            }
        }
        ModelBinding::NonDiffusion { architecture } => {
            if !is_registry_ident(architecture) {
                errs.push(format!(
                    "{ctx}: NonDiffusion architecture {:?} is not a valid registry identifier",
                    architecture
                ));
            }
        }
    }

    // 4. Adapter surface & install policy consistency.
    let has_adapter_surface = s.adapters.supports_lora || s.adapters.supports_lokr;
    if has_adapter_surface && s.adapters.install_policy == AdapterInstallPolicy::Unsupported {
        errs.push(format!(
            "{ctx}: declares supports_lora/supports_lokr but adapter install_policy is Unsupported"
        ));
    }
    if !has_adapter_surface && s.adapters.install_policy != AdapterInstallPolicy::Unsupported {
        errs.push(format!(
            "{ctx}: declares adapter install_policy {:?} while neither LoRA nor LoKr is supported",
            s.adapters.install_policy
        ));
    }

    // 5. Preview policy consistency.
    if matches!(record.modality, Modality::Audio | Modality::Vector)
        && s.output.preview != PreviewSupport::Unsupported
    {
        errs.push(format!(
            "{ctx}: {:?} route cannot advertise preview {:?}",
            record.modality, s.output.preview
        ));
    }

    // 6. Conditioning surface & pass-0 consistency.
    match s.conditioning.pass0_requirement {
        Pass0ConditioningRequirement::NoiseStartSupported => {}
        Pass0ConditioningRequirement::RequiresControl => {
            if !s.conditioning.kinds.contains(&ConditioningKind::Control) {
                errs.push(format!(
                    "{ctx}: pass0_requirement is RequiresControl but ConditioningKind::Control is not in conditioning.kinds"
                ));
            }
        }
        Pass0ConditioningRequirement::RequiresReference => {
            let has_ref = s.conditioning.kinds.iter().any(|k| {
                matches!(
                    k,
                    ConditioningKind::Reference
                        | ConditioningKind::ReferenceRgba
                        | ConditioningKind::MultiReference
                        | ConditioningKind::ReduxRefs
                )
            });
            if !has_ref {
                errs.push(format!(
                    "{ctx}: pass0_requirement is RequiresReference but no reference ConditioningKind is declared"
                ));
            }
        }
        Pass0ConditioningRequirement::RequiresEdit => {}
    }
    if s.conditioning.control_kinds.is_some()
        && !s.conditioning.kinds.contains(&ConditioningKind::Control)
    {
        errs.push(format!(
            "{ctx}: declares control_kinds without ConditioningKind::Control in conditioning.kinds"
        ));
    }

    // 7. Defaults within ProductLimits and declared surfaces.
    let defaults = &record.defaults;
    if let Some(steps) = defaults.steps {
        if steps == 0 {
            errs.push(format!("{ctx}: defaults.steps must be >= 1, got 0"));
        } else if !record.limits.supported_steps.admits(steps) {
            errs.push(format!(
                "{ctx}: defaults.steps ({steps}) is outside limits.supported_steps ({:?})",
                record.limits.supported_steps
            ));
        }
    }
    if let Some(guidance) = defaults.guidance {
        if !guidance.is_finite() {
            errs.push(format!("{ctx}: defaults.guidance ({guidance}) must be finite"));
        }
        if !s.sampling.supports_guidance {
            errs.push(format!(
                "{ctx}: declares defaults.guidance ({guidance}) while sampling.supports_guidance is false"
            ));
        }
    }
    if let Some(shift) = defaults.shift {
        if !shift.is_finite() {
            errs.push(format!("{ctx}: defaults.shift ({shift}) must be finite"));
        }
    }
    if let Some(strength) = defaults.strength {
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
            errs.push(format!(
                "{ctx}: defaults.strength ({strength}) must be finite and within 0.0..=1.0"
            ));
        }
    }
    if let Some(sampler) = defaults.sampler {
        if sampler.is_empty() || sampler.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: defaults.sampler {sampler:?} is empty or contains whitespace"
            ));
        } else if !s.sampling.samplers.is_empty() && !s.sampling.samplers.contains(&sampler) {
            errs.push(format!(
                "{ctx}: defaults.sampler {sampler:?} is not in declared sampling.samplers"
            ));
        }
    }
    if let Some(schedule) = defaults.schedule {
        if schedule.is_empty() || schedule.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: defaults.schedule {schedule:?} is empty or contains whitespace"
            ));
        } else if !s.sampling.schedulers.is_empty() && !s.sampling.schedulers.contains(&schedule) {
            errs.push(format!(
                "{ctx}: defaults.schedule {schedule:?} is not in declared sampling.schedulers"
            ));
        }
    }
    if let Some((w, h)) = defaults.dimensions {
        if matches!(record.modality, Modality::Audio | Modality::Vector) {
            errs.push(format!(
                "{ctx}: {:?} route should not declare spatial defaults.dimensions ({w}x{h})",
                record.modality
            ));
        } else {
            if w < record.limits.min_size
                || w > record.limits.max_size
                || h < record.limits.min_size
                || h > record.limits.max_size
            {
                errs.push(format!(
                    "{ctx}: defaults.dimensions ({w}x{h}) outside limits.min_size..=max_size ({}..={})",
                    record.limits.min_size, record.limits.max_size
                ));
            }
            if let Some(multiple) = record.limits.size_floor.explicit_size_multiple() {
                if multiple > 0 && (!w.is_multiple_of(multiple) || !h.is_multiple_of(multiple)) {
                    errs.push(format!(
                        "{ctx}: defaults.dimensions ({w}x{h}) are not multiples of size_floor multiple {multiple}"
                    ));
                }
                if let SizeFloor::RangeCheckedOnGridArea { max_cells, .. } =
                    record.limits.size_floor
                {
                    if let (Some(cw), Some(ch)) = (w.checked_div(multiple), h.checked_div(multiple))
                    {
                        let cells = u64::from(cw) * u64::from(ch);
                        if cells > u64::from(max_cells) {
                            errs.push(format!(
                                "{ctx}: defaults.dimensions ({w}x{h}) exceed size_floor max_cells ({max_cells})"
                            ));
                        }
                    }
                }
            }
        }
    }
    if let Some(voice) = defaults.audio_voice {
        if voice.is_empty() || voice.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: defaults.audio_voice {voice:?} is empty or contains whitespace"
            ));
        } else if !s.audio.voices.is_empty() && !s.audio.voices.contains(&voice) {
            errs.push(format!(
                "{ctx}: defaults.audio_voice {voice:?} is not in declared audio.voices"
            ));
        }
    }
    if let Some(sr) = defaults.audio_sample_rate {
        if sr == 0 {
            errs.push(format!("{ctx}: defaults.audio_sample_rate must be > 0"));
        } else if !s.audio.sample_rates.is_empty() && !s.audio.sample_rates.contains(&sr) {
            errs.push(format!(
                "{ctx}: defaults.audio_sample_rate ({sr}) is not in declared audio.sample_rates"
            ));
        }
    }
    if let Some(lang) = defaults.audio_language {
        if lang.is_empty() || lang.chars().any(char::is_whitespace) {
            errs.push(format!(
                "{ctx}: defaults.audio_language {lang:?} is empty or contains whitespace"
            ));
        } else if !s.audio.languages.is_empty() && !s.audio.languages.contains(&lang) {
            errs.push(format!(
                "{ctx}: defaults.audio_language {lang:?} is not in declared audio.languages"
            ));
        }
    }

    // 8. Audio & limit consistency.
    if let Some(max_dur) = record.limits.max_audio_duration_secs {
        if !max_dur.is_finite() || max_dur <= 0.0 {
            errs.push(format!(
                "{ctx}: limits.max_audio_duration_secs ({max_dur}) must be finite and > 0"
            ));
        }
    }
    if s.audio.max_speakers.is_some() && !s.audio.supports_multi_speaker {
        errs.push(format!(
            "{ctx}: declares audio.max_speakers while audio.supports_multi_speaker is false"
        ));
    }
    if (s.audio.supports_song_plan_only || s.audio.supports_song_cover)
        && !s.audio.supports_symbolic_song
    {
        errs.push(format!(
            "{ctx}: declares song_plan_only/song_cover while audio.supports_symbolic_song is false"
        ));
    }

    // 9. Residency lifetime graph consistency.
    for (idx, node) in s.residency.lifetime_graph.iter().enumerate() {
        if node.components.is_empty() {
            errs.push(format!(
                "{ctx}: residency.lifetime_graph[{idx}] ({:?}) has no components",
                node.phase
            ));
        }
        for comp in &node.components {
            if comp.is_empty() || comp.chars().any(char::is_whitespace) {
                errs.push(format!(
                    "{ctx}: residency.lifetime_graph[{idx}] has empty or whitespace component {comp:?}"
                ));
            }
        }
    }

    // 10. RouteBindings provider_id and render-plan scope consistency.
    if record.bindings.render_plan_scope == RenderPlanScope::MultiphaseOnly {
        if record.bindings.render_plan.is_none() {
            errs.push(format!(
                "{ctx}: render_plan_scope is MultiphaseOnly but no render_plan compiler is attached"
            ));
        }
        if !s.denoiser.supports_multiphase_denoise() {
            errs.push(format!(
                "{ctx}: render_plan_scope is MultiphaseOnly but denoiser does not declare supports_multiphase_denoise"
            ));
        }
    }
    if let Some(vp) = record.bindings.video_plan {
        if vp.provider_id != record.route_id {
            errs.push(format!(
                "{ctx}: video_plan provider_id '{}' does not match route_id",
                vp.provider_id
            ));
        }
    }
    if let Some((Some(pid), _, _)) = record.bindings.memory.contract {
        if pid != record.route_id {
            errs.push(format!(
                "{ctx}: memory contract provider_id '{pid}' does not match route_id"
            ));
        }
    }
    if let Some((Some(pid), _, _)) = record.bindings.memory.fixture {
        if pid != record.route_id {
            errs.push(format!(
                "{ctx}: memory fixture provider_id '{pid}' does not match route_id"
            ));
        }
    }
    if let Some((Some(pid), _)) = record.bindings.memory.surface_resolver {
        if pid != record.route_id {
            errs.push(format!(
                "{ctx}: memory surface resolver provider_id '{pid}' does not match route_id"
            ));
        }
    }
    if let Some((Some(pid), _, _)) = record.bindings.memory.resident_only {
        if pid != record.route_id {
            errs.push(format!(
                "{ctx}: resident-only memory witness provider_id '{pid}' does not match route_id"
            ));
        }
    }
    if let Some((Some(pid), _, _)) = record.bindings.memory.behavior {
        if pid != record.route_id {
            errs.push(format!(
                "{ctx}: memory behavior provider_id '{pid}' does not match route_id"
            ));
        }
    }
    if record.bindings.memory.fixture.is_some() && record.bindings.memory.resident_only.is_some() {
        errs.push(format!(
            "{ctx}: route bindings declare both a memory fixture and a resident-only witness"
        ));
    }
    if let Some(anchor) = record.bindings.activation_memory {
        if anchor.bytes_1024 == 0 {
            errs.push(format!(
                "{ctx}: activation_memory anchor is zero — omit unmeasured routes"
            ));
        }
    }

    errs
}

/// Fail-closed Tier 0 conformance gate for one [`RouteRecord`].
pub fn verify_route_conformance(record: &RouteRecord) -> Result<()> {
    let errs = route_record_conformance_errors(record);
    if errs.is_empty() {
        Ok(())
    } else {
        Err(Error::Msg(format!(
            "route '{}' failed Tier 0 conformance ({} violation(s)): {}",
            record.route_id,
            errs.len(),
            errs.join("; ")
        )))
    }
}
