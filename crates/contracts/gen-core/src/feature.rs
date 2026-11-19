//! Tensor-free, object-safe Feature specification, hook-point map, structural applicability
//! derivation, and deterministic composition (SW-6938: HOOK-1..HOOK-5, E1).
//!
//! Nothing in this module touches tensors, weights, or a hardcoded model-ID capability table:
//! applicability is computed purely from a route's structural declaration ([`ModelDescriptor`]
//! plus its optional pre-load [`RenderPlanRegistration`]).

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::{
    Conditioning, ConditioningKind, ControlKind, Error, GenerationPhase, GenerationRequest,
    IdentityWeights, Image, LatentPatchLayout, LoadSpec, Modality, ModelDescriptor, OffloadPolicy,
    PhaseAdapter, ProducedReference, RenderPlan, RenderPlanRegistration, RenderStage, Result,
    WeightsSource,
};

// -------------------------------------------------------------------------------------------------
// HOOK-1: Documented hook map
// -------------------------------------------------------------------------------------------------

/// When a hook executes relative to its owning stage or block operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HookTiming {
    Before,
    Replace,
    After,
    Observe,
}

impl HookTiming {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::Replace => "replace",
            Self::After => "after",
            Self::Observe => "observe",
        }
    }
}

/// Every legal hook point in the planned image/video render lifecycle (HOOK-1).
///
/// Each variant owns a fixed stage, default timing, typed input/output contract, deterministic
/// ordering slot, and result-changing classification, and has at least one registered
/// production user in [`FeatureRegistry::standard`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HookPoint {
    ScheduleTransform,
    ConditioningSlot,
    PhaseTransition,
    GuidancePredictor,
    AttentionBlock,
}

impl HookPoint {
    pub const ALL: [HookPoint; 5] = [
        Self::ScheduleTransform,
        Self::ConditioningSlot,
        Self::PhaseTransition,
        Self::GuidancePredictor,
        Self::AttentionBlock,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ScheduleTransform => "schedule_transform",
            Self::ConditioningSlot => "conditioning_slot",
            Self::PhaseTransition => "phase_transition",
            Self::GuidancePredictor => "guidance_predictor",
            Self::AttentionBlock => "attention_block",
        }
    }

    pub const fn stage(self) -> RenderStage {
        match self {
            Self::ScheduleTransform => RenderStage::Schedule,
            Self::ConditioningSlot => RenderStage::Encode,
            Self::PhaseTransition => RenderStage::Denoise,
            Self::GuidancePredictor => RenderStage::Denoise,
            Self::AttentionBlock => RenderStage::Denoise,
        }
    }

    pub const fn default_timing(self) -> HookTiming {
        match self {
            Self::ScheduleTransform => HookTiming::After,
            Self::ConditioningSlot => HookTiming::After,
            Self::PhaseTransition => HookTiming::Before,
            Self::GuidancePredictor => HookTiming::Replace,
            Self::AttentionBlock => HookTiming::Replace,
        }
    }

    pub const fn inputs(self) -> &'static [&'static str] {
        match self {
            Self::ScheduleTransform => &["base_sigmas", "executed_steps", "resolution_mu"],
            Self::ConditioningSlot => &[
                "prompt_embeddings",
                "reference_or_control_inputs",
                "pass_index",
            ],
            Self::PhaseTransition => &[
                "phase_latents",
                "boundary_step",
                "boundary_sigma",
                "source_stream_geometry",
            ],
            Self::GuidancePredictor => &[
                "conditional_velocity",
                "unconditional_velocity",
                "guidance_scale",
                "step_index",
            ],
            Self::AttentionBlock => &[
                "block_hidden_states",
                "block_index",
                "step_index",
                "active_adapters",
            ],
        }
    }

    pub const fn outputs(self) -> &'static [&'static str] {
        match self {
            Self::ScheduleTransform => &["sigmas", "denoise_phase_spans"],
            Self::ConditioningSlot => &["bound_conditioning_slots", "produced_reference_binding"],
            Self::PhaseTransition => &[
                "next_phase_latents",
                "target_stream_geometry",
                "active_phase_binding",
            ],
            Self::GuidancePredictor => &["guided_velocity"],
            Self::AttentionBlock => &["transformed_block_hidden_states"],
        }
    }

    pub const fn ordering(self) -> u32 {
        match self {
            Self::ScheduleTransform => 10,
            Self::ConditioningSlot => 20,
            Self::PhaseTransition => 30,
            Self::AttentionBlock => 40,
            Self::GuidancePredictor => 50,
        }
    }

    pub const fn result_changing(self) -> bool {
        match self {
            Self::ScheduleTransform
            | Self::ConditioningSlot
            | Self::PhaseTransition
            | Self::GuidancePredictor
            | Self::AttentionBlock => true,
        }
    }
}

/// Complete machine-readable metadata for one [`HookPoint`] (HOOK-1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct HookPointMetadata {
    pub hook_point: HookPoint,
    pub name: &'static str,
    pub stage: RenderStage,
    pub timing: HookTiming,
    pub inputs: &'static [&'static str],
    pub outputs: &'static [&'static str],
    pub ordering: u32,
    pub result_changing: bool,
    pub registered_users: Vec<&'static str>,
}

/// Inspect one [`HookPoint`]'s metadata and its registered users in [`FeatureRegistry::standard`].
pub fn hook_point_metadata(point: HookPoint) -> HookPointMetadata {
    let registry = FeatureRegistry::standard();
    hook_point_metadata_in(point, &registry)
}

fn hook_point_metadata_in(point: HookPoint, registry: &FeatureRegistry) -> HookPointMetadata {
    let mut registered_users: Vec<&'static str> = registry
        .specs()
        .iter()
        .filter(|spec| {
            spec.hook_bindings()
                .iter()
                .any(|binding| binding.hook_point == point)
        })
        .map(|spec| spec.id())
        .collect();
    registered_users.sort_unstable();
    registered_users.dedup();
    HookPointMetadata {
        hook_point: point,
        name: point.as_str(),
        stage: point.stage(),
        timing: point.default_timing(),
        inputs: point.inputs(),
        outputs: point.outputs(),
        ordering: point.ordering(),
        result_changing: point.result_changing(),
        registered_users,
    }
}

/// Return the complete ordered hook map for the standard feature registry (HOOK-1).
pub fn hook_map() -> Vec<HookPointMetadata> {
    let registry = FeatureRegistry::standard();
    hook_map_for_registry(&registry)
}

/// Return the complete ordered hook map for `registry`.
pub fn hook_map_for_registry(registry: &FeatureRegistry) -> Vec<HookPointMetadata> {
    let mut entries: Vec<HookPointMetadata> = HookPoint::ALL
        .into_iter()
        .map(|point| hook_point_metadata_in(point, registry))
        .collect();
    entries.sort_by_key(|entry| entry.ordering);
    entries
}

/// Typed JSON dump helper for the hook map (HOOK-1; computed on demand, not a checked-in matrix).
pub fn dump_hook_map_json() -> serde_json::Value {
    serde_json::to_value(hook_map()).expect("HookPointMetadata serializes")
}

// -------------------------------------------------------------------------------------------------
// HOOK-2: FeatureSpec registry and DEC-8 classes
// -------------------------------------------------------------------------------------------------

/// DEC-8 Feature equivalence/intent classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureClass {
    /// Result-changing creative or trajectory recipe (e.g. `low_res_start`, `hires_fix`, `krea_multiphase`, `pag`).
    Recipe,
    /// Numerically or bitwise equivalent execution restructuring (e.g. `inert_phase_split`).
    Equivalent,
    /// Non-mutating observation / telemetry hook (e.g. `trajectory_observer`).
    Diagnostic,
}

impl FeatureClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Recipe => "recipe",
            Self::Equivalent => "equivalent",
            Self::Diagnostic => "diagnostic",
        }
    }
}

/// One hook binding declared by a [`FeatureSpec`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct FeatureHookBinding {
    pub hook_point: HookPoint,
    pub stage: RenderStage,
    pub timing: HookTiming,
    /// Which pass this hook targets (`Some(0)` = pass 0 only, `Some(1)` = pass 1, `None` = all passes).
    pub pass_scope: Option<usize>,
    pub result_changing: bool,
}

/// Typed parameter domain for one parameter in a [`FeatureParameterSchema`].
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FeatureParameterDomain {
    /// Step index `min..executed_steps` (`1..executed_steps` for `low_res_start.k`).
    StepIndex {
        min: usize,
        max_exclusive: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_inclusive_for_default_steps: Option<usize>,
        #[serde(skip_serializing_if = "Option::is_none")]
        default_executed_steps: Option<u32>,
    },
    /// Bounded floating-point parameter (`0.25..=0.9` with default `0.5` for `low_res_start.scale`).
    FloatRange {
        min: f32,
        max: f32,
        inclusive: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        default: Option<f32>,
    },
    /// Bounded integer parameter.
    IntegerRange {
        min: u32,
        max: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        default: Option<u32>,
    },
    /// Ordered list of denoise phases over one shared schedule.
    PhaseList {
        min_phases: usize,
        min_steps_per_phase: u32,
    },
}

/// One typed parameter specification inside [`FeatureParameterSchema`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureParameterSpec {
    pub name: &'static str,
    pub required: bool,
    pub domain: FeatureParameterDomain,
    pub description: &'static str,
}

/// Versioned parameter schema of a [`FeatureSpec`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureParameterSchema {
    pub version: u32,
    pub parameters: Vec<FeatureParameterSpec>,
}

impl FeatureParameterSchema {
    pub fn to_json_map(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("version".into(), serde_json::json!(self.version));
        for param in &self.parameters {
            let value = match &param.domain {
                FeatureParameterDomain::StepIndex {
                    min,
                    max_exclusive,
                    max_inclusive_for_default_steps,
                    default_executed_steps,
                } => {
                    let mut obj = serde_json::json!({
                        "type": "step_index",
                        "required": param.required,
                        "min": min,
                        "max_exclusive": max_exclusive,
                        "range": format!("{min}..{max_exclusive}"),
                    });
                    if let Some(max_inc) = max_inclusive_for_default_steps {
                        obj["max"] = serde_json::json!(max_inc);
                    }
                    if let Some(def_steps) = default_executed_steps {
                        obj["default_executed_steps"] = serde_json::json!(def_steps);
                    }
                    obj
                }
                FeatureParameterDomain::FloatRange {
                    min,
                    max,
                    inclusive,
                    default,
                } => {
                    let clean = |v: f32| ((f64::from(v)) * 1_000_000.0).round() / 1_000_000.0;
                    let mut obj = serde_json::json!({
                        "type": "float_range",
                        "required": param.required,
                        "min": clean(*min),
                        "max": clean(*max),
                        "inclusive": inclusive,
                    });
                    if let Some(default) = default {
                        obj["default"] = serde_json::json!(clean(*default));
                    }
                    obj
                }
                FeatureParameterDomain::IntegerRange { min, max, default } => {
                    let mut obj = serde_json::json!({
                        "type": "integer_range",
                        "required": param.required,
                        "min": min,
                        "max": max,
                    });
                    if let Some(default) = default {
                        obj["default"] = serde_json::json!(default);
                    }
                    obj
                }
                FeatureParameterDomain::PhaseList {
                    min_phases,
                    min_steps_per_phase,
                } => serde_json::json!({
                    "type": "phase_list",
                    "required": param.required,
                    "min_phases": min_phases,
                    "min_steps_per_phase": min_steps_per_phase,
                }),
            };
            map.insert(param.name.to_string(), value);
        }
        serde_json::Value::Object(map)
    }
}

/// Inter-feature and request-axis relations declared by a [`FeatureSpec`] (HOOK-2).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct FeatureRelations {
    /// Other feature IDs that cannot be composed with this feature.
    pub conflicts_with_features: Vec<&'static str>,
    /// Request/route axes that conflict with this feature (e.g., `"phases"`, `"pass0_reference_conditioning"`, `"pass0_control_conditioning"`, `"pass0_edit_conditioning"`).
    pub conflicts_with_axes: Vec<&'static str>,
    /// Other feature IDs that must also be active when this feature is selected.
    pub requires_features: Vec<&'static str>,
    /// Structural route capabilities required by this feature.
    pub requires_capabilities: Vec<&'static str>,
    /// Features that this feature must execute before during plan composition.
    pub before: Vec<&'static str>,
    /// Features that this feature must execute after during plan composition.
    pub after: Vec<&'static str>,
}

impl FeatureRelations {
    /// Combined list of conflicting features and request/conditioning axes for catalog export.
    pub fn all_conflicts(&self) -> Vec<&'static str> {
        let mut out = self.conflicts_with_axes.clone();
        for &feature in &self.conflicts_with_features {
            if !out.contains(&feature) {
                out.push(feature);
            }
        }
        out
    }
}

/// Transition residency behavior at a phase or pass boundary (HOOK-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionResidency {
    /// No transition buffer allocation beyond the running trajectory tensor.
    None,
    /// In-device latent spatial unpack -> clean-estimate resize -> repack -> re-noise at boundary sigma; zero extra model weights.
    InDeviceLatentUnpackResizeRepack,
    /// Pass-0 VAE decode to RGB pixels -> host/device resize -> pass-1 `ProducedReference` conditioning.
    PixelRoundTripProducedReference,
    /// Per-phase low-rank adapter residual swap on a job-local transformer without mutating shared base weights.
    JobLocalAdapterResidualSwap,
    /// MoE expert weight handoff at a boundary step (shedding the inactive expert under staged residency).
    StagedExpertWeightHandoff,
}

/// Memory and residency effect declared by a [`FeatureSpec`] (HOOK-2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureMemoryEffect {
    pub extra_resident_weight_bytes: u64,
    pub transition_residency: TransitionResidency,
    pub introduces_denoise_phase_boundary: bool,
    pub introduces_pass_boundary: bool,
    pub retains_single_pass_schedule: bool,
}

/// What pass 0 of a route requires on its conditioning axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pass0ConditioningRequirement {
    /// Pass 0 can start from pure noise (text-to-image).
    NoiseStartSupported,
    /// Pass 0 requires a reference/identity image (e.g. IP-Adapter or PuLID route).
    RequiresReference,
    /// Pass 0 requires a structural control image (e.g. ControlNet route).
    RequiresControl,
    /// Pass 0 requires a source edit image (e.g. instruction-edit / infill route).
    RequiresEdit,
}

/// Structural facts of a route derived weights-free from [`ModelDescriptor`] and its optional
/// [`RenderPlanRegistration`] (HOOK-2 / HOOK-4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FeatureRouteStructure {
    pub route_id: String,
    pub family: String,
    pub backend: String,
    pub modality: Modality,
    pub has_render_plan: bool,
    pub has_packed_image_trajectory: bool,
    pub supports_spatial_unpack_repack: bool,
    pub supports_clean_estimate_renoise_transition: bool,
    pub pass0_conditioning: Pass0ConditioningRequirement,
    pub supports_multiphase_denoise: bool,
    pub supports_produced_reference_multipass: bool,
    pub supports_expert_switching: bool,
    pub supports_pag: bool,
    pub default_executed_steps: Option<u32>,
    pub pixel_stride: Option<u32>,
    pub token_width: Option<u32>,
}

impl FeatureRouteStructure {
    /// Derive structural route facts from a [`ModelDescriptor`] and its optional pre-load
    /// [`RenderPlanRegistration`] without any hardcoded model-ID lookup table.
    pub fn from_descriptor_and_plan(
        descriptor: &ModelDescriptor,
        render_plan: Option<&RenderPlanRegistration>,
    ) -> Self {
        let mut has_render_plan = render_plan.is_some();
        let mut has_packed_image_trajectory = false;
        let mut pass0_conditioning = classify_descriptor_pass0_conditioning(descriptor);
        let mut default_executed_steps = None;
        let mut pixel_stride = descriptor
            .capabilities
            .size_floor
            .explicit_size_multiple()
            .or_else(|| {
                descriptor
                    .denoiser_output_latent_space
                    .map(|space| match space.patch_layout {
                        LatentPatchLayout::Packed { patch_width, .. } => {
                            u32::from(space.spatial_compression.width)
                                * u32::from(patch_width.max(1))
                        }
                        LatentPatchLayout::Unpacked => {
                            u32::from(space.spatial_compression.width) * 2
                        }
                    })
            });
        let mut token_width = None;

        if let Some(planner) = render_plan {
            let probe_size = probe_dimension(descriptor, pixel_stride.unwrap_or(16));
            let base_spec = LoadSpec::new(WeightsSource::Dir("/__feature_probe__".into()))
                .with_offload_policy(OffloadPolicy::Sequential);
            let noise_req = GenerationRequest {
                prompt: "feature applicability probe".into(),
                width: probe_size,
                height: probe_size,
                count: 1,
                seed: Some(1),
                ..Default::default()
            };

            match (planner.compile)(&base_spec, &noise_req) {
                Ok(Some(plan)) => {
                    has_render_plan = true;
                    has_packed_image_trajectory = true;
                    pass0_conditioning = Pass0ConditioningRequirement::NoiseStartSupported;
                    default_executed_steps = u32::try_from(plan.executed_steps()).ok();
                    pixel_stride = Some(plan.pixel_stride());
                    token_width = Some(plan.token_width());
                }
                Ok(None) => {
                    has_render_plan = false;
                }
                Err(_) => {
                    // Probe conditioned trajectories (Control, then Reference/Identity) to derive
                    // the exact reason pass 0 did not compile as noise-start.
                    let probe_img = Image {
                        width: 16,
                        height: 16,
                        pixels: vec![0; 16 * 16 * 3],
                    };
                    let control_spec = base_spec.clone().with_control(WeightsSource::File(
                        "/__feature_probe_control__.safetensors".into(),
                    ));
                    let control_req = GenerationRequest {
                        conditioning: vec![Conditioning::Control {
                            image: probe_img.clone(),
                            kind: ControlKind::Pose,
                            scale: Some(0.7),
                        }],
                        ..noise_req.clone()
                    };
                    if let Ok(Some(plan)) = (planner.compile)(&control_spec, &control_req) {
                        has_render_plan = true;
                        has_packed_image_trajectory = true;
                        pass0_conditioning = Pass0ConditioningRequirement::RequiresControl;
                        default_executed_steps = u32::try_from(plan.executed_steps()).ok();
                        pixel_stride = Some(plan.pixel_stride());
                        token_width = Some(plan.token_width());
                    } else {
                        let mut ref_spec = base_spec
                            .clone()
                            .with_ip_adapter(WeightsSource::File(
                                "/__feature_probe_ip__.safetensors".into(),
                            ))
                            .with_component(
                                "flux_ip_image_encoder",
                                WeightsSource::Dir("/__feature_probe_ip_encoder__".into()),
                            );
                        ref_spec.identity = Some(IdentityWeights {
                            encoder: Some(WeightsSource::File(
                                "/__feature_probe_id_encoder__.safetensors".into(),
                            )),
                            eva: Some(WeightsSource::File(
                                "/__feature_probe_eva__.safetensors".into(),
                            )),
                            face_dir: Some(WeightsSource::Dir("/__feature_probe_face__".into())),
                        });
                        let ref_req = GenerationRequest {
                            conditioning: vec![Conditioning::Reference {
                                image: probe_img,
                                strength: Some(0.7),
                            }],
                            ..noise_req
                        };
                        if let Ok(Some(plan)) = (planner.compile)(&ref_spec, &ref_req) {
                            has_render_plan = true;
                            has_packed_image_trajectory = true;
                            pass0_conditioning = Pass0ConditioningRequirement::RequiresReference;
                            default_executed_steps = u32::try_from(plan.executed_steps()).ok();
                            pixel_stride = Some(plan.pixel_stride());
                            token_width = Some(plan.token_width());
                        }
                    }
                }
            }
        }

        let has_packed_latent_space =
            descriptor
                .denoiser_output_latent_space
                .is_some_and(|space| {
                    space.is_well_formed()
                        && match space.patch_layout {
                            LatentPatchLayout::Packed {
                                patch_height: 2,
                                patch_width: 2,
                            } => true,
                            LatentPatchLayout::Unpacked => {
                                let expected_stride =
                                    u32::from(space.spatial_compression.width) * 2;
                                let expected_tokens = u32::from(space.channels) * 4;
                                pixel_stride == Some(expected_stride)
                                    && token_width == Some(expected_tokens)
                            }
                            _ => false,
                        }
                });
        let supports_spatial_unpack_repack = descriptor.modality == Modality::Image
            && has_packed_image_trajectory
            && has_packed_latent_space
            && pixel_stride.is_some_and(|stride| stride > 0)
            && token_width.is_some_and(|width| width > 0);
        let supports_clean_estimate_renoise_transition = has_render_plan
            && supports_spatial_unpack_repack
            && default_executed_steps.is_some_and(|steps| steps >= 2);
        let supports_multiphase_denoise = descriptor.capabilities.supports_phases;
        let supports_produced_reference_multipass = has_render_plan
            && has_packed_image_trajectory
            && descriptor.capabilities.accepts(ConditioningKind::Reference);
        let supports_pag = descriptor.capabilities.pag.is_some();

        Self {
            route_id: descriptor.id.to_string(),
            family: descriptor.family.to_string(),
            backend: descriptor.backend.to_string(),
            modality: descriptor.modality,
            has_render_plan,
            has_packed_image_trajectory,
            supports_spatial_unpack_repack,
            supports_clean_estimate_renoise_transition,
            pass0_conditioning,
            supports_multiphase_denoise,
            supports_produced_reference_multipass,
            supports_expert_switching: false,
            supports_pag,
            default_executed_steps,
            pixel_stride,
            token_width,
        }
    }
}

fn probe_dimension(descriptor: &ModelDescriptor, stride: u32) -> u32 {
    let stride = stride.max(1);
    let min = descriptor.capabilities.min_size.max(stride);
    let max = if descriptor.capabilities.max_size >= min {
        descriptor.capabilities.max_size
    } else {
        1024
    };
    let candidate = 512.clamp(min, max);
    let aligned = (candidate / stride) * stride;
    if aligned >= min {
        aligned
    } else {
        min.div_ceil(stride) * stride
    }
}

fn classify_descriptor_pass0_conditioning(
    descriptor: &ModelDescriptor,
) -> Pass0ConditioningRequirement {
    if descriptor.control_kinds.is_some()
        || descriptor.capabilities.conditioning == [ConditioningKind::Control]
    {
        Pass0ConditioningRequirement::RequiresControl
    } else if descriptor.id.contains("edit") || descriptor.id.contains("fill") {
        Pass0ConditioningRequirement::RequiresEdit
    } else if descriptor.family == "pulid"
        || descriptor.id.contains("pulid")
        || descriptor.id.contains("ip_adapter")
        || descriptor.id.contains("instantid")
    {
        Pass0ConditioningRequirement::RequiresReference
    } else {
        Pass0ConditioningRequirement::NoiseStartSupported
    }
}

/// Typed machine-readable code explaining why a [`FeatureSpec`] does not apply to a route (HOOK-4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureInapplicabilityCode {
    MissingRenderPlan,
    UnsupportedModality,
    MissingPackedImageTrajectory,
    MissingSpatialUnpackRepack,
    MissingCleanEstimateRenoiseTransition,
    ConflictsWithPass0ReferenceConditioning,
    ConflictsWithPass0ControlConditioning,
    ConflictsWithPass0EditConditioning,
    ConflictsWithMultiphaseDenoise,
    InsufficientExecutedSteps,
    MissingMultiphaseDenoiseSupport,
    MissingProducedReferenceSupport,
    MissingExpertSwitchingSupport,
    MissingPagCapability,
}

impl FeatureInapplicabilityCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MissingRenderPlan => "missing_render_plan",
            Self::UnsupportedModality => "unsupported_modality",
            Self::MissingPackedImageTrajectory => "missing_packed_image_trajectory",
            Self::MissingSpatialUnpackRepack => "missing_spatial_unpack_repack",
            Self::MissingCleanEstimateRenoiseTransition => {
                "missing_clean_estimate_renoise_transition"
            }
            Self::ConflictsWithPass0ReferenceConditioning => {
                "conflicts_with_pass0_reference_conditioning"
            }
            Self::ConflictsWithPass0ControlConditioning => {
                "conflicts_with_pass0_control_conditioning"
            }
            Self::ConflictsWithPass0EditConditioning => "conflicts_with_pass0_edit_conditioning",
            Self::ConflictsWithMultiphaseDenoise => "conflicts_with_multiphase_denoise",
            Self::InsufficientExecutedSteps => "insufficient_executed_steps",
            Self::MissingMultiphaseDenoiseSupport => "missing_multiphase_denoise_support",
            Self::MissingProducedReferenceSupport => "missing_produced_reference_support",
            Self::MissingExpertSwitchingSupport => "missing_expert_switching_support",
            Self::MissingPagCapability => "missing_pag_capability",
        }
    }
}

/// One typed reason a [`FeatureSpec`] is inapplicable on a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct FeatureInapplicabilityReason {
    pub code: FeatureInapplicabilityCode,
    pub message: String,
}

/// Computed applicability, classification, parameter limits/defaults, conflicts, and typed
/// inapplicability reasons for one `(route, feature)` pair (HOOK-4).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FeatureApplicability {
    pub feature_id: &'static str,
    pub version: u32,
    pub class: FeatureClass,
    pub applicable: bool,
    pub parameter_schema: FeatureParameterSchema,
    pub relations: FeatureRelations,
    pub hook_points: Vec<HookPoint>,
    pub memory_effect: FeatureMemoryEffect,
    pub reasons: Vec<FeatureInapplicabilityReason>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub conditioned_routes: BTreeMap<String, FeatureApplicability>,
}

impl FeatureApplicability {
    /// Serialize to the catalog JSON representation consumed by release/product capability checks.
    pub fn to_json(&self) -> serde_json::Value {
        let schema_json = self.parameter_schema.to_json_map();
        let parameters_json = if self.applicable {
            schema_json.clone()
        } else {
            serde_json::Value::Null
        };
        let reason_codes: Vec<&'static str> =
            self.reasons.iter().map(|reason| reason.code.as_str()).collect();
        let hook_names: Vec<&'static str> =
            self.hook_points.iter().map(|point| point.as_str()).collect();
        let mut obj = serde_json::json!({
            "id": self.feature_id,
            "version": self.version,
            "class": self.class.as_str(),
            "classification": self.class.as_str(),
            "applicable": self.applicable,
            "supported": self.applicable,
            "available": self.applicable,
            "parameter_schema": schema_json,
            "parameters": parameters_json,
            "conflicts": self.relations.all_conflicts(),
            "requires": self.relations.requires_capabilities,
            "hook_points": hook_names,
            "memory_effect": self.memory_effect,
            "reasons": self.reasons,
            "inapplicability_reasons": reason_codes,
        });
        if !self.conditioned_routes.is_empty() {
            let subroutes: serde_json::Map<String, serde_json::Value> = self
                .conditioned_routes
                .iter()
                .map(|(route_id, app)| (route_id.clone(), app.to_json()))
                .collect();
            obj["conditioned_routes"] = serde_json::Value::Object(subroutes);
        }
        obj
    }
}

// -------------------------------------------------------------------------------------------------
// HOOK-3: Composed render plan representation and deterministic Feature composition
// -------------------------------------------------------------------------------------------------

/// One transition action executed at a phase boundary inside a [`ComposedPassPlan`].
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhaseTransitionAction {
    /// Inert proof split: no geometry, schedule, guidance, or model change.
    InertSplit,
    /// `low_res_start` transition: spatial unpack -> clean-estimate x0 resize -> repack -> re-noise at `boundary_sigma`.
    CleanEstimateSpatialResizeRenoise {
        scale_bits: u32,
        from_width: u32,
        from_height: u32,
        to_width: u32,
        to_height: u32,
    },
    /// `krea_multiphase` transition: per-phase low-rank adapter subset & guidance switch on a job-local DiT.
    AdapterAndGuidanceSwitch {
        guidance_bits: u32,
        adapters: Vec<(usize, Option<u32>)>,
    },
    /// `wan_expert_switch` transition: hand off between MoE experts at the boundary step.
    ExpertSwitch {
        from_expert: String,
        to_expert: String,
    },
    /// `trajectory_observer` transition: non-mutating diagnostic observation at a phase boundary.
    DiagnosticObserve { label: String },
}

impl PhaseTransitionAction {
    fn canonical_rank(&self) -> u8 {
        match self {
            Self::CleanEstimateSpatialResizeRenoise { .. } => 1,
            Self::ExpertSwitch { .. } => 2,
            Self::AdapterAndGuidanceSwitch { .. } => 3,
            Self::InertSplit => 4,
            Self::DiagnosticObserve { .. } => 5,
        }
    }
}

/// A phase boundary at `step` in a [`ComposedPassPlan`], carrying all merged transition actions.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedPhaseTransition {
    pub step: usize,
    pub boundary_sigma: f32,
    pub actions: Vec<PhaseTransitionAction>,
}

/// One contiguous denoise phase within a [`ComposedPassPlan`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedDenoisePhase {
    pub phase_index: usize,
    pub first_step: usize,
    pub end_step: usize,
    pub width: u32,
    pub height: u32,
    pub pixel_stride: u32,
    pub packed_tokens: u32,
    pub guidance: f32,
    pub active_adapters: Vec<(usize, Option<u32>)>,
    pub active_expert: Option<String>,
}

/// One pass inside a [`ComposedRenderPlan`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedPassPlan {
    pub pass_index: usize,
    pub stages: [RenderStage; 10],
    pub sigmas: Vec<f32>,
    pub executed_steps: usize,
    pub sampler: String,
    pub target_width: u32,
    pub target_height: u32,
    pub pixel_stride: u32,
    pub token_width: u32,
    pub base_guidance: f32,
    pub produced_reference: Option<ProducedReference>,
    pub phases: Vec<ComposedDenoisePhase>,
    pub transitions: Vec<ComposedPhaseTransition>,
    pub pag_active: bool,
    pub diagnostic_observers: Vec<String>,
}

impl ComposedPassPlan {
    /// Construct a single-phase pass plan over one shared schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn single_phase(
        pass_index: usize,
        width: u32,
        height: u32,
        pixel_stride: u32,
        token_width: u32,
        sampler: impl Into<String>,
        guidance: f32,
        sigmas: Vec<f32>,
        produced_reference: Option<ProducedReference>,
    ) -> Result<Self> {
        if pixel_stride == 0
            || width == 0
            || height == 0
            || !width.is_multiple_of(pixel_stride)
            || !height.is_multiple_of(pixel_stride)
            || sigmas.len() < 2
        {
            return Err(Error::Unsupported(
                "composed pass plan has invalid geometry or schedule".into(),
            ));
        }
        let executed_steps = sigmas.len() - 1;
        let packed_tokens = (width / pixel_stride)
            .checked_mul(height / pixel_stride)
            .ok_or_else(|| Error::Unsupported("packed token count overflow".into()))?;
        Ok(Self {
            pass_index,
            stages: [
                RenderStage::Schedule,
                RenderStage::PromptPrep,
                RenderStage::Encode,
                RenderStage::Init,
                RenderStage::Enter,
                RenderStage::Denoise,
                RenderStage::Exit,
                RenderStage::PostLatent,
                RenderStage::Decode,
                RenderStage::Output,
            ],
            sigmas,
            executed_steps,
            sampler: sampler.into(),
            target_width: width,
            target_height: height,
            pixel_stride,
            token_width,
            base_guidance: guidance,
            produced_reference,
            phases: vec![ComposedDenoisePhase {
                phase_index: 0,
                first_step: 0,
                end_step: executed_steps,
                width,
                height,
                pixel_stride,
                packed_tokens,
                guidance,
                active_adapters: Vec::new(),
                active_expert: None,
            }],
            transitions: Vec::new(),
            pag_active: false,
            diagnostic_observers: Vec::new(),
        })
    }

    /// Insert or merge a phase boundary at `step` (`0 < step < executed_steps`), attaching `action`
    /// in deterministic canonical order without creating zero-step phases.
    pub fn insert_boundary_action(
        &mut self,
        step: usize,
        action: PhaseTransitionAction,
    ) -> Result<()> {
        if step == 0 || step >= self.executed_steps {
            return Err(Error::Unsupported(format!(
                "phase transition step {step} must lie in 1..{}",
                self.executed_steps
            )));
        }
        let boundary_sigma = self.sigmas[step];
        match self.transitions.binary_search_by_key(&step, |t| t.step) {
            Ok(idx) => {
                self.transitions[idx].actions.push(action);
                self.transitions[idx]
                    .actions
                    .sort_by_key(PhaseTransitionAction::canonical_rank);
            }
            Err(idx) => {
                self.transitions.insert(
                    idx,
                    ComposedPhaseTransition {
                        step,
                        boundary_sigma,
                        actions: vec![action],
                    },
                );
            }
        }
        self.rebuild_phases_from_transitions()
    }

    fn rebuild_phases_from_transitions(&mut self) -> Result<()> {
        let mut boundaries = Vec::with_capacity(self.transitions.len() + 2);
        boundaries.push(0usize);
        for transition in &self.transitions {
            boundaries.push(transition.step);
        }
        boundaries.push(self.executed_steps);

        // Determine whether a low-res spatial transition and/or expert switch or multiphase override exists.
        let low_res_transition = self.transitions.iter().find_map(|t| {
            t.actions.iter().find_map(|action| match action {
                PhaseTransitionAction::CleanEstimateSpatialResizeRenoise {
                    from_width,
                    from_height,
                    ..
                } => Some((t.step, *from_width, *from_height)),
                _ => None,
            })
        });
        let expert_transition = self.transitions.iter().find_map(|t| {
            t.actions.iter().find_map(|action| match action {
                PhaseTransitionAction::ExpertSwitch {
                    from_expert,
                    to_expert,
                } => Some((t.step, from_expert.clone(), to_expert.clone())),
                _ => None,
            })
        });

        let mut phases = Vec::with_capacity(boundaries.len() - 1);
        for (phase_index, window) in boundaries.windows(2).enumerate() {
            let (first_step, end_step) = (window[0], window[1]);
            if end_step <= first_step {
                return Err(Error::Unsupported(
                    "composed denoise phases must be non-empty and strictly increasing".into(),
                ));
            }
            let (width, height) = match low_res_transition {
                Some((split_step, low_w, low_h)) if end_step <= split_step => (low_w, low_h),
                _ => (self.target_width, self.target_height),
            };
            let packed_tokens = (width / self.pixel_stride)
                .checked_mul(height / self.pixel_stride)
                .ok_or_else(|| Error::Unsupported("packed token count overflow".into()))?;
            let active_expert = match &expert_transition {
                Some((split_step, from_expert, to_expert)) => {
                    if end_step <= *split_step {
                        Some(from_expert.clone())
                    } else {
                        Some(to_expert.clone())
                    }
                }
                None => None,
            };
            // Preserve per-phase guidance/adapters if already set for this span.
            let existing = self
                .phases
                .iter()
                .find(|p| p.first_step <= first_step && p.end_step >= end_step);
            let guidance = existing.map_or(self.base_guidance, |p| p.guidance);
            let active_adapters = existing.map_or_else(Vec::new, |p| p.active_adapters.clone());
            phases.push(ComposedDenoisePhase {
                phase_index,
                first_step,
                end_step,
                width,
                height,
                pixel_stride: self.pixel_stride,
                packed_tokens,
                guidance,
                active_adapters,
                active_expert,
            });
        }
        self.phases = phases;
        Ok(())
    }
}

/// Complete tensor-free plan produced by deterministic [`FeatureRegistry::compose`] (HOOK-3).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComposedRenderPlan {
    pub route_id: String,
    pub backend: String,
    pub request_has_phases: bool,
    pub pass0_has_conditioning: bool,
    pub resolved_feature_order: Vec<String>,
    pub passes: Vec<ComposedPassPlan>,
}

impl ComposedRenderPlan {
    /// Lift an ordinary single-pass [`RenderPlan`] into a [`ComposedRenderPlan`] ready for Feature composition.
    pub fn from_render_plan(
        route: &FeatureRouteStructure,
        plan: &RenderPlan,
        req: &GenerationRequest,
    ) -> Result<Self> {
        let (width, height) = plan.output_size();
        let pass = ComposedPassPlan::single_phase(
            plan.pass_index(),
            width,
            height,
            plan.pixel_stride(),
            plan.token_width(),
            plan.sampler(),
            plan.guidance(),
            plan.denoise_sigmas().to_vec(),
            plan.produced_reference(),
        )?;
        Ok(Self {
            route_id: route.route_id.clone(),
            backend: route.backend.clone(),
            request_has_phases: req.phases.is_some(),
            pass0_has_conditioning: !req.conditioning.is_empty() || plan.has_conditioning(),
            resolved_feature_order: Vec::new(),
            passes: vec![pass],
        })
    }
}

/// Typed parameters supplied to a [`FeatureSpec`] invocation.
#[derive(Clone, Debug, PartialEq)]
pub enum FeatureParameters {
    LowResStart {
        k: usize,
        scale: Option<f32>,
    },
    HiresFix {
        width: u32,
        height: u32,
        steps: u32,
        strength: Option<f32>,
        sigmas: Vec<f32>,
    },
    KreaMultiphase {
        phases: Vec<GenerationPhase>,
        loaded_adapter_count: usize,
    },
    WanExpertSwitch {
        boundary_step: usize,
        high_noise_expert: String,
        low_noise_expert: String,
    },
    Pag {
        scale: f32,
        layers: Vec<u32>,
    },
    InertPhaseSplit {
        step: usize,
    },
    TrajectoryObserver {
        label: String,
    },
}

/// One requested Feature activation for [`FeatureRegistry::compose`].
#[derive(Clone, Debug, PartialEq)]
pub struct FeatureInvocation {
    pub feature_id: String,
    pub parameters: FeatureParameters,
}

/// Tensor-free, object-safe Feature specification contract (HOOK-2).
pub trait FeatureSpec: Send + Sync {
    /// Stable lowercase feature identifier (e.g. `"low_res_start"`).
    fn id(&self) -> &'static str;

    /// Feature specification version (`1` for `"low_res_start"`).
    fn version(&self) -> u32;

    /// DEC-8 classification (`Recipe`, `Equivalent`, or `Diagnostic`).
    fn class(&self) -> FeatureClass;

    /// Versioned parameter schema for `route`.
    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema;

    /// Inter-feature conflicts, requirements, and ordering constraints.
    fn relations(&self) -> FeatureRelations;

    /// Hook points and stage timings bound by this feature.
    fn hook_bindings(&self) -> Vec<FeatureHookBinding>;

    /// Memory and transition-residency effect of activating this feature.
    fn memory_effect(&self) -> FeatureMemoryEffect;

    /// Structural applicability predicate over `route` (computed weights-free, never a table).
    fn applicability(&self, route: &FeatureRouteStructure) -> FeatureApplicability {
        let reasons = self.inapplicability_reasons(route);
        let hook_points = self
            .hook_bindings()
            .into_iter()
            .map(|binding| binding.hook_point)
            .collect();
        FeatureApplicability {
            feature_id: self.id(),
            version: self.version(),
            class: self.class(),
            applicable: reasons.is_empty(),
            parameter_schema: self.parameter_schema(Some(route)),
            relations: self.relations(),
            hook_points,
            memory_effect: self.memory_effect(),
            reasons,
            conditioned_routes: BTreeMap::new(),
        }
    }

    /// Compute every structural reason this feature cannot run on `route` (empty => applicable).
    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason>;

    /// Validate parameters and rewrite `plan` in place during deterministic composition (HOOK-3).
    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()>;
}

// Compile-time proof that `FeatureSpec` is object-safe (HOOK-2).
const _: Option<&dyn FeatureSpec> = None;

// -------------------------------------------------------------------------------------------------
// Registered FeatureSpec implementations
// -------------------------------------------------------------------------------------------------

/// `"low_res_start"` (`FeatureClass::Recipe`, version 1):
/// Runs steps `0..k` of pass 0 at `scale` (in `0.25..=0.9`, default `0.5`) of target width/height,
/// then transitions at `sigmas[k]` via spatial unpack -> clean-estimate resize -> repack -> re-noise
/// to full target geometry for steps `k..executed_steps` over the single pass-0 schedule.
pub struct LowResStartFeature;

impl LowResStartFeature {
    pub const ID: &'static str = "low_res_start";
    pub const VERSION: u32 = 1;
    pub const SCHEMA_VERSION: u32 = 1;
    pub const SCALE_MIN: f32 = 0.25;
    pub const SCALE_MAX: f32 = 0.9;
    pub const SCALE_DEFAULT: f32 = 0.5;
}

impl FeatureSpec for LowResStartFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        let default_steps = route.and_then(|r| r.default_executed_steps);
        let max_inc = default_steps
            .and_then(|steps| usize::try_from(steps).ok())
            .and_then(|steps| steps.checked_sub(1))
            .filter(|&max| max >= 1);
        FeatureParameterSchema {
            version: Self::SCHEMA_VERSION,
            parameters: vec![
                FeatureParameterSpec {
                    name: "k",
                    required: true,
                    domain: FeatureParameterDomain::StepIndex {
                        min: 1,
                        max_exclusive: "executed_steps",
                        max_inclusive_for_default_steps: max_inc,
                        default_executed_steps: default_steps,
                    },
                    description: "Pass-0 step index k in 1..executed_steps at which the trajectory transitions from low-res to target geometry",
                },
                FeatureParameterSpec {
                    name: "scale",
                    required: false,
                    domain: FeatureParameterDomain::FloatRange {
                        min: Self::SCALE_MIN,
                        max: Self::SCALE_MAX,
                        inclusive: true,
                        default: Some(Self::SCALE_DEFAULT),
                    },
                    description: "Low-resolution spatial scale factor in 0.25..=0.9 (default 0.5)",
                },
            ],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: vec![KreaMultiphaseFeature::ID],
            conflicts_with_axes: vec![
                "phases",
                "pass0_reference_conditioning",
                "pass0_control_conditioning",
                "pass0_edit_conditioning",
            ],
            requires_features: Vec::new(),
            requires_capabilities: vec![
                "render_plan",
                "packed_image_trajectory",
                "spatial_unpack_repack",
                "clean_estimate_renoise_transition",
                "noise_start_pass0",
            ],
            before: vec![
                WanExpertSwitchFeature::ID,
                HiresFixFeature::ID,
                PagFeature::ID,
                TrajectoryObserverFeature::ID,
            ],
            after: Vec::new(),
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::ScheduleTransform,
                stage: RenderStage::Schedule,
                timing: HookTiming::After,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Denoise,
                timing: HookTiming::Before,
                pass_scope: Some(0),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::InDeviceLatentUnpackResizeRepack,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if route.modality != Modality::Image {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::UnsupportedModality,
                message: format!(
                    "route '{}' has modality {:?}; low_res_start applies only to image trajectories",
                    route.route_id, route.modality
                ),
            });
        }
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        if !route.has_packed_image_trajectory {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingPackedImageTrajectory,
                message: format!(
                    "route '{}' does not compile a pass-0 packed image trajectory",
                    route.route_id
                ),
            });
        }
        if !route.supports_spatial_unpack_repack {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingSpatialUnpackRepack,
                message: format!(
                    "route '{}' does not declare a 2x2-packed spatial latent space supporting unpack/repack",
                    route.route_id
                ),
            });
        }
        if !route.supports_clean_estimate_renoise_transition {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingCleanEstimateRenoiseTransition,
                message: format!(
                    "route '{}' does not support clean-estimate spatial resize and re-noise phase transitions",
                    route.route_id
                ),
            });
        }
        match route.pass0_conditioning {
            Pass0ConditioningRequirement::NoiseStartSupported => {}
            Pass0ConditioningRequirement::RequiresReference => {
                reasons.push(FeatureInapplicabilityReason {
                    code: FeatureInapplicabilityCode::ConflictsWithPass0ReferenceConditioning,
                    message: format!(
                        "route '{}' requires pass-0 reference/identity conditioning, which conflicts with noise-start low_res_start",
                        route.route_id
                    ),
                });
            }
            Pass0ConditioningRequirement::RequiresControl => {
                reasons.push(FeatureInapplicabilityReason {
                    code: FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning,
                    message: format!(
                        "route '{}' requires pass-0 control conditioning, which conflicts with noise-start low_res_start",
                        route.route_id
                    ),
                });
            }
            Pass0ConditioningRequirement::RequiresEdit => {
                reasons.push(FeatureInapplicabilityReason {
                    code: FeatureInapplicabilityCode::ConflictsWithPass0EditConditioning,
                    message: format!(
                        "route '{}' requires pass-0 edit conditioning, which conflicts with noise-start low_res_start",
                        route.route_id
                    ),
                });
            }
        }
        if route.supports_multiphase_denoise {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::ConflictsWithMultiphaseDenoise,
                message: format!(
                    "route '{}' uses user-supplied multi-phase denoise (phases), which conflicts with low_res_start",
                    route.route_id
                ),
            });
        }
        if route.default_executed_steps.is_some_and(|steps| steps < 2) {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::InsufficientExecutedSteps,
                message: format!(
                    "route '{}' executes fewer than 2 steps, leaving 1..executed_steps empty",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        if plan.request_has_phases {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start conflicts with multi-phase denoise (phases)",
                route.route_id
            )));
        }
        if plan.pass0_has_conditioning {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start requires a noise-start pass 0 and conflicts with pass-0 edit/reference/control conditioning",
                route.route_id
            )));
        }
        let FeatureParameters::LowResStart { k, scale } = parameters else {
            return Err(Error::Unsupported(
                "low_res_start requires FeatureParameters::LowResStart".into(),
            ));
        };
        let scale = scale.unwrap_or(Self::SCALE_DEFAULT);
        if !scale.is_finite() || !(Self::SCALE_MIN..=Self::SCALE_MAX).contains(&scale) {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start scale {scale} must lie in {}..={}",
                route.route_id,
                Self::SCALE_MIN,
                Self::SCALE_MAX
            )));
        }
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        if *k == 0 || *k >= pass0.executed_steps {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start step k={k} must lie in 1..{}",
                route.route_id, pass0.executed_steps
            )));
        }
        let stride = pass0.pixel_stride.max(1);
        let low_w = (((pass0.target_width as f32 * scale).round() as u32) / stride) * stride;
        let low_h = (((pass0.target_height as f32 * scale).round() as u32) / stride) * stride;
        if low_w == 0 || low_h == 0 || low_w >= pass0.target_width || low_h >= pass0.target_height {
            return Err(Error::Unsupported(format!(
                "{}: low_res_start scale {scale} on {}x{} produces invalid aligned low-res geometry {low_w}x{low_h}",
                route.route_id, pass0.target_width, pass0.target_height
            )));
        }
        pass0.insert_boundary_action(
            *k,
            PhaseTransitionAction::CleanEstimateSpatialResizeRenoise {
                scale_bits: scale.to_bits(),
                from_width: low_w,
                from_height: low_h,
                to_width: pass0.target_width,
                to_height: pass0.target_height,
            },
        )
    }
}

/// `"hires_fix"` (`FeatureClass::Recipe`, version 1):
/// Appends a second pass with pass-local stage bindings and a pixel round-trip transition
/// (`ProducedReference { from_pass: 0, ... }`). Composes cleanly with `"low_res_start"` on pass 0.
pub struct HiresFixFeature;

impl HiresFixFeature {
    pub const ID: &'static str = "hires_fix";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for HiresFixFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: vec![
                FeatureParameterSpec {
                    name: "strength",
                    required: false,
                    domain: FeatureParameterDomain::FloatRange {
                        min: 0.0,
                        max: 1.0,
                        inclusive: true,
                        default: Some(0.3),
                    },
                    description: "Pass-1 produced-reference denoise strength",
                },
                FeatureParameterSpec {
                    name: "steps",
                    required: true,
                    domain: FeatureParameterDomain::IntegerRange {
                        min: 1,
                        max: 1000,
                        default: Some(12),
                    },
                    description: "Pass-1 executed denoise steps",
                },
            ],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec![
                "render_plan",
                "packed_image_trajectory",
                "produced_reference_multipass",
            ],
            before: vec![TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID, WanExpertSwitchFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::ConditioningSlot,
                stage: RenderStage::Encode,
                timing: HookTiming::After,
                pass_scope: Some(1),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Decode,
                timing: HookTiming::After,
                pass_scope: Some(1),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::PixelRoundTripProducedReference,
            introduces_denoise_phase_boundary: false,
            introduces_pass_boundary: true,
            retains_single_pass_schedule: false,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        if !route.supports_produced_reference_multipass {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingProducedReferenceSupport,
                message: format!(
                    "route '{}' does not accept a ProducedReference pass-1 input",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::HiresFix {
            width,
            height,
            steps,
            strength,
            sigmas,
        } = parameters
        else {
            return Err(Error::Unsupported(
                "hires_fix requires FeatureParameters::HiresFix".into(),
            ));
        };
        if *steps == 0 || sigmas.len() != (*steps as usize) + 1 {
            return Err(Error::Unsupported(format!(
                "{}: hires_fix pass-1 sigmas length {} must equal steps ({steps}) + 1",
                route.route_id,
                sigmas.len()
            )));
        }
        let pass0 = plan
            .passes
            .first()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        let pass_index = plan.passes.len();
        let produced = ProducedReference {
            from_pass: 0,
            width: *width,
            height: *height,
            strength: *strength,
        };
        let pass1 = ComposedPassPlan::single_phase(
            pass_index,
            *width,
            *height,
            pass0.pixel_stride,
            pass0.token_width,
            pass0.sampler.clone(),
            pass0.base_guidance,
            sigmas.clone(),
            Some(produced),
        )?;
        plan.passes.push(pass1);
        Ok(())
    }
}

/// `"krea_multiphase"` (`FeatureClass::Recipe`, version 1):
/// Slices one coherent global schedule into contiguous denoise phases with per-phase guidance and
/// active low-rank adapter subsets. Conflicts with `"low_res_start"`.
pub struct KreaMultiphaseFeature;

impl KreaMultiphaseFeature {
    pub const ID: &'static str = "krea_multiphase";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for KreaMultiphaseFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "phases",
                required: true,
                domain: FeatureParameterDomain::PhaseList {
                    min_phases: 1,
                    min_steps_per_phase: 1,
                },
                description: "Ordered denoise phases partitioning the single global schedule",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: vec![LowResStartFeature::ID, InertPhaseSplitFeature::ID],
            conflicts_with_axes: vec![
                "pass0_reference_conditioning",
                "pass0_control_conditioning",
                "pass0_edit_conditioning",
            ],
            requires_features: Vec::new(),
            requires_capabilities: vec!["multiphase_denoise"],
            before: vec![HiresFixFeature::ID, TrajectoryObserverFeature::ID],
            after: Vec::new(),
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::ScheduleTransform,
                stage: RenderStage::Schedule,
                timing: HookTiming::Replace,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Denoise,
                timing: HookTiming::Before,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::GuidancePredictor,
                stage: RenderStage::Denoise,
                timing: HookTiming::Replace,
                pass_scope: Some(0),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::JobLocalAdapterResidualSwap,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.supports_multiphase_denoise {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingMultiphaseDenoiseSupport,
                message: format!(
                    "route '{}' does not support multi-phase denoise (phases)",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::KreaMultiphase {
            phases,
            loaded_adapter_count,
        } = parameters
        else {
            return Err(Error::Unsupported(
                "krea_multiphase requires FeatureParameters::KreaMultiphase".into(),
            ));
        };
        if phases.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: krea_multiphase requires at least one phase",
                route.route_id
            )));
        }
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        let total_steps: usize = phases.iter().map(|p| p.steps as usize).sum();
        if total_steps != pass0.executed_steps {
            return Err(Error::Unsupported(format!(
                "{}: krea_multiphase total steps {total_steps} does not match pass-0 schedule executed_steps {}",
                route.route_id, pass0.executed_steps
            )));
        }
        let mut cursor = 0usize;
        let mut composed_phases = Vec::with_capacity(phases.len());
        let packed_tokens = (pass0.target_width / pass0.pixel_stride)
            * (pass0.target_height / pass0.pixel_stride);
        for (idx, phase) in phases.iter().enumerate() {
            if phase.steps == 0 {
                return Err(Error::Unsupported(format!(
                    "{}: krea_multiphase phase {idx} has 0 steps",
                    route.route_id
                )));
            }
            let mut adapters = Vec::with_capacity(phase.adapters.len());
            for PhaseAdapter { adapter, weight } in &phase.adapters {
                if *adapter >= *loaded_adapter_count {
                    return Err(Error::Unsupported(format!(
                        "{}: phase {idx} adapter index {adapter} exceeds loaded_adapter_count {loaded_adapter_count}",
                        route.route_id
                    )));
                }
                adapters.push((*adapter, weight.map(f32::to_bits)));
            }
            let first_step = cursor;
            let end_step = cursor + phase.steps as usize;
            let guidance = phase.guidance.unwrap_or(pass0.base_guidance);
            if first_step > 0 {
                pass0.transitions.push(ComposedPhaseTransition {
                    step: first_step,
                    boundary_sigma: pass0.sigmas[first_step],
                    actions: vec![PhaseTransitionAction::AdapterAndGuidanceSwitch {
                        guidance_bits: guidance.to_bits(),
                        adapters: adapters.clone(),
                    }],
                });
            }
            composed_phases.push(ComposedDenoisePhase {
                phase_index: idx,
                first_step,
                end_step,
                width: pass0.target_width,
                height: pass0.target_height,
                pixel_stride: pass0.pixel_stride,
                packed_tokens,
                guidance,
                active_adapters: adapters,
                active_expert: None,
            });
            cursor = end_step;
        }
        pass0.phases = composed_phases;
        Ok(())
    }
}

/// `"wan_expert_switch"` (`FeatureClass::Recipe`, version 1):
/// Switches between MoE experts (`high_noise` -> `low_noise`) at a schedule boundary step.
/// Composes with `"low_res_start"` even when both target the exact same boundary step `k` (HOOK-3).
pub struct WanExpertSwitchFeature;

impl WanExpertSwitchFeature {
    pub const ID: &'static str = "wan_expert_switch";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for WanExpertSwitchFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        let default_steps = route.and_then(|r| r.default_executed_steps);
        let max_inc = default_steps
            .and_then(|steps| usize::try_from(steps).ok())
            .and_then(|steps| steps.checked_sub(1))
            .filter(|&max| max >= 1);
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "boundary_step",
                required: true,
                domain: FeatureParameterDomain::StepIndex {
                    min: 1,
                    max_exclusive: "executed_steps",
                    max_inclusive_for_default_steps: max_inc,
                    default_executed_steps: default_steps,
                },
                description: "Step index in 1..executed_steps at which the active MoE expert switches",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["expert_switching"],
            before: vec![HiresFixFeature::ID, TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![FeatureHookBinding {
            hook_point: HookPoint::PhaseTransition,
            stage: RenderStage::Denoise,
            timing: HookTiming::Before,
            pass_scope: Some(0),
            result_changing: true,
        }]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::StagedExpertWeightHandoff,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.supports_expert_switching {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingExpertSwitchingSupport,
                message: format!(
                    "route '{}' does not use dual-expert MoE boundary switching",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::WanExpertSwitch {
            boundary_step,
            high_noise_expert,
            low_noise_expert,
        } = parameters
        else {
            return Err(Error::Unsupported(
                "wan_expert_switch requires FeatureParameters::WanExpertSwitch".into(),
            ));
        };
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        if *boundary_step == 0 || *boundary_step >= pass0.executed_steps {
            return Err(Error::Unsupported(format!(
                "{}: wan_expert_switch boundary_step {boundary_step} must lie in 1..{}",
                route.route_id, pass0.executed_steps
            )));
        }
        pass0.insert_boundary_action(
            *boundary_step,
            PhaseTransitionAction::ExpertSwitch {
                from_expert: high_noise_expert.clone(),
                to_expert: low_noise_expert.clone(),
            },
        )
    }
}

/// `"pag"` (`FeatureClass::Recipe`, version 1):
/// Perturbed-Attention Guidance binding `HookPoint::AttentionBlock` and `HookPoint::GuidancePredictor`.
pub struct PagFeature;

impl PagFeature {
    pub const ID: &'static str = "pag";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for PagFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Recipe
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "scale",
                required: true,
                domain: FeatureParameterDomain::FloatRange {
                    min: 0.0,
                    max: crate::DEFAULT_PAG_MAX_SCALE,
                    inclusive: true,
                    default: Some(1.5),
                },
                description: "Perturbed-Attention Guidance scale",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["pag"],
            before: vec![TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::AttentionBlock,
                stage: RenderStage::Denoise,
                timing: HookTiming::Replace,
                pass_scope: Some(0),
                result_changing: true,
            },
            FeatureHookBinding {
                hook_point: HookPoint::GuidancePredictor,
                stage: RenderStage::Denoise,
                timing: HookTiming::After,
                pass_scope: Some(0),
                result_changing: true,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::None,
            introduces_denoise_phase_boundary: false,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.supports_pag {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingPagCapability,
                message: format!(
                    "route '{}' does not advertise Perturbed-Attention Guidance (PAG)",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::Pag { scale, layers } = parameters else {
            return Err(Error::Unsupported(
                "pag requires FeatureParameters::Pag".into(),
            ));
        };
        if !scale.is_finite() || *scale <= 0.0 || layers.is_empty() {
            return Err(Error::Unsupported(format!(
                "{}: pag requires positive finite scale and non-empty layers",
                route.route_id
            )));
        }
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        pass0.pag_active = true;
        Ok(())
    }
}

/// `"inert_phase_split"` (`FeatureClass::Equivalent`, version 1):
/// Proof-only result-equivalent phase boundary split over a single schedule.
pub struct InertPhaseSplitFeature;

impl InertPhaseSplitFeature {
    pub const ID: &'static str = "inert_phase_split";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for InertPhaseSplitFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Equivalent
    }

    fn parameter_schema(&self, route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        let default_steps = route.and_then(|r| r.default_executed_steps);
        let max_inc = default_steps
            .and_then(|steps| usize::try_from(steps).ok())
            .and_then(|steps| steps.checked_sub(1))
            .filter(|&max| max >= 1);
        FeatureParameterSchema {
            version: 1,
            parameters: vec![FeatureParameterSpec {
                name: "step",
                required: true,
                domain: FeatureParameterDomain::StepIndex {
                    min: 1,
                    max_exclusive: "executed_steps",
                    max_inclusive_for_default_steps: max_inc,
                    default_executed_steps: default_steps,
                },
                description: "Inert phase split step in 1..executed_steps",
            }],
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: vec![KreaMultiphaseFeature::ID],
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["render_plan", "packed_image_trajectory"],
            before: vec![HiresFixFeature::ID, TrajectoryObserverFeature::ID],
            after: vec![LowResStartFeature::ID, WanExpertSwitchFeature::ID],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![FeatureHookBinding {
            hook_point: HookPoint::PhaseTransition,
            stage: RenderStage::Denoise,
            timing: HookTiming::Observe,
            pass_scope: Some(0),
            result_changing: false,
        }]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::None,
            introduces_denoise_phase_boundary: true,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        if !route.has_packed_image_trajectory {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingPackedImageTrajectory,
                message: format!(
                    "route '{}' does not compile a packed image trajectory",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::InertPhaseSplit { step } = parameters else {
            return Err(Error::Unsupported(
                "inert_phase_split requires FeatureParameters::InertPhaseSplit".into(),
            ));
        };
        let pass0 = plan
            .passes
            .first_mut()
            .ok_or_else(|| Error::Unsupported("composed plan has no pass 0".into()))?;
        if *step == 0 || *step >= pass0.executed_steps {
            return Err(Error::Unsupported(format!(
                "{}: inert_phase_split step {step} must lie in 1..{}",
                route.route_id, pass0.executed_steps
            )));
        }
        pass0.insert_boundary_action(*step, PhaseTransitionAction::InertSplit)
    }
}

/// `"trajectory_observer"` (`FeatureClass::Diagnostic`, version 1):
/// Non-mutating diagnostic observer on `HookPoint::AttentionBlock` and `HookPoint::PhaseTransition`.
pub struct TrajectoryObserverFeature;

impl TrajectoryObserverFeature {
    pub const ID: &'static str = "trajectory_observer";
    pub const VERSION: u32 = 1;
}

impl FeatureSpec for TrajectoryObserverFeature {
    fn id(&self) -> &'static str {
        Self::ID
    }

    fn version(&self) -> u32 {
        Self::VERSION
    }

    fn class(&self) -> FeatureClass {
        FeatureClass::Diagnostic
    }

    fn parameter_schema(&self, _route: Option<&FeatureRouteStructure>) -> FeatureParameterSchema {
        FeatureParameterSchema {
            version: 1,
            parameters: Vec::new(),
        }
    }

    fn relations(&self) -> FeatureRelations {
        FeatureRelations {
            conflicts_with_features: Vec::new(),
            conflicts_with_axes: Vec::new(),
            requires_features: Vec::new(),
            requires_capabilities: vec!["render_plan"],
            before: Vec::new(),
            after: vec![
                LowResStartFeature::ID,
                WanExpertSwitchFeature::ID,
                KreaMultiphaseFeature::ID,
                HiresFixFeature::ID,
                PagFeature::ID,
                InertPhaseSplitFeature::ID,
            ],
        }
    }

    fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
        vec![
            FeatureHookBinding {
                hook_point: HookPoint::AttentionBlock,
                stage: RenderStage::Denoise,
                timing: HookTiming::Observe,
                pass_scope: None,
                result_changing: false,
            },
            FeatureHookBinding {
                hook_point: HookPoint::PhaseTransition,
                stage: RenderStage::Denoise,
                timing: HookTiming::Observe,
                pass_scope: None,
                result_changing: false,
            },
        ]
    }

    fn memory_effect(&self) -> FeatureMemoryEffect {
        FeatureMemoryEffect {
            extra_resident_weight_bytes: 0,
            transition_residency: TransitionResidency::None,
            introduces_denoise_phase_boundary: false,
            introduces_pass_boundary: false,
            retains_single_pass_schedule: true,
        }
    }

    fn inapplicability_reasons(
        &self,
        route: &FeatureRouteStructure,
    ) -> Vec<FeatureInapplicabilityReason> {
        let mut reasons = Vec::new();
        if !route.has_render_plan {
            reasons.push(FeatureInapplicabilityReason {
                code: FeatureInapplicabilityCode::MissingRenderPlan,
                message: format!(
                    "route '{}' does not register a pre-load RenderPlan",
                    route.route_id
                ),
            });
        }
        reasons
    }

    fn rewrite_plan(
        &self,
        _route: &FeatureRouteStructure,
        plan: &mut ComposedRenderPlan,
        parameters: &FeatureParameters,
    ) -> Result<()> {
        let FeatureParameters::TrajectoryObserver { label } = parameters else {
            return Err(Error::Unsupported(
                "trajectory_observer requires FeatureParameters::TrajectoryObserver".into(),
            ));
        };
        for pass in &mut plan.passes {
            pass.diagnostic_observers.push(label.clone());
            pass.diagnostic_observers.sort();
            pass.diagnostic_observers.dedup();
        }
        Ok(())
    }
}

// -------------------------------------------------------------------------------------------------
// FeatureRegistry: object-safe registry, applicability evaluation, and deterministic composition
// -------------------------------------------------------------------------------------------------

/// Registry of object-safe [`FeatureSpec`]s (HOOK-2 / HOOK-3).
pub struct FeatureRegistry {
    specs: Vec<Box<dyn FeatureSpec>>,
}

impl Default for FeatureRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

impl FeatureRegistry {
    /// Construct the standard production Feature registry.
    pub fn standard() -> Self {
        Self {
            specs: vec![
                Box::new(LowResStartFeature),
                Box::new(HiresFixFeature),
                Box::new(KreaMultiphaseFeature),
                Box::new(WanExpertSwitchFeature),
                Box::new(PagFeature),
                Box::new(InertPhaseSplitFeature),
                Box::new(TrajectoryObserverFeature),
            ],
        }
    }

    /// Construct a registry from an explicit list of boxed [`FeatureSpec`]s (used to prove
    /// registration-order invariance and test synthetic cycle/conflict configurations).
    pub fn with_specs(specs: Vec<Box<dyn FeatureSpec>>) -> Result<Self> {
        let mut seen = BTreeSet::new();
        for spec in &specs {
            if !seen.insert(spec.id()) {
                return Err(Error::Unsupported(format!(
                    "duplicate FeatureSpec id '{}' in FeatureRegistry",
                    spec.id()
                )));
            }
        }
        Ok(Self { specs })
    }

    /// Every registered [`FeatureSpec`] in this registry.
    pub fn specs(&self) -> &[Box<dyn FeatureSpec>] {
        &self.specs
    }

    /// Look up a registered [`FeatureSpec`] by id.
    pub fn get(&self, id: &str) -> Option<&dyn FeatureSpec> {
        self.specs
            .iter()
            .find(|spec| spec.id() == id)
            .map(Box::as_ref)
    }

    /// Compute applicability for every registered [`FeatureSpec`] on `route`, keyed by feature id.
    pub fn evaluate_route(
        &self,
        route: &FeatureRouteStructure,
    ) -> BTreeMap<String, FeatureApplicability> {
        let mut map = BTreeMap::new();
        for spec in &self.specs {
            map.insert(spec.id().to_string(), spec.applicability(route));
        }
        map
    }

    /// Validate and compose an active set of [`FeatureInvocation`]s onto `base_plan` (HOOK-3).
    ///
    /// Guarantees:
    /// - Whole-active-set applicability, requirement, conflict, exclusive-`Replace`-hook ownership,
    ///   and cycle validation run before any rewrite.
    /// - Topological ordering breaks ties by canonical `(earliest_hook_ordering, feature_id)`, never
    ///   by registration order or caller invocation order.
    pub fn compose(
        &self,
        route: &FeatureRouteStructure,
        mut base_plan: ComposedRenderPlan,
        invocations: &[FeatureInvocation],
    ) -> Result<ComposedRenderPlan> {
        // 1. Resolve all invoked specs and reject unknown or duplicate feature IDs.
        let mut active_by_id: BTreeMap<&'static str, (&dyn FeatureSpec, &FeatureInvocation)> =
            BTreeMap::new();
        for invocation in invocations {
            let spec = self.get(&invocation.feature_id).ok_or_else(|| {
                Error::Unsupported(format!(
                    "unknown feature '{}' requested on route '{}'",
                    invocation.feature_id, route.route_id
                ))
            })?;
            if active_by_id.insert(spec.id(), (spec, invocation)).is_some() {
                return Err(Error::Unsupported(format!(
                    "duplicate feature '{}' in active set on route '{}'",
                    spec.id(),
                    route.route_id
                )));
            }
        }

        // 2. Per-feature route applicability check.
        for (&id, &(spec, _)) in &active_by_id {
            let applicability = spec.applicability(route);
            if !applicability.applicable {
                let reasons: Vec<String> = applicability
                    .reasons
                    .iter()
                    .map(|r| format!("{}: {}", r.code.as_str(), r.message))
                    .collect();
                return Err(Error::Unsupported(format!(
                    "feature '{id}' is not applicable to route '{}': {}",
                    route.route_id,
                    reasons.join("; ")
                )));
            }
        }

        // 3. Whole-active-set requirements and symmetric conflicts check.
        let active_ids: Vec<&'static str> = active_by_id.keys().copied().collect();
        for &id in &active_ids {
            let (spec, _) = active_by_id[id];
            let relations = spec.relations();
            for &required in &relations.requires_features {
                if !active_by_id.contains_key(required) {
                    return Err(Error::Unsupported(format!(
                        "feature '{id}' requires feature '{required}', which is not in the active set"
                    )));
                }
            }
            for &conflict in &relations.conflicts_with_features {
                if active_by_id.contains_key(conflict) {
                    let (a, b) = if id <= conflict {
                        (id, conflict)
                    } else {
                        (conflict, id)
                    };
                    return Err(Error::Unsupported(format!(
                        "features '{a}' and '{b}' conflict and cannot be composed together"
                    )));
                }
            }
        }

        // 4. Whole-active-set exclusive `HookTiming::Replace` ownership check on `(pass_scope, hook_point)`.
        let mut replace_owners: BTreeMap<(Option<usize>, HookPoint), &'static str> =
            BTreeMap::new();
        for &id in &active_ids {
            let (spec, _) = active_by_id[id];
            for binding in spec.hook_bindings() {
                if binding.timing == HookTiming::Replace {
                    let key = (binding.pass_scope, binding.hook_point);
                    if let Some(&existing_owner) = replace_owners.get(&key) {
                        if existing_owner != id {
                            return Err(Error::Unsupported(format!(
                                "hook ownership conflict on {:?} (pass {:?}): both '{}' and '{}' claim Replace timing",
                                binding.hook_point, binding.pass_scope, existing_owner, id
                            )));
                        }
                    } else {
                        replace_owners.insert(key, id);
                    }
                }
            }
        }

        // 5. Build ordering graph over the active set and run deterministic Kahn topological sort.
        let mut edges: BTreeMap<&'static str, BTreeSet<&'static str>> = active_ids
            .iter()
            .copied()
            .map(|id| (id, BTreeSet::new()))
            .collect();
        let mut in_degree: BTreeMap<&'static str, usize> =
            active_ids.iter().copied().map(|id| (id, 0usize)).collect();

        for &id in &active_ids {
            let (spec, _) = active_by_id[id];
            let relations = spec.relations();
            for &target in &relations.before {
                if active_by_id.contains_key(target) {
                    if target == id {
                        return Err(Error::Unsupported(format!(
                            "feature '{id}' declares a self-cycle in `before`"
                        )));
                    }
                    if edges.get_mut(id).unwrap().insert(target) {
                        *in_degree.get_mut(target).unwrap() += 1;
                    }
                }
            }
            for &source in &relations.after {
                if active_by_id.contains_key(source) {
                    if source == id {
                        return Err(Error::Unsupported(format!(
                            "feature '{id}' declares a self-cycle in `after`"
                        )));
                    }
                    if edges.get_mut(source).unwrap().insert(id) {
                        *in_degree.get_mut(id).unwrap() += 1;
                    }
                }
            }
        }

        let priority_key = |id: &'static str| -> (u32, &'static str) {
            let (spec, _) = active_by_id[id];
            let min_hook_order = spec
                .hook_bindings()
                .iter()
                .map(|b| b.hook_point.ordering())
                .min()
                .unwrap_or(u32::MAX);
            (min_hook_order, id)
        };

        let mut ready: BTreeSet<(u32, &'static str)> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&id, _)| priority_key(id))
            .collect();

        let mut ordered_ids = Vec::with_capacity(active_ids.len());
        while let Some((_, current)) = ready.pop_first() {
            ordered_ids.push(current);
            for &neighbor in &edges[current] {
                let deg = in_degree.get_mut(neighbor).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    ready.insert(priority_key(neighbor));
                }
            }
        }

        if ordered_ids.len() != active_ids.len() {
            let cycle_members: Vec<&'static str> = in_degree
                .into_iter()
                .filter_map(|(id, deg)| (deg > 0).then_some(id))
                .collect();
            return Err(Error::Unsupported(format!(
                "feature ordering cycle detected among active set: {:?}",
                cycle_members
            )));
        }

        // 6. Rewrite plan in canonical topological order.
        for id in ordered_ids {
            let (spec, invocation) = active_by_id[id];
            spec.rewrite_plan(route, &mut base_plan, &invocation.parameters)?;
            base_plan.resolved_feature_order.push(id.to_string());
        }

        Ok(base_plan)
    }
}

/// Validate a request attempting to use `"low_res_start"` on `descriptor` / `render_plan` (HOOK-5).
///
/// Refuses with a typed [`Error::Unsupported`] when:
/// - the route is structurally inapplicable (e.g. `flux1_dev_control`, `flux1_dev_ip_adapter`,
///   `flux1_schnell_ip_adapter`, `flux1_dev_pulid` / `pulid_flux`, or an unmigrated route);
/// - `req.phases` is `Some(...)`;
/// - `req.conditioning` is non-empty on pass 0 (reference, control, or edit conditioning);
/// - `k` is outside `1..executed_steps` or `scale` is outside `0.25..=0.9`.
pub fn validate_low_res_start_request(
    descriptor: &ModelDescriptor,
    render_plan: Option<&RenderPlanRegistration>,
    req: &GenerationRequest,
    executed_steps: usize,
    k: usize,
    scale: Option<f32>,
) -> Result<ComposedRenderPlan> {
    if req.phases.is_some() {
        return Err(Error::Unsupported(format!(
            "{}: low_res_start conflicts with multi-phase denoise (phases)",
            descriptor.id
        )));
    }
    if !req.conditioning.is_empty() {
        return Err(Error::Unsupported(format!(
            "{}: low_res_start requires a noise-start pass and conflicts with pass-0 conditioning",
            descriptor.id
        )));
    }
    let route = FeatureRouteStructure::from_descriptor_and_plan(descriptor, render_plan);
    let stride = route.pixel_stride.unwrap_or(16);
    let token_width = route.token_width.unwrap_or(64);
    let sigmas: Vec<f32> = (0..=executed_steps)
        .map(|i| 1.0 - (i as f32 / executed_steps.max(1) as f32))
        .collect();
    let pass0 = ComposedPassPlan::single_phase(
        0,
        req.width,
        req.height,
        stride,
        token_width,
        req.sampler.as_deref().unwrap_or("euler"),
        req.guidance.unwrap_or(3.5),
        sigmas,
        None,
    )?;
    let base_plan = ComposedRenderPlan {
        route_id: route.route_id.clone(),
        backend: route.backend.clone(),
        request_has_phases: req.phases.is_some(),
        pass0_has_conditioning: !req.conditioning.is_empty(),
        resolved_feature_order: Vec::new(),
        passes: vec![pass0],
    };
    FeatureRegistry::standard().compose(
        &route,
        base_plan,
        &[FeatureInvocation {
            feature_id: LowResStartFeature::ID.to_string(),
            parameters: FeatureParameters::LowResStart { k, scale },
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AcceptedControlKinds, Capabilities, Conditioning, ControlKind, GenerationPhase, Image,
        Modality, ModelDescriptor, PhaseAdapter, FLUX1_LATENT_SPACE,
    };

    fn synthetic_full_route() -> FeatureRouteStructure {
        FeatureRouteStructure {
            route_id: "synthetic_unified_route".into(),
            family: "synthetic".into(),
            backend: "mlx".into(),
            modality: Modality::Image,
            has_render_plan: true,
            has_packed_image_trajectory: true,
            supports_spatial_unpack_repack: true,
            supports_clean_estimate_renoise_transition: true,
            pass0_conditioning: Pass0ConditioningRequirement::NoiseStartSupported,
            supports_multiphase_denoise: false,
            supports_produced_reference_multipass: true,
            supports_expert_switching: true,
            supports_pag: true,
            default_executed_steps: Some(20),
            pixel_stride: Some(16),
            token_width: Some(64),
        }
    }

    fn synthetic_base_plan(route: &FeatureRouteStructure, steps: usize) -> ComposedRenderPlan {
        let sigmas: Vec<f32> = (0..=steps).map(|i| 1.0 - (i as f32 / steps as f32)).collect();
        let pass0 = ComposedPassPlan::single_phase(
            0,
            1024,
            1024,
            route.pixel_stride.unwrap_or(16),
            route.token_width.unwrap_or(64),
            "euler",
            3.5,
            sigmas,
            None,
        )
        .unwrap();
        ComposedRenderPlan {
            route_id: route.route_id.clone(),
            backend: route.backend.clone(),
            request_has_phases: false,
            pass0_has_conditioning: false,
            resolved_feature_order: Vec::new(),
            passes: vec![pass0],
        }
    }

    fn sample_invocation(feature_id: &str) -> FeatureInvocation {
        let parameters = match feature_id {
            LowResStartFeature::ID => FeatureParameters::LowResStart {
                k: 6,
                scale: Some(0.5),
            },
            HiresFixFeature::ID => FeatureParameters::HiresFix {
                width: 2048,
                height: 2048,
                steps: 4,
                strength: Some(0.3),
                sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
            },
            KreaMultiphaseFeature::ID => FeatureParameters::KreaMultiphase {
                phases: vec![
                    GenerationPhase {
                        steps: 12,
                        guidance: Some(3.5),
                        adapters: vec![],
                    },
                    GenerationPhase {
                        steps: 8,
                        guidance: Some(0.0),
                        adapters: vec![PhaseAdapter {
                            adapter: 0,
                            weight: Some(0.8),
                        }],
                    },
                ],
                loaded_adapter_count: 2,
            },
            WanExpertSwitchFeature::ID => FeatureParameters::WanExpertSwitch {
                boundary_step: 6,
                high_noise_expert: "high_noise".into(),
                low_noise_expert: "low_noise".into(),
            },
            PagFeature::ID => FeatureParameters::Pag {
                scale: 1.5,
                layers: vec![28],
            },
            InertPhaseSplitFeature::ID => FeatureParameters::InertPhaseSplit { step: 10 },
            TrajectoryObserverFeature::ID => FeatureParameters::TrajectoryObserver {
                label: "step_trace".into(),
            },
            other => panic!("unexpected feature_id {other}"),
        };
        FeatureInvocation {
            feature_id: feature_id.to_string(),
            parameters,
        }
    }

    /// HOOK-1: Every HookPoint in the enum declares its stage, timing, inputs, outputs, ordering,
    /// and result-changing classification, has at least one registered production user, and dumps
    /// to a deterministic JSON value without a checked-in matrix.
    #[test]
    fn hook_map_covers_every_enum_variant_with_registered_users_and_typed_dump() {
        let entries = hook_map();
        assert_eq!(entries.len(), HookPoint::ALL.len());
        for (window_idx, pair) in entries.windows(2).enumerate() {
            assert!(
                pair[0].ordering < pair[1].ordering,
                "hook map must be strictly ordered at index {window_idx}"
            );
        }
        for point in HookPoint::ALL {
            let meta = hook_point_metadata(point);
            assert!(!meta.name.is_empty());
            assert!(!meta.inputs.is_empty(), "{point:?} must declare inputs");
            assert!(!meta.outputs.is_empty(), "{point:?} must declare outputs");
            assert!(
                !meta.registered_users.is_empty(),
                "{point:?} must have at least one registered production user"
            );
        }
        let dumped = dump_hook_map_json();
        let arr = dumped.as_array().expect("hook map JSON dump is an array");
        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0]["name"], "schedule_transform");
        assert_eq!(arr[0]["stage"], "Schedule");
        assert_eq!(arr[0]["result_changing"], true);
    }

    /// HOOK-2: `FeatureSpec` is object-safe and tensor-free; `"low_res_start"` registers with
    /// DEC-8 class `Recipe`, version 1, schema version 1 (`k: 1..executed_steps`, `scale: 0.25..=0.9`,
    /// default `0.5`), conflicts with `phases` and pass-0 conditioning, and transition residency.
    #[test]
    fn low_res_start_spec_matches_hook2_contract_and_object_safety() {
        let registry = FeatureRegistry::standard();
        let spec: &dyn FeatureSpec = registry
            .get("low_res_start")
            .expect("low_res_start must be registered");
        assert_eq!(spec.id(), "low_res_start");
        assert_eq!(spec.version(), 1);
        assert_eq!(spec.class(), FeatureClass::Recipe);

        let route = synthetic_full_route();
        let schema = spec.parameter_schema(Some(&route));
        assert_eq!(schema.version, 1);
        let schema_json = schema.to_json_map();
        assert_eq!(schema_json["version"], 1);
        assert_eq!(schema_json["k"]["min"], 1);
        assert_eq!(schema_json["k"]["max_exclusive"], "executed_steps");
        assert_eq!(schema_json["k"]["max"], 19);
        assert_eq!(schema_json["scale"]["min"], 0.25);
        assert_eq!(schema_json["scale"]["max"], 0.9);
        assert_eq!(schema_json["scale"]["default"], 0.5);
        assert_eq!(schema_json["scale"]["inclusive"], true);

        let relations = spec.relations();
        assert!(relations.all_conflicts().contains(&"phases"));
        assert!(relations
            .all_conflicts()
            .contains(&"pass0_reference_conditioning"));
        assert!(relations
            .all_conflicts()
            .contains(&"pass0_control_conditioning"));
        assert!(relations
            .all_conflicts()
            .contains(&"pass0_edit_conditioning"));

        let memory = spec.memory_effect();
        assert_eq!(
            memory.transition_residency,
            TransitionResidency::InDeviceLatentUnpackResizeRepack
        );
        assert!(memory.introduces_denoise_phase_boundary);
        assert!(memory.retains_single_pass_schedule);
    }

    /// HOOK-3: Pairwise composition conformance across every pair of registered `FeatureSpec`s on a
    /// synthetic route, proving symmetry and deterministic outcomes.
    #[test]
    fn pairwise_feature_composition_conformance_on_synthetic_route() {
        let registry = FeatureRegistry::standard();
        let ids: Vec<&'static str> = registry.specs().iter().map(|s| s.id()).collect();

        for (i, &id_a) in ids.iter().enumerate() {
            for &id_b in &ids[i + 1..] {
                // Enable multiphase support on the synthetic route only when testing krea_multiphase
                // without low_res_start, so pairwise composition tests feature-to-feature rules.
                let mut route = synthetic_full_route();
                if id_a == KreaMultiphaseFeature::ID || id_b == KreaMultiphaseFeature::ID {
                    route.supports_multiphase_denoise = true;
                }
                let inv_a = sample_invocation(id_a);
                let inv_b = sample_invocation(id_b);

                let res_ab = registry.compose(
                    &route,
                    synthetic_base_plan(&route, 20),
                    &[inv_a.clone(), inv_b.clone()],
                );
                let res_ba = registry.compose(
                    &route,
                    synthetic_base_plan(&route, 20),
                    &[inv_b, inv_a],
                );

                let spec_a = registry.get(id_a).unwrap();
                let spec_b = registry.get(id_b).unwrap();
                let declared_conflict = spec_a.relations().conflicts_with_features.contains(&id_b)
                    || spec_b.relations().conflicts_with_features.contains(&id_a)
                    || (id_a == LowResStartFeature::ID && route.supports_multiphase_denoise)
                    || (id_b == LowResStartFeature::ID && route.supports_multiphase_denoise);

                if declared_conflict {
                    assert!(
                        res_ab.is_err() && res_ba.is_err(),
                        "conflicting pair ({id_a}, {id_b}) must be refused in both orders"
                    );
                } else {
                    let plan_ab = res_ab.unwrap_or_else(|e| {
                        panic!("compatible pair ({id_a}, {id_b}) failed in A->B order: {e}")
                    });
                    let plan_ba = res_ba.unwrap_or_else(|e| {
                        panic!("compatible pair ({id_a}, {id_b}) failed in B->A order: {e}")
                    });
                    assert_eq!(
                        plan_ab, plan_ba,
                        "compatible pair ({id_a}, {id_b}) must resolve to identical plan in both orders"
                    );
                }
            }
        }
    }

    /// HOOK-3: Whole-active-set cycle and exclusive `HookTiming::Replace` ownership validation.
    #[test]
    fn whole_active_set_detects_ordering_cycles_and_replace_ownership_conflicts() {
        struct SyntheticFeature {
            id: &'static str,
            before: Vec<&'static str>,
            hook_point: HookPoint,
            timing: HookTiming,
        }

        impl FeatureSpec for SyntheticFeature {
            fn id(&self) -> &'static str {
                self.id
            }
            fn version(&self) -> u32 {
                1
            }
            fn class(&self) -> FeatureClass {
                FeatureClass::Equivalent
            }
            fn parameter_schema(
                &self,
                _route: Option<&FeatureRouteStructure>,
            ) -> FeatureParameterSchema {
                FeatureParameterSchema {
                    version: 1,
                    parameters: Vec::new(),
                }
            }
            fn relations(&self) -> FeatureRelations {
                FeatureRelations {
                    before: self.before.clone(),
                    ..Default::default()
                }
            }
            fn hook_bindings(&self) -> Vec<FeatureHookBinding> {
                vec![FeatureHookBinding {
                    hook_point: self.hook_point,
                    stage: self.hook_point.stage(),
                    timing: self.timing,
                    pass_scope: Some(0),
                    result_changing: false,
                }]
            }
            fn memory_effect(&self) -> FeatureMemoryEffect {
                FeatureMemoryEffect {
                    extra_resident_weight_bytes: 0,
                    transition_residency: TransitionResidency::None,
                    introduces_denoise_phase_boundary: false,
                    introduces_pass_boundary: false,
                    retains_single_pass_schedule: true,
                }
            }
            fn inapplicability_reasons(
                &self,
                _route: &FeatureRouteStructure,
            ) -> Vec<FeatureInapplicabilityReason> {
                Vec::new()
            }
            fn rewrite_plan(
                &self,
                _route: &FeatureRouteStructure,
                _plan: &mut ComposedRenderPlan,
                _parameters: &FeatureParameters,
            ) -> Result<()> {
                Ok(())
            }
        }

        let route = synthetic_full_route();

        // 3-way ordering cycle: alpha -> beta -> gamma -> alpha.
        let cycle_registry = FeatureRegistry::with_specs(vec![
            Box::new(SyntheticFeature {
                id: "alpha",
                before: vec!["beta"],
                hook_point: HookPoint::ScheduleTransform,
                timing: HookTiming::After,
            }),
            Box::new(SyntheticFeature {
                id: "beta",
                before: vec!["gamma"],
                hook_point: HookPoint::ConditioningSlot,
                timing: HookTiming::After,
            }),
            Box::new(SyntheticFeature {
                id: "gamma",
                before: vec!["alpha"],
                hook_point: HookPoint::PhaseTransition,
                timing: HookTiming::Before,
            }),
        ])
        .unwrap();

        let cycle_err = cycle_registry
            .compose(
                &route,
                synthetic_base_plan(&route, 10),
                &[
                    FeatureInvocation {
                        feature_id: "alpha".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "a".into() },
                    },
                    FeatureInvocation {
                        feature_id: "beta".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "b".into() },
                    },
                    FeatureInvocation {
                        feature_id: "gamma".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "c".into() },
                    },
                ],
            )
            .unwrap_err();
        assert!(
            cycle_err.to_string().contains("cycle"),
            "expected cycle refusal, got {cycle_err}"
        );

        // Two features claiming `HookTiming::Replace` on the same hook point and pass scope.
        let conflict_registry = FeatureRegistry::with_specs(vec![
            Box::new(SyntheticFeature {
                id: "guider_one",
                before: Vec::new(),
                hook_point: HookPoint::GuidancePredictor,
                timing: HookTiming::Replace,
            }),
            Box::new(SyntheticFeature {
                id: "guider_two",
                before: Vec::new(),
                hook_point: HookPoint::GuidancePredictor,
                timing: HookTiming::Replace,
            }),
        ])
        .unwrap();

        let owner_err = conflict_registry
            .compose(
                &route,
                synthetic_base_plan(&route, 10),
                &[
                    FeatureInvocation {
                        feature_id: "guider_one".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "1".into() },
                    },
                    FeatureInvocation {
                        feature_id: "guider_two".into(),
                        parameters: FeatureParameters::TrajectoryObserver { label: "2".into() },
                    },
                ],
            )
            .unwrap_err();
        assert!(
            owner_err.to_string().contains("hook ownership conflict"),
            "expected hook ownership refusal, got {owner_err}"
        );
    }

    /// HOOK-3: Worked plan tests for:
    /// 1) low-res + Hires (`"low_res_start"` + `"hires_fix"`)
    /// 2) Krea phases (`"krea_multiphase"`)
    /// 3) Coincident Wan expert/geometry boundary (`"low_res_start"` + `"wan_expert_switch"` at the same step k)
    ///
    /// Also proves that permuting registration order in `FeatureRegistry` and invocation order cannot
    /// change the resolved plan.
    #[test]
    fn worked_plans_and_registration_order_invariance() {
        let route = synthetic_full_route();
        let standard = FeatureRegistry::standard();

        // Worked plan 1: low_res_start (pass 0: 0..6 at 512x512, 6..20 at 1024x1024) + hires_fix (pass 1: 2048x2048).
        let low_res_hires = standard
            .compose(
                &route,
                synthetic_base_plan(&route, 20),
                &[
                    sample_invocation(HiresFixFeature::ID),
                    sample_invocation(LowResStartFeature::ID),
                ],
            )
            .unwrap();
        assert_eq!(
            low_res_hires.resolved_feature_order,
            vec!["low_res_start", "hires_fix"]
        );
        assert_eq!(low_res_hires.passes.len(), 2);
        assert_eq!(low_res_hires.passes[0].phases.len(), 2);
        assert_eq!(
            (
                low_res_hires.passes[0].phases[0].first_step,
                low_res_hires.passes[0].phases[0].end_step,
                low_res_hires.passes[0].phases[0].width,
                low_res_hires.passes[0].phases[0].height,
            ),
            (0, 6, 512, 512)
        );
        assert_eq!(
            (
                low_res_hires.passes[0].phases[1].first_step,
                low_res_hires.passes[0].phases[1].end_step,
                low_res_hires.passes[0].phases[1].width,
                low_res_hires.passes[0].phases[1].height,
            ),
            (6, 20, 1024, 1024)
        );
        assert_eq!(
            low_res_hires.passes[1].produced_reference,
            Some(ProducedReference {
                from_pass: 0,
                width: 2048,
                height: 2048,
                strength: Some(0.3),
            })
        );

        // Worked plan 2: Krea multiphase over one shared 20-step global schedule.
        let mut krea_route = synthetic_full_route();
        krea_route.supports_multiphase_denoise = true;
        let krea_plan = standard
            .compose(
                &krea_route,
                synthetic_base_plan(&krea_route, 20),
                &[sample_invocation(KreaMultiphaseFeature::ID)],
            )
            .unwrap();
        assert_eq!(krea_plan.passes[0].phases.len(), 2);
        assert_eq!(
            (
                krea_plan.passes[0].phases[0].first_step,
                krea_plan.passes[0].phases[0].end_step,
                krea_plan.passes[0].phases[0].guidance,
            ),
            (0, 12, 3.5)
        );
        assert_eq!(
            (
                krea_plan.passes[0].phases[1].first_step,
                krea_plan.passes[0].phases[1].end_step,
                krea_plan.passes[0].phases[1].guidance,
            ),
            (12, 20, 0.0)
        );
        assert_eq!(krea_plan.passes[0].transitions.len(), 1);
        assert_eq!(krea_plan.passes[0].transitions[0].step, 12);

        // Worked plan 3: Coincident Wan expert/geometry boundary at step 6 + Hires.fix + observer.
        // Also prove registration-order invariance across reversed and rotated registries.
        let active_set = vec![
            sample_invocation(TrajectoryObserverFeature::ID),
            sample_invocation(HiresFixFeature::ID),
            sample_invocation(WanExpertSwitchFeature::ID),
            sample_invocation(LowResStartFeature::ID),
        ];

        let baseline_plan = standard
            .compose(&route, synthetic_base_plan(&route, 20), &active_set)
            .unwrap();
        // Step 6 has ONE merged transition boundary with BOTH actions in canonical order, and 2 spans (0..6, 6..20).
        assert_eq!(baseline_plan.passes[0].phases.len(), 2);
        assert_eq!(baseline_plan.passes[0].transitions.len(), 1);
        assert_eq!(baseline_plan.passes[0].transitions[0].step, 6);
        assert_eq!(baseline_plan.passes[0].transitions[0].actions.len(), 2);
        assert_eq!(
            baseline_plan.passes[0].phases[0].active_expert.as_deref(),
            Some("high_noise")
        );
        assert_eq!(
            (
                baseline_plan.passes[0].phases[0].width,
                baseline_plan.passes[0].phases[0].height
            ),
            (512, 512)
        );
        assert_eq!(
            baseline_plan.passes[0].phases[1].active_expert.as_deref(),
            Some("low_noise")
        );
        assert_eq!(
            (
                baseline_plan.passes[0].phases[1].width,
                baseline_plan.passes[0].phases[1].height
            ),
            (1024, 1024)
        );

        let reversed_registry = FeatureRegistry::with_specs(vec![
            Box::new(TrajectoryObserverFeature),
            Box::new(InertPhaseSplitFeature),
            Box::new(PagFeature),
            Box::new(WanExpertSwitchFeature),
            Box::new(KreaMultiphaseFeature),
            Box::new(HiresFixFeature),
            Box::new(LowResStartFeature),
        ])
        .unwrap();
        let rotated_registry = FeatureRegistry::with_specs(vec![
            Box::new(WanExpertSwitchFeature),
            Box::new(HiresFixFeature),
            Box::new(TrajectoryObserverFeature),
            Box::new(LowResStartFeature),
            Box::new(PagFeature),
            Box::new(KreaMultiphaseFeature),
            Box::new(InertPhaseSplitFeature),
        ])
        .unwrap();

        let mut reversed_invocations = active_set.clone();
        reversed_invocations.reverse();

        for reg in [&standard, &reversed_registry, &rotated_registry] {
            for invs in [&active_set, &reversed_invocations] {
                let candidate = reg
                    .compose(&route, synthetic_base_plan(&route, 20), invs)
                    .unwrap();
                assert_eq!(
                    serde_json::to_value(&candidate).unwrap(),
                    serde_json::to_value(&baseline_plan).unwrap(),
                    "changing registration order or invocation order must never change the resolved plan"
                );
            }
        }
    }

    /// HOOK-4: Removing a structural precondition from a route descriptor or RenderPlanRegistration
    /// flips feature availability automatically with no table edits.
    #[test]
    fn removing_structural_precondition_flips_applicability_without_table_edits() {
        fn compile_fixture(
            _spec: &LoadSpec,
            req: &GenerationRequest,
        ) -> Result<Option<RenderPlan>> {
            let desc = fixture_descriptor();
            RenderPlan::base_image(
                &desc,
                req,
                crate::render::PackedImagePass {
                    pixel_stride: 16,
                    token_width: 64,
                    sampler: "euler",
                    guidance: 3.5,
                    sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                    executed_steps: 4,
                    capture_sigma: 0.0,
                },
            )
            .map(Some)
        }

        fn fixture_descriptor() -> ModelDescriptor {
            ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: Some(&FLUX1_LATENT_SPACE),
                control_kinds: None,
                required_components: &[],
                id: "structural_probe_route",
                family: "flux",
                backend: "mlx",
                modality: Modality::Image,
                capabilities: Capabilities {
                    min_size: 256,
                    max_size: 2048,
                    max_count: 1,
                    ..Default::default()
                },
            }
        }

        let planner = RenderPlanRegistration {
            provider_id: "structural_probe_route",
            compile: compile_fixture,
        };
        let full_desc = fixture_descriptor();
        let full_route =
            FeatureRouteStructure::from_descriptor_and_plan(&full_desc, Some(&planner));
        let full_app = LowResStartFeature.applicability(&full_route);
        assert!(
            full_app.applicable,
            "route satisfying all structural preconditions must be applicable"
        );
        assert_eq!(full_app.to_json()["applicable"], true);
        assert_eq!(full_app.to_json()["parameter_schema"]["k"]["max"], 3);

        // 1. Remove the RenderPlanRegistration -> flips to false with `missing_render_plan`.
        let without_plan = FeatureRouteStructure::from_descriptor_and_plan(&full_desc, None);
        let no_plan_app = LowResStartFeature.applicability(&without_plan);
        assert!(!no_plan_app.applicable);
        assert!(no_plan_app
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::MissingRenderPlan));

        // 2. Remove the packed latent space from the descriptor -> flips to false with `missing_spatial_unpack_repack`.
        let mut without_latent = full_desc.clone();
        without_latent.denoiser_output_latent_space = None;
        let no_latent_route =
            FeatureRouteStructure::from_descriptor_and_plan(&without_latent, Some(&planner));
        let no_latent_app = LowResStartFeature.applicability(&no_latent_route);
        assert!(!no_latent_app.applicable);
        assert!(no_latent_app
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::MissingSpatialUnpackRepack));

        // 3. Make the route require pass-0 control conditioning -> flips to false with `conflicts_with_pass0_control_conditioning`.
        fn compile_control_only(
            _spec: &LoadSpec,
            req: &GenerationRequest,
        ) -> Result<Option<RenderPlan>> {
            if !matches!(req.conditioning.as_slice(), [Conditioning::Control { .. }]) {
                return Err(Error::Unsupported("requires control".into()));
            }
            let mut desc = fixture_descriptor();
            desc.control_kinds = Some(AcceptedControlKinds::Only(vec![ControlKind::Pose]));
            desc.capabilities.conditioning = vec![ConditioningKind::Control];
            RenderPlan::packed_image_trajectory(
                &desc,
                req,
                crate::render::PackedImagePass {
                    pixel_stride: 16,
                    token_width: 64,
                    sampler: "euler",
                    guidance: 3.5,
                    sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                    executed_steps: 4,
                    capture_sigma: 0.0,
                },
            )
            .map(Some)
        }
        let mut control_desc = full_desc;
        control_desc.control_kinds = Some(AcceptedControlKinds::Only(vec![ControlKind::Pose]));
        control_desc.capabilities.conditioning = vec![ConditioningKind::Control];
        let control_planner = RenderPlanRegistration {
            provider_id: "structural_probe_route",
            compile: compile_control_only,
        };
        let control_route = FeatureRouteStructure::from_descriptor_and_plan(
            &control_desc,
            Some(&control_planner),
        );
        let control_app = LowResStartFeature.applicability(&control_route);
        assert!(!control_app.applicable);
        assert!(control_app
            .reasons
            .iter()
            .any(|r| r.code == FeatureInapplicabilityCode::ConflictsWithPass0ControlConditioning));
    }

    /// HOOK-5: `validate_low_res_start_request` refuses `req.phases` and pass-0 reference/control
    /// conditioning with a typed `Error::Unsupported`.
    #[test]
    fn low_res_start_refuses_phases_and_pass0_conditioning_with_typed_error() {
        fn compile_fixture(
            _spec: &LoadSpec,
            req: &GenerationRequest,
        ) -> Result<Option<RenderPlan>> {
            let desc = fixture_descriptor();
            RenderPlan::base_image(
                &desc,
                req,
                crate::render::PackedImagePass {
                    pixel_stride: 16,
                    token_width: 64,
                    sampler: "euler",
                    guidance: 3.5,
                    sigmas: vec![1.0, 0.75, 0.5, 0.25, 0.0],
                    executed_steps: 4,
                    capture_sigma: 0.0,
                },
            )
            .map(Some)
        }
        fn fixture_descriptor() -> ModelDescriptor {
            ModelDescriptor {
                encoder_contract: None,
                denoiser_output_latent_space: Some(&FLUX1_LATENT_SPACE),
                control_kinds: None,
                required_components: &[],
                id: "flux1_dev",
                family: "flux",
                backend: "mlx",
                modality: Modality::Image,
                capabilities: Capabilities {
                    min_size: 256,
                    max_size: 2048,
                    max_count: 1,
                    ..Default::default()
                },
            }
        }

        let desc = fixture_descriptor();
        let planner = RenderPlanRegistration {
            provider_id: "flux1_dev",
            compile: compile_fixture,
        };
        let clean_req = GenerationRequest {
            prompt: "a lighthouse".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_low_res_start_request(
            &desc,
            Some(&planner),
            &clean_req,
            4,
            2,
            Some(0.5)
        )
        .is_ok());

        let phased_req = GenerationRequest {
            phases: Some(vec![GenerationPhase {
                steps: 4,
                guidance: Some(3.5),
                adapters: vec![],
            }]),
            ..clean_req.clone()
        };
        assert!(matches!(
            validate_low_res_start_request(&desc, Some(&planner), &phased_req, 4, 2, Some(0.5)),
            Err(Error::Unsupported(_))
        ));

        let ref_req = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 16,
                    height: 16,
                    pixels: vec![0; 16 * 16 * 3],
                },
                strength: Some(0.7),
            }],
            ..clean_req
        };
        assert!(matches!(
            validate_low_res_start_request(&desc, Some(&planner), &ref_req, 4, 2, Some(0.5)),
            Err(Error::Unsupported(_))
        ));
    }
}
