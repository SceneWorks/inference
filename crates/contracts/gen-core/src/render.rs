//! Weight-free execution contract and owner-local lifecycle for fixed image passes.

pub mod conditional;
mod passes;
pub mod video;
pub mod video_plan;

pub use passes::{MultiPassRenderPlan, PlannedPassRequest, ProducedReference};

use crate::phase::{ConditioningBranches, PhaseAdapterScale, PhaseArguments};
use crate::{
    default_seed, CancelFlag, ComposedRenderPlan, Conditioning, ControlKind, Error,
    FeatureInvocation, FeatureParameters, FeatureRegistry, FeatureRouteStructure,
    GenerationRequest, Image, KreaMultiphaseFeature, LoadSpec, MemoryPhase, MemoryRequestScope,
    ModelDescriptor, Progress, Result,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::ops::Range;

#[derive(Clone, Copy)]
pub struct RenderPlanRegistration {
    pub provider_id: &'static str,
    pub compile: fn(&LoadSpec, &GenerationRequest) -> Result<Option<RenderPlan>>,
}

/// A serialized plan exposes the actual execution order, including once-per-request encoding
/// and the subset repeated for each sequential image. This noise-start pass has no reference stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RenderStage {
    PromptPrep,
    Encode,
    Schedule,
    Init,
    Enter,
    Denoise,
    Exit,
    PostLatent,
    Decode,
    Output,
}

type Stage = RenderStage;

#[derive(Clone, Debug, PartialEq, Serialize)]
struct PackedImageStream {
    width: u32,
    height: u32,
    patch_stride: u32,
    tokens: u32,
    token_width: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ImagePhase {
    first_step: usize,
    end_step: usize,
    stream: PackedImageStream,
    sampler: String,
    guidance: f32,
}

/// A compiled adapter/guidance phase using absolute steps of its pass's global schedule.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AdapterDenoisePhase {
    index: usize,
    span: Range<usize>,
    arguments: PhaseArguments,
}

impl AdapterDenoisePhase {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    pub fn arguments(&self) -> &PhaseArguments {
        &self.arguments
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum AdapterPhaseSampling {
    // Legacy Krea sampling resets solver history and starts phase-local stochastic draw
    // coordinates on the original image seed. The noisy latent and boundary sigma continue.
    ResetSolverAndReuseImageSeed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum AdapterPhaseInstallation {
    // Every phase selects from the model's installed, immutable residuals. Their placement
    // and any retained preparation are live-set allocations, not a weight-swap transition.
    ImmutableResiduals,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct AdapterPhasePlan {
    phases: Vec<AdapterDenoisePhase>,
    need_negative: bool,
    sampling: AdapterPhaseSampling,
    installation: AdapterPhaseInstallation,
}

#[derive(Clone, Debug)]
struct ExactRenderLoad(std::sync::Arc<LoadSpec>);

impl PartialEq for ExactRenderLoad {
    fn eq(&self, other: &Self) -> bool {
        self.0.same_load_except_placement(&other.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ImagePass {
    stages: [Stage; 10],
    sigmas: Vec<f32>,
    phase: ImagePhase,
    #[serde(skip_serializing_if = "Option::is_none")]
    low_res_start: Option<crate::ResolvedLowResStartPlan>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    inert_splits: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_phases: Option<AdapterPhasePlan>,
    capture_sigma: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
struct LowResStartInput {
    version: u32,
    k: usize,
    scale_bits: Option<u32>,
    estimate_source: crate::EstimateSource,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct SourceImageInput {
    width: u32,
    height: u32,
    pixels_sha256: [u8; 32],
    strength_bits: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ReferenceImageInput {
    width: u32,
    height: u32,
    pixels_sha256: [u8; 32],
    #[serde(skip_serializing_if = "Option::is_none")]
    strength_bits: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct AdapterPhaseInputs {
    authored: Vec<AuthoredAdapterPhase>,
    pag: Option<crate::PagConfig>,
    text_style_gain_bits: Option<u32>,
    output_rgba: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct AuthoredAdapterPhase {
    steps: u32,
    guidance_bits: Option<u32>,
    adapters: Vec<(usize, Option<u32>)>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ExactImageControls {
    strength_bits: Option<u32>,
    pag: Option<crate::PagConfig>,
    text_style_gain_bits: Option<u32>,
    output_rgba: bool,
}

impl From<&GenerationRequest> for ExactImageControls {
    fn from(req: &GenerationRequest) -> Self {
        Self {
            strength_bits: req.strength.map(f32::to_bits),
            pag: req.pag.clone(),
            text_style_gain_bits: req.text_style_gain.map(f32::to_bits),
            output_rgba: req.output_channels == crate::OutputChannels::Rgba,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct ImageInputs {
    prompt_sha256: [u8; 32],
    negative_prompt: Option<String>,
    width: u32,
    height: u32,
    count: u32,
    requested_seed: Option<u64>,
    frames: u32,
    references: u32,
    conditioning: PackedImageConditioning,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_image: Option<SourceImageInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    low_res_start: Option<LowResStartInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adapter_phases: Option<AdapterPhaseInputs>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exact_controls: Option<ExactImageControls>,
    steps: Option<u32>,
    sampler: Option<String>,
    scheduler: Option<String>,
    scheduler_shift: Option<f32>,
    guidance: Option<f32>,
    guidance_method: Option<String>,
    true_cfg_bits: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_guidance_bits: Option<u32>,
    timestep_to_start_cfg: Option<u32>,
    use_pid: bool,
    pid_capture_sigma: Option<f32>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    enhance_prompt: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    enhance_temperature_bits: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enhance_max_tokens: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct ControlImageInput {
    width: u32,
    height: u32,
    pixels_sha256: [u8; 32],
    kind: String,
    scale_bits: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct MaskImageInput {
    width: u32,
    height: u32,
    pixels_sha256: [u8; 32],
}

/// Bind the actual bytes and controls of an image pass before load so a queued request
/// cannot swap its source or conditioning inputs after plan compilation.
#[derive(Clone, Debug, PartialEq, Serialize)]
enum PackedImageConditioning {
    None,
    Reference {
        width: u32,
        height: u32,
        pixels_sha256: [u8; 32],
        strength_bits: Option<u32>,
    },
    MultiReference {
        references: Vec<ReferenceImageInput>,
    },
    Control {
        width: u32,
        height: u32,
        pixels_sha256: [u8; 32],
        kind: String,
        scale_bits: Option<u32>,
    },
    MultiControl {
        controls: Vec<ControlImageInput>,
    },
    ControlWithReference {
        width: u32,
        height: u32,
        pixels_sha256: [u8; 32],
        kind: String,
        scale_bits: Option<u32>,
        reference: ReferenceImageInput,
    },
    MultiControlWithReference {
        controls: Vec<ControlImageInput>,
        reference: ReferenceImageInput,
    },
    Inpaint {
        reference: ReferenceImageInput,
        mask: MaskImageInput,
    },
    /// A reference an earlier pass of the same multi-pass plan produces. Its pixels do not exist
    /// before load, so the plan binds the declared geometry and strength instead of a digest.
    ProducedReference {
        from_pass: usize,
        width: u32,
        height: u32,
        strength_bits: Option<u32>,
    },
}

fn control_kind_key(kind: &ControlKind) -> String {
    match kind {
        ControlKind::Pose => "pose".to_owned(),
        ControlKind::Canny => "canny".to_owned(),
        ControlKind::Depth => "depth".to_owned(),
        ControlKind::Other(value) => format!("other:{value}"),
    }
}

fn image_digest(image: &Image) -> Result<[u8; 32]> {
    let expected = u64::from(image.width)
        .checked_mul(u64::from(image.height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| Error::Unsupported("conditioning image geometry overflows".into()))?;
    if image.width == 0 || image.height == 0 || expected != image.pixels.len() as u64 {
        return Err(Error::Unsupported(
            "conditioning image needs width × height × 3 RGB bytes".into(),
        ));
    }
    Ok(Sha256::digest(&image.pixels).into())
}

impl TryFrom<&GenerationRequest> for ImageInputs {
    type Error = Error;

    fn try_from(req: &GenerationRequest) -> Result<Self> {
        let conditioning = match req.conditioning.as_slice() {
            [] => PackedImageConditioning::None,
            [Conditioning::Reference { image, strength }] => PackedImageConditioning::Reference {
                width: image.width,
                height: image.height,
                pixels_sha256: image_digest(image)?,
                strength_bits: strength.map(f32::to_bits),
            },
            [Conditioning::Control { image, kind, scale }] => PackedImageConditioning::Control {
                width: image.width,
                height: image.height,
                pixels_sha256: image_digest(image)?,
                kind: control_kind_key(kind),
                scale_bits: scale.map(f32::to_bits),
            },
            [Conditioning::Control {
                image: ctrl_img,
                kind,
                scale,
            }, Conditioning::Reference {
                image: ref_img,
                strength,
            }]
            | [Conditioning::Reference {
                image: ref_img,
                strength,
            }, Conditioning::Control {
                image: ctrl_img,
                kind,
                scale,
            }] => PackedImageConditioning::ControlWithReference {
                width: ctrl_img.width,
                height: ctrl_img.height,
                pixels_sha256: image_digest(ctrl_img)?,
                kind: control_kind_key(kind),
                scale_bits: scale.map(f32::to_bits),
                reference: ReferenceImageInput {
                    width: ref_img.width,
                    height: ref_img.height,
                    pixels_sha256: image_digest(ref_img)?,
                    strength_bits: strength.map(f32::to_bits),
                },
            },
            [Conditioning::Reference {
                image: ref_img,
                strength,
            }, Conditioning::Mask { image: mask_img }]
            | [Conditioning::Mask { image: mask_img }, Conditioning::Reference {
                image: ref_img,
                strength,
            }] => PackedImageConditioning::Inpaint {
                reference: ReferenceImageInput {
                    width: ref_img.width,
                    height: ref_img.height,
                    pixels_sha256: image_digest(ref_img)?,
                    strength_bits: strength.map(f32::to_bits),
                },
                mask: MaskImageInput {
                    width: mask_img.width,
                    height: mask_img.height,
                    pixels_sha256: image_digest(mask_img)?,
                },
            },
            items
                if items.len() > 1
                    && items
                        .iter()
                        .all(|c| matches!(c, Conditioning::Control { .. })) =>
            {
                let mut controls = Vec::with_capacity(items.len());
                for c in items {
                    let Conditioning::Control { image, kind, scale } = c else {
                        unreachable!()
                    };
                    controls.push(ControlImageInput {
                        width: image.width,
                        height: image.height,
                        pixels_sha256: image_digest(image)?,
                        kind: control_kind_key(kind),
                        scale_bits: scale.map(f32::to_bits),
                    });
                }
                PackedImageConditioning::MultiControl { controls }
            }
            items
                if items.len() > 2
                    && items.iter().all(|c| {
                        matches!(
                            c,
                            Conditioning::Control { .. } | Conditioning::Reference { .. }
                        )
                    })
                    && items
                        .iter()
                        .filter(|c| matches!(c, Conditioning::Reference { .. }))
                        .count()
                        == 1 =>
            {
                let mut controls = Vec::with_capacity(items.len() - 1);
                let mut reference = None;
                for c in items {
                    match c {
                        Conditioning::Control { image, kind, scale } => {
                            controls.push(ControlImageInput {
                                width: image.width,
                                height: image.height,
                                pixels_sha256: image_digest(image)?,
                                kind: control_kind_key(kind),
                                scale_bits: scale.map(f32::to_bits),
                            });
                        }
                        Conditioning::Reference { image, strength } => {
                            reference = Some(ReferenceImageInput {
                                width: image.width,
                                height: image.height,
                                pixels_sha256: image_digest(image)?,
                                strength_bits: strength.map(f32::to_bits),
                            });
                        }
                        _ => unreachable!(),
                    }
                }
                PackedImageConditioning::MultiControlWithReference {
                    controls,
                    reference: reference.expect("guard ensures one Reference"),
                }
            }
            items
                if items.iter().all(|c| {
                    matches!(
                        c,
                        Conditioning::Reference { .. } | Conditioning::MultiReference { .. }
                    )
                }) =>
            {
                let mut references = Vec::new();
                for c in items {
                    match c {
                        Conditioning::Reference { image, strength } => {
                            references.push(ReferenceImageInput {
                                width: image.width,
                                height: image.height,
                                pixels_sha256: image_digest(image)?,
                                strength_bits: strength.map(f32::to_bits),
                            });
                        }
                        Conditioning::MultiReference { images } => {
                            if images.is_empty() {
                                return Err(Error::Unsupported(
                                    "multi-reference conditioning requires at least one image"
                                        .into(),
                                ));
                            }
                            for image in images {
                                references.push(ReferenceImageInput {
                                    width: image.width,
                                    height: image.height,
                                    pixels_sha256: image_digest(image)?,
                                    strength_bits: None,
                                });
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                PackedImageConditioning::MultiReference { references }
            }
            _ => {
                return Err(Error::Unsupported(
                    "packed image plan needs one supported image conditioning input".into(),
                ))
            }
        };
        let source_image = match &req.source_image {
            None => None,
            Some(image) => {
                if let Some(s) = req.strength {
                    if !s.is_finite() || !(0.0..=1.0).contains(&s) {
                        return Err(Error::Unsupported(format!(
                            "img2img strength {s} must be finite and lie in 0.0..=1.0"
                        )));
                    }
                }
                Some(SourceImageInput {
                    width: image.width,
                    height: image.height,
                    pixels_sha256: image_digest(image)?,
                    strength_bits: req.strength.map(f32::to_bits),
                })
            }
        };
        let low_res_start = req.low_res_start.map(|spec| LowResStartInput {
            version: spec.version,
            k: spec.k,
            scale_bits: spec.scale.map(f32::to_bits),
            estimate_source: spec.estimate_source,
        });
        let adapter_phases = req.phases.as_ref().map(|phases| AdapterPhaseInputs {
            authored: phases
                .iter()
                .map(|phase| AuthoredAdapterPhase {
                    steps: phase.steps,
                    guidance_bits: phase.guidance.map(f32::to_bits),
                    adapters: phase
                        .adapters
                        .iter()
                        .map(|adapter| (adapter.adapter, adapter.weight.map(f32::to_bits)))
                        .collect(),
                })
                .collect(),
            pag: req.pag.clone(),
            text_style_gain_bits: req.text_style_gain.map(f32::to_bits),
            output_rgba: req.output_channels == crate::OutputChannels::Rgba,
        });
        Ok(Self {
            prompt_sha256: Sha256::digest(req.prompt.as_bytes()).into(),
            negative_prompt: req.negative_prompt.clone(),
            width: req.width,
            height: req.height,
            count: req.count,
            requested_seed: req.seed,
            frames: req.frames.unwrap_or(1),
            references: req.memory_reference_count(),
            conditioning,
            source_image,
            low_res_start,
            adapter_phases,
            exact_controls: None,
            steps: req.steps,
            sampler: req.sampler.clone(),
            scheduler: req.scheduler.clone(),
            scheduler_shift: req.scheduler_shift,
            guidance: req.guidance,
            guidance_method: req.guidance_method.clone(),
            true_cfg_bits: req.true_cfg.map(f32::to_bits),
            image_guidance_bits: req.image_guidance.map(f32::to_bits),
            timestep_to_start_cfg: req.timestep_to_start_cfg,
            use_pid: req.use_pid,
            pid_capture_sigma: req.pid_capture_sigma,
            enhance_prompt: req.enhance_prompt,
            enhance_temperature_bits: req.enhance_temperature.map(f32::to_bits),
            enhance_max_tokens: req.enhance_max_tokens,
        })
    }
}

/// Dependencies of base FLUX preparation at load. The cache/loader independently enforces
/// its prepared source pins and admitted offload policy. Neither pins nor tensors are cloned.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RenderLoadBinding {
    #[serde(skip)]
    video_source: Option<std::sync::Arc<VideoSourceBinding>>,
    weights: crate::WeightsSource,
    resolved_route: Option<String>,
    quant: Option<crate::Quant>,
    precision: crate::Precision,
    sequential_offload: bool,
    deferred_materialization: bool,
    load_shape_declaration: String,
    adapters: Vec<crate::AdapterSpec>,
    control: Option<crate::WeightsSource>,
    extra_controls: Vec<crate::WeightsSource>,
    ip_adapter: Option<crate::WeightsSource>,
    identity: Option<[Option<crate::WeightsSource>; 3]>,
    text_encoder: Option<crate::WeightsSource>,
    components: std::collections::BTreeMap<String, crate::WeightsSource>,
    pid: Option<crate::PidWeights>,
}

#[derive(Clone, Debug, PartialEq)]
struct VideoSourceBinding {
    pins: crate::PreparedFilePins,
    inventory: Option<std::sync::Arc<crate::SourceInventory>>,
    policies: Vec<crate::MemoryDecodeGeometryPolicy>,
    authoritative: bool,
    runtime: Option<crate::MemoryDecodeQualityRuntimeIdentity>,
}

impl RenderLoadBinding {
    /// Capture the same source and decode-policy metadata at pre-load and at factory entry.
    pub fn from_video_spec(spec: &LoadSpec) -> Self {
        Self {
            video_source: Some(std::sync::Arc::new(VideoSourceBinding {
                inventory: spec.retained_source_inventory(),
                pins: spec.prepared_file_pins().clone(),
                policies: spec.decode_geometry_policies.clone(),
                authoritative: spec.decode_geometry_policy_authoritative,
                runtime: spec.decode_quality_runtime_identity.clone(),
            })),
            ..Self::from_spec(spec)
        }
    }

    pub fn from_spec(spec: &LoadSpec) -> Self {
        Self {
            video_source: None,
            weights: spec.weights.clone(),
            resolved_route: spec.resolved_route.clone(),
            quant: spec.quantize,
            precision: spec.precision,
            sequential_offload: matches!(spec.offload_policy, crate::OffloadPolicy::Sequential),
            deferred_materialization: matches!(
                spec.load_shape,
                crate::LoadShape::DeferredMaterialization
            ),
            load_shape_declaration: format!("{:?}", spec.load_shape_declaration_result),
            adapters: spec.adapters.clone(),
            control: spec.control.clone(),
            extra_controls: spec.extra_controls.clone(),
            ip_adapter: spec.ip_adapter.clone(),
            identity: spec.identity.as_ref().map(|identity| {
                [
                    identity.encoder.clone(),
                    identity.eva.clone(),
                    identity.face_dir.clone(),
                ]
            }),
            text_encoder: spec.text_encoder.clone(),
            components: spec.components.clone(),
            pid: spec.pid.clone(),
        }
    }
}

/// Count and seed order belong to the plan. Each image gets its own preview sequence
/// and is decoded before the next image's initial noise is drawn.
#[derive(Clone, Debug, PartialEq, Serialize)]
enum CountPlan {
    SequentialWrappingSeeds { count: u32, base_seed: u64 },
}

impl CountPlan {
    fn seeds(&self) -> impl Iterator<Item = u64> {
        let Self::SequentialWrappingSeeds { count, base_seed } = *self;
        (0..count).map(move |index| base_seed.wrapping_add(u64::from(index)))
    }

    fn base_seed(&self) -> u64 {
        let Self::SequentialWrappingSeeds { base_seed, .. } = *self;
        base_seed
    }
}
/// Family-resolved packing, solver and schedule for one image pass. The compiler
/// validates these values before the provider can materialize its components.
pub struct PackedImagePass<'a> {
    pub pixel_stride: u32,
    pub token_width: u32,
    pub sampler: &'a str,
    pub guidance: f32,
    pub sigmas: Vec<f32>,
    pub executed_steps: usize,
    pub capture_sigma: f32,
}

/// Complete resolved base image pass. Tensor operations, callbacks and mutable trajectory
/// state stay with the executor. Only the plan can travel across the caller/owner boundary.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RenderPlan {
    backend: String,
    provider: String,
    inputs: ImageInputs,
    preparation: Option<RenderLoadBinding>,
    #[serde(skip)]
    exact_load: Option<ExactRenderLoad>,
    count: CountPlan,
    pass: ImagePass,
    /// Position inside a [`MultiPassRenderPlan`]; 0 for a single-pass plan.
    #[serde(skip_serializing_if = "is_first_pass")]
    pass_index: usize,
    /// True when compiled via [`Self::decode_stage`] for standalone S10 (`RenderStage::Decode`)
    /// inner-driver execution (for example PiD pixel-diffusion).
    #[serde(skip)]
    decode_only: bool,
}

fn is_first_pass(index: &usize) -> bool {
    *index == 0
}

impl RenderPlan {
    pub fn base_image(
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
        pass: PackedImagePass<'_>,
    ) -> Result<Self> {
        if !req.conditioning.is_empty() {
            return Err(Error::Unsupported(
                "base image plan needs noise-start inputs".into(),
            ));
        }
        Self::packed_image_trajectory(descriptor, req, pass)
    }

    /// Inner image trajectory for a provider that already owns its conditioning/injector.
    /// The enclosing reference route still owns preparation and admission.
    pub fn packed_image_trajectory(
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
        pass: PackedImagePass<'_>,
    ) -> Result<Self> {
        if req.phases.is_some() {
            return Err(Error::Unsupported(format!(
                "{}: multi-phase denoise (phases) is not supported on this route",
                descriptor.id
            )));
        }
        Self::packed_image_trajectory_inner(descriptor, req, pass)
    }

    /// Compile an ordinary packed image pass with its prepared sources and controls for
    /// reference strength, PAG, text style and output channels bound before placement.
    pub fn packed_image_trajectory_with_exact_load(
        descriptor: &ModelDescriptor,
        spec: &LoadSpec,
        req: &GenerationRequest,
        pass: PackedImagePass<'_>,
    ) -> Result<Self> {
        let pins = spec.prepared_file_pins();
        if !pins.is_finalized() && (pins.is_prepared() || !spec.file_source_paths().is_empty()) {
            return Err(Error::Unsupported(
                "exact image plans require finalized file source pins".into(),
            ));
        }
        let mut plan = Self::packed_image_trajectory(descriptor, req, pass)?;
        plan.inputs.exact_controls = Some(req.into());
        plan.seal_load(spec);
        Ok(plan)
    }

    /// Compile the Krea Raw phase Feature over the provider's untrimmed global schedule.
    ///
    /// The Feature owns the executed step count and capture sigma. This constructor ignores
    /// those two fields of `pass` and binds the final plan to `spec`'s ordered adapter stack.
    /// Multi-phase denoise explicitly excludes `req.use_pid` and `req.low_res_start` (Krea
    /// Raw phased adapter execution decodes through its standard VAE rather than PiD).
    pub fn krea_multiphase(
        descriptor: &ModelDescriptor,
        spec: &LoadSpec,
        req: &GenerationRequest,
        mut pass: PackedImagePass<'_>,
    ) -> Result<Self> {
        let authored = req.phases.as_ref().ok_or_else(|| {
            Error::Unsupported("Krea multi-phase plan requires requested phases".into())
        })?;
        if !req.conditioning.is_empty()
            || req.source_image.is_some()
            || req.use_pid
            || req.low_res_start.is_some()
        {
            return Err(Error::Unsupported(format!(
                "{}: multi-phase denoise requires a noise start and excludes PiD and low-res start",
                descriptor.id
            )));
        }
        pass.executed_steps = pass.sigmas.len().saturating_sub(1);
        pass.capture_sigma = 0.0;
        let mut plan = Self::packed_image_trajectory_inner(descriptor, req, pass)?;
        // This is the planner itself. Derive capabilities from the descriptor without probing
        // its registered compiler, which would recursively call this method.
        let mut route = FeatureRouteStructure::from_descriptor_and_plan(descriptor, None);
        route.has_render_plan = true;
        route.has_packed_image_trajectory = true;
        let composed = FeatureRegistry::standard().compose(
            &route,
            ComposedRenderPlan::from_render_plan(&route, &plan, req)?,
            &[FeatureInvocation {
                feature_id: KreaMultiphaseFeature::ID.to_owned(),
                parameters: FeatureParameters::KreaMultiphase {
                    phases: authored.clone(),
                    loaded_adapter_count: spec.adapters.len(),
                },
            }],
        )?;
        let [lowered] = composed.passes.as_slice() else {
            return Err(Error::Unsupported(
                "Krea multi-phase Feature must produce one render pass".into(),
            ));
        };
        let phases = lowered
            .phases
            .iter()
            .map(|phase| {
                let adapters = phase
                    .active_adapters
                    .iter()
                    .map(|&(slot, override_bits)| {
                        let scale = override_bits.map_or(spec.adapters[slot].scale, f32::from_bits);
                        if !scale.is_finite() {
                            return Err(Error::Unsupported(format!(
                                "{}: adapter slot {slot} has nonfinite scale",
                                descriptor.id
                            )));
                        }
                        Ok(PhaseAdapterScale::new(slot, scale))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let branches = if phase.guidance != 0.0 {
                    ConditioningBranches::PositiveAndNegative
                } else {
                    ConditioningBranches::Positive
                };
                Ok(AdapterDenoisePhase {
                    index: phase.phase_index,
                    span: phase.first_step..phase.end_step,
                    arguments: PhaseArguments::new(adapters, phase.guidance, branches),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        plan.pass.phase.end_step = lowered.executed_steps;
        plan.pass.capture_sigma = plan.pass.sigmas[lowered.executed_steps];
        plan.pass.adapter_phases = Some(AdapterPhasePlan {
            phases,
            need_negative: authored
                .iter()
                .any(|phase| phase.guidance.unwrap_or(plan.guidance()) != 0.0),
            sampling: AdapterPhaseSampling::ResetSolverAndReuseImageSeed,
            installation: AdapterPhaseInstallation::ImmutableResiduals,
        });
        plan.seal_load(spec);
        Ok(plan)
    }

    fn packed_image_trajectory_inner(
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
        pass: PackedImagePass<'_>,
    ) -> Result<Self> {
        let PackedImagePass {
            pixel_stride,
            token_width,
            sampler,
            guidance,
            sigmas,
            executed_steps,
            capture_sigma,
        } = pass;
        let has_reference_or_source = req.source_image.is_some()
            || req
                .conditioning
                .iter()
                .any(|c| matches!(c, Conditioning::Reference { .. }));
        let terminal_reference_noop = has_reference_or_source
            && sigmas.as_slice() == [0.0]
            && executed_steps == 0
            && capture_sigma == 0.0;
        if pixel_stride == 0
            || token_width == 0
            || req.count == 0
            || req.width == 0
            || req.height == 0
            || !req.width.is_multiple_of(pixel_stride)
            || !req.height.is_multiple_of(pixel_stride)
            || !guidance.is_finite()
            || !capture_sigma.is_finite()
            || (!terminal_reference_noop && (sigmas.len() < 2 || executed_steps == 0))
            || executed_steps >= sigmas.len()
            || sigmas
                .iter()
                .any(|sigma| !sigma.is_finite() || *sigma < 0.0)
            || sigmas.windows(2).any(|nodes| nodes[0] < nodes[1])
            || sigmas.last() != Some(&0.0)
            || sigmas[executed_steps] != capture_sigma
        {
            return Err(Error::Unsupported(format!(
                "{} has an invalid base image pass",
                descriptor.id
            )));
        }
        let tokens = (req.width / pixel_stride)
            .checked_mul(req.height / pixel_stride)
            .ok_or_else(|| Error::Unsupported("packed image token count overflow".into()))?;
        let low_res_start = match &req.low_res_start {
            Some(spec) => {
                Some(spec.resolve_for_descriptor(descriptor, req, executed_steps, pixel_stride)?)
            }
            None => None,
        };
        let inputs = ImageInputs::try_from(req)?;
        match &inputs.conditioning {
            PackedImageConditioning::MultiReference { .. }
                if !descriptor
                    .capabilities
                    .accepts(crate::ConditioningKind::MultiReference) =>
            {
                return Err(Error::Unsupported(
                    "packed image plan needs one supported image conditioning input".into(),
                ));
            }
            PackedImageConditioning::ControlWithReference { .. }
            | PackedImageConditioning::MultiControlWithReference { .. }
                if !(descriptor
                    .capabilities
                    .accepts(crate::ConditioningKind::Control)
                    && descriptor
                        .capabilities
                        .accepts(crate::ConditioningKind::Reference)) =>
            {
                return Err(Error::Unsupported(
                    "packed image plan needs one supported image conditioning input".into(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            backend: descriptor.backend.to_owned(),
            provider: descriptor.id.to_owned(),
            inputs,
            preparation: None,
            exact_load: None,
            count: CountPlan::SequentialWrappingSeeds {
                count: req.count,
                base_seed: req.seed.unwrap_or_else(default_seed),
            },
            pass: ImagePass {
                stages: [
                    Stage::Schedule,
                    Stage::PromptPrep,
                    Stage::Encode,
                    Stage::Init,
                    Stage::Enter,
                    Stage::Denoise,
                    Stage::Exit,
                    Stage::PostLatent,
                    Stage::Decode,
                    Stage::Output,
                ],
                sigmas,
                capture_sigma,
                phase: ImagePhase {
                    first_step: 0,
                    end_step: executed_steps,
                    sampler: sampler.to_owned(),
                    guidance,
                    stream: PackedImageStream {
                        width: req.width,
                        height: req.height,
                        patch_stride: pixel_stride,
                        tokens,
                        token_width,
                    },
                },
                low_res_start,
                inert_splits: Vec::new(),
                adapter_phases: None,
            },
            pass_index: 0,
            decode_only: false,
        })
    }

    /// Compile a standalone S10 (`RenderStage::Decode`) plan for decode-stage inner drivers
    /// (for example PiD pixel-diffusion decoding a denoised latent without an upstream denoise pass).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_stage(
        descriptor: &ModelDescriptor,
        width: u32,
        height: u32,
        count: u32,
        seed: u64,
        sampler: &str,
        sigmas: Vec<f32>,
        executed_steps: usize,
    ) -> Result<Self> {
        let capture_sigma = sigmas.get(executed_steps).copied().unwrap_or(f32::NAN);
        if width == 0
            || height == 0
            || count == 0
            || sampler.trim().is_empty()
            || executed_steps == 0
            || sigmas.len() < 2
            || executed_steps >= sigmas.len()
            || !capture_sigma.is_finite()
            || sigmas
                .iter()
                .any(|sigma| !sigma.is_finite() || *sigma < 0.0)
            || sigmas.windows(2).any(|nodes| nodes[0] < nodes[1])
        {
            return Err(Error::Unsupported(format!(
                "{} has an invalid S10 decode stage plan",
                descriptor.id
            )));
        }
        let tokens = width
            .checked_mul(height)
            .ok_or_else(|| Error::Unsupported("decode stage pixel count overflow".into()))?;
        Ok(Self {
            backend: descriptor.backend.to_owned(),
            provider: descriptor.id.to_owned(),
            inputs: ImageInputs {
                prompt_sha256: Sha256::digest(b"").into(),
                negative_prompt: None,
                width,
                height,
                count,
                requested_seed: Some(seed),
                frames: 1,
                references: 0,
                conditioning: PackedImageConditioning::None,
                source_image: None,
                low_res_start: None,
                adapter_phases: None,
                exact_controls: None,
                steps: u32::try_from(executed_steps).ok(),
                sampler: Some(sampler.to_owned()),
                scheduler: None,
                scheduler_shift: None,
                guidance: None,
                guidance_method: None,
                true_cfg_bits: None,
                image_guidance_bits: None,
                timestep_to_start_cfg: None,
                use_pid: false,
                pid_capture_sigma: None,
                enhance_prompt: false,
                enhance_temperature_bits: None,
                enhance_max_tokens: None,
            },
            preparation: None,
            exact_load: None,
            count: CountPlan::SequentialWrappingSeeds {
                count,
                base_seed: seed,
            },
            pass: ImagePass {
                stages: [
                    Stage::Schedule,
                    Stage::PromptPrep,
                    Stage::Encode,
                    Stage::Init,
                    Stage::Enter,
                    Stage::Denoise,
                    Stage::Exit,
                    Stage::PostLatent,
                    Stage::Decode,
                    Stage::Output,
                ],
                sigmas,
                capture_sigma,
                phase: ImagePhase {
                    first_step: 0,
                    end_step: executed_steps,
                    sampler: sampler.to_owned(),
                    guidance: 0.0,
                    stream: PackedImageStream {
                        width,
                        height,
                        patch_stride: 1,
                        tokens,
                        token_width: 3,
                    },
                },
                low_res_start: None,
                inert_splits: Vec::new(),
                adapter_phases: None,
            },
            pass_index: 0,
            decode_only: true,
        })
    }

    /// True when this plan was compiled specifically for standalone S10 (`RenderStage::Decode`)
    /// inner-driver execution via [`Self::decode_stage`].
    pub fn is_decode_stage(&self) -> bool {
        self.decode_only
    }

    /// Admission may configure memory controls, preview and cancellation after compilation.
    /// Output-affecting inputs of the base route must still describe this exact trajectory.
    pub fn check_request(
        &self,
        descriptor: &ModelDescriptor,
        req: &GenerationRequest,
    ) -> Result<()> {
        if self.decode_only {
            return Err(Error::Unsupported(
                "standalone decode-stage render plan does not validate a full generation request"
                    .into(),
            ));
        }
        let mut live = ImageInputs::try_from(req)?;
        if self.inputs.exact_controls.is_some() {
            live.exact_controls = Some(req.into());
        }
        if self.pass_index > 0 {
            live.low_res_start = None;
        }
        if let PackedImageConditioning::ProducedReference {
            from_pass,
            width,
            height,
            strength_bits,
        } = &self.inputs.conditioning
        {
            // The produced image is bound by shape and strength: its bytes come from the earlier
            // pass and cannot be digested before load.
            match &live.conditioning {
                PackedImageConditioning::Reference {
                    width: live_width,
                    height: live_height,
                    strength_bits: live_strength,
                    ..
                } if (live_width, live_height, live_strength) == (width, height, strength_bits) => {
                    live.conditioning = self.inputs.conditioning.clone();
                }
                _ => {
                    return Err(Error::Unsupported(format!(
                        "pass {} reference must be the {width}x{height} output of pass {from_pass} \
                         at its declared strength",
                        self.pass_index
                    )))
                }
            }
        }
        if descriptor.id != self.provider
            || descriptor.backend != self.backend
            || self.inputs != live
        {
            return Err(Error::Unsupported(
                "compiled render plan does not match this request/provider".into(),
            ));
        }
        Ok(())
    }

    /// The pass position inside a [`MultiPassRenderPlan`] (0 for a single-pass plan).
    pub fn pass_index(&self) -> usize {
        self.pass_index
    }

    /// The earlier-pass output this pass consumes as its reference, when it declares one.
    pub fn produced_reference(&self) -> Option<ProducedReference> {
        match self.inputs.conditioning {
            PackedImageConditioning::ProducedReference {
                from_pass,
                width,
                height,
                strength_bits,
            } => Some(ProducedReference {
                from_pass,
                width,
                height,
                strength: strength_bits.map(f32::from_bits),
            }),
            _ => None,
        }
    }

    /// Requests per image (the plan's sequential count).
    pub fn image_count(&self) -> u32 {
        let CountPlan::SequentialWrappingSeeds { count, .. } = self.count;
        count
    }

    /// Executed denoise steps of this pass (the progress total its sampler reports).
    pub fn executed_steps(&self) -> usize {
        self.pass.phase.end_step - self.pass.phase.first_step
    }

    /// The (width, height) of this pass's output image.
    pub fn output_size(&self) -> (u32, u32) {
        (self.pass.phase.stream.width, self.pass.phase.stream.height)
    }

    /// Packed image spatial pixel stride (e.g., 16 for 2x2-packed /8 VAE latents).
    pub fn pixel_stride(&self) -> u32 {
        self.pass.phase.stream.patch_stride
    }

    /// Packed token channel width (e.g., 64 for FLUX.1).
    pub fn token_width(&self) -> u32 {
        self.pass.phase.stream.token_width
    }

    /// Declared stage execution order for this pass.
    pub fn stages(&self) -> &[RenderStage; 10] {
        &self.pass.stages
    }

    /// Whether this pass carries reference or control conditioning.
    pub fn has_conditioning(&self) -> bool {
        !matches!(self.inputs.conditioning, PackedImageConditioning::None)
    }

    /// Number of image references used to admit this pass, including its produced reference.
    pub fn reference_count(&self) -> u32 {
        self.inputs.references
    }

    /// Replace a compiled single-reference input with a declaration that an earlier pass
    /// produces it, keeping the compiled geometry and strength.
    fn into_produced_reference(mut self, produced: ProducedReference) -> Result<Self> {
        let PackedImageConditioning::Reference {
            width,
            height,
            strength_bits,
            ..
        } = self.inputs.conditioning
        else {
            return Err(Error::Unsupported(
                "a produced reference needs a single-reference pass".into(),
            ));
        };
        if (width, height, strength_bits)
            != (
                produced.width,
                produced.height,
                produced.strength.map(f32::to_bits),
            )
        {
            return Err(Error::Unsupported(
                "the compiled reference differs from the produced reference declaration".into(),
            ));
        }
        self.inputs.conditioning = PackedImageConditioning::ProducedReference {
            from_pass: produced.from_pass,
            width,
            height,
            strength_bits,
        };
        Ok(self)
    }

    fn preparation(&self) -> Option<&RenderLoadBinding> {
        self.preparation.as_ref()
    }
    /// Bind a plan to its load contract. An exactly bound plan keeps its original source and
    /// placement. Use [`Self::rebind_load_placement`] to update its placement.
    pub fn bind_load(mut self, spec: &LoadSpec) -> Self {
        if self.exact_load.is_none() {
            self.preparation = Some(RenderLoadBinding::from_spec(spec));
        }
        self
    }

    fn seal_load(&mut self, spec: &LoadSpec) {
        self.preparation = Some(RenderLoadBinding::from_spec(spec));
        self.exact_load = Some(ExactRenderLoad(std::sync::Arc::new(spec.clone())));
    }

    /// Update an exactly bound plan with the actual loader placement. Source pins, adapter order
    /// and default scales remain as compiled.
    pub fn rebind_load_placement(mut self, spec: &LoadSpec) -> Result<Self> {
        self.check_exact_sources(spec)?;
        self.preparation = Some(RenderLoadBinding::from_spec(spec));
        Ok(self)
    }

    /// Require an exact binding and check the loaded spec, including its placement, prepared
    /// source pins, and ordered adapter identity.
    pub fn check_load_spec(&self, spec: &LoadSpec) -> Result<()> {
        self.check_exact_sources(spec)?;
        self.check_bound_load(&RenderLoadBinding::from_spec(spec))
    }

    fn check_exact_sources(&self, spec: &LoadSpec) -> Result<()> {
        let exact = self.exact_load.as_ref().ok_or_else(|| {
            Error::Unsupported("render plan requires an exact load binding".into())
        })?;
        if !exact.0.same_load_except_placement(spec) {
            return Err(Error::Unsupported(
                "render plan belongs to a different model or adapter stack".into(),
            ));
        }
        Ok(())
    }

    pub fn check_load(&self, loaded: &RenderLoadBinding) -> Result<()> {
        if self.exact_load.is_some() {
            return Err(Error::Unsupported(
                "exactly bound render plans require checking the load spec".into(),
            ));
        }
        self.check_bound_load(loaded)
    }

    /// Check `spec` using exact source/pin verification when the plan was sealed with
    /// [`Self::packed_image_trajectory_with_exact_load`], or binding equality otherwise.
    pub fn check_load_compatible(&self, spec: &LoadSpec) -> Result<()> {
        if self.exact_load.is_some() {
            self.check_load_spec(spec)
        } else {
            self.check_load(&RenderLoadBinding::from_spec(spec))
        }
    }

    fn check_bound_load(&self, loaded: &RenderLoadBinding) -> Result<()> {
        if self
            .preparation
            .as_ref()
            .is_some_and(|expected| expected != loaded)
        {
            return Err(Error::Unsupported(
                "render plan belongs to a different model or adapter stack".into(),
            ));
        }
        Ok(())
    }

    pub fn base_seed(&self) -> u64 {
        self.count.base_seed()
    }
    pub fn guidance(&self) -> f32 {
        self.pass.phase.guidance
    }
    pub fn sampler(&self) -> &str {
        &self.pass.phase.sampler
    }
    pub fn capture_sigma(&self) -> f32 {
        self.pass.capture_sigma
    }
    pub fn sigmas(&self) -> &[f32] {
        &self.pass.sigmas
    }
    pub fn packed_tokens(&self) -> usize {
        self.pass.phase.stream.tokens as usize
    }

    /// Resolved two-phase low-resolution start plan when active on this pass (`pass_index == 0`).
    pub fn low_res_start(&self) -> Option<&crate::ResolvedLowResStartPlan> {
        self.pass.low_res_start.as_ref()
    }

    /// Build the persisted [`crate::LowResStartExecutionReport`] receipt when `low_res_start` is active
    /// on this pass (`SW-6940`: `FEAT-1`, `PIPE-3`, `PIPE-4`, `PERF-3`, `E3`).
    pub fn low_res_start_execution_report(
        &self,
        dtype: &str,
        dtype_bytes: u64,
        slot_configuration: &str,
    ) -> Result<Option<crate::LowResStartExecutionReport>> {
        let Some(low_res) = &self.pass.low_res_start else {
            return Ok(None);
        };
        let solver = low_res.estimate_source.validate_sampler_name(
            &self.provider,
            Some(self.sampler()),
            self.inputs.guidance_method.as_deref(),
        )?;
        let guidance_mode = if let Some(method) = &self.inputs.guidance_method {
            method.as_str()
        } else if self.inputs.true_cfg_bits.is_some() {
            "true_cfg"
        } else if self.guidance() > 0.0 {
            "embedded_guidance"
        } else {
            "distilled_unconditional"
        };
        let report = low_res.execution_report(
            self.pass_index,
            self.denoise_sigmas(),
            solver,
            guidance_mode,
            self.guidance(),
            slot_configuration,
            dtype,
            dtype_bytes,
            1,
            self.pixel_stride(),
            self.token_width(),
        )?;
        Ok(Some(report))
    }

    /// Spatial `(width, height)` for denoise `phase` (`0..denoise_spans().len()`).
    pub fn phase_output_size(&self, phase: usize) -> (u32, u32) {
        if let Some(low_res) = &self.pass.low_res_start {
            let spans = self.denoise_spans();
            if spans.get(phase).is_some_and(|span| span.end <= low_res.k) {
                return (low_res.low_width, low_res.low_height);
            }
        }
        (self.pass.phase.stream.width, self.pass.phase.stream.height)
    }

    /// Packed token count for denoise `phase` (`0..denoise_spans().len()`).
    pub fn phase_packed_tokens(&self, phase: usize) -> usize {
        let (w, h) = self.phase_output_size(phase);
        let stride = self.pixel_stride().max(1);
        ((w / stride) * (h / stride)) as usize
    }

    pub fn denoise_sigmas(&self) -> &[f32] {
        &self.pass.sigmas[self.pass.phase.first_step..=self.pass.phase.end_step]
    }

    /// Compiled request-local adapter and guidance arguments in execution order.
    pub fn adapter_phases(&self) -> &[AdapterDenoisePhase] {
        self.pass
            .adapter_phases
            .as_ref()
            .map_or(&[], |plan| &plan.phases)
    }

    /// Preserve Krea's negative prompt encoding when any authored phase uses CFG, even
    /// when a shorter resolved schedule leaves the last authored phase with no evaluations.
    pub fn adapter_phases_need_negative(&self) -> bool {
        self.pass
            .adapter_phases
            .as_ref()
            .is_some_and(|plan| plan.need_negative)
    }

    /// A proof-only phase boundary with no changes to geometry, model, guidance or schedule.
    /// It is compiled before load; product Feature controls introduce their own validated phases.
    #[doc(hidden)]
    pub fn with_inert_split_at(mut self, step: usize) -> Result<Self> {
        if self.pass.adapter_phases.is_some() {
            return Err(Error::Unsupported(
                "inert split cannot be combined with Krea multi-phase denoise".into(),
            ));
        }
        if step <= self.pass.phase.first_step || step >= self.pass.phase.end_step {
            return Err(Error::Unsupported(
                "inert split must lie inside the executed pass steps".into(),
            ));
        }
        if self.pass.inert_splits.binary_search(&step).is_err() {
            self.pass.inert_splits.push(step);
            self.pass.inert_splits.sort_unstable();
        }
        Ok(self)
    }

    /// Half-open absolute steps in the executed pass-local schedule.
    pub fn denoise_spans(&self) -> Vec<Range<usize>> {
        if let Some(plan) = &self.pass.adapter_phases {
            return plan.phases.iter().map(AdapterDenoisePhase::span).collect();
        }
        let mut splits = self.pass.inert_splits.clone();
        if let Some(low_res) = &self.pass.low_res_start {
            if splits.binary_search(&low_res.k).is_err() {
                splits.push(low_res.k);
                splits.sort_unstable();
            }
        }
        let mut first = self.pass.phase.first_step;
        let mut spans = Vec::with_capacity(splits.len() + 1);
        for split in splits {
            spans.push(first..split);
            first = split;
        }
        spans.push(first..self.pass.phase.end_step);
        spans
    }

    fn denoise_phase_count(&self) -> usize {
        self.pass
            .adapter_phases
            .as_ref()
            .map_or_else(|| self.denoise_spans().len(), |plan| plan.phases.len())
    }
}

/// Physical geometry of the stage about to operate under the request's already selected
/// memory rung. Sequential images process one trajectory at a time, including when the
/// admission reservation covers the whole count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderPhaseIdentity {
    pub pass: usize,
    pub phase: usize,
    pub geometry: crate::MemoryGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunState {
    BeforeEncode,
    Encoding,
    Encoded,
    Rendering,
    Finished,
}

/// Recorded execution summary for an inner [`FamilyDriver`](crate::driver::FamilyDriver)
/// hosted inside S10 (`RenderStage::Decode`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DecodeDriverExecution {
    pub driver_id: &'static str,
    pub completed_outer_steps: usize,
    pub internal_evaluations: u64,
    pub boundary_evaluations: u64,
    pub stochastic_draw_index: u64,
}

struct DecodePhaseGuard<'a> {
    flag: &'a Cell<bool>,
    prev: bool,
}

impl Drop for DecodePhaseGuard<'_> {
    fn drop(&mut self) {
        self.flag.set(self.prev);
    }
}

/// Owner-local state for an invocation. A provider's existing residency driver temporarily
/// owns its text and heavy bundles. The two callbacks share one request scope without holding
/// a dynamic borrow while running provider code or progress callbacks.
pub struct RenderRun<'p, 's> {
    plan: &'p RenderPlan,
    cancel: &'p CancelFlag,
    state: Cell<RunState>,
    scope: RefCell<Option<&'s mut dyn MemoryRequestScope>>,
    next_denoise_phase: Cell<usize>,
    active_denoise_phase: Cell<Option<usize>>,
    in_decode_phase: Cell<bool>,
    decode_executions: RefCell<Vec<DecodeDriverExecution>>,
}

impl<'p, 's> RenderRun<'p, 's> {
    pub fn new(
        plan: &'p RenderPlan,
        cancel: &'p CancelFlag,
        scope: Option<&'s mut dyn MemoryRequestScope>,
    ) -> Self {
        let initial_state = if plan.decode_only {
            RunState::Rendering
        } else {
            RunState::BeforeEncode
        };
        Self {
            plan,
            cancel,
            scope: RefCell::new(scope),
            state: Cell::new(initial_state),
            next_denoise_phase: Cell::new(0),
            active_denoise_phase: Cell::new(None),
            in_decode_phase: Cell::new(false),
            decode_executions: RefCell::new(Vec::new()),
        }
    }

    fn target_phase_identity(&self) -> RenderPhaseIdentity {
        let stream = &self.plan.pass.phase.stream;
        RenderPhaseIdentity {
            pass: self.plan.pass_index,
            phase: 0,
            geometry: crate::MemoryGeometry {
                width: stream.width,
                height: stream.height,
                batch: 1,
                frames: self.plan.inputs.frames,
                reference_count: self.plan.inputs.references,
            },
        }
    }

    fn denoise_phase_identity(&self, phase: usize) -> RenderPhaseIdentity {
        let (width, height) = self.plan.phase_output_size(phase);
        RenderPhaseIdentity {
            pass: self.plan.pass_index,
            phase,
            geometry: crate::MemoryGeometry {
                width,
                height,
                batch: 1,
                frames: self.plan.inputs.frames,
                reference_count: self.plan.inputs.references,
            },
        }
    }

    fn phase<T, E: From<Error>>(
        &self,
        phase: MemoryPhase,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled.into());
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            let identity = self.target_phase_identity();
            scope.enter_render_phase(phase, &identity)?;
        }
        let _decode_guard = (phase == MemoryPhase::Decode).then(|| {
            let prev = self.in_decode_phase.replace(true);
            DecodePhaseGuard {
                flag: &self.in_decode_phase,
                prev,
            }
        });
        // A failed stage leaves teardown to the scope's owner, preserving the original error.
        let result = body()?;
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.leave_phase(phase)?;
        }
        Ok(result)
    }

    fn begin(&self, expected: RunState, next: RunState) -> Result<()> {
        if self.state.get() != expected {
            return Err(Error::Unsupported(format!(
                "render stage {next:?} cannot start from {:?}",
                self.state.get()
            )));
        }
        self.state.set(next);
        Ok(())
    }

    pub fn plan(&self) -> &RenderPlan {
        self.plan
    }

    /// Invoke a closure with the active [`MemoryRequestScope`] when present (`MEM-5`).
    pub fn with_active_scope<T>(
        &self,
        body: impl FnOnce(&mut dyn MemoryRequestScope) -> Result<T>,
    ) -> Result<Option<T>> {
        let mut guard = self.scope.borrow_mut();
        match guard.as_deref_mut() {
            Some(scope) => Ok(Some(body(scope)?)),
            None => Ok(None),
        }
    }

    /// Enter the next declared denoise span on the owner thread. A failed body leaves scope
    /// teardown to the owner, as in `phase`; the next image cannot start with an active span.
    pub fn begin_denoise_phase(&self, phase: usize) -> Result<()> {
        if self.plan.decode_only {
            return Err(Error::Unsupported(
                "decode-stage render plan cannot enter a denoise phase".into(),
            ));
        }
        if self.state.get() != RunState::Rendering
            || self.active_denoise_phase.get().is_some()
            || phase != self.next_denoise_phase.get()
            || phase >= self.plan.denoise_phase_count()
        {
            return Err(Error::Unsupported(
                "denoise phase is out of order or outside the compiled plan".into(),
            ));
        }
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.enter_render_phase(MemoryPhase::Denoise, &self.denoise_phase_identity(phase))?;
        }
        self.active_denoise_phase.set(Some(phase));
        Ok(())
    }

    pub fn end_denoise_phase(&self, phase: usize) -> Result<()> {
        if self.active_denoise_phase.get() != Some(phase) {
            return Err(Error::Unsupported(
                "denoise phase end does not match its active phase".into(),
            ));
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.leave_phase(MemoryPhase::Denoise)?;
        }
        self.active_denoise_phase.set(None);
        self.next_denoise_phase.set(phase + 1);
        Ok(())
    }

    /// Run the compiled adapter phases, threading the noisy latent through each shared sigma
    /// boundary. The backend callback runs a fresh solver over each inclusive slice, from the
    /// original image seed and local draw index zero. Any backend fence needed before leaving
    /// a phase belongs in the callback. The scope owner tears down a failed phase.
    pub fn denoise_adapter_phases<T, E: From<Error>>(
        &self,
        image_seed: u64,
        initial: T,
        mut body: impl FnMut(&AdapterDenoisePhase, &[f32], u64, T) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let plan = self.plan.pass.adapter_phases.as_ref().ok_or_else(|| {
            Error::Unsupported("render plan does not contain adapter phases".into())
        })?;
        match plan.sampling {
            AdapterPhaseSampling::ResetSolverAndReuseImageSeed => {
                let mut latent = initial;
                for phase in &plan.phases {
                    self.begin_denoise_phase(phase.index)?;
                    let sigmas = &self.plan.sigmas()[phase.span.start..=phase.span.end];
                    latent = body(phase, sigmas, image_seed, latent)?;
                    self.end_denoise_phase(phase.index)?;
                }
                Ok(latent)
            }
        }
    }

    pub fn encode<T, E: From<Error>>(
        &self,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        self.begin_conditioning()?;
        let result = body();
        match result {
            Ok(value) => {
                self.end_conditioning()?;
                Ok(value)
            }
            Err(error) => {
                // As with `phase`, the request-scope owner tears down a failed stage.
                self.state.set(RunState::Finished);
                Err(error)
            }
        }
    }

    /// Open Conditioning across a staged residency handoff. The owner can encode text, shed its
    /// weights, load the VAE/identity stack, then close this same phase after source preparation.
    pub fn begin_conditioning(&self) -> Result<()> {
        self.begin(RunState::BeforeEncode, RunState::Encoding)?;
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.enter_render_phase(MemoryPhase::Conditioning, &self.target_phase_identity())?;
        }
        Ok(())
    }

    pub fn end_conditioning(&self) -> Result<()> {
        if self.state.get() != RunState::Encoding {
            return Err(Error::Unsupported(
                "conditioning phase is not active".into(),
            ));
        }
        if self.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        if let Some(scope) = self.scope.borrow_mut().as_deref_mut() {
            scope.leave_phase(MemoryPhase::Conditioning)?;
        }
        self.state.set(RunState::Encoded);
        Ok(())
    }

    pub fn images<I, D, O, E: From<Error>>(
        &self,
        progress: &mut dyn FnMut(Progress),
        mut init: impl FnMut(u64) -> std::result::Result<I, E>,
        mut denoise: impl FnMut(u64, I, &mut dyn FnMut(Progress)) -> std::result::Result<D, E>,
        mut decode: impl FnMut(u64, D, &mut dyn FnMut(Progress)) -> std::result::Result<O, E>,
    ) -> std::result::Result<Vec<O>, E> {
        if self.plan.pass.adapter_phases.is_some() {
            return Err(
                Error::Unsupported("Krea requires phase-aware denoise execution".into()).into(),
            );
        }
        self.begin(RunState::Encoded, RunState::Rendering)?;
        let terminal_reference =
            self.plan.produced_reference().is_some() && self.plan.executed_steps() == 0;
        let images = self
            .plan
            .count
            .seeds()
            .map(|seed| {
                if self.cancel.is_cancelled() {
                    return Err(Error::Canceled.into());
                }
                let initial = init(seed)?;
                if terminal_reference {
                    // The backend still materializes the initialized reference before decoding.
                    return self.phase(MemoryPhase::Decode, || {
                        let terminal = denoise(seed, initial, progress)?;
                        decode(seed, terminal, progress)
                    });
                }
                let denoised =
                    self.phase(MemoryPhase::Denoise, || denoise(seed, initial, progress))?;
                self.phase(MemoryPhase::Decode, || decode(seed, denoised, progress))
            })
            .collect();
        self.state.set(RunState::Finished);
        images
    }

    /// The phase-aware FLUX path keeps one image trajectory while each declared denoise span
    /// enters/leaves its own memory scope through `begin_denoise_phase`/`end_denoise_phase`.
    /// Preparation and decode remain once per image, and the existing one-span `images` path
    /// remains available to callers without explicit phase driving.
    pub fn images_with_denoise_phases<I, D, O, E: From<Error>>(
        &self,
        progress: &mut dyn FnMut(Progress),
        mut init: impl FnMut(u64) -> std::result::Result<I, E>,
        mut denoise: impl FnMut(u64, I, &mut dyn FnMut(Progress)) -> std::result::Result<D, E>,
        mut decode: impl FnMut(u64, D, &mut dyn FnMut(Progress)) -> std::result::Result<O, E>,
    ) -> std::result::Result<Vec<O>, E> {
        self.begin(RunState::Encoded, RunState::Rendering)?;
        let expected_phases = self.plan.denoise_phase_count();
        let images = self
            .plan
            .count
            .seeds()
            .map(|seed| {
                if self.cancel.is_cancelled() {
                    return Err(Error::Canceled.into());
                }
                self.next_denoise_phase.set(0);
                self.active_denoise_phase.set(None);
                let initial = init(seed)?;
                let denoised = denoise(seed, initial, progress)?;
                if self.active_denoise_phase.get().is_some()
                    || self.next_denoise_phase.get() != expected_phases
                {
                    return Err(Error::Unsupported(
                        "denoise did not finish every compiled phase".into(),
                    )
                    .into());
                }
                self.phase(MemoryPhase::Decode, || decode(seed, denoised, progress))
            })
            .collect();
        self.state.set(RunState::Finished);
        images
    }

    /// Phase-aware staged trajectory where all `count` images are denoised while the denoiser
    /// is resident (each entering/leaving its declared denoise phase spans via
    /// `begin_denoise_phase`/`end_denoise_phase`), `between_denoise_and_decode` runs to shed
    /// denoiser weights and load decoder weights, and then all `count` denoised latents are
    /// decoded under `MemoryPhase::Decode`.
    pub fn denoise_then_decode_with_phases<I, D, Dec, O, E: From<Error>>(
        &self,
        progress: &mut dyn FnMut(Progress),
        mut init: impl FnMut(u64) -> std::result::Result<I, E>,
        mut denoise: impl FnMut(u64, I, &mut dyn FnMut(Progress)) -> std::result::Result<D, E>,
        between_denoise_and_decode: impl FnOnce() -> std::result::Result<Dec, E>,
        mut decode: impl FnMut(&mut Dec, u64, D, &mut dyn FnMut(Progress)) -> std::result::Result<O, E>,
    ) -> std::result::Result<Vec<O>, E> {
        self.begin(RunState::Encoded, RunState::Rendering)?;
        let expected_phases = self.plan.denoise_phase_count();
        let mut denoised_batch = Vec::with_capacity(self.plan.image_count() as usize);
        for seed in self.plan.count.seeds() {
            if self.cancel.is_cancelled() {
                self.state.set(RunState::Finished);
                return Err(Error::Canceled.into());
            }
            self.next_denoise_phase.set(0);
            self.active_denoise_phase.set(None);
            let initial = init(seed)?;
            let denoised = denoise(seed, initial, progress)?;
            if self.active_denoise_phase.get().is_some()
                || self.next_denoise_phase.get() != expected_phases
            {
                self.state.set(RunState::Finished);
                return Err(Error::Unsupported(
                    "denoise did not finish every compiled phase".into(),
                )
                .into());
            }
            denoised_batch.push((seed, denoised));
        }
        let mut decoder_ctx = between_denoise_and_decode()?;
        let mut images = Vec::with_capacity(denoised_batch.len());
        for (seed, denoised) in denoised_batch {
            if self.cancel.is_cancelled() {
                self.state.set(RunState::Finished);
                return Err(Error::Canceled.into());
            }
            let image = self.phase(MemoryPhase::Decode, || {
                decode(&mut decoder_ctx, seed, denoised, progress)
            })?;
            images.push(image);
        }
        self.state.set(RunState::Finished);
        Ok(images)
    }

    /// Transition from `Encoded` to `Rendering` when driving staged denoise/decode closures
    /// across an external residency seam (such as `Residency::run_staged_request_scoped`).
    pub fn begin_rendering(&self) -> Result<()> {
        self.begin(RunState::Encoded, RunState::Rendering)
    }

    /// Reset per-image denoise phase counters at the start of an image trajectory.
    pub fn reset_image_denoise_phases(&self) {
        self.next_denoise_phase.set(0);
        self.active_denoise_phase.set(None);
    }

    /// Verify that an image trajectory completed all compiled denoise phases.
    pub fn finish_image_denoise_phases(&self) -> Result<()> {
        let expected_phases = self.plan.denoise_phase_count();
        if self.active_denoise_phase.get().is_some()
            || self.next_denoise_phase.get() != expected_phases
        {
            self.state.set(RunState::Finished);
            return Err(Error::Unsupported(
                "denoise did not finish every compiled phase".into(),
            ));
        }
        Ok(())
    }

    /// Execute one image's decode body inside `MemoryPhase::Decode`.
    pub fn decode_phase<T, E: From<Error>>(
        &self,
        body: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        self.phase(MemoryPhase::Decode, body)
    }

    /// True when execution is currently inside an active S10 (`MemoryPhase::Decode`) stage.
    pub fn is_in_decode_phase(&self) -> bool {
        self.in_decode_phase.get()
    }

    /// Recorded inner-driver executions hosted inside S10 (`RenderStage::Decode`) on this run.
    pub fn decode_driver_executions(&self) -> Vec<DecodeDriverExecution> {
        self.decode_executions.borrow().clone()
    }

    /// Execute an S10 (`RenderStage::Decode`) inner [`FamilyDriver`](crate::driver::FamilyDriver)
    /// over `0..expected_steps` inside `MemoryPhase::Decode`.
    ///
    /// When already hosted inside an enclosing [`Self::decode_phase`] or `Self::images*` decode
    /// closure, reuses the active `MemoryPhase::Decode` scope without double-entering
    /// `MemoryRequestScope::enter_render_phase`. Otherwise opens and closes `MemoryPhase::Decode`
    /// via `self.phase`, leaving teardown to the scope owner on error.
    pub fn decode_phase_with_driver_with<O, E: From<Error>>(
        &self,
        support: &crate::driver::DriverSupport,
        expected_steps: usize,
        body: impl FnOnce(&mut crate::driver::DriverRunState) -> std::result::Result<O, E>,
    ) -> std::result::Result<O, E> {
        if self.state.get() != RunState::Rendering {
            return Err(Error::Unsupported(format!(
                "decode inner driver cannot start from {:?}",
                self.state.get()
            ))
            .into());
        }
        if self.active_denoise_phase.get().is_some() {
            return Err(Error::Unsupported(
                "decode inner driver cannot start while a denoise phase is active".into(),
            )
            .into());
        }
        if expected_steps == 0 {
            return Err(Error::Unsupported(format!(
                "{}: decode inner driver requires at least 1 outer step",
                support.driver_id
            ))
            .into());
        }
        if self.plan.is_decode_stage() && expected_steps != self.plan.executed_steps() {
            return Err(Error::Unsupported(format!(
                "{}: decode inner driver expected {expected_steps} steps but S10 plan compiled {}",
                support.driver_id,
                self.plan.executed_steps()
            ))
            .into());
        }
        let execute = || -> std::result::Result<O, E> {
            if self.cancel.is_cancelled() {
                return Err(Error::Canceled.into());
            }
            let mut driver_state = crate::driver::DriverRunState::new();
            let output = body(&mut driver_state)?;
            if self.cancel.is_cancelled() {
                return Err(Error::Canceled.into());
            }
            if driver_state.completed_outer_steps != expected_steps {
                return Err(Error::Unsupported(format!(
                    "{}: decode inner driver completed {} outer steps, expected {}",
                    support.driver_id, driver_state.completed_outer_steps, expected_steps
                ))
                .into());
            }
            self.decode_executions
                .borrow_mut()
                .push(DecodeDriverExecution {
                    driver_id: support.driver_id,
                    completed_outer_steps: driver_state.completed_outer_steps,
                    internal_evaluations: driver_state.internal_evaluations,
                    boundary_evaluations: driver_state.boundary_evaluations,
                    stochastic_draw_index: driver_state.stochastic_draw_index,
                });
            Ok(output)
        };
        if self.in_decode_phase.get() {
            execute()
        } else {
            self.phase(MemoryPhase::Decode, execute)
        }
    }

    /// Execute a generic S10 (`RenderStage::Decode`) inner [`FamilyDriver`](crate::driver::FamilyDriver)
    /// over `0..total_steps` inside `MemoryPhase::Decode` using [`crate::driver::run_driver_slices`].
    pub fn decode_phase_with_driver<L, D: crate::driver::FamilyDriver<L> + ?Sized>(
        &self,
        driver: &D,
        initial_state: crate::BoundaryState<L>,
        total_steps: usize,
        seed: u64,
        hooks: &mut crate::driver::StepHooks<'_, L>,
    ) -> Result<crate::BoundaryState<L>> {
        self.decode_phase_with_driver_with(driver.support(), total_steps, |driver_state| {
            let mut spec = crate::driver::SliceSpec::full(total_steps);
            spec.pass = self.plan.pass_index();
            crate::driver::run_driver_slices(
                driver,
                [spec],
                initial_state,
                total_steps,
                seed,
                self.cancel,
                None,
                driver_state,
                hooks,
                |_, out, _| Ok(out),
            )
        })
    }

    /// Complete the S10 output lifecycle (`RenderStage::Output`) and mark the render run finished.
    pub fn finish_output<O, E: From<Error>>(&self, output: O) -> std::result::Result<O, E> {
        if self.state.get() != RunState::Rendering {
            return Err(Error::Unsupported(format!(
                "render stage Output cannot finish from {:?}",
                self.state.get()
            ))
            .into());
        }
        if self.active_denoise_phase.get().is_some() || self.in_decode_phase.get() {
            return Err(Error::Unsupported(
                "render stage Output cannot finish while another stage is active".into(),
            )
            .into());
        }
        if self.cancel.is_cancelled() {
            self.state.set(RunState::Finished);
            return Err(Error::Canceled.into());
        }
        self.state.set(RunState::Finished);
        Ok(output)
    }

    /// Mark the render run finished after staged decode completes.
    pub fn finish_rendering(&self) {
        self.state.set(RunState::Finished);
    }
}

#[cfg(test)]
mod tests;
