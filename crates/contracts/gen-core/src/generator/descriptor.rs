//! `ModelDescriptor` and the descriptor-level floors and support facts.

use super::*;

/// What kind of media a model emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Image,
    Video,
    /// Emits both image and video (e.g. the SeedVR2 upscaler over either).
    Both,
    /// Pure audio synthesis — TTS / music (sc-12834). Emits [`GenerationOutput::Audio`];
    /// `width`/`height` are unused, so audio models validate through the size-skipping floor
    /// ([`Capabilities::validate_request_audio`]).
    Audio,
}

/// A model's stable identity + advertised capabilities. Returned by `descriptor()` and also
/// constructible without loading weights (registry introspection).
#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    /// Exact text-encoder architecture/output contract for safe [`crate::LoadSpec::text_encoder`]
    /// substitution. `None` means the provider has not advertised a substitutable encoder.
    pub encoder_contract: Option<crate::EncoderContract>,
    /// The exact latent tensor emitted by the denoiser at its decoder boundary. `None` means the
    /// provider has not advertised enough information; consumers must treat it as incompatible with
    /// every decoder rather than infer compatibility from family names or channel counts.
    pub denoiser_output_latent_space: Option<&'static crate::latent::LatentSpace>,
    pub id: &'static str,
    pub family: &'static str,
    /// `"mlx"` | `"candle"` — the tensor backend whose provider crate registered this engine.
    /// Drives the worker's registry-derived capability advertisement (sc-3723); MLX families set
    /// `"mlx"`.
    pub backend: &'static str,
    pub modality: Modality,
    pub capabilities: Capabilities,
    /// **Named model components this engine requires** at load (epic 13657) — a weights-free
    /// advertisement of the extra artifacts a consumer must provision, beyond the base `weights` and
    /// the typed [`LoadSpec`](crate::LoadSpec) overlays, before calling `load`. The complement of
    /// [`LoadSpec::components`](crate::LoadSpec::components): the model declares its required ids here
    /// so SceneWorks knows what to stage (and the load fails fast if it doesn't — see
    /// [`require_component`](crate::control::require_component)), and the caller supplies each id's
    /// resolved local path in the load spec's `components` map.
    ///
    /// `Default` / the shipped value for every image/video provider and every single-file audio model
    /// is `&[]` — no extra components; the field is strictly additive. Each id is a lowercase
    /// `snake_case` registry identifier; the descriptor conformance sweep
    /// ([`model_descriptor_errors`](crate::registry::model_descriptor_errors)) requires the declared
    /// ids to be non-empty and unique. The concrete ids per model are the registry documented on
    /// [`LoadSpec::components`](crate::LoadSpec::components) (e.g. chatterbox `["perth",
    /// "voice_embedding"]`).
    pub required_components: &'static [&'static str],
    /// **Which control signals this model admits**, weights-free — the descriptor-level twin of
    /// [`ControlBranch::accepted_control_kinds`](crate::control::ControlBranch::accepted_control_kinds).
    ///
    /// [`capabilities.conditioning`](Capabilities::conditioning) says *whether* a model takes
    /// [`Conditioning::Control`] but never *which kind*, and the kind policy lived only on the
    /// loaded control struct (`&self` on a `ControlBranch`). A consumer planning a render therefore
    /// could not tell a pose-only branch from a pose/canny/depth union without loading multi-GB
    /// weights, and a depth request aimed at a pose-only branch failed inside `generate` — after
    /// residency — instead of before it.
    ///
    /// The distinction between the two `None`-ish answers is the point, because the permissive
    /// answer is the dangerous one:
    ///
    /// - `None` — **not advertised**. The model may still reject kinds at render time; a consumer
    ///   must treat control kind as unchecked. This is the `Default` for a reason: a model that
    ///   forgot to declare must not read as "accepts anything".
    /// - `Some(`[`Any`](crate::control::AcceptedControlKinds::Any)`)` — deliberately
    ///   input-agnostic, the Fun-Controlnet-Union position (pose/canny/depth share one VAE-encoded
    ///   path and differ only by the host-side preprocessor).
    /// - `Some(`[`Only`](crate::control::AcceptedControlKinds::Only)`(..))` — exactly these kinds;
    ///   anything else is rejected rather than silently coerced.
    ///
    /// A [`ControlBranch`](crate::control::ControlBranch) implementor should declare it here rather
    /// than override the trait method: the trait's default **reads this field**, so the descriptor
    /// is the single source of truth and the advertised policy cannot drift from the enforced one.
    ///
    /// [`Conditioning::Control`]: Conditioning::Control
    pub control_kinds: Option<crate::control::AcceptedControlKinds>,
}

impl ModelDescriptor {
    /// Alternate decoders this exact provider has wired and whose input space is compatible with its
    /// advertised denoiser output. Missing/learned normalization evidence therefore fails closed.
    pub fn compatible_decoder_options(&self) -> Vec<crate::latent::DecoderOption> {
        crate::latent::DECODER_OPTIONS
            .iter()
            .copied()
            .filter(|option| option.eligible_backends.contains(&self.backend))
            .filter(|option| option.eligible_provider_ids.contains(&self.id))
            .filter(|option| {
                crate::latent::latent_spaces_compatible(
                    self.denoiser_output_latent_space,
                    Some(option.input_latent_space),
                )
            })
            .collect()
    }
}

/// How a model's advertised size range is enforced.
///
/// The distinction already existed as two entry points —
/// [`Capabilities::validate_request`] and
/// [`Capabilities::validate_request_skip_size`] — with the choice made by the
/// provider and advertised nowhere. This makes it readable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SizeFloor {
    /// `width`/`height` must lie within `min_size..=max_size`. The common case.
    #[default]
    RangeChecked,
    /// The ordinary range check plus a required grid for every explicit `width`/`height`.
    ///
    /// A request is accepted only when both dimensions are multiples of `multiple`. This is
    /// advertised here, rather than hidden in provider code, so a weights-free consumer can predict
    /// whether the engine will render the exact geometry it was asked for.
    RangeCheckedOnGrid { multiple: u32 },
    /// `width`/`height == 0` is a "resolve from the driving media" sentinel, and the shared range
    /// check does not apply **to that pair of values** (F-158). Any other size — including a
    /// half-sentinel like `0x512` — is still range-checked normally.
    ///
    /// # What this does not promise
    ///
    /// It says nothing about the size the provider *resolves* to. A consumer must not read this
    /// variant as "the resolved geometry is bounded by `min_size..=max_size`": whether a provider
    /// re-checks after resolving is per-provider and is **not advertised here**.
    ///
    /// SCAIL-2, the only model using this resolved-downstream policy family, sets
    /// [`SizeFloor::ResolvedDownstreamExplicitGrid`] on both backends and does re-check — it refuses a
    /// resolved geometry outside `min_size..=max_size` or over its area cap before the render, naming
    /// the largest in-envelope geometry at the source aspect. But that is SCAIL-2's own guarantee,
    /// made by each SCAIL-2 provider, **not** something this variant asserts on behalf of a provider that
    /// sets it later. A consumer needing the bound must still read the provider, or treat the
    /// resolved size as unbounded.
    ///
    /// The Candle provider adopted the same safe sentinel policy in sc-16199; before that change it
    /// declared [`SizeFloor::RangeCheckedOnGrid`] and its resolve-from-the-clip branch was unreachable.
    ResolvedDownstream,
    /// [`ResolvedDownstream`](Self::ResolvedDownstream), with an additional grid requirement for
    /// **explicit** dimensions.
    ///
    /// The `0x0` sentinel remains exempt because the caller did not choose the source-media
    /// geometry; the provider may align that resolved geometry as part of its documented downstream
    /// policy. Any non-sentinel request must already land on `multiple`, so it is either rendered
    /// exactly or rejected before generation. SCAIL-2 uses this to preserve ordinary `640x360`
    /// driving clips while refusing an explicit `1280x730` instead of silently rendering
    /// `1280x704` (sc-16198).
    ResolvedDownstreamExplicitGrid { multiple: u32 },
}

impl SizeFloor {
    /// The required multiple for explicit dimensions, when this floor advertises one.
    ///
    /// `None` means the shared floor has no grid opinion; a provider may still layer a model-local
    /// rule. Consumers can call this without loading weights.
    pub fn explicit_size_multiple(self) -> Option<u32> {
        match self {
            Self::RangeChecked | Self::ResolvedDownstream => None,
            Self::RangeCheckedOnGrid { multiple }
            | Self::ResolvedDownstreamExplicitGrid { multiple } => Some(multiple),
        }
    }
}

/// A model component whose resident numeric tier may deliberately differ from the tier selected for
/// the model as a whole.
///
/// This vocabulary is intentionally shared with SceneWorks' tier-integrity ledger. Providers use it
/// to expose precision floors to callers before weights are loaded, so tier selection, telemetry,
/// and memory-evidence identity do not have to infer provider-local packing exceptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrecisionFloorComponent {
    TextEncoder,
    TransformerHead,
}

impl PrecisionFloorComponent {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TextEncoder => "textEncoder",
            Self::TransformerHead => "transformerHead",
        }
    }
}

/// One worker-visible component precision floor.
///
/// When `selected_tier` is requested, the named component is resident at no less than
/// `resident_tier`. A provider that raises a component above the selected tier must declare that
/// substitution here; callers include the declaration in labels and memory evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentPrecisionFloor {
    pub component: PrecisionFloorComponent,
    pub selected_tier: Quant,
    pub resident_tier: Quant,
}

impl ComponentPrecisionFloor {
    pub const fn applies_to(self, selected: Quant) -> bool {
        matches!(
            (self.selected_tier, selected),
            (Quant::Q4, Quant::Q4)
                | (Quant::Q8, Quant::Q8)
                | (Quant::Nvfp4, Quant::Nvfp4)
                | (Quant::Fp8Rowwise, Quant::Fp8Rowwise)
        )
    }
}

/// Resolve the quant tier a provider must use for one component from its advertised floor table.
/// Providers and callers share this function so the load path cannot apply a different substitution
/// from the one visible in descriptor introspection.
pub fn effective_component_quant(
    floors: &[ComponentPrecisionFloor],
    component: PrecisionFloorComponent,
    selected: Quant,
) -> Quant {
    floors
        .iter()
        .copied()
        .find(|floor| floor.component == component && floor.applies_to(selected))
        .map_or(selected, |floor| floor.resident_tier)
}

/// Provider-owned warm activation transient measured at 1024×1024, in bytes.
///
/// `bytes_1024` is the bare engine allocation (`peak − resident`) for one warm image. It excludes
/// model weights and OS/application reserve; consumers add those independently and may scale this
/// anchor for request geometry. A route-wide anchor is valid only when measurements establish that
/// its activation high-water is tier-independent. A provider with storage- or tier-dependent
/// activation memory must omit this route-only carrier until a spec-aware contract exists. Distinct
/// edit/control routes retain their own provider ids and must register separately. Providers publish
/// only real on-device measurements at or above the observed high-water mark; no registration means
/// "unmeasured".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationMemoryAnchor {
    pub bytes_1024: u64,
}

/// Static descriptor classification for the provider's staged-residency behavior. This describes
/// physical execution independent of request-selected memory-strategy evidence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StagedResidencyAvailability {
    /// The provider neither stages components unconditionally nor offers the shared selectable
    /// sequential-residency control.
    #[default]
    Absent,
    /// The provider honors the shared selectable sequential-residency control, but does not perform
    /// the staged load/use/drop lifecycle on every request by default.
    Selectable,
    /// The provider performs a staged component load/use/drop lifecycle on every request, regardless
    /// of whether it also offers a stronger selectable sequential-residency control.
    UnconditionallyEngaged,
}

/// The denoise step counts a model can render — the ONE representation for every step-count
/// shape the catalog has needed (sc-19559).
///
/// Three shapes accreted before this type existed: a **minimum** (SceneWorks'
/// `limits.hardMinSteps`, sc-19426), an **exact set** (`supported_steps: Vec<u32>`, sc-19502),
/// and a **ceiling**, which nothing could express at all — SVD's `MAX_STEPS = 200` lived only
/// inside `mlx-gen-svd`'s and `candle-gen-svd`'s `validate`, so a consumer could not learn the
/// bound without dispatching a job. Rather than adding a fourth key, the shapes collapse here:
/// a model is unconstrained, pinned to an exact menu, or bounded by an inclusive range.
///
/// **Rejected alternatives.** A separate `max_steps: Option<u32>` beside the existing `Vec<u32>`
/// was cheaper but lets a descriptor declare `supported_steps: [8]` *and* `max_steps: 4` — a
/// contradiction only a cross-check could catch, and it leaves the minimum still unexpressible,
/// so the next model needing a floor adds a fourth key. Extending `Vec<u32>` to enumerate
/// `1..=200` as 200 elements is a set pretending to be a range: it makes the ceiling
/// undiscoverable as a *bound*, and Kolors' `1..=1100` would be an 1100-element vector in every
/// descriptor snapshot.
///
/// **Polarity.** [`Unconstrained`](Self::Unconstrained) is the `Default` and means **no
/// constraint** — deliberately the opposite of [`Capabilities::samplers`], where empty means
/// "reject any explicit value". A model opts *in* to being constrained. Inverting it would make
/// a bare `Default::default()` refuse every step count in the repo.
///
/// Only an EXPLICIT `req.steps` is judged; `None` means "the model picks its baked default" and
/// always passes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum StepSupport {
    /// No advertised constraint: any count the shared sanity caps admit.
    #[default]
    Unconstrained,
    /// The **only** legal counts. A distilled model bakes its σ waypoints into training, so the
    /// step count is not a knob at all — LTX-2.3 runs a fixed 8-step stage-1 schedule
    /// (`STAGE1_SIGMAS`, 9 waypoints) and cannot honor any other count without going
    /// out-of-distribution.
    ///
    /// An empty set would refuse every count including the model's own default, which no model
    /// wants; [`model_descriptor_errors`](crate::registry::model_descriptor_errors) rejects it as
    /// a descriptor mistake.
    Exact(Vec<u32>),
    /// Any count in `min..=max`, inclusive on both ends. This is the shape a model with a real
    /// engine bound has: SVD-XT's 200-step ceiling, ACE-Step's 200, MMAudio's 500, Kolors'
    /// 1100 train timesteps.
    ///
    /// `min > max` is an unsatisfiable declaration and is rejected by
    /// [`model_descriptor_errors`](crate::registry::model_descriptor_errors).
    Range { min: u32, max: u32 },
}

impl StepSupport {
    /// Whether this model can render exactly `steps` denoise steps.
    pub fn admits(&self, steps: u32) -> bool {
        match self {
            Self::Unconstrained => true,
            Self::Exact(counts) => counts.contains(&steps),
            Self::Range { min, max } => (*min..=*max).contains(&steps),
        }
    }

    /// The largest step count this model renders, or `None` when it advertises no ceiling.
    ///
    /// This is the weights-free ceiling read: a consumer sizing a Steps control asks the
    /// descriptor rather than dispatching a job and reading the failure. An
    /// [`Exact`](Self::Exact) menu's ceiling is its largest member; an empty menu has none.
    pub fn ceiling(&self) -> Option<u32> {
        match self {
            Self::Unconstrained => None,
            Self::Exact(counts) => counts.iter().copied().max(),
            Self::Range { max, .. } => Some(*max),
        }
    }

    /// The smallest step count this model renders, or `None` when it advertises no floor.
    ///
    /// ⚠️ This is a **bound, not a default**. A consumer must not seed a Steps control with it:
    /// omitting `steps` selects the model's own baked default, which is generally not the floor.
    pub fn floor(&self) -> Option<u32> {
        match self {
            Self::Unconstrained => None,
            Self::Exact(counts) => counts.iter().copied().min(),
            Self::Range { min, .. } => Some(*min),
        }
    }

    /// Whether this advertises no step constraint at all (the `Default`).
    pub fn is_unconstrained(&self) -> bool {
        matches!(self, Self::Unconstrained)
    }
}
