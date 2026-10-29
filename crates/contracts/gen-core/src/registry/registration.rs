//! The registration records provider crates publish: models, imported checkpoints, memory contracts and every provider kind.

use super::*;

/// The provider-owned per-component resident-weight estimate (bytes), used by pre-load fit gates for
/// staged residency (sc-10894/sc-11924). Each component defaults to its summed on-disk
/// `.safetensors` size; a provider whose load materializes a larger representation must replace that
/// component with the conservative in-memory size:
///
/// - `text_encoder` — the phase-A prompt encoder(s) that [`OffloadPolicy::Sequential`](crate::runtime::OffloadPolicy)
///   drops *before* the heavy render bundle loads (one or more, e.g. SDXL's two CLIPs, SD3's three);
/// - `dit` — the heavy transformer / U-Net (the "DiT"), the dominant render-phase component;
/// - `vae` — the autoencoder, co-resident with the DiT through the render.
///
/// Why a provider owns this rather than the consumer inferring it: the Sequential/staged peak is
/// `max(text_encoder, dit + vae)` (the encoder is freed before the renderer materializes — see
/// [`LoadPhase::Renderer`](crate::runtime::LoadPhase::Renderer)), not the resident sum. A consumer that
/// guesses the text-encoder size from `text_encoder*` subdir NAMING reads **zero** for any family whose
/// encoder is not under such a subdir — or has no separable encoder at all (a flat unified checkpoint) —
/// collapsing the staged peak back to the resident peak so no saving is ever selected. Each provider,
/// by contrast, computes the split from the exact subdir paths its own loader resolves.
///
/// The default constructors are tensor-free on-disk sums ([`crate::safetensors_dir_bytes`]) — **zero**
/// MLX allocation and no whole-file reads — so this remains safe in a pre-load gate. A component a
/// model does not have (or cannot separate) is `0`.
///
/// **On-disk byte SUMS, not load-exact.** Each field totals *every* `.safetensors` under the named
/// path(s), which can exceed what a single load materializes: one component dir may ship multiple
/// interchangeable variant files (anima's `diffusion_models/` holds the base/aesthetic/turbo DiTs, but
/// a run loads exactly one — so `dit` over-counts by the unused variants), or side-by-side dtype shards
/// (an SD3 `text_encoder_3/` carrying both f32 and fp16 double-counts). Today the worker consumes only
/// `text_encoder` plus the true whole-model total; `dit` / `vae` are **informational** for a future
/// consumer, which must treat them as an upper-bound on-disk footprint, not the resident size of one load.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerComponentBytes {
    pub text_encoder: u64,
    pub dit: u64,
    pub vae: u64,
}

impl PerComponentBytes {
    /// Best-effort footprint for a diffusers-style snapshot: sum the `.safetensors` bytes under each
    /// named component subdir of the spec's weights DIRECTORY. Each list is the exact subdir(s) the
    /// caller's own loader resolves — `["text_encoder", "text_encoder_2"]` for the two SDXL CLIPs,
    /// `["unet"]` / `["transformer"]` for the DiT, `["vae"]` — so the paths are always correct per
    /// engine. A subdir that is absent contributes `0` ([`crate::safetensors_dir_bytes`]).
    ///
    /// Each name may be a component *subdir* OR a flat component *file* ([`safetensors_path_bytes`]),
    /// so this also covers the bernini / anima flat-file layouts. Errors only when `spec.weights` is a
    /// single [`WeightsSource::File`]: a one-file checkpoint has no component tree to split (the consumer
    /// then falls back to whole-file / resident accounting).
    pub fn from_spec_subdirs(
        spec: &LoadSpec,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.as_path(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "per-component footprint requires a snapshot directory, not a single .safetensors \
                     file"
                    .to_owned(),
            )),
        };
        Ok(Self::from_root_subdirs(root, text_encoder, dit, vae))
    }

    /// Sum each component's `.safetensors` bytes under an already-resolved `root` — for a provider whose
    /// component tree is NOT directly under `spec.weights` (e.g. anima's `split_files/` nesting resolves
    /// the root itself, then names `text_encoders` / `diffusion_models` / `vae` under it). Each name is a
    /// subdir or a flat file ([`safetensors_path_bytes`]); a missing one contributes `0`. Infallible —
    /// the root is the caller's to validate.
    pub fn from_root_subdirs(
        root: &Path,
        text_encoder: &[&str],
        dit: &[&str],
        vae: &[&str],
    ) -> Self {
        let sum = |names: &[&str]| -> u64 {
            names
                .iter()
                .map(|n| safetensors_path_bytes(root.join(n)))
                .sum()
        };
        Self {
            text_encoder: sum(text_encoder),
            dit: sum(dit),
            vae: sum(vae),
        }
    }
}

/// A generator provider's registration — `descriptor` for introspection (no weights loaded),
/// `load` to construct the model, and the optional [`footprint`](Self::footprint) size seam.
/// ≈ `services.AddKeyedSingleton<IGenerator>("id", factory)`.
#[derive(Clone, Copy)]
pub struct ModelRegistration {
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
    /// Optional per-component on-disk footprint (sc-10894) — `Some` for a provider that has declared its
    /// [`PerComponentBytes`] split (via `register_generators! { … ; footprint = … }`), `None` otherwise.
    /// `None` is the default so **every** provider that does not set it registers unchanged; a consumer
    /// reaching [`footprint`](Self::footprint) then gets `Ok(None)` and falls back to its own
    /// accounting. Mirrors the [`load`](Self::load) fn-pointer shape (a spec in, a `Result` out).
    pub footprint: Option<fn(&LoadSpec) -> Result<PerComponentBytes>>,
}

/// The validated on-disk shape of a caller-owned model routed through a registered generator.
///
/// The shape is explicit because family identity alone cannot prove that a File, fused checkpoint,
/// component directory, or ComfyUI tree is loadable by the same provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportedModelSource {
    TransformerFile,
    FusedCheckpoint,
    TransformerDirectory,
    ComfyUiTree,
}

/// The request surface selected for an imported source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImportedModelOperation {
    Generate,
    Edit,
    Pose,
    MultiPhase,
}

/// A platform backend that may bind a portable checkpoint adapter to a real generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CheckpointBackend {
    Mlx,
    Candle,
}

impl CheckpointBackend {
    pub(super) fn descriptor_label(self) -> &'static str {
        match self {
            Self::Mlx => "mlx",
            Self::Candle => "candle",
        }
    }
}

/// One named on-disk dialect accepted by a family checkpoint adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointDialectRegistration {
    pub id: &'static str,
    pub source: ImportedModelSource,
}

/// Header-only tensor evidence that identifies one registered dialect.
///
/// The inspector owns wildcard/cardinality evaluation; the adapter owns the stable signature id
/// and the exact tensor names that must all be present before that signature can claim a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointSignatureRegistration {
    pub id: &'static str,
    pub dialect: &'static str,
    pub required_tensor_names: &'static [&'static str],
}

/// A component role and its accepted cardinality in the portable checkpoint graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointComponentRegistration {
    pub role: &'static str,
    pub min_count: u16,
    pub max_count: u16,
}

/// Families that may satisfy one component role's resident/base dependency.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointBaseCompatibilityRegistration {
    pub component_role: &'static str,
    pub compatible_families: &'static [&'static str],
}

/// Stable canonical key-mapping authority for one checkpoint dialect.
///
/// The mapping id names the provider-owned exhaustive mapper exercised by family migration tests;
/// callers never infer a mapper from a family allow-list.
///
/// # `plan_driven_backends`: which declarations are backed by code
///
/// A `mapping_id` is a *declaration*. Whether some crate actually ships a
/// [`LogicalKeyMapping`](crate::checkpoint_codec::LogicalKeyMapping) with that id is a separate
/// fact, and for most shipped families the answer is **no**: their loaders read the checkpoint on
/// their own native path and never compile a
/// [`LogicalWeightPlan`](crate::checkpoint_codec::LogicalWeightPlan). Declaring that difference
/// here is what lets each catalog's conformance test prove reachability in both directions —
/// a backend listed here **must** ship the impl, and a backend not listed **must not**, so neither
/// an unbacked declaration nor an undeclared plan route can appear silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointCanonicalMappingRegistration {
    pub dialect: &'static str,
    pub mapping_id: &'static str,
    /// Backends that ship a real `LogicalKeyMapping` implementation whose `mapping_id()` equals
    /// [`mapping_id`](Self::mapping_id). Empty = **loader-native**: no backend routes this dialect
    /// through the plan compiler today, and the id names the correspondence only. Must be a
    /// duplicate-free subset of the adapter's
    /// [`eligible_backends`](CheckpointAdapterRegistration::eligible_backends).
    pub plan_driven_backends: &'static [CheckpointBackend],
}

/// Stable config-recovery authority for one semantic config field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointConfigRecoveryRegistration {
    pub field: &'static str,
    pub recovery_id: &'static str,
}

/// Portable capability policy for one imported operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointAdapterCapabilityRegistration {
    pub operation: ImportedModelOperation,
    /// The binding projects the real provider descriptor instead of copying a second capability
    /// table into SceneWorks.
    pub inherit_provider_capabilities: bool,
    /// Whether a backend binding may inherit the provider's LoRA/LoKr support for this operation.
    pub supports_adapter_inheritance: bool,
}

/// A real platform-provider binding for one portable family adapter route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointBackendBindingRegistration {
    pub backend: CheckpointBackend,
    pub source: ImportedModelSource,
    pub operation: ImportedModelOperation,
    pub provider_id: &'static str,
    pub required_components: Option<&'static [&'static str]>,
    pub inherit_adapters: bool,
}

/// The explicit temporary projection into the legacy imported-model family namespace.
///
/// [`CheckpointAdapterRegistration::family`] remains the provider-truth portable identity used
/// for binding and cross-catalog conformance. This separate, validated field exists only so the
/// derived [`ImportedModelRegistration`] view can remain byte-compatible while its consumers are
/// migrated away from the legacy family spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportedModelCompatibilityProjectionRegistration {
    pub family: &'static str,
}

/// The complete portable checkpoint-adapter authority for one model family.
///
/// Platform catalogs register this metadata together with only the bindings they actually ship.
/// [`ProviderRegistryBuilder::build`] validates the portable graph, refuses missing or dangling
/// implementations, and derives the temporary [`ImportedModelRegistration`] compatibility view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointAdapterRegistration {
    pub adapter_id: &'static str,
    pub family: &'static str,
    pub compatibility_projection: ImportedModelCompatibilityProjectionRegistration,
    pub signatures: &'static [CheckpointSignatureRegistration],
    pub dialects: &'static [CheckpointDialectRegistration],
    pub component_topology: &'static [CheckpointComponentRegistration],
    pub base_compatibility: &'static [CheckpointBaseCompatibilityRegistration],
    pub canonical_mappings: &'static [CheckpointCanonicalMappingRegistration],
    pub config_recovery: &'static [CheckpointConfigRecoveryRegistration],
    pub eligible_backends: &'static [CheckpointBackend],
    pub backend_bindings: &'static [CheckpointBackendBindingRegistration],
    pub operations: &'static [ImportedModelOperation],
    pub capabilities: &'static [CheckpointAdapterCapabilityRegistration],
}

impl CheckpointAdapterRegistration {
    /// Whether two platform registrations describe the same portable checkpoint contract.
    ///
    /// Backend bindings are deliberately excluded: provider ids and supported operations may be
    /// asymmetric between MLX and Candle, while every other field remains shared family authority.
    pub fn has_same_portable_metadata(&self, other: &Self) -> bool {
        self.adapter_id == other.adapter_id
            && self.family == other.family
            && self.compatibility_projection == other.compatibility_projection
            && self.signatures == other.signatures
            && self.dialects == other.dialects
            && self.component_topology == other.component_topology
            && self.base_compatibility == other.base_compatibility
            && self.canonical_mappings == other.canonical_mappings
            && self.config_recovery == other.config_recovery
            && self.eligible_backends == other.eligible_backends
            && self.operations == other.operations
            && self.capabilities == other.capabilities
    }
}

/// Provider-owned route from one imported source shape and operation to the ordinary generator
/// registration that actually validates and loads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportedModelRegistration {
    /// Historical family spelling emitted by the adapter's explicit compatibility projection.
    /// Portable code must use [`CheckpointAdapterRegistration::family`] instead.
    pub family: &'static str,
    pub source: ImportedModelSource,
    pub operation: ImportedModelOperation,
    pub provider_id: &'static str,
    /// Source-shape-specific staged components. When present, this replaces the ordinary
    /// provider descriptor's component list because an imported loader can consume a different
    /// artifact shape from the published snapshot loader (for example an SDXL fused checkpoint
    /// still needs caller-staged tokenizer assets).
    pub required_components: Option<&'static [&'static str]>,
    /// Whether this source shape inherits the provider's LoRA/LoKr flags. Set false only for a
    /// structural loader refusal that is narrower than the ordinary provider (for example a full
    /// Mage fine-tune whose moved base weights cannot safely take an adapter fitted to the published
    /// checkpoint).
    pub inherit_adapters: bool,
}

/// Provider-owned encoder-contract alias for a real route assembled outside an ordinary
/// [`ModelRegistration`].
///
/// Bespoke edit/control routes often reuse a registered base generator's prompt encoder while
/// loading the denoiser through a platform-specific path.  The alias keeps that relationship in
/// the inference composition root: callers resolve the authored route id and never need to know or
/// hardcode which registered base provider owns its [`crate::EncoderContract`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderContractRouteRegistration {
    /// The externally routed provider id (for example a composed control route).
    pub route_id: &'static str,
    /// The ordinary registered generator whose descriptor is the sole contract oracle.
    pub provider_id: &'static str,
}

/// Optional, pre-load memory-strategy contract registration for one provider route id.
///
/// Kept separate from [`ModelRegistration`] so every existing provider remains source-compatible:
/// ordinary generators opt in with [`ProviderRegistryBuilder::register_memory_strategy`] as they
/// migrate. A platform composition root may instead use
/// [`ProviderRegistryBuilder::register_composed_memory_strategy`] for a real route assembled outside
/// a single gen-core [`Generator`] (for example a base generator plus a native control overlay).
#[derive(Clone, Copy)]
pub struct MemoryRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    /// The provider's real admission check, callable before weights are loaded. The load spec lets
    /// tier-sensitive providers reproduce the loaded generator's exact check without opening any
    /// weight files. This must be the same function the loaded [`Generator`] delegates to; registry
    /// conformance uses it to prove that a route-specific request is rejected at admission rather
    /// than later during generation.
    pub safety_check:
        fn(&LoadSpec, &MemoryProviderContract, &MemoryRunContext) -> MemorySafetyDecision,
}

/// Provider-owned, weights-free contract fixture paired with a [`MemoryRegistration`].
///
/// Catalog conformance uses this factory when required model assets are unavailable. Production
/// registry resolution never consults it and continues to call [`MemoryRegistration::contract`].
/// The fixture must preserve the route declaration and inject zero asset facts without filesystem
/// traversal.
#[derive(Clone, Copy)]
pub struct MemoryContractFixtureRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    /// Complete, weights-free registry-load surface for this provider.
    ///
    /// A caller-selected `LoadSpec` is not a contract inventory: it can hide a rung whose
    /// availability changes with numeric tier, residency policy, or materialization shape.  The
    /// provider therefore owns an explicit finite witness set.  Catalog dumps and conformance walk
    /// every witness; they never substitute one convenient default spec.
    pub surface_specs: fn() -> Vec<MemoryContractSurfaceSpec>,
}

/// Optional selector-aware resolver for a [`MemoryContractFixtureRegistration`].
///
/// Most providers can construct a weights-free contract from the witness [`LoadSpec`] alone. A
/// prepacked provider cannot: `LoadSpec::quantize` means "pack this dense source at load time", while
/// [`MemoryContractSurfaceSelector::tier`] names the already-resolved artifact tier. Those are
/// deliberately different production shapes. Registering this additive resolver lets such a
/// provider consume the explicit selector tier without overloading `LoadSpec::quantize` or teaching
/// a downstream capability consumer a provider-specific interpretation.
#[derive(Clone, Copy)]
pub struct MemoryContractSurfaceResolverRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&MemoryContractSurfaceSpec) -> Result<MemoryProviderContract>,
}

/// Explicit, weights-free proof that a [`MemoryRegistration`] has no optimized contract surface.
///
/// Resident-only providers still participate in the reconciliation gate: they must publish this
/// typed witness rather than disappearing from the inventory by omission. Every declared selector
/// is constructed and checked to ensure all non-resident strategies remain [`MemoryStrategySupport::Missing`].
#[derive(Clone, Copy)]
pub struct ResidentOnlyMemoryContractRegistration {
    pub provider_id: &'static str,
    pub contract: fn(&LoadSpec) -> Result<MemoryProviderContract>,
    pub surface_specs: fn() -> Vec<MemoryContractSurfaceSpec>,
}

/// Provider-owned, weights-free executable fixture for one implemented memory strategy.
///
/// The request is intentionally provider-owned: edit/control routes can supply their real mode and
/// conditioning shape instead of being forced through a synthetic text-to-image fixture.
pub struct MemoryBehaviorFixture {
    pub context: crate::MemoryRunContext,
    pub request: crate::GenerationRequest,
    /// Exact provider-owned load identity used by executable conformance. `None` reuses the
    /// catalog's common weights-free spec; providers with route/path-sensitive admission attach a
    /// synthetic but canonical receipt so their positive control exercises production validation.
    pub load_spec: Option<LoadSpec>,
}

impl MemoryBehaviorFixture {
    /// Build a fixture whose request mirrors `context.geometry`.
    ///
    /// All five geometry axes are carried across: width, height, batch (as
    /// [`count`](crate::GenerationRequest::count)), reference count (as that many
    /// [`Conditioning::Reference`](crate::Conditioning::Reference) entries) and
    /// [`frames`](crate::GenerationRequest::frames). The fixture is therefore self-consistent by
    /// construction — a provider re-grading the request against the geometry it just admitted sees
    /// the same numbers on both sides.
    ///
    /// **The `frames` mapping is `u32` → `Option<u32>` (sc-19591).**
    /// [`MemoryGeometry::frames`](crate::MemoryGeometry) is a concrete frame count, while
    /// [`GenerationRequest::frames`](crate::GenerationRequest::frames) is optional, where `None`
    /// means *unstated* and delegates to the provider's own default clip length — every consumer
    /// resolves it as `request.frames.unwrap_or(<that provider's default>)`, which is
    /// `default_frames` in `MlxRequestScopeCore::configure_request` and
    /// `CandleRequestScopeCore::configure_request`, and `1` in, for example,
    /// `candle_gen_flux2::memory_strategy`. A shared builder cannot know that per-provider default,
    /// so it *states* the geometry's count instead of delegating: `Some(frames)` re-reads as the
    /// admitted geometry for every provider, whereas `None` would only do so for one whose default
    /// happens to equal the geometry. A geometry of zero frames states no clip length at all and no
    /// provider can render zero frames, so it maps back to the unstated `None` rather than to an
    /// explicit zero-frame request.
    ///
    /// Providers whose frame lattice excludes 1 — `mlx-gen-minimax-h3` (`17n+5`, minimum 124
    /// frames) is the first — need only declare the frame count on `context.geometry`; no post-hoc
    /// override of `fixture.request.frames` is required. Guarded in `memory_strategy.rs`'s tests,
    /// alongside `behavior_fixture_preserves_exact_reference_cardinality`, by
    /// `behavior_fixture_propagates_a_multi_frame_geometry`,
    /// `behavior_fixture_propagates_a_single_frame_geometry` and
    /// `behavior_fixture_leaves_a_zero_frame_geometry_unstated`.
    pub fn new(context: crate::MemoryRunContext) -> Self {
        let mut request = crate::GenerationRequest {
            width: context.geometry.width,
            height: context.geometry.height,
            count: context.geometry.batch,
            frames: (context.geometry.frames > 0).then_some(context.geometry.frames),
            use_pid: context.use_pid,
            ..Default::default()
        };
        for _ in 0..context.geometry.reference_count {
            request.conditioning.push(crate::Conditioning::Reference {
                image: crate::Image {
                    width: 1,
                    height: 1,
                    pixels: vec![0, 0, 0],
                },
                strength: Some(1.0),
            });
        }
        Self {
            context,
            request,
            load_spec: None,
        }
    }

    /// Bind this positive-control fixture to its provider-owned exact load identity.
    pub fn with_load_spec(mut self, load_spec: LoadSpec) -> Self {
        self.load_spec = Some(load_spec);
        self
    }
}

pub type MemoryBehaviorBeginRequest = fn(
    &LoadSpec,
    &MemoryProviderContract,
    &crate::MemoryRunContext,
) -> Result<Option<Box<dyn crate::MemoryRequestScope>>>;

/// Additive executable conformance seam paired by `provider_id` with a [`MemoryRegistration`].
/// Existing resident-only adopters need no behavior registration; every provider advertising an
/// optimized implemented rung must register one.
#[derive(Clone, Copy)]
pub struct MemoryBehaviorRegistration {
    pub provider_id: &'static str,
    pub valid_fixtures: fn(
        &LoadSpec,
        &MemoryProviderContract,
        crate::MemoryStrategy,
    ) -> Result<Vec<MemoryBehaviorFixture>>,
    pub begin_request: MemoryBehaviorBeginRequest,
}

/// Provider-owned, weights-free activation measurements for one registered generator route.
///
/// Kept separate from [`ModelRegistration`] and [`crate::Capabilities`] so providers opt in without a
/// breaking migration of every unrelated descriptor. The anchor is a static, family-route-wide,
/// warm 1024² transient; omission is the explicit unmeasured state.
#[derive(Clone, Copy)]
pub struct ActivationMemoryRegistration {
    pub provider_id: &'static str,
    pub anchor: crate::ActivationMemoryAnchor,
}

/// Provider-owned PRE-FIT pricing of a request's reference work (sc-26241): the additional bytes
/// the resolved [`crate::MemoryReferenceResourceFacts`] need on the loaded package `spec`. Opt-in
/// and separate from [`MemoryRegistration`], like [`ActivationMemoryRegistration`]; omission is
/// the explicit unpriced state, and a request without references prices zero.
#[derive(Clone, Copy)]
pub struct ReferenceWorkspaceRegistration {
    pub provider_id: &'static str,
    pub reference_workspace_bytes:
        fn(&LoadSpec, &crate::MemoryReferenceResourceFacts) -> Result<u64>,
}

/// A transform provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct TransformRegistration {
    pub descriptor: fn() -> TransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transform>>,
}

/// An audio-transform provider's registration (parallel to [`TransformRegistration`]; sc-12839). The
/// audio sibling of the (image) transform: a non-prompt audio→audio / audio→stems transform (voice
/// conversion, stem separation, super-resolution) resolved and loaded exactly like every other kind.
#[derive(Clone, Copy)]
pub struct AudioTransformRegistration {
    pub descriptor: fn() -> AudioTransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn AudioTransform>>,
}

/// A trainer provider's registration (parallel to [`ModelRegistration`]) — `descriptor` for
/// introspection, `load` to construct the trainer with its (frozen) base model from a [`LoadSpec`].
#[derive(Clone, Copy)]
pub struct TrainerRegistration {
    pub descriptor: fn() -> TrainerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Trainer>>,
}

/// A captioner provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct CaptionerRegistration {
    pub descriptor: fn() -> CaptionerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Captioner>>,
}

/// A transcriber provider's registration (parallel to [`CaptionerRegistration`]; sc-12850). The
/// audio-to-text sibling of the captioner: an ASR provider resolved and loaded exactly like every
/// other kind.
#[derive(Clone, Copy)]
pub struct TranscriberRegistration {
    pub descriptor: fn() -> TranscriberDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transcriber>>,
}

/// An image-embedder provider's registration (parallel to [`ModelRegistration`]).
#[derive(Clone, Copy)]
pub struct ImageEmbedderRegistration {
    pub descriptor: fn() -> ImageEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn ImageEmbedder>>,
}

/// A text-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]). Used by the
/// worker's `dataset_analysis` job for caption/image alignment in CLIP's joint space.
#[derive(Clone, Copy)]
pub struct TextEmbedderRegistration {
    pub descriptor: fn() -> TextEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn TextEmbedder>>,
}

/// A voice-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]; sc-12838).
/// The audio-identity sibling of the (unregistered) face embedder: a cloned-voice speaker vector
/// that conditions TTS via [`Conditioning::VoiceEmbedding`](crate::generator::Conditioning::VoiceEmbedding).
#[derive(Clone, Copy)]
pub struct VoiceEmbedderRegistration {
    pub descriptor: fn() -> VoiceEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn VoiceEmbedder>>,
}

/// An audio-embedder provider's registration (parallel to [`ImageEmbedderRegistration`]; sc-12851).
/// The semantic audio-text (CLAP-style) sibling of the image embedder: a whole-clip vector in a
/// joint audio-text space, for retrieval / search / auto-tagging.
#[derive(Clone, Copy)]
pub struct AudioEmbedderRegistration {
    pub descriptor: fn() -> AudioEmbedderDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn AudioEmbedder>>,
}
