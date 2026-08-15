//! Tensor-neutral memory strategy and provider contract (SC-15449).
//!
//! This module is the inference-side authority for what a provider can do. Providers describe
//! structure, lifecycle hooks, formula shape, backend realization, asset facts, and the
//! calibration ABI/fingerprint. Measured coefficients and envelopes remain external evidence, and
//! the caller remains the sole owner of live-budget accounting and least-cost selection.
//!
//! The separation is deliberate: a provider safety gate may reject a shared selection, but the
//! return type cannot substitute a different strategy or numeric tier. Unknown, stale,
//! fingerprint-mismatched, or out-of-envelope evidence is therefore never turned into a claimed
//! optimized fit by provider-local policy.
//!
//! # Why `memory_strategy` and not `memory` (SC-15804)
//!
//! The contract carried an `Image*` prefix through SC-15449 because the image lane adopted it
//! first. Nothing in the four-rung ladder's mechanisms is image-specific: rung 1 sheds a
//! conditioning component, rung 2 bounds decoder scratch, rung 3 bounds attention, and rung 4
//! bounds transformer residency. Those mechanisms can apply to video and audio providers, so the
//! vocabulary is lane-neutral; that applicability does not claim adoption coverage. At the
//! sc-16595 review point (F-200, 2026-08-01 review), audio had no registered memory-strategy
//! contract. sc-16998 separately tracks the first Stable Audio 3 adoption and non-vacuous audio
//! catalog conformance.
//!
//! It is deliberately **not** the bare name `memory`. Five crates in this workspace already have a
//! `memory` module, meaning two different things: `mlx_gen::memory` is the MLX budget interface
//! (`safe_budget_gib`, `clamp_budget_to_cap`, `apply_memory_cap_env`, `MEMORY_CAP_ENV`) that MLX
//! provider files import directly, while `mlx-gen-sam2`'s `memory` is SAM2's *model* memory bank
//! (`MemoryEncoder`, `MemoryAttention`). A third, `mlx_rs::memory`, is the allocator itself
//! (`clear_cache`, `get_peak_memory`). A `gen_core::memory` would land in the same `use` block as
//! the first and read as the second. `memory_strategy` collides with none of them.
//!
//! The **types** stay bare (`MemoryStrategy`, `MemoryProviderContract`, ...): the existing bare
//! `Memory*` names in the workspace (`MemoryEncoder`, `MemoryAttention`, ...) are SAM2/model-bank
//! concepts in other crates and never appear alongside these.
//!
//! # The cost order is ordinal, and it stays backend-neutral (SC-16090)
//!
//! SC-15791 measured rung 4 at ~8.0 s per denoise step on Candle against ~0.309 s on MLX — ~26× for
//! the same rung on the same file — and asked whether the ladder's cost order must therefore become
//! per-backend, or be replaced by a selector that consumes measured per-backend cost. **Neither. The
//! order is unchanged, static, and shared.** Three independent reasons, each sufficient:
//!
//! 1. **The ladder never trades cost against cost.** The shared selector walks
//!    [`MemoryStrategy::ALL`] and takes the *first* rung whose measured peak fits the live budget —
//!    the cheapest sufficient rung. This is the one authoritative least-cost decision path.
//!    Rung 4 is therefore reached only when no cheaper rung is *selectable*, so it never wins over a
//!    cheaper rung that fits: a cost multiplier on it cannot flip that comparison in either
//!    direction. Its alternative is no render — usually a rejection, or an `Unverified` verdict where
//!    a cheaper rung's peak would have fit but its evidence was stale, out-of-envelope or
//!    fingerprint-mismatched (that path exists and is separately tested; it does not restore a
//!    cheaper *render*, only a cheaper non-answer). Nothing in the ladder consults a cost, which is
//!    why [`MemoryStrategyCapability`] carries no cost field.
//! 2. **A cost *order* is ordinal; 26× is a magnitude, and no consumer of the order reads one.** The
//!    order's whole job is "try the cheaper mechanism first", and it discharges that on any backend
//!    where rung 4 remains the most expensive rung. Per-backend *magnitudes* are already carried
//!    where they belong: per cell, as calibration evidence, keyed by backend. (SC-15791 timed no
//!    Candle rung other than 4, so that rung 4 is still last on Candle is the epic's premise and
//!    remains unmeasured; it is not evidence this decision claims.)
//! 3. **The 26× is not a backend cost.** It decomposes entirely into per-window work that is not
//!    per-window work — see [`MemoryWindowMaterialization`]. Forking the shared order would encode a
//!    fixable implementation defect into the contract 20 family stories then have to honour.
//!
//! ## Where the magnitude reaches a TIER decision, and why that is allowed (SC-16104)
//!
//! Reason 1 is a statement about the *ladder*, and it must not be over-read into "the cost never
//! matters anywhere". One layer up, a caller choosing a numeric **tier** can consume this ladder's
//! verdict as a boolean: SceneWorks' capability-downtier collapses "fits, resident" and "fits, but
//! only by engaging rung 4" into the same `Fits`, then keeps the highest-fidelity tier that fits. So
//! under an automatic tier choice, a tier whose *only* fit is rung 4 is preferred over a lower tier
//! that fits resident, and the cost of that rung is what the higher tier is bought with — tens of
//! seconds per step on q8, extrapolating z-image's measured ~1466 ms/block.
//!
//! **That is intended.** Fidelity the user asked for is not a memory lever: the catalog refuses to
//! substitute precision (tier integrity) and refuses to substitute geometry (SC-15807), and refusing
//! to substitute speed-for-fidelity is the same refusal. It does mean this deliberately **declines**
//! SC-15791's recommendation "do not enable rung 4 on q8 until the repack is hoisted" — gating q8
//! would be the substitution the rule forbids — which raises the value of the repack fix (SC-16096)
//! rather than lowering it. What remains open on SC-16104 is disclosure, not policy: nothing tells a
//! caller its render is slow *because* it is streaming blocks to hold the tier.
//!
//! # A rung declares no non-VRAM resource cost (SC-16090)
//!
//! SC-15791 also asked how a rung declares a **host RAM** cost, because its recommended Candle fix
//! (cache the repacked bytes host-side) holds ~3.16 GiB of host memory for the whole request — a
//! derived figure for a *proposed* fix, not a measurement; nothing in the spike watched host RSS —
//! while MLX's mmap realization pays nothing host-side. A rung whose resource cost changes *kind* per
//! backend cannot be expressed as one cost order.
//!
//! **No such axis is added, because the cost belongs to that one candidate fix rather than to the
//! rung.** A realization that materializes windows from device-format bytes on disk holds its host copy
//! as the mapped file — reclaimable page cache, the same kind of resource MLX's mmap is — and makes no
//! anonymous per-request allocation at all. Adding a resource-kind axis for an implementation nobody
//! has to write would be the same mistake as (3) above, one layer down.
//!
//! ## What that leaves undeclared, stated rather than glossed
//!
//! A [`MemoryWindowMaterialization::HostFormatConversion`] realization does have a host cost, and this
//! decision does **not** surface its size. The original packed Candle realization converted each
//! window from MLX-affine source tensors, allocating anonymous host memory per projection: on the
//! order of the block's unpacked codes for q4, and for q8 a **full dense f32 grid**
//! (`repack::dequant_mlx_q8_gs`), which for a 3840 × 15360 projection is a few hundred MiB. Those
//! transients were not held for the request and SC-15791 did not measure them. SC-16096 replaced that
//! Krea/Lens path with content-addressed device-format sidecars; SC-16510 completed the same transition
//! for streamed Z-Image blocks. No current media provider declares `HostFormatConversion`, but the
//! variant remains so a future converting realization must describe itself honestly rather than claim
//! a device-format transfer.
//!
//! **Revisit trigger — deliberately broad enough to catch that case.** The axis is owed if a
//! realization is measured to need host memory proportional to the model that a fit gate would have to
//! account for: whether it is held for the request or transient, reclaimable or not, and whether or not
//! a device-format alternative exists. A provider that crosses that threshold must raise the contract
//! question rather than absorb it. What is *not* owed is an axis built ahead of that measurement.

use crate::{
    weightsmeta::safetensors_path_bytes, AdapterSpec, Error, GenerationRequest, LoadShape,
    Precision, Quant, Result,
};
use sha2::{Digest, Sha256};

/// Current ABI of the provider/evidence calibration handshake.
///
/// Providers also supply a content fingerprint. The ABI changes when this contract changes how
/// formula inputs or lifecycle semantics are interpreted; the fingerprint changes whenever one
/// provider changes tensor layout, quantization floors, execution structure, or another detail that
/// invalidates its measurements.
///
/// ABI 2 added the typed load-shape axis to calibration identities, run contexts, and evidence
/// keys. ABI 3 adds the typed reference-count geometry axis. Decode-quality admission is versioned
/// independently below: adding a semantic policy must not invalidate or imply a refresh of the
/// model-memory calibration matrix.
pub const MEMORY_CALIBRATION_ABI: u32 = 3;

/// Current ABI of production-latent tiled-decode quality records.
///
/// This identity is deliberately independent from [`MEMORY_CALIBRATION_ABI`]: decode quality is a
/// semantic admission input, not a memory or performance measurement. Changing its fixture or
/// verdict shape invalidates quality records without pretending that a model-memory matrix was
/// captured or refreshed.
pub const MEMORY_DECODE_QUALITY_ABI: u32 = 2;

/// Conservative source-closure fingerprint for the production decode paths and correctness-only
/// harness that define [`MEMORY_DECODE_QUALITY_ABI`]. The repository test beside the collector
/// recursively re-derives this value from a fail-closed superset of the involved shared/provider
/// crate sources, manifests, and embedded assets. Changing any included byte invalidates receipts,
/// even when a particular route does not exercise that source.
///
/// This is intentionally independent from a Git commit: a reviewed head and its byte-identical
/// merge commit have different object ids but the same implementation. Derivation reads source
/// bytes only; it does not execute measurement code or consume measurement artifacts.
pub const MEMORY_DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT: &str =
    "63c6f89da806e02f098598c595d1e5b46a619d05f3beb269fe40334e60c39cd9";
#[cfg(test)]
const MEMORY_DECODE_QUALITY_CANONICAL_FIXTURE_SHA256: &str =
    "87e9ecca9ab5d9d3721a316c22fed63dfb8c5aac519ddb66cf0a80384425d6fb";

/// Prefix for the single-line calibration observation protocol consumed by release tooling.
///
/// The payload after this prefix is compact JSON. Keeping the version outside the JSON makes mixed
/// test output cheap to scan without accepting a legacy `SEQ_AB` line by accident.
pub const MEMORY_EVIDENCE_V1_PREFIX: &str = "MEMORY_EVIDENCE_V1 ";

/// The normative least-cost memory-strategy ladder, in selection order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum MemoryStrategy {
    Resident = 0,
    StagedResidency = 1,
    BoundedDecode = 2,
    BoundedAttention = 3,
    BoundedTransformerResidency = 4,
}

impl MemoryStrategy {
    pub const ALL: [Self; 5] = [
        Self::Resident,
        Self::StagedResidency,
        Self::BoundedDecode,
        Self::BoundedAttention,
        Self::BoundedTransformerResidency,
    ];

    /// `true` for a constrained strategy. The resident baseline is not an optimized fit.
    pub const fn is_optimized(self) -> bool {
        !matches!(self, Self::Resident)
    }

    /// The shared prerequisite graph for this rung (SC-15805, corrected by SC-15998).
    ///
    /// ```text
    /// rung 1  ->  (none)
    /// rung 2  ->  (none)
    /// rung 3  ->  (none)
    /// rung 4  ->  requires a DeferredMaterialization load shape
    /// ```
    ///
    /// That is the entire graph, and it is **declared**, never derived from this enum's numeric
    /// order. The order is a *cost* ordering (see [`MemoryStrategy::engages`]); reading it as a
    /// dependency made rungs 2 and 3 unselectable on any provider that honestly declares rung 1
    /// `Missing`, even though bounding decoder or attention scratch has nothing to do with whether
    /// the conditioning component was shed.
    ///
    /// # Why rung 4's single edge exists
    ///
    /// A block window over an already-materialized trunk bounds nothing — it *adds* a copy on top,
    /// so you pay the windowing machinery's synchronization cost for zero residency saving. That is
    /// arithmetic, not policy.
    ///
    /// # What the precondition physically is
    ///
    /// SC-15998 measured the four combinations independently on MLX/Metal. Resident phase policy
    /// plus deferred block materialization cut Z-Image's request peak, retained the warm generator,
    /// and performed no phase reloads. The previous rung-1 edge therefore encoded one provider's
    /// coupled loader shape as universal arithmetic. [`LoadShape::DeferredMaterialization`] is the
    /// actual prerequisite, independent from phase-level [`crate::OffloadPolicy`].
    ///
    /// # Why the graph is contract-owned rather than provider-declared
    ///
    /// Providers cannot replace this graph. They may only append realization-specific constraints
    /// through [`MemoryProviderContract::additional_prerequisites`], which preserves the shared
    /// deferred-materialization requirement while giving a backend-specific coupling a truthful
    /// contract seam.
    ///
    pub const fn requires(self) -> &'static [MemoryStrategyPrerequisite] {
        match self {
            Self::Resident
            | Self::StagedResidency
            | Self::BoundedDecode
            | Self::BoundedAttention => NO_PREREQUISITES,
            Self::BoundedTransformerResidency => BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES,
        }
    }

    /// Cost-order **selector policy**: whether selecting `self` also engages `rung`.
    ///
    /// This is the one legitimate reading of the enum's numeric order — it is a least-cost ladder,
    /// so *if you needed the most expensive rung, you would have taken the cheaper ones on the way
    /// up*. The order is **ordinal and shared across backends**, and stays so even where one backend's
    /// rung costs an order of magnitude more than another's: see the module docs (SC-16090). It is
    /// emphatically **not** a dependency: prerequisites live in
    /// [`MemoryStrategy::requires`] and violating one is an error, whereas this default is
    /// **defeasible**. The epic words it as *"strategies are cumulative unless a provider documents
    /// and verifies a cheaper equivalent composition"*, and
    /// [`MemoryProviderContract::engages`] is where a provider opts out: a rung it does not declare
    /// `Implemented` is not engaged, and the validator then stops requiring that rung's parameters
    /// instead of refusing the selection. A provider can therefore publish a verified cheaper
    /// composition without fighting the validator.
    ///
    /// # Rung 1 is excluded from the cost-order default (SC-15805)
    ///
    /// [`Self::StagedResidency`] is engaged **only** when the selection *is* rung 1, or when the
    /// selection's declared [`requires`](Self::requires) graph names it. It is deliberately **not**
    /// dragged in by cost order the way rungs 2 and 3 are, because the three are not the same kind
    /// of thing:
    ///
    /// - Rungs 2 and 3 bound **scratch** — activations born and dying inside one request. Turning
    ///   them on costs nothing a caller would decline, which is what makes a cumulative default
    ///   harmless there.
    /// - Rung 1 bounds **residency**, and per the epic engaging it *"may evict the warm
    ///   cross-request cache"*. That is a real cost paid by the *next* request, so applying it by
    ///   default to a selection that never needed it is a straight loss for zero benefit.
    ///
    /// This is also what SC-15805's own root cause says — *"bounding scratch is correct whether or
    /// not a residency rung engaged"* — read in the other direction, and it is what the story's AC
    /// states: *"cost-order defaults still apply (rung 4 engages 2 and 3)"*. The AC never says rung
    /// 1.
    ///
    /// **Rung 4 does not engage rung 1.** Its shared prerequisite is
    /// [`LoadShape::DeferredMaterialization`], which can coexist with resident phase policy. A
    /// provider-specific rung prerequisite is handled by [`MemoryProviderContract::engages`].
    ///
    /// [`Self::Resident`] remains engaged by every selection — it is the baseline, not an
    /// optimization, and nothing reads it as a lever.
    pub const fn engages(self, rung: Self) -> bool {
        match rung {
            // The baseline. Unchanged: `(0 <= anything)` was always true.
            Self::Resident => true,
            // Residency-bounding, and not free — explicit selection only in the shared policy.
            Self::StagedResidency => matches!(self, Self::StagedResidency),
            // Scratch-bounding and free: the plain cost-order default.
            Self::BoundedDecode | Self::BoundedAttention | Self::BoundedTransformerResidency => {
                (rung as u8) <= (self as u8)
            }
        }
    }
}

/// What a declared prerequisite demands of the rung it names (SC-15805).
///
/// The distinction is load-bearing, which is why it is a type rather than a comment: rung 4's edge
/// is an **engagement** prerequisite, not an availability one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryPrerequisiteScope {
    /// The named rung's mechanism must be active in **this same request**. A provider declaring the
    /// rung `Implemented` is necessary but *not* sufficient — the selected composition must actually
    /// engage it (see [`MemoryProviderContract::engages`]). Availability elsewhere, on another
    /// request, or on a warm generator that engaged it previously, does not satisfy this.
    EngagedInSameRequest,
}

/// One edge of the declared prerequisite graph. See [`MemoryStrategy::requires`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryStrategyPrerequisite {
    /// A strategy mechanism must engage in the stated scope.
    Rung {
        rung: MemoryStrategy,
        scope: MemoryPrerequisiteScope,
    },
    /// The generator must have been loaded with this independent materialization shape.
    LoadShape(LoadShape),
}

const NO_PREREQUISITES: &[MemoryStrategyPrerequisite] = &[];

const BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES: &[MemoryStrategyPrerequisite] =
    &[MemoryStrategyPrerequisite::LoadShape(
        LoadShape::DeferredMaterialization,
    )];

/// Static provider disposition for one rung. Dynamic verification never belongs here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryStrategySupport {
    Implemented,
    /// The architecture lacks the component or independent work the rung would optimize.
    StructurallyNotApplicable {
        reason: String,
    },
    Missing,
}

/// Provider-declared support and parameter domain for one ladder rung.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryStrategyCapability {
    pub strategy: MemoryStrategy,
    pub support: MemoryStrategySupport,
    /// Production parameter candidates accepted by the provider. Empty fields mean the strategy
    /// does not use that parameter; evidence covers only the exact values it exercised.
    pub parameters: MemoryParameterRanges,
}

impl MemoryStrategyCapability {
    /// This rung's shared cross-provider prerequisites.
    ///
    /// Provider-specific additive edges are deliberately absent here. Consumers validating a
    /// concrete provider must use [`MemoryProviderContract::requires`] so those edges are included.
    pub const fn requires(&self) -> &'static [MemoryStrategyPrerequisite] {
        self.strategy.requires()
    }
}

/// Which transformer(s) rung 4's block window applies to (SC-15794).
///
/// Rung 4 is defined as *"only an active transformer block or bounded block window is
/// wired/materialized at once"* — which says nothing about **which** transformer. It was implemented
/// for the DiT by convention, not by definition, and a text encoder is a transformer. This is a
/// **scope on rung 4, not a new rung**: a fifth rung would cost new conformance cells across every
/// catalog entry, new contract vocabulary, and an incoherent cost ordering (there is no sensible
/// request that wants rung 5 without rung 4).
///
/// Ordered cheapest-scope-first so a selector can widen the scope monotonically.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransformerComponent {
    /// The denoising transformer only — rung 4's original by-convention scope, and the default so an
    /// existing provider's behaviour is unchanged until it declares otherwise.
    #[default]
    Dit,
    /// The text encoder only. Worth scoping separately because the encoder re-materializes **once per
    /// generation** while the DiT re-materializes once per step, so the two have very different
    /// cost sides for the same mechanism.
    TextEncoder,
    /// Both transformers stream.
    Both,
}

impl TransformerComponent {
    /// Whether this scope streams the denoising transformer.
    pub const fn includes_dit(self) -> bool {
        matches!(self, Self::Dit | Self::Both)
    }

    /// Whether this scope streams the text encoder.
    pub const fn includes_text_encoder(self) -> bool {
        matches!(self, Self::TextEncoder | Self::Both)
    }
}

/// Production parameter domains. The values are candidates, not calibration evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryParameterRanges {
    pub decode_tile_edges: Vec<u32>,
    pub decode_overlaps: Vec<u32>,
    /// Exact production-latent quality verdicts for geometry-aware bounded decode. Empty preserves
    /// the legacy route-blind parameter contract. Once non-empty, absence of an exact admitted row
    /// is a refusal; a tile edge in the union above is never authority by itself.
    pub decode_geometry_policies: Vec<MemoryDecodeGeometryPolicy>,
    pub attention_chunk_sizes: Vec<u32>,
    pub transformer_window_sizes: Vec<u32>,
    /// Component scopes the provider actually implements for rung 4 (SC-15794). Empty means the
    /// provider streams only the DiT, which is how every pre-SC-15794 provider behaves.
    pub transformer_window_components: Vec<TransformerComponent>,
}

/// One decode route's exact tile domain.
///
/// A route owns one overlap and one or more tile edges. Keeping the pair together matters for
/// conformance: the route-blind [`MemoryParameterRanges`] publishes the union of all edges and
/// overlaps, while admission must reject a geometry assembled for a different route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeRouteDomain {
    pub tile_edges: Vec<u32>,
    pub tile_overlap: u32,
}

/// Native-VAE and PiD decode domains for a provider that can execute either route.
///
/// `None` on [`MemoryProviderContract::pid_decode_routes`] means the provider has no PiD route. This
/// backend-neutral declaration is deliberately expressed only in contract geometry: `gen-core`
/// knows about the request's `use_pid` choice, but it does not depend on the MLX PiD implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPidDecodeRoutes {
    pub native: MemoryDecodeRouteDomain,
    pub pid: MemoryDecodeRouteDomain,
}

/// Concrete parameters selected for one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryStrategyParameters {
    pub decode_tile_edge: Option<u32>,
    pub decode_overlap: Option<u32>,
    pub attention_chunk_size: Option<u32>,
    pub transformer_window_size: Option<u32>,
    /// Which transformer(s) the window applies to. `None` ⇒ [`TransformerComponent::Dit`], so a
    /// selection written before SC-15794 keeps its exact previous meaning.
    pub transformer_window_component: Option<TransformerComponent>,
}

impl MemoryStrategyParameters {
    /// The effective rung-4 component scope: the declared one, or the DiT-only default.
    pub fn window_component(&self) -> TransformerComponent {
        self.transformer_window_component.unwrap_or_default()
    }
}

/// Provider lifecycle phases whose scarce backend residency may be separated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryPhase {
    Conditioning,
    Denoise,
    Decode,
}

/// Lifecycle and bounded-work hooks implemented by a provider.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryLifecycleCapabilities {
    pub phases: Vec<MemoryPhase>,
    /// Completed phases are synchronized before their scarce residency is released.
    pub synchronized_phase_release: bool,
    pub decode_tiling: bool,
    pub attention_chunking: bool,
    pub transformer_window_materialization: bool,
}

/// Formula inputs whose coefficients live in manifest/generated evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryFormulaVariable {
    AssetBytes,
    PixelCount,
    LatentPixelCount,
    BatchCount,
    FrameCount,
    ConditioningTokenCount,
    OverlayBytes,
    DecodeTileArea,
    AttentionChunkSize,
    TransformerWindowSize,
}

/// Provider-owned shape of the peak-memory formula. Coefficients do not live here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryFormulaKind {
    /// Generic MLX-compatible seam: materialized asset bytes plus calibrated headroom.
    AssetBytesPlusHeadroom,
    /// One affine expression over the declared request/strategy variables.
    Affine {
        variables: Vec<MemoryFormulaVariable>,
    },
    /// Maximum of independently calibrated lifecycle phase expressions (the generalized Krea
    /// phase-curve shape, also suitable for request-aware provider estimators such as Mage).
    PhaseEnvelope {
        phases: Vec<MemoryPhase>,
        variables: Vec<MemoryFormulaVariable>,
    },
    /// A phase envelope which separately declares resident component contributions. This is a new
    /// formula shape instead of a field on [`MemoryAssetFacts`], so every existing provider keeps
    /// its source-compatible contract literal and adopts the component axis only by choosing this
    /// variant.
    ComponentPhaseEnvelope {
        phases: Vec<MemoryPhase>,
        variables: Vec<MemoryFormulaVariable>,
        resident_components: Vec<MemoryResidentComponent>,
    },
}

impl MemoryFormulaKind {
    /// Whether this formula consumes `variable`.
    pub fn uses(&self, variable: MemoryFormulaVariable) -> bool {
        match self {
            Self::AssetBytesPlusHeadroom => variable == MemoryFormulaVariable::AssetBytes,
            Self::Affine { variables }
            | Self::PhaseEnvelope { variables, .. }
            | Self::ComponentPhaseEnvelope { variables, .. } => variables.contains(&variable),
        }
    }

    fn resident_components(&self) -> &[MemoryResidentComponent] {
        match self {
            Self::ComponentPhaseEnvelope {
                resident_components,
                ..
            } => resident_components,
            _ => &[],
        }
    }
}

/// What one rung-4 window materialization physically does on a realization (SC-16090).
///
/// This is the invariant that makes the ladder's shared cost order *true* rather than assumed. Rung 4
/// re-materializes the whole transformer once per **forward**, so a window's cost is multiplied by
/// `n_blocks × forwards`: 240 times for a 30-block stack over an 8-step schedule, and 480 where true
/// CFG runs a second forward per step. Whatever a window does is paid every one of those times, so only
/// work that is *irreducibly* per-window belongs there.
///
/// # Why this is declared rather than assumed
///
/// SC-15791 measured Candle's rung 4 at ~8.0 s/step against MLX's ~0.309 s on the identical file, and
/// ~100% of the ~26× is a host-side format conversion sitting in the per-window path. Per 97.1 MiB
/// block, medians on a host whose run-to-run spread reaches ±38%:
///
/// - **~204 ms** for the leg the spike labels *host read + repack* — a mapped read plus the MLX affine
///   triple → GGML `Q4_1` conversion. The read is inside that number and is not separated out; on a
///   cold page cache it is the part that dominates, since Candle re-reads the tier every step.
/// - **62.5 ms** attributable to a device→host round trip: the packed triple is loaded onto the CUDA
///   device and `repack::q4_parts` pulls all three straight back with `to_device(&Cpu)`. This figure is
///   a **residual** (266.8 production − 204.3 corrected), not an independently timed leg.
/// - The PCIe transfer itself: **unresolvable against a ±34.9 ms noise floor.** The only irreducibly
///   per-window part is a rounding error.
///
/// The conversion is a pure deterministic function of bytes that never change, recomputed once per
/// block per forward for one answer.
///
/// So the difference between the two backends is not device residency, PCIe, or unified memory. It is
/// that MLX's window materialization reads bytes the accelerator can consume as they sit on disk,
/// and Candle's does not — yet. Nothing about Candle prevents it: the conversion is
/// content-addressed by the source tensor and can be done once, ahead of the render, into a
/// device-format artifact a window then maps and copies.
///
/// # The obligation
///
/// A rung-4 realization SHOULD be [`Self::DeviceFormatTransfer`]. [`Self::HostFormatConversion`] is
/// admitted only as a **transitional** state that names the story removing it, because a rung whose
/// per-window path recomputes model-proportional work is not the rung the ladder costs.
///
/// # Where this lives, and why not on `MemoryStrategy`
///
/// SC-16090's brief was to layer additively on [`MemoryProviderContract`] "rather than patching the
/// bare enum". The prohibition is on [`MemoryStrategy`]: that enum is contract-owned precisely so no
/// provider can opt out of arithmetic, and a rung-4 field there would change the rung for MLX too. This
/// fact is the opposite kind — a property of one provider's loader — so it is declared on the
/// realization and read through [`MemoryProviderContract::window_materialization`], mirroring how
/// [`MemoryProviderContract::engages`] layers over [`MemoryStrategy::engages`].
///
/// It is a **required** field rather than an `Option` or a defaulted one, which does mean every existing
/// construction site had to change. That is the intended cost: a default is an opt-out that reads as a
/// declaration, and the whole point is that a Candle provider adopting rung 4 cannot stay silent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryWindowMaterialization {
    /// A window is a transfer of bytes **already in the form the accelerator consumes** — a mapped
    /// read plus, where the memory is not unified, a host-to-device copy. No format conversion, no
    /// device→host round trip, no re-derivation of anything. Cost is bounded by the transfer, so the
    /// rung costs what the ladder says it costs.
    ///
    /// MLX/Metal realizations satisfy this structurally: the mmap *is* the residency.
    DeviceFormatTransfer,
    /// A window performs a **host-side format conversion** whose cost is proportional to the block's
    /// weights, on every window of every step. Non-conforming, and admitted only transitionally.
    ///
    /// `converts` names the conversion in the terms the code uses, so a reviewer can find it.
    /// `owner_story` names the story that removes it — required, and the reason this variant is not a
    /// silent opt-out: an existing provider may keep shipping while a *new* one cannot quietly
    /// reproduce the defect without pointing at its fix.
    HostFormatConversion {
        converts: String,
        owner_story: String,
    },
}

impl MemoryWindowMaterialization {
    /// Whether this realization meets rung 4's per-window cost obligation.
    pub const fn is_conforming(&self) -> bool {
        matches!(self, Self::DeviceFormatTransfer)
    }
}

/// Backend-specific realization without imposing CUDA transfer language on unified memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryBackendRealization {
    CandleCuda {
        device_residency: bool,
        host_backed_weights: bool,
        host_to_device_block_materialization: bool,
        /// What one rung-4 window materialization does here (SC-16090). Stated per provider rather
        /// than per backend: it is a property of the loader a provider's streamed path calls, and two
        /// Candle providers can differ.
        ///
        /// Deliberately has **no default**. A defaulted or `Option` field would let a provider adopt
        /// rung 4 without answering, which is the silent state this exists to remove — so a new Candle
        /// provider cannot compile until it states one. What the contract cannot do is *verify* the
        /// answer: a provider that types `DeviceFormatTransfer` over a converting loader has made a
        /// false declaration, and nothing here catches that. The compile-forced choice buys a
        /// deliberate statement a reviewer can check against the loader, not a proof.
        block_materialization: MemoryWindowMaterialization,
    },
    MlxMetal {
        bounded_wired_residency: bool,
        lazy_or_mmap_materialization: bool,
        explicit_evaluation_and_synchronization: bool,
        cache_eviction: bool,
    },
}

/// Stable backend identity used by evidence and caller-side selection.
///
/// This deliberately excludes the realization capability flags above. Those flags describe one
/// loaded provider and are covered by its calibration identity; the evidence-key axis is the
/// canonical backend family carried by persisted calibration records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryBackend {
    Candle,
    Mlx,
}

impl MemoryBackend {
    /// Stable spelling at the persisted evidence/protocol boundary.
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Mlx => "mlx",
        }
    }
}

impl MemoryBackendRealization {
    pub const fn backend_kind(&self) -> MemoryBackend {
        match self {
            Self::CandleCuda { .. } => MemoryBackend::Candle,
            Self::MlxMetal { .. } => MemoryBackend::Mlx,
        }
    }

    pub const fn backend_id(&self) -> &'static str {
        self.backend_kind().as_key()
    }

    /// This realization's rung-4 window materialization (SC-16090) — the backend-neutral question,
    /// with a per-realization answer. This is the shape the contract layers everything backend-varying
    /// in: one shared question asked of every backend, not a forked ladder.
    ///
    /// `None` means **unstated**, which is a third condition and deliberately not folded into
    /// [`MemoryWindowMaterialization::HostFormatConversion`]: an MLX realization that declares neither
    /// lazy nor mmap materialization has not told us what its windows do, and encoding that as a
    /// *declared host conversion* would both assert something unmeasured and demand an `owner_story`
    /// field `MlxMetal` does not have — leaving the provider with an error whose only named fix is
    /// impossible. Its actual fix is to declare `lazy_or_mmap_materialization`.
    ///
    /// An MLX/Metal realization that does declare it answers
    /// [`MemoryWindowMaterialization::DeviceFormatTransfer`], because there the mapped file *is* the
    /// residency and there is no conversion seam to violate the obligation.
    pub fn window_materialization(&self) -> Option<&MemoryWindowMaterialization> {
        const MAPPED: &MemoryWindowMaterialization =
            &MemoryWindowMaterialization::DeviceFormatTransfer;
        match self {
            Self::CandleCuda {
                block_materialization,
                ..
            } => Some(block_materialization),
            Self::MlxMetal {
                lazy_or_mmap_materialization: true,
                ..
            } => Some(MAPPED),
            Self::MlxMetal { .. } => None,
        }
    }
}

/// Provider-owned calibration identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryCalibrationIdentity {
    pub abi: u32,
    pub fingerprint: String,
    /// Exact intra-phase materialization shape covered by this calibration.
    pub load_shape: LoadShape,
}

impl MemoryCalibrationIdentity {
    pub fn new(fingerprint: impl Into<String>, load_shape: LoadShape) -> Self {
        Self {
            abi: MEMORY_CALIBRATION_ABI,
            fingerprint: fingerprint.into(),
            load_shape,
        }
    }
}

/// Typed identity of one resident network component.
///
/// [`TransformerComponent`] keeps its original three meanings exactly; the wrapper lets the same
/// component axis name a network that is not one of the base model's transformers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryComponentKind {
    Transformer(TransformerComponent),
    ControlBranch,
    AdapterStack,
    IpAdapter,
    IdentityEncoder,
}

/// Whether adapter factors remain independently resident after a provider finishes loading.
///
/// Dense providers may fold the factors into their base weights, while packed quantized providers
/// commonly preserve them as forward-time residuals because the packed base is immutable. Keeping
/// this distinction typed prevents a fit gate from charging every adapter file as resident bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterResidencyMode {
    Folded,
    Additive,
}

/// Load-exact resident bytes for a single-stack adapter installation.
///
/// `Some(0)` is positive evidence that factors are folded and therefore add no independent
/// residency. `None` means an additive stack was requested but at least one source could not be
/// sized; callers must fail closed instead of guessing. Multi-expert providers should account for
/// shared adapters once per simultaneously resident expert rather than calling this helper once for
/// the whole model.
pub fn adapter_stack_resident_bytes(
    adapters: &[AdapterSpec],
    mode: AdapterResidencyMode,
) -> Option<u64> {
    if adapters.is_empty() || mode == AdapterResidencyMode::Folded {
        return Some(0);
    }
    adapters.iter().try_fold(0_u64, |total, adapter| {
        let bytes = safetensors_path_bytes(&adapter.path);
        (bytes > 0).then(|| total.saturating_add(bytes))
    })
}

impl MemoryComponentKind {
    pub const fn is_auxiliary(self) -> bool {
        matches!(
            self,
            Self::ControlBranch | Self::AdapterStack | Self::IpAdapter | Self::IdentityEncoder
        )
    }
}

/// Load-exact resident bytes for one named component.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryResidentComponent {
    /// Stable provider-local identity. The kind is intentionally not the identity: MultiControlNet
    /// may hold several distinct [`MemoryComponentKind::ControlBranch`] components at once.
    pub id: String,
    pub kind: MemoryComponentKind,
    pub resident_bytes: u64,
    /// The rung that bounds this component's residency, if any. `None` means it remains fully
    /// resident for every strategy the provider declares.
    pub bounded_by: Option<MemoryStrategy>,
}

/// A scalar predicted peak with the resident auxiliary contributions exposed separately.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoryPeakBreakdown {
    predicted_peak_bytes: u64,
    /// The part of the peak not attributable to a separately declared auxiliary network. This
    /// includes base-model residency and calibrated transients; the contract does not invent a split
    /// that the provider did not declare.
    pub unattributed_bytes: u64,
    pub components: Vec<MemoryResidentComponent>,
}

impl MemoryPeakBreakdown {
    pub const fn from_unattributed(predicted_peak_bytes: u64) -> Self {
        Self {
            predicted_peak_bytes,
            unattributed_bytes: predicted_peak_bytes,
            components: Vec::new(),
        }
    }

    pub const fn predicted_peak_bytes(&self) -> u64 {
        self.predicted_peak_bytes
    }
}

/// Provider-owned, load-exact asset facts used as formula inputs.
///
/// `base_bytes` is exactly the sum of the three base-model component fields. Auxiliary networks
/// never belong in any of those four fields; they are declared once in `overlay_bytes` and, when
/// non-zero alongside both formula variables, by typed resident components.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryAssetFacts {
    pub base_bytes: u64,
    pub conditioning_bytes: u64,
    pub transformer_bytes: u64,
    pub decoder_bytes: u64,
    /// Compatibility aggregate for auxiliary resident networks. An adopting provider declares the
    /// corresponding typed contributions in [`MemoryFormulaKind::ComponentPhaseEnvelope`].
    pub overlay_bytes: u64,
}

/// Request-level cache keys must include every axis that can change residency or execution.
///
/// A warm generator is owned by one resolved provider, backend realization, and load shape, so
/// those axes are invariant inside this contract. A caller whose outer cache spans loaded
/// generators must key that outer cache by those loader identities too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCacheSemantics {
    StrategyTierParametersModeGeometryOverlayAndEngagedComposition,
}

/// Warm generators must not inherit a prior request's memory decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryWarmRunSemantics {
    RevalidateBudgetAndReapplyRequestState,
}

/// Cancellation and error cleanup must synchronize backend work before releasing active state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCleanupSemantics {
    SynchronizeAndReleaseActivePhasesAndWindows,
}

/// Runtime semantics required of every adopting provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryRuntimeSemantics {
    pub cache: MemoryCacheSemantics,
    pub warm_run: MemoryWarmRunSemantics,
    pub cancellation: MemoryCleanupSemantics,
    pub error: MemoryCleanupSemantics,
}

impl Default for MemoryRuntimeSemantics {
    fn default() -> Self {
        Self {
            cache:
                MemoryCacheSemantics::StrategyTierParametersModeGeometryOverlayAndEngagedComposition,
            warm_run: MemoryWarmRunSemantics::RevalidateBudgetAndReapplyRequestState,
            cancellation: MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
            error: MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
        }
    }
}

/// A provider-verified exception to the shared cumulative cost-order composition.
///
/// This can remove only a default engagement edge. It cannot remove a prerequisite, and the provider
/// must name the calibration or executable proof that established the cheaper composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryStrategyEngagementExclusion {
    pub selection: MemoryStrategy,
    pub excluded_rung: MemoryStrategy,
    pub evidence: String,
}

/// Representation of an explicit resident selection in [`crate::GenerationRequest::memory`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResidentRequestMemory {
    /// Leave the request memory block absent and preserve the loaded provider's historical defaults.
    #[default]
    PreserveLoadDefaults,
    /// Write `Some(GenerationMemory::default())` to disable load-time sequential/windowed defaults.
    ExplicitResident,
}

/// Static provider contract returned before weights are loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryProviderContract {
    pub provider_id: String,
    pub backend: MemoryBackendRealization,
    pub strategies: Vec<MemoryStrategyCapability>,
    /// Whether the caller declared semantic decode authority for this loaded route even when no
    /// row survives the exact tier/load-shape projection. This distinguishes a genuinely legacy
    /// empty table from a declared-but-inapplicable/refused table, which must remain fail closed.
    pub decode_geometry_policy_authoritative: bool,
    /// Exact native/PiD route split when this provider supports PiD decode. The ordinary strategy
    /// ranges publish the union; this declaration makes the route distinction visible to weights-free
    /// registry conformance and provider admission checks.
    pub pid_decode_routes: Option<MemoryPidDecodeRoutes>,
    /// The load-time materialization shape this concrete provider instance was built with.
    pub load_shape: LoadShape,
    /// Realization-specific constraints appended to the shared graph. Providers cannot remove or
    /// replace [`MemoryStrategy::requires`].
    pub additional_prerequisites: Vec<(MemoryStrategy, MemoryStrategyPrerequisite)>,
    /// Documented, provider-verified exceptions to the shared cumulative cost-order default.
    /// Prerequisite edges remain authoritative and cannot be removed through this list.
    pub default_engagement_exclusions: Vec<MemoryStrategyEngagementExclusion>,
    /// How an explicit shared-contract `Resident` selection is represented on the generation
    /// request. Most providers preserve their historical load-time defaults by leaving the memory
    /// block absent. Providers whose load defaults themselves enable sequential/windowed execution
    /// opt into an explicit all-disabled block so `Resident` can override those defaults.
    pub resident_request_memory: ResidentRequestMemory,
    pub lifecycle: MemoryLifecycleCapabilities,
    pub formula: MemoryFormulaKind,
    /// `None` is the compatibility default for a provider that has not adopted calibration yet.
    /// Such a provider can run its resident path but can never claim a verified optimized fit.
    pub calibration: Option<MemoryCalibrationIdentity>,
    pub asset_facts: MemoryAssetFacts,
    pub runtime: MemoryRuntimeSemantics,
}

impl MemoryProviderContract {
    /// Safe compatibility view for an existing provider: resident only, optimized rungs missing,
    /// and no calibration identity. This preserves existing behavior without fabricating evidence.
    pub fn compatibility_default(
        provider_id: impl Into<String>,
        backend: MemoryBackendRealization,
    ) -> Self {
        let provider_id = provider_id.into();
        Self {
            provider_id,
            backend,
            strategies: MemoryStrategy::ALL
                .into_iter()
                .map(|strategy| MemoryStrategyCapability {
                    strategy,
                    support: if strategy == MemoryStrategy::Resident {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    parameters: MemoryParameterRanges::default(),
                })
                .collect(),
            decode_geometry_policy_authoritative: false,
            pid_decode_routes: None,
            load_shape: LoadShape::EagerMaterialization,
            additional_prerequisites: Vec::new(),
            default_engagement_exclusions: Vec::new(),
            resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
            lifecycle: MemoryLifecycleCapabilities::default(),
            formula: MemoryFormulaKind::AssetBytesPlusHeadroom,
            calibration: None,
            asset_facts: MemoryAssetFacts::default(),
            runtime: MemoryRuntimeSemantics::default(),
        }
    }

    /// Add load-exact auxiliary residency to a provider's existing base prediction.
    ///
    /// Both declaration legs are required: typed components make the total decomposable, while
    /// [`MemoryFormulaVariable::OverlayBytes`] makes the compatibility aggregate load-bearing. A
    /// pre-SC-16065 provider has no typed components, so this returns the byte-identical input scalar.
    pub fn predicted_peak_from_base(&self, base_predicted_peak_bytes: u64) -> MemoryPeakBreakdown {
        let components = self.auxiliary_components().cloned().collect::<Vec<_>>();
        let auxiliary_bytes = self.auxiliary_overlay_bytes_for_prediction();
        MemoryPeakBreakdown {
            predicted_peak_bytes: base_predicted_peak_bytes.saturating_add(auxiliary_bytes),
            unattributed_bytes: base_predicted_peak_bytes,
            components,
        }
    }

    /// Decompose a scalar which already includes this provider's auxiliary residency. The scalar is
    /// never rewritten; this is the inspection seam for run contexts and evidence records.
    pub fn decompose_predicted_peak(&self, predicted_peak_bytes: u64) -> MemoryPeakBreakdown {
        let components = self.auxiliary_components().cloned().collect::<Vec<_>>();
        let auxiliary_bytes = self.auxiliary_overlay_bytes_for_prediction();
        MemoryPeakBreakdown {
            predicted_peak_bytes,
            unattributed_bytes: predicted_peak_bytes.saturating_sub(auxiliary_bytes),
            components,
        }
    }

    fn auxiliary_components(&self) -> impl Iterator<Item = &MemoryResidentComponent> {
        self.formula
            .resident_components()
            .iter()
            .filter(|component| component.kind.is_auxiliary())
    }

    /// Compatibility aggregate attributed to typed auxiliary components in peak calculations.
    ///
    /// The typed-component sum and this aggregate are intentionally separate declaration legs;
    /// conformance proves they agree. Legacy contracts with only an aggregate must not gain a new
    /// contribution merely by calling the decomposition helpers.
    fn auxiliary_overlay_bytes_for_prediction(&self) -> u64 {
        if self.auxiliary_components().next().is_some() {
            self.asset_facts.overlay_bytes
        } else {
            0
        }
    }

    /// Typed resident component declarations, empty for every non-adopting provider.
    pub fn resident_components(&self) -> &[MemoryResidentComponent] {
        self.formula.resident_components()
    }

    /// Load-exact bytes attributable to auxiliary component declarations.
    pub fn auxiliary_resident_bytes(&self) -> u64 {
        self.auxiliary_components().fold(0_u64, |total, component| {
            total.saturating_add(component.resident_bytes)
        })
    }

    /// Resident bytes safe to credit to a warm provider. A legacy declaration preserves the old
    /// `base_bytes` meaning; an adopter adds the separately declared overlay.
    pub fn total_resident_bytes(&self) -> u64 {
        if self.auxiliary_components().next().is_some() {
            self.asset_facts
                .base_bytes
                .saturating_add(self.asset_facts.overlay_bytes)
        } else {
            self.asset_facts.base_bytes
        }
    }

    pub fn capability(&self, strategy: MemoryStrategy) -> Option<&MemoryStrategyCapability> {
        self.strategies
            .iter()
            .find(|capability| capability.strategy == strategy)
    }

    /// Replace a withheld provider's route-blind rung-2 declaration with sealed, geometry-aware
    /// quality policy. An empty input is a no-op so existing defaults and estimated fallback remain
    /// byte-for-byte compatible until a packaged quality table exists.
    ///
    /// Refused rows remain published as fail-closed knowledge. Only admitted rows contribute to the
    /// selectable edge/overlap union or arm the production decode lifecycle.
    pub fn adopt_decode_geometry_policies(
        &mut self,
        expected_family: &str,
        policies: Vec<MemoryDecodeGeometryPolicy>,
    ) -> Result<bool> {
        if policies.is_empty() {
            return Ok(false);
        }
        self.decode_geometry_policy_authoritative = true;
        if expected_family.trim().is_empty()
            || policies
                .iter()
                .any(|policy| policy.family != expected_family)
        {
            return Err(Error::Unsupported(format!(
                "{}: decode-quality policy does not match expected family {:?}",
                self.provider_id, expected_family
            )));
        }
        let capability = self
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{}: decode-quality policy requires a BoundedDecode capability",
                    self.provider_id
                ))
            })?;
        let mut edges = policies
            .iter()
            .filter(|policy| matches!(policy.disposition, MemoryDecodeQualityDisposition::Admitted))
            .map(|policy| policy.tile_edge)
            .collect::<Vec<_>>();
        let mut overlaps = policies
            .iter()
            .filter(|policy| matches!(policy.disposition, MemoryDecodeQualityDisposition::Admitted))
            .map(|policy| policy.overlap)
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges.dedup();
        overlaps.sort_unstable();
        overlaps.dedup();
        let admitted = !edges.is_empty();
        capability.support = if admitted {
            MemoryStrategySupport::Implemented
        } else {
            MemoryStrategySupport::Missing
        };
        capability.parameters.decode_tile_edges = edges;
        capability.parameters.decode_overlaps = overlaps;
        capability.parameters.decode_geometry_policies = policies;
        self.lifecycle.decode_tiling = admitted;

        let mut errors = Vec::new();
        validate_decode_geometry_policies(self, &mut errors);
        if errors.is_empty() {
            Ok(admitted)
        } else {
            Err(Error::Unsupported(format!(
                "{}: invalid decode-quality policy: {}",
                self.provider_id,
                errors.join("; ")
            )))
        }
    }

    /// Resolve the exact production-latent quality row for bounded decode.
    ///
    /// `Ok(None)` means this provider has not adopted geometry-aware quality policy and retains its
    /// legacy route-blind domain. Once any policy row is declared, every coordinate is closed: an
    /// admitted row returns `Some`, while a recorded failure, an unmeasured coordinate, or a
    /// tile/overlap mismatch returns a typed refusal. The tile pair is part of the exact coordinate:
    /// a selector that has not chosen one must enumerate
    /// [`decode_geometry_policies_for_request`](Self::decode_geometry_policies_for_request) rather
    /// than rely on table order.
    pub fn decode_geometry_policies_for_request(
        &self,
        query: MemoryDecodePolicyQuery<'_>,
    ) -> Result<Vec<&MemoryDecodeGeometryPolicy>> {
        let Some(capability) = self.capability(MemoryStrategy::BoundedDecode) else {
            return Ok(Vec::new());
        };
        let policies = &capability.parameters.decode_geometry_policies;
        if policies.is_empty() {
            return Ok(Vec::new());
        }
        Ok(policies
            .iter()
            .filter(|policy| policy.matches_request_axes(self, query))
            .collect())
    }

    pub fn decode_geometry_policy(
        &self,
        query: MemoryDecodePolicyQuery<'_>,
        selected_parameters: Option<MemoryStrategyParameters>,
    ) -> Result<Option<&MemoryDecodeGeometryPolicy>> {
        let Some(capability) = self.capability(MemoryStrategy::BoundedDecode) else {
            return Ok(None);
        };
        let policies = &capability.parameters.decode_geometry_policies;
        if policies.is_empty() {
            if self.decode_geometry_policy_authoritative {
                return Err(Error::Unsupported(format!(
                    "{}: bounded decode has a declared quality scope but no applicable row for the loaded tier/load shape",
                    self.provider_id
                )));
            }
            return Ok(None);
        }
        let request_rows = self.decode_geometry_policies_for_request(query)?;
        let had_request_rows = !request_rows.is_empty();
        let selected_pair = selected_parameters
            .and_then(|parameters| parameters.decode_tile_edge.zip(parameters.decode_overlap));
        let policy = if let Some((tile_edge, overlap)) = selected_pair {
            request_rows
                .into_iter()
                .find(|policy| policy.matches_coordinate(self, query, tile_edge, overlap))
        } else if request_rows.len() == 1 {
            request_rows.into_iter().next()
        } else if request_rows.is_empty() {
            None
        } else {
            return Err(Error::Unsupported(format!(
                "{}: bounded decode quality lookup for {}x{} matches multiple tile pairs; an exact selected edge/overlap is required",
                self.provider_id, query.geometry.width, query.geometry.height,
            )));
        };
        let Some(policy) = policy else {
            if had_request_rows && selected_pair.is_some() {
                return Err(Error::Unsupported(format!(
                    "{}: bounded decode parameters {:?} do not match any admitted/refused exact tile pair for {}x{}",
                    self.provider_id,
                    selected_pair,
                    query.geometry.width,
                    query.geometry.height,
                )));
            }
            return Err(Error::Unsupported(format!(
                "{}: bounded decode has no admitted production-latent quality record for \
                 backend={} tier={:?} load_shape={:?} mode={} overlay={:?} geometry={}x{}x{} \
                 frames={} references={} use_pid={} tile_pair={:?}",
                self.provider_id,
                self.backend.backend_id(),
                query.tier,
                query.load_shape,
                query.mode_key,
                query.overlay,
                query.geometry.width,
                query.geometry.height,
                query.geometry.batch,
                query.geometry.frames,
                query.geometry.reference_count,
                query.use_pid,
                selected_pair,
            )));
        };
        if let MemoryDecodeQualityDisposition::Refused { reason } = &policy.disposition {
            return Err(Error::Unsupported(format!(
                "{}: bounded decode was refused by production-latent quality admission for \
                 {}x{}: {reason} (evidence {})",
                self.provider_id,
                query.geometry.width,
                query.geometry.height,
                policy.production_evidence_sha256,
            )));
        }
        Ok(Some(policy))
    }

    fn support(&self, strategy: MemoryStrategy) -> Option<&MemoryStrategySupport> {
        self.capability(strategy)
            .map(|capability| &capability.support)
    }

    /// Shared prerequisites followed by additive realization-specific constraints.
    pub fn requires(
        &self,
        strategy: MemoryStrategy,
    ) -> impl Iterator<Item = MemoryStrategyPrerequisite> + '_ {
        strategy.requires().iter().copied().chain(
            self.additional_prerequisites
                .iter()
                .filter_map(move |(owner, prerequisite)| {
                    (*owner == strategy).then_some(*prerequisite)
                }),
        )
    }

    /// Whether `rung`'s mechanism is active in a request that selected `strategy`.
    ///
    /// This is the **cost-order selector policy** ([`MemoryStrategy::engages`]) intersected with
    /// what this provider actually implements, and it is the seam that makes the default defeasible:
    /// a provider which does not declare a cheaper rung `Implemented` simply does not engage it, and
    /// [`MemoryProviderContract::validate_selection`] then stops requiring that rung's parameters
    /// rather than refusing the composition. It is deliberately separate from
    /// [`MemoryStrategy::requires`] — engagement is a default, a prerequisite is a constraint.
    pub fn engages(&self, strategy: MemoryStrategy, rung: MemoryStrategy) -> bool {
        let required_by_realization = self.requires(strategy).any(|prerequisite| {
            matches!(
                prerequisite,
                MemoryStrategyPrerequisite::Rung {
                    rung: required,
                    ..
                } if required == rung
            )
        });
        let excludes_default = self
            .default_engagement_exclusions
            .iter()
            .any(|exclusion| exclusion.selection == strategy && exclusion.excluded_rung == rung);
        ((strategy.engages(rung) && !excludes_default) || required_by_realization)
            && matches!(self.support(rung), Some(MemoryStrategySupport::Implemented))
    }

    /// Request-selection-aware engagement. Geometry-quality adoption makes bounded decode a
    /// conditional cost-order default: the exact tile pair is present only on selections for an
    /// admitted request coordinate. Higher independent rungs therefore remain selectable with
    /// `decode_* = None` on off-grid/refused requests, while an exact admitted pair composes bounded
    /// decode as before. Selecting `BoundedDecode` itself never makes its own mechanism optional.
    pub fn engages_selection(&self, selection: &MemorySelection, rung: MemoryStrategy) -> bool {
        let engaged = self.engages(selection.strategy, rung);
        if !engaged || rung != MemoryStrategy::BoundedDecode {
            return engaged;
        }
        let has_geometry_policy = self.decode_geometry_policy_authoritative
            || self
                .capability(MemoryStrategy::BoundedDecode)
                .is_some_and(|capability| {
                    !capability.parameters.decode_geometry_policies.is_empty()
                });
        if !has_geometry_policy || selection.strategy == MemoryStrategy::BoundedDecode {
            return true;
        }
        selection.parameters.decode_tile_edge.is_some()
            && selection.parameters.decode_overlap.is_some()
    }

    /// Canonical composition for one concrete selection, including conditional bounded-decode
    /// engagement.
    pub fn engaged_composition_for_selection(
        &self,
        selection: &MemorySelection,
    ) -> Vec<MemoryStrategy> {
        MemoryStrategy::ALL
            .into_iter()
            .filter(|rung| self.engages_selection(selection, *rung))
            .collect()
    }

    /// Canonical, ordered set of strategy mechanisms active for one selection.
    ///
    /// Evidence records this exact set at measurement time. Walking [`MemoryStrategy::ALL`] keeps
    /// the identity deterministic while routing every engagement decision through the provider
    /// contract, including realization-specific prerequisite edges.
    pub fn engaged_composition(&self, strategy: MemoryStrategy) -> Vec<MemoryStrategy> {
        MemoryStrategy::ALL
            .into_iter()
            .filter(|rung| self.engages(strategy, *rung))
            .collect()
    }

    /// Canonical translation from an admitted shared selection into the tensor-neutral request
    /// controls every provider executes. Engagement is read only through this contract, including
    /// provider prerequisite edges and verified exclusions; provider implementations must not
    /// duplicate this mapping.
    pub fn generation_memory(
        &self,
        selection: &MemorySelection,
    ) -> Option<crate::GenerationMemory> {
        if selection.strategy == MemoryStrategy::Resident {
            return match self.resident_request_memory {
                ResidentRequestMemory::PreserveLoadDefaults => None,
                ResidentRequestMemory::ExplicitResident => Some(crate::GenerationMemory::default()),
            };
        }

        let parameters = selection.parameters;
        let stage_residency = self.engages_selection(selection, MemoryStrategy::StagedResidency);
        let tile_vae_decode = self.engages_selection(selection, MemoryStrategy::BoundedDecode);
        let chunk_attention = self.engages_selection(selection, MemoryStrategy::BoundedAttention);
        let stream_transformer_blocks =
            self.engages_selection(selection, MemoryStrategy::BoundedTransformerResidency);
        Some(crate::GenerationMemory {
            stage_residency,
            tile_vae_decode,
            chunk_attention,
            stream_transformer_blocks,
            decode_tile_edge: tile_vae_decode
                .then_some(parameters.decode_tile_edge)
                .flatten(),
            decode_overlap: tile_vae_decode
                .then_some(parameters.decode_overlap)
                .flatten(),
            attention_chunk_size: chunk_attention
                .then_some(parameters.attention_chunk_size)
                .flatten(),
            transformer_window_size: stream_transformer_blocks
                .then_some(parameters.transformer_window_size)
                .flatten(),
            transformer_window_component: stream_transformer_blocks
                .then(|| parameters.window_component()),
            ..Default::default()
        })
    }

    /// Deterministic provider-declared selection used by weights-free behavioral conformance.
    pub fn representative_selection(
        &self,
        strategy: MemoryStrategy,
        tier: MemoryNumericTier,
        use_pid: bool,
    ) -> crate::Result<MemorySelection> {
        let mut parameters = MemoryStrategyParameters::default();
        if self.engages(strategy, MemoryStrategy::BoundedDecode) {
            if let Some(routes) = &self.pid_decode_routes {
                let route = if use_pid { &routes.pid } else { &routes.native };
                parameters.decode_tile_edge = route.tile_edges.first().copied();
                parameters.decode_overlap = Some(route.tile_overlap);
            } else if let Some(capability) = self.capability(MemoryStrategy::BoundedDecode) {
                parameters.decode_tile_edge =
                    capability.parameters.decode_tile_edges.first().copied();
                parameters.decode_overlap = capability.parameters.decode_overlaps.first().copied();
            }
        }
        if self.engages(strategy, MemoryStrategy::BoundedAttention) {
            parameters.attention_chunk_size = self
                .capability(MemoryStrategy::BoundedAttention)
                .and_then(|capability| {
                    capability.parameters.attention_chunk_sizes.first().copied()
                });
        }
        if self.engages(strategy, MemoryStrategy::BoundedTransformerResidency) {
            if let Some(capability) = self.capability(MemoryStrategy::BoundedTransformerResidency) {
                parameters.transformer_window_size = capability
                    .parameters
                    .transformer_window_sizes
                    .first()
                    .copied();
                parameters.transformer_window_component = capability
                    .parameters
                    .transformer_window_components
                    .first()
                    .copied();
            }
        }
        let selection = MemorySelection {
            strategy,
            parameters,
            tier,
        };
        self.validate_selection(&selection)?;
        Ok(selection)
    }

    /// What one rung-4 window materialization does for this provider (SC-16090), or `None` if the
    /// realization has not stated it.
    ///
    /// Layered on [`MemoryBackendRealization::window_materialization`] the way
    /// [`MemoryProviderContract::engages`] is layered on [`MemoryStrategy::engages`]: the contract is
    /// where a caller asks, so a later per-provider override has a seam to land in without every caller
    /// changing. It currently forwards unchanged — the fact is already per provider, since the
    /// realization is declared per provider rather than per backend.
    ///
    /// **Not `Option<bool>`**: a caller that only wants "is this conforming" still has to distinguish
    /// *declared non-conforming* from *never declared*, because the two have different fixes.
    pub fn window_materialization(&self) -> Option<&MemoryWindowMaterialization> {
        self.backend.window_materialization()
    }

    /// Static conformance errors. An empty result means the provider declaration is internally
    /// coherent; it does not mean measurements are verified.
    pub fn conformance_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.provider_id.trim().is_empty() {
            errors.push("provider_id must be non-empty".to_owned());
        }

        for strategy in MemoryStrategy::ALL {
            let count = self
                .strategies
                .iter()
                .filter(|capability| capability.strategy == strategy)
                .count();
            if count != 1 {
                errors.push(format!(
                    "strategy {strategy:?} must appear exactly once (found {count})"
                ));
            }
        }
        if self.strategies.len() != MemoryStrategy::ALL.len() {
            errors.push(format!(
                "strategy table must contain exactly five entries (found {})",
                self.strategies.len()
            ));
        }

        let implemented = |strategy| {
            matches!(
                self.capability(strategy).map(|c| &c.support),
                Some(MemoryStrategySupport::Implemented)
            )
        };
        let mut exclusions = std::collections::HashSet::new();
        for exclusion in &self.default_engagement_exclusions {
            if !exclusions.insert((exclusion.selection, exclusion.excluded_rung)) {
                errors.push(format!(
                    "duplicate default engagement exclusion {:?} -> {:?}",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if exclusion.evidence.trim().is_empty() {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} has no verification evidence",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if !exclusion.selection.engages(exclusion.excluded_rung) {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} does not remove a shared cost-order edge",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if exclusion.selection == exclusion.excluded_rung
                || exclusion.excluded_rung == MemoryStrategy::Resident
            {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} cannot remove the selected strategy or resident baseline",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if !implemented(exclusion.selection) || !implemented(exclusion.excluded_rung) {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} requires both strategies to be implemented",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
            if self.requires(exclusion.selection).any(|prerequisite| {
                matches!(
                    prerequisite,
                    MemoryStrategyPrerequisite::Rung { rung, .. }
                        if rung == exclusion.excluded_rung
                )
            }) {
                errors.push(format!(
                    "default engagement exclusion {:?} -> {:?} cannot remove a prerequisite edge",
                    exclusion.selection, exclusion.excluded_rung
                ));
            }
        }
        if implemented(MemoryStrategy::StagedResidency) {
            for phase in [
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ] {
                if !self.lifecycle.phases.contains(&phase) {
                    errors.push(format!(
                        "StagedResidency requires a {phase:?} lifecycle phase"
                    ));
                }
            }
            if !self.lifecycle.synchronized_phase_release {
                errors.push(
                    "StagedResidency requires synchronized release of completed phases".to_owned(),
                );
            }
        }
        if implemented(MemoryStrategy::BoundedDecode) && !self.lifecycle.decode_tiling {
            errors.push("BoundedDecode requires the decode_tiling hook".to_owned());
        }
        if let Some(routes) = &self.pid_decode_routes {
            validate_pid_decode_routes(self, routes, &mut errors);
        }
        validate_decode_geometry_policies(self, &mut errors);
        if implemented(MemoryStrategy::BoundedAttention) && !self.lifecycle.attention_chunking {
            errors.push("BoundedAttention requires the attention_chunking hook".to_owned());
        }
        if implemented(MemoryStrategy::BoundedTransformerResidency) {
            if !self.lifecycle.transformer_window_materialization {
                errors.push(
                    "BoundedTransformerResidency requires the transformer-window hook".to_owned(),
                );
            }
            // SC-16090. What is refused here is an UNDECLARED window realization, never a slow one.
            //
            // A provider that honestly declares `HostFormatConversion` and names its owner story has no
            // error and keeps every rung, including rung 4 — deliberately, because the selector reaches
            // rung 4 only when nothing cheaper fits, so refusing it would trade a slow render for no
            // render. Cost is not what this rule reads.
            //
            // Be clear about the blast radius, because it is wider than rung 4: like every other rule
            // in this function, an error here is CONTRACT-level, and a caller that bails on a non-empty
            // `conformance_errors` (the shared selector does) loses rungs 0-4, not just rung 4. That is
            // the right severity for the case that produces it — a declaration this contract cannot
            // interpret — and the wrong severity for a cost verdict, which is a second reason this rule
            // does not attempt one.
            match self.backend.window_materialization() {
                Some(MemoryWindowMaterialization::HostFormatConversion {
                    converts,
                    owner_story,
                }) => {
                    // `owner_story` is the promise; `converts` is the pointer a reviewer follows to the
                    // code. Both are load-bearing, so both are required to be present. Neither can be
                    // checked for TRUTH here — a wrong pointer or a closed story still passes.
                    if owner_story.trim().is_empty() {
                        errors.push(format!(
                            "BoundedTransformerResidency on a HostFormatConversion realization \
                             ({converts}) must name the owner_story that removes it (SC-16090)"
                        ));
                    }
                    if converts.trim().is_empty() {
                        errors.push(
                            "BoundedTransformerResidency on a HostFormatConversion realization must \
                             say what it converts, so a reviewer can find it (SC-16090)"
                                .to_owned(),
                        );
                    }
                }
                Some(MemoryWindowMaterialization::DeviceFormatTransfer) => {}
                // Unstated. The actionable fix is naming the mechanism, not naming a story — see
                // `MemoryBackendRealization::window_materialization`.
                None => errors.push(
                    "BoundedTransformerResidency requires the realization to state what a window \
                     materialization does; an MLX realization does so by declaring \
                     lazy_or_mmap_materialization (SC-16090)"
                        .to_owned(),
                ),
            }
        }

        for capability in &self.strategies {
            if let MemoryStrategySupport::StructurallyNotApplicable { reason } = &capability.support
            {
                if reason.trim().is_empty() {
                    errors.push(format!(
                        "{:?} is StructurallyNotApplicable without a reason",
                        capability.strategy
                    ));
                }
                let has_implementation_hook = match capability.strategy {
                    MemoryStrategy::Resident => false,
                    MemoryStrategy::StagedResidency => {
                        !self.lifecycle.phases.is_empty()
                            || self.lifecycle.synchronized_phase_release
                    }
                    MemoryStrategy::BoundedDecode => self.lifecycle.decode_tiling,
                    MemoryStrategy::BoundedAttention => self.lifecycle.attention_chunking,
                    MemoryStrategy::BoundedTransformerResidency => {
                        self.lifecycle.transformer_window_materialization
                    }
                };
                if has_implementation_hook {
                    errors.push(format!(
                        "{:?} cannot be StructurallyNotApplicable while its implementation hook is declared",
                        capability.strategy
                    ));
                }
            }
            validate_ranges(capability, &mut errors);
            validate_owned_parameter_domain(capability, &mut errors);
        }

        let mut component_ids = std::collections::HashSet::new();
        for component in self.formula.resident_components() {
            if component.id.trim().is_empty() {
                errors.push("resident component id must be non-empty".to_owned());
            } else if !component_ids.insert(component.id.as_str()) {
                errors.push(format!(
                    "resident component id {:?} must be unique",
                    component.id
                ));
            }
            if component.resident_bytes == 0 {
                errors.push(format!(
                    "resident component {:?} must declare non-zero bytes",
                    component.id
                ));
            }
            if let Some(strategy) = component.bounded_by {
                if !implemented(strategy) {
                    errors.push(format!(
                        "resident component {:?} is bounded by {strategy:?}, but that strategy is not implemented",
                        component.id
                    ));
                }
            }
        }
        let auxiliary_bytes = self.auxiliary_resident_bytes();
        match self
            .asset_facts
            .conditioning_bytes
            .checked_add(self.asset_facts.transformer_bytes)
            .and_then(|bytes| bytes.checked_add(self.asset_facts.decoder_bytes))
        {
            Some(base_components) if base_components != self.asset_facts.base_bytes => {
                errors.push(format!(
                    "base_bytes {} must equal base component bytes {} and exclude overlay_bytes",
                    self.asset_facts.base_bytes, base_components
                ));
            }
            None => errors.push("base component byte sum overflow".to_owned()),
            _ => {}
        }
        if self.asset_facts.overlay_bytes > 0
            && self.formula.uses(MemoryFormulaVariable::AssetBytes)
            && self.formula.uses(MemoryFormulaVariable::OverlayBytes)
            && auxiliary_bytes == 0
        {
            errors.push(
                "a non-zero overlay declared with AssetBytes and OverlayBytes must use typed auxiliary resident components"
                    .to_owned(),
            );
        }
        if auxiliary_bytes > 0 {
            if auxiliary_bytes != self.asset_facts.overlay_bytes {
                errors.push(format!(
                    "overlay_bytes {} must equal declared auxiliary component bytes {}",
                    self.asset_facts.overlay_bytes, auxiliary_bytes
                ));
            }
            if !self.formula.uses(MemoryFormulaVariable::OverlayBytes) {
                errors.push(
                    "a provider with auxiliary resident components must include OverlayBytes in its formula"
                        .to_owned(),
                );
            }
        }

        if let Some(calibration) = &self.calibration {
            if calibration.abi != MEMORY_CALIBRATION_ABI {
                errors.push(format!(
                    "calibration ABI {} does not match contract ABI {}",
                    calibration.abi, MEMORY_CALIBRATION_ABI
                ));
            }
            if let Err(reason) = validate_calibration_fingerprint(&calibration.fingerprint) {
                errors.push(format!("calibration fingerprint {reason}"));
            }
            if calibration.load_shape != self.load_shape {
                errors.push(format!(
                    "calibration load shape {:?} does not match contract load shape {:?}",
                    calibration.load_shape, self.load_shape
                ));
            }
        }

        if self.backend.backend_id().is_empty() {
            errors.push("backend realization must have an id".to_owned());
        }
        errors
    }

    /// Validate a worker-owned selection against static provider capability and parameter ranges.
    pub fn validate_selection(&self, selection: &MemorySelection) -> Result<()> {
        let capability = self.capability(selection.strategy).ok_or_else(|| {
            Error::Unsupported(format!(
                "{} does not declare {:?}",
                self.provider_id, selection.strategy
            ))
        })?;
        if !matches!(capability.support, MemoryStrategySupport::Implemented) {
            return Err(Error::Unsupported(format!(
                "{} cannot execute {:?}: {:?}",
                self.provider_id, selection.strategy, capability.support
            )));
        }
        // SC-15805: walk the DECLARED prerequisite graph, never `< selection.strategy`. The numeric
        // order is a cost ordering; reading it as a dependency refused compositions that are
        // perfectly correct (rungs 2 and 3 bound scratch and depend on nothing).
        for prerequisite in self.requires(selection.strategy) {
            match prerequisite {
                MemoryStrategyPrerequisite::Rung { rung, scope } => match scope {
                    MemoryPrerequisiteScope::EngagedInSameRequest => {
                        if self.engages_selection(selection, rung) {
                            continue;
                        }
                        // `StructurallyNotApplicable` satisfies the edge vacuously: it asserts the
                        // architecture has no such component to shed.
                        if matches!(
                            self.support(rung),
                            Some(MemoryStrategySupport::StructurallyNotApplicable { .. })
                        ) {
                            continue;
                        }
                        let (why, caveat) = match self.support(rung) {
                            Some(MemoryStrategySupport::Implemented) => (
                                "the selected composition does not engage it".to_owned(),
                                format!(
                                    "This prerequisite is engagement, not availability — rung \
                                     {rung:?} being implemented, or engaged by some other request, \
                                     does not satisfy it."
                                ),
                            ),
                            support => (
                                match support {
                                    Some(support) => {
                                        format!("the provider declares it {support:?}")
                                    }
                                    None => {
                                        "the provider does not declare it at all".to_owned()
                                    }
                                },
                                format!(
                                    "Declaring rung {rung:?} Implemented — or \
                                     StructurallyNotApplicable, if this architecture has no such \
                                     component to stage — is what makes {:?} selectable. Engagement \
                                     is per request.",
                                    selection.strategy
                                ),
                            ),
                        };
                        return Err(Error::Unsupported(format!(
                            "{} cannot execute {:?}: it requires rung {rung:?} to be ENGAGED IN THE \
                             SAME REQUEST, and {why}. {caveat}",
                            self.provider_id, selection.strategy
                        )));
                    }
                },
                MemoryStrategyPrerequisite::LoadShape(required) => {
                    if self.load_shape == required {
                        continue;
                    }
                    return Err(Error::Unsupported(format!(
                        "{} cannot execute {:?}: it requires load shape {required:?}, but this \
                         generator was loaded with {:?}",
                        self.provider_id, selection.strategy, self.load_shape
                    )));
                }
            }
        }
        validate_selected_parameters(selection, self)
            .map_err(|message| Error::Unsupported(format!("{}: {message}", self.provider_id)))
    }
}

fn validate_pid_decode_routes(
    contract: &MemoryProviderContract,
    routes: &MemoryPidDecodeRoutes,
    errors: &mut Vec<String>,
) {
    let Some(bounded_decode) = contract.capability(MemoryStrategy::BoundedDecode) else {
        errors.push("PiD decode routes require a BoundedDecode capability".to_owned());
        return;
    };
    if !matches!(bounded_decode.support, MemoryStrategySupport::Implemented) {
        errors.push("PiD decode routes require BoundedDecode to be Implemented".to_owned());
    }

    for (label, domain) in [("native", &routes.native), ("PiD", &routes.pid)] {
        if domain.tile_edges.is_empty() {
            errors.push(format!(
                "{label} decode route must declare at least one tile edge"
            ));
        }
        if domain.tile_edges.contains(&0) {
            errors.push(format!("{label} decode route contains a zero tile edge"));
        }
        let mut unique = domain.tile_edges.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != domain.tile_edges.len() {
            errors.push(format!(
                "{label} decode route contains duplicate tile edges"
            ));
        }
        if domain.tile_overlap == 0 {
            errors.push(format!(
                "{label} decode route must declare a non-zero overlap"
            ));
        }
    }

    if routes.native.tile_overlap == routes.pid.tile_overlap
        && routes
            .native
            .tile_edges
            .iter()
            .any(|edge| routes.pid.tile_edges.contains(edge))
    {
        errors.push("native and PiD decode route geometries must be disjoint".to_owned());
    }

    let mut declared_edges = bounded_decode.parameters.decode_tile_edges.clone();
    declared_edges.sort_unstable();
    declared_edges.dedup();
    let mut route_edges = routes
        .native
        .tile_edges
        .iter()
        .chain(&routes.pid.tile_edges)
        .copied()
        .collect::<Vec<_>>();
    route_edges.sort_unstable();
    route_edges.dedup();
    if declared_edges != route_edges {
        errors.push(format!(
            "BoundedDecode tile edges {declared_edges:?} must equal the native/PiD route union \
             {route_edges:?}"
        ));
    }

    let mut declared_overlaps = bounded_decode.parameters.decode_overlaps.clone();
    declared_overlaps.sort_unstable();
    declared_overlaps.dedup();
    let mut route_overlaps = vec![routes.native.tile_overlap, routes.pid.tile_overlap];
    route_overlaps.sort_unstable();
    route_overlaps.dedup();
    if declared_overlaps != route_overlaps {
        errors.push(format!(
            "BoundedDecode overlaps {declared_overlaps:?} must equal the native/PiD route union \
             {route_overlaps:?}"
        ));
    }
}

fn validate_decode_geometry_policies(contract: &MemoryProviderContract, errors: &mut Vec<String>) {
    let Some(bounded_decode) = contract.capability(MemoryStrategy::BoundedDecode) else {
        return;
    };
    let policies = &bounded_decode.parameters.decode_geometry_policies;
    if policies.is_empty() {
        return;
    }

    let implemented = matches!(bounded_decode.support, MemoryStrategySupport::Implemented);
    let mut coordinates = std::collections::HashSet::new();
    let mut admitted_edges = Vec::new();
    let mut admitted_overlaps = Vec::new();
    let mut admitted_count = 0usize;
    let mut family = None;
    let mut resolved_route = None;
    let mut artifact = None;
    for policy in policies {
        if policy.quality_abi != MEMORY_DECODE_QUALITY_ABI {
            errors.push(format!(
                "decode quality ABI {} does not match {}",
                policy.quality_abi, MEMORY_DECODE_QUALITY_ABI
            ));
        }
        if policy.family.trim().is_empty() {
            errors.push("decode quality family must be non-empty".to_owned());
        }
        if let Some(expected) = family {
            if policy.family != expected {
                errors.push(format!(
                    "one loaded provider contract cannot mix decode quality families {:?} and {:?}",
                    expected, policy.family
                ));
            }
        } else {
            family = Some(policy.family.as_str());
        }
        if policy.resolved_route.trim().is_empty() {
            errors.push("decode quality resolved route must be non-empty".to_owned());
        }
        if let Some(expected) = resolved_route {
            if policy.resolved_route != expected {
                errors.push(format!(
                    "one loaded provider contract cannot mix decode quality routes {:?} and {:?}",
                    expected, policy.resolved_route
                ));
            }
        } else {
            resolved_route = Some(policy.resolved_route.as_str());
        }
        if policy.backend != contract.backend.backend_kind() {
            errors.push(format!(
                "decode quality backend {:?} does not match provider backend {:?}",
                policy.backend,
                contract.backend.backend_kind()
            ));
        }
        if policy.load_shape != contract.load_shape {
            errors.push(format!(
                "decode quality load shape {:?} does not match provider load shape {:?}",
                policy.load_shape, contract.load_shape
            ));
        }
        if policy.artifact.repository.trim().is_empty()
            || policy.artifact.variant.trim().is_empty()
            || policy.artifact.fingerprint.trim().is_empty()
            || !is_lower_hex_sha1(&policy.artifact.revision)
        {
            errors.push(
                "decode quality artifact requires repository, exact 40-hex revision, variant, and fingerprint"
                    .to_owned(),
            );
        }
        if let Some(expected) = artifact {
            if &policy.artifact != expected {
                errors.push(
                    "one loaded provider contract cannot mix decode quality artifact identities"
                        .to_owned(),
                );
            }
        } else {
            artifact = Some(&policy.artifact);
        }
        if !is_lower_hex_sha256(&policy.implementation_fingerprint)
            || policy.implementation_fingerprint != MEMORY_DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT
        {
            errors.push(
                "decode quality implementation fingerprint does not match the running quality source closure"
                    .to_owned(),
            );
        }
        if policy.mode.as_key().trim().is_empty() {
            errors.push("decode quality mode must be non-empty".to_owned());
        }
        if policy
            .overlay
            .as_deref()
            .is_some_and(|overlay| overlay.trim().is_empty() || overlay.contains('='))
        {
            errors.push("decode quality overlay must be a non-empty identity axis".to_owned());
        }
        if policy.geometry.width == 0
            || policy.geometry.height == 0
            || policy.geometry.batch == 0
            || policy.geometry.frames == 0
        {
            errors.push("decode quality geometry dimensions must be non-zero".to_owned());
        }
        if policy.use_pid && contract.pid_decode_routes.is_none() {
            errors.push(
                "decode quality cannot admit a PiD route when the provider declares no PiD decode geometry"
                    .to_owned(),
            );
        }
        if policy.tile_edge == 0 || policy.overlap == 0 || policy.overlap >= policy.tile_edge {
            errors.push(format!(
                "decode quality tile edge {} must exceed non-zero overlap {}",
                policy.tile_edge, policy.overlap
            ));
        }
        if policy.metric.trim().is_empty() {
            errors.push("decode quality metric must be non-empty".to_owned());
        }
        if !is_lower_hex_sha256(&policy.production_evidence_sha256) {
            errors.push(
                "decode production evidence SHA-256 must be 64 lowercase hexadecimal characters"
                    .to_owned(),
            );
        } else if policy.production_evidence_sha256 != policy.computed_evidence_sha256() {
            errors.push(format!(
                "decode production evidence SHA-256 does not bind the canonical {}x{} quality record",
                policy.geometry.width, policy.geometry.height
            ));
        }
        if policy.fixtures.len() < 2 {
            errors.push(format!(
                "decode quality coordinate {}x{} requires multiple fixed-seed production latents",
                policy.geometry.width, policy.geometry.height
            ));
        }
        let mut seeds = std::collections::HashSet::new();
        for fixture in &policy.fixtures {
            if !seeds.insert(fixture.seed) {
                errors.push(format!(
                    "decode quality coordinate {}x{} contains duplicate seed {}",
                    policy.geometry.width, policy.geometry.height, fixture.seed
                ));
            }
            if fixture.production_latent_provenance.trim().is_empty() {
                errors.push(format!(
                    "decode quality seed {} has no production-latent provenance",
                    fixture.seed
                ));
            }
            for (label, sha256) in [
                (
                    "production latent",
                    fixture.production_latent_sha256.as_str(),
                ),
                ("dense output", fixture.dense_output_sha256.as_str()),
                ("tiled output", fixture.tiled_output_sha256.as_str()),
            ] {
                if !is_lower_hex_sha256(sha256) {
                    errors.push(format!(
                        "decode quality seed {} {label} SHA-256 must be 64 lowercase hexadecimal characters",
                        fixture.seed
                    ));
                }
            }
        }

        let coordinate = format!(
            "{}|{}|{}|{:?}|{:?}|{}|{:?}|{}x{}x{}x{}r{}|pid={}|tile={}+{}",
            policy.family,
            policy.resolved_route,
            policy.backend.as_key(),
            policy.tier,
            policy.load_shape,
            policy.mode.as_key(),
            policy.overlay,
            policy.geometry.width,
            policy.geometry.height,
            policy.geometry.batch,
            policy.geometry.frames,
            policy.geometry.reference_count,
            policy.use_pid,
            policy.tile_edge,
            policy.overlap,
        );
        if !coordinates.insert(coordinate) {
            errors.push(format!(
                "duplicate decode quality coordinate for {}x{}",
                policy.geometry.width, policy.geometry.height
            ));
        }

        match &policy.disposition {
            MemoryDecodeQualityDisposition::Admitted => {
                admitted_count += 1;
                admitted_edges.push(policy.tile_edge);
                admitted_overlaps.push(policy.overlap);
                if !implemented {
                    errors.push(
                        "an admitted decode quality row requires BoundedDecode Implemented"
                            .to_owned(),
                    );
                }
                if policy
                    .fixtures
                    .iter()
                    .any(|fixture| fixture.observed_error > policy.maximum_error)
                {
                    errors.push(format!(
                        "admitted decode quality coordinate {}x{} exceeds its {} maximum",
                        policy.geometry.width, policy.geometry.height, policy.maximum_error
                    ));
                }
            }
            MemoryDecodeQualityDisposition::Refused { reason } => {
                if reason.trim().is_empty() {
                    errors.push("refused decode quality row requires a reason".to_owned());
                }
                if policy
                    .fixtures
                    .iter()
                    .all(|fixture| fixture.observed_error <= policy.maximum_error)
                {
                    errors.push(format!(
                        "refused decode quality coordinate {}x{} has no failing fixture",
                        policy.geometry.width, policy.geometry.height
                    ));
                }
            }
        }
    }

    if implemented && admitted_count == 0 {
        errors.push(
            "geometry-aware BoundedDecode requires at least one admitted quality row".to_owned(),
        );
    }
    admitted_edges.sort_unstable();
    admitted_edges.dedup();
    admitted_overlaps.sort_unstable();
    admitted_overlaps.dedup();
    let mut declared_edges = bounded_decode.parameters.decode_tile_edges.clone();
    declared_edges.sort_unstable();
    declared_edges.dedup();
    let mut declared_overlaps = bounded_decode.parameters.decode_overlaps.clone();
    declared_overlaps.sort_unstable();
    declared_overlaps.dedup();
    if implemented && admitted_edges != declared_edges {
        errors.push(format!(
            "BoundedDecode tile edges {declared_edges:?} must equal admitted geometry-policy union {admitted_edges:?}"
        ));
    }
    if implemented && admitted_overlaps != declared_overlaps {
        errors.push(format!(
            "BoundedDecode overlaps {declared_overlaps:?} must equal admitted geometry-policy union {admitted_overlaps:?}"
        ));
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_lower_hex_sha1(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// The shared safety pipeline for an adopted provider.
///
/// `loaded_tier` binds a weights-free registration or loaded generator to the numeric tier it
/// actually loaded. `route_gate` is the only provider-owned part of the standard pipeline and runs
/// after the shared handshake, tier, and selection checks but before the shared budget check.
pub fn standard_memory_strategy_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_tier: Option<MemoryNumericTier>,
    route_gate: Option<&dyn Fn() -> Result<()>>,
) -> MemorySafetyDecision {
    let reject = |reason| MemorySafetyDecision::Reject { reason };
    if context.budget.reclaimable_bytes > context.budget.committed_bytes {
        return reject(format!(
            "{}: reclaimable bytes {} exceed committed bytes {}; reclaim credit must be a subset \
             of the charged snapshot",
            contract.provider_id, context.budget.reclaimable_bytes, context.budget.committed_bytes
        ));
    }
    if context.has_reference != (context.geometry.reference_count > 0) {
        return reject(format!(
            "{}: has_reference={} is inconsistent with reference_count={}",
            contract.provider_id, context.has_reference, context.geometry.reference_count
        ));
    }
    if context
        .overlay
        .as_deref()
        .is_some_and(|overlay| overlay.contains('='))
    {
        return reject(format!(
            "{}: overlay is an identity axis and must not encode structured key/value data",
            contract.provider_id
        ));
    }
    if context.selection.strategy.is_optimized()
        && context.optimization_authority == MemoryOptimizationAuthority::Resident
    {
        return reject(format!(
            "{}: optimized memory strategy {:?} has no explicit measured or estimated authority",
            contract.provider_id, context.selection.strategy
        ));
    }
    if let Some(calibration) = contract.calibration.as_ref() {
        if context.calibration_abi != calibration.abi
            || context.calibration_fingerprint != calibration.fingerprint
            || context.load_shape != calibration.load_shape
        {
            return reject(format!(
                "{}: calibration handshake mismatch",
                contract.provider_id
            ));
        }
    } else if context.selection.strategy.is_optimized()
        && context.optimization_authority != MemoryOptimizationAuthority::Estimated
    {
        return reject(format!(
            "{}: optimized memory strategy {:?} without a calibration identity requires explicit estimate authority",
            contract.provider_id, context.selection.strategy
        ));
    }
    if let Some(loaded_tier) = loaded_tier {
        if context.selection.tier != loaded_tier {
            return reject(format!(
                "{}: selected tier {:?} does not match loaded tier {:?}",
                contract.provider_id, context.selection.tier, loaded_tier
            ));
        }
    }
    if let Err(error) = contract.validate_selection(&context.selection) {
        return reject(error.to_string());
    }
    if contract.engages_selection(&context.selection, MemoryStrategy::BoundedDecode) {
        if let Err(error) = contract.decode_geometry_policy(
            MemoryDecodePolicyQuery {
                tier: context.selection.tier,
                load_shape: context.load_shape,
                mode_key: context.mode.as_key(),
                overlay: context.overlay.as_deref(),
                geometry: context.geometry,
                use_pid: context.use_pid,
            },
            Some(context.selection.parameters),
        ) {
            return reject(error.to_string());
        }
    }
    if let Some(route_gate) = route_gate {
        if let Err(error) = route_gate() {
            return reject(error.to_string());
        }
    }
    if !context.budget.fits(context.predicted_peak_bytes) {
        return MemorySafetyDecision::Reject {
            reason: format!(
                "{}: incremental live demand {} exceeds effective budget {}",
                contract.provider_id,
                context.predicted_peak_bytes,
                context.budget.effective_bytes()
            ),
        };
    }
    MemorySafetyDecision::Accept
}

/// The shared safety behavior for an adopted provider with no additional admission rules.
///
/// Provider registrations can point at this function without loading a [`Generator`](crate::Generator).
/// Providers with extra safety rules delegate to [`standard_memory_strategy_safety_check`].
pub fn default_memory_strategy_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    standard_memory_strategy_safety_check(contract, context, None, None)
}

/// Registry adapter for [`default_memory_strategy_safety_check`].
///
/// The load specification is part of the weights-free registration callback so providers whose
/// admission rules depend on the requested tier can reproduce their loaded generator's check.
pub fn default_registered_memory_strategy_safety_check(
    spec: &crate::LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        None,
    )
}

fn validate_ranges(capability: &MemoryStrategyCapability, errors: &mut Vec<String>) {
    let ranges = &capability.parameters;
    for (name, values) in [
        ("decode_tile_edges", &ranges.decode_tile_edges),
        ("decode_overlaps", &ranges.decode_overlaps),
        ("attention_chunk_sizes", &ranges.attention_chunk_sizes),
        ("transformer_window_sizes", &ranges.transformer_window_sizes),
    ] {
        if values.contains(&0) {
            errors.push(format!("{:?} {name} contains zero", capability.strategy));
        }
        let mut unique = values.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != values.len() {
            errors.push(format!(
                "{:?} {name} contains duplicate candidates",
                capability.strategy
            ));
        }
    }
}

fn validate_owned_parameter_domain(
    capability: &MemoryStrategyCapability,
    errors: &mut Vec<String>,
) {
    if capability.strategy != MemoryStrategy::BoundedDecode
        && !capability.parameters.decode_geometry_policies.is_empty()
    {
        errors.push(format!(
            "{:?} must not own decode geometry policies",
            capability.strategy
        ));
    }
    if !matches!(capability.support, MemoryStrategySupport::Implemented) {
        return;
    }
    let ranges = &capability.parameters;
    let (decode, attention, transformer) = match capability.strategy {
        MemoryStrategy::Resident | MemoryStrategy::StagedResidency => (false, false, false),
        MemoryStrategy::BoundedDecode => (true, false, false),
        MemoryStrategy::BoundedAttention => (false, true, false),
        MemoryStrategy::BoundedTransformerResidency => (false, false, true),
    };
    if decode && (ranges.decode_tile_edges.is_empty() || ranges.decode_overlaps.is_empty()) {
        errors.push(
            "BoundedDecode requires non-empty decode tile-edge and overlap candidates".to_owned(),
        );
    }
    if attention && ranges.attention_chunk_sizes.is_empty() {
        errors
            .push("BoundedAttention requires non-empty attention chunk-size candidates".to_owned());
    }
    if transformer && ranges.transformer_window_sizes.is_empty() {
        errors.push(
            "BoundedTransformerResidency requires non-empty transformer window candidates"
                .to_owned(),
        );
    }
    if !decode && (!ranges.decode_tile_edges.is_empty() || !ranges.decode_overlaps.is_empty()) {
        errors.push(format!(
            "{:?} must not own decode parameter candidates",
            capability.strategy
        ));
    }
    if !attention && !ranges.attention_chunk_sizes.is_empty() {
        errors.push(format!(
            "{:?} must not own attention parameter candidates",
            capability.strategy
        ));
    }
    if !transformer && !ranges.transformer_window_sizes.is_empty() {
        errors.push(format!(
            "{:?} must not own transformer-window candidates",
            capability.strategy
        ));
    }
    // The component scope is owned by the same rung as the window it scopes (SC-15794). It is
    // deliberately allowed to be empty on an implementing provider — that reads as the pre-SC-15794
    // DiT-only behaviour rather than as an incomplete declaration, so existing providers stay valid.
    if !transformer && !ranges.transformer_window_components.is_empty() {
        errors.push(format!(
            "{:?} must not own transformer-window component candidates",
            capability.strategy
        ));
    }
    let components = &ranges.transformer_window_components;
    let mut unique = components.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != components.len() {
        errors.push(format!(
            "{:?} transformer_window_components contains duplicate candidates",
            capability.strategy
        ));
    }
}

fn validate_selected_parameters(
    selection: &MemorySelection,
    contract: &MemoryProviderContract,
) -> std::result::Result<(), String> {
    let selected = selection.parameters;
    // Which rungs this selection ENGAGES — the defeasible cost-order default, read through the one
    // named policy seam rather than re-deriving `>=` here (SC-15805). A rung the provider does not
    // implement is not engaged, so its parameters stop being required instead of the selection being
    // refused: that is how a verified cheaper composition is published without fighting this
    // validator.
    let requires_decode = contract.engages_selection(selection, MemoryStrategy::BoundedDecode);
    let requires_attention =
        contract.engages_selection(selection, MemoryStrategy::BoundedAttention);
    let requires_transformer =
        contract.engages_selection(selection, MemoryStrategy::BoundedTransformerResidency);

    validate_required_parameter(
        "decode_tile_edge",
        selected.decode_tile_edge,
        requires_decode,
        contract
            .capability(MemoryStrategy::BoundedDecode)
            .map(|capability| capability.parameters.decode_tile_edges.as_slice())
            .unwrap_or_default(),
    )?;
    validate_required_parameter(
        "decode_overlap",
        selected.decode_overlap,
        requires_decode,
        contract
            .capability(MemoryStrategy::BoundedDecode)
            .map(|capability| capability.parameters.decode_overlaps.as_slice())
            .unwrap_or_default(),
    )?;
    validate_required_parameter(
        "attention_chunk_size",
        selected.attention_chunk_size,
        requires_attention,
        contract
            .capability(MemoryStrategy::BoundedAttention)
            .map(|capability| capability.parameters.attention_chunk_sizes.as_slice())
            .unwrap_or_default(),
    )?;
    validate_required_parameter(
        "transformer_window_size",
        selected.transformer_window_size,
        requires_transformer,
        contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .map(|capability| capability.parameters.transformer_window_sizes.as_slice())
            .unwrap_or_default(),
    )?;
    // The component scope (SC-15794) is validated separately from the numeric parameters: unlike a
    // tile edge or a window size it has a meaningful DEFAULT (DiT-only), so `None` is legal at every
    // rung and only an explicitly-declared scope is checked. That keeps a pre-SC-15794 selection —
    // which cannot carry a component — valid rather than retroactively incomplete.
    if let Some(component) = selected.transformer_window_component {
        if !requires_transformer {
            return Err(format!(
                "transformer_window_component={component:?} is irrelevant: the selection does not \
                 engage its owning strategy rung"
            ));
        }
        let allowed = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .map(|capability| {
                capability
                    .parameters
                    .transformer_window_components
                    .as_slice()
            })
            .unwrap_or_default();
        // An empty declaration means the provider implements the DiT-only default and nothing else,
        // so that is the one scope a request may still name explicitly.
        let permitted = if allowed.is_empty() {
            component == TransformerComponent::Dit
        } else {
            allowed.contains(&component)
        };
        if !permitted {
            return Err(format!(
                "transformer_window_component={component:?} is outside the declared production \
                 candidates {allowed:?}"
            ));
        }
    }
    Ok(())
}

fn validate_required_parameter(
    name: &str,
    value: Option<u32>,
    required: bool,
    allowed: &[u32],
) -> std::result::Result<(), String> {
    match (required, value) {
        (true, None) => Err(format!(
            "{name} is required by a strategy rung this selection engages"
        )),
        (false, Some(value)) => Err(format!(
            "{name}={value} is irrelevant: the selection does not engage its owning strategy rung"
        )),
        (true, Some(value)) if !allowed.contains(&value) => Err(format!(
            "{name}={value} is outside the declared production candidates {allowed:?}"
        )),
        _ => Ok(()),
    }
}

/// Numeric tier is one immutable axis of a selection. Strategies cannot change it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryNumericTier {
    pub precision: Precision,
    pub quant: Option<Quant>,
    /// The provider-declared component floors active for this numeric tier. Part of evidence/cache
    /// identity: a uniform q4 run and a q4 run with q8 components are different numeric regimes.
    pub component_precision_floors: &'static [crate::ComponentPrecisionFloor],
}

/// Shared-worker selection presented to the provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySelection {
    pub strategy: MemoryStrategy,
    pub parameters: MemoryStrategyParameters,
    pub tier: MemoryNumericTier,
}

/// Request geometry used by evidence and cache identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryGeometry {
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub frames: u32,
    /// Number of reference images consumed by this provider forward pass. This is request
    /// geometry, not overlay identity; zero means the request has no reference input.
    pub reference_count: u32,
}

/// One deterministic production latent used to judge a tiled-decode policy.
///
/// `production_latent_provenance` names the immutable model revision, prompt fixture, denoise
/// schedule, and harness source that reproduce the latent. The enclosing evidence SHA binds the
/// complete ordered fixture set and its observed errors; this record intentionally carries no peak,
/// duration, or allocator field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeQualityFixture {
    pub seed: u64,
    pub production_latent_provenance: String,
    pub production_latent_sha256: String,
    pub dense_output_sha256: String,
    pub tiled_output_sha256: String,
    pub observed_error: u32,
}

/// Production-latent quality verdict for one exact decode coordinate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryDecodeQualityDisposition {
    Admitted,
    Refused { reason: String },
}

/// Immutable identity of the exact model artifact whose production latents were admitted.
///
/// `fingerprint` is the caller's already-validated resolved-tree identity. For an immutable hub
/// snapshot it is normally `repository@revision:variant`; app-managed artifacts may instead carry
/// a strong `sha256:` content identity. The runtime never derives this value from a request or a
/// manifest: the loader supplies an independently resolved identity and [`LoadSpec`](crate::LoadSpec)
/// compares it before a row can enter the provider contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeArtifactIdentity {
    pub repository: String,
    pub revision: String,
    pub variant: String,
    pub fingerprint: String,
}

/// Independently resolved identity of the artifact and running quality implementation used by one
/// concrete load. Callers attach this beside packaged rows; [`LoadSpec`](crate::LoadSpec) refuses to
/// pass rows into a provider unless both halves match exactly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeQualityRuntimeIdentity {
    pub artifact: MemoryDecodeArtifactIdentity,
    pub implementation_fingerprint: String,
}

/// Geometry-aware tiled-decode policy and its immutable production-quality receipt.
///
/// Every runtime identity axis that can change the latent or decoder route is explicit. A provider
/// adopting this table may execute bounded decode only when the loaded contract and request match
/// one `Admitted` row exactly. Failed rows remain first-class records so a known-bad coordinate is
/// distinguishable from an unmeasured one; both fail closed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryDecodeGeometryPolicy {
    pub quality_abi: u32,
    pub family: String,
    pub resolved_route: String,
    pub backend: MemoryBackend,
    pub tier: MemoryNumericTier,
    pub load_shape: LoadShape,
    pub artifact: MemoryDecodeArtifactIdentity,
    /// Exact quality-relevant inference/harness source closure.
    pub implementation_fingerprint: String,
    pub mode: MemoryMode,
    pub overlay: Option<String>,
    pub geometry: MemoryGeometry,
    pub use_pid: bool,
    pub tile_edge: u32,
    pub overlap: u32,
    pub metric: String,
    pub maximum_error: u32,
    pub fixtures: Vec<MemoryDecodeQualityFixture>,
    /// SHA-256 of the canonical quality record, excluding this field.
    pub production_evidence_sha256: String,
    pub disposition: MemoryDecodeQualityDisposition,
}

/// Exact request coordinate supplied before a selector chooses tiled-decode parameters.
#[derive(Clone, Copy, Debug)]
pub struct MemoryDecodePolicyQuery<'a> {
    pub tier: MemoryNumericTier,
    pub load_shape: LoadShape,
    pub mode_key: &'a str,
    pub overlay: Option<&'a str>,
    pub geometry: MemoryGeometry,
    pub use_pid: bool,
}

impl MemoryDecodeGeometryPolicy {
    /// SHA-256 of the canonical semantic quality record. No timing, memory, or performance field is
    /// present in the input by construction.
    pub fn computed_evidence_sha256(&self) -> String {
        let component_precision_floors = self
            .tier
            .component_precision_floors
            .iter()
            .map(|floor| {
                serde_json::json!({
                    "component": floor.component.as_str(),
                    "selected_tier": quant_key(floor.selected_tier),
                    "resident_tier": quant_key(floor.resident_tier),
                })
            })
            .collect::<Vec<_>>();
        let fixtures = self
            .fixtures
            .iter()
            .map(|fixture| {
                serde_json::json!({
                    "seed": fixture.seed,
                    "production_latent_provenance": fixture.production_latent_provenance,
                    "production_latent_sha256": fixture.production_latent_sha256,
                    "dense_output_sha256": fixture.dense_output_sha256,
                    "tiled_output_sha256": fixture.tiled_output_sha256,
                    "observed_error": fixture.observed_error,
                })
            })
            .collect::<Vec<_>>();
        let disposition = match &self.disposition {
            MemoryDecodeQualityDisposition::Admitted => {
                serde_json::json!({ "kind": "admitted" })
            }
            MemoryDecodeQualityDisposition::Refused { reason } => {
                serde_json::json!({ "kind": "refused", "reason": reason })
            }
        };
        let canonical = serde_json::json!({
            "quality_abi": self.quality_abi,
            "family": self.family,
            "resolved_route": self.resolved_route,
            "backend": self.backend.as_key(),
            "tier": {
                "precision": precision_key(self.tier.precision),
                "quant": self.tier.quant.map(quant_key),
                "component_precision_floors": component_precision_floors,
            },
            "load_shape": load_shape_key(self.load_shape),
            "artifact": {
                "repository": self.artifact.repository,
                "revision": self.artifact.revision,
                "variant": self.artifact.variant,
                "fingerprint": self.artifact.fingerprint,
            },
            "implementation_fingerprint": self.implementation_fingerprint,
            "mode": self.mode.as_key(),
            "overlay": self.overlay,
            "geometry": {
                "width": self.geometry.width,
                "height": self.geometry.height,
                "batch": self.geometry.batch,
                "frames": self.geometry.frames,
                "reference_count": self.geometry.reference_count,
            },
            "use_pid": self.use_pid,
            "tile_edge": self.tile_edge,
            "overlap": self.overlap,
            "metric": self.metric,
            "maximum_error": self.maximum_error,
            "fixtures": fixtures,
            "disposition": disposition,
        });
        format!("{:x}", Sha256::digest(canonical.to_string().as_bytes()))
    }

    /// Fill the evidence identity from the canonical quality fields.
    pub fn seal(mut self) -> Self {
        self.production_evidence_sha256 = self.computed_evidence_sha256();
        self
    }

    fn matches_request_axes(
        &self,
        contract: &MemoryProviderContract,
        query: MemoryDecodePolicyQuery<'_>,
    ) -> bool {
        self.backend == contract.backend.backend_kind()
            && self.tier == query.tier
            && self.load_shape == query.load_shape
            && self.mode.as_key() == query.mode_key
            && self.overlay.as_deref() == query.overlay
            && self.geometry == query.geometry
            && self.use_pid == query.use_pid
    }

    fn matches_coordinate(
        &self,
        contract: &MemoryProviderContract,
        query: MemoryDecodePolicyQuery<'_>,
        tile_edge: u32,
        overlap: u32,
    ) -> bool {
        self.matches_request_axes(contract, query)
            && self.tile_edge == tile_edge
            && self.overlap == overlap
    }
}

/// Canonical live-budget accounting owned by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    /// The platform-enforced capacity available to this process or worker, in bytes. This and every
    /// other budget field are one caller snapshot taken immediately before request admission.
    pub total_bytes: u64,
    /// Bytes already charged against [`total_bytes`](Self::total_bytes) at that snapshot, including
    /// unrelated live allocations and any request-owned resident assets loaded before admission.
    /// A caller must not subtract these bytes again from `total_bytes` before constructing the
    /// budget.
    pub committed_bytes: u64,
    /// Memory a backend is holding but could return under pressure — chiefly an allocator's
    /// freed-buffer reuse cache.
    ///
    /// **Creditable only if something will actually reclaim it before the platform's killer fires.**
    /// [`effective_bytes`](Self::effective_bytes) adds this back into the budget, which assumes a
    /// reclaimer exists. That assumption holds where the OS applies backpressure and fails where it
    /// terminates the process instead.
    ///
    /// iOS is the failing case, measured: jetsam reads `phys_footprint` — which counts an allocator's
    /// reuse cache in full — and does not ask anyone to release anything first. An MLX process there
    /// sat at `active + cache` pinned to the per-process cap across an entire decode, with ~3.9 GB
    /// cached and reclaimable, and was killed with 4 MiB of headroom. Every byte counted here was
    /// genuinely reclaimable and none of it was reclaimed, because nothing asked.
    ///
    /// What makes it creditable is bounding the allocator's own cache limit so the *allocator*
    /// reclaims continuously rather than waiting to be asked (`runtime_ios`'s
    /// `bound_mlx_to_platform_limits`). A caller that cannot name such a mechanism should pass `0`
    /// here rather than credit memory that will still be resident at the moment of the kill.
    /// Reclaimable bytes are a subset of `committed_bytes`, never a separate pool: the shared safety
    /// gate rejects `reclaimable_bytes > committed_bytes` rather than inventing capacity that was not
    /// first charged to the snapshot.
    pub reclaimable_bytes: u64,
    /// Capacity deliberately unavailable to the request, in the same process-residency currency as
    /// `total_bytes` and `committed_bytes`.
    pub reserved_headroom_bytes: u64,
}

impl MemoryBudget {
    /// Effective request budget. Reserved headroom is removed from the currently free plus
    /// reclaimable memory, while the total-minus-headroom ceiling prevents reclaimable accounting
    /// from exceeding physical capacity. Signed intermediate arithmetic preserves an existing
    /// overcommit deficit instead of saturating it away before reclamation is credited.
    pub fn effective_bytes(self) -> u64 {
        let ceiling = self
            .total_bytes
            .saturating_sub(self.reserved_headroom_bytes);
        let available = i128::from(self.total_bytes) - i128::from(self.committed_bytes)
            + i128::from(self.reclaimable_bytes)
            - i128::from(self.reserved_headroom_bytes);
        available.clamp(0, i128::from(ceiling)) as u64
    }

    /// Whether the request's **incremental** live demand fits; exact boundaries are accepted.
    ///
    /// `incremental_live_demand_bytes` is the additional live allocation above this budget's
    /// committed snapshot, not an absolute request high-water and not process footprint. Evidence
    /// records an absolute request live peak; before calling `fits`, the caller subtracts only the
    /// request-owned resident bytes already included in `committed_bytes`. Unrelated committed bytes
    /// are already charged by [`effective_bytes`](Self::effective_bytes) and must not be subtracted
    /// from the request peak.
    pub fn fits(self, incremental_live_demand_bytes: u64) -> bool {
        incremental_live_demand_bytes <= self.effective_bytes()
    }

    /// Total platform capacity required by this snapshot and an incremental request demand.
    ///
    /// This is the user-facing inverse of [`fits`](Self::fits): already committed bytes remain
    /// charged, valid reclaim credit is removed, then reserved headroom and new live demand are
    /// added. Saturation keeps an invalidly large input conservative.
    pub fn required_total_bytes(self, incremental_live_demand_bytes: u64) -> u64 {
        self.committed_bytes
            .saturating_sub(self.reclaimable_bytes.min(self.committed_bytes))
            .saturating_add(self.reserved_headroom_bytes)
            .saturating_add(incremental_live_demand_bytes)
    }
}

/// Cold/warm state is request-scoped and never implies reuse of a prior selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCacheState {
    Cold,
    Warm,
}

/// Provenance class under which the caller admitted the selected peak.
///
/// This is deliberately distinct from calibration identity. A conservative estimate is allowed to
/// drive an implemented optimization when a route has no model-memory calibration, but it must be
/// named explicitly; an absent calibration never turns an arbitrary hand-built optimized context
/// into an authorized estimate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryOptimizationAuthority {
    /// Resident baseline; no optimized claim is made.
    #[default]
    Resident,
    /// Exact/stale measured evidence that remains bound to the provider calibration handshake.
    Calibrated,
    /// Caller-synthesized conservative estimate admitted behind the estimate margin.
    Estimated,
}

/// Advertised request surface. Optimized evidence never transfers between these modes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryMode {
    TextToImage,
    ImageToImage,
    Edit,
    Other(String),
}

impl MemoryMode {
    /// Stable spelling at the persisted evidence/protocol boundary.
    pub fn as_key(&self) -> &str {
        match self {
            Self::TextToImage => "text_to_image",
            Self::ImageToImage => "image_to_image",
            Self::Edit => "edit",
            Self::Other(mode) => mode,
        }
    }
}

/// Context for provider safety and lifecycle hooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRunContext {
    pub selection: MemorySelection,
    pub optimization_authority: MemoryOptimizationAuthority,
    /// Provider calibration identity that the caller used for admission. Resident requests carry
    /// this handshake too; optimized-only evidence validation is not sufficient.
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    /// Typed materialization shape covered by the calibration handshake.
    pub load_shape: LoadShape,
    pub mode: MemoryMode,
    /// Compatibility summary for older callers. It must equal `geometry.reference_count > 0`;
    /// provider safety rejects inconsistent contexts before any provider-owned route gate runs.
    pub has_reference: bool,
    pub use_pid: bool,
    pub has_phases: bool,
    pub geometry: MemoryGeometry,
    /// Exact overlay identity used for evidence keying. Structured request data belongs in typed
    /// context or geometry fields; key/value encodings such as `references=2` are rejected.
    pub overlay: Option<String>,
    pub budget: MemoryBudget,
    /// Incremental live-allocation demand above `budget.committed_bytes`. The selected evidence owns
    /// an absolute request peak; the caller removes exactly the request-resident portion already
    /// present in the budget snapshot before building this context.
    pub predicted_peak_bytes: u64,
    /// Caller-facing telemetry for cold/warm admission and cache-poisoning regression checks.
    pub cache_state: MemoryCacheState,
    /// Caller-facing identity of the exact evidence record selected for this request.
    pub evidence_revision: String,
}

impl MemoryRunContext {
    pub fn predicted_peak_breakdown(
        &self,
        _contract: &MemoryProviderContract,
    ) -> MemoryPeakBreakdown {
        // Context demand is incremental after the caller credits request-owned resident bytes.
        // Provider formula components describe the evidence-owned absolute peak and cannot be
        // subtracted from this scalar a second time.
        MemoryPeakBreakdown::from_unattributed(self.predicted_peak_bytes)
    }
}

/// Build a deterministic, finite, weights-free context from provider-owned route facts.
pub struct MemoryBehaviorRoute {
    pub mode: MemoryMode,
    /// Exact number of reference images in the provider-owned weights-free fixture.
    pub reference_count: u32,
    pub use_pid: bool,
    pub has_phases: bool,
    pub overlay: Option<String>,
}

pub fn standard_memory_behavior_context(
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
    tier: MemoryNumericTier,
    route: MemoryBehaviorRoute,
) -> crate::Result<MemoryRunContext> {
    let calibration = contract.calibration.as_ref().ok_or_else(|| {
        crate::Error::Unsupported(format!(
            "{} has no calibration identity for optimized behavior",
            contract.provider_id
        ))
    })?;
    Ok(MemoryRunContext {
        selection: contract.representative_selection(strategy, tier, route.use_pid)?,
        optimization_authority: MemoryOptimizationAuthority::Calibrated,
        calibration_abi: calibration.abi,
        calibration_fingerprint: calibration.fingerprint.clone(),
        load_shape: calibration.load_shape,
        mode: route.mode,
        has_reference: route.reference_count > 0,
        use_pid: route.use_pid,
        has_phases: route.has_phases,
        geometry: MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: route.reference_count,
        },
        overlay: route.overlay,
        budget: MemoryBudget {
            total_bytes: 8 * 1024 * 1024 * 1024,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        predicted_peak_bytes: 1024 * 1024 * 1024,
        cache_state: MemoryCacheState::Cold,
        evidence_revision: "weights-free-provider-behavior".to_owned(),
    })
}

/// Defense-in-depth result. It can accept or reject; it cannot replace the worker's selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemorySafetyDecision {
    Accept,
    Reject { reason: String },
}

/// Terminal cleanup reason supplied to a request scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryRunOutcome {
    Complete,
    Canceled,
    Error { message: String },
}

/// Tensor-neutral lifecycle scope implemented by adopting providers.
pub trait MemoryRequestScope {
    /// Translate the selected strategy into the provider's existing request controls. This is the
    /// executable bridge for providers that predate the shared contract.
    fn configure_request(&mut self, request: &mut GenerationRequest) -> Result<()>;
    fn enter_phase(&mut self, phase: MemoryPhase) -> Result<()>;
    fn leave_phase(&mut self, phase: MemoryPhase) -> Result<()>;
    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> Result<()>;
    fn configure_attention(&mut self, chunk_size: u32) -> Result<()>;
    fn materialize_transformer_window(&mut self, first_block: u32, block_count: u32) -> Result<()>;
    /// Must be called exactly once on success, cancellation, or error. Providers synchronize and
    /// release active phases/windows according to [`MemoryRuntimeSemantics`].
    fn finish(&mut self, outcome: MemoryRunOutcome) -> Result<()>;
}

/// Five catalog conformance states from epic SC-15448.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryConformanceState {
    Verified,
    ImplementedUnverified,
    StructurallyNotApplicable,
    Missing,
    RouteUnavailableOrBroken,
}

/// Six independent evidence dimensions from epic SC-15448.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryEvidenceDimension {
    StaticImplementation,
    DeclaredCalibration,
    HistoricalVerification,
    CurrentEnvironmentVerification,
    CanonicalRouteLoadability,
    ExactStrategyParameters,
}

impl MemoryEvidenceDimension {
    pub const ALL: [Self; 6] = [
        Self::StaticImplementation,
        Self::DeclaredCalibration,
        Self::HistoricalVerification,
        Self::CurrentEnvironmentVerification,
        Self::CanonicalRouteLoadability,
        Self::ExactStrategyParameters,
    ];
}

/// Why one evidence dimension does or does not cover a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryEvidenceVerdict {
    Satisfied,
    Missing,
    Unverified,
    Stale,
    FingerprintMismatch,
    CompositionMismatch,
    OutOfEnvelope,
    Invalid,
}

/// Explicit six-dimensional evidence result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryEvidenceDimensions {
    pub static_implementation: MemoryEvidenceVerdict,
    pub declared_calibration: MemoryEvidenceVerdict,
    pub historical_verification: MemoryEvidenceVerdict,
    pub current_environment_verification: MemoryEvidenceVerdict,
    pub canonical_route_loadability: MemoryEvidenceVerdict,
    pub exact_strategy_parameters: MemoryEvidenceVerdict,
}

impl MemoryEvidenceDimensions {
    pub const VERIFIED: Self = Self {
        static_implementation: MemoryEvidenceVerdict::Satisfied,
        declared_calibration: MemoryEvidenceVerdict::Satisfied,
        historical_verification: MemoryEvidenceVerdict::Satisfied,
        current_environment_verification: MemoryEvidenceVerdict::Satisfied,
        canonical_route_loadability: MemoryEvidenceVerdict::Satisfied,
        exact_strategy_parameters: MemoryEvidenceVerdict::Satisfied,
    };

    pub fn verdict(self, dimension: MemoryEvidenceDimension) -> MemoryEvidenceVerdict {
        match dimension {
            MemoryEvidenceDimension::StaticImplementation => self.static_implementation,
            MemoryEvidenceDimension::DeclaredCalibration => self.declared_calibration,
            MemoryEvidenceDimension::HistoricalVerification => self.historical_verification,
            MemoryEvidenceDimension::CurrentEnvironmentVerification => {
                self.current_environment_verification
            }
            MemoryEvidenceDimension::CanonicalRouteLoadability => self.canonical_route_loadability,
            MemoryEvidenceDimension::ExactStrategyParameters => self.exact_strategy_parameters,
        }
    }

    pub fn all_satisfied(self) -> bool {
        MemoryEvidenceDimension::ALL
            .into_iter()
            .all(|dimension| self.verdict(dimension) == MemoryEvidenceVerdict::Satisfied)
    }
}

/// Fully-qualified key for one evidence cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryEvidenceKey {
    pub resolved_route: String,
    pub backend: MemoryBackend,
    pub tier: MemoryNumericTier,
    /// Exact intra-phase materialization shape measured by this evidence cell.
    pub load_shape: LoadShape,
    pub mode: MemoryMode,
    pub overlay: Option<String>,
    pub geometry: MemoryGeometry,
    pub strategy: MemoryStrategy,
    /// Exact, canonically ordered strategy set active when this evidence was measured.
    pub engaged_composition: Vec<MemoryStrategy>,
    pub parameters: MemoryStrategyParameters,
}

impl MemoryEvidenceKey {
    fn has_canonical_engaged_composition(&self) -> bool {
        !self.engaged_composition.is_empty()
            && self
                .engaged_composition
                .windows(2)
                .all(|pair| pair[0] < pair[1])
    }
}

/// Numerical parity contract for an evidence record.
#[derive(Clone, Debug, PartialEq)]
pub enum MemoryParityContract {
    Exact,
    Tolerance {
        metric: String,
        maximum_error: f64,
    },
    Golden {
        fixture: String,
        metric: String,
        maximum_error: f64,
    },
}

/// Result of executing the declared numerical contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryParityResult {
    Passed,
    Failed { reason: String },
    NotRun,
}

/// Dynamic/static evidence handshake populated and consumed by caller-side admission selectors.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryEvidence {
    pub key: MemoryEvidenceKey,
    pub conformance: MemoryConformanceState,
    pub dimensions: MemoryEvidenceDimensions,
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    pub sceneworks_revision: String,
    pub inference_revision: String,
    pub harness_version: String,
    /// Absolute request peak **live allocation** the formula predicts — bytes held by the request's
    /// live tensors at the high-water mark, including request-resident assets but excluding any
    /// allocator reuse cache. A caller converts this to `MemoryRunContext::predicted_peak_bytes` by
    /// subtracting exactly the request-resident bytes already charged in its budget snapshot. See
    /// [`observed_peak_bytes`](Self::observed_peak_bytes) for why the distinction is load-bearing
    /// and not pedantic.
    pub predicted_peak_bytes: u64,
    /// Absolute request peak **live allocation** measured by the calibration harness. `None` until
    /// measured; it uses the same baseline as `predicted_peak_bytes` above.
    ///
    /// **Live allocation, not process footprint, and the two are not close.** An allocator's
    /// freed-buffer reuse cache is excluded here and counted by the OS: a Z-Image 1024² render
    /// measured 3102 MiB of live peak and 6488 MiB of `active + cache` on the same run. Comparing the
    /// wrong one against a cap is not conservative in a predictable direction — of three
    /// configurations measured together, the one with the **lowest** live peak had the **highest**
    /// footprint and was the only one an iOS device refused. A ladder built on the wrong quantity can
    /// rank a fatal configuration as the safest of its set.
    ///
    /// Live allocation is the right quantity *here* because it is the irreducible working set — the
    /// part no tuning removes. The cache is modelled on the other side of the comparison, by
    /// [`MemoryBudget::reclaimable_bytes`], whose docs carry the precondition that makes that
    /// modelling valid. A calibration that records footprint in this field silently double-counts
    /// the cache against a budget that already credits it.
    pub observed_peak_bytes: Option<u64>,
    pub parity: MemoryParityContract,
    pub parity_result: MemoryParityResult,
}

/// One machine-readable calibration observation emitted by a real-weight harness.
///
/// This is deliberately separate from [`MemoryEvidence`]. A harness observation is the provenance
/// input used to promote evidence; it must carry both the provider identity declared before the run
/// and the identity observed at the measurement seam. Making those two fields explicit lets the
/// verifier reject the exact stale-layout failure calibration fingerprints exist to detect.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryEvidenceLogRecord {
    pub key: MemoryEvidenceKey,
    pub declared_calibration: MemoryCalibrationIdentity,
    pub observed_calibration: MemoryCalibrationIdentity,
    pub predicted_peak_bytes: u64,
    pub observed_peak_bytes: u64,
    pub inference_revision: String,
    pub sceneworks_revision: String,
    /// Exact immutable model revision used by the probe.
    pub model_revision: String,
    /// SHA-256 of the canonical, dereferenced model-file inventory persisted with the evidence.
    pub model_inventory_sha256: String,
    pub harness_version: String,
    /// SHA-256 of the exact output bytes written by this probe. The A/B verifier binds each record
    /// to its artifact before it may promote the declared parity contract.
    pub output_sha256: String,
    pub parity: MemoryParityContract,
    pub parity_result: MemoryParityResult,
}

impl MemoryEvidenceLogRecord {
    /// Validate the persisted protocol invariants before emitting a line.
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.key.has_canonical_engaged_composition() {
            errors.push(
                "evidence engaged composition must be a non-empty canonical strategy set"
                    .to_owned(),
            );
        } else if self.key.engaged_composition.first() != Some(&MemoryStrategy::Resident)
            || self.key.engaged_composition.last() != Some(&self.key.strategy)
        {
            errors.push(
                "evidence engaged composition must start with resident and end with the measured strategy"
                    .to_owned(),
            );
        }
        if self.predicted_peak_bytes == 0 {
            errors.push("predicted peak bytes must be positive".to_owned());
        }
        if self.observed_peak_bytes == 0 {
            errors.push("observed peak bytes must be positive".to_owned());
        }
        for (label, identity) in [
            ("declared", &self.declared_calibration),
            ("observed", &self.observed_calibration),
        ] {
            if identity.abi != MEMORY_CALIBRATION_ABI {
                errors.push(format!(
                    "{label} calibration ABI {} does not match contract ABI {MEMORY_CALIBRATION_ABI}",
                    identity.abi
                ));
            }
            if identity.load_shape != self.key.load_shape {
                errors.push(format!(
                    "{label} calibration load shape {:?} does not match evidence key {:?}",
                    identity.load_shape, self.key.load_shape
                ));
            }
            if let Err(reason) = validate_calibration_fingerprint(&identity.fingerprint) {
                errors.push(format!("{label} calibration fingerprint {reason}"));
            }
        }
        if self.declared_calibration != self.observed_calibration {
            errors.push("declared and observed calibration identities do not match".to_owned());
        }
        for (label, revision) in [
            ("inference", self.inference_revision.as_str()),
            ("SceneWorks", self.sceneworks_revision.as_str()),
            ("model", self.model_revision.as_str()),
        ] {
            if revision.len() != 40
                || !revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                errors.push(format!(
                    "{label} revision must be an exact 40-character lowercase Git commit"
                ));
            }
        }
        if self.model_inventory_sha256.len() != 64
            || !self
                .model_inventory_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            errors.push(
                "model inventory SHA-256 must be 64 lowercase hexadecimal characters".to_owned(),
            );
        }
        if self.harness_version.trim().is_empty() {
            errors.push("harness version must be non-empty".to_owned());
        }
        if self.output_sha256.len() != 64
            || !self
                .output_sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            errors.push("output SHA-256 must be 64 lowercase hexadecimal characters".to_owned());
        }
        match &self.parity {
            MemoryParityContract::Exact => {}
            MemoryParityContract::Tolerance {
                metric,
                maximum_error,
            } => validate_parity_limit(metric, *maximum_error, &mut errors),
            MemoryParityContract::Golden {
                fixture,
                metric,
                maximum_error,
            } => {
                if fixture.trim().is_empty() {
                    errors.push("golden parity fixture must be non-empty".to_owned());
                }
                validate_parity_limit(metric, *maximum_error, &mut errors);
            }
        }
        errors
    }

    /// Serialize this observation as one canonical `MEMORY_EVIDENCE_V1` line.
    pub fn to_json_line(&self) -> Result<String> {
        let errors = self.validation_errors();
        if !errors.is_empty() {
            return Err(Error::Msg(format!(
                "invalid memory evidence log record: {}",
                errors.join("; ")
            )));
        }

        let json = serde_json::json!({
            "schema_version": 1,
            "key": evidence_key_json(&self.key),
            "declared_calibration": calibration_identity_json(&self.declared_calibration),
            "observed_calibration": calibration_identity_json(&self.observed_calibration),
            "predicted_peak_bytes": self.predicted_peak_bytes,
            "observed_peak_bytes": self.observed_peak_bytes,
            "inference_revision": self.inference_revision,
            "sceneworks_revision": self.sceneworks_revision,
            "model_revision": self.model_revision,
            "model_inventory_sha256": self.model_inventory_sha256,
            "harness_version": self.harness_version,
            "output_sha256": self.output_sha256,
            "parity": parity_contract_json(&self.parity),
            "parity_result": parity_result_json(&self.parity_result),
        });
        serde_json::to_string(&json)
            .map(|payload| format!("{MEMORY_EVIDENCE_V1_PREFIX}{payload}"))
            .map_err(|error| Error::Msg(format!("serialize memory evidence log record: {error}")))
    }
}

/// Validate the stable content-fingerprint grammar used at the persisted evidence boundary.
///
/// Fingerprints are lowercase kebab tokens and contain exactly one positive `vN` token. Backend,
/// mode, tier, load shape, geometry, and strategy parameters are typed evidence-key axes and must
/// not be encoded as substitutes for those fields. Existing descriptive tokens may still mention
/// implementation facts; the version token is what makes a deliberate calibration change lintable.
pub fn validate_calibration_fingerprint(fingerprint: &str) -> std::result::Result<(), String> {
    if fingerprint.is_empty() {
        return Err("must be non-empty".to_owned());
    }
    let tokens = fingerprint.split('-').collect::<Vec<_>>();
    if tokens.iter().any(|token| {
        token.is_empty()
            || !token
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    }) {
        return Err("must contain only non-empty lowercase ASCII kebab tokens".to_owned());
    }
    let version_tokens = tokens
        .iter()
        .filter(|token| {
            token.strip_prefix('v').is_some_and(|digits| {
                !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
            })
        })
        .collect::<Vec<_>>();
    if version_tokens.len() != 1 {
        return Err("must contain exactly one vN version token".to_owned());
    }
    let digits = version_tokens[0].strip_prefix('v').unwrap_or_default();
    if digits.starts_with('0') {
        return Err(
            "version token must be positive and must not contain leading zeroes".to_owned(),
        );
    }
    Ok(())
}

fn calibration_identity_json(identity: &MemoryCalibrationIdentity) -> serde_json::Value {
    serde_json::json!({
        "abi": identity.abi,
        "fingerprint": identity.fingerprint,
        "load_shape": load_shape_key(identity.load_shape),
    })
}

fn evidence_key_json(key: &MemoryEvidenceKey) -> serde_json::Value {
    let component_precision_floors = key
        .tier
        .component_precision_floors
        .iter()
        .map(|floor| {
            serde_json::json!({
                "component": floor.component.as_str(),
                "selected_tier": quant_key(floor.selected_tier),
                "resident_tier": quant_key(floor.resident_tier),
            })
        })
        .collect::<Vec<_>>();
    serde_json::json!({
        "resolved_route": key.resolved_route,
        "backend": key.backend.as_key(),
        "tier": {
            "precision": precision_key(key.tier.precision),
            "quant": key.tier.quant.map(quant_key),
            "component_precision_floors": component_precision_floors,
        },
        "load_shape": load_shape_key(key.load_shape),
        "mode": key.mode.as_key(),
        "overlay": key.overlay,
        "geometry": {
            "width": key.geometry.width,
            "height": key.geometry.height,
            "batch": key.geometry.batch,
            "frames": key.geometry.frames,
            "reference_count": key.geometry.reference_count,
        },
        "strategy": memory_strategy_key(key.strategy),
        "engaged_composition": key.engaged_composition.iter().copied().map(memory_strategy_key).collect::<Vec<_>>(),
        "parameters": {
            "decode_tile_edge": key.parameters.decode_tile_edge,
            "decode_overlap": key.parameters.decode_overlap,
            "attention_chunk_size": key.parameters.attention_chunk_size,
            "transformer_window_size": key.parameters.transformer_window_size,
            "transformer_window_component": key.parameters.transformer_window_component.map(transformer_component_key),
        },
    })
}

fn parity_contract_json(parity: &MemoryParityContract) -> serde_json::Value {
    match parity {
        MemoryParityContract::Exact => serde_json::json!({ "kind": "exact" }),
        MemoryParityContract::Tolerance {
            metric,
            maximum_error,
        } => serde_json::json!({
            "kind": "tolerance",
            "metric": metric,
            "maximum_error": maximum_error,
        }),
        MemoryParityContract::Golden {
            fixture,
            metric,
            maximum_error,
        } => serde_json::json!({
            "kind": "golden",
            "fixture": fixture,
            "metric": metric,
            "maximum_error": maximum_error,
        }),
    }
}

fn parity_result_json(result: &MemoryParityResult) -> serde_json::Value {
    match result {
        MemoryParityResult::Passed => serde_json::json!({ "kind": "passed" }),
        MemoryParityResult::Failed { reason } => {
            serde_json::json!({ "kind": "failed", "reason": reason })
        }
        MemoryParityResult::NotRun => serde_json::json!({ "kind": "not_run" }),
    }
}

const fn memory_strategy_key(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

const fn transformer_component_key(component: TransformerComponent) -> &'static str {
    match component {
        TransformerComponent::Dit => "dit",
        TransformerComponent::TextEncoder => "text_encoder",
        TransformerComponent::Both => "both",
    }
}

const fn load_shape_key(load_shape: LoadShape) -> &'static str {
    match load_shape {
        LoadShape::EagerMaterialization => "eager_materialization",
        LoadShape::DeferredMaterialization => "deferred_materialization",
    }
}

const fn precision_key(precision: Precision) -> &'static str {
    match precision {
        Precision::Bf16 => "bf16",
        Precision::Fp32 => "fp32",
    }
}

const fn quant_key(quant: Quant) -> &'static str {
    match quant {
        Quant::Q4 => "q4",
        Quant::Q8 => "q8",
        Quant::Nvfp4 => "nvfp4",
    }
}

impl MemoryEvidence {
    pub fn predicted_peak_breakdown(
        &self,
        contract: &MemoryProviderContract,
    ) -> MemoryPeakBreakdown {
        contract.decompose_predicted_peak(self.predicted_peak_bytes)
    }

    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if !self.key.has_canonical_engaged_composition() {
            errors.push(
                "evidence engaged composition must be a non-empty canonical strategy set"
                    .to_owned(),
            );
        }
        if self.observed_peak_bytes.is_none() {
            errors.push("verified evidence requires an observed peak".to_owned());
        }
        match &self.parity {
            MemoryParityContract::Exact => {}
            MemoryParityContract::Tolerance {
                metric,
                maximum_error,
            } => validate_parity_limit(metric, *maximum_error, &mut errors),
            MemoryParityContract::Golden {
                fixture,
                metric,
                maximum_error,
            } => {
                if fixture.trim().is_empty() {
                    errors.push("golden parity fixture must be non-empty".to_owned());
                }
                validate_parity_limit(metric, *maximum_error, &mut errors);
            }
        }
        match &self.parity_result {
            MemoryParityResult::Passed => {}
            MemoryParityResult::Failed { reason } if reason.trim().is_empty() => {
                errors.push("failed parity result requires a reason".to_owned());
            }
            MemoryParityResult::Failed { .. } | MemoryParityResult::NotRun => {}
        }
        errors
    }

    /// Only a fully verified, six-dimension record matching the current provider handshake may
    /// authorize an optimized fit.
    pub fn optimized_eligibility(
        &self,
        contract: &MemoryProviderContract,
    ) -> std::result::Result<(), MemoryEvidenceVerdict> {
        if !self.key.has_canonical_engaged_composition() {
            return Err(MemoryEvidenceVerdict::Invalid);
        }
        let selection = MemorySelection {
            strategy: self.key.strategy,
            parameters: self.key.parameters,
            tier: self.key.tier,
        };
        if self.key.engaged_composition != contract.engaged_composition_for_selection(&selection) {
            return Err(MemoryEvidenceVerdict::CompositionMismatch);
        }
        if !self.key.strategy.is_optimized() {
            return Ok(());
        }
        if self.conformance != MemoryConformanceState::Verified {
            return Err(MemoryEvidenceVerdict::Unverified);
        }
        if !self.validation_errors().is_empty() {
            return Err(MemoryEvidenceVerdict::Invalid);
        }
        if self.parity_result != MemoryParityResult::Passed {
            return Err(MemoryEvidenceVerdict::Unverified);
        }
        for dimension in MemoryEvidenceDimension::ALL {
            let verdict = self.dimensions.verdict(dimension);
            if verdict != MemoryEvidenceVerdict::Satisfied {
                return Err(verdict);
            }
        }
        let Some(identity) = &contract.calibration else {
            return Err(MemoryEvidenceVerdict::Unverified);
        };
        if self.key.load_shape != contract.load_shape || self.key.load_shape != identity.load_shape
        {
            return Err(MemoryEvidenceVerdict::FingerprintMismatch);
        }
        if self.calibration_abi != identity.abi
            || self.calibration_fingerprint != identity.fingerprint
        {
            return Err(MemoryEvidenceVerdict::FingerprintMismatch);
        }
        Ok(())
    }
}

fn validate_parity_limit(metric: &str, maximum_error: f64, errors: &mut Vec<String>) {
    if metric.trim().is_empty() {
        errors.push("parity metric must be non-empty".to_owned());
    }
    if !maximum_error.is_finite() || maximum_error < 0.0 {
        errors.push("parity maximum_error must be finite and non-negative".to_owned());
    }
}

/// Truthful caller-facing pre-OOM rejection payload. The caller may populate a measured smaller
/// geometry; it must not invent advice from unknown or unverified evidence. The contract defines
/// this vocabulary even when an in-process provider does not construct it directly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRejection {
    pub available_bytes: u64,
    pub minimum_verified_peak_bytes: Option<u64>,
    pub smaller_verified_geometry: Option<MemoryGeometry>,
    pub exclusion_reason: MemoryEvidenceVerdict,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LoadSpec, WeightsSource};

    #[test]
    fn adapter_residency_distinguishes_folded_additive_and_missing_evidence() {
        // sc-17755: a `TempDir` guard, so the tree leaves on `Drop` even when an assertion below
        // panics — the trailing `remove_dir_all` this replaced never ran on a failing test.
        let tmp = tempfile::Builder::new()
            .prefix("gen-core-adapter-residency-")
            .tempdir()
            .expect("fixture temp dir");
        let root = tmp.path();
        let first = root.join("first.safetensors");
        let second = root.join("second.safetensors");
        std::fs::write(&first, vec![0_u8; 11]).unwrap();
        std::fs::write(&second, vec![0_u8; 17]).unwrap();
        let adapters = vec![
            AdapterSpec::new(first, 1.0, crate::AdapterKind::Lora),
            AdapterSpec::new(second, 0.5, crate::AdapterKind::Lokr),
        ];

        assert_eq!(
            adapter_stack_resident_bytes(&adapters, AdapterResidencyMode::Folded),
            Some(0)
        );
        assert_eq!(
            adapter_stack_resident_bytes(&adapters, AdapterResidencyMode::Additive),
            Some(28)
        );

        let missing = vec![AdapterSpec::new(
            root.join("missing.safetensors"),
            1.0,
            crate::AdapterKind::Lora,
        )];
        assert_eq!(
            adapter_stack_resident_bytes(&missing, AdapterResidencyMode::Additive),
            None
        );
    }

    fn mlx_backend() -> MemoryBackendRealization {
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        }
    }

    fn implemented(strategy: MemoryStrategy) -> MemoryStrategyCapability {
        MemoryStrategyCapability {
            strategy,
            support: MemoryStrategySupport::Implemented,
            parameters: MemoryParameterRanges::default(),
        }
    }

    fn adopted_contract() -> MemoryProviderContract {
        let mut strategies: Vec<_> = MemoryStrategy::ALL.into_iter().map(implemented).collect();
        strategies[2].parameters.decode_tile_edges = vec![512, 384];
        strategies[2].parameters.decode_overlaps = vec![64];
        strategies[3].parameters.attention_chunk_sizes = vec![256, 128];
        strategies[4].parameters.transformer_window_sizes = vec![2, 1];
        MemoryProviderContract {
            provider_id: "test-provider".to_owned(),
            backend: mlx_backend(),
            strategies,
            decode_geometry_policy_authoritative: false,
            pid_decode_routes: None,
            load_shape: LoadShape::DeferredMaterialization,
            additional_prerequisites: Vec::new(),
            default_engagement_exclusions: Vec::new(),
            resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
            lifecycle: MemoryLifecycleCapabilities {
                phases: vec![
                    MemoryPhase::Conditioning,
                    MemoryPhase::Denoise,
                    MemoryPhase::Decode,
                ],
                synchronized_phase_release: true,
                decode_tiling: true,
                attention_chunking: true,
                transformer_window_materialization: true,
            },
            formula: MemoryFormulaKind::PhaseEnvelope {
                phases: vec![
                    MemoryPhase::Conditioning,
                    MemoryPhase::Denoise,
                    MemoryPhase::Decode,
                ],
                variables: vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::PixelCount,
                ],
            },
            calibration: Some(MemoryCalibrationIdentity::new(
                "test-layout-v1",
                LoadShape::DeferredMaterialization,
            )),
            asset_facts: MemoryAssetFacts::default(),
            runtime: MemoryRuntimeSemantics::default(),
        }
    }

    fn admitted_context(contract: &MemoryProviderContract) -> MemoryRunContext {
        let calibration = contract.calibration.as_ref().expect("calibration");
        MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            optimization_authority: MemoryOptimizationAuthority::Resident,
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 512,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    fn decode_quality_policy(contract: &MemoryProviderContract) -> MemoryDecodeGeometryPolicy {
        MemoryDecodeGeometryPolicy {
            quality_abi: MEMORY_DECODE_QUALITY_ABI,
            family: "test-family".to_owned(),
            resolved_route: contract.provider_id.clone(),
            backend: MemoryBackend::Mlx,
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
            load_shape: contract.load_shape,
            artifact: MemoryDecodeArtifactIdentity {
                repository: "SceneWorks/test-model".to_owned(),
                revision: "a".repeat(40),
                variant: "q4".to_owned(),
                fingerprint: format!("SceneWorks/test-model@{}:q4", "a".repeat(40)),
            },
            implementation_fingerprint: MEMORY_DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT.to_owned(),
            mode: MemoryMode::TextToImage,
            overlay: None,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            use_pid: false,
            tile_edge: 512,
            overlap: 64,
            metric: "max_abs_rgb_u8".to_owned(),
            maximum_error: 48,
            fixtures: [(7, 'a', 'b', 'c', 31), (99, 'd', 'e', 'f', 42)]
                .into_iter()
                .map(
                    |(seed, latent, dense, tiled, observed_error)| MemoryDecodeQualityFixture {
                        seed,
                        production_latent_provenance: format!("test-model@revision seed={seed}"),
                        production_latent_sha256: latent.to_string().repeat(64),
                        dense_output_sha256: dense.to_string().repeat(64),
                        tiled_output_sha256: tiled.to_string().repeat(64),
                        observed_error,
                    },
                )
                .collect(),
            production_evidence_sha256: String::new(),
            disposition: MemoryDecodeQualityDisposition::Admitted,
        }
        .seal()
    }

    fn decode_quality_runtime_identity(
        policy: &MemoryDecodeGeometryPolicy,
    ) -> MemoryDecodeQualityRuntimeIdentity {
        MemoryDecodeQualityRuntimeIdentity {
            artifact: policy.artifact.clone(),
            implementation_fingerprint: policy.implementation_fingerprint.clone(),
        }
    }

    #[test]
    fn geometry_decode_quality_is_exact_fail_closed_and_self_authenticating() {
        let mut contract = adopted_contract();
        contract.provider_id = "sdxl".to_owned();
        let mut policy = decode_quality_policy(&contract);
        policy.resolved_route = "realvisxl".to_owned();
        policy = policy.seal();
        assert_eq!(
            policy.production_evidence_sha256, MEMORY_DECODE_QUALITY_CANONICAL_FIXTURE_SHA256,
            "Rust and the correctness-only collector must share one canonical quality ABI"
        );
        assert!(contract
            .adopt_decode_geometry_policies("test-family", vec![policy.clone()])
            .unwrap());
        assert!(contract.lifecycle.decode_tiling);
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.parameters.decode_tile_edges, [512]);
        assert_eq!(decode.parameters.decode_overlaps, [64]);
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );

        let query = MemoryDecodePolicyQuery {
            tier: policy.tier,
            load_shape: policy.load_shape,
            mode_key: "text_to_image",
            overlay: None,
            geometry: policy.geometry,
            use_pid: false,
        };
        let admitted = contract
            .decode_geometry_policy(query, None)
            .unwrap()
            .expect("policy adopter returns the exact row");
        assert_eq!((admitted.tile_edge, admitted.overlap), (512, 64));
        assert!(contract
            .decode_geometry_policy(
                query,
                Some(MemoryStrategyParameters {
                    decode_tile_edge: Some(384),
                    decode_overlap: Some(64),
                    ..Default::default()
                })
            )
            .unwrap_err()
            .to_string()
            .contains("do not match"));

        let off_grid = MemoryDecodePolicyQuery {
            geometry: MemoryGeometry {
                width: 513,
                ..policy.geometry
            },
            ..query
        };
        assert!(contract
            .decode_geometry_policy(off_grid, None)
            .unwrap_err()
            .to_string()
            .contains("no admitted"));

        contract.strategies[MemoryStrategy::BoundedDecode as usize]
            .parameters
            .decode_geometry_policies[0]
            .fixtures[0]
            .observed_error += 1;
        assert!(contract
            .conformance_errors()
            .iter()
            .any(|error| error.contains("does not bind the canonical")));
    }

    #[test]
    fn load_quality_table_requires_the_exact_resolved_route() {
        let contract = adopted_contract();
        let mut policy = decode_quality_policy(&contract);
        policy.resolved_route = "realvisxl".to_owned();
        policy = policy.seal();

        let unbound = LoadSpec::new(WeightsSource::Dir("/models/sdxl".into()))
            .with_decode_geometry_policies(vec![policy.clone()])
            .with_decode_quality_runtime_identity(decode_quality_runtime_identity(&policy));
        assert!(unbound.route_bound_decode_geometry_policies().is_err());

        let sibling = unbound.clone().with_resolved_route("realvisxl_lightning");
        assert!(sibling.route_bound_decode_geometry_policies().is_err());

        let empty = unbound.clone().with_resolved_route("   ");
        assert!(empty.route_bound_decode_geometry_policies().is_err());

        let exact = unbound.with_resolved_route("realvisxl");
        assert_eq!(
            exact.route_bound_decode_geometry_policies().unwrap(),
            [policy]
        );
    }

    #[test]
    fn load_quality_table_requires_every_artifact_and_implementation_identity_axis() {
        let contract = adopted_contract();
        let mut policy = decode_quality_policy(&contract);
        policy.resolved_route = "realvisxl".to_owned();
        policy = policy.seal();
        let base = LoadSpec::new(WeightsSource::Dir("/models/sdxl".into()))
            .with_resolved_route("realvisxl")
            .with_decode_geometry_policies(vec![policy.clone()]);
        assert!(base
            .route_bound_decode_geometry_policies()
            .unwrap_err()
            .to_string()
            .contains("independently resolved"));

        let exact = decode_quality_runtime_identity(&policy);
        for (label, identity) in [
            ("repository", {
                let mut value = exact.clone();
                value.artifact.repository.push_str("-sibling");
                value
            }),
            ("revision", {
                let mut value = exact.clone();
                value.artifact.revision = "b".repeat(40);
                value
            }),
            ("variant", {
                let mut value = exact.clone();
                value.artifact.variant = "q8".to_owned();
                value
            }),
            ("fingerprint", {
                let mut value = exact.clone();
                value.artifact.fingerprint.push_str("-drift");
                value
            }),
            ("implementation", {
                let mut value = exact.clone();
                value.implementation_fingerprint = "f".repeat(64);
                value
            }),
        ] {
            let error = base
                .clone()
                .with_decode_quality_runtime_identity(identity)
                .route_bound_decode_geometry_policies()
                .expect_err(label);
            assert!(
                error
                    .to_string()
                    .contains("cannot authorize loaded identity"),
                "{label}: {error}"
            );
        }
        assert_eq!(
            base.with_decode_quality_runtime_identity(exact)
                .route_bound_decode_geometry_policies()
                .unwrap(),
            [policy]
        );
    }

    #[test]
    fn exact_tile_pair_is_policy_identity_and_selection_controls_decode_engagement() {
        let mut contract = adopted_contract();
        let first = decode_quality_policy(&contract);
        let mut second = first.clone();
        second.tile_edge = 384;
        second.overlap = 32;
        second = second.seal();
        contract
            .adopt_decode_geometry_policies("test-family", vec![first.clone(), second.clone()])
            .unwrap();
        let query = MemoryDecodePolicyQuery {
            tier: first.tier,
            load_shape: first.load_shape,
            mode_key: "text_to_image",
            overlay: None,
            geometry: first.geometry,
            use_pid: false,
        };
        assert!(contract
            .decode_geometry_policy(query, None)
            .unwrap_err()
            .to_string()
            .contains("multiple tile pairs"));
        let select_pair = |tile_edge, overlap| MemoryStrategyParameters {
            decode_tile_edge: Some(tile_edge),
            decode_overlap: Some(overlap),
            ..Default::default()
        };
        assert_eq!(
            contract
                .decode_geometry_policy(query, Some(select_pair(384, 32)))
                .unwrap()
                .unwrap()
                .production_evidence_sha256,
            second.production_evidence_sha256
        );

        let independent_attention = MemorySelection {
            strategy: MemoryStrategy::BoundedAttention,
            parameters: MemoryStrategyParameters {
                attention_chunk_size: Some(128),
                ..Default::default()
            },
            tier: first.tier,
        };
        contract.validate_selection(&independent_attention).unwrap();
        assert!(!contract.engages_selection(&independent_attention, MemoryStrategy::BoundedDecode));
        let request = contract
            .generation_memory(&independent_attention)
            .expect("optimized request memory");
        assert!(!request.tile_vae_decode);
        assert!(request.chunk_attention);

        let admitted_attention = MemorySelection {
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(384),
                decode_overlap: Some(32),
                attention_chunk_size: Some(128),
                ..Default::default()
            },
            ..independent_attention
        };
        contract.validate_selection(&admitted_attention).unwrap();
        assert!(contract.engages_selection(&admitted_attention, MemoryStrategy::BoundedDecode));
        assert!(
            contract
                .generation_memory(&admitted_attention)
                .unwrap()
                .tile_vae_decode
        );

        let decode_without_pair = MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters::default(),
            tier: first.tier,
        };
        assert!(contract.validate_selection(&decode_without_pair).is_err());

        let mut duplicate = first.clone();
        duplicate.fixtures[0].seed += 1;
        duplicate = duplicate.seal();
        let mut duplicate_contract = adopted_contract();
        assert!(duplicate_contract
            .adopt_decode_geometry_policies("test-family", vec![first, duplicate])
            .unwrap_err()
            .to_string()
            .contains("duplicate decode quality coordinate"));
    }

    #[test]
    fn calibration_free_optimization_requires_explicit_estimate_authority() {
        let mut contract = adopted_contract();
        let mut context = admitted_context(&contract);
        contract.calibration = None;
        context.selection.strategy = MemoryStrategy::StagedResidency;
        context.optimization_authority = MemoryOptimizationAuthority::Estimated;
        assert_eq!(
            default_memory_strategy_safety_check(&contract, &context),
            MemorySafetyDecision::Accept,
            "a caller-admitted conservative estimate authorizes an implemented calibration-free route"
        );

        context.optimization_authority = MemoryOptimizationAuthority::Calibrated;
        assert!(matches!(
            default_memory_strategy_safety_check(&contract, &context),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("requires explicit estimate authority")
        ));
        context.optimization_authority = MemoryOptimizationAuthority::Resident;
        assert!(matches!(
            default_memory_strategy_safety_check(&contract, &context),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("no explicit measured or estimated authority")
        ));
    }

    #[test]
    fn loaded_quality_table_selects_the_exact_tier_and_load_shape() {
        let contract = adopted_contract();
        let mut q4 = decode_quality_policy(&contract);
        q4.resolved_route = "realvisxl".to_owned();
        q4 = q4.seal();
        let mut q8 = q4.clone();
        q8.tier.quant = Some(Quant::Q8);
        q8.artifact.variant = "q8".to_owned();
        q8.artifact.fingerprint = format!(
            "{}@{}:{}",
            q8.artifact.repository, q8.artifact.revision, q8.artifact.variant
        );
        q8 = q8.seal();

        let spec = LoadSpec::new(WeightsSource::Dir("/models/sdxl".into()))
            .with_resolved_route("realvisxl")
            .with_decode_geometry_policies(vec![q4.clone(), q8.clone()])
            .with_decode_quality_runtime_identity(decode_quality_runtime_identity(&q4))
            .with_quant(Quant::Q4)
            .with_load_shape(LoadShape::DeferredMaterialization);
        assert_eq!(
            spec.decode_geometry_policies_for_loaded_contract(
                MemoryBackend::Mlx,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            )
            .unwrap(),
            [q4.clone()]
        );
        let q8_spec = spec
            .clone()
            .with_decode_quality_runtime_identity(decode_quality_runtime_identity(&q8));
        assert_eq!(
            q8_spec
                .decode_geometry_policies_for_loaded_contract(
                    MemoryBackend::Mlx,
                    MemoryNumericTier {
                        precision: Precision::Bf16,
                        quant: Some(Quant::Q8),
                        component_precision_floors: &[],
                    },
                )
                .unwrap(),
            [q8]
        );
        let eager = spec.with_load_shape(LoadShape::EagerMaterialization);
        assert!(
            eager.decode_geometry_policy_authoritative,
            "an exact projection that yields no row must retain the caller-declared semantic scope"
        );
        assert_eq!(
            eager
                .decode_geometry_policies_for_loaded_contract(
                    MemoryBackend::Mlx,
                    MemoryNumericTier {
                        precision: Precision::Bf16,
                        quant: Some(Quant::Q4),
                        component_precision_floors: &[],
                    },
                )
                .unwrap(),
            Vec::<MemoryDecodeGeometryPolicy>::new()
        );
    }

    #[test]
    fn loaded_quality_table_cannot_mix_semantic_families() {
        let mut contract = adopted_contract();
        let first = decode_quality_policy(&contract);
        let mut second = first.clone();
        second.family = "another-family".to_owned();
        second.geometry.width = 768;
        second = second.seal();

        let error = contract
            .adopt_decode_geometry_policies("test-family", vec![first, second])
            .unwrap_err();
        assert!(error.to_string().contains("does not match expected family"));
    }

    #[test]
    fn empty_decode_quality_table_preserves_legacy_parameter_fallback() {
        let contract = adopted_contract();
        let context = admitted_context(&contract);
        assert!(contract
            .decode_geometry_policy(
                MemoryDecodePolicyQuery {
                    tier: context.selection.tier,
                    load_shape: context.load_shape,
                    mode_key: context.mode.as_key(),
                    overlay: context.overlay.as_deref(),
                    geometry: context.geometry,
                    use_pid: context.use_pid,
                },
                None,
            )
            .unwrap()
            .is_none());
    }

    #[test]
    fn declared_empty_decode_quality_scope_fails_closed_without_blocking_independent_rungs() {
        let mut contract = adopted_contract();
        contract.decode_geometry_policy_authoritative = true;
        let context = admitted_context(&contract);
        let query = MemoryDecodePolicyQuery {
            tier: context.selection.tier,
            load_shape: context.load_shape,
            mode_key: context.mode.as_key(),
            overlay: context.overlay.as_deref(),
            geometry: context.geometry,
            use_pid: context.use_pid,
        };
        let error = contract
            .decode_geometry_policy(query, None)
            .expect_err("declared semantic scope cannot fall back to route-blind decode");
        assert!(error.to_string().contains("declared quality scope"));

        let independent = MemorySelection {
            strategy: MemoryStrategy::BoundedAttention,
            parameters: MemoryStrategyParameters {
                attention_chunk_size: Some(256),
                ..Default::default()
            },
            tier: context.selection.tier,
        };
        assert!(!contract.engages_selection(&independent, MemoryStrategy::BoundedDecode));
        assert!(contract.engages_selection(&independent, MemoryStrategy::BoundedAttention));
        let memory = contract
            .generation_memory(&independent)
            .expect("optimized selection memory controls");
        assert!(!memory.tile_vae_decode);
        assert!(memory.chunk_attention);
    }

    #[test]
    fn behavior_fixture_preserves_exact_reference_cardinality() {
        let contract = adopted_contract();
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let context = standard_memory_behavior_context(
            &contract,
            MemoryStrategy::Resident,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::Edit,
                reference_count: 4,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .expect("behavior context");
        let fixture = crate::MemoryBehaviorFixture::new(context);
        assert!(fixture.context.has_reference);
        assert_eq!(fixture.context.geometry.reference_count, 4);
        assert_eq!(fixture.request.conditioning.len(), 4);
        assert!(fixture
            .request
            .conditioning
            .iter()
            .all(|conditioning| matches!(conditioning, crate::Conditioning::Reference { .. })));
    }

    #[test]
    fn default_safety_check_rejects_mutated_calibration_handshake() {
        let contract = adopted_contract();
        let context = admitted_context(&contract);
        assert_eq!(
            default_memory_strategy_safety_check(&contract, &context),
            MemorySafetyDecision::Accept
        );

        let mut wrong_fingerprint = context.clone();
        wrong_fingerprint.calibration_fingerprint.push_str("-stale");
        assert!(matches!(
            default_memory_strategy_safety_check(&contract, &wrong_fingerprint),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("calibration handshake mismatch")
        ));

        let mut wrong_shape = context.clone();
        wrong_shape.load_shape = LoadShape::EagerMaterialization;
        assert!(matches!(
            default_memory_strategy_safety_check(&contract, &wrong_shape),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("calibration handshake mismatch")
        ));

        let mut wrong_abi = context;
        wrong_abi.calibration_abi += 1;
        assert!(matches!(
            default_memory_strategy_safety_check(&contract, &wrong_abi),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("calibration handshake mismatch")
        ));
    }

    #[test]
    fn calibration_identity_load_shape_must_match_contract() {
        let mut contract = adopted_contract();
        contract.calibration.as_mut().unwrap().load_shape = LoadShape::EagerMaterialization;
        assert!(contract
            .conformance_errors()
            .iter()
            .any(|error| error.contains("calibration load shape")));
    }

    #[test]
    fn standard_safety_check_runs_tier_route_and_budget_gates() {
        let contract = adopted_contract();
        let context = admitted_context(&contract);
        let loaded_tier = context.selection.tier;
        assert_eq!(
            standard_memory_strategy_safety_check(&contract, &context, Some(loaded_tier), None),
            MemorySafetyDecision::Accept
        );
        let matching_spec = crate::LoadSpec::new(crate::WeightsSource::Dir("/weights".into()))
            .with_quant(Quant::Q4);
        assert_eq!(
            default_registered_memory_strategy_safety_check(&matching_spec, &contract, &context),
            MemorySafetyDecision::Accept
        );
        let wrong_spec = matching_spec.with_quant(Quant::Q8);
        assert!(matches!(
            default_registered_memory_strategy_safety_check(&wrong_spec, &contract, &context),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("does not match loaded tier")
        ));

        let mut wrong_tier = context.clone();
        wrong_tier.selection.tier.quant = Some(Quant::Q8);
        assert!(matches!(
            standard_memory_strategy_safety_check(
                &contract,
                &wrong_tier,
                Some(loaded_tier),
                None
            ),
            MemorySafetyDecision::Reject { reason } if reason.contains("does not match loaded tier")
        ));

        let route_gate = || Err(Error::Unsupported("route refused".to_owned()));
        assert!(matches!(
            standard_memory_strategy_safety_check(
                &contract,
                &context,
                Some(loaded_tier),
                Some(&route_gate)
            ),
            MemorySafetyDecision::Reject { reason } if reason.contains("route refused")
        ));

        let mut invalid_reclaim_credit = context.clone();
        invalid_reclaim_credit.budget.committed_bytes = 10;
        invalid_reclaim_credit.budget.reclaimable_bytes = 11;
        assert!(matches!(
            standard_memory_strategy_safety_check(
                &contract,
                &invalid_reclaim_credit,
                Some(loaded_tier),
                None
            ),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("reclaim credit must be a subset")
        ));

        let mut over_budget = context;
        over_budget.predicted_peak_bytes = 1025;
        assert!(matches!(
            standard_memory_strategy_safety_check(
                &contract,
                &over_budget,
                Some(loaded_tier),
                None
            ),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("incremental live demand 1025 exceeds effective budget 1024")
        ));
    }

    fn pid_contract() -> MemoryProviderContract {
        let mut contract = adopted_contract();
        contract.strategies[2].parameters.decode_tile_edges = vec![512, 2048];
        contract.strategies[2].parameters.decode_overlaps = vec![64, 256];
        contract.pid_decode_routes = Some(MemoryPidDecodeRoutes {
            native: MemoryDecodeRouteDomain {
                tile_edges: vec![512],
                tile_overlap: 64,
            },
            pid: MemoryDecodeRouteDomain {
                tile_edges: vec![2048],
                tile_overlap: 256,
            },
        });
        contract
    }

    #[test]
    fn pid_decode_routes_must_be_disjoint_and_match_the_published_ranges() {
        let valid = pid_contract();
        assert!(valid.conformance_errors().is_empty());

        let mut overlapping = valid.clone();
        overlapping.pid_decode_routes.as_mut().unwrap().pid = MemoryDecodeRouteDomain {
            tile_edges: vec![512],
            tile_overlap: 64,
        };
        let errors = overlapping.conformance_errors();
        assert!(errors
            .iter()
            .any(|error| error.contains("must be disjoint")));

        let mut under_declared = valid;
        under_declared.strategies[2]
            .parameters
            .decode_tile_edges
            .retain(|edge| *edge != 2048);
        let errors = under_declared.conformance_errors();
        assert!(errors
            .iter()
            .any(|error| error.contains("must equal the native/PiD route union")));
    }

    #[test]
    fn compatibility_default_never_advertises_an_optimized_rung() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<MemoryAssetFacts>();
        let legacy_literal = MemoryAssetFacts {
            base_bytes: 1,
            conditioning_bytes: 2,
            transformer_bytes: 3,
            decoder_bytes: 4,
            overlay_bytes: 5,
        };
        assert_eq!(legacy_literal.overlay_bytes, 5);

        let contract = MemoryProviderContract::compatibility_default("legacy", mlx_backend());
        assert!(contract.conformance_errors().is_empty());
        assert!(contract.calibration.is_none());
        assert!(contract.formula.resident_components().is_empty());
        assert_eq!(contract.asset_facts.overlay_bytes, 0);
        let prediction = contract.predicted_peak_from_base(123);
        assert_eq!(prediction.predicted_peak_bytes(), 123);
        assert_eq!(prediction.unattributed_bytes, 123);
        assert!(prediction.components.is_empty());
        for strategy in MemoryStrategy::ALL {
            let support = &contract.capability(strategy).unwrap().support;
            if strategy == MemoryStrategy::Resident {
                assert_eq!(support, &MemoryStrategySupport::Implemented);
            } else {
                assert_eq!(support, &MemoryStrategySupport::Missing);
            }
        }

        let mut context = admitted_context(&adopted_contract());
        assert_eq!(
            default_memory_strategy_safety_check(&contract, &context),
            MemorySafetyDecision::Accept,
            "calibration-free compatibility contracts preserve resident admission"
        );
        context.selection.strategy = MemoryStrategy::StagedResidency;
        assert!(matches!(
            default_memory_strategy_safety_check(&contract, &context),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("has no explicit measured or estimated authority")
        ));

        let mut mutated = contract.clone();
        mutated
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .unwrap()
            .support = MemoryStrategySupport::Implemented;
        assert!(matches!(
            default_memory_strategy_safety_check(&mutated, &context),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("has no explicit measured or estimated authority")
        ));
        context.optimization_authority = MemoryOptimizationAuthority::Estimated;
        assert_eq!(
            default_memory_strategy_safety_check(&mutated, &context),
            MemorySafetyDecision::Accept,
            "explicitly authorized conservative estimates preserve calibration-free optimized fallback"
        );
    }

    #[test]
    fn asset_facts_reject_overlay_contamination_and_ambiguous_overlay_formulas() {
        let mut contract = adopted_contract();
        contract.asset_facts = MemoryAssetFacts {
            base_bytes: 60,
            conditioning_bytes: 10,
            transformer_bytes: 40,
            decoder_bytes: 10,
            overlay_bytes: 7,
        };
        contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: contract.lifecycle.phases.clone(),
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::OverlayBytes,
            ],
            resident_components: vec![MemoryResidentComponent {
                id: "control".to_owned(),
                kind: MemoryComponentKind::ControlBranch,
                resident_bytes: 7,
                bounded_by: None,
            }],
        };
        assert!(contract.conformance_errors().is_empty());

        let mut contaminated = contract.clone();
        contaminated.asset_facts.base_bytes += contaminated.asset_facts.overlay_bytes;
        assert!(contaminated
            .conformance_errors()
            .iter()
            .any(|error| error.contains("exclude overlay_bytes")));

        let mut ambiguous = contract;
        ambiguous.formula = MemoryFormulaKind::PhaseEnvelope {
            phases: ambiguous.lifecycle.phases.clone(),
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::OverlayBytes,
            ],
        };
        assert!(ambiguous
            .conformance_errors()
            .iter()
            .any(|error| error.contains("typed auxiliary resident components")));

        let mut candle_krea_shape = adopted_contract();
        candle_krea_shape.formula = MemoryFormulaKind::PhaseEnvelope {
            phases: candle_krea_shape.lifecycle.phases.clone(),
            variables: vec![
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::OverlayBytes,
            ],
        };
        assert!(candle_krea_shape.conformance_errors().is_empty());
    }

    #[test]
    fn optimized_evidence_requires_every_dimension_and_matching_fingerprint() {
        let contract = adopted_contract();
        let mut evidence = MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "test".to_owned(),
                backend: MemoryBackend::Mlx,
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
                load_shape: contract.load_shape,
                mode: MemoryMode::TextToImage,
                overlay: None,
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                strategy: MemoryStrategy::BoundedDecode,
                engaged_composition: contract.engaged_composition(MemoryStrategy::BoundedDecode),
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    ..Default::default()
                },
            },
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: "test-layout-v1".to_owned(),
            sceneworks_revision: "scene".to_owned(),
            inference_revision: "inference".to_owned(),
            harness_version: "v1".to_owned(),
            predicted_peak_bytes: 10,
            observed_peak_bytes: Some(9),
            parity: MemoryParityContract::Exact,
            parity_result: MemoryParityResult::Passed,
        };
        assert_eq!(evidence.optimized_eligibility(&contract), Ok(()));

        let canonical_composition = evidence.key.engaged_composition.clone();
        evidence.key.engaged_composition = vec![MemoryStrategy::Resident, MemoryStrategy::Resident];
        assert!(evidence
            .validation_errors()
            .iter()
            .any(|error| error.contains("canonical strategy set")));
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::Invalid)
        );
        evidence.key.engaged_composition = canonical_composition;

        evidence.key.load_shape = LoadShape::EagerMaterialization;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::FingerprintMismatch)
        );
        evidence.key.load_shape = contract.load_shape;

        let mut eager_contract = contract.clone();
        eager_contract.load_shape = LoadShape::EagerMaterialization;
        eager_contract.calibration.as_mut().unwrap().load_shape = LoadShape::EagerMaterialization;
        assert_eq!(
            evidence.optimized_eligibility(&eager_contract),
            Err(MemoryEvidenceVerdict::FingerprintMismatch)
        );

        evidence.calibration_abi = 1;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::FingerprintMismatch)
        );
        evidence.calibration_abi = MEMORY_CALIBRATION_ABI;

        let mut changed_composition = contract.clone();
        changed_composition.additional_prerequisites.push((
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        assert_eq!(
            evidence.optimized_eligibility(&changed_composition),
            Err(MemoryEvidenceVerdict::CompositionMismatch)
        );

        evidence.dimensions.current_environment_verification = MemoryEvidenceVerdict::Stale;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::Stale)
        );
        evidence.dimensions = MemoryEvidenceDimensions::VERIFIED;
        evidence.calibration_fingerprint = "old-layout".to_owned();
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::FingerprintMismatch)
        );
    }

    #[test]
    fn engaged_composition_tracks_a_non_default_prerequisite_edge() {
        let mut contract = adopted_contract();
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            vec![MemoryStrategy::Resident, MemoryStrategy::BoundedDecode,]
        );

        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedDecode,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedDecode),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
            ]
        );
    }

    #[test]
    fn effective_budget_reserves_headroom_from_free_and_reclaimable_memory() {
        let budget = MemoryBudget {
            total_bytes: 100,
            committed_bytes: 80,
            reclaimable_bytes: 50,
            reserved_headroom_bytes: 10,
        };
        assert_eq!(budget.effective_bytes(), 60);
        assert!(budget.fits(60));
        assert!(!budget.fits(61));

        assert_eq!(
            MemoryBudget {
                reclaimable_bytes: 0,
                ..budget
            }
            .effective_bytes(),
            10
        );
    }

    #[test]
    fn fits_consumes_incremental_demand_not_the_absolute_request_peak() {
        let budget = MemoryBudget {
            total_bytes: 100,
            committed_bytes: 40,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 10,
        };
        let absolute_request_peak = 80;
        let request_resident_already_committed = 30;
        let incremental_live_demand = absolute_request_peak - request_resident_already_committed;

        assert_eq!(budget.effective_bytes(), 50);
        assert!(budget.fits(incremental_live_demand));
        assert_eq!(budget.required_total_bytes(incremental_live_demand), 100);
        assert!(!budget.fits(absolute_request_peak));
        assert_eq!(budget.required_total_bytes(absolute_request_peak), 130);
        assert!(!budget.fits(incremental_live_demand + 1));
    }

    #[test]
    fn effective_budget_saturates_at_zero_and_total_minus_headroom() {
        assert_eq!(
            MemoryBudget {
                total_bytes: 100,
                committed_bytes: 110,
                reclaimable_bytes: 5,
                reserved_headroom_bytes: 10,
            }
            .effective_bytes(),
            0
        );
        assert_eq!(
            MemoryBudget {
                total_bytes: 100,
                committed_bytes: 0,
                reclaimable_bytes: u64::MAX,
                reserved_headroom_bytes: 10,
            }
            .effective_bytes(),
            90
        );
        assert_eq!(
            MemoryBudget {
                total_bytes: 100,
                committed_bytes: 0,
                reclaimable_bytes: u64::MAX,
                reserved_headroom_bytes: 101,
            }
            .effective_bytes(),
            0
        );
    }

    #[test]
    fn effective_budget_preserves_an_overcommit_deficit_before_reclamation() {
        let budget = MemoryBudget {
            total_bytes: 100,
            committed_bytes: 110,
            reclaimable_bytes: 50,
            reserved_headroom_bytes: 10,
        };
        assert_eq!(budget.effective_bytes(), 30);
        assert!(budget.fits(30));
        assert!(!budget.fits(31));

        assert_eq!(
            MemoryBudget {
                committed_bytes: 160,
                ..budget
            }
            .effective_bytes(),
            0,
            "an overcommit larger than all reclaimable memory must leave no request budget"
        );
    }

    #[test]
    fn evidence_axes_have_stable_protocol_keys_without_stringly_runtime_identity() {
        assert_eq!(MemoryBackend::Mlx.as_key(), "mlx");
        assert_eq!(MemoryBackend::Candle.as_key(), "candle");
        assert_eq!(MemoryMode::TextToImage.as_key(), "text_to_image");
        assert_eq!(MemoryMode::ImageToImage.as_key(), "image_to_image");
        assert_eq!(MemoryMode::Edit.as_key(), "edit");
        assert_eq!(MemoryMode::Other("custom".to_owned()).as_key(), "custom");
    }

    #[test]
    fn provider_contract_requires_hooks_for_implemented_rungs() {
        let mut contract = adopted_contract();
        contract.lifecycle.attention_chunking = false;
        assert!(contract
            .conformance_errors()
            .iter()
            .any(|error| error.contains("attention_chunking")));
    }

    #[test]
    fn provider_contract_rejects_structural_absence_when_an_implementation_hook_exists() {
        let mut contract = adopted_contract();
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedAttention)
            .expect("bounded attention capability")
            .support = MemoryStrategySupport::StructurallyNotApplicable {
            reason: "mutated contradiction".to_owned(),
        };
        assert!(contract.conformance_errors().iter().any(|error| {
            error.contains("BoundedAttention cannot be StructurallyNotApplicable")
                && error.contains("implementation hook")
        }));
    }

    /// **SC-16090.** A rung-4 realization that converts formats per window may ship, but not silently:
    /// it must name the story that removes it, and say what it converts. The arms are asserted against
    /// each other so the test cannot pass with the rule deleted (an unconditional error would fail the
    /// declared arm, an absent one the undeclared arms), and the unrelated-rung arm pins that the rule
    /// is scoped to rung 4.
    #[test]
    fn a_nonconforming_window_materialization_must_name_its_owner_story() {
        let converting = |converts: &str, owner_story: &str| MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::HostFormatConversion {
                converts: converts.to_owned(),
                owner_story: owner_story.to_owned(),
            },
        };
        const CONVERTS: &str = "MLX affine triple -> GGML Q4_1 per projection per window";
        let errors_matching = |contract: &MemoryProviderContract, needle: &str| {
            contract
                .conformance_errors()
                .iter()
                .any(|error| error.contains(needle))
        };

        let mut unnamed = adopted_contract();
        unnamed.backend = converting(CONVERTS, "");
        assert!(
            errors_matching(&unnamed, "owner_story"),
            "a per-window format conversion with no owner story must be a conformance error"
        );
        // Whitespace is not a name — otherwise the escape hatch is one space wide.
        let mut blank = adopted_contract();
        blank.backend = converting(CONVERTS, "   ");
        assert!(
            errors_matching(&blank, "owner_story"),
            "a blank owner_story must not satisfy it"
        );

        // `converts` is the pointer a reviewer follows to the code, so it is required too. Asserted
        // separately from `owner_story` so satisfying one cannot mask the other.
        let mut unexplained = adopted_contract();
        unexplained.backend = converting("", "sc-16096");
        assert!(
            errors_matching(&unexplained, "say what it converts"),
            "an unexplained conversion must be an error even with an owner story named"
        );

        let mut named = adopted_contract();
        named.backend = converting(CONVERTS, "sc-16096");
        assert!(
            named.conformance_errors().is_empty(),
            "a fully declared conversion is the escape hatch, and must leave the contract clean — \
             a contract-level error would cost the shared selector rungs 0-4, not just rung 4: {:?}",
            named.conformance_errors()
        );

        // Scoped to rung 4: the same realization with rung 4 unimplemented is silent, because nothing
        // is materializing windows.
        let mut without_rung_four = adopted_contract();
        without_rung_four.backend = converting("", "");
        without_rung_four.strategies[4].support = MemoryStrategySupport::Missing;
        without_rung_four.strategies[4].parameters = MemoryParameterRanges::default();
        assert!(
            without_rung_four.conformance_errors().is_empty(),
            "the rule must not fire for a provider that does not implement rung 4: {:?}",
            without_rung_four.conformance_errors()
        );
    }

    /// **SC-16090.** The obligation is asked backend-neutrally and answered per realization: one shared
    /// question, not a forked ladder. MLX's mapped materialization satisfies it structurally; an MLX
    /// realization that declares neither lazy nor mmap materialization is **unstated** — refused, but
    /// with the fix it can actually perform, since `MlxMetal` has no `owner_story` to name.
    #[test]
    fn window_materialization_is_asked_of_every_backend_and_not_granted_by_backend_identity() {
        assert!(mlx_backend()
            .window_materialization()
            .is_some_and(MemoryWindowMaterialization::is_conforming));

        let unstated = MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: false,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        };
        assert_eq!(
            unstated.window_materialization(),
            None,
            "being MLX is not itself evidence that a window is a mapped read — but nor is it a \
             declared host conversion, which would assert something unmeasured"
        );

        // Unstated is refused, and the message must name the fix that exists (declare the mechanism),
        // not the one MlxMetal cannot perform (name an owner_story it has no field for).
        let mut under_declared = adopted_contract();
        under_declared.backend = unstated.clone();
        let errors = under_declared.conformance_errors();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("lazy_or_mmap_materialization")),
            "the refusal must name the achievable fix: {errors:?}"
        );
        assert!(
            !errors.iter().any(|error| error.contains("owner_story")),
            "must not demand an owner_story from a realization with no such field: {errors:?}"
        );

        let candle_conforming = MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        };
        assert!(
            candle_conforming
                .window_materialization()
                .is_some_and(MemoryWindowMaterialization::is_conforming),
            "a host-to-device copy of device-format bytes conforms — the rung is not MLX-only"
        );
        assert_eq!(candle_conforming.backend_id(), "candle");

        // The contract-level accessor is the seam a caller uses (mirroring how `engages` is layered),
        // so it must agree with the realization rather than being a second source of truth.
        let mut contract = adopted_contract();
        contract.backend = candle_conforming.clone();
        assert_eq!(
            contract.window_materialization(),
            candle_conforming.window_materialization()
        );
        assert!(
            contract.conformance_errors().is_empty(),
            "a conforming candle realization needs no escape hatch: {:?}",
            contract.conformance_errors()
        );
    }

    #[test]
    fn selection_preserves_tier_and_rejects_unknown_parameter_values() {
        let contract = adopted_contract();
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let mut selection = MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            },
            tier,
        };
        contract.validate_selection(&selection).unwrap();
        assert_eq!(selection.tier, tier);

        selection.parameters.decode_tile_edge = Some(320);
        assert!(matches!(
            contract.validate_selection(&selection),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn cumulative_parameters_are_required_and_irrelevant_parameters_are_rejected() {
        let contract = adopted_contract();
        let tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        };
        let mut selection = MemorySelection {
            strategy: MemoryStrategy::BoundedAttention,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(128),
                transformer_window_size: None,
                transformer_window_component: None,
            },
            tier,
        };
        contract.validate_selection(&selection).unwrap();

        selection.parameters.decode_overlap = None;
        assert!(contract
            .validate_selection(&selection)
            .unwrap_err()
            .to_string()
            .contains("decode_overlap is required"));

        selection.strategy = MemoryStrategy::Resident;
        selection.parameters = MemoryStrategyParameters {
            attention_chunk_size: Some(128),
            ..Default::default()
        };
        assert!(contract
            .validate_selection(&selection)
            .unwrap_err()
            .to_string()
            .contains("irrelevant"));
    }

    #[test]
    fn optimized_evidence_requires_valid_passing_parity_and_an_observed_peak() {
        let contract = adopted_contract();
        let mut evidence = MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "test".to_owned(),
                backend: MemoryBackend::Mlx,
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                load_shape: contract.load_shape,
                mode: MemoryMode::TextToImage,
                overlay: None,
                geometry: MemoryGeometry {
                    width: 512,
                    height: 512,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                strategy: MemoryStrategy::BoundedDecode,
                engaged_composition: contract.engaged_composition(MemoryStrategy::BoundedDecode),
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    ..Default::default()
                },
            },
            conformance: MemoryConformanceState::Verified,
            dimensions: MemoryEvidenceDimensions::VERIFIED,
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: "test-layout-v1".to_owned(),
            sceneworks_revision: "scene".to_owned(),
            inference_revision: "inference".to_owned(),
            harness_version: "v1".to_owned(),
            predicted_peak_bytes: 10,
            observed_peak_bytes: Some(9),
            parity: MemoryParityContract::Tolerance {
                metric: "max_abs".to_owned(),
                maximum_error: 0.001,
            },
            parity_result: MemoryParityResult::Passed,
        };
        assert_eq!(evidence.optimized_eligibility(&contract), Ok(()));

        evidence.parity = MemoryParityContract::Tolerance {
            metric: String::new(),
            maximum_error: f64::NAN,
        };
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::Invalid)
        );
        evidence.parity = MemoryParityContract::Golden {
            fixture: String::new(),
            metric: "max_abs".to_owned(),
            maximum_error: -1.0,
        };
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::Invalid)
        );
        evidence.parity = MemoryParityContract::Exact;
        evidence.observed_peak_bytes = None;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::Invalid)
        );
        evidence.observed_peak_bytes = Some(9);
        evidence.parity_result = MemoryParityResult::NotRun;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(MemoryEvidenceVerdict::Unverified)
        );
    }
    /// SC-15794: the component scope's accessors and its DiT-only default.
    ///
    /// The default is the whole backward-compatibility story — every provider and every selection
    /// written before this type existed must keep meaning "stream the DiT", so `None` resolving to
    /// anything else would silently re-scope work already in production.
    #[test]
    fn the_transformer_component_defaults_to_dit_and_reports_its_membership() {
        assert_eq!(TransformerComponent::default(), TransformerComponent::Dit);
        assert_eq!(
            MemoryStrategyParameters::default().window_component(),
            TransformerComponent::Dit
        );
        let explicit = MemoryStrategyParameters {
            transformer_window_component: Some(TransformerComponent::TextEncoder),
            ..Default::default()
        };
        assert_eq!(
            explicit.window_component(),
            TransformerComponent::TextEncoder
        );

        // Membership, asserted per variant rather than via a loop so a swapped pair cannot pass.
        assert!(TransformerComponent::Dit.includes_dit());
        assert!(!TransformerComponent::Dit.includes_text_encoder());
        assert!(!TransformerComponent::TextEncoder.includes_dit());
        assert!(TransformerComponent::TextEncoder.includes_text_encoder());
        assert!(TransformerComponent::Both.includes_dit());
        assert!(TransformerComponent::Both.includes_text_encoder());
    }

    /// The selected scope is validated against what the provider actually declared — the check that
    /// stops a selector asking for an encoder stream from a provider that only implements the DiT.
    #[test]
    fn a_selected_component_must_be_one_the_provider_declared() {
        let contract = |components: Vec<TransformerComponent>| {
            let mut c = adopted_contract();
            let rung4 = c
                .strategies
                .iter_mut()
                .find(|s| s.strategy == MemoryStrategy::BoundedTransformerResidency)
                .expect("rung 4 capability");
            rung4.parameters.transformer_window_sizes = vec![1];
            rung4.parameters.transformer_window_components = components;
            c
        };
        let select = |component| MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                // Rung 4 is cumulative, so the lower rungs' parameters are required too; these are
                // `adopted_contract`'s own declared candidates.
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: Some(component),
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        };

        let dit_only = contract(vec![TransformerComponent::Dit]);
        assert!(
            validate_selected_parameters(&select(TransformerComponent::Dit), &dit_only).is_ok()
        );
        let err =
            validate_selected_parameters(&select(TransformerComponent::TextEncoder), &dit_only)
                .expect_err(
                    "a provider that declares only Dit must reject a TextEncoder selection",
                );
        assert!(err.contains("TextEncoder"), "{err}");

        // An empty declaration means DiT-only, so DiT is still selectable and nothing else is.
        let empty = contract(Vec::new());
        assert!(validate_selected_parameters(&select(TransformerComponent::Dit), &empty).is_ok());
        assert!(validate_selected_parameters(&select(TransformerComponent::Both), &empty).is_err());

        // A provider that declares the encoder scope accepts it.
        let both = contract(vec![TransformerComponent::Dit, TransformerComponent::Both]);
        assert!(validate_selected_parameters(&select(TransformerComponent::Both), &both).is_ok());
    }

    /// A provider may only own component candidates on the rung that owns the window they scope.
    #[test]
    fn only_rung_four_may_own_component_candidates() {
        let with_components = |strategy| MemoryStrategyCapability {
            strategy,
            support: MemoryStrategySupport::Implemented,
            parameters: MemoryParameterRanges {
                transformer_window_components: vec![TransformerComponent::Both],
                ..Default::default()
            },
        };
        let mut errors = Vec::new();
        validate_owned_parameter_domain(
            &with_components(MemoryStrategy::BoundedDecode),
            &mut errors,
        );
        assert!(
            errors.iter().any(|e| e.contains("component candidates")),
            "a non-rung-4 strategy owning component candidates must be rejected, got {errors:?}"
        );

        // Duplicates are a declaration bug the selector would silently tolerate.
        let mut errors = Vec::new();
        validate_owned_parameter_domain(
            &MemoryStrategyCapability {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                support: MemoryStrategySupport::Implemented,
                parameters: MemoryParameterRanges {
                    transformer_window_sizes: vec![1],
                    transformer_window_components: vec![
                        TransformerComponent::Dit,
                        TransformerComponent::Dit,
                    ],
                    ..Default::default()
                },
            },
            &mut errors,
        );
        assert!(
            errors.iter().any(|e| e.contains("duplicate")),
            "duplicate component candidates must be rejected, got {errors:?}"
        );

        // An EMPTY declaration is legal and means DiT-only — that is what keeps every pre-SC-15794
        // provider valid rather than retroactively incomplete.
        let mut errors = Vec::new();
        validate_owned_parameter_domain(
            &MemoryStrategyCapability {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                support: MemoryStrategySupport::Implemented,
                parameters: MemoryParameterRanges {
                    transformer_window_sizes: vec![1],
                    ..Default::default()
                },
            },
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "an empty component declaration must be legal, got {errors:?}"
        );
    }

    /// Set one rung's support, returning the mutated contract.
    fn with_support(
        mut contract: MemoryProviderContract,
        strategy: MemoryStrategy,
        support: MemoryStrategySupport,
    ) -> MemoryProviderContract {
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == strategy)
            .expect("declared rung")
            .support = support;
        contract
    }

    fn bf16() -> MemoryNumericTier {
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        }
    }

    /// SC-15998: the graph itself, asserted edge by edge so a regression to the old rung-1 edge
    /// cannot pass.
    #[test]
    fn rung_four_requires_deferred_materialization_not_rung_one() {
        assert_eq!(MemoryStrategy::Resident.requires(), &[]);
        assert_eq!(MemoryStrategy::StagedResidency.requires(), &[]);
        assert_eq!(MemoryStrategy::BoundedDecode.requires(), &[]);
        assert_eq!(MemoryStrategy::BoundedAttention.requires(), &[]);
        assert_eq!(
            MemoryStrategy::BoundedTransformerResidency.requires(),
            &[MemoryStrategyPrerequisite::LoadShape(
                LoadShape::DeferredMaterialization
            )]
        );
        // The per-rung accessor is the same declaration, so a provider cannot diverge from it.
        for capability in &adopted_contract().strategies {
            assert_eq!(capability.requires(), capability.strategy.requires());
        }
    }

    /// SC-15805 — the case that is impossible today: rung 1 `Missing` must not make rungs 2 and 3
    /// unselectable. Bounding decoder or attention scratch has nothing to do with whether the
    /// conditioning component was shed.
    #[test]
    fn a_missing_rung_one_leaves_rungs_two_and_three_selectable() {
        let contract = with_support(
            adopted_contract(),
            MemoryStrategy::StagedResidency,
            MemoryStrategySupport::Missing,
        );
        assert!(contract.conformance_errors().is_empty());

        contract
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedDecode,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    ..Default::default()
                },
                tier: bf16(),
            })
            .expect("rung 2 depends on nothing and must stay selectable under a Missing rung 1");

        contract
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedAttention,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(256),
                    ..Default::default()
                },
                tier: bf16(),
            })
            .expect("rung 3 depends on nothing and must stay selectable under a Missing rung 1");
    }

    /// SC-15998 — rung 4 is refused for an eager load, and accepted for deferred materialization
    /// even when rung 1 is unavailable.
    #[test]
    fn rung_four_is_gated_by_load_shape_not_staged_residency() {
        let rung_four = |contract: &MemoryProviderContract| {
            contract.validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(256),
                    transformer_window_size: Some(1),
                    transformer_window_component: None,
                },
                tier: bf16(),
            })
        };

        let deferred_without_rung_one = with_support(
            adopted_contract(),
            MemoryStrategy::StagedResidency,
            MemoryStrategySupport::Missing,
        );
        rung_four(&deferred_without_rung_one)
            .expect("Resident+Deferred rung 4 must not require staged residency");

        let mut eager = adopted_contract();
        eager.load_shape = LoadShape::EagerMaterialization;
        let error = rung_four(&eager)
            .expect_err("a block window over an eagerly materialized trunk must be refused")
            .to_string();
        assert!(error.contains("DeferredMaterialization"), "{error}");
        assert!(error.contains("EagerMaterialization"), "{error}");

        // Component scope does not change the physical prerequisite.
        let mut encoder_scoped = eager;
        encoder_scoped
            .strategies
            .iter_mut()
            .find(|c| c.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("rung 4")
            .parameters
            .transformer_window_components = vec![TransformerComponent::TextEncoder];
        let error = encoder_scoped
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(256),
                    transformer_window_size: Some(1),
                    transformer_window_component: Some(TransformerComponent::TextEncoder),
                },
                tier: bf16(),
            })
            .expect_err("a TextEncoder-scoped window does not escape the load-shape prerequisite")
            .to_string();
        assert!(error.contains("DeferredMaterialization"), "{error}");
    }

    /// A backend can append its real implementation coupling without changing the shared graph.
    #[test]
    fn an_additive_backend_prerequisite_can_require_staged_residency() {
        let mut contract = adopted_contract();
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        contract
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(256),
                    transformer_window_size: Some(1),
                    transformer_window_component: None,
                },
                tier: bf16(),
            })
            .expect("the additive rung prerequisite engages staged residency");

        let missing = with_support(
            contract,
            MemoryStrategy::StagedResidency,
            MemoryStrategySupport::Missing,
        );
        let error = missing
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(256),
                    transformer_window_size: Some(1),
                    transformer_window_component: None,
                },
                tier: bf16(),
            })
            .expect_err("the backend-specific rung edge must be enforced")
            .to_string();
        assert!(error.contains("StagedResidency"), "{error}");
        assert!(error.contains("ENGAGED IN THE SAME REQUEST"), "{error}");
    }

    /// SC-15998: rung 1 is neither part of the cost-order default nor rung 4's shared prerequisite.
    ///
    /// Rungs 2 and 3 bound scratch and are free, so a cumulative default costs the caller nothing.
    /// Rung 1 bounds residency and *"may evict the warm cross-request cache"* — a real cost paid by
    /// the next request. Dragging it into a rung-2 selection that never needed it is a straight loss.
    ///
    #[test]
    fn rung_four_keeps_phase_residency_warm_while_cost_order_still_engages_scratch_rungs() {
        // 1. The behavior chosen here: a free scratch rung does not drag in the costly residency one.
        assert!(
            !MemoryStrategy::BoundedDecode.engages(MemoryStrategy::StagedResidency),
            "rung 2 bounds scratch and must not evict the warm cache by engaging rung 1"
        );
        assert!(
            !MemoryStrategy::BoundedAttention.engages(MemoryStrategy::StagedResidency),
            "rung 3 bounds scratch and must not evict the warm cache by engaging rung 1"
        );
        assert!(!MemoryStrategy::Resident.engages(MemoryStrategy::StagedResidency));

        // THE REGRESSION GUARD: rung 4 alone does not evict the warm cross-request cache.
        assert!(
            !MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::StagedResidency)
        );
        assert!(MemoryStrategy::StagedResidency.engages(MemoryStrategy::StagedResidency));
        // End to end: the independent deferred load shape keeps rung 4 selectable.
        adopted_contract()
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    attention_chunk_size: Some(256),
                    transformer_window_size: Some(1),
                    transformer_window_component: None,
                },
                tier: bf16(),
            })
            .expect("Deferred materialization satisfies rung 4 without engaging rung 1");

        // The AC's stated cost-order default is untouched: rung 4 still engages 2 and 3.
        assert!(MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::BoundedDecode));
        assert!(
            MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::BoundedAttention)
        );
        assert!(MemoryStrategy::BoundedAttention.engages(MemoryStrategy::BoundedDecode));
        assert!(!MemoryStrategy::BoundedDecode.engages(MemoryStrategy::BoundedAttention));
        // Rung 0 stays engaged by everything — baseline, not a lever.
        for strategy in MemoryStrategy::ALL {
            assert!(strategy.engages(MemoryStrategy::Resident), "{strategy:?}");
        }
    }

    /// The cost-order default is engagement, is defeasible, and is NOT the prerequisite graph.
    ///
    /// A provider that publishes a cheaper verified composition — rung 4 without rung 2 — must not
    /// have to fight the validator: the unimplemented rung is simply not engaged, so its parameters
    /// become irrelevant rather than required.
    #[test]
    fn the_cost_order_default_is_defeasible_selector_policy_not_a_dependency() {
        // Pure policy: a deeper selection engages every cheaper SCRATCH rung, and never the reverse.
        // (Rung 1 is deliberately outside this default — see
        // `rung_one_is_not_dragged_in_by_cost_order_but_the_rung_four_edge_still_engages_it`.)
        assert!(MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::BoundedDecode));
        assert!(MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::Resident));
        assert!(MemoryStrategy::BoundedDecode.engages(MemoryStrategy::BoundedDecode));
        assert!(!MemoryStrategy::BoundedDecode.engages(MemoryStrategy::BoundedAttention));

        // …and the contract intersects it with what the provider implements.
        let mut cheaper = with_support(
            adopted_contract(),
            MemoryStrategy::BoundedDecode,
            MemoryStrategySupport::Missing,
        );
        cheaper
            .strategies
            .iter_mut()
            .find(|c| c.strategy == MemoryStrategy::BoundedDecode)
            .expect("rung 2")
            .parameters = MemoryParameterRanges::default();
        cheaper.lifecycle.decode_tiling = false;
        assert!(cheaper.conformance_errors().is_empty());
        assert!(!cheaper.engages(
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategy::BoundedDecode
        ));

        // Rung 4 without rung 2 — accepted, and the decode parameters are now the irrelevant ones.
        let mut selection = MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: None,
                decode_overlap: None,
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: None,
            },
            tier: bf16(),
        };
        cheaper
            .validate_selection(&selection)
            .expect("a verified cheaper composition must not fight the validator");
        selection.parameters.decode_tile_edge = Some(512);
        assert!(cheaper
            .validate_selection(&selection)
            .unwrap_err()
            .to_string()
            .contains("irrelevant"));
    }

    #[test]
    fn a_verified_exclusion_defeats_only_the_default_engagement_edge() {
        let mut contract = adopted_contract();
        contract
            .default_engagement_exclusions
            .push(MemoryStrategyEngagementExclusion {
                selection: MemoryStrategy::BoundedAttention,
                excluded_rung: MemoryStrategy::BoundedDecode,
                evidence: "direct calibration: attention chunking is independent of decode tiling"
                    .to_owned(),
            });
        assert!(contract.conformance_errors().is_empty());
        assert!(!contract.engages(
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedDecode
        ));
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedAttention),
            vec![MemoryStrategy::Resident, MemoryStrategy::BoundedAttention]
        );
        contract
            .validate_selection(&MemorySelection {
                strategy: MemoryStrategy::BoundedAttention,
                parameters: MemoryStrategyParameters {
                    attention_chunk_size: Some(256),
                    ..Default::default()
                },
                tier: bf16(),
            })
            .expect("the documented cheaper composition must validate without decode parameters");

        contract.default_engagement_exclusions[0].evidence.clear();
        assert!(contract
            .conformance_errors()
            .iter()
            .any(|error| error.contains("has no verification evidence")));
    }

    fn evidence_log_record() -> MemoryEvidenceLogRecord {
        let calibration =
            MemoryCalibrationIdentity::new("test-layout-v1", LoadShape::EagerMaterialization);
        MemoryEvidenceLogRecord {
            key: MemoryEvidenceKey {
                resolved_route: "test_provider".to_owned(),
                backend: MemoryBackend::Mlx,
                tier: bf16(),
                load_shape: LoadShape::EagerMaterialization,
                mode: MemoryMode::TextToImage,
                overlay: None,
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                },
                strategy: MemoryStrategy::StagedResidency,
                engaged_composition: vec![
                    MemoryStrategy::Resident,
                    MemoryStrategy::StagedResidency,
                ],
                parameters: MemoryStrategyParameters::default(),
            },
            declared_calibration: calibration.clone(),
            observed_calibration: calibration,
            predicted_peak_bytes: 8_000_000_000,
            observed_peak_bytes: 7_500_000_000,
            inference_revision: "a".repeat(40),
            sceneworks_revision: "b".repeat(40),
            model_revision: "d".repeat(40),
            model_inventory_sha256: "e".repeat(64),
            harness_version: "test-harness-v1".to_owned(),
            output_sha256: "c".repeat(64),
            parity: MemoryParityContract::Exact,
            parity_result: MemoryParityResult::Passed,
        }
    }

    #[test]
    fn memory_evidence_v1_line_contains_the_complete_typed_key_and_provenance() {
        let line = evidence_log_record().to_json_line().unwrap();
        assert!(line.starts_with(MEMORY_EVIDENCE_V1_PREFIX));
        let value: serde_json::Value =
            serde_json::from_str(&line[MEMORY_EVIDENCE_V1_PREFIX.len()..]).unwrap();
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["key"]["backend"], "mlx");
        assert_eq!(value["key"]["mode"], "text_to_image");
        assert_eq!(value["key"]["load_shape"], "eager_materialization");
        assert_eq!(value["key"]["geometry"]["reference_count"], 0);
        assert_eq!(value["key"]["strategy"], "staged_residency");
        assert_eq!(
            value["key"]["engaged_composition"],
            serde_json::json!(["resident", "staged_residency"])
        );
        assert_eq!(value["declared_calibration"], value["observed_calibration"]);
        assert_eq!(value["predicted_peak_bytes"], 8_000_000_000_u64);
        assert_eq!(value["observed_peak_bytes"], 7_500_000_000_u64);
        assert_eq!(value["model_revision"], "d".repeat(40));
        assert_eq!(value["model_inventory_sha256"], "e".repeat(64));
        assert_eq!(value["output_sha256"], "c".repeat(64));
        assert_eq!(value["parity_result"]["kind"], "passed");
    }

    #[test]
    fn evidence_writer_rejects_identity_revision_and_fingerprint_drift() {
        let mut record = evidence_log_record();
        record.key.engaged_composition = vec![MemoryStrategy::Resident];
        assert!(record
            .to_json_line()
            .unwrap_err()
            .to_string()
            .contains("end with the measured strategy"));

        let mut record = evidence_log_record();
        record.observed_calibration.fingerprint = "other-layout-v1".to_owned();
        assert!(record
            .to_json_line()
            .unwrap_err()
            .to_string()
            .contains("identities do not match"));

        let mut record = evidence_log_record();
        record.inference_revision = "main".to_owned();
        assert!(record
            .to_json_line()
            .unwrap_err()
            .to_string()
            .contains("exact 40-character"));

        let mut record = evidence_log_record();
        record.model_inventory_sha256 = "mutable-cache".to_owned();
        assert!(record
            .to_json_line()
            .unwrap_err()
            .to_string()
            .contains("model inventory SHA-256"));

        let mut record = evidence_log_record();
        record.output_sha256 = "not-a-sha".to_owned();
        assert!(record
            .to_json_line()
            .unwrap_err()
            .to_string()
            .contains("output SHA-256"));

        for invalid in [
            "",
            "Layout-v1",
            "layout_v1",
            "layout",
            "layout-v0",
            "layout-v01",
            "layout-v1-v2",
        ] {
            assert!(
                validate_calibration_fingerprint(invalid).is_err(),
                "{invalid:?}"
            );
        }
        for valid in ["layout-v1", "sc-16594-layout-v2", "layout-q4-512-v3"] {
            validate_calibration_fingerprint(valid).unwrap();
        }
    }
}
