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
//! first. Nothing in the four-rung ladder is image-specific — rung 1 sheds a conditioning
//! component, rung 2 bounds decoder scratch, rung 3 bounds attention, rung 4 bounds transformer
//! residency, and video and audio have all four — so the vocabulary is lane-neutral here.
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
//!    the cheapest sufficient rung, the same shape [`crate::mempolicy`] uses for its lever ladder.
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
//! A [`MemoryWindowMaterialization::HostFormatConversion`] realization — which ships today — does have
//! a host cost, and this decision does **not** surface its size. Its per-window conversion allocates
//! anonymous host memory per projection and frees it again: on the order of the block's unpacked codes
//! for q4, and for q8 a **full dense f32 grid** (`repack::dequant_mlx_q8_gs`), which for a
//! 3840 × 15360 projection is a few hundred MiB. Those are transients, not held for the request, and
//! they are **not measured** — SC-15791 states plainly that q8 host memory is unverified. So on a
//! low-RAM host a rung-4 Candle selection today carries an unquantified host transient that no gate
//! sees. That is a consequence of the non-conforming realization, and SC-16096 both removes the
//! conversion and owes the host measurement; it is recorded here so it is not mistaken for zero.
//!
//! **Revisit trigger — deliberately broad enough to catch that case.** The axis is owed if a
//! realization is measured to need host memory proportional to the model that a fit gate would have to
//! account for: whether it is held for the request or transient, reclaimable or not, and whether or not
//! a device-format alternative exists. SC-16096 must raise it rather than absorb it. What is *not* owed
//! is an axis built ahead of that measurement, on the strength of a figure derived for a fix nobody has
//! written.

use crate::{Error, GenerationRequest, Precision, Quant, Result};

/// Current ABI of the provider/evidence calibration handshake.
///
/// Providers also supply a content fingerprint. The ABI changes when this contract changes how
/// formula inputs or lifecycle semantics are interpreted; the fingerprint changes whenever one
/// provider changes tensor layout, quantization floors, execution structure, or another detail that
/// invalidates its measurements.
pub const MEMORY_CALIBRATION_ABI: u32 = 1;

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

    /// The declared prerequisite graph for this rung (SC-15805).
    ///
    /// ```text
    /// rung 1  ->  (none)
    /// rung 2  ->  (none)
    /// rung 3  ->  (none)
    /// rung 4  ->  requires rung 1 ENGAGED in the same request
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
    /// The true physical precondition is not rung 1 *the strategy* — it is a **load shape that
    /// permits deferred materialization** of the trunk. Rung 1 is the mechanism that produces that
    /// shape, which is why the edge names it. The two facts are currently conflated inside one
    /// provider's private condition: `mlx-gen-z-image` computes
    /// `streamable = Dir(_) && OffloadPolicy::Sequential` and declares rung 4 `Missing` when it is
    /// false. **SC-15806** converges them: once rung 1 is request-scoped, `streamable` is threaded
    /// through the loader per request, after which "rung 1 engaged in this request" and "this
    /// request has a deferred-materialization load" are the same statement. Until then the edge is
    /// the contract-level expression of a fact a provider also enforces privately — the two
    /// **coexist** and answer different questions:
    ///
    /// - *availability* (z-image's load-spec-conditional declaration, SC-15754): can this **load**
    ///   execute the rung at all? Decided once, when the contract is built from the `LoadSpec`.
    /// - *engagement* (this graph): is the prerequisite rung active in **this request**? Decided
    ///   per selection, by [`MemoryProviderContract::validate_selection`].
    ///
    /// # The edge is unconditional
    ///
    /// It is deliberately **not** conditioned on
    /// [`MemoryStrategyParameters::transformer_window_component`] (SC-15794). For the `Dit` and
    /// `Both` scopes the arithmetic above applies directly, and `Dit` is both the published
    /// production default and the only scope a Krea request may name. A `TextEncoder`-scoped window
    /// bounds the encoder *during conditioning*, the phase rung 1 has not yet shed, so in theory a
    /// TE-dominant model whose **conditioning sets the request peak** would want TE-scoped rung 4
    /// with rung 1 not engaged. No such model is measured: z-image is the only measurement and it is
    /// decisive the other way — TE scope cut conditioning 46.5% and moved the request peak 0.0%.
    /// Encoding that exception now would build selector surface for exactly the case the epic's
    /// non-negotiable tells the selector to ignore ("a strategy that bounds a phase but does not
    /// move the REQUEST peak is not a saving").
    ///
    /// **Revisit trigger: SC-15800.** If it measures a TE-dominant family where conditioning sets
    /// the request peak, this edge becomes scope-conditional *on that evidence*. Until then it does
    /// not.
    ///
    /// # Why the graph is contract-owned rather than provider-declared
    ///
    /// The whole point of the story is to move this fact "from a provider's private condition into
    /// the contract". A per-provider `requires` field would let a provider silently drop the edge by
    /// leaving it unset — the false-green shape this contract exists to prevent — and no provider
    /// may opt out of arithmetic. Providers still control the graph's *effect* through their support
    /// declarations: declaring rung 1 `Missing` makes rung 4 unselectable, which is exactly the
    /// honest outcome.
    ///
    /// # Seam asymmetry with `engages` (recorded, not a defect)
    ///
    /// [`MemoryStrategy::engages`] exists in two forms — the pure cost-order policy here and the
    /// provider-intersected [`MemoryProviderContract::engages`] — so a provider can differentiate
    /// engagement. `requires` deliberately has only the pure form: it is a `const fn` on the bare
    /// enum taking `self` by value, with no contract in scope, so **there is no seam at which a
    /// backend-differentiated prerequisite edge could be declared.** That is the intended shape
    /// today (see the section above: the edge is arithmetic, and no provider may opt out of it), and
    /// nothing in the current graph wants a per-backend edge.
    ///
    /// Should a genuinely Candle-specific edge surface — one whose *arithmetic* differs by backend
    /// rather than one a provider merely wishes away — the additive fix is a
    /// `MemoryProviderContract::requires(strategy)` that defaults to this enum graph and is
    /// overridden only where a backend realization changes the arithmetic. That keeps the
    /// no-silent-opt-out property (the default is still contract-owned) while giving the edge set a
    /// place to vary. It is deliberately not built ahead of that evidence.
    ///
    /// **SC-15791 has since reported, and it is not owed.** The Candle spike confirmed rung 4's single
    /// edge holds there for the same arithmetic reason, needing no backend differentiation. What the
    /// spike did surface was on the *cost* axis, and it is answered without differentiating anything:
    /// see the module docs on why the cost order stays shared, and
    /// [`MemoryWindowMaterialization`] for the per-realization obligation that keeps it true (SC-16090).
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
    /// **The rung-4 edge still resolves.** Rung 4 engages rung 1 because
    /// `BoundedTransformerResidency::requires()` names it, not because `1 <= 4` — so
    /// [`MemoryProviderContract::validate_selection`]'s prerequisite walk is satisfied exactly as
    /// before and rung 4 stays selectable. Deriving this from `requires()` rather than hardcoding
    /// "rung 4" is the point: a future edge added to the graph implies its own engagement, with no
    /// second place to update.
    ///
    /// [`Self::Resident`] remains engaged by every selection — it is the baseline, not an
    /// optimization, and nothing reads it as a lever.
    pub const fn engages(self, rung: Self) -> bool {
        match rung {
            // The baseline. Unchanged: `(0 <= anything)` was always true.
            Self::Resident => true,
            // Residency-bounding, and not free — declared prerequisite or explicit selection only.
            Self::StagedResidency => {
                matches!(self, Self::StagedResidency) || self.requires_rung(Self::StagedResidency)
            }
            // Scratch-bounding and free: the plain cost-order default.
            Self::BoundedDecode | Self::BoundedAttention | Self::BoundedTransformerResidency => {
                (rung as u8) <= (self as u8)
            }
        }
    }

    /// Whether `self`'s declared prerequisite graph names `rung`.
    ///
    /// Kept as a `const fn` slice walk (rather than an iterator) so [`Self::engages`] stays `const`.
    const fn requires_rung(self, rung: Self) -> bool {
        let edges = self.requires();
        let mut index = 0;
        while index < edges.len() {
            if edges[index].rung as u8 == rung as u8 {
                return true;
            }
            index += 1;
        }
        false
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

/// One edge of the declared prerequisite graph (SC-15805). See [`MemoryStrategy::requires`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryStrategyPrerequisite {
    /// The rung this edge points at.
    pub rung: MemoryStrategy,
    /// What is demanded of it.
    pub scope: MemoryPrerequisiteScope,
}

const NO_PREREQUISITES: &[MemoryStrategyPrerequisite] = &[];

const BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES: &[MemoryStrategyPrerequisite] =
    &[MemoryStrategyPrerequisite {
        rung: MemoryStrategy::StagedResidency,
        scope: MemoryPrerequisiteScope::EngagedInSameRequest,
    }];

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
    /// This rung's declared prerequisites (SC-15805). Contract-owned, not provider-settable — see
    /// [`MemoryStrategy::requires`] for why.
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
    pub attention_chunk_sizes: Vec<u32>,
    pub transformer_window_sizes: Vec<u32>,
    /// Component scopes the provider actually implements for rung 4 (SC-15794). Empty means the
    /// provider streams only the DiT, which is how every pre-SC-15794 provider behaves.
    pub transformer_window_components: Vec<TransformerComponent>,
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

impl MemoryBackendRealization {
    pub const fn backend_id(&self) -> &'static str {
        match self {
            Self::CandleCuda { .. } => "candle",
            Self::MlxMetal { .. } => "mlx",
        }
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
}

impl MemoryCalibrationIdentity {
    pub fn new(fingerprint: impl Into<String>) -> Self {
        Self {
            abi: MEMORY_CALIBRATION_ABI,
            fingerprint: fingerprint.into(),
        }
    }
}

/// Provider-owned, load-exact asset facts used as formula inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryAssetFacts {
    pub base_bytes: u64,
    pub conditioning_bytes: u64,
    pub transformer_bytes: u64,
    pub decoder_bytes: u64,
    pub overlay_bytes: u64,
}

/// Cache keys must include every axis that can change residency or execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCacheSemantics {
    StrategyTierParametersGeometryAndOverlay,
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
            cache: MemoryCacheSemantics::StrategyTierParametersGeometryAndOverlay,
            warm_run: MemoryWarmRunSemantics::RevalidateBudgetAndReapplyRequestState,
            cancellation: MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
            error: MemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
        }
    }
}

/// Static provider contract returned before weights are loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryProviderContract {
    pub provider_id: String,
    pub backend: MemoryBackendRealization,
    pub strategies: Vec<MemoryStrategyCapability>,
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
        Self {
            provider_id: provider_id.into(),
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
            lifecycle: MemoryLifecycleCapabilities::default(),
            formula: MemoryFormulaKind::AssetBytesPlusHeadroom,
            calibration: None,
            asset_facts: MemoryAssetFacts::default(),
            runtime: MemoryRuntimeSemantics::default(),
        }
    }

    pub fn capability(&self, strategy: MemoryStrategy) -> Option<&MemoryStrategyCapability> {
        self.strategies
            .iter()
            .find(|capability| capability.strategy == strategy)
    }

    fn support(&self, strategy: MemoryStrategy) -> Option<&MemoryStrategySupport> {
        self.capability(strategy)
            .map(|capability| &capability.support)
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
        strategy.engages(rung)
            && matches!(self.support(rung), Some(MemoryStrategySupport::Implemented))
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
            }
            validate_ranges(capability, &mut errors);
            validate_owned_parameter_domain(capability, &mut errors);
        }

        if let Some(calibration) = &self.calibration {
            if calibration.abi != MEMORY_CALIBRATION_ABI {
                errors.push(format!(
                    "calibration ABI {} does not match contract ABI {}",
                    calibration.abi, MEMORY_CALIBRATION_ABI
                ));
            }
            if calibration.fingerprint.trim().is_empty() {
                errors.push("calibration fingerprint must be non-empty".to_owned());
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
        for prerequisite in selection.strategy.requires() {
            match prerequisite.scope {
                MemoryPrerequisiteScope::EngagedInSameRequest => {
                    if self.engages(selection.strategy, prerequisite.rung) {
                        continue;
                    }
                    // `StructurallyNotApplicable` satisfies the edge vacuously: it asserts the
                    // architecture has no such component to shed, which is not evidence that the
                    // trunk is eagerly materialized. Refusing here would invent an over-refusal for
                    // an architecture that legitimately streams a trunk it never staged.
                    if matches!(
                        self.support(prerequisite.rung),
                        Some(MemoryStrategySupport::StructurallyNotApplicable { .. })
                    ) {
                        continue;
                    }
                    // The caveat must match the case that actually produced the refusal, or the
                    // message tells the reader that the one fix available to them will not work.
                    //
                    // Reachability (SC-15805 review): `engages` is
                    // `rung <= strategy && support(rung) == Implemented`. EVERY edge in the current
                    // graph points DOWN the cost ladder (rung 4 -> rung 1), so the ordering half is
                    // always true here and `Some(Implemented)` short-circuits at the `continue`
                    // above — that arm is **dead for the current graph**. It is retained rather than
                    // unreachable-panicked because it is the correct answer the moment an edge
                    // points sideways or up the ladder, and getting there is a one-line change to
                    // `requires`. Only the `Missing` / undeclared arms are reachable today, and for
                    // those "implement it" IS the fix — so that arm must not deny it.
                    let (why, caveat) = match self.support(prerequisite.rung) {
                        Some(MemoryStrategySupport::Implemented) => (
                            "the selected composition does not engage it".to_owned(),
                            format!(
                                "This prerequisite is engagement, not availability — rung {:?} \
                                 being implemented, or engaged by some other request, does not \
                                 satisfy it.",
                                prerequisite.rung
                            ),
                        ),
                        support => (
                            match support {
                                Some(support) => format!("the provider declares it {support:?}"),
                                None => "the provider does not declare it at all".to_owned(),
                            },
                            format!(
                                "Declaring rung {:?} Implemented — or \
                                 StructurallyNotApplicable, if this architecture has no such \
                                 component to stage — is what makes {:?} selectable. Engagement is \
                                 per request, so rung {:?} running on some other request, or \
                                 engaged earlier on this warm generator, does not satisfy it.",
                                prerequisite.rung, selection.strategy, prerequisite.rung
                            ),
                        ),
                    };
                    return Err(Error::Unsupported(format!(
                        "{} cannot execute {:?}: it requires rung {:?} to be ENGAGED IN THE SAME \
                         REQUEST, and {why}. {caveat}",
                        self.provider_id, selection.strategy, prerequisite.rung
                    )));
                }
            }
        }
        validate_selected_parameters(selection, self)
            .map_err(|message| Error::Unsupported(format!("{}: {message}", self.provider_id)))
    }
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
    let requires_decode = contract.engages(selection.strategy, MemoryStrategy::BoundedDecode);
    let requires_attention = contract.engages(selection.strategy, MemoryStrategy::BoundedAttention);
    let requires_transformer = contract.engages(
        selection.strategy,
        MemoryStrategy::BoundedTransformerResidency,
    );

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
}

/// Canonical live-budget accounting owned by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryBudget {
    pub total_bytes: u64,
    pub committed_bytes: u64,
    pub reclaimable_bytes: u64,
    pub reserved_headroom_bytes: u64,
}

impl MemoryBudget {
    /// Effective request budget. Reserved headroom is removed from the currently free plus
    /// reclaimable memory, while the total-minus-headroom ceiling prevents reclaimable accounting
    /// from exceeding physical capacity. All arithmetic is saturating.
    pub fn effective_bytes(self) -> u64 {
        let ceiling = self
            .total_bytes
            .saturating_sub(self.reserved_headroom_bytes);
        self.total_bytes
            .saturating_sub(self.committed_bytes)
            .saturating_add(self.reclaimable_bytes)
            .saturating_sub(self.reserved_headroom_bytes)
            .min(ceiling)
    }

    /// Exact-boundary fits are accepted.
    pub fn fits(self, predicted_peak_bytes: u64) -> bool {
        predicted_peak_bytes <= self.effective_bytes()
    }
}

/// Cold/warm state is request-scoped and never implies reuse of a prior selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryCacheState {
    Cold,
    Warm,
}

/// Advertised request surface. Optimized evidence never transfers between these modes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryMode {
    TextToImage,
    ImageToImage,
    Edit,
    Other(String),
}

/// Context for provider safety and lifecycle hooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryRunContext {
    pub selection: MemorySelection,
    /// Provider calibration identity that the caller used for admission. Resident requests carry
    /// this handshake too; optimized-only evidence validation is not sufficient.
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    pub mode: MemoryMode,
    pub has_reference: bool,
    pub use_pid: bool,
    pub has_phases: bool,
    pub geometry: MemoryGeometry,
    pub overlay: Option<String>,
    pub budget: MemoryBudget,
    pub predicted_peak_bytes: u64,
    pub cache_state: MemoryCacheState,
    pub evidence_revision: String,
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
    pub backend: String,
    pub tier: MemoryNumericTier,
    pub mode: String,
    pub overlay: Option<String>,
    pub geometry: MemoryGeometry,
    pub strategy: MemoryStrategy,
    pub parameters: MemoryStrategyParameters,
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

/// Dynamic/static evidence handshake consumed by the shared selector.
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
    pub predicted_peak_bytes: u64,
    pub observed_peak_bytes: Option<u64>,
    pub parity: MemoryParityContract,
    pub parity_result: MemoryParityResult,
}

impl MemoryEvidence {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
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

/// Truthful pre-OOM rejection payload. The caller may populate a measured smaller geometry; it must
/// not invent advice from unknown or unverified evidence.
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
            calibration: Some(MemoryCalibrationIdentity::new("test-layout-v1")),
            asset_facts: MemoryAssetFacts::default(),
            runtime: MemoryRuntimeSemantics::default(),
        }
    }

    #[test]
    fn compatibility_default_never_advertises_an_optimized_rung() {
        let contract = MemoryProviderContract::compatibility_default("legacy", mlx_backend());
        assert!(contract.conformance_errors().is_empty());
        assert!(contract.calibration.is_none());
        for strategy in MemoryStrategy::ALL {
            let support = &contract.capability(strategy).unwrap().support;
            if strategy == MemoryStrategy::Resident {
                assert_eq!(support, &MemoryStrategySupport::Implemented);
            } else {
                assert_eq!(support, &MemoryStrategySupport::Missing);
            }
        }
    }

    #[test]
    fn optimized_evidence_requires_every_dimension_and_matching_fingerprint() {
        let contract = adopted_contract();
        let mut evidence = MemoryEvidence {
            key: MemoryEvidenceKey {
                resolved_route: "test".to_owned(),
                backend: "mlx".to_owned(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                },
                mode: "text_to_image".to_owned(),
                overlay: None,
                geometry: MemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                },
                strategy: MemoryStrategy::BoundedDecode,
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
    fn provider_contract_requires_hooks_for_implemented_rungs() {
        let mut contract = adopted_contract();
        contract.lifecycle.attention_chunking = false;
        assert!(contract
            .conformance_errors()
            .iter()
            .any(|error| error.contains("attention_chunking")));
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
                backend: "mlx".to_owned(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                },
                mode: "text_to_image".to_owned(),
                overlay: None,
                geometry: MemoryGeometry {
                    width: 512,
                    height: 512,
                    batch: 1,
                    frames: 1,
                },
                strategy: MemoryStrategy::BoundedDecode,
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
        }
    }

    /// SC-15805: the graph itself, asserted edge by edge rather than through a loop so that moving
    /// the edge to a different rung — or dropping it — cannot pass.
    #[test]
    fn the_declared_prerequisite_graph_is_one_engagement_edge_on_rung_four() {
        assert_eq!(MemoryStrategy::Resident.requires(), &[]);
        assert_eq!(MemoryStrategy::StagedResidency.requires(), &[]);
        assert_eq!(MemoryStrategy::BoundedDecode.requires(), &[]);
        assert_eq!(MemoryStrategy::BoundedAttention.requires(), &[]);
        assert_eq!(
            MemoryStrategy::BoundedTransformerResidency.requires(),
            &[MemoryStrategyPrerequisite {
                rung: MemoryStrategy::StagedResidency,
                // ENGAGEMENT, not availability. If this ever reads as an availability scope the
                // rung-4 refusal below stops meaning what its test name says.
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            }]
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

    /// SC-15805 — rung 4 is refused when rung 1 is not engaged in the same request, and the refusal
    /// says so. A window over an already-materialized trunk bounds nothing.
    #[test]
    fn rung_four_is_refused_when_rung_one_is_not_engaged() {
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

        // Baseline: rung 1 implemented ⇒ engaged by the cost-order default ⇒ accepted.
        rung_four(&adopted_contract()).expect("rung 4 is executable when rung 1 is engaged");

        let missing = with_support(
            adopted_contract(),
            MemoryStrategy::StagedResidency,
            MemoryStrategySupport::Missing,
        );
        let error = rung_four(&missing)
            .expect_err("rung 4 must be refused when its prerequisite rung is not engaged")
            .to_string();
        assert!(error.contains("StagedResidency"), "{error}");
        assert!(
            error.contains("ENGAGED IN THE SAME REQUEST"),
            "the refusal must name engagement, not availability: {error}"
        );

        // The edge is UNCONDITIONAL: naming the encoder scope does not buy an exemption. (The
        // parameter itself is rejected by `adopted_contract`'s empty component declaration, so this
        // asserts the prerequisite fires FIRST and on its own terms.)
        let mut encoder_scoped = with_support(
            adopted_contract(),
            MemoryStrategy::StagedResidency,
            MemoryStrategySupport::Missing,
        );
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
            .expect_err("a TextEncoder-scoped window does not escape the prerequisite")
            .to_string();
        assert!(error.contains("ENGAGED IN THE SAME REQUEST"), "{error}");
    }

    /// `StructurallyNotApplicable` satisfies the edge vacuously — it asserts the architecture has no
    /// such component, which is not evidence that the trunk is eagerly materialized. Pinned so the
    /// choice stays visible rather than accidental.
    #[test]
    fn a_structurally_inapplicable_rung_one_satisfies_the_edge_vacuously() {
        let contract = with_support(
            adopted_contract(),
            MemoryStrategy::StagedResidency,
            MemoryStrategySupport::StructurallyNotApplicable {
                reason: "no separable conditioning component".to_owned(),
            },
        );
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
            .expect("a structurally inapplicable prerequisite is vacuous, not fatal");
    }

    /// SC-15805: rung 1 is NOT part of the cost-order default, but rung 4's declared edge still
    /// engages it.
    ///
    /// Rungs 2 and 3 bound scratch and are free, so a cumulative default costs the caller nothing.
    /// Rung 1 bounds residency and *"may evict the warm cross-request cache"* — a real cost paid by
    /// the next request. Dragging it into a rung-2 selection that never needed it is a straight loss.
    ///
    /// The regression this guards is severe and non-obvious: `engages` is what SATISFIES rung 4's
    /// prerequisite in `validate_selection`, so the naive fix — excluding rung 1 from `engages`
    /// outright — makes rung 4 **permanently unselectable**. Rung 1 is therefore engaged when the
    /// selection IS rung 1, or when the selection's `requires()` graph names it (derived from the
    /// graph, not hardcoded to rung 4, so a future edge implies its own engagement).
    #[test]
    fn rung_one_is_not_dragged_in_by_cost_order_but_the_rung_four_edge_still_engages_it() {
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

        // 2. THE REGRESSION GUARD. Rung 4 declares rung 1 as a prerequisite, so it engages it — and
        //    `validate_selection` therefore still admits rung 4 on a provider that implements rung 1.
        assert!(
            MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::StagedResidency),
            "rung 4's declared prerequisite must keep engaging rung 1, or rung 4 becomes \
             permanently unselectable"
        );
        assert!(MemoryStrategy::StagedResidency.engages(MemoryStrategy::StagedResidency));
        // Engagement follows the DECLARED graph, so the two agree by construction rather than by a
        // hardcoded `BoundedTransformerResidency` arm that could drift away from `requires()`.
        for strategy in MemoryStrategy::ALL {
            let declared = strategy
                .requires()
                .iter()
                .any(|edge| edge.rung == MemoryStrategy::StagedResidency);
            assert_eq!(
                strategy.engages(MemoryStrategy::StagedResidency),
                declared || strategy == MemoryStrategy::StagedResidency,
                "{strategy:?}'s rung-1 engagement drifted from its declared prerequisite graph"
            );
        }
        // End to end: rung 4 is still selectable on a provider implementing every rung.
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
            .expect("rung 4 must remain selectable after rung 1 left the cost-order default");

        // 3. The AC's stated cost-order default is untouched: rung 4 still engages 2 and 3.
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
}
