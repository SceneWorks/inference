//! The strategy ladder, its prerequisites and capabilities, and the parameter domains a provider owns.

use super::*;

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
    /// Add phase eviction to an otherwise independent bounded request. None preserves the
    /// historical composition; this never disables a declared staging prerequisite.
    pub stage_residency: Option<bool>,
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
