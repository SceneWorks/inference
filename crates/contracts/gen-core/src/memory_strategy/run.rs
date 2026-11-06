//! Per-run context, behavior routes, byte totals, and the provider request-scope lifecycle.

use super::*;

/// Reference-only geometry resolved from actual owned image carriers before
/// admission/cache sizing, not user-authored numeric hints. `conditioning_tokens`
/// excludes TEXT tokens and is not multiplied by CFG/job/decode batch. A backend
/// prices encoder work by evaluating ALL ordered `(width, height)` canvases at
/// its actual precision/tile strategy and taking the largest allocation demand.
/// There is deliberately no caller-supplied peak or preselected maximum canvas.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryReferenceResourceFacts {
    pub conditioning_tokens: u64,
    pub prepared_canvases: Vec<(u32, u32)>,
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
    /// Actual ordered reference geometry, when the provider requires full token/
    /// encoder-workspace proof. `None` preserves legacy/no-reference callers.
    pub reference_resources: Option<MemoryReferenceResourceFacts>,
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
        reference_resources: None,
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

/// Sum resident-byte `values` for a memory contract, refusing `u64` overflow with a typed
/// [`Error::Msg`] naming the `owner` (a provider id) and the `label` of the component being priced.
///
/// One definition for the per-provider `checked_sum` copies (sc-27341): every byte total a
/// contract publishes must fail closed rather than wrap.
pub fn checked_byte_total(
    owner: &str,
    label: &str,
    values: impl IntoIterator<Item = u64>,
) -> Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| Error::Msg(format!("{owner}: {label} byte total overflows u64")))
    })
}

/// Forward the five pure pass-through [`MemoryRequestScope`] hooks — `enter_phase`, `leave_phase`,
/// `configure_decode`, `configure_attention` and `materialize_transformer_window` — to a wrapped
/// scope field (sc-27424, sc-27341).
///
/// Invoke it inside an `impl MemoryRequestScope for Wrapper { .. }` block, naming the field that
/// holds the backend's shared request-scope core (or any other `MemoryRequestScope`); the trait must
/// be in scope at the call site, which implementing it by name already guarantees. The wrapper still
/// writes `configure_request` and `finish` itself, since those are where wrappers add their checks.
///
/// It deliberately does **not** forward `configure_admitted_request`. That method has a refusing
/// default, and forwarding it blindly would let the immutable admitted path bypass whatever the
/// wrapper's `configure_request` checks (an evidence revision, an overlay or source receipt). A
/// wrapper that supports the admitted path opts in by writing it.
#[macro_export]
macro_rules! delegate_memory_request_scope {
    ($field:ident) => {
        fn enter_phase(&mut self, phase: $crate::MemoryPhase) -> $crate::Result<()> {
            self.$field.enter_phase(phase)
        }

        fn leave_phase(&mut self, phase: $crate::MemoryPhase) -> $crate::Result<()> {
            self.$field.leave_phase(phase)
        }

        fn configure_decode(
            &mut self,
            tile_edge: u32,
            overlap: u32,
            geometry: $crate::MemoryGeometry,
        ) -> $crate::Result<()> {
            self.$field.configure_decode(tile_edge, overlap, geometry)
        }

        fn configure_attention(&mut self, chunk_size: u32) -> $crate::Result<()> {
            self.$field.configure_attention(chunk_size)
        }

        fn materialize_transformer_window(
            &mut self,
            first_block: u32,
            block_count: u32,
        ) -> $crate::Result<()> {
            self.$field
                .materialize_transformer_window(first_block, block_count)
        }
    };
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
    /// Configure an ALREADY canonical, admission-bound immutable request. No
    /// media/prompt clone and no fabricated metadata carrier is permitted.
    /// Existing providers keep the old mutable contract unless they explicitly
    /// implement this source-backed path; unsupported stops before generation.
    fn configure_admitted_request(&mut self, _request: &GenerationRequest) -> Result<()> {
        Err(Error::Unsupported(
            "provider lifecycle does not support immutable admitted configuration".into(),
        ))
    }
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
