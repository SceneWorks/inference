//! Shared MLX memory-strategy request lifecycle.
//!
//! Provider families configure their admitted route and rung parameters here; this core owns the
//! invariants which must not drift between them: request geometry/route identity, exact attention
//! and transformer-window parameters (including ragged tails), terminal evaluation/cache eviction,
//! and the finished-state guard for every lifecycle hook.

use crate::gen_core::{
    Error, GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryDecodeArtifactIdentity,
    MemoryDecodeGeometryPolicy, MemoryDecodePolicyQuery, MemoryDecodeQualityDisposition,
    MemoryGeometry, MemoryNumericTier, MemoryPhase, MemoryProviderContract, MemoryRequestScope,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy, Result,
};
use std::cell::RefCell;

type DecodeValidator = Box<dyn Fn(bool, u32, u32) -> Result<()> + 'static>;
type CleanupAction = Box<dyn FnMut() -> Result<()> + 'static>;

/// Whether a decode-quality row carries a well-formed forensic source stamp.
///
/// This is a shape check, not a currency check. sc-19728 removed the comparison against a constant
/// compiled into the running build: the stamp records which source closure produced the
/// measurement, and measurements stand until they are remeasured. What must still hold is that a
/// row admitted into a request scope carries a real digest rather than an empty or malformed one,
/// so the receipt a decode is authorized against is always traceable.
fn is_quality_stamp(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GeometryDecodeAuthorization {
    provider_id: &'static str,
    geometry: MemoryGeometry,
    load_shape: LoadShape,
    artifact: MemoryDecodeArtifactIdentity,
    use_pid: bool,
    tile_edge: u32,
    overlap: u32,
    production_evidence_sha256: String,
}

thread_local! {
    static GEOMETRY_DECODE_AUTHORIZATION: RefCell<Option<GeometryDecodeAuthorization>> =
        const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LifecycleState {
    Active,
    CleanupPending,
    Complete,
}

/// Provider-owned parameters for [`MlxRequestScopeCore`].
pub struct MlxRequestScopeConfig {
    pub provider_id: &'static str,
    pub geometry: MemoryGeometry,
    pub load_shape: LoadShape,
    /// Provider default used when [`GenerationRequest::frames`] is omitted. Image providers retain
    /// the historical value `1`; video providers set their real default so an omitted request value
    /// is still bound to the exact admitted frame geometry.
    pub default_frames: u32,
    pub memory: Option<GenerationMemory>,
    pub use_pid: bool,
    pub attention_chunk_size: Option<u32>,
    pub transformer_window: Option<u32>,
    /// The layer count read from the provider's model configuration.
    pub transformer_blocks: u32,
    pub decode_validator: DecodeValidator,
    decode_authorization: Option<GeometryDecodeAuthorization>,
}

impl MlxRequestScopeConfig {
    pub fn new(
        provider_id: &'static str,
        geometry: MemoryGeometry,
        memory: Option<GenerationMemory>,
        use_pid: bool,
        transformer_blocks: usize,
        decode_validator: impl Fn(bool, u32, u32) -> Result<()> + 'static,
    ) -> Result<Self> {
        let transformer_blocks = u32::try_from(transformer_blocks).map_err(|_| {
            Error::Unsupported(format!(
                "{provider_id}: transformer layer count {transformer_blocks} exceeds u32"
            ))
        })?;
        Ok(Self {
            provider_id,
            geometry,
            load_shape: LoadShape::EagerMaterialization,
            default_frames: 1,
            memory,
            use_pid,
            attention_chunk_size: None,
            transformer_window: None,
            transformer_blocks,
            decode_validator: Box::new(decode_validator),
            decode_authorization: None,
        })
    }

    /// Bind the exact policy row admitted by the shared safety gate to this request scope. The
    /// production decoder consumes the request-local authorization; a hand-built
    /// `GenerationMemory` with no scope cannot replay the tile pair on another coordinate.
    pub fn authorize_geometry_decode(&mut self, policy: &MemoryDecodeGeometryPolicy) -> Result<()> {
        if !matches!(policy.disposition, MemoryDecodeQualityDisposition::Admitted) {
            return Err(Error::Unsupported(format!(
                "{}: cannot authorize a refused decode-quality row",
                self.provider_id
            )));
        }
        if policy.artifact.repository.trim().is_empty()
            || policy.artifact.revision.len() != 40
            || !policy
                .artifact
                .revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || policy.artifact.variant.trim().is_empty()
            || policy.artifact.fingerprint.trim().is_empty()
            || !is_quality_stamp(&policy.implementation_fingerprint)
        {
            return Err(Error::Unsupported(format!(
                "{}: decode-quality authorization has an incomplete artifact/source identity",
                self.provider_id
            )));
        }
        let memory = self
            .memory
            .filter(|memory| memory.tile_vae_decode)
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{}: decode-quality authorization requires bounded decode selected",
                    self.provider_id
                ))
            })?;
        if policy.geometry != self.geometry
            || policy.load_shape != self.load_shape
            || policy.use_pid != self.use_pid
            || memory.decode_tile_edge != Some(policy.tile_edge)
            || memory.decode_overlap != Some(policy.overlap)
        {
            return Err(Error::Unsupported(format!(
                "{}: decode-quality authorization does not match the request geometry/parameters",
                self.provider_id
            )));
        }
        self.decode_authorization = Some(GeometryDecodeAuthorization {
            provider_id: self.provider_id,
            geometry: policy.geometry,
            load_shape: policy.load_shape,
            artifact: policy.artifact.clone(),
            use_pid: policy.use_pid,
            tile_edge: policy.tile_edge,
            overlap: policy.overlap,
            production_evidence_sha256: policy.production_evidence_sha256.clone(),
        });
        let expected_use_pid = policy.use_pid;
        let expected_edge = policy.tile_edge;
        let expected_overlap = policy.overlap;
        let provider_id = self.provider_id;
        self.decode_validator = Box::new(move |use_pid, edge, overlap| {
            if (use_pid, edge, overlap) == (expected_use_pid, expected_edge, expected_overlap) {
                Ok(())
            } else {
                Err(Error::Unsupported(format!(
                    "{provider_id}: decode parameters do not match the admitted geometry-quality row"
                )))
            }
        });
        Ok(())
    }
}

/// Install the exact shared-contract quality row selected for this request, when bounded decode is
/// engaged. Legacy providers with an empty policy table retain their existing validator path.
pub fn authorize_selected_geometry_decode(
    config: &mut MlxRequestScopeConfig,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> Result<()> {
    if !contract.engages_selection(&context.selection, MemoryStrategy::BoundedDecode) {
        return Ok(());
    }
    if let Some(policy) = contract.decode_geometry_policy(
        MemoryDecodePolicyQuery {
            tier: context.selection.tier,
            load_shape: context.load_shape,
            mode_key: context.mode.as_key(),
            overlay: context.overlay.as_deref(),
            geometry: context.geometry,
            use_pid: context.use_pid,
        },
        Some(context.selection.parameters),
    )? {
        config.authorize_geometry_decode(policy)?;
    }
    Ok(())
}

/// Validate and consume the active request scope's exact geometry-quality authority.
///
/// The authorization remains active for all images in the request count and is cleared only when
/// the surrounding [`MlxRequestScopeCore`] finishes or drops.
pub fn validate_geometry_decode_authorization(
    provider_id: &'static str,
    request: &GenerationRequest,
    tile_edge: u32,
    overlap: u32,
) -> Result<String> {
    GEOMETRY_DECODE_AUTHORIZATION.with(|slot| {
        let binding = slot.borrow();
        let binding = binding.as_ref().ok_or_else(|| {
            Error::Unsupported(format!(
                "{provider_id}: bounded decode requires request-scoped production-latent quality authorization"
            ))
        })?;
        let geometry = MemoryGeometry {
            width: request.width,
            height: request.height,
            batch: request.count,
            frames: 1,
            reference_count: request.memory_reference_count(),
        };
        if binding.provider_id != provider_id
            || binding.geometry != geometry
            || !matches!(
                binding.load_shape,
                LoadShape::EagerMaterialization | LoadShape::DeferredMaterialization
            )
            || binding.artifact.repository.trim().is_empty()
            || binding.artifact.revision.trim().is_empty()
            || binding.artifact.variant.trim().is_empty()
            || binding.artifact.fingerprint.trim().is_empty()
            // No stamp check here. It used to compare against a build-local constant (sc-19728).
            // The shape check lives at `authorize_geometry_decode` instead, which is the public
            // API boundary a caller can hand an arbitrary policy to; this site reads a
            // thread-local that only that call populates, so repeating it here would add a
            // condition that cannot fail on any reachable path.
            || binding.use_pid != request.use_pid
            || binding.tile_edge != tile_edge
            || binding.overlap != overlap
        {
            return Err(Error::Unsupported(format!(
                "{provider_id}: bounded decode does not match the active geometry-quality authorization"
            )));
        }
        Ok(binding.production_evidence_sha256.clone())
    })
}

/// Terminal-cleanup policy. `None` is used only by weights-free behavior probes.
#[derive(Clone, Copy)]
pub enum MlxScopeCleanup {
    Device,
    None,
}

/// The single MLX request-scope implementation shared by adopting provider families.
pub struct MlxRequestScopeCore {
    config: MlxRequestScopeConfig,
    cleanup: CleanupAction,
    state: LifecycleState,
    decode_authorization_armed: bool,
}

impl MlxRequestScopeCore {
    pub fn new(config: MlxRequestScopeConfig) -> Self {
        Self::with_cleanup(config, MlxScopeCleanup::Device)
    }

    pub fn with_cleanup(config: MlxRequestScopeConfig, cleanup: MlxScopeCleanup) -> Self {
        let cleanup: CleanupAction = match cleanup {
            MlxScopeCleanup::Device => Box::new(|| {
                let barrier = mlx_rs::Array::from(0.0_f32);
                barrier.eval().map_err(crate::Error::from)?;
                drop(barrier);
                mlx_rs::memory::clear_cache();
                Ok(())
            }),
            MlxScopeCleanup::None => Box::new(|| Ok(())),
        };
        Self::with_cleanup_action(config, cleanup)
    }

    fn with_cleanup_action(config: MlxRequestScopeConfig, cleanup: CleanupAction) -> Self {
        Self {
            config,
            cleanup,
            state: LifecycleState::Active,
            decode_authorization_armed: false,
        }
    }

    fn ensure_active(&self) -> Result<()> {
        if self.state != LifecycleState::Active {
            Err(Error::Msg(format!(
                "{}: memory-strategy request scope is already finished",
                self.config.provider_id
            )))
        } else {
            Ok(())
        }
    }

    fn cleanup_pending(&mut self) -> Result<()> {
        debug_assert_eq!(self.state, LifecycleState::CleanupPending);
        match (self.cleanup)() {
            Ok(()) => {
                self.state = LifecycleState::Complete;
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn validate_geometry(&self, geometry: MemoryGeometry) -> Result<()> {
        if geometry.width == self.config.geometry.width
            && geometry.height == self.config.geometry.height
            && geometry.frames == self.config.geometry.frames
            && geometry.reference_count == self.config.geometry.reference_count
            && geometry.batch > 0
            && geometry.batch <= self.config.geometry.batch
        {
            return Ok(());
        }
        Err(Error::Unsupported(format!(
            "{}: hook geometry {}x{}x{} frames={} references={} does not fit admitted {}x{}x{} frames={} references={}",
            self.config.provider_id,
            geometry.width,
            geometry.height,
            geometry.batch,
            geometry.frames,
            geometry.reference_count,
            self.config.geometry.width,
            self.config.geometry.height,
            self.config.geometry.batch,
            self.config.geometry.frames,
            self.config.geometry.reference_count
        )))
    }
}

impl MlxRequestScopeCore {
    fn validate_configuration(&self, request: &GenerationRequest) -> Result<()> {
        self.ensure_active()?;
        if request.use_pid != self.config.use_pid {
            return Err(Error::Unsupported(format!(
                "{}: request use_pid={} does not match admitted use_pid={}",
                self.config.provider_id, request.use_pid, self.config.use_pid
            )));
        }
        if request.width != self.config.geometry.width
            || request.height != self.config.geometry.height
            || request.count == 0
            || request.count > self.config.geometry.batch
            || request.frames.unwrap_or(self.config.default_frames) != self.config.geometry.frames
            || request.memory_reference_count() != self.config.geometry.reference_count
        {
            return Err(Error::Unsupported(format!(
                "{}: request geometry {}x{}x{} frames={} references={} does not fit admitted {}x{}x{} frames={} references={}",
                self.config.provider_id,
                request.width,
                request.height,
                request.count,
                request.frames.unwrap_or(self.config.default_frames),
                request.memory_reference_count(),
                self.config.geometry.width,
                self.config.geometry.height,
                self.config.geometry.batch,
                self.config.geometry.frames,
                self.config.geometry.reference_count
            )));
        }
        Ok(())
    }
    fn arm_configuration(&mut self) -> Result<()> {
        if let Some(authorization) = self.config.decode_authorization.clone() {
            GEOMETRY_DECODE_AUTHORIZATION.with(|slot| {
                let mut active = slot.borrow_mut();
                if active.is_some() {
                    return Err(Error::Unsupported(format!(
                        "{}: a geometry-decode authorization is already active on this thread",
                        self.config.provider_id
                    )));
                }
                *active = Some(authorization);
                Ok(())
            })?;
            self.decode_authorization_armed = true;
        }
        Ok(())
    }
}

impl MemoryRequestScope for MlxRequestScopeCore {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> Result<()> {
        self.validate_configuration(request)?;
        self.arm_configuration()?;
        request.memory = self.config.memory;
        Ok(())
    }
    fn configure_admitted_request(&mut self, request: &GenerationRequest) -> Result<()> {
        self.validate_configuration(request)?;
        // The scope selects the ladder, not independent execution domains.
        // Preserve the original admitted cadence/CFG/FFN/cache/fault controls;
        // the provider's ordinary request floor validates those domains.
        let mut ladder = request.memory.ok_or_else(|| {
            Error::Unsupported(format!(
                "{}: immutable admitted configuration requires explicit memory, not legacy None",
                self.config.provider_id
            ))
        })?;
        let selected = self.config.memory.ok_or_else(|| {
            Error::Unsupported(format!(
                "{}: immutable admitted configuration has no selected ladder",
                self.config.provider_id
            ))
        })?;
        ladder.graph_eval_cadence = selected.graph_eval_cadence;
        ladder.ffn_chunk = selected.ffn_chunk;
        ladder.cfg_batching = selected.cfg_batching;
        ladder.prefix_kv_cache = selected.prefix_kv_cache;
        ladder.calibration_error_phase = selected.calibration_error_phase;
        ladder.calibration_fault_harness_authorized = selected.calibration_fault_harness_authorized;
        if ladder != selected {
            return Err(Error::Unsupported(format!(
                "{}: immutable request ladder differs from admitted canonical configuration",
                self.config.provider_id
            )));
        }
        self.arm_configuration()
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> Result<()> {
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> Result<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> Result<()> {
        self.ensure_active()?;
        self.validate_geometry(geometry)?;
        (self.config.decode_validator)(self.config.use_pid, tile_edge, overlap)
    }

    fn configure_attention(&mut self, chunk_size: u32) -> Result<()> {
        self.ensure_active()?;
        if self.config.attention_chunk_size == Some(chunk_size) {
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "{}: attention chunk size {chunk_size} was not admitted",
                self.config.provider_id
            )))
        }
    }

    fn materialize_transformer_window(&mut self, first_block: u32, block_count: u32) -> Result<()> {
        self.ensure_active()?;
        let Some(window) = self.config.transformer_window else {
            return Err(Error::Unsupported(format!(
                "{}: bounded transformer residency was not selected",
                self.config.provider_id
            )));
        };
        if window == 0 || block_count == 0 || !first_block.is_multiple_of(window) {
            return Err(Error::Unsupported(format!(
                "{}: transformer window {window} requires a non-zero block count and aligned start, got {block_count} blocks at {first_block}",
                self.config.provider_id
            )));
        }
        if first_block >= self.config.transformer_blocks {
            return Err(Error::Unsupported(format!(
                "{}: transformer window starts at block {first_block}, past the {}-block stack",
                self.config.provider_id, self.config.transformer_blocks
            )));
        }
        let expected = window.min(self.config.transformer_blocks - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "{}: admitted window {window} requires {expected} blocks at {first_block}, got {block_count}",
                self.config.provider_id
            )))
        }
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> Result<()> {
        self.ensure_active()?;
        if self.decode_authorization_armed {
            GEOMETRY_DECODE_AUTHORIZATION.with(|slot| *slot.borrow_mut() = None);
            self.decode_authorization_armed = false;
        }
        self.state = LifecycleState::CleanupPending;
        self.cleanup_pending()
    }
}

impl Drop for MlxRequestScopeCore {
    fn drop(&mut self) {
        if self.decode_authorization_armed {
            GEOMETRY_DECODE_AUTHORIZATION.with(|slot| *slot.borrow_mut() = None);
            self.decode_authorization_armed = false;
        }
        if self.state == LifecycleState::Active {
            self.state = LifecycleState::CleanupPending;
        }
        if self.state == LifecycleState::CleanupPending {
            let _ = self.cleanup_pending();
        }
    }
}

/// The numeric tier a loader runs for `spec` when the provider declares no component precision
/// floors: the spec's precision and load-time quantization, exactly as requested.
///
/// Shared by every provider whose tier is a pure function of the spec (sc-27257). That includes the
/// convert-at-install families (Anima), whose worker records the packed on-disk tier in
/// `LoadSpec::quantize` for the resolved directory, and the whole-pipeline-tier families (Qwen-Image
/// 2.1, Iris, FLUX 3), for which no component is resident above the selected width. A provider that
/// reads packed bits off disk itself, or publishes component floors, keeps its own resolver.
pub fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

/// The rung-4 window cadence this request admits: the selected `transformer_window_size` when the
/// contract engages [`MemoryStrategy::BoundedTransformerResidency`] for the selected strategy,
/// otherwise `None`.
///
/// Contract-aware `engages` rather than an ordinal or equality compare, so a provider exclusion or
/// realization prerequisite moves the window with it (sc-27325).
pub fn selected_transformer_window(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> Option<u32> {
    contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten()
}

/// The one place a provider's refused safety decision becomes a typed error, shared with every
/// backend. Providers whose scope is not built by [`MlxRequestScopeCore::begin`] call this first,
/// before any other fallible admission step.
pub use crate::gen_core::admit_memory_safety as admit;

// ── Request-side resolution shared by the adopting providers (sc-27257) ──────────────────────────

/// Rung 3's plan for a provider whose only bounded-attention lever is the constrained score budget.
///
/// An unselected request returns the exact historical unbounded, uncancellable plan; a request that
/// chunks attention gets [`AttentionBudget::CONSTRAINED`](crate::attention::AttentionBudget) with
/// the request's cancel flag threaded through, so a per-chunk `eval` observes a cancel promptly.
pub fn constrained_attention_plan(req: &GenerationRequest) -> crate::attention::AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => crate::attention::AttentionPlan::budgeted(
            crate::attention::AttentionBudget::CONSTRAINED,
        )
        .with_cancel(&req.cancel),
        _ => crate::attention::AttentionPlan::UNBOUNDED,
    }
}

/// Rung 1: whether a request's memory block stages component residency. `default_staged` is the
/// load-time verdict a request that names nothing keeps.
pub fn requested_stage_residency(memory: Option<GenerationMemory>, default_staged: bool) -> bool {
    memory.map_or(default_staged, |memory| memory.stage_residency)
}

/// Rung 2: the spatial decode tiling a request's memory block selects, or `None` for the
/// single-pass decode.
///
/// An unnamed edge or overlap takes the provider's published default. `validate` sees the resolved
/// pair before a tiling is returned, so the provider's own domain check (or a cancel check) runs on
/// the production render path and not only at admission.
pub fn requested_decode_tiling(
    memory: Option<GenerationMemory>,
    default_edge: u32,
    default_overlap: u32,
    validate: impl FnOnce(u32, u32) -> crate::Result<()>,
) -> crate::Result<Option<crate::TilingConfig>> {
    let Some(memory) = memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    let edge = memory.decode_tile_edge.unwrap_or(default_edge);
    let overlap = memory.decode_overlap.unwrap_or(default_overlap);
    validate(edge, overlap)?;
    Ok(Some(crate::TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    )))
}

/// Rung 4: the window cadence a request's memory block selects, or `None` for the resident stack.
///
/// A component outside `components`, or a cadence `validate_size` refuses, is a typed rejection
/// rather than a silently narrowed (or silently *widened*) execution. An unnamed component takes
/// `default_component`, an unnamed size `default_size`.
pub fn requested_transformer_window(
    provider_id: &str,
    memory: Option<GenerationMemory>,
    default_component: crate::gen_core::TransformerComponent,
    components: &[crate::gen_core::TransformerComponent],
    default_size: u32,
    validate_size: impl FnOnce(u32) -> Result<()>,
) -> crate::Result<Option<usize>> {
    let Some(memory) = memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory
        .transformer_window_component
        .unwrap_or(default_component);
    if !components.contains(&component) {
        return Err(crate::Error::Unsupported(format!(
            "{provider_id} implements only the {default_component:?} transformer window \
             component, got {component:?}"
        )));
    }
    let size = memory.transformer_window_size.unwrap_or(default_size);
    validate_size(size).map_err(|error| {
        crate::Error::Unsupported(format!(
            "{provider_id} transformer window rejected: {error}"
        ))
    })?;
    Ok(Some(size as usize))
}

/// Refuse a rung-4 cadence outside the calibrated `sizes` domain.
///
/// The shared `validate_selection` already refuses an out-of-domain window at admission, so this is
/// defense in depth for the render path: *executing* at a wider cadence than the declared domain
/// **under**-predicts peak, and a harness that hand-builds a [`GenerationMemory`] never crosses
/// admission.
pub fn validate_calibrated_window(provider_id: &str, size: u32, sizes: &[u32]) -> Result<()> {
    if !sizes.contains(&size) {
        return Err(Error::Unsupported(format!(
            "{provider_id} transformer window {size} is outside the calibrated domain {sizes:?}"
        )));
    }
    Ok(())
}

/// Whether `contract` publishes sealed geometry-quality rows for rung 2, i.e. whether a bounded
/// decode must go through [`geometry_decode_tiling`] rather than a route-blind domain.
pub fn has_decode_geometry_table(contract: &MemoryProviderContract) -> bool {
    contract
        .capability(MemoryStrategy::BoundedDecode)
        .is_some_and(|capability| !capability.parameters.decode_geometry_policies.is_empty())
}

/// Record the single-pass decode decision for a request that did not select rung 2.
pub fn record_dense_decode() {
    crate::diagnostics::record_geometry_decode_decision(
        crate::diagnostics::DecodePolicyDisposition::Unchanged,
        crate::diagnostics::DecodePathDisposition::Dense,
        None,
    );
}

/// Resolve a geometry-aware bounded decode: the exact tile pair `memory` names, authorized by the
/// request-scoped production-latent quality row, recorded, and returned as a spatial tiling.
///
/// A missing edge or overlap is a typed refusal — a geometry row admits exact coordinates only.
pub fn geometry_decode_tiling(
    provider_id: &'static str,
    req: &GenerationRequest,
    memory: GenerationMemory,
) -> crate::Result<crate::TilingConfig> {
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        crate::Error::Unsupported(format!(
            "{provider_id}: geometry-aware bounded decode requires an exact tile edge"
        ))
    })?;
    let overlap = memory.decode_overlap.ok_or_else(|| {
        crate::Error::Unsupported(format!(
            "{provider_id}: geometry-aware bounded decode requires an exact overlap"
        ))
    })?;
    let evidence = validate_geometry_decode_authorization(provider_id, req, edge, overlap)?;
    crate::diagnostics::record_geometry_decode_decision(
        crate::diagnostics::DecodePolicyDisposition::GeometryTiled,
        crate::diagnostics::DecodePathDisposition::Tiled,
        Some(&evidence),
    );
    Ok(crate::TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    ))
}

/// Record one whole-render overlay component that no published rung bounds, skipping a zero: the
/// shared validator refuses a declared component with zero bytes, and a component that measured zero
/// is not evidence of residency anyway.
pub fn push_unbounded_overlay(
    into: &mut Vec<crate::gen_core::MemoryResidentComponent>,
    id: &str,
    kind: crate::gen_core::MemoryComponentKind,
    resident_bytes: u64,
) {
    if resident_bytes == 0 {
        return;
    }
    into.push(crate::gen_core::MemoryResidentComponent {
        id: id.to_owned(),
        kind,
        resident_bytes,
        bounded_by: None,
        residency: crate::gen_core::MemoryComponentResidency::WholeRender,
    });
}

impl MlxRequestScopeCore {
    /// The admission skeleton every adopting provider shares (sc-27257, sc-27325, sc-27341).
    ///
    /// `safety` is the provider's own safety-check decision, computed by the caller. A rejection
    /// becomes the one typed [`Error::Unsupported`] (the decision carries an already-rendered
    /// reason), and `configure` is not run. Otherwise `configure` builds the provider's
    /// [`MlxRequestScopeConfig`] — its decode validator, stack depth and rung parameters stay
    /// provider-owned — and the core is wrapped with `cleanup`.
    ///
    /// The ordering is the historical per-provider one: safety first, then any fallible
    /// configuration (route tables, layer counts), then construction.
    pub fn begin(
        safety: MemorySafetyDecision,
        cleanup: MlxScopeCleanup,
        configure: impl FnOnce() -> Result<MlxRequestScopeConfig>,
    ) -> Result<Self> {
        admit(safety)?;
        Ok(Self::with_cleanup(configure()?, cleanup))
    }
}

/// [`MlxRequestScopeCore::begin`] for a provider whose request scope is the shared core itself, boxed
/// in the shape `begin_memory_strategy_request` returns.
pub fn begin_scope(
    safety: MemorySafetyDecision,
    cleanup: MlxScopeCleanup,
    configure: impl FnOnce() -> Result<MlxRequestScopeConfig>,
) -> Result<Option<Box<dyn MemoryRequestScope + 'static>>> {
    Ok(Some(Box::new(MlxRequestScopeCore::begin(
        safety, cleanup, configure,
    )?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gen_core::{
        Conditioning, Image, MemoryBackend, MemoryDecodeArtifactIdentity,
        MemoryDecodeQualityFixture, MemoryMode, MemoryNumericTier, MemoryRunOutcome, Precision,
        Quant, MEMORY_DECODE_QUALITY_ABI,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    /// An arbitrary well-formed source stamp. Any 64-hex value must now be admitted — the point of
    /// sc-19728 is that this value is recorded, not matched — so the fixture deliberately does not
    /// derive from anything in the running build.
    const STAMP: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn config(blocks: usize) -> MlxRequestScopeConfig {
        MlxRequestScopeConfig::new(
            "fixture",
            MemoryGeometry {
                width: 64,
                height: 64,
                batch: 3,
                frames: 1,
                reference_count: 0,
            },
            None,
            false,
            blocks,
            |_use_pid, edge, overlap| {
                (edge == 32 && overlap == 4)
                    .then_some(())
                    .ok_or_else(|| Error::Unsupported("wrong decode route".into()))
            },
        )
        .unwrap()
    }

    #[test]
    fn immutable_admitted_configuration_borrows_original_and_never_rewrites_legacy_none() {
        let mut admitted = config(20);
        admitted.memory = Some(GenerationMemory::default());
        let mut scope = MlxRequestScopeCore::with_cleanup_action(admitted, Box::new(|| Ok(())));
        let memory = GenerationMemory {
            cfg_batching: Some(crate::gen_core::CfgBatching::Sequential),
            ..Default::default()
        };
        let request = GenerationRequest {
            width: 64,
            height: 64,
            count: 1,
            memory: Some(memory),
            ..Default::default()
        };
        let original = &request as *const GenerationRequest;
        scope.configure_admitted_request(&request).unwrap();
        assert_eq!(&request as *const GenerationRequest, original);
        assert_eq!(
            request.memory,
            Some(memory),
            "independent admitted execution must survive read-only ladder setup"
        );
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        let mut admitted = config(20);
        admitted.memory = Some(GenerationMemory::default());
        let mut scope = MlxRequestScopeCore::with_cleanup_action(admitted, Box::new(|| Ok(())));
        let mut legacy = GenerationRequest {
            width: 64,
            height: 64,
            count: 1,
            memory: None,
            ..Default::default()
        };
        assert!(scope.configure_admitted_request(&legacy).is_err());
        assert_eq!(
            legacy.memory, None,
            "immutable failure must not normalize/clone a fake carrier"
        );
        // Existing explicit mutable bridge retains exactly its former behavior.
        scope.configure_request(&mut legacy).unwrap();
        assert_eq!(legacy.memory, Some(GenerationMemory::default()));
        scope.finish(MemoryRunOutcome::Complete).unwrap();
    }

    fn reference() -> Conditioning {
        Conditioning::Reference {
            image: Image {
                width: 1,
                height: 1,
                pixels: vec![0, 0, 0],
            },
            strength: None,
        }
    }

    fn quality_policy(geometry: MemoryGeometry) -> MemoryDecodeGeometryPolicy {
        MemoryDecodeGeometryPolicy {
            quality_abi: MEMORY_DECODE_QUALITY_ABI,
            family: "fixture".to_owned(),
            resolved_route: "fixture-route".to_owned(),
            backend: MemoryBackend::Mlx,
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
            load_shape: LoadShape::EagerMaterialization,
            artifact: MemoryDecodeArtifactIdentity {
                repository: "SceneWorks/fixture".to_owned(),
                revision: "a".repeat(40),
                variant: "q4".to_owned(),
                fingerprint: format!("SceneWorks/fixture@{}:q4", "a".repeat(40)),
            },
            implementation_fingerprint: STAMP.to_owned(),
            mode: MemoryMode::TextToImage,
            overlay: None,
            geometry,
            use_pid: false,
            tile_edge: 32,
            overlap: 4,
            metric: "max_abs_rgb_u8".to_owned(),
            maximum_error: 48,
            fixtures: [7_u64, 99]
                .into_iter()
                .map(|seed| MemoryDecodeQualityFixture {
                    seed,
                    production_latent_provenance: format!("fixture@revision seed={seed}"),
                    production_latent_sha256: "a".repeat(64),
                    dense_output_sha256: "b".repeat(64),
                    tiled_output_sha256: "c".repeat(64),
                    observed_error: 31,
                })
                .collect(),
            production_evidence_sha256: String::new(),
            disposition: MemoryDecodeQualityDisposition::Admitted,
        }
        .seal()
    }

    #[test]
    fn geometry_decode_authority_is_exact_request_scoped_and_cleared() {
        let mut cfg = config(7);
        cfg.memory = Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(32),
            decode_overlap: Some(4),
            ..Default::default()
        });
        let policy = quality_policy(cfg.geometry);
        let evidence = policy.production_evidence_sha256.clone();
        cfg.authorize_geometry_decode(&policy).unwrap();
        let mut scope = MlxRequestScopeCore::with_cleanup(cfg, MlxScopeCleanup::None);
        let mut request = GenerationRequest {
            width: 64,
            height: 64,
            count: 3,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        assert_eq!(
            validate_geometry_decode_authorization("fixture", &request, 32, 4).unwrap(),
            evidence
        );
        assert!(validate_geometry_decode_authorization("fixture", &request, 48, 4).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(validate_geometry_decode_authorization("fixture", &request, 32, 4).is_err());
    }

    fn tiled_decode_config() -> MlxRequestScopeConfig {
        let mut cfg = config(7);
        cfg.memory = Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(32),
            decode_overlap: Some(4),
            ..Default::default()
        });
        cfg
    }

    #[test]
    fn geometry_decode_authority_rejects_load_shape_and_artifact_identity_drift() {
        let mut cfg = tiled_decode_config();
        let policy = quality_policy(cfg.geometry);

        cfg.load_shape = LoadShape::DeferredMaterialization;
        assert!(cfg.authorize_geometry_decode(&policy).is_err());
        cfg.load_shape = LoadShape::EagerMaterialization;

        let mut stale_artifact = policy;
        stale_artifact.artifact.revision = "not-a-revision".to_owned();
        assert!(cfg.authorize_geometry_decode(&stale_artifact).is_err());
    }

    /// sc-19728: a row measured by a different source closure now authorizes a decode.
    ///
    /// This case used to live in the test above, asserting the opposite — a `"f".repeat(64)` stamp
    /// was refused as "stale source identity". That refusal is the one this story removed, so the
    /// assertion is inverted rather than deleted: it is the observable proof for
    /// [`MlxRequestScopeConfig::authorize_geometry_decode`], and restoring the comparison against a
    /// build-local constant turns it red.
    #[test]
    fn a_foreign_source_closure_stamp_no_longer_refuses_a_decode_authorization() {
        for stamp in [
            "800d06acf579a36e604d91955fd6a6852ec70bc39701f7a320f1fdd2bf5ff29d",
            &"f".repeat(64),
        ] {
            let mut cfg = tiled_decode_config();
            let mut policy = quality_policy(cfg.geometry);
            policy.implementation_fingerprint = stamp.to_owned();
            let policy = policy.seal();
            let evidence = policy.production_evidence_sha256.clone();

            cfg.authorize_geometry_decode(&policy)
                .unwrap_or_else(|error| panic!("{stamp}: {error}"));

            // The stamp is sealed into `production_evidence_sha256`, so authorizing against a
            // differently-stamped row yields a different evidence digest. Asserting the digest
            // therefore pins the stamp's traceability without the binding having to re-read it.
            let mut scope = MlxRequestScopeCore::with_cleanup(cfg, MlxScopeCleanup::None);
            let mut request = GenerationRequest {
                width: 64,
                height: 64,
                count: 3,
                ..Default::default()
            };
            scope.configure_request(&mut request).unwrap();
            assert_eq!(
                validate_geometry_decode_authorization("fixture", &request, 32, 4)
                    .unwrap_or_else(|error| panic!("{stamp}: {error}")),
                evidence
            );
            scope.finish(MemoryRunOutcome::Complete).unwrap();
        }
    }

    /// The guard that survived at both authorization sites still discriminates on shape.
    #[test]
    fn a_malformed_source_closure_stamp_is_still_refused() {
        for malformed in ["", &"a".repeat(63), &"A".repeat(64), &"g".repeat(64)] {
            let mut cfg = tiled_decode_config();
            let mut policy = quality_policy(cfg.geometry);
            policy.implementation_fingerprint = malformed.to_owned();
            let policy = policy.seal();
            assert!(
                cfg.authorize_geometry_decode(&policy).is_err(),
                "{malformed:?} must not authorize a decode"
            );
        }
    }

    #[test]
    fn configured_request_must_match_exact_reference_cardinality() {
        let mut cfg = config(7);
        cfg.geometry.reference_count = 2;
        let mut scope = MlxRequestScopeCore::with_cleanup(cfg, MlxScopeCleanup::None);
        let mut exact = GenerationRequest {
            width: 64,
            height: 64,
            count: 1,
            conditioning: vec![reference(), reference()],
            ..Default::default()
        };
        scope.configure_request(&mut exact).unwrap();

        for conditioning in [
            vec![],
            vec![reference()],
            vec![reference(), reference(), reference(), reference()],
        ] {
            let mut mismatch = GenerationRequest {
                width: 64,
                height: 64,
                count: 1,
                conditioning,
                ..Default::default()
            };
            assert!(scope.configure_request(&mut mismatch).is_err());
        }
    }

    #[test]
    fn configured_request_must_match_the_admitted_frame_count() {
        let mut cfg = config(7);
        cfg.geometry.frames = 97;
        let mut scope = MlxRequestScopeCore::with_cleanup(cfg, MlxScopeCleanup::None);
        let mut exact = GenerationRequest {
            width: 64,
            height: 64,
            count: 1,
            frames: Some(97),
            ..Default::default()
        };
        scope.configure_request(&mut exact).unwrap();

        for frames in [None, Some(1), Some(89), Some(105)] {
            let mut mismatch = GenerationRequest {
                width: 64,
                height: 64,
                count: 1,
                frames,
                ..Default::default()
            };
            assert!(scope.configure_request(&mut mismatch).is_err());
        }
    }

    #[test]
    fn configured_request_resolves_frames_through_the_provider_default() {
        let mut cfg = config(7);
        cfg.geometry.frames = 81;
        cfg.default_frames = 81;
        let mut scope = MlxRequestScopeCore::with_cleanup(cfg, MlxScopeCleanup::None);

        for frames in [None, Some(81)] {
            let mut exact = GenerationRequest {
                width: 64,
                height: 64,
                count: 1,
                frames,
                ..Default::default()
            };
            scope.configure_request(&mut exact).unwrap();
        }
        for frames in [Some(1), Some(77), Some(85)] {
            let mut mismatch = GenerationRequest {
                width: 64,
                height: 64,
                count: 1,
                frames,
                ..Default::default()
            };
            assert!(scope.configure_request(&mut mismatch).is_err());
        }
    }

    #[test]
    fn lifecycle_rejects_double_finish_and_every_post_finish_hook() {
        let mut scope = MlxRequestScopeCore::with_cleanup(config(7), MlxScopeCleanup::None);
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
        let mut request = GenerationRequest {
            prompt: "x".into(),
            width: 64,
            height: 64,
            count: 1,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut request).is_err());
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.leave_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.configure_decode(32, 4, config(7).geometry).is_err());
        assert!(scope.configure_attention(4).is_err());
        assert!(scope.materialize_transformer_window(0, 1).is_err());
    }

    #[test]
    fn transformer_windows_accept_only_full_windows_and_the_ragged_tail() {
        let mut cfg = config(7);
        cfg.transformer_window = Some(3);
        let mut scope = MlxRequestScopeCore::with_cleanup(cfg, MlxScopeCleanup::None);
        scope.materialize_transformer_window(0, 3).unwrap();
        scope.materialize_transformer_window(3, 3).unwrap();
        scope.materialize_transformer_window(6, 1).unwrap();
        assert!(scope.materialize_transformer_window(0, 0).is_err());
        assert!(scope.materialize_transformer_window(1, 3).is_err());
        assert!(scope.materialize_transformer_window(6, 3).is_err());
        assert!(scope.materialize_transformer_window(7, 1).is_err());

        let mut zero = config(7);
        zero.transformer_window = Some(0);
        let mut zero = MlxRequestScopeCore::with_cleanup(zero, MlxScopeCleanup::None);
        assert!(zero.materialize_transformer_window(0, 1).is_err());
    }

    #[test]
    fn decode_geometry_must_match_the_admitted_shape_and_batch_maximum() {
        let mut scope = MlxRequestScopeCore::with_cleanup(config(7), MlxScopeCleanup::None);
        let admitted = config(7).geometry;
        scope
            .configure_decode(
                32,
                4,
                MemoryGeometry {
                    batch: 1,
                    ..admitted
                },
            )
            .unwrap();
        for geometry in [
            MemoryGeometry {
                width: 32,
                ..admitted
            },
            MemoryGeometry {
                height: 32,
                ..admitted
            },
            MemoryGeometry {
                frames: 2,
                ..admitted
            },
            MemoryGeometry {
                reference_count: 1,
                ..admitted
            },
            MemoryGeometry {
                batch: 0,
                ..admitted
            },
            MemoryGeometry {
                batch: 4,
                ..admitted
            },
        ] {
            assert!(scope.configure_decode(32, 4, geometry).is_err());
        }
    }

    #[test]
    fn cleanup_failure_is_terminal_and_drop_retries_only_cleanup() {
        let calls = Rc::new(Cell::new(0));
        let completed = Rc::new(Cell::new(false));
        let cleanup_calls = Rc::clone(&calls);
        let cleanup_completed = Rc::clone(&completed);
        let mut cfg = config(7);
        cfg.attention_chunk_size = Some(4);
        cfg.transformer_window = Some(3);
        let mut scope = MlxRequestScopeCore::with_cleanup_action(
            cfg,
            Box::new(move || {
                let call = cleanup_calls.get() + 1;
                cleanup_calls.set(call);
                if call == 1 {
                    Err(Error::Msg("injected cleanup failure".into()))
                } else {
                    cleanup_completed.set(true);
                    Ok(())
                }
            }),
        );
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());

        let mut request = GenerationRequest {
            prompt: "x".into(),
            width: 64,
            height: 64,
            count: 1,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut request).is_err());
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.leave_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.configure_decode(32, 4, config(7).geometry).is_err());
        assert!(scope.configure_attention(4).is_err());
        assert!(scope.materialize_transformer_window(0, 3).is_err());
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
        assert_eq!(calls.get(), 1, "hooks must not retry cleanup");

        drop(scope);
        assert_eq!(calls.get(), 2);
        assert!(
            completed.get(),
            "Drop must complete the pending cleanup retry"
        );
    }

    #[test]
    fn every_adopting_provider_constructs_the_shared_core() {
        for (name, source, config_marker) in [
            (
                "z-image",
                include_str!("../mlx-gen-z-image/src/memory_strategy.rs"),
                "ZImageTransformerConfig::turbo().n_layers",
            ),
            (
                "qwen-image",
                include_str!("../mlx-gen-qwen-image/src/memory_strategy.rs"),
                "QwenTransformerConfig::qwen_image_edit().num_layers",
            ),
            (
                "krea-base",
                include_str!("../mlx-gen-krea/src/block_memory_strategy.rs"),
                "Krea2Config::turbo().num_layers",
            ),
            (
                "krea-control",
                include_str!("../mlx-gen-krea/src/memory_strategy.rs"),
                "Krea2Config::turbo().num_layers",
            ),
            (
                "lens",
                include_str!("../mlx-gen-lens/src/memory_strategy.rs"),
                "GptOssConfig::lens().num_layers",
            ),
            (
                "mage",
                include_str!("../mlx-gen-mage/src/model.rs"),
                "MageFlowConfig::mage_flow().depth",
            ),
            (
                "sana",
                include_str!("../mlx-gen-sana/src/memory_strategy.rs"),
                "SanaTransformerConfig::sana_1600m().num_layers",
            ),
            // sc-15528. The marker is the DUAL-expert derivation: Bernini's rung-4 window covers
            // both experts in one global index space, so the count handed to this core is
            // `2 * num_layers`. A regression to one expert's depth would silently admit a window at
            // block 39 and reject one at block 40 — the low expert's first block.
            (
                "bernini",
                include_str!("../mlx-gen-bernini/src/memory_strategy.rs"),
                "WanModelConfig::wan22_t2v_14b().num_layers",
            ),
            (
                "ltx-video",
                include_str!("../mlx-gen-ltx/src/memory_strategy.rs"),
                "LtxConfig::video_only_defaults().num_layers",
            ),
        ] {
            // `begin_scope` is the shared admission skeleton that constructs the core (sc-27257).
            assert!(
                source.contains("request_scope::MlxRequestScopeCore::")
                    || source.contains("request_scope::begin_scope("),
                "{name} bypassed MlxRequestScopeCore"
            );
            assert!(
                !source.contains("struct ZImageMemoryScope")
                    && !source.contains("struct QwenMemoryScope")
                    && !source.contains("struct KreaMemoryScope")
                    && !source.contains("struct KreaControlMemoryScope")
                    && !source.contains("struct LensMemoryScope")
                    && !source.contains("struct MageMemoryScope"),
                "{name} retained a duplicate provider scope"
            );
            assert!(
                source.contains(config_marker),
                "{name} stopped deriving its block count from model config"
            );
        }
    }

    #[test]
    fn begin_refuses_a_rejected_safety_decision_without_building_a_config() {
        let built = Cell::new(false);
        let result = MlxRequestScopeCore::begin(
            MemorySafetyDecision::Reject {
                reason: "fixture: refused".into(),
            },
            MlxScopeCleanup::None,
            || {
                built.set(true);
                Ok(config(20))
            },
        );
        match result {
            Err(Error::Unsupported(reason)) => assert_eq!(reason, "fixture: refused"),
            Err(other) => panic!("expected the typed rejection, got {other:?}"),
            Ok(_) => panic!("a rejected safety decision must not open a scope"),
        }
        assert!(!built.get(), "configuration must not run after a rejection");
    }

    #[test]
    fn begin_scope_opens_the_configured_core_on_an_accepted_decision() {
        let mut scope = begin_scope(MemorySafetyDecision::Accept, MlxScopeCleanup::None, || {
            Ok(config(20))
        })
        .unwrap()
        .expect("an accepted decision opens a scope");
        let mut request = GenerationRequest {
            width: 64,
            height: 64,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        scope.finish(MemoryRunOutcome::Complete).unwrap();
    }
}
