//! The ONE iOS admission seam (epic 16774, E4/E5/E7) for every registered model and mode.
//!
//! A candidate names a registered memory-strategy provider, its load spec, the provider's own
//! `MemoryRunContext` (built by [`build_candidate`]) and the actual work. Admission:
//!
//! 1. validates the actual work against the context (geometry, batch, references, execution) and
//!    refuses a mode the provider does not serve ([`RefusalKind::Mode`]);
//! 2. computes the evidence key: hardware, provider, route, tier, path-free layout digest, mode,
//!    geometry, reference count, execution levers, token window — never prompt, seed or path;
//! 3. looks the key up in the single reviewed table ([`crate::evidence`]); no row refuses
//!    ([`RefusalKind::NoEvidence`]);
//! 4. checks the execution implements the selected ladder ([`RefusalKind::Strategy`]);
//! 5. runs the provider's REGISTERED safety check (`standard_memory_strategy_safety_check` plus the
//!    provider's route gate, E5) with the captured absolute footprint as the predicted peak; a
//!    rejection over the budget is [`RefusalKind::Capacity`], any other is `Strategy`;
//! 6. seals the source inventory and an exact work binding for the bound wrappers' lifetime check.
//!
//! [`capture_permit`] is the explicitly labelled, NON-circular evidence path: it performs 1, 2 and 4
//! but needs no row, and is only granted where every quantum of the model is bounded by live OS
//! headroom before its graph is constructed (`mlx_gen::ios_quantum::enabled()`: a physical iOS
//! device, or the CPU test fixture). A capture never changes admission: its output is a candidate
//! row a human reviews into the table.

use mlx_gen::gen_core::iris::depth::DepthRequest;
use mlx_gen::gen_core::{
    standard_memory_behavior_context, CfgBatching, Conditioning, ConditioningKind, Error,
    GenerationMemory, GenerationRequest, LoadSpec, MemoryBehaviorRoute, MemoryBudget,
    MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryRunContext, MemorySafetyDecision,
    MemorySelection, MemoryStrategySupport, Precision, ProviderRegistry, Quant, Result,
    SourceInventory, TransformRequest,
};
use sha2::{Digest, Sha256};

use crate::evidence::{self, EvidenceKey, EvidenceRow, EVIDENCE_TABLE_PATH};
use crate::refusal::{artifact, AdmissionError, AdmissionResult, RefusalKind};

/// The Iris restoration transform's memory provider id.
pub const IRIS_RESTORE_ID: &str = "iris_3b_restore";
/// The Iris depth task's memory provider id.
pub const IRIS_DEPTH_ID: &str = "iris_3b_depth";

/// The actual work a candidate will run.
// A handful of short-lived candidates per admission; boxing the request would only complicate
// every caller's pattern match.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum IosWork {
    Generate(GenerationRequest),
    Restore(TransformRequest),
    Depth(DepthRequest),
}

/// One caller-ordered candidate. Build with [`build_candidate`]; preference order stays the
/// caller's, never a hidden quality downgrade.
#[derive(Clone, Debug)]
pub struct IosCandidate {
    pub provider: String,
    pub spec: LoadSpec,
    pub context: MemoryRunContext,
    /// Execution levers the work runs with (`GenerationRequest::memory` for generation).
    pub execution: GenerationMemory,
    pub work: IosWork,
}

/// The tier label a spec loads at: `q4`/`q8` for a packed tier, else the dense precision
/// (`bf16`; any other precision gets its own label and so matches no reviewed row).
pub fn tier_label(spec: &LoadSpec) -> String {
    match spec.quantize {
        Some(Quant::Q4) => "q4".into(),
        Some(Quant::Q8) => "q8".into(),
        Some(other) => format!("{other:?}").to_ascii_lowercase(),
        None if spec.precision == Precision::Bf16 => "bf16".into(),
        None => format!("{:?}", spec.precision).to_ascii_lowercase(),
    }
}

fn numeric_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

fn generation_batch(req: &GenerationRequest) -> u32 {
    if req.prompt_batch.is_empty() {
        req.count
    } else {
        (req.prompt_batch.len() as u32).saturating_mul(req.count.max(1))
    }
}

/// The memory route a work item takes on `provider` with `spec`.
pub fn work_mode(provider: &str, spec: &LoadSpec, work: &IosWork) -> MemoryMode {
    match work {
        IosWork::Restore(_) => MemoryMode::Other("restoration".into()),
        IosWork::Depth(_) => MemoryMode::Other("depth".into()),
        IosWork::Generate(req) => {
            let refs = req.memory_reference_count();
            let mask = req
                .conditioning
                .iter()
                .any(|c| matches!(c, Conditioning::Mask { .. }));
            let control = req
                .conditioning
                .iter()
                .any(|c| matches!(c, Conditioning::Control { .. }));
            if control {
                MemoryMode::Other("control".into())
            } else if refs > 0 && mask {
                MemoryMode::Edit
            } else if refs > 0 && provider == crate::sdxl::SDXL_ID && spec.ip_adapter.is_some() {
                MemoryMode::Other("ip_adapter".into())
            } else if refs > 0 {
                MemoryMode::ImageToImage
            } else {
                MemoryMode::TextToImage
            }
        }
    }
}

fn work_geometry(work: &IosWork) -> MemoryGeometry {
    match work {
        IosWork::Generate(req) => MemoryGeometry {
            width: req.width,
            height: req.height,
            batch: generation_batch(req),
            frames: req.frames.unwrap_or(1),
            reference_count: req.memory_reference_count(),
        },
        IosWork::Restore(req) => MemoryGeometry {
            width: req.image.width,
            height: req.image.height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        IosWork::Depth(req) => MemoryGeometry {
            width: req.image.width,
            height: req.image.height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
    }
}

/// Build a candidate from the provider's own contract: the shared weights-free context for
/// `selection`'s strategy, with `selection` itself, the work's real geometry and route, and the
/// provider's ladder execution (plus SDXL's sequential CFG on iOS) written onto the request.
pub fn build_candidate(
    registry: &ProviderRegistry,
    provider: &str,
    spec: LoadSpec,
    selection: Option<MemorySelection>,
    strategy: mlx_gen::gen_core::MemoryStrategy,
    mut work: IosWork,
) -> AdmissionResult<IosCandidate> {
    let contract = registry
        .memory_strategy_contract(provider, &spec)?
        .ok_or_else(|| {
            AdmissionError::refused(
                RefusalKind::Strategy,
                format!("{provider}: no memory contract"),
            )
        })?;
    // A provider publishes no calibration identity when the requested tier is not the tier the
    // artifact actually stores (SANA: a dense snapshot requested as q4/q8 would be a load-time
    // requantization no capture measured). That is an ARTIFACT refusal, typed, before any key or
    // price is computed for the wrong tier.
    if contract.calibration.is_none() {
        return Err(AdmissionError::refused(
            RefusalKind::Artifact,
            format!(
                "{provider}: the requested tier {} does not match the artifact's stored tier \
                 (the provider publishes no calibration identity for this load)",
                tier_label(&spec)
            ),
        ));
    }
    let geometry = work_geometry(&work);
    let mut context = standard_memory_behavior_context(
        &contract,
        strategy,
        numeric_tier(&spec),
        MemoryBehaviorRoute {
            mode: work_mode(provider, &spec, &work),
            reference_count: geometry.reference_count,
            use_pid: false,
            has_phases: matches!(work, IosWork::Generate(_)),
            overlay: None,
        },
    )?;
    if let Some(selection) = selection {
        context.selection = selection;
    }
    context.geometry = geometry;
    context.load_shape = spec.load_shape;
    let mut execution = contract
        .generation_memory(&context.selection)
        .unwrap_or_default();
    if provider == crate::sdxl::SDXL_ID {
        execution.cfg_batching = Some(CfgBatching::Sequential);
    }
    if let IosWork::Generate(req) = &mut work {
        req.memory = Some(execution);
    }
    Ok(IosCandidate {
        provider: provider.into(),
        spec,
        context,
        execution,
        work,
    })
}

/// Canonical memory-relevant execution identity: strategy, every ladder parameter, every request
/// execution lever, and the task-specific sizing knobs.
pub fn execution_key(candidate: &IosCandidate) -> String {
    let task = match &candidate.work {
        IosWork::Generate(_) => String::new(),
        IosWork::Restore(r) => format!(
            "|target={:?}|sizing={:?}|steps={:?}|color_fix={:?}",
            r.target, r.input_sizing, r.steps, r.color_fix
        ),
        IosWork::Depth(r) => format!("|resolution={:?}", r.resolution),
    };
    format!(
        "{:?}|{:?}|{:?}{task}",
        candidate.context.selection.strategy,
        candidate.context.selection.parameters,
        candidate.execution
    )
}

/// Encoded text window. SDXL's long-prompt path encodes `n` CLIP windows of 77; every other iOS
/// provider pads to a fixed, artifact-determined window (0 here, covered by the layout digest).
pub fn token_window(candidate: &IosCandidate) -> Result<u32> {
    match (&candidate.work, candidate.provider.as_str()) {
        (IosWork::Generate(req), crate::sdxl::SDXL_ID) => {
            // The negative row is always counted (an empty one is one window): conservative when
            // CFG is off, and identical for a capture and a product request.
            let windows = mlx_gen_sdxl::conditioning_window_count(
                &candidate.spec,
                &req.prompt,
                Some(req.negative_prompt.as_deref().unwrap_or("")),
            )?;
            Ok(windows.saturating_mul(77))
        }
        _ => Ok(0),
    }
}

fn route_of(candidate: &IosCandidate) -> AdmissionResult<String> {
    if candidate.provider == crate::sdxl::SDXL_ID {
        return candidate
            .spec
            .resolved_route
            .clone()
            .filter(|r| crate::sdxl::SDXL_FAMILY_ROUTES.contains(&r.as_str()))
            .ok_or_else(|| {
                AdmissionError::refused(RefusalKind::Artifact, "unknown SDXL family route")
            });
    }
    match &candidate.spec.resolved_route {
        Some(route) if route != &candidate.provider => Err(AdmissionError::refused(
            RefusalKind::Artifact,
            format!("{}: foreign route {route}", candidate.provider),
        )),
        _ => Ok(candidate.provider.clone()),
    }
}

fn mismatch(what: &str) -> AdmissionError {
    AdmissionError::Failed(Error::Unsupported(format!(
        "iOS actual work disagrees with its admission context: {what}"
    )))
}

/// Validate the actual work against its context and the provider's registered surface.
pub fn validate_actual(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
) -> AdmissionResult<()> {
    let c = &candidate.context;
    let provider = candidate.provider.as_str();
    if work_mode(provider, &candidate.spec, &candidate.work) != c.mode {
        return Err(mismatch("mode"));
    }
    if work_geometry(&candidate.work) != c.geometry
        || c.has_reference != (c.geometry.reference_count > 0)
        || c.use_pid
    {
        return Err(mismatch("geometry/batch/references"));
    }
    match &candidate.work {
        IosWork::Generate(req) => {
            let descriptor = registry
                .generators()
                .map(|r| (r.descriptor)())
                .find(|d| d.id == provider)
                .ok_or_else(|| {
                    AdmissionError::refused(
                        RefusalKind::Mode,
                        format!("{provider} is not a registered iOS generator"),
                    )
                })?;
            let caps = &descriptor.capabilities;
            for conditioning in &req.conditioning {
                let kind = match conditioning {
                    Conditioning::Reference { .. } => ConditioningKind::Reference,
                    Conditioning::Mask { .. } => ConditioningKind::Mask,
                    Conditioning::Control { .. } => ConditioningKind::Control,
                    other => {
                        return Err(AdmissionError::refused(
                            RefusalKind::Mode,
                            format!("{provider}: conditioning {other:?} has no iOS carrier"),
                        ))
                    }
                };
                if !caps.conditioning.contains(&kind) {
                    return Err(AdmissionError::refused(
                        RefusalKind::Mode,
                        format!("{provider} does not serve {kind:?}"),
                    ));
                }
            }
            if !req.prompt_batch.is_empty() && !caps.supports_prompt_batch {
                return Err(AdmissionError::refused(
                    RefusalKind::Mode,
                    format!("{provider} does not serve prompt batches"),
                ));
            }
            if req.memory.unwrap_or_default() != candidate.execution || req.use_pid {
                return Err(mismatch("request execution"));
            }
            caps.validate_request(provider, req)?;
        }
        IosWork::Restore(_) => {
            if provider != IRIS_RESTORE_ID
                || !registry
                    .transforms()
                    .any(|r| (r.descriptor)().id == provider)
            {
                return Err(AdmissionError::refused(
                    RefusalKind::Mode,
                    format!("{provider} is not a registered iOS restoration transform"),
                ));
            }
        }
        IosWork::Depth(req) => {
            if provider != IRIS_DEPTH_ID {
                return Err(AdmissionError::refused(
                    RefusalKind::Mode,
                    format!("{provider} is not the registered iOS depth task"),
                ));
            }
            mlx_gen::gen_core::iris::depth::validate_depth_request(req)?;
        }
    }
    Ok(())
}

/// Exact binding of the work for a bound wrapper's lifetime check: everything but the seed and the
/// cancellation handle. This is NOT the evidence key (which ignores prompt/pixels).
pub fn work_binding(work: &IosWork) -> [u8; 32] {
    let text = match work {
        IosWork::Generate(req) => {
            let mut r = req.clone();
            r.cancel = Default::default();
            r.seed = None;
            format!("generate:{r:?}")
        }
        IosWork::Restore(req) => {
            let mut r = req.clone();
            r.cancel = Default::default();
            r.seed = None;
            format!("restore:{r:?}")
        }
        IosWork::Depth(req) => {
            let mut r = req.clone();
            r.cancel = Default::default();
            format!("depth:{r:?}")
        }
    };
    Sha256::digest(text.as_bytes()).into()
}

/// Total absolute-process capacity with no speculative reclaim credit. A captured process peak is
/// converted to incremental demand exactly once at the provider check.
pub fn capacity_budget(cap: u64, committed: u64, live_incremental: Option<u64>) -> MemoryBudget {
    MemoryBudget {
        total_bytes: live_incremental.map_or(cap, |available| {
            cap.min(committed.saturating_add(available))
        }),
        committed_bytes: committed,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

/// The evidence key of a candidate on `machine`. Opens tensor headers and small configs only.
pub fn evidence_key(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    machine: &str,
) -> AdmissionResult<EvidenceKey> {
    validate_actual(registry, candidate)?;
    let route = route_of(candidate)?;
    key_unchecked(candidate, machine, route)
}

fn key_unchecked(
    candidate: &IosCandidate,
    machine: &str,
    route: String,
) -> AdmissionResult<EvidenceKey> {
    let layout = if candidate.provider.starts_with("sana_") {
        artifact(crate::admission::sana_snapshot_profile(&candidate.spec))?.layout_sha256
    } else {
        artifact(crate::layout::artifact_layout_digest(&candidate.spec))?
    };
    let c = &candidate.context;
    Ok(EvidenceKey {
        machine: machine.into(),
        provider: candidate.provider.clone(),
        route,
        tier: tier_label(&candidate.spec),
        layout_sha256: crate::layout::hex(&layout),
        mode: c.mode.as_key().into(),
        width: c.geometry.width,
        height: c.geometry.height,
        batch: c.geometry.batch,
        reference_count: c.geometry.reference_count,
        execution: execution_key(candidate),
        token_window: token_window(candidate)?,
    })
}

/// A sealed admission: the source inventory, the admitted context and the exact work binding.
#[derive(Debug)]
pub struct Admission {
    pub(crate) index: usize,
    pub(crate) provider: String,
    pub(crate) footprint_bytes: u64,
    pub(crate) spec: LoadSpec,
    pub(crate) context: MemoryRunContext,
    pub(crate) binding: [u8; 32],
    pub(crate) key: EvidenceKey,
    /// `true` only for a [`capture_permit`]: execution bounded per quantum, no evidence row.
    pub(crate) capture: bool,
}

impl Admission {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn provider(&self) -> &str {
        &self.provider
    }
    /// The captured footprint this admission was granted on (0 for a capture permit).
    pub fn footprint_bytes(&self) -> u64 {
        self.footprint_bytes
    }
    pub fn footprint_mib(&self) -> u32 {
        u32::try_from(self.footprint_bytes.div_ceil(1 << 20)).unwrap_or(u32::MAX)
    }
    pub fn context(&self) -> &MemoryRunContext {
        &self.context
    }
    pub fn evidence_key(&self) -> &EvidenceKey {
        &self.key
    }
    /// Whether this is an evidence-collection capture permit rather than a product admission.
    pub fn is_capture(&self) -> bool {
        self.capture
    }
}

/// Steps 3–6 for one candidate, under its own source inventory.
fn admit_one(
    registry: &ProviderRegistry,
    index: usize,
    candidate: &IosCandidate,
    machine: &str,
    budget: MemoryBudget,
    rows: Option<&[EvidenceRow]>,
) -> AdmissionResult<Admission> {
    validate_actual(registry, candidate)?;
    let route = route_of(candidate)?;
    if let Some(rows) = rows {
        // Cheap pre-check: refuse before opening any source when no row can match.
        if !rows.iter().any(|r| {
            r.key.machine == machine && r.key.provider == candidate.provider && r.key.route == route
        }) {
            return Err(AdmissionError::refused(
                RefusalKind::NoEvidence,
                format!(
                    "{}: no reviewed device capture for route {route} on {machine} in \
                     {EVIDENCE_TABLE_PATH} (source unopened)",
                    candidate.provider
                ),
            ));
        }
    }
    let guard = SourceInventory::capture(&candidate.spec)?;
    guard.read_unchanged(|| -> AdmissionResult<Admission> {
        let key = key_unchecked(candidate, machine, route.clone())?;
        let footprint_bytes = match rows {
            None => 0,
            Some(rows) => {
                let row = evidence::lookup(rows, &key).ok_or_else(|| {
                    AdmissionError::refused(
                        RefusalKind::NoEvidence,
                        format!("no reviewed device capture in {EVIDENCE_TABLE_PATH} for {key:?}"),
                    )
                })?;
                row.footprint_bytes
            }
        };
        let contract = registry
            .memory_strategy_contract(&candidate.provider, &candidate.spec)?
            .ok_or_else(|| {
                AdmissionError::refused(RefusalKind::Strategy, "provider has no memory contract")
            })?;
        let registration = registry
            .memory_strategy_registrations()
            .find(|r| r.provider_id == candidate.provider)
            .ok_or_else(|| {
                AdmissionError::refused(
                    RefusalKind::Strategy,
                    "provider registers no memory safety check",
                )
            })?;
        let strategy = candidate.context.selection.strategy;
        if contract
            .capability(strategy)
            .is_none_or(|cap| !matches!(cap.support, MemoryStrategySupport::Implemented))
        {
            return Err(AdmissionError::refused(
                RefusalKind::Strategy,
                format!(
                    "{}: {strategy:?} is not implemented on this load",
                    candidate.provider
                ),
            ));
        }
        let expected = contract
            .generation_memory(&candidate.context.selection)
            .unwrap_or_default();
        let actual = candidate.execution;
        if (
            actual.stage_residency,
            actual.tile_vae_decode,
            actual.chunk_attention,
            actual.stream_transformer_blocks,
            actual.decode_tile_edge,
            actual.decode_overlap,
            actual.attention_chunk_size,
            actual.transformer_window_size,
            actual.transformer_window_component,
        ) != (
            expected.stage_residency,
            expected.tile_vae_decode,
            expected.chunk_attention,
            expected.stream_transformer_blocks,
            expected.decode_tile_edge,
            expected.decode_overlap,
            expected.attention_chunk_size,
            expected.transformer_window_size,
            expected.transformer_window_component,
        ) {
            return Err(AdmissionError::refused(
                RefusalKind::Strategy,
                "actual execution does not implement the selected ladder",
            ));
        }
        let mut context = candidate.context.clone();
        if rows.is_some() {
            context.budget = budget;
            context.predicted_peak_bytes = footprint_bytes.saturating_sub(budget.committed_bytes);
            context.evidence_revision = format!("ios-device-capture:{}", key.layout_sha256);
            if let MemorySafetyDecision::Reject { reason } =
                (registration.safety_check)(&candidate.spec, &contract, &context)
            {
                let kind = if context.predicted_peak_bytes > budget.effective_bytes() {
                    RefusalKind::Capacity
                } else {
                    RefusalKind::Strategy
                };
                return Err(AdmissionError::refused(kind, reason));
            }
        } else {
            context.evidence_revision = "ios-capture-permit:no-evidence-row".into();
        }
        let mut spec = candidate.spec.clone();
        spec.bind_source_inventory(guard.clone())?;
        Ok(Admission {
            index,
            provider: candidate.provider.clone(),
            footprint_bytes,
            spec,
            context,
            binding: work_binding(&candidate.work),
            key,
            capture: rows.is_none(),
        })
    })
}

/// First (highest caller preference) candidate admitted against `rows` under `budget`.
///
/// Aggregation: if every candidate FAILED the first failure is returned; otherwise the refusal
/// carries the kind of the first refused candidate and lists every candidate's outcome.
pub fn admit_with_rows(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    rows: &[EvidenceRow],
) -> AdmissionResult<Admission> {
    let mut first_refusal: Option<RefusalKind> = None;
    let mut first_failure: Option<Error> = None;
    let mut reasons = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        match admit_one(registry, index, candidate, machine, budget, Some(rows)) {
            Ok(admitted) => return Ok(admitted),
            Err(AdmissionError::Refused(r)) => {
                first_refusal.get_or_insert(r.kind);
                reasons.push(format!(
                    "candidate {index}: {} ({})",
                    r.reason,
                    r.kind.label()
                ));
            }
            Err(AdmissionError::Failed(e)) => {
                reasons.push(format!("candidate {index}: failed: {e}"));
                first_failure.get_or_insert(e);
            }
        }
    }
    match (first_refusal, first_failure) {
        (Some(kind), _) => Err(AdmissionError::refused(
            kind,
            format!(
                "iOS admission on {machine}: no candidate admitted; {}",
                reasons.join("; ")
            ),
        )),
        (None, Some(error)) => Err(AdmissionError::Failed(error)),
        (None, None) => Err(AdmissionError::refused(
            RefusalKind::NoEvidence,
            "no candidates offered",
        )),
    }
}

/// [`admit_with_rows`] against the compiled-in reviewed table. A malformed table admits nothing.
pub fn admit(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
) -> AdmissionResult<Admission> {
    let rows = evidence::rows().map_err(|e| AdmissionError::refused(RefusalKind::NoEvidence, e))?;
    admit_with_rows(registry, candidates, machine, budget, rows)
}

/// Explicitly-labelled capture permit. NOT admission: no evidence row is required (that would be
/// circular) and none is created. Granted only where `mlx_gen::ios_quantum` bounds every quantum of
/// this model by live OS headroom BEFORE its graph is constructed; a refusal at quantum k of size N
/// is the "next size refused before jetsam" evidence. Never raises the cap.
pub fn capture_permit(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    machine: &str,
) -> AdmissionResult<Admission> {
    if !mlx_gen::ios_quantum::enabled() {
        return Err(AdmissionError::Failed(Error::Unsupported(
            "iOS capture requires bounded per-quantum execution (a physical iOS device); \
             no weights loaded"
                .into(),
        )));
    }
    admit_one(
        registry,
        0,
        candidate,
        machine,
        MemoryBudget {
            total_bytes: 0,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        },
        None,
    )
}

impl Admission {
    fn check_kind(&self, expected: &str) -> Result<()> {
        let ok = match expected {
            "generate" => {
                self.context.mode.as_key() != "restoration" && self.context.mode.as_key() != "depth"
            }
            other => self.context.mode.as_key() == other,
        };
        if ok {
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "{}: admitted work is not a {expected} load",
                self.provider
            )))
        }
    }

    /// Load the admitted generator. `capacity` is rechecked before load and before every call.
    pub fn load_generator(
        self,
        registry: &ProviderRegistry,
        capacity: impl Fn(u64) -> Result<()> + 'static,
    ) -> Result<crate::bound::BoundGenerator> {
        self.check_kind("generate")?;
        let provider = self.provider.clone();
        self.load_generator_using(|spec| registry.load(&provider, spec), Box::new(capacity))
    }

    pub(crate) fn load_generator_using(
        self,
        load: impl FnOnce(&LoadSpec) -> Result<Box<dyn mlx_gen::gen_core::Generator>>,
        capacity: Box<dyn Fn(u64) -> Result<()>>,
    ) -> Result<crate::bound::BoundGenerator> {
        capacity(self.footprint_bytes)?;
        let inner = self
            .spec
            .read_prepared_files_unchanged(|| load(&self.spec))?;
        Ok(crate::bound::BoundGenerator::new(
            inner,
            self.spec,
            self.context,
            self.binding,
            self.footprint_bytes,
            capacity,
        ))
    }

    /// Load the admitted Iris restoration transform.
    pub fn load_restorer(
        self,
        registry: &ProviderRegistry,
        capacity: impl Fn(u64) -> Result<()> + 'static,
    ) -> Result<crate::bound::BoundTransform> {
        self.check_kind("restoration")?;
        capacity(self.footprint_bytes)?;
        let provider = self.provider.clone();
        let inner = self
            .spec
            .read_prepared_files_unchanged(|| registry.load_transform(&provider, &self.spec))?;
        Ok(crate::bound::BoundTransform::new(
            inner,
            self.spec,
            self.binding,
            self.footprint_bytes,
            Box::new(capacity),
        ))
    }

    /// Load the admitted Iris depth task.
    pub fn load_depth(
        self,
        capacity: impl Fn(u64) -> Result<()> + 'static,
    ) -> Result<crate::bound::BoundDepth> {
        self.check_kind("depth")?;
        capacity(self.footprint_bytes)?;
        let inner = self.spec.read_prepared_files_unchanged(|| {
            mlx_gen_iris::depth::load_depth(&self.spec).map_err(Error::from)
        })?;
        Ok(crate::bound::BoundDepth::new(
            Box::new(inner),
            self.spec,
            self.binding,
            self.footprint_bytes,
            Box::new(capacity),
        ))
    }
}

#[cfg(test)]
#[path = "ios_admission_tests.rs"]
mod tests;
