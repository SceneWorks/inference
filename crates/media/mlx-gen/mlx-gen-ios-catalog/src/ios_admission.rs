//! The ONE iOS admission seam (epic 16774, E4/E5/E7) for every registered model and mode.
//!
//! A candidate names a registered memory-strategy provider, its load spec, the provider's own
//! `MemoryRunContext` (built by [`build_candidate`]) and the actual work. Admission:
//!
//! 1. validates the actual work against the context (geometry, batch, references, execution) and
//!    refuses a mode the provider does not serve ([`RefusalKind::Mode`]);
//! 2. computes the evidence key: hardware, provider, route, tier, path-free layout digest, mode,
//!    geometry, reference count, execution levers, token window — never prompt, seed or path;
//! 3. prices the request from the single reviewed ANCHOR table ([`crate::evidence`]) by the anchor
//!    law ([`crate::anchor`]): at an anchor's own key, its measured demand; otherwise the demand
//!    DERIVED from the family's anchors (the optimized anchor, or a same-regime resident one), after
//!    the two anchors' self-check. Derivation consumes ONLY a registered provider bounder for the
//!    execution facts stated at the boundary ([`admit_with_boundary`]), through the certified law;
//!    missing facts, bounders or certificates in the anchors' counter domain refuse exactly.
//!    Alternatively, an explicitly approved [`crate::engineering`] policy uses measured process
//!    endpoints and source-generated phase pressure, labelled [`AdmissionBasis::EngineeringEstimate`].
//!    It preserves typed resident refusal and records separate process, HOST/residual and source-
//!    preparation assumptions; it supplies no mathematical physical/native certificate.
//!    No compatible family, an underivable request, or a self-check
//!    failure refuses ([`RefusalKind::NoEvidence`]); a measured resident refusal at its own key is
//!    [`RefusalKind::Capacity`];
//! 4. checks the execution implements the selected ladder ([`RefusalKind::Strategy`]);
//! 5. runs the provider's REGISTERED safety check (`standard_memory_strategy_safety_check` plus the
//!    provider's route gate, E5) with the demand plus the anchor's measured margin as the predicted
//!    peak; a rejection over the budget is [`RefusalKind::Capacity`], any other is `Strategy`;
//! 6. seals the source inventory and an exact work binding for the bound wrappers' lifetime check.
//!
//! `capture_permit` (feature `device-capture`) is the explicitly labelled, NON-circular evidence path: it performs 1, 2 and 4
//! but needs no row, and is only granted where every quantum of the model is bounded by live OS
//! headroom before its graph is constructed (`mlx_gen::ios_quantum::enabled()`: a physical iOS
//! device, or the CPU test fixture). A capture never changes admission: its output is a candidate
//! row a human reviews into the table.

use mlx_gen::gen_core::iris::depth::DepthRequest;
use mlx_gen::gen_core::{
    standard_memory_behavior_context, CfgBatching, Conditioning, ConditioningKind, Error,
    GenerationMemory, GenerationRequest, LoadSpec, MemoryBehaviorRoute, MemoryBudget,
    MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryProviderContract, MemoryRunContext,
    MemorySafetyDecision, MemorySelection, MemoryStrategySupport, Precision, ProviderRegistry,
    Quant, Result, SourceInventory, TransformRequest,
};
use mlx_gen::ios_pricing::{
    IosBoundRequest, IosCertificatePremise, IosCertifiedBounds, IosExecutionFacts, IosHostGrowth,
    IosMemoryBaseline, IosMemoryCoordinate, IosMemoryDomain, IosNativeToProcessModel, IosPricing,
    IosPricingRequest, IosReplay, IosWindowBound, NATIVE_TO_PROCESS_HOST_PREMISE_ID,
    NATIVE_TO_PROCESS_PREMISE_ID,
};

use crate::anchor;
use crate::evidence::{self, AnchorKind, AnchorRow, EvidenceKey, EVIDENCE_TABLE_PATH};
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

fn generation_batch(req: &GenerationRequest) -> AdmissionResult<u32> {
    let rows = if req.prompt_batch.is_empty() {
        1
    } else {
        u32::try_from(req.prompt_batch.len()).map_err(|_| {
            AdmissionError::refused(
                RefusalKind::Mode,
                "prompt row count exceeds request geometry domain",
            )
        })?
    };
    rows.checked_mul(req.count)
        .filter(|count| *count > 0)
        .ok_or_else(|| {
            AdmissionError::refused(
                RefusalKind::Mode,
                "serial count × prompt rows must be nonzero and fit request geometry",
            )
        })
}

/// The memory route a work item takes on `provider` with `spec`.
/// SDXL has two source stages: resolve/validate its ACTUAL sampler first (including early
/// accel+Reference and mask/control refusals), then apply the runtime's configured-IP predicate.
/// A predicate excluding IP is not permission to invent an accepted INIT route after refusal.
pub fn work_mode(provider: &str, spec: &LoadSpec, work: &IosWork) -> AdmissionResult<MemoryMode> {
    Ok(match work {
        IosWork::Restore(_) => MemoryMode::Other("restoration".into()),
        IosWork::Depth(_) => MemoryMode::Other("depth".into()),
        IosWork::Generate(req) => {
            if let Some(role) = req.conditioning.iter().find(|role| {
                !matches!(
                    role,
                    Conditioning::Reference { .. }
                        | Conditioning::Mask { .. }
                        | Conditioning::Control { .. }
                )
            }) {
                return Err(pricing_error(Error::Unsupported(format!(
                    "iOS role partition has no source classification for {:?}",
                    role.kind()
                ))));
            }
            let refs = req
                .conditioning
                .iter()
                .filter(|role| matches!(role, Conditioning::Reference { .. }))
                .count();
            let masks = req
                .conditioning
                .iter()
                .filter(|role| matches!(role, Conditioning::Mask { .. }))
                .count();
            let controls = req
                .conditioning
                .iter()
                .filter(|role| matches!(role, Conditioning::Control { .. }))
                .count();
            let mask_count = u32::try_from(masks).map_err(|_| {
                pricing_error(Error::Unsupported(
                    "mask role count exceeds geometry domain".into(),
                ))
            })?;
            let control_count = u32::try_from(controls).map_err(|_| {
                pricing_error(Error::Unsupported(
                    "control role count exceeds geometry domain".into(),
                ))
            })?;
            let partitioned = crate::pricing::reference_role_count(
                req.memory_reference_count(),
                mask_count,
                control_count,
            )
            .map_err(pricing_error)?;
            if refs != partitioned as usize {
                return Err(pricing_error(Error::Unsupported(
                    "actual Reference roles differ from aggregate partition".into(),
                )));
            }
            if crate::zimage::is_z_image(provider) {
                mlx_gen_z_image::ios_pricing::validate_pricing_roles(
                    provider,
                    req.memory_reference_count(),
                    partitioned,
                    mask_count,
                    control_count,
                )
                .map_err(pricing_error)?;
            }
            let mask = masks != 0;
            let control = controls != 0;
            // Only SDXL owns this resolver/predicate. Other providers retain their own actual
            // generation/task semantics; do not reinterpret Iris or Sprint controls as SDXL.
            let ip = if provider == crate::sdxl::SDXL_ID {
                let sampling = mlx_gen_sdxl::sampling::ResolvedSampling::resolve(req)
                    .map_err(pricing_error)?;
                mlx_gen_sdxl::model::uses_ip_reference(
                    spec.ip_adapter.is_some(),
                    refs != 0,
                    mask,
                    controls,
                    sampling.route().is_acceleration(),
                )
            } else {
                false
            };
            if control {
                MemoryMode::Other("control".into())
            } else if refs > 0 && mask {
                MemoryMode::Edit
            } else if ip {
                MemoryMode::Other("ip_adapter".into())
            } else if refs > 0 {
                MemoryMode::ImageToImage
            } else {
                MemoryMode::TextToImage
            }
        }
    })
}

fn work_geometry(work: &IosWork) -> AdmissionResult<MemoryGeometry> {
    Ok(match work {
        IosWork::Generate(req) => MemoryGeometry {
            width: req.width,
            height: req.height,
            batch: generation_batch(req)?,
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
    })
}

/// Select an ACTUALLY legal configured form for a requested rung. Only an existing provider's
/// dedicated check and pure selection-space incompatibility are skippable; contract/config/I/O
/// failures propagate. This does not change ordinary catalog defaults or caller creative controls.
pub fn configured_load_form(
    registry: &ProviderRegistry,
    provider: &str,
    base: &LoadSpec,
    strategy: mlx_gen::gen_core::MemoryStrategy,
) -> AdmissionResult<LoadSpec> {
    use evidence::AnchorLoadForm;
    use mlx_gen::gen_core::MemoryStrategy;
    let forms: &[AnchorLoadForm] = if strategy == MemoryStrategy::Resident {
        &[AnchorLoadForm::ResidentEager]
    } else {
        &[
            AnchorLoadForm::SequentialDeferred,
            AnchorLoadForm::SequentialEager,
            AnchorLoadForm::ResidentDeferred,
            AnchorLoadForm::ResidentEager,
        ]
    };
    for form in forms {
        let mut spec = base.clone();
        form.apply(&mut spec);
        if provider == "iris_3b" {
            // EXACT same metadata-only predicate called by the real Iris loader and contract.
            match mlx_gen_iris::memory_strategy::refuse_streamed_adapters(provider, &spec) {
                Ok(()) => {}
                Err(Error::Unsupported(_)) => continue,
                Err(error) => return Err(AdmissionError::Failed(error)),
            }
        }
        let contract = registry
            .memory_strategy_contract(provider, &spec)?
            .ok_or_else(|| {
                AdmissionError::refused(RefusalKind::Strategy, "provider has no memory contract")
            })?;
        if contract
            .capability(strategy)
            .is_none_or(|cap| cap.support != MemoryStrategySupport::Implemented)
        {
            continue;
        }
        match contract.selection_space(strategy, numeric_tier(&spec), false) {
            Ok(space) if !space.is_empty() => return Ok(spec),
            Ok(_) | Err(Error::Unsupported(_)) => continue,
            Err(error) => return Err(AdmissionError::Failed(error)),
        }
    }
    Err(AdmissionError::refused(
        RefusalKind::Strategy,
        format!("{provider}: no legal configured load form for {strategy:?}"),
    ))
}

/// Highest legal configured endpoint. This is metadata enumeration, not extra physical captures.
pub fn configured_endpoint(
    registry: &ProviderRegistry,
    provider: &str,
    base: &LoadSpec,
    endpoint: AnchorKind,
) -> AdmissionResult<(LoadSpec, mlx_gen::gen_core::MemoryStrategy)> {
    use mlx_gen::gen_core::MemoryStrategy;
    if endpoint == AnchorKind::Resident {
        return configured_load_form(registry, provider, base, MemoryStrategy::Resident)
            .map(|spec| (spec, MemoryStrategy::Resident));
    }
    for strategy in MemoryStrategy::ALL
        .into_iter()
        .rev()
        .filter(|strategy| strategy.is_optimized())
    {
        match configured_load_form(registry, provider, base, strategy) {
            Ok(spec) => return Ok((spec, strategy)),
            Err(AdmissionError::Refused(refusal)) if refusal.kind == RefusalKind::Strategy => {
                continue
            }
            Err(error) => return Err(error),
        }
    }
    Err(AdmissionError::refused(
        RefusalKind::Strategy,
        format!("{provider}: no legal configured optimized endpoint"),
    ))
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
    // Z-Image Turbo's artifact policy (stored tier, inventory, the control route's checkpoint) is
    // checked from small configs before its contract reads any tensor header.
    if crate::zimage::is_z_image(provider) {
        artifact(crate::zimage::check_artifact(provider, &spec))?;
    }
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
    let geometry = work_geometry(&work)?;
    let mut context = standard_memory_behavior_context(
        &contract,
        strategy,
        numeric_tier(&spec),
        MemoryBehaviorRoute {
            mode: work_mode(provider, &spec, &work)?,
            reference_count: geometry.reference_count,
            use_pid: false,
            has_phases: matches!(work, IosWork::Generate(_)),
            overlay: None,
        },
    )?;
    // A caller-chosen selection (a campaign cell's) must be one the provider's own contract
    // declares and can execute: a typed strategy refusal otherwise, before any price or key.
    if let Some(selection) = selection {
        contract.validate_selection(&selection).map_err(|e| {
            AdmissionError::refused(
                RefusalKind::Strategy,
                format!(
                    "{provider}: selection {:?} is not executable: {e}",
                    selection.parameters
                ),
            )
        })?;
        if selection.strategy != strategy {
            return Err(AdmissionError::refused(
                RefusalKind::Strategy,
                format!(
                    "{provider}: selection is for {:?}, the request for {strategy:?}",
                    selection.strategy
                ),
            ));
        }
        context.selection = selection;
    }
    context.geometry = geometry;
    context.load_shape = spec.load_shape;
    let mut execution = execution_for(provider, &contract, &context.selection);
    if let IosWork::Generate(req) = &mut work {
        if provider == crate::sdxl::SDXL_ID {
            // Keep the existing iOS sequential default, but never erase an explicit caller choice.
            if let Some(batching) = req.memory.and_then(|memory| memory.cfg_batching) {
                execution.cfg_batching = Some(batching);
            }
        }
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

/// The execution levers a selection runs with on iOS: the provider's ladder execution, plus SDXL's
/// sequential CFG.
pub fn execution_for(
    provider: &str,
    contract: &MemoryProviderContract,
    selection: &MemorySelection,
) -> GenerationMemory {
    let mut execution = contract.generation_memory(selection).unwrap_or_default();
    if provider == crate::sdxl::SDXL_ID {
        execution.cfg_batching = Some(CfgBatching::Sequential);
    }
    execution
}

/// Canonical memory-relevant execution identity: strategy, every ladder parameter, every request
/// execution lever, and the task-specific sizing knobs.
pub fn execution_key(candidate: &IosCandidate) -> AdmissionResult<String> {
    let task = match &candidate.work {
        IosWork::Generate(request) => {
            let sampling =
                crate::sampling::resolve_for_spec(&candidate.provider, &candidate.spec, request)
                    .map_err(pricing_error)?;
            format!("|recipe={}", sampling.token())
        }
        IosWork::Restore(r) => {
            let recipe =
                crate::sampling::resolve_transform(&candidate.provider, &candidate.spec, r)
                    .map_err(pricing_error)?;
            format!(
                "|target={:?}|sizing={:?}|steps={:?}|color_fix={:?}|recipe={}",
                r.target,
                r.input_sizing,
                r.steps,
                r.color_fix,
                recipe.token()
            )
        }
        IosWork::Depth(r) => {
            let recipe = crate::sampling::resolve_depth(&candidate.provider, &candidate.spec, r)
                .map_err(pricing_error)?;
            format!("|resolution={:?}|recipe={}", r.resolution, recipe.token())
        }
    };
    Ok(format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}{task}",
        candidate.spec.offload_policy,
        candidate.spec.load_shape,
        candidate.context.selection.strategy,
        candidate.context.selection.parameters,
        candidate.execution
    ))
}

/// Describe the SDK's actually selected SANA memory program. Identity only:
/// this does not configure a scope, grant credit, or change an ordinary key.
/// A guarded caller reuses its SAME source-capacity check for the new String.
pub fn bind_sana_retirement_key(
    key: &mut EvidenceKey,
    candidate: &IosCandidate,
    requirement: mlx_gen::ios_pricing::IosNativeRetirementExecution,
    next: Option<KeyCapacity<'_>>,
) -> AdmissionResult<()> {
    if !matches!(
        candidate.provider.as_str(),
        "sana_1600m" | "sana_sprint_1600m"
    ) {
        return Ok(());
    }
    if key.provider != candidate.provider {
        return Err(mismatch("retirement key provider"));
    }
    let suffix = mlx_gen_sana::ios_pricing::NATIVE_RETIREMENT_EXECUTION_SUFFIX;
    let joined = requirement
        == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
        && matches!(&candidate.work,IosWork::Generate(request) if request.memory.is_some());
    if !joined {
        if key.execution.ends_with(suffix) {
            key.execution.truncate(key.execution.len() - suffix.len());
        }
        return Ok(());
    }
    if key.execution.ends_with(suffix) {
        return Ok(());
    }
    key.execution = match next {
        Some(next) => source_string_parts(
            std::iter::once(key.execution.as_str()).chain(std::iter::once(suffix)),
            next,
        )?,
        None => format!("{}{suffix}", key.execution),
    };
    Ok(())
}

fn key_negative_prompt(request: &GenerationRequest) -> Result<Option<&str>> {
    let cfg_on = crate::sdxl::sampling::ResolvedSampling::resolve(request)?.guidance() > 1.0;
    Ok(cfg_on.then(|| request.negative_prompt.as_deref().unwrap_or("")))
}

/// Encoded text window. SDXL's long-prompt path encodes `n` CLIP windows of 77; every other iOS
/// provider pads to a fixed, artifact-determined window (0 here, covered by the layout digest).
pub fn token_window(candidate: &IosCandidate) -> Result<u32> {
    match (&candidate.work, candidate.provider.as_str()) {
        (IosWork::Generate(req), crate::sdxl::SDXL_ID) => {
            // The actual sampler defaults decide CFG (acceleration defaults to guidance 1).
            // SDXL does not tokenize/retain a negative row when effective guidance is off.
            let negative = key_negative_prompt(req)?;
            let windows =
                mlx_gen_sdxl::conditioning_window_count(&candidate.spec, &req.prompt, negative)?;
            Ok(windows.saturating_mul(77))
        }
        _ => Ok(0),
    }
}

type KeyCapacity<'a> = &'a mut dyn FnMut(u64) -> mlx_gen::Result<()>;

fn token_window_using_key_capacity(
    candidate: &IosCandidate,
    source: &SourceInventory,
    next: KeyCapacity<'_>,
) -> AdmissionResult<u32> {
    let (IosWork::Generate(request), crate::sdxl::SDXL_ID) =
        (&candidate.work, candidate.provider.as_str())
    else {
        return token_window(candidate).map_err(AdmissionError::Failed);
    };
    let negative = key_negative_prompt(request).map_err(pricing_error)?;
    let mut capacity_refused = false;
    let mut checked = |bytes| {
        let result = next(bytes);
        if result.is_err() {
            capacity_refused = true;
        }
        result
    };
    mlx_gen_sdxl::conditioning_window_count_with_source_capacity(
        &candidate.spec,
        &request.prompt,
        negative,
        source,
        &mut checked,
    )
    .map(|windows| windows.saturating_mul(77))
    .map_err(|error| {
        if capacity_refused {
            AdmissionError::refused(RefusalKind::Capacity, error.to_string())
        } else {
            AdmissionError::Failed(error.into())
        }
    })
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
    if c.load_shape != candidate.spec.load_shape {
        return Err(mismatch("load shape"));
    }
    if work_mode(provider, &candidate.spec, &candidate.work)? != c.mode {
        return Err(mismatch("mode"));
    }
    if work_geometry(&candidate.work)? != c.geometry
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
            // A PiD decode is not an iOS route (no provider's iOS contract offers it): a typed mode
            // refusal, never an untyped mismatch.
            if req.use_pid {
                return Err(AdmissionError::refused(
                    RefusalKind::Mode,
                    format!("{provider}: a PiD decode (`use_pid`) is not served on iOS"),
                ));
            }
            if req.memory.unwrap_or_default() != candidate.execution {
                return Err(mismatch("request execution"));
            }
            caps.validate_request(provider, req)?;
            if provider == crate::sdxl::SDXL_ID {
                crate::sdxl::sampling::ResolvedSampling::resolve(req).map_err(|error| {
                    AdmissionError::refused(RefusalKind::Mode, error.to_string())
                })?;
            }
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
    crate::binding::work_binding(work)
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

/// Exact SANA source event cardinalities for capture-storage preflight only.
/// The actual original request is borrowed; this is not a RAM/model permit.
pub fn sana_capture_observation_plan(
    candidate: &IosCandidate,
    key: &EvidenceKey,
    page_bytes: u64,
    analysis_budget: u64,
) -> AdmissionResult<mlx_gen_sana::ios_pricing::CaptureObservationPlan> {
    let IosWork::Generate(actual) = &candidate.work else {
        return Err(mismatch("SANA capture work kind"));
    };
    if candidate.provider != key.provider
        || !matches!(
            candidate.provider.as_str(),
            "sana_1600m" | "sana_sprint_1600m"
        )
    {
        return Err(mismatch("SANA capture plan provider"));
    }
    let pricing = IosPricingRequest {
        provider: &candidate.provider,
        spec: &candidate.spec,
        mode: &candidate.context.mode,
        geometry: candidate.context.geometry,
        selection: &candidate.context.selection,
        execution: &candidate.execution,
        token_window: key.token_window,
        generation: Some(actual),
        transform: None,
        depth: None,
    };
    mlx_gen_sana::ios_pricing::capture_observation_plan(&pricing, page_bytes, analysis_budget)
        .map_err(pricing_error)
}

/// SDK capture-program identity. SANA captures with explicit memory use the
/// configured/joined program; catalog ordinary keys remain unchanged. This
/// describes the future required program, never authorizes execution itself.
pub fn capture_evidence_key(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    machine: &str,
) -> AdmissionResult<EvidenceKey> {
    let mut key = evidence_key(registry, candidate, machine)?;
    bind_sana_retirement_key(
        &mut key,
        candidate,
        mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle,
        None,
    )?;
    Ok(key)
}

/// Same SDK capture identity with the SAME real source-capacity callback for
/// source capture/tokenization AND execution-tag String ownership.
pub fn capture_evidence_key_with_key_capacity(
    registry: &ProviderRegistry,
    candidate: &mut IosCandidate,
    machine: &str,
    next: KeyCapacity<'_>,
) -> AdmissionResult<EvidenceKey> {
    let mut key = evidence_key_with_key_capacity(registry, candidate, machine, &mut *next)?;
    bind_sana_retirement_key(
        &mut key,
        candidate,
        mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle,
        Some(next),
    )?;
    Ok(key)
}

/// Capture key for EVERY provider under the real byte-sensitive source callback.
/// Only token_window uses the SDXL-specific tokenizer guard. Reuses an existing inventory;
/// otherwise retains the ordinary inventory on the candidate for its later capture permit.
pub fn evidence_key_with_key_capacity(
    registry: &ProviderRegistry,
    candidate: &mut IosCandidate,
    machine: &str,
    next: KeyCapacity<'_>,
) -> AdmissionResult<EvidenceKey> {
    validate_actual(registry, candidate)?;
    let route = route_of(candidate)?;
    if candidate.spec.source_inventory().is_none() {
        let source = with_source_inventory_capacity(next, |next| {
            SourceInventory::capture_with_capacity(&candidate.spec, next)
        })?;
        with_source_inventory_capacity(next, |next| {
            candidate
                .spec
                .bind_source_inventory_with_capacity(source, next)
        })?;
    }
    let source = candidate
        .spec
        .source_inventory()
        .expect("retained ordinary inventory");
    with_source_inventory_capacity(next, |next| {
        source.verify_for_with_capacity(&candidate.spec, next)
    })?;
    with_source_inventory_capacity(next, |next| {
        Ok(source.read_unchanged_with_capacity(next, |same_next| {
            let mut provider_next = |bytes| same_next(bytes).map_err(mlx_gen::Error::from);
            key_unchecked_using_key_capacity(
                candidate,
                machine,
                route,
                Some((source, &mut provider_next)),
            )
        }))
    })?
    .map_err(source_metadata_gap_refusal)
}

fn key_unchecked(
    candidate: &IosCandidate,
    machine: &str,
    route: String,
) -> AdmissionResult<EvidenceKey> {
    // Metadata-only compatibility API. Real SDK/capture routes supply the callback below.
    key_unchecked_using_key_capacity(candidate, machine, route, None)
}

fn key_unchecked_using_key_capacity(
    candidate: &IosCandidate,
    machine: &str,
    route: String,
    pre_key: Option<(&SourceInventory, KeyCapacity<'_>)>,
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
        architecture_sha256: crate::layout::hex(&artifact(crate::layout::architecture_digest(
            &candidate.spec,
        ))?),
        configured_variant_sha256: crate::layout::configured_variant_digest(
            &candidate.provider,
            &candidate.spec,
        )
        .map_err(pricing_error)?
        .map(|digest| crate::layout::hex(&digest)),
        mode: c.mode.as_key().into(),
        width: c.geometry.width,
        height: c.geometry.height,
        batch: c.geometry.batch,
        reference_count: c.geometry.reference_count,
        mask_count: match &candidate.work {
            IosWork::Generate(request) => u32::try_from(
                request
                    .conditioning
                    .iter()
                    .filter(|item| matches!(item, Conditioning::Mask { .. }))
                    .count(),
            )
            .map_err(|_| no_evidence("mask role count exceeds wire integer".into()))?,
            _ => 0,
        },
        control_count: match &candidate.work {
            IosWork::Generate(request) => u32::try_from(
                request
                    .conditioning
                    .iter()
                    .filter(|item| matches!(item, Conditioning::Control { .. }))
                    .count(),
            )
            .map_err(|_| no_evidence("control role count exceeds wire integer".into()))?,
            _ => 0,
        },
        load_form: evidence::AnchorLoadForm::from_spec(&candidate.spec),
        execution: execution_key(candidate)?,
        sampling_recipe: match &candidate.work {
            IosWork::Generate(request) => Some(
                crate::sampling::resolve_for_spec(&candidate.provider, &candidate.spec, request)
                    .map_err(pricing_error)?
                    .token()
                    .into(),
            ),
            IosWork::Restore(request) => Some(
                crate::sampling::resolve_transform(&candidate.provider, &candidate.spec, request)
                    .map_err(pricing_error)?
                    .token()
                    .into(),
            ),
            IosWork::Depth(request) => Some(
                crate::sampling::resolve_depth(&candidate.provider, &candidate.spec, request)
                    .map_err(pricing_error)?
                    .token()
                    .into(),
            ),
        },
        sdxl_work: match &candidate.work {
            IosWork::Generate(request) if candidate.provider == crate::sdxl::SDXL_ID => {
                Some(crate::SdxlWorkBinding::query(request))
            }
            _ => None,
        },
        token_window: match pre_key {
            Some((source, next)) => token_window_using_key_capacity(candidate, source, next)?,
            None => token_window(candidate)?,
        },
    })
}

/// Per quantum label, the device's calibration of that formula (from the anchor captures).
pub type Calibration = std::collections::BTreeMap<String, mlx_gen::ios_quantum::QuantumCalibration>;

/// What an admission's demand rests on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmissionBasis {
    /// The request IS this anchor: its measured demand, no derivation.
    Measured(AnchorKind),
    /// Derived by the anchor law from this anchor of the family.
    Derived { from: AnchorKind },
    /// An explicitly approved empirical process-pressure estimate, not a physical certificate.
    EngineeringEstimate,
    /// A labelled device-capture permit: no evidence, bounded per quantum.
    CapturePermit,
}

/// A sealed admission: the source inventory, the admitted context and the exact work binding.
#[derive(Debug)]
pub struct Admission {
    pub(crate) index: usize,
    pub(crate) provider: String,
    /// The render's own growth over the process's starting footprint (anchor-measured or derived).
    pub(crate) demand_bytes: u64,
    /// The anchor's measured repeat spread (floored): the explicit low-end margin.
    pub(crate) margin_bytes: u64,
    pub(crate) basis: AdmissionBasis,
    /// The anchor's per-quantum DEVICE calibration the admitted work's guards run under.
    pub(crate) calibration: Calibration,
    pub(crate) spec: LoadSpec,
    pub(crate) context: MemoryRunContext,
    pub(crate) binding: [u8; 32],
    pub(crate) key: EvidenceKey,
    /// Every unproved premise the admitted derivation rests on; non-empty only under an explicit
    /// policy that approved each of them. Recorded, never implicit.
    pub(crate) premises: Vec<IosCertificatePremise>,
    /// Required SOURCE execution, carried into the bound generator. This is
    /// never satisfied merely by the caller stating an execution-fact enum.
    pub(crate) native_retirement: mlx_gen::ios_pricing::IosNativeRetirementExecution,
}

fn bound_domain(bound: &IosWindowBound) -> IosMemoryDomain {
    match bound {
        IosWindowBound::Transfer { domain, .. } | IosWindowBound::RawPhysical { domain, .. } => {
            *domain
        }
    }
}

/// Bring a provider's certified bounds into the measured anchor's counter domain under the
/// policy: every provider premise must be approved; native-domain bounds are mapped to process
/// ONLY through the approved activity-26569 model, whose premise is then recorded. Measured
/// evidence is never altered; only certificate deltas are transformed.
fn in_anchor_domain(
    certified: IosCertifiedBounds,
    measured: &anchor::MeasuredAnchor,
    policy: &PremisePolicy,
) -> std::result::Result<(Vec<IosWindowBound>, Vec<IosCertificatePremise>), String> {
    policy.accept(&certified.premises)?;
    let mut premises = certified.premises;
    let declared_host = !matches!(certified.host, IosHostGrowth::Undeclared);
    if certified
        .bounds
        .iter()
        .all(|bound| bound_domain(bound) == measured.domain)
    {
        // A process-domain certificate already includes host growth: a declared (or unproved)
        // host term beside it is unsupported evidence, never silently dropped. A native anchor's
        // counter excludes host allocations by construction, so there it is not consumed.
        if declared_host && measured.domain == IosMemoryDomain::ProcessGrowth {
            return Err(
                "a declared host-growth term accompanies process-domain certificates".into(),
            );
        }
        return Ok((certified.bounds, premises));
    }
    if measured.domain != IosMemoryDomain::ProcessGrowth {
        return Err(format!(
            "certificates are not in the measured anchor's {:?} domain",
            measured.domain
        ));
    }
    let model = policy.approved_native_to_process().ok_or_else(|| {
        format!(
            "native-allocator certificates do not authorize process-growth admission without the \
             explicitly approved premise {NATIVE_TO_PROCESS_PREMISE_ID} and its model"
        )
    })?;
    if declared_host && !policy.approved.contains(NATIVE_TO_PROCESS_HOST_PREMISE_ID) {
        return Err(format!(
            "certificates declaring source-derived host growth need the separately approved \
             host-extended premise {NATIVE_TO_PROCESS_HOST_PREMISE_ID} ({})",
            model.host_premise().statement
        ));
    }
    let bounds = model
        .to_process_with_host(&certified.bounds, &certified.host)
        .map_err(|e| e.to_string())?;
    premises.push(model.premise());
    if declared_host {
        premises.push(model.host_premise());
    }
    Ok((bounds, premises))
}

/// Explicit, user-approved acceptance of named certificate premises. The default approves
/// NOTHING: premise-dependent bounds then refuse exactly, naming each premise.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PremisePolicy {
    approved: std::collections::BTreeSet<String>,
    /// The activity-26569 engineering model; consulted only if its premise id is ALSO approved.
    native_to_process: Option<IosNativeToProcessModel>,
    engineering_process: Option<crate::engineering::EngineeringProcessModel>,
}

impl PremisePolicy {
    /// Approve exactly these premise ids (the caller records the user approval that justifies it).
    pub fn approving(ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            approved: ids.into_iter().collect(),
            native_to_process: None,
            engineering_process: None,
        }
    }
    /// Supply the target-device native-to-process model. It takes effect only when
    /// [`NATIVE_TO_PROCESS_PREMISE_ID`] is also approved; supplying it alone approves nothing.
    pub fn with_native_to_process(mut self, model: IosNativeToProcessModel) -> Self {
        self.native_to_process = Some(model);
        self
    }
    /// Attach the reviewable empirical model. Both separately named engineering premises must
    /// ALSO be approved. This does not approve physical, HOST or source-certificate premises.
    pub fn with_engineering_process(
        mut self,
        model: crate::engineering::EngineeringProcessModel,
    ) -> Self {
        self.engineering_process = Some(model);
        self
    }
    /// The runtime verifies the model's hardware/OS/build scope before installation.
    pub fn engineering_process(&self) -> Option<&crate::engineering::EngineeringProcessModel> {
        self.engineering_process.as_ref()
    }
    /// Resolve an approved conservative envelope before parsing model/component metadata or
    /// constructing request carriers. Absence of an engineering model grants no empirical bound.
    pub fn preparation_envelope(
        &self,
        rows: &[AnchorRow],
        scope: crate::engineering::PreparationScope<'_>,
    ) -> std::result::Result<Option<u64>, String> {
        let Some(model) = self.engineering_process() else {
            return Ok(None);
        };
        use crate::engineering::{
            PROCESS_PREPARATION_PREMISE_ID, PROCESS_PRESSURE_PREMISE_ID,
            PROCESS_RESIDUAL_PREMISE_ID,
        };
        for id in [
            PROCESS_PREPARATION_PREMISE_ID,
            PROCESS_PRESSURE_PREMISE_ID,
            PROCESS_RESIDUAL_PREMISE_ID,
        ] {
            if !self.approved.contains(id) {
                return Err(format!(
                    "engineering source preparation has unapproved premise {id}"
                ));
            }
        }
        let mut envelope = None;
        for row in rows.iter().filter(|row| {
            row.anchor_kind == AnchorKind::Optimized
                && row.key.machine == scope.machine
                && row.key.provider == scope.provider
                && row.key.route == scope.route
                && row.key.mode == scope.mode
                && row.key.tier == scope.tier
        }) {
            let bytes = model.preparation_estimate(row)?;
            envelope = Some(envelope.map_or(bytes, |old: u64| old.max(bytes)));
        }
        envelope.map(Some).ok_or_else(|| {
            "no scoped optimized source-preparation capture exists before source inspection".into()
        })
    }

    fn approved_native_to_process(&self) -> Option<&IosNativeToProcessModel> {
        self.native_to_process
            .as_ref()
            .filter(|_| self.approved.contains(NATIVE_TO_PROCESS_PREMISE_ID))
    }
    fn accept(&self, premises: &[IosCertificatePremise]) -> std::result::Result<(), String> {
        let unapproved: Vec<_> = premises
            .iter()
            .filter(|premise| !self.approved.contains(&premise.id))
            .map(|premise| format!("{} ({})", premise.id, premise.statement))
            .collect();
        if unapproved.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "bounds rest on unapproved premise(s): {}",
                unapproved.join("; ")
            ))
        }
    }
}

impl Admission {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn provider(&self) -> &str {
        &self.provider
    }
    /// What the admitted work itself adds to a process (0 for a capture permit): the anchor's
    /// measured growth at its own key, else the anchor-law derivation. Never an absolute peak.
    pub fn demand_bytes(&self) -> u64 {
        self.demand_bytes
    }
    /// The low-end margin charged on top of the demand: the anchor's measured repeat spread.
    pub fn margin_bytes(&self) -> u64 {
        self.margin_bytes
    }
    /// `demand + margin`: what live headroom must cover, and the measured envelope the admitted
    /// work's quantum guards run under ([`mlx_gen::ios_quantum::with_admitted_envelope`]).
    pub fn required_bytes(&self) -> u64 {
        self.demand_bytes
            .checked_add(self.margin_bytes)
            .expect("sealed admission construction checked the complete envelope")
    }
    pub fn required_mib(&self) -> u32 {
        u32::try_from(self.required_bytes().div_ceil(1 << 20)).unwrap_or(u32::MAX)
    }
    pub fn basis(&self) -> AdmissionBasis {
        self.basis
    }
    /// Actual sealed selection requirement, descriptive only (not a successful
    /// native range, physical permission or a substitute for execution).
    pub fn native_retirement_execution(
        &self,
    ) -> mlx_gen::ios_pricing::IosNativeRetirementExecution {
        self.native_retirement
    }
    /// The anchor's per-quantum DEVICE calibration (label → priced scratch vs device-observed
    /// growth) the admitted work's guards use inside the measured envelope.
    pub fn quantum_calibration(&self) -> &Calibration {
        &self.calibration
    }
    pub fn context(&self) -> &MemoryRunContext {
        &self.context
    }
    pub fn evidence_key(&self) -> &EvidenceKey {
        &self.key
    }
    /// The approved, unproved premises this admission's derivation rests on (empty when fully
    /// source-proved or measured at the anchor's own key).
    pub fn certificate_premises(&self) -> &[IosCertificatePremise] {
        &self.premises
    }
    /// Whether this is an evidence-collection capture permit rather than a product admission.
    pub fn is_capture(&self) -> bool {
        self.basis == AdmissionBasis::CapturePermit
    }
}

fn no_evidence(reason: String) -> AdmissionError {
    AdmissionError::refused(RefusalKind::NoEvidence, reason)
}

fn completed(row: &AnchorRow) -> bool {
    row.demand_bytes().is_some()
}

/// The `MemoryMode` a key string names.
fn mode_from_key(key: &str) -> MemoryMode {
    match key {
        "text_to_image" => MemoryMode::TextToImage,
        "image_to_image" => MemoryMode::ImageToImage,
        "edit" => MemoryMode::Edit,
        other => MemoryMode::Other(other.to_string()),
    }
}

/// The ANCHOR's own resolved selection, load form, geometry and replay recipe, retained so the
/// SAME request both prices the anchor and is handed to a provider bounder.
struct AnchorContext<'row> {
    provider: &'row str,
    spec: LoadSpec,
    mode: MemoryMode,
    geometry: MemoryGeometry,
    selection: MemorySelection,
    execution: GenerationMemory,
    token_window: u32,
    token: &'row str,
    mask_count: u32,
    control_count: u32,
    recorded_source: Option<&'row mlx_gen_sdxl::sampling::SamplingSourceFacts>,
}

impl AnchorContext<'_> {
    /// The ANCHOR's replay side for a provider bounder: `request` (from [`Self::request`]) plus
    /// the recorded token and typed roles, so a bounder resolves its exact execution plan.
    fn bound<'a>(
        &'a self,
        request: &'a IosPricingRequest<'a>,
    ) -> AdmissionResult<IosBoundRequest<'a>> {
        Ok(IosBoundRequest {
            pricing: request,
            replay: Some(IosReplay {
                token: self.token,
                true_reference_count: crate::pricing::reference_role_count(
                    self.geometry.reference_count,
                    self.mask_count,
                    self.control_count,
                )
                .map_err(pricing_error)?,
                mask_count: self.mask_count,
                control_count: self.control_count,
                source: self
                    .recorded_source
                    .map(|source| mlx_gen::ios_pricing::IosReplaySource(source)),
            }),
        })
    }
    /// The ANCHOR's own payload-free replay request (no GenerationRequest carrier).
    fn request(&self) -> IosPricingRequest<'_> {
        IosPricingRequest {
            provider: self.provider,
            spec: &self.spec,
            mode: &self.mode,
            geometry: self.geometry,
            selection: &self.selection,
            execution: &self.execution,
            token_window: self.token_window,
            generation: None,
            transform: None,
            depth: None,
        }
    }
}

fn anchor_context<'row>(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    row: &'row AnchorRow,
    next: Option<KeyCapacity<'_>>,
) -> AdmissionResult<AnchorContext<'row>> {
    anchor_context_with_spec(registry, candidate, &candidate.spec, row, next)
}

fn anchor_context_with_spec<'row>(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    pricing_spec: &LoadSpec,
    row: &'row AnchorRow,
    mut next: Option<KeyCapacity<'_>>,
) -> AdmissionResult<AnchorContext<'row>> {
    let (Some(strategy), Some(parameters)) = (row.selection.strategy(), row.selection.parameters())
    else {
        return Err(no_evidence(format!(
            "anchor {:?} carries an unreadable selection",
            row.key
        )));
    };
    let provider = row.key.provider.as_str();
    let Some(_) = crate::pricing::provider_pricing(provider) else {
        return Err(no_evidence(format!("{provider}: no counted iOS pricing")));
    };
    let recorded_source = if provider == crate::sdxl::SDXL_ID {
        let binding = row.key.sdxl_work.as_ref().ok_or_else(|| {
            pricing_error(Error::Unsupported(
                "SDXL anchor lacks actual image/text source context".into(),
            ))
        })?;
        let facts = binding.require_facts().map_err(pricing_error)?;
        crate::source_work::validate_key_roles(&row.key, facts).map_err(pricing_error)?;
        Some(facts)
    } else {
        None
    };
    let mut spec = match next.as_mut() {
        Some(next) => with_source_inventory_capacity(&mut **next, |core_next| {
            pricing_spec.clone_with_capacity(core_next)
        })?,
        None => pricing_spec.clone(),
    };
    // This override copies only the two load-form enums; it allocates nothing.
    // Re-price the anchor's actual load, not the requesting endpoint's load form.
    row.key.load_form.apply(&mut spec);
    // Retain the actual matched configured graph, including Control+Reference/IP combinations.
    // Family identity carries normalized overlay semantics; mode-name stripping would silently
    // change the load and defeat the provider's real init/IP interpretation.
    if provider == crate::sdxl::SDXL_ID {
        spec.resolved_route = Some(match next.as_mut() {
            Some(next) => source_string_clone(&row.key.route, Some(&mut **next))?,
            None => row.key.route.clone(),
        });
    }
    let contract = registry
        .memory_strategy_contract(provider, &spec)?
        .ok_or_else(|| no_evidence(format!("{provider}: no memory contract")))?;
    let selection = MemorySelection {
        strategy,
        parameters,
        tier: candidate.context.selection.tier,
    };
    contract.validate_selection(&selection).map_err(|e| {
        no_evidence(format!(
            "{provider}: anchor selection is not executable: {e}"
        ))
    })?;
    let execution = execution_for(provider, &contract, &selection);
    let geometry = MemoryGeometry {
        width: row.key.width,
        height: row.key.height,
        batch: row.key.batch,
        frames: 1,
        reference_count: row.key.reference_count,
    };
    let token = row.key.sampling_recipe.as_deref().ok_or_else(|| {
        no_evidence(format!(
            "{provider} anchor missing resolved execution recipe"
        ))
    })?;
    Ok(AnchorContext {
        provider,
        spec,
        mode: mode_from_key(&row.key.mode),
        geometry,
        selection,
        execution,
        token_window: row.key.token_window,
        token,
        mask_count: row.key.mask_count,
        control_count: row.key.control_count,
        recorded_source,
    })
}

fn validate_engineering_inventory(
    candidate: &IosCandidate,
    key: &EvidenceKey,
    row: &AnchorRow,
    context: &AnchorContext<'_>,
) -> AdmissionResult<()> {
    if row.key.provider != candidate.provider
        || row.key.route != key.route
        || row.key.mode != key.mode
    {
        return Err(no_evidence(
            "engineering estimate cannot borrow a different provider, route or mode's endpoint"
                .into(),
        ));
    }
    let layout = if context.provider.starts_with("sana_") {
        artifact(crate::admission::sana_snapshot_profile(&context.spec))?.layout_sha256
    } else {
        artifact(crate::layout::artifact_layout_digest(&context.spec))?
    };
    if crate::layout::hex(&layout) != row.key.layout_sha256 {
        return Err(no_evidence("engineering estimate requires the exact captured header/config/tokenizer/adapter inventory".into()));
    }
    Ok(())
}

/// Count the ANCHOR's own resolved recipe/roles, selection, load form and geometry on the
/// matched artifact/configured semantic graph. A mode label never decides which configured
/// overlay is loaded: the actual provider factory/pricer owns that interpretation.
fn price_anchor<'row>(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    row: &'row AnchorRow,
) -> AdmissionResult<(AnchorContext<'row>, IosPricing)> {
    price_anchor_using_source_capacity(registry, candidate, row, None)
}

fn price_anchor_using_source_capacity<'row>(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    row: &'row AnchorRow,
    next: Option<KeyCapacity<'_>>,
) -> AdmissionResult<(AnchorContext<'row>, IosPricing)> {
    price_anchor_using_estimated_preparation(registry, candidate, &candidate.spec, row, next, None)
}

fn price_anchor_using_estimated_preparation<'row>(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    pricing_spec: &LoadSpec,
    row: &'row AnchorRow,
    mut next: Option<KeyCapacity<'_>>,
    estimate: Option<mlx_gen_iris::ios_pricing::EstimatedSourcePreparation>,
) -> AdmissionResult<(AnchorContext<'row>, IosPricing)> {
    let context = match next.as_mut() {
        Some(next) => {
            anchor_context_with_spec(registry, candidate, pricing_spec, row, Some(&mut **next))?
        }
        None => anchor_context_with_spec(registry, candidate, pricing_spec, row, None)?,
    };
    let mut pricing = match next {
        Some(next) => with_source_metadata_capacity(next, |next| {
            crate::sampling::price_metadata_with_estimated_preparation(
                &context.request(),
                context.token,
                context.mask_count,
                context.control_count,
                context.recorded_source,
                estimate,
                next,
            )
        })?,
        None => crate::sampling::price_metadata(
            &context.request(),
            context.token,
            context.mask_count,
            context.control_count,
            context.recorded_source,
        )
        .map_err(pricing_error)?,
    };
    anchor::attach_observations(&mut pricing, row).map_err(no_evidence)?;
    if matches!(context.provider, "sana_1600m" | "sana_sprint_1600m")
        && row
            .key
            .execution
            .ends_with(mlx_gen_sana::ios_pricing::NATIVE_RETIREMENT_EXECUTION_SUFFIX)
        && mlx_gen_sana::ios_pricing::observed_pricing_regime(&pricing)
            == mlx_gen_sana::ios_pricing::SanaPricingRegime::RetiredLifecycle
    {
        // Ordinary replay above retains original source/error chronology.
        // Only complete actual observations select the reachable split windows.
        let mut joined = crate::sampling::price_sana_metadata_in_regime(
            &context.request(),
            context.token,
            mlx_gen_sana::ios_pricing::SanaPricingRegime::RetiredLifecycle,
        )
        .map_err(pricing_error)?;
        joined.observations = pricing.observations;
        pricing = joined;
    }
    Ok((context, pricing))
}

/// A provider's pricing that declines a configuration (`Unsupported`: an adapter stack, a PiD
/// decode, a layout it does not price) makes it not derivable — a typed `no_evidence`; anything else
/// (an unreadable header) is a failure.
// Only these exact unresolved SOURCE-authority reasons are proof-missing refusals.
// Malformed artifacts, unrelated Unsupported/IO errors and real Capacity stay unchanged.
fn source_metadata_gap_refusal(error: AdmissionError) -> AdmissionError {
    match error {
        AdmissionError::Failed(Error::Unsupported(reason))
            if [
                SourceInventory::READDIR_METADATA_GAP,
                SourceInventory::PIN_VERIFY_METADATA_GAP,
                SourceInventory::PIN_CAPTURE_METADATA_GAP,
                SourceInventory::ABSOLUTE_PATH_METADATA_GAP,
                SourceInventory::MISSING_INVENTORY_METADATA_GAP,
            ]
            .contains(&reason.as_str()) =>
        {
            no_evidence(reason)
        }
        other => other,
    }
}

fn with_source_metadata_capacity<T>(
    next: KeyCapacity<'_>,
    work: impl FnOnce(KeyCapacity<'_>) -> mlx_gen::Result<T>,
) -> AdmissionResult<T> {
    with_source_capacity_using(next, work, pricing_error)
}

fn with_source_inventory_capacity<T>(
    next: KeyCapacity<'_>,
    work: impl FnOnce(
        &mut dyn FnMut(u64) -> mlx_gen::gen_core::Result<()>,
    ) -> mlx_gen::gen_core::Result<T>,
) -> AdmissionResult<T> {
    with_source_capacity_using(
        next,
        |same_next| {
            mlx_gen::ios_pricing::with_core_source_capacity(same_next, |core_next| {
                work(core_next).map_err(mlx_gen::Error::from)
            })
        },
        |error| source_metadata_gap_refusal(AdmissionError::Failed(error)),
    )
}

/// Build exactly the borrowed UTF-8 segments billed BEFORE allocation. Both
/// iterator passes are source-owned, immutable and allocation-free. No format!
/// growth heuristic or temporary collected Vec participates in the guarded path.
fn source_string_parts<'a>(
    parts: impl Iterator<Item = &'a str> + Clone,
    next: KeyCapacity<'_>,
) -> AdmissionResult<String> {
    let bytes = parts
        .clone()
        .try_fold(0usize, |bytes, part| bytes.checked_add(part.len()))
        .ok_or_else(|| no_evidence("source string requested-layout overflow".into()))?;
    let requested = u64::try_from(bytes)
        .map_err(|_| no_evidence("source string requested-layout exceeds u64".into()))?;
    with_source_metadata_capacity(next, |same_next| {
        same_next(requested)?;
        let mut value = String::new();
        value.try_reserve_exact(bytes).map_err(|_| {
            mlx_gen::Error::Msg("source string allocation failed after capacity check".into())
        })?;
        for part in parts {
            value.push_str(part);
        }
        Ok(value)
    })
}

fn source_string_clone(value: &str, next: Option<KeyCapacity<'_>>) -> AdmissionResult<String> {
    match next {
        Some(next) => source_string_parts(std::iter::once(value), next),
        None => Ok(value.to_owned()),
    }
}

/// Full concrete context Clone law: four strings and the ordered reference canvases allocate.
/// Exhaustive destructuring/matching forces a source audit if a field/variant is
/// added; selection, authority, geometry, budget and cache state are Copy data.
fn source_context_clone(
    context: &MemoryRunContext,
    next: Option<KeyCapacity<'_>>,
) -> AdmissionResult<MemoryRunContext> {
    let Some(next) = next else {
        return Ok(context.clone());
    };
    fn require_copy<T: Copy>(_: T) {}
    require_copy((
        context.selection,
        context.optimization_authority,
        context.calibration_abi,
        context.load_shape,
        context.has_reference,
        context.use_pid,
        context.has_phases,
        context.geometry,
        context.budget,
        context.predicted_peak_bytes,
        context.cache_state,
    ));
    let MemoryRunContext {
        selection: _,
        optimization_authority: _,
        calibration_abi: _,
        calibration_fingerprint,
        load_shape: _,
        mode,
        has_reference: _,
        use_pid: _,
        has_phases: _,
        geometry: _,
        reference_resources,
        overlay,
        budget: _,
        predicted_peak_bytes: _,
        cache_state: _,
        evidence_revision,
    } = context;
    let mode_bytes = match mode {
        MemoryMode::TextToImage | MemoryMode::ImageToImage | MemoryMode::Edit => 0,
        MemoryMode::Other(value) => value.len(),
    };
    let reference_bytes = match reference_resources {
        Some(mlx_gen::gen_core::MemoryReferenceResourceFacts {
            conditioning_tokens,
            prepared_canvases,
        }) => {
            require_copy(*conditioning_tokens);
            prepared_canvases
                .len()
                .checked_mul(std::mem::size_of::<(u32, u32)>())
                .ok_or_else(|| {
                    no_evidence("reference canvas clone requested-layout overflow".into())
                })?
        }
        None => 0,
    };
    let bytes = [
        calibration_fingerprint.len(),
        mode_bytes,
        overlay.as_ref().map_or(0, String::len),
        evidence_revision.len(),
        reference_bytes,
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .ok_or_else(|| no_evidence("context clone requested-layout overflow".into()))?;
    let requested = u64::try_from(bytes)
        .map_err(|_| no_evidence("context clone requested-layout exceeds u64".into()))?;
    with_source_metadata_capacity(next, |same_next| {
        same_next(requested)?;
        Ok(context.clone())
    })
}

fn source_evidence_revision(
    key: &EvidenceKey,
    premises: &[IosCertificatePremise],
    next: Option<KeyCapacity<'_>>,
) -> AdmissionResult<String> {
    let Some(next) = next else {
        let revision = format!("ios-device-anchor:{}", key.layout_sha256);
        return Ok(if premises.is_empty() {
            revision
        } else {
            let ids: Vec<_> = premises.iter().map(|premise| premise.id.as_str()).collect();
            format!("{revision};premises={}", ids.join(","))
        });
    };
    let prefix =
        std::iter::once("ios-device-anchor:").chain(std::iter::once(key.layout_sha256.as_str()));
    let premise_prefix = (!premises.is_empty()).then_some(";premises=").into_iter();
    let ids = premises
        .iter()
        .enumerate()
        .flat_map(|(index, premise)| [if index == 0 { "" } else { "," }, premise.id.as_str()]);
    source_string_parts(prefix.chain(premise_prefix).chain(ids), next)
}

fn with_source_capacity_using<T>(
    next: KeyCapacity<'_>,
    work: impl FnOnce(KeyCapacity<'_>) -> mlx_gen::Result<T>,
    classify_error: impl FnOnce(Error) -> AdmissionError,
) -> AdmissionResult<T> {
    let mut refused = false;
    let mut checked = |bytes| {
        let result = next(bytes);
        if result.is_err() {
            refused = true;
        }
        result
    };
    let result = work(&mut checked);
    if refused {
        return Err(AdmissionError::refused(
            RefusalKind::Capacity,
            "source metadata capacity check refused before pricing allocation",
        ));
    }
    result.map_err(|error| classify_error(error.into()))
}

fn pricing_error(error: impl Into<Error>) -> AdmissionError {
    match error.into() {
        Error::Unsupported(reason) => no_evidence(format!("not derivable: {reason}")),
        other => AdmissionError::Failed(other),
    }
}

/// Borrowed facts from one SDK admission boundary. Grouping preserves absent facts and
/// the caller's explicit premise policy; it grants no defaults or additional authority.
#[derive(Clone, Copy)]
pub(crate) struct AdmissionEvidence<'a> {
    request_baseline: Option<&'a IosMemoryBaseline>,
    execution_facts: Option<&'a IosExecutionFacts>,
    premise_policy: &'a PremisePolicy,
}

/// Step 3: the request's demand and margin from the anchor table (see the module docs).
fn anchored_demand(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    contract: &MemoryProviderContract,
    key: &EvidenceKey,
    rows: &[AnchorRow],
    evidence: AdmissionEvidence<'_>,
) -> AdmissionResult<(
    u64,
    u64,
    AdmissionBasis,
    Calibration,
    Vec<IosCertificatePremise>,
)> {
    anchored_demand_using_source_capacity(
        registry,
        (candidate, &candidate.spec),
        contract,
        key,
        rows,
        evidence,
        None,
    )
}

fn anchored_demand_using_source_capacity(
    registry: &ProviderRegistry,
    priced_candidate: (&IosCandidate, &LoadSpec),
    contract: &MemoryProviderContract,
    key: &EvidenceKey,
    rows: &[AnchorRow],
    evidence: AdmissionEvidence<'_>,
    mut source_metadata_capacity: Option<KeyCapacity<'_>>,
) -> AdmissionResult<(
    u64,
    u64,
    AdmissionBasis,
    Calibration,
    Vec<IosCertificatePremise>,
)> {
    let (candidate, pricing_spec) = priced_candidate;
    let AdmissionEvidence {
        request_baseline,
        execution_facts,
        premise_policy,
    } = evidence;
    if let Some(row) = evidence::lookup(rows, key) {
        if let Some(model) = premise_policy.engineering_process() {
            model.validate_row(row).map_err(no_evidence)?;
        }
        return match &row.outcome {
            evidence::AnchorOutcome::Completed { .. } => {
                let baseline = request_baseline.ok_or_else(|| no_evidence(
                    "an exact measured key requires the actual SDK BeforeModelLoad process baseline".into()))?;
                if baseline.domain != IosMemoryDomain::ProcessGrowth
                    || baseline.coordinate != IosMemoryCoordinate::BeforeModelLoad
                    || baseline.bytes == 0
                    || baseline.source.is_empty()
                {
                    return Err(no_evidence(
                        "exact measured key has an unspecified/incompatible request baseline"
                            .into(),
                    ));
                }
                // Same complete key only: this is direct PROCESS evidence, not a Native-to-
                // Process transfer. A masked lifetime upper cannot claim accurate attained demand
                // or measured repeat variability. Retain the original pre-request ledger separately.
                let (demand, margin) = row
                    .exact_growth_and_margin_at(IosMemoryCoordinate::BeforeModelLoad)
                    .map_err(|error| {
                        no_evidence(format!(
                            "exact measured key lacks accurate SDK-coordinate peaks: {error}"
                        ))
                    })?;
                Ok((
                    demand,
                    margin,
                    AdmissionBasis::Measured(row.anchor_kind),
                    row.quantum_calibration(),
                    Vec::new(),
                ))
            }
            evidence::AnchorOutcome::Refused {
                refused_quantum, ..
            } => Err(AdmissionError::refused(
                RefusalKind::Capacity,
                format!(
                    "measured on {}: this {:?} anchor refused before allocation at quantum \
                         {} ({} B + {} B reserve, {:?} B live)",
                    key.machine,
                    row.anchor_kind,
                    refused_quantum.label,
                    refused_quantum.scratch_bytes,
                    refused_quantum.reserve_bytes,
                    refused_quantum.available_bytes
                ),
            )),
        };
    }
    let family = key.family().map_err(no_evidence)?;
    let (resident, optimized) = if premise_policy.engineering_process().is_some() {
        crate::engineering::pressure_family_anchors(rows, key).map_err(no_evidence)?
    } else {
        evidence::family_anchors(rows, &family)
    };
    let Some(optimized) = optimized.filter(|r| completed(r)) else {
        return Err(no_evidence(format!(
            "no completed optimized anchor for {family:?} in {EVIDENCE_TABLE_PATH}: another \
             architecture, tier, non-additive mode or device is not derivable"
        )));
    };
    // The ladder owns only its selected controls. SDXL's explicit CFG choice is independently
    // resolved, priced and matched by the sampling allocation class; other extra controls still
    // cannot drift from the declared execution.
    let mut expected = execution_for(&candidate.provider, contract, &candidate.context.selection);
    if candidate.provider == crate::sdxl::SDXL_ID {
        expected.cfg_batching = candidate.execution.cfg_batching;
    }
    if candidate.execution != expected {
        return Err(no_evidence(format!(
            "{key:?}: execution levers outside the iOS ladder are not derivable from an anchor"
        )));
    }
    let Some((pricer, _)) = crate::pricing::provider_pricing(&candidate.provider) else {
        return Err(no_evidence(format!(
            "{}: no counted iOS pricing, so only its anchors' own keys are admitted",
            candidate.provider
        )));
    };
    let engineering_preparation = if let Some(model) = premise_policy.engineering_process() {
        if source_metadata_capacity.is_none() {
            return Err(no_evidence(
                "engineering estimate requires the actual source-preparation capacity callback"
                    .into(),
            ));
        }
        let resident = resident.ok_or_else(|| no_evidence("engineering estimate requires a recorded resident endpoint (completed or typed refusal)".into()))?;
        let premises = model.premises(
            optimized,
            resident,
            &IosHostGrowth::Unproved(format!(
                "{}: this empirical path makes no finite source HOST/driver certificate claim",
                candidate.provider
            )),
        );
        premise_policy.accept(&premises).map_err(no_evidence)?;
        let bytes = model.preparation_estimate(optimized).map_err(no_evidence)?;
        if let Some(next) = source_metadata_capacity.as_mut() {
            with_source_metadata_capacity(&mut **next, |next| next(bytes))?;
        }
        Some(mlx_gen_iris::ios_pricing::EstimatedSourcePreparation::new(bytes,
            mlx_gen_iris::ios_pricing::SourcePreparationEstimateProvenance::BeforeRequestConstructionPeakWithMargin).map_err(pricing_error)?)
    } else {
        None
    };
    let request_context = IosPricingRequest {
        provider: &candidate.provider,
        spec: pricing_spec,
        mode: &candidate.context.mode,
        geometry: candidate.context.geometry,
        selection: &candidate.context.selection,
        execution: &candidate.execution,
        token_window: key.token_window,
        generation: match &candidate.work {
            IosWork::Generate(request) => Some(request),
            _ => None,
        },
        transform: match &candidate.work {
            IosWork::Restore(request) => Some(request),
            _ => None,
        },
        depth: match &candidate.work {
            IosWork::Depth(request) => Some(request),
            _ => None,
        },
    };
    let iris_source_pricing = matches!(
        candidate.provider.as_str(),
        "iris_3b" | IRIS_RESTORE_ID | IRIS_DEPTH_ID,
    );
    #[cfg(test)]
    let iris_source_pricing = iris_source_pricing && !crate::pricing::test_pricing_active();
    let sana_joined = matches!(
        candidate.provider.as_str(),
        "sana_1600m" | "sana_sprint_1600m"
    ) && execution_facts.is_some_and(|facts| {
        facts.native_retirement
            == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
    }) && matches!(&candidate.work,IosWork::Generate(request) if request.memory.is_some());
    #[cfg(test)]
    let sana_joined = sana_joined && !crate::pricing::test_pricing_active();
    let mut request = if sana_joined {
        mlx_gen_sana::ios_pricing::ios_pricing_in_regime(
            &request_context,
            mlx_gen_sana::ios_pricing::SanaPricingRegime::RetiredLifecycle,
        )
        .map_err(pricing_error)?
    } else {
        match (iris_source_pricing, source_metadata_capacity.as_mut()) {
            (true, Some(next)) => with_source_metadata_capacity(&mut **next, |next| {
                match engineering_preparation {
                    Some(estimate) => {
                        mlx_gen_iris::ios_pricing::pricing_facts_with_estimated_preparation(
                            &request_context,
                            None,
                            estimate,
                            next,
                        )
                    }
                    None => mlx_gen_iris::ios_pricing::pricing_facts_with_source_capacity(
                        &request_context,
                        None,
                        next,
                    ),
                }
                .map(|facts| facts.pricing)
            })?,
            _ => pricer(&request_context).map_err(pricing_error)?,
        }
    };
    // A fact from the SDK boundary, not an inferred budget/zero or a process-transfer proof.
    request.request_baseline = request_baseline.cloned();
    let resident_endpoint = resident;
    let resident = resident.filter(|r| completed(r));
    // Derive UPWARD from the optimized anchor; a measured resident anchor prices only its own
    // regime at other geometries (never derived downward toward the optimized end).
    let base = match resident {
        Some(r) if r.key.execution == key.execution => r,
        _ => optimized,
    };
    // Certified derivation (sc-26270) requires a registered source-owned provider bounder for
    // the STATED execution, consumed by the complete certified law. The explicitly approved
    // empirical path below has a distinct basis and never claims those certificates. Legacy
    // logical derive/consistency carry no authority and are not consulted. Measured anchors are
    // ProcessGrowth; native-allocator certificates reach that domain ONLY through the explicitly
    // approved activity-26569 engineering premise, validated against both endpoint anchors.
    let execution_facts = execution_facts.ok_or_else(|| {
        no_evidence(format!(
            "{key:?}: derivation away from an anchor key requires the actual certified execution \
             facts (stream, iOS quantum scope) at the admission boundary"
        ))
    })?;
    if let Some(model) = premise_policy.engineering_process() {
        if !execution_facts.ios_quantum_scope {
            return Err(no_evidence(
                "engineering estimate requires the actual iOS quantum scope".into(),
            ));
        }
        let baseline = request_baseline.ok_or_else(|| {
            no_evidence(
                "engineering estimate requires the actual SDK BeforeModelLoad process baseline"
                    .into(),
            )
        })?;
        if baseline.domain != IosMemoryDomain::ProcessGrowth
            || baseline.coordinate != IosMemoryCoordinate::BeforeModelLoad
            || baseline.bytes == 0
            || baseline.source.is_empty()
        {
            return Err(no_evidence(
                "engineering estimate has an incompatible request baseline".into(),
            ));
        }
        let resident = resident_endpoint.ok_or_else(|| no_evidence(
            "engineering estimate requires a recorded resident endpoint (completed or typed refusal)".into()
        ))?;
        let mut premises = model.premises(
            optimized,
            resident,
            &IosHostGrowth::Unproved(format!(
                "{}: this empirical path makes no finite source HOST/driver certificate claim",
                candidate.provider
            )),
        );
        premise_policy.accept(&premises).map_err(no_evidence)?;
        let (optimized_context, at_optimized) = match source_metadata_capacity.as_mut() {
            Some(next) => price_anchor_using_estimated_preparation(
                registry,
                candidate,
                pricing_spec,
                optimized,
                Some(&mut **next),
                engineering_preparation,
            )?,
            None => price_anchor_using_estimated_preparation(
                registry,
                candidate,
                pricing_spec,
                optimized,
                None,
                engineering_preparation,
            )?,
        };
        // This path consumes the actual source-generated phase pressure and measured process
        // endpoints. It neither requests nor manufactures mathematical native/HOST certificates.
        // The separate certified path below retains all its source-proof requirements.
        let (demand, margin) = model
            .estimate(optimized, &at_optimized, &request)
            .map_err(no_evidence)?;
        let resident_pricing = if completed(resident) {
            Some(match source_metadata_capacity.as_mut() {
                Some(next) => {
                    price_anchor_using_estimated_preparation(
                        registry,
                        candidate,
                        pricing_spec,
                        resident,
                        Some(&mut **next),
                        engineering_preparation,
                    )?
                    .1
                }
                None => {
                    price_anchor_using_estimated_preparation(
                        registry,
                        candidate,
                        pricing_spec,
                        resident,
                        None,
                        engineering_preparation,
                    )?
                    .1
                }
            })
        } else {
            None
        };
        model
            .validate_resident(
                optimized,
                &at_optimized,
                resident,
                resident_pricing.as_ref(),
            )
            .map_err(no_evidence)?;
        validate_engineering_inventory(candidate, key, optimized, &optimized_context)?;
        let resident_context = match source_metadata_capacity.as_mut() {
            Some(next) => anchor_context(registry, candidate, resident, Some(&mut **next))?,
            None => anchor_context(registry, candidate, resident, None)?,
        };
        validate_engineering_inventory(candidate, key, resident, &resident_context)?;
        premises.sort();
        premises.dedup();
        return Ok((
            demand,
            margin,
            AdmissionBasis::EngineeringEstimate,
            optimized.quantum_calibration(),
            premises,
        ));
    }
    // A provider's premise-mode bounder is selected ONLY when the policy approves its exact
    // premise id; its certificates then carry that premise, which the policy re-checks.
    let Some((bounder, gate, _premise_mode)) =
        crate::pricing::provider_bounder_under(&candidate.provider, |id| {
            premise_policy.approved.contains(id)
        })
    else {
        return Err(no_evidence(format!(
            "{}: no source-owned transfer bounder, so only its anchors' own keys are admitted",
            candidate.provider
        )));
    };
    gate(&candidate.provider, execution_facts)
        .map_err(|e| no_evidence(format!("{key:?}: execution not certifiable: {e}")))?;
    // Explicitly borrowed source-allocation check, not TLS, an analysis scheduling
    // budget, a new arbitrary cap, or a native-to-process premise. The same check
    // protects request/base and optimized/resident endpoint guarded analyses.
    let source_capacity_refused = std::cell::Cell::new(false);
    let certify = |anchor_request: &IosBoundRequest<'_>,
                   anchor_pricing: &IosPricing,
                   request: &IosBoundRequest<'_>,
                   request_pricing: &IosPricing,
                   source_metadata_capacity: &mut Option<KeyCapacity<'_>>|
     -> mlx_gen::Result<IosCertifiedBounds> {
        if let (true, Some(next)) = (
            matches!(
                candidate.provider.as_str(),
                "iris_3b" | IRIS_RESTORE_ID | IRIS_DEPTH_ID
            ),
            source_metadata_capacity.as_mut(),
        ) {
            let mut checked = |bytes| {
                let result = (**next)(bytes);
                if result.is_err() {
                    source_capacity_refused.set(true);
                }
                result
            };
            let result = crate::pricing::iris_certified_bounder_with_source_capacity(
                anchor_request,
                anchor_pricing,
                request,
                request_pricing,
                execution_facts,
                &mut checked,
            );
            // A provider may represent refused analysis as an incomplete-host gap.
            // The real boundary must still refuse Capacity, never reinterpret it
            // as successful native output or later turn it into NoEvidence.
            if source_capacity_refused.get() {
                return Err(mlx_gen::Error::Unsupported(
                    "source metadata capacity check refused before guarded repricing".into(),
                ));
            }
            return result;
        }
        // Existing native/metadata compatibility path remains separate. Iris's
        // five-arg wrapper must record missing capacity as a typed host-proof gap,
        // not fabricate a successful guarded metadata analysis or perform new reads.
        bounder(
            anchor_request,
            anchor_pricing,
            request,
            request_pricing,
            execution_facts,
        )
    };
    let measured = anchor::measured_anchor(base).map_err(no_evidence)?;
    let (base_context, at_base) = match source_metadata_capacity.as_mut() {
        Some(next) => {
            price_anchor_using_source_capacity(registry, candidate, base, Some(&mut **next))?
        }
        None => price_anchor(registry, candidate, base)?,
    };
    let request_side = IosBoundRequest {
        pricing: &request_context,
        replay: None,
    };
    let base_request = base_context.request();
    let certified = certify(
        &base_context.bound(&base_request)?,
        &at_base,
        &request_side,
        &request,
        &mut source_metadata_capacity,
    )
    .map_err(|e| {
        let reason = format!(
            "{key:?}: no transfer certificate from its {:?} anchor: {e}",
            base.anchor_kind,
        );
        if source_capacity_refused.get() {
            AdmissionError::refused(RefusalKind::Capacity, reason)
        } else {
            no_evidence(reason)
        }
    })?;
    let (bounds, mut premises) = in_anchor_domain(certified, &measured, premise_policy)
        .map_err(|e| no_evidence(format!("{key:?}: {e}")))?;
    let mapped = premises
        .iter()
        .any(|premise| premise.id == NATIVE_TO_PROCESS_PREMISE_ID);
    if mapped && resident.is_none() {
        return Err(no_evidence(format!(
            "{family:?}: the native-to-process engineering model must be validated against BOTH \
             measured endpoint anchors; this family has no completed resident anchor"
        )));
    }
    let derived =
        anchor::derive_certified(&measured, &at_base, &request, &bounds).map_err(|e| {
            no_evidence(format!(
                "{key:?} is not derivable from its {:?} anchor: {e}",
                base.anchor_kind
            ))
        })?;
    // The two endpoints check each other through the SAME certified consumer (and, when mapped,
    // the SAME approved model): an under-prediction of the measured resident anchor fails the
    // whole family's derivations closed. This is the model's two-endpoint validation.
    if let Some(resident) = resident {
        let (optimized_context, at_optimized) = match source_metadata_capacity.as_mut() {
            Some(next) => price_anchor_using_source_capacity(
                registry,
                candidate,
                optimized,
                Some(&mut **next),
            )?,
            None => price_anchor(registry, candidate, optimized)?,
        };
        let (resident_context, at_resident) = match source_metadata_capacity.as_mut() {
            Some(next) => price_anchor_using_source_capacity(
                registry,
                candidate,
                resident,
                Some(&mut **next),
            )?,
            None => price_anchor(registry, candidate, resident)?,
        };
        let optimized_request = optimized_context.request();
        let resident_request = resident_context.request();
        let endpoint = certify(
            &optimized_context.bound(&optimized_request)?,
            &at_optimized,
            &resident_context.bound(&resident_request)?,
            &at_resident,
            &mut source_metadata_capacity,
        )
        .map_err(|e| {
            let reason = format!("{family:?}: no endpoint transfer certificate: {e}");
            if source_capacity_refused.get() {
                AdmissionError::refused(RefusalKind::Capacity, reason)
            } else {
                no_evidence(reason)
            }
        })?;
        let optimized_measured = anchor::measured_anchor(optimized).map_err(no_evidence)?;
        let (endpoint_bounds, endpoint_premises) =
            in_anchor_domain(endpoint, &optimized_measured, premise_policy)
                .map_err(|e| no_evidence(format!("{family:?} endpoint check: {e}")))?;
        premises.extend(endpoint_premises);
        let check = anchor::consistency_certified(
            &optimized_measured,
            &at_optimized,
            &anchor::measured_anchor(resident).map_err(no_evidence)?,
            &at_resident,
            &endpoint_bounds,
        )
        .map_err(|e| no_evidence(format!("{family:?}: anchors not mutually derivable: {e}")))?;
        if check.fails_closed() {
            return Err(no_evidence(format!(
                "{family:?}: the optimized anchor under-predicts the measured resident anchor \
                 ({} B predicted vs {} B measured, {:+.1}% beyond the {:.1}% tolerance); \
                 derivations of this family fail closed",
                check.predicted,
                check.measured,
                check.relative_error * 100.0,
                anchor::CONSISTENCY_TOLERANCE * 100.0
            )));
        }
    }
    let margin = base
        .margin_bytes()
        .ok_or_else(|| no_evidence("no capture".into()))?;
    Ok((
        derived.demand,
        margin,
        AdmissionBasis::Derived {
            from: base.anchor_kind,
        },
        base.quantum_calibration(),
        {
            premises.sort();
            premises.dedup();
            premises
        },
    ))
}

/// Steps 3–6 for one candidate, under its own source inventory.
fn admit_one(
    registry: &ProviderRegistry,
    index: usize,
    candidate: &IosCandidate,
    machine: &str,
    budget: MemoryBudget,
    rows: Option<&[AnchorRow]>,
    evidence: AdmissionEvidence<'_>,
) -> AdmissionResult<Admission> {
    admit_one_using_key_capacity(
        registry,
        index,
        candidate,
        machine,
        budget,
        rows,
        AdmissionAttempt {
            evidence,
            next: None,
        },
    )
}

struct AdmissionAttempt<'facts, 'capacity> {
    evidence: AdmissionEvidence<'facts>,
    next: Option<KeyCapacity<'capacity>>,
}

fn admit_one_using_key_capacity(
    registry: &ProviderRegistry,
    index: usize,
    candidate: &IosCandidate,
    machine: &str,
    budget: MemoryBudget,
    rows: Option<&[AnchorRow]>,
    attempt: AdmissionAttempt<'_, '_>,
) -> AdmissionResult<Admission> {
    admit_one_with_retirement(
        registry,
        index,
        candidate,
        machine,
        budget,
        RetirementInput {
            rows,
            simulator_functional: false,
        },
        attempt,
    )
}

struct RetirementInput<'a> {
    rows: Option<&'a [AnchorRow]>,
    simulator_functional: bool,
}

fn admit_one_with_retirement(
    registry: &ProviderRegistry,
    index: usize,
    candidate: &IosCandidate,
    machine: &str,
    budget: MemoryBudget,
    retirement: RetirementInput<'_>,
    attempt: AdmissionAttempt<'_, '_>,
) -> AdmissionResult<Admission> {
    let RetirementInput {
        rows,
        simulator_functional,
    } = retirement;
    let AdmissionAttempt { evidence, mut next } = attempt;
    validate_actual(registry, candidate)?;
    let native_retirement = if !simulator_functional
        && matches!(
            candidate.provider.as_str(),
            "sana_1600m" | "sana_sprint_1600m"
        )
        && matches!(&candidate.work,IosWork::Generate(request) if request.memory.is_some())
        && (rows.is_none()
            || evidence.execution_facts.is_some_and(|facts| {
                facts.native_retirement
                    == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
            })) {
        mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
    } else {
        mlx_gen::ios_pricing::IosNativeRetirementExecution::Ordinary
    };
    let route = route_of(candidate)?;
    if let Some(rows) = rows {
        // Cheap pre-check: refuse before opening any source when no anchor can match.
        if !rows
            .iter()
            .any(|r| r.key.machine == machine && r.key.tier == tier_label(&candidate.spec))
        {
            return Err(AdmissionError::refused(
                RefusalKind::NoEvidence,
                format!(
                    "{}: no reviewed device anchor for route {route} on {machine} in \
                     {EVIDENCE_TABLE_PATH} (source unopened)",
                    candidate.provider
                ),
            ));
        }
    }
    if let Some(rows) = rows {
        let tier = tier_label(&candidate.spec);
        if let Some(bytes) = evidence
            .premise_policy
            .preparation_envelope(
                rows,
                crate::engineering::PreparationScope {
                    machine,
                    provider: &candidate.provider,
                    route: &route,
                    mode: candidate.context.mode.as_key(),
                    tier: &tier,
                },
            )
            .map_err(no_evidence)?
        {
            let Some(next) = next.as_mut() else {
                return Err(no_evidence(
                    "engineering estimate requires the actual source-preparation capacity callback"
                        .into(),
                ));
            };
            with_source_metadata_capacity(&mut **next, |next| next(bytes))?;
        }
    }
    let fresh_source;
    let guard = if let Some(source) = candidate.spec.source_inventory() {
        match next.as_mut() {
            Some(next) => with_source_inventory_capacity(&mut **next, |next| {
                source.verify_for_with_capacity(&candidate.spec, next)
            })?,
            None => source.verify_for(&candidate.spec)?,
        }
        source
    } else {
        fresh_source = match next.as_mut() {
            Some(next) => with_source_inventory_capacity(&mut **next, |next| {
                SourceInventory::capture_with_capacity(&candidate.spec, next)
            })?,
            None => SourceInventory::capture(&candidate.spec)?,
        };
        &fresh_source
    };
    // One inner body; the outer wrapper supplies guarded before/after source checks
    // for the real SDK/capture path and preserves native legacy semantics separately.
    let consume = |mut next: Option<KeyCapacity<'_>>| -> AdmissionResult<Admission> {
        // The actual full Clone and inventory binding are behind their own
        // finite requested-layout laws, not a preflight followed by unchecked
        // copies. Keep the SAME mutable capacity callback for every owner.
        let mut spec = match next.as_mut() {
            Some(next) => with_source_inventory_capacity(&mut **next, |core_next| {
                candidate.spec.clone_with_capacity(core_next)
            })?,
            None => candidate.spec.clone(),
        };
        if spec.source_inventory().is_none() {
            match next.as_mut() {
                Some(next) => with_source_inventory_capacity(&mut **next, |core_next| {
                    let source = guard.clone_with_capacity(core_next)?;
                    spec.bind_source_inventory_with_capacity(source, core_next)
                })?,
                None => spec.bind_source_inventory((*guard).clone())?,
            }
        }
        // The retained inventory is bound before request AND endpoint metadata pricing.
        // Reborrow the SAME callback for the key, retaining it for both guarded
        // request/anchor repricings below. Never spend it with Option::map.
        let mut key = match next.as_mut() {
            Some(next) => key_unchecked_using_key_capacity(
                candidate,
                machine,
                route.clone(),
                Some((guard, &mut **next)),
            ),
            None => key_unchecked(candidate, machine, route.clone()),
        }?;
        match next.as_mut() {
            Some(next) => {
                bind_sana_retirement_key(&mut key, candidate, native_retirement, Some(&mut **next))?
            }
            None => bind_sana_retirement_key(&mut key, candidate, native_retirement, None)?,
        }
        if native_retirement
            == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
            && evidence
                .execution_facts
                .is_some_and(|facts| !facts.ios_quantum_scope)
        {
            return Err(no_evidence(
                "required SANA native lifecycle has no actual quantum execution scope".into(),
            ));
        }
        if let (Some(binding), IosWork::Generate(request)) = (&key.sdxl_work, &candidate.work) {
            binding.validate_actual(request).map_err(pricing_error)?;
        }
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
        let expected = execution_for(&candidate.provider, &contract, &candidate.context.selection);
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
        let (demand_bytes, margin_bytes, basis, calibration, premises) = match rows {
            None => (
                0,
                0,
                AdmissionBasis::CapturePermit,
                Calibration::new(),
                Vec::new(),
            ),
            Some(rows) => match next.as_mut() {
                Some(next) => anchored_demand_using_source_capacity(
                    registry,
                    (candidate, &spec),
                    &contract,
                    &key,
                    rows,
                    evidence,
                    Some(&mut **next),
                )?,
                None => anchored_demand(registry, candidate, &contract, &key, rows, evidence)?,
            },
        };
        let required_bytes = anchor::with_margin(demand_bytes, margin_bytes).map_err(|error| {
            no_evidence(format!("complete admitted envelope arithmetic: {error}"))
        })?;
        let mut context = match next.as_mut() {
            Some(next) => source_context_clone(&candidate.context, Some(&mut **next))?,
            None => candidate.context.clone(),
        };
        if rows.is_some() {
            context.budget = budget;
            // The render's own growth plus the anchor's measured margin; what the process already
            // holds is the budget's committed bytes and is never credited as reusable.
            context.predicted_peak_bytes = required_bytes;
            // Preserve the exact approved identity without a temporary IDs Vec
            // or unchecked format/String growth in the guarded path.
            context.evidence_revision = match next.as_mut() {
                Some(next) => source_evidence_revision(&key, &premises, Some(&mut **next))?,
                None => source_evidence_revision(&key, &premises, None)?,
            };
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
            context.evidence_revision = match next.as_mut() {
                Some(next) => {
                    source_string_clone("ios-capture-permit:no-evidence-row", Some(&mut **next))?
                }
                None => "ios-capture-permit:no-evidence-row".into(),
            };
        }
        let provider = match next.as_mut() {
            Some(next) => source_string_clone(&candidate.provider, Some(&mut **next))?,
            None => candidate.provider.clone(),
        };
        Ok(Admission {
            index,
            provider,
            demand_bytes,
            margin_bytes,
            basis,
            calibration,
            spec,
            context,
            binding: work_binding(&candidate.work),
            key,
            premises,
            native_retirement,
        })
    };
    match next {
        Some(next) => with_source_inventory_capacity(next, |next| {
            Ok(guard.read_unchanged_with_capacity(next, |same_next| {
                let mut provider_next = |bytes| same_next(bytes).map_err(mlx_gen::Error::from);
                consume(Some(&mut provider_next))
            }))
        })?,
        None => guard.read_unchanged(|| consume(None)),
    }
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
    rows: &[AnchorRow],
) -> AdmissionResult<Admission> {
    admit_rows_at_boundary(registry, candidates, machine, budget, rows, None)
}

pub(crate) fn admit_rows_at_boundary(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    rows: &[AnchorRow],
    request_baseline: Option<&IosMemoryBaseline>,
) -> AdmissionResult<Admission> {
    admit_rows_with_facts(
        registry,
        candidates,
        machine,
        budget,
        rows,
        AdmissionEvidence {
            request_baseline,
            execution_facts: None,
            premise_policy: &PremisePolicy::default(),
        },
    )
}

/// The complete SDK boundary: the process baseline AND the certified execution facts. Absent
/// facts never default; derivation then refuses exactly while exact anchor keys still admit.
pub(crate) fn admit_rows_with_facts(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    rows: &[AnchorRow],
    evidence: AdmissionEvidence<'_>,
) -> AdmissionResult<Admission> {
    admit_rows_with_facts_using_key_capacity(
        registry, candidates, machine, budget, rows, evidence, None,
    )
}

fn admit_rows_with_facts_using_key_capacity(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    rows: &[AnchorRow],
    evidence: AdmissionEvidence<'_>,
    next: Option<KeyCapacity<'_>>,
) -> AdmissionResult<Admission> {
    admit_rows_using_lifecycle_policy(
        registry,
        candidates,
        machine,
        budget,
        rows,
        evidence,
        AdmissionRunPolicy {
            next,
            sana_fallback: false,
        },
    )
}

struct AdmissionRunPolicy<'a> {
    next: Option<KeyCapacity<'a>>,
    sana_fallback: bool,
}
fn admit_rows_using_lifecycle_policy(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    rows: &[AnchorRow],
    evidence: AdmissionEvidence<'_>,
    policy: AdmissionRunPolicy<'_>,
) -> AdmissionResult<Admission> {
    let AdmissionRunPolicy {
        mut next,
        sana_fallback,
    } = policy;
    if let Some(baseline) = evidence.request_baseline {
        if baseline.domain != IosMemoryDomain::ProcessGrowth
            || baseline.coordinate != IosMemoryCoordinate::BeforeModelLoad
            || baseline.bytes == 0
            || baseline.source.is_empty()
            || baseline.bytes != budget.committed_bytes
        {
            return Err(no_evidence("request baseline must be the actual BeforeModelLoad process snapshot used by the committed budget".into()));
        }
    }
    evidence::validate_rows(rows).map_err(no_evidence)?;
    let mut first_refusal: Option<RefusalKind> = None;
    let mut first_failure: Option<Error> = None;
    let mut reasons = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let mut result = match next.as_mut() {
            Some(next) => admit_one_using_key_capacity(
                registry,
                index,
                candidate,
                machine,
                budget,
                Some(rows),
                AdmissionAttempt {
                    evidence,
                    next: Some(&mut **next),
                },
            ),
            None => admit_one(
                registry,
                index,
                candidate,
                machine,
                budget,
                Some(rows),
                evidence,
            ),
        };
        // Preserve candidate preference AND existing ordinary measurements:
        // attempt ordinary first for THIS candidate, then only its explicitly
        // supported joined program. Never replace a malformed-artifact failure
        // with a speculative fallback or try a lower-preference tier first.
        if sana_fallback
            && evidence.execution_facts.is_some_and(|facts| {
                facts.native_retirement
                    == mlx_gen::ios_pricing::IosNativeRetirementExecution::Ordinary
            })
            && matches!(
                candidate.provider.as_str(),
                "sana_1600m" | "sana_sprint_1600m"
            )
            && matches!(&candidate.work,IosWork::Generate(request) if request.memory.is_some())
            && matches!(&result,Err(AdmissionError::Refused(refusal)) if matches!(refusal.kind,RefusalKind::NoEvidence|RefusalKind::Capacity))
        {
            let actual = evidence
                .execution_facts
                .expect("explicit ordinary execution facts checked above");
            let mut native = *actual;
            native.native_retirement =
                mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle;
            let native_evidence = AdmissionEvidence {
                execution_facts: Some(&native),
                ..evidence
            };
            result = match next.as_mut() {
                Some(next) => admit_one_using_key_capacity(
                    registry,
                    index,
                    candidate,
                    machine,
                    budget,
                    Some(rows),
                    AdmissionAttempt {
                        evidence: native_evidence,
                        next: Some(&mut **next),
                    },
                ),
                None => admit_one(
                    registry,
                    index,
                    candidate,
                    machine,
                    budget,
                    Some(rows),
                    native_evidence,
                ),
            };
        }
        match result {
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

/// Admission with the real SDK request boundary. The snapshot is a coordinate fact, not a
/// certificate that native allocation growth converts to process footprint. Pure metadata callers
/// that cannot supply this fact use `admit`; they must not invent a zero baseline.
pub fn admit_with_baseline(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    request_baseline: &IosMemoryBaseline,
) -> AdmissionResult<Admission> {
    let rows = evidence::rows().map_err(no_evidence)?;
    admit_rows_at_boundary(
        registry,
        candidates,
        machine,
        budget,
        rows,
        Some(request_baseline),
    )
}

/// Admission with the full real SDK boundary: the BeforeModelLoad process snapshot and the facts
/// of the execution that will run (stream family, iOS quantum scope). Derivation away from an
/// anchor's own key requires a registered provider bounder that certifies those facts AND
/// certificates in the measured anchors' counter domain; otherwise it refuses exactly.
pub fn admit_with_boundary(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    request_baseline: &IosMemoryBaseline,
    execution_facts: &IosExecutionFacts,
    premise_policy: &PremisePolicy,
) -> AdmissionResult<Admission> {
    let rows = evidence::rows().map_err(no_evidence)?;
    admit_rows_with_facts(
        registry,
        candidates,
        machine,
        budget,
        rows,
        AdmissionEvidence {
            request_baseline: Some(request_baseline),
            execution_facts: Some(execution_facts),
            premise_policy,
        },
    )
}

/// The real SDK boundary facts and its borrowed byte-sensitive source-allocation check.
/// No baseline, execution fact, premise approval or capacity default is invented here.
pub struct SourceCapacityBoundary<'a> {
    /// Actual BeforeModelLoad process snapshot, unchanged from the SDK boundary.
    pub request_baseline: &'a IosMemoryBaseline,
    /// Execution facts for the operation that will really run.
    pub execution_facts: &'a IosExecutionFacts,
    /// Explicit caller approvals, never expanded by this context.
    pub premise_policy: &'a PremisePolicy,
    /// Real byte-sensitive NEXT requested-layout check; no unlimited default.
    pub source_metadata_capacity: &'a mut dyn FnMut(u64) -> mlx_gen::Result<()>,
}

/// Borrowed metadata for the ACTUAL selected admission, before any model payload is loaded.
/// Observing or constructing this view confers no load authority and approves no premise.
/// An observer may refuse known recorder/storage work using the same next-allocation guard.
pub struct SelectedAdmissionView<'a> {
    pub candidate: &'a IosCandidate,
    pub admission: &'a Admission,
    pub boundary: SourceCapacityBoundary<'a>,
}

impl std::fmt::Debug for SourceCapacityBoundary<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceCapacityBoundary")
            .field("request_baseline", self.request_baseline)
            .field("execution_facts", self.execution_facts)
            .field("premise_policy", self.premise_policy)
            .finish_non_exhaustive()
    }
}

/// Full SDK boundary plus the real byte-sensitive guard before fresh key tokenization.
/// Execution facts, baseline and premise policy are the SAME facts used for admission.
pub fn admit_with_boundary_and_key_capacity(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    boundary: SourceCapacityBoundary<'_>,
) -> AdmissionResult<Admission> {
    let rows = evidence::rows().map_err(no_evidence)?;
    admit_rows_with_facts_using_key_capacity(
        registry,
        candidates,
        machine,
        budget,
        rows,
        AdmissionEvidence {
            request_baseline: Some(boundary.request_baseline),
            execution_facts: Some(boundary.execution_facts),
            premise_policy: boundary.premise_policy,
        },
        Some(boundary.source_metadata_capacity),
    )
    .map_err(source_metadata_gap_refusal)
}

/// SDK policy: keep every candidate's existing ordinary measured/derived path
/// first, then its required SANA native lifecycle only when ordinary refused.
/// This preserves caller preference, legacy None, and valid ordinary evidence.
/// The selected receipt—not this preference—carries the execution requirement.
pub fn admit_with_sana_lifecycle_fallback(
    registry: &ProviderRegistry,
    candidates: &[IosCandidate],
    machine: &str,
    budget: MemoryBudget,
    boundary: SourceCapacityBoundary<'_>,
) -> AdmissionResult<Admission> {
    let rows = evidence::rows().map_err(no_evidence)?;
    admit_rows_using_lifecycle_policy(
        registry,
        candidates,
        machine,
        budget,
        rows,
        AdmissionEvidence {
            request_baseline: Some(boundary.request_baseline),
            execution_facts: Some(boundary.execution_facts),
            premise_policy: boundary.premise_policy,
        },
        AdmissionRunPolicy {
            next: Some(boundary.source_metadata_capacity),
            sana_fallback: true,
        },
    )
    .map_err(source_metadata_gap_refusal)
}

/// Explicitly-labelled capture permit. NOT admission: no evidence row is required (that would be
/// circular) and none is created. Granted only where `mlx_gen::ios_quantum` bounds every quantum of
/// this model by live OS headroom BEFORE its graph is constructed; a refusal at quantum k of size N
/// is the "next size refused before jetsam" evidence. Never raises the cap.
#[cfg(feature = "device-capture")]
pub fn capture_permit(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    machine: &str,
) -> AdmissionResult<Admission> {
    capture_permit_using_key_capacity(registry, candidate, machine, None, false)
}

/// The served device-capture route, guarded before fresh key tokenization.
#[cfg(feature = "device-capture")]
pub fn capture_permit_with_key_capacity(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    machine: &str,
    next: KeyCapacity<'_>,
) -> AdmissionResult<Admission> {
    capture_permit_using_key_capacity(registry, candidate, machine, Some(next), false)
        .map_err(source_metadata_gap_refusal)
}

/// Functional-only simulator permit. It has no phone admission, cap, or quantum claim. The
/// separately dispatched simulator controller must own and bound the app before release.
#[cfg(all(feature = "simulator-functional", target_abi = "sim"))]
pub fn simulator_functional_permit(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    machine: &str,
) -> AdmissionResult<Admission> {
    capture_permit_using_key_capacity(registry, candidate, machine, None, true)
}

#[cfg(feature = "device-capture")]
fn capture_permit_using_key_capacity(
    registry: &ProviderRegistry,
    candidate: &IosCandidate,
    machine: &str,
    mut next: Option<KeyCapacity<'_>>,
    simulator_functional: bool,
) -> AdmissionResult<Admission> {
    if !mlx_gen::ios_quantum::enabled() && !simulator_functional {
        return Err(AdmissionError::Failed(Error::Unsupported(
            "iOS capture requires bounded per-quantum execution (a physical iOS device); \
             no weights loaded"
                .into(),
        )));
    }
    let premise_policy = PremisePolicy::default();
    let permit = admit_one_with_retirement(
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
        RetirementInput {
            rows: None,
            simulator_functional,
        },
        AdmissionAttempt {
            evidence: AdmissionEvidence {
                request_baseline: None,
                execution_facts: None,
                premise_policy: &premise_policy,
            },
            next: next.as_mut().map(|next| &mut **next as KeyCapacity<'_>),
        },
    )?;
    if !simulator_functional
        && matches!(permit.provider.as_str(), "sana_1600m" | "sana_sprint_1600m")
    {
        // The real key/inventory already exists. This header/decoder-plan preflight must precede
        // runtime capture observers, whose source-event planning may construct a tokenizer.
        let IosWork::Generate(request) = &candidate.work else {
            return Err(AdmissionError::refused(
                RefusalKind::Mode,
                "SANA capture needs generation work",
            ));
        };
        match next {
            Some(next) => with_source_inventory_capacity(next, |core_next| {
                permit
                    .spec
                    .read_prepared_files_unchanged_with_capacity(core_next, |_| {
                        mlx_gen_sana::model::preflight_resident_request(
                            &permit.spec,
                            &permit.provider,
                            request,
                        )
                        .map_err(Error::from)
                    })
            }),
            None => permit
                .spec
                .read_prepared_files_unchanged(|| {
                    mlx_gen_sana::model::preflight_resident_request(
                        &permit.spec,
                        &permit.provider,
                        request,
                    )
                    .map_err(Error::from)
                })
                .map_err(AdmissionError::Failed),
        }?;
    }
    Ok(permit)
}

impl Admission {
    /// The measured envelope an admitted (non-capture) load and its renders run under, opened
    /// before the load so it spans the weights too.
    fn envelope(&self) -> Option<mlx_gen::ios_quantum::AdmittedEnvelope> {
        (!self.is_capture()).then(|| {
            mlx_gen::ios_quantum::open_admitted_envelope(
                self.required_bytes(),
                self.calibration.clone(),
                self.margin_bytes,
            )
        })
    }

    fn check_load_form(&self) -> Result<()> {
        if matches!(self.provider.as_str(), "sana_1600m" | "sana_sprint_1600m")
            && (self.native_retirement
                == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle)
                != self
                    .key
                    .execution
                    .ends_with(mlx_gen_sana::ios_pricing::NATIVE_RETIREMENT_EXECUTION_SUFFIX)
        {
            return Err(Error::Unsupported(
                "sealed SANA execution requirement disagrees with its key".into(),
            ));
        }
        if self.native_retirement
            == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
            && !mlx_gen::ios_quantum::enabled()
        {
            return Err(Error::Unsupported(
                "required SANA native lifecycle lost quantum scope before model load".into(),
            ));
        }
        let prefix = format!("{:?}|{:?}|", self.spec.offload_policy, self.spec.load_shape);
        if self.context.load_shape != self.spec.load_shape
            || !self.key.execution.starts_with(&prefix)
        {
            return Err(Error::Unsupported(
                "admitted load form disagrees with its execution identity".into(),
            ));
        }
        Ok(())
    }

    fn check_kind(&self, expected: &str) -> Result<()> {
        self.check_load_form()?;
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
        self.check_load_form()?;
        capacity(self.required_bytes())?;
        let streams = if self.native_retirement
            == mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
        {
            // Resolve/drop allocating device metadata BEFORE any model load or
            // critical cleanup. The initial range is preexisting stream work,
            // not attributed to this request and never credited as its demand.
            let streams = mlx_gen::ios_quantum::NativeRetirementStreams::prepare()?;
            streams.drain()?;
            Some(streams)
        } else {
            None
        };
        let envelope = self.envelope();
        let required = self.required_bytes();
        let inner = match self.spec.read_prepared_files_unchanged(|| load(&self.spec)) {
            Ok(inner) => inner,
            Err(primary) => {
                if let Some(streams) = &streams {
                    if let Err(cleanup) = streams.drain() {
                        eprintln!("native load-error cleanup failed: {cleanup}");
                    }
                }
                return Err(primary);
            }
        };
        Ok(crate::bound::BoundGenerator::new(
            inner,
            self.spec,
            self.context,
            self.binding,
            required,
            capacity,
            crate::bound::GeneratorExecution {
                info: crate::bound::GeneratorAdmissionInfo {
                    index: self.index,
                    key: self.key,
                    demand_bytes: self.demand_bytes,
                    margin_bytes: self.margin_bytes,
                    required_bytes: required,
                    basis: self.basis,
                },
                envelope,
                native_retirement: self.native_retirement,
                streams,
            },
        ))
    }

    /// Load the admitted Iris restoration transform.
    pub fn load_restorer(
        self,
        registry: &ProviderRegistry,
        capacity: impl Fn(u64) -> Result<()> + 'static,
    ) -> Result<crate::bound::BoundTransform> {
        self.check_kind("restoration")?;
        capacity(self.required_bytes())?;
        let envelope = self.envelope();
        let required = self.required_bytes();
        let provider = self.provider.clone();
        let inner = self
            .spec
            .read_prepared_files_unchanged(|| registry.load_transform(&provider, &self.spec))?;
        Ok(crate::bound::BoundTransform::new(
            inner,
            self.spec,
            self.binding,
            required,
            Box::new(capacity),
            envelope,
        ))
    }

    /// Load the admitted Iris depth task.
    pub fn load_depth(
        self,
        capacity: impl Fn(u64) -> Result<()> + 'static,
    ) -> Result<crate::bound::BoundDepth> {
        self.check_kind("depth")?;
        capacity(self.required_bytes())?;
        let envelope = self.envelope();
        let required = self.required_bytes();
        let inner = self.spec.read_prepared_files_unchanged(|| {
            mlx_gen_iris::depth::load_depth(&self.spec).map_err(Error::from)
        })?;
        Ok(crate::bound::BoundDepth::new(
            Box::new(inner),
            self.spec,
            self.binding,
            required,
            Box::new(capacity),
            envelope,
        ))
    }
}

#[cfg(test)]
mod guarded_clone_tests {
    use super::*;
    use crate::binding::tests::allocations_during;
    use std::cell::Cell;

    #[test]
    fn guarded_string_checks_utf8_requested_bytes_before_its_only_allocation() {
        let parts = ["route/", "日本語", "/tail"];
        let bill = Cell::new(0);
        let mut next = |bytes| {
            bill.set(bytes);
            Ok(())
        };
        let (result, allocated) =
            allocations_during(|| source_string_parts(parts.into_iter(), &mut next));
        let value = result.unwrap();
        assert_eq!(value, "route/日本語/tail");
        assert_eq!(bill.get(), value.len() as u64);
        assert_eq!(
            allocated as u64,
            bill.get(),
            "unbilled temporary/growth allocation"
        );
    }

    #[test]
    fn guarded_string_refusal_precedes_large_owner_creation() {
        let original = "x".repeat(1 << 20);
        let calls = Cell::new(0);
        let mut next = |bytes| {
            assert_eq!(bytes, original.len() as u64);
            calls.set(calls.get() + 1);
            Err(mlx_gen::Error::Canceled)
        };
        let (result, allocated) =
            allocations_during(|| source_string_clone(&original, Some(&mut next)));
        assert!(
            matches!(result, Err(AdmissionError::Refused(ref refusal)) if refusal.kind == RefusalKind::Capacity)
        );
        assert_eq!(calls.get(), 1);
        assert!(
            allocated < original.len(),
            "copy builder ran after capacity refusal"
        );
    }

    #[test]
    fn guarded_context_clone_bills_all_actual_string_owners() {
        let registry = crate::provider_registry().unwrap();
        let root = crate::sdxl::tests::fixture();
        let mut context = crate::sdxl::tests::candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            None,
            "text_to_image",
        )
        .context;
        context.calibration_fingerprint = "fingerprint".repeat(8);
        context.mode = MemoryMode::Other("mode".repeat(9));
        context.overlay = Some("overlay".repeat(7));
        context.evidence_revision = "previous".repeat(11);
        context.reference_resources = Some(mlx_gen::gen_core::MemoryReferenceResourceFacts {
            conditioning_tokens: 4096,
            prepared_canvases: vec![(512, 512), (768, 512), (1024, 1024)],
        });
        let bill = Cell::new(0);
        let mut next = |bytes| {
            bill.set(bytes);
            Ok(())
        };
        let (result, allocated) =
            allocations_during(|| source_context_clone(&context, Some(&mut next)));
        assert_eq!(result.unwrap(), context);
        assert_eq!(allocated as u64, bill.get());
        assert!(bill.get() > 0);
    }

    #[test]
    fn guarded_evidence_revision_preserves_exact_legacy_identity_without_ids_vec() {
        let registry = crate::provider_registry().unwrap();
        let root = crate::sdxl::tests::fixture();
        let candidate = crate::sdxl::tests::candidate(
            &registry,
            root.path(),
            "sdxl",
            512,
            None,
            "text_to_image",
        );
        let key =
            crate::source_work::fixture_key(&registry, &candidate, crate::sdxl::tests::DEVICE);
        let premises = [
            IosCertificatePremise {
                id: "first".into(),
                statement: "statement must not be copied".repeat(100),
            },
            IosCertificatePremise {
                id: "日本語".into(),
                statement: "other".into(),
            },
        ];
        for count in 0..=premises.len() {
            let expected = source_evidence_revision(&key, &premises[..count], None).unwrap();
            let bill = Cell::new(0);
            let mut next = |bytes| {
                bill.set(bytes);
                Ok(())
            };
            let (result, allocated) = allocations_during(|| {
                source_evidence_revision(&key, &premises[..count], Some(&mut next))
            });
            let value = result.unwrap();
            assert_eq!(value, expected);
            assert_eq!(bill.get(), value.len() as u64);
            assert_eq!(
                allocated as u64,
                bill.get(),
                "temporary IDs or format growth escaped bill"
            );
        }
    }
}

#[cfg(test)]
#[path = "ios_admission_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "anchor_admission_tests.rs"]
mod anchor_tests;

#[cfg(all(test, target_os = "macos"))]
mod original_z_execution_sidecar_tests {
    use super::*;
    use crate::evidence::AnchorKind;
    use mlx_gen::gen_core::{ControlKind, Image, MemoryStrategy, Quant, WeightsSource};
    use sha2::{Digest, Sha256};
    mod shared {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../mlx-gen-z-image/tests/common/original_z_execution_sidecar.rs"
        ));
    }
    use shared::{Inputs, ManualMode, Position};

    fn image() -> Image {
        Image {
            width: 32,
            height: 32,
            pixels: vec![19; 32 * 32 * 3],
        }
    }

    fn private_execution_binding(core: [u8; 32], program: &serde_json::Value) -> [u8; 32] {
        let mut hash = Sha256::new();
        hash.update(b"z-image-original-private-execution-sidecar-v1\0");
        hash.update(core);
        hash.update(serde_json::to_vec(program).unwrap());
        hash.finalize().into()
    }

    /// Full canonical constructor inventory for this PUBLIC metadata-only fixture. No Array,
    /// coefficient payload, model load or native query is created. FF=2*dim and the VAE channel
    /// ladder deliberately follow the existing original fixture's valid topology; this is not a
    /// claim that sparse zero payload can execute or that it is the historical private program.
    struct PublicHeaders(std::collections::BTreeMap<String, (&'static str, Vec<usize>)>);
    impl PublicHeaders {
        fn new() -> Self {
            Self(std::collections::BTreeMap::new())
        }
        fn put(&mut self, name: impl Into<String>, dtype: &'static str, shape: &[usize]) {
            assert!(
                self.0
                    .insert(name.into(), (dtype, shape.to_vec()))
                    .is_none(),
                "PUBLIC_HEADER_UNIQUE_NAME"
            );
        }
        fn linear(&mut self, path: &str, output: usize, input: usize, packed: bool, bias: bool) {
            if packed {
                assert_eq!(input % 64, 0, "PUBLIC_Q8_GROUP_ALIGNMENT");
                self.put(format!("{path}.weight"), "U32", &[output, input / 4]);
                self.put(format!("{path}.scales"), "BF16", &[output, input / 64]);
                self.put(format!("{path}.biases"), "BF16", &[output, input / 64]);
            } else {
                self.put(format!("{path}.weight"), "BF16", &[output, input]);
            }
            if bias {
                self.put(format!("{path}.bias"), "BF16", &[output]);
            }
        }
        fn block(&mut self, path: &str, dim: usize, heads: usize, modulation: bool, packed: bool) {
            for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                self.linear(&format!("{path}.attention.{leaf}"), dim, dim, packed, false);
            }
            for leaf in ["norm_q", "norm_k"] {
                self.put(
                    format!("{path}.attention.{leaf}.weight"),
                    "BF16",
                    &[dim / heads],
                );
            }
            for leaf in [
                "attention_norm1",
                "attention_norm2",
                "ffn_norm1",
                "ffn_norm2",
            ] {
                self.put(format!("{path}.{leaf}.weight"), "BF16", &[dim]);
            }
            self.linear(
                &format!("{path}.feed_forward.w1"),
                2 * dim,
                dim,
                packed,
                false,
            );
            self.linear(
                &format!("{path}.feed_forward.w2"),
                dim,
                2 * dim,
                packed,
                false,
            );
            self.linear(
                &format!("{path}.feed_forward.w3"),
                2 * dim,
                dim,
                packed,
                false,
            );
            if modulation {
                self.linear(
                    &format!("{path}.adaLN_modulation.0"),
                    4 * dim,
                    dim,
                    packed,
                    true,
                );
            }
        }
        fn norm(&mut self, path: &str, channels: usize) {
            self.put(format!("{path}.weight"), "BF16", &[channels]);
            self.put(format!("{path}.bias"), "BF16", &[channels]);
        }
        fn conv(&mut self, path: &str, output: usize, input: usize, kernel: usize) {
            self.put(
                format!("{path}.weight"),
                "BF16",
                &[output, input, kernel, kernel],
            );
            self.put(format!("{path}.bias"), "BF16", &[output]);
        }
        fn resnet(&mut self, path: &str, input: usize, output: usize) {
            self.norm(&format!("{path}.norm1"), input);
            self.conv(&format!("{path}.conv1"), output, input, 3);
            self.norm(&format!("{path}.norm2"), output);
            self.conv(&format!("{path}.conv2"), output, output, 3);
            if input != output {
                self.conv(&format!("{path}.conv_shortcut"), output, input, 1);
            }
        }
        fn mid(&mut self, path: &str, channels: usize) {
            self.resnet(&format!("{path}.resnets.0"), channels, channels);
            self.norm(&format!("{path}.attentions.0.group_norm"), channels);
            for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
                self.linear(
                    &format!("{path}.attentions.0.{leaf}"),
                    channels,
                    channels,
                    true,
                    true,
                );
            }
            self.resnet(&format!("{path}.resnets.1"), channels, channels);
        }
        fn save(&self, path: &std::path::Path) {
            use std::io::Write;
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut offset = 0u64;
            let mut header = serde_json::Map::new();
            for (name, (dtype, shape)) in &self.0 {
                let width = match *dtype {
                    "U32" => 4u64,
                    "BF16" => 2,
                    _ => unreachable!(),
                };
                let bytes = shape
                    .iter()
                    .try_fold(width, |n, &d| n.checked_mul(u64::try_from(d).unwrap()))
                    .unwrap();
                let end = offset.checked_add(bytes).unwrap();
                header.insert(
                    name.clone(),
                    serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,end]}),
                );
                offset = end;
            }
            let encoded = serde_json::to_vec(&header).unwrap();
            let mut file = std::fs::File::create(path).unwrap();
            gen_core_testkit::mark_sparse(path);
            file.write_all(&(encoded.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&encoded).unwrap();
            let header_bytes = u64::try_from(encoded.len()).unwrap();
            assert!(
                header_bytes <= 100_000_000,
                "PUBLIC_FIXTURE_TESTKIT_HEADER_LIMIT"
            );
            let logical = 8u64
                .checked_add(header_bytes)
                .and_then(|n| n.checked_add(offset))
                .unwrap();
            file.set_len(logical).unwrap();
        }
    }

    fn public_te_epoch(root: &std::path::Path) -> (u64, u64, std::time::SystemTime) {
        use std::os::unix::fs::MetadataExt;
        let metadata = std::fs::metadata(root.join("text_encoder/model.safetensors")).unwrap();
        (metadata.ino(), metadata.len(), metadata.modified().unwrap())
    }

    fn complete_public_metadata_fixture(root: &std::path::Path, control: &std::path::Path) {
        // Retain the exact existing production TE contract/tokenizer sparse writer. Replace BOTH
        // placeholder component files and Control's placeholder with complete actual schemas.
        crate::zimage::tests::snapshot(root, Some(8));
        let cfg = mlx_gen_z_image::ZImageTransformerConfig::turbo();
        let dim = usize::try_from(cfg.dim).unwrap();
        let heads = usize::try_from(cfg.n_heads).unwrap();
        let patch = usize::try_from(cfg.patch_size * cfg.patch_size * cfg.in_channels).unwrap();
        let mut base = PublicHeaders::new();
        base.linear("all_x_embedder.2-1", dim, patch, true, true);
        base.linear(
            "t_embedder.mlp.0",
            1024,
            usize::try_from(cfg.frequency_embedding_size).unwrap(),
            true,
            true,
        );
        base.linear("t_embedder.mlp.2", dim, 1024, true, true);
        base.put(
            "cap_embedder.0.weight",
            "BF16",
            &[usize::try_from(cfg.cap_feat_dim).unwrap()],
        );
        base.linear(
            "cap_embedder.1",
            dim,
            usize::try_from(cfg.cap_feat_dim).unwrap(),
            true,
            true,
        );
        base.put("x_pad_token", "BF16", &[1, dim]);
        base.put("cap_pad_token", "BF16", &[1, dim]);
        base.linear("all_final_layer.2-1.linear", patch, dim, true, true);
        base.linear(
            "all_final_layer.2-1.adaLN_modulation.1",
            dim,
            dim,
            true,
            true,
        );
        for i in 0..cfg.n_layers {
            base.block(&format!("layers.{i}"), dim, heads, true, true);
        }
        for i in 0..cfg.n_refiner_layers {
            base.block(&format!("noise_refiner.{i}"), dim, heads, true, true);
            base.block(&format!("context_refiner.{i}"), dim, heads, false, true);
        }
        base.save(&root.join("transformer/model.safetensors"));
        std::fs::write(root.join("transformer/config.json"), serde_json::json!({"_class_name":"ZImageTransformer2DModel",
            "dim":cfg.dim,"n_layers":cfg.n_layers,"n_refiner_layers":cfg.n_refiner_layers,"n_heads":cfg.n_heads,
            "patch_size":cfg.patch_size,"f_patch_size":cfg.f_patch_size,"in_channels":cfg.in_channels,
            "cap_feat_dim":cfg.cap_feat_dim,"fixture_ff_hidden":2*cfg.dim,
            "quantization":{"bits":8,"group_size":64}}).to_string()).unwrap();
        let mut c = PublicHeaders::new();
        c.linear(
            "control_all_x_embedder.2-1",
            dim,
            usize::try_from(
                cfg.patch_size
                    * cfg.patch_size
                    * mlx_gen_z_image::control_transformer::CONTROL_IN_DIM,
            )
            .unwrap(),
            false,
            true,
        );
        // The production Control topology in control_transformer.rs: places0,2,...,28 and0,1.
        for (stack, count) in [("control_layers", 15), ("control_noise_refiner", 2)] {
            for i in 0..count {
                let p = format!("{stack}.{i}");
                c.block(&p, dim, heads, true, false);
                c.linear(&format!("{p}.after_proj"), dim, dim, false, true);
                if i == 0 {
                    c.linear(&format!("{p}.before_proj"), dim, dim, false, true);
                }
            }
        }
        c.save(control);
        let channels = [32usize, 32, 64, 64];
        let mut vae = PublicHeaders::new();
        vae.conv("encoder.conv_in", channels[0], 3, 3);
        let mut previous = channels[0];
        for (i, &out) in channels.iter().enumerate() {
            for r in 0..2 {
                vae.resnet(
                    &format!("encoder.down_blocks.{i}.resnets.{r}"),
                    if r == 0 { previous } else { out },
                    out,
                );
            }
            if i < 3 {
                vae.conv(
                    &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                    out,
                    out,
                    3,
                );
            }
            previous = out;
        }
        vae.mid("encoder.mid_block", previous);
        vae.norm("encoder.conv_norm_out", previous);
        vae.conv("encoder.conv_out", 32, previous, 3);
        vae.conv("decoder.conv_in", previous, 16, 3);
        vae.mid("decoder.mid_block", previous);
        for (i, &out) in channels.iter().rev().enumerate() {
            for r in 0..3 {
                vae.resnet(
                    &format!("decoder.up_blocks.{i}.resnets.{r}"),
                    if r == 0 { previous } else { out },
                    out,
                );
            }
            if i < 3 {
                vae.conv(
                    &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                    out,
                    out,
                    3,
                );
            }
            previous = out;
        }
        vae.norm("decoder.conv_norm_out", previous);
        vae.conv("decoder.conv_out", 3, previous, 3);
        vae.save(&root.join("vae/model.safetensors"));
        std::fs::write(
            root.join("vae/config.json"),
            serde_json::json!({"_class_name":"AutoencoderKL",
            "block_out_channels":channels,"layers_per_block":2,"latent_channels":16,
            "quantization":{"bits":8,"group_size":64}})
            .to_string(),
        )
        .unwrap();

        // Real source_for_load acquires an ArtifactSeal: it may sequentially SHA256 the whole
        // sparse TE once per acquisition without its identity memo. This is CPU/IO test-integrity
        // work, NOT semantic coefficient/scalar pricing and not a model load. Never bypass it.
        use std::os::unix::fs::MetadataExt;
        fn storage(
            dir: &std::path::Path,
            logical: &mut u64,
            allocated: &mut u64,
            small: &mut u64,
            count: &mut u64,
        ) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    storage(&path, logical, allocated, small, count);
                    continue;
                }
                let metadata = std::fs::metadata(&path).unwrap();
                *logical = logical.checked_add(metadata.len()).unwrap();
                *allocated = allocated
                    .checked_add(metadata.blocks().checked_mul(512).unwrap())
                    .unwrap();
                if path.extension().and_then(|x| x.to_str()) == Some("safetensors") {
                    *count += 1;
                } else {
                    *small = small.checked_add(metadata.len()).unwrap();
                }
            }
        }
        let (mut logical, mut allocated, mut small, mut count) = (0, 0, 0, 0);
        storage(root, &mut logical, &mut allocated, &mut small, &mut count);
        let te_bytes = std::fs::metadata(root.join("text_encoder/model.safetensors"))
            .unwrap()
            .len();
        // Three endpoint calls plus four mode calls. Every classified call acquires its contract
        // directly AND via build_candidate. Resident's configured form adds1; optimized explores
        // at most four forms for every actual optimized strategy. Do not assume the memo hits.
        let contract_acquisitions = 2 * (3u64 + 4)
            + 1
            + 4 * u64::try_from(
                MemoryStrategy::ALL
                    .into_iter()
                    .filter(|strategy| strategy.is_optimized())
                    .count(),
            )
            .unwrap();
        let max_te_scan = te_bytes.checked_mul(contract_acquisitions).unwrap();
        eprintln!("PUBLIC_FIXTURE_STORAGE_V1 sparse_files={count} total_logical_bytes={logical} actual_allocated_bytes={allocated} metadata_file_bytes={small} TE_file_bytes={te_bytes} unique_TE_epochs=1 contract_acquisition_max={contract_acquisitions} TE_stream_scan_bytes_if_no_memo={max_te_scan} hash_buffer_bytes=65536");

        // The aggregate raw-source/real-constructor oracle runs after truthful TE discovery
        // in the HOST below. Retain the independent complete VAE endpoint check here.
        use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
        let vae_headers = safetensors_path_tensor_headers(root.join("vae")).unwrap();
        assert!(
            vae_headers
                .iter()
                .any(|h| h.name == "decoder.up_blocks.3.resnets.2.conv2.weight"
                    && h.shape == [32, 32, 3, 3]),
            "PUBLIC_COMPLETE_VAE_DECODER_TOPOLOGY"
        );
    }

    /// Independent raw-source oracle, not the PublicHeaders writer's inventory. Constructor
    /// citations: transformer.rs::adapter_header_targets, loader.rs::TRANSFORMER_KEY_ALIASES,
    /// timestep_embedder.rs, final_layer.rs, transformer_block.rs/context_block.rs and
    /// control_transformer.rs::{CONTROL_LAYERS_PLACES,CONTROL_REFINER_PLACES}. Those metadata
    /// helpers are crate-private; no production API is widened to make this fixture pass.
    fn public_raw_schema(
        control: bool,
    ) -> std::collections::BTreeMap<String, (&'static str, Vec<usize>)> {
        type Schema = std::collections::BTreeMap<String, (&'static str, Vec<usize>)>;
        fn put(out: &mut Schema, name: String, dtype: &'static str, shape: Vec<usize>) {
            assert!(
                out.insert(name, (dtype, shape)).is_none(),
                "PUBLIC_ORACLE_UNIQUE_RAW_KEY"
            );
        }
        fn linear(
            out: &mut Schema,
            path: &str,
            output: usize,
            input: usize,
            packed: bool,
            bias: bool,
        ) {
            if packed {
                assert_eq!(input % 64, 0, "PUBLIC_ORACLE_GROUP64");
                for (suffix, dtype, columns) in [
                    ("weight", "U32", input / 4),
                    ("scales", "BF16", input / 64),
                    ("biases", "BF16", input / 64),
                ] {
                    put(
                        out,
                        format!("{path}.{suffix}"),
                        dtype,
                        vec![output, columns],
                    );
                }
            } else {
                put(out, format!("{path}.weight"), "BF16", vec![output, input]);
            }
            if bias {
                put(out, format!("{path}.bias"), "BF16", vec![output]);
            }
        }
        fn block(
            out: &mut Schema,
            path: &str,
            dim: usize,
            heads: usize,
            context: bool,
            packed: bool,
        ) {
            // Each shape is the constructor's [out,in], not a count of whatever the writer emits.
            for (leaf, output, input, bias) in [
                ("attention.to_q", dim, dim, false),
                ("attention.to_k", dim, dim, false),
                ("attention.to_v", dim, dim, false),
                ("attention.to_out.0", dim, dim, false),
                ("feed_forward.w1", 2 * dim, dim, false),
                ("feed_forward.w2", dim, 2 * dim, false),
                ("feed_forward.w3", 2 * dim, dim, false),
            ] {
                linear(out, &format!("{path}.{leaf}"), output, input, packed, bias);
            }
            if !context {
                linear(
                    out,
                    &format!("{path}.adaLN_modulation.0"),
                    4 * dim,
                    dim,
                    packed,
                    true,
                );
            }
            for leaf in ["attention.norm_q", "attention.norm_k"] {
                put(
                    out,
                    format!("{path}.{leaf}.weight"),
                    "BF16",
                    vec![dim / heads],
                );
            }
            for leaf in [
                "attention_norm1",
                "attention_norm2",
                "ffn_norm1",
                "ffn_norm2",
            ] {
                put(out, format!("{path}.{leaf}.weight"), "BF16", vec![dim]);
            }
        }
        let cfg = mlx_gen_z_image::ZImageTransformerConfig::turbo();
        let dim = usize::try_from(cfg.dim).unwrap();
        let heads = usize::try_from(cfg.n_heads).unwrap();
        let patch = usize::try_from(cfg.patch_size * cfg.patch_size * cfg.in_channels).unwrap();
        let mut out = Schema::new();
        if control {
            linear(
                &mut out,
                "control_all_x_embedder.2-1",
                dim,
                usize::try_from(cfg.patch_size * cfg.patch_size * mlx_gen_z_image::CONTROL_IN_DIM)
                    .unwrap(),
                false,
                true,
            );
            // Private placement constants are [0,2,..,28] and [0,1]; use their explicit source
            // positions, not a claim that a stack count alone establishes constructor placement.
            for (stack, places) in [
                (
                    "control_layers",
                    &[0usize, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28][..],
                ),
                ("control_noise_refiner", &[0usize, 1][..]),
            ] {
                for (index, _) in places.iter().enumerate() {
                    let path = format!("{stack}.{index}");
                    block(&mut out, &path, dim, heads, false, false);
                    linear(
                        &mut out,
                        &format!("{path}.after_proj"),
                        dim,
                        dim,
                        false,
                        true,
                    );
                    if index == 0 {
                        linear(
                            &mut out,
                            &format!("{path}.before_proj"),
                            dim,
                            dim,
                            false,
                            true,
                        );
                    }
                }
            }
        } else {
            for (path, output, input) in [
                ("all_x_embedder.2-1", dim, patch),
                (
                    "t_embedder.mlp.0",
                    1024,
                    usize::try_from(cfg.frequency_embedding_size).unwrap(),
                ),
                ("t_embedder.mlp.2", dim, 1024),
                (
                    "cap_embedder.1",
                    dim,
                    usize::try_from(cfg.cap_feat_dim).unwrap(),
                ),
                ("all_final_layer.2-1.linear", patch, dim),
                ("all_final_layer.2-1.adaLN_modulation.1", dim, dim),
            ] {
                linear(&mut out, path, output, input, true, true);
            }
            put(
                &mut out,
                "cap_embedder.0.weight".into(),
                "BF16",
                vec![usize::try_from(cfg.cap_feat_dim).unwrap()],
            );
            for name in ["x_pad_token", "cap_pad_token"] {
                put(&mut out, name.into(), "BF16", vec![1, dim]);
            }
            for (stack, count, context) in [
                ("layers", cfg.n_layers, false),
                ("noise_refiner", cfg.n_refiner_layers, false),
                ("context_refiner", cfg.n_refiner_layers, true),
            ] {
                for index in 0..count {
                    block(
                        &mut out,
                        &format!("{stack}.{index}"),
                        dim,
                        heads,
                        context,
                        true,
                    );
                }
            }
        }
        out
    }

    fn public_raw_header_failures(
        component: &str,
        actual: &std::collections::BTreeMap<
            String,
            mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader,
        >,
        expected: &std::collections::BTreeMap<String, (&'static str, Vec<usize>)>,
    ) -> Vec<String> {
        use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
        let mut failures = Vec::new();
        for (name, (dtype, shape)) in expected {
            let category = if name.ends_with(".scales") || name.ends_with(".biases") {
                "companion/group64"
            } else if name.ends_with(".bias") {
                "bias"
            } else {
                "tensor"
            };
            match actual.get(name) {
                None => failures.push(format!("{component}: missing {category} {name}")),
                Some(header) => {
                    let dtype_matches = matches!(
                        (*dtype, header.dtype),
                        ("U32", Stored::U32) | ("BF16", Stored::BF16)
                    );
                    if !dtype_matches {
                        failures.push(format!(
                            "{component}: dtype {name}: {:?}, expected {dtype}",
                            header.dtype
                        ));
                    }
                    if header.shape != *shape {
                        failures.push(format!(
                            "{component}: {category} shape {name}: {:?}, expected {shape:?}",
                            header.shape
                        ));
                    }
                }
            }
        }
        for name in actual.keys().filter(|name| !expected.contains_key(*name)) {
            failures.push(format!("{component}: unexpected raw key {name}"));
        }
        // These are constructor-facing aliases, not extra authored source tensors. Check ALL
        // suffixes and reject duplicates too: adding correct raw keys beside bad internal ones
        // must not make a supposedly complete fixture pass.
        if component == "base" {
            for alias in [
                "t_embedder.linear1",
                "t_embedder.linear2",
                "all_final_layer.2-1.adaLN_modulation.0",
            ] {
                for suffix in ["weight", "scales", "biases", "bias"] {
                    let name = format!("{alias}.{suffix}");
                    if actual.contains_key(&name) {
                        failures.push(format!("base: internal raw-alias forbidden {name}"));
                    }
                }
            }
        }
        failures
    }

    fn validate_public_raw_and_constructor_schemas(
        root: &std::path::Path,
        control: &std::path::Path,
    ) {
        use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
        let read = |source: &std::path::Path| {
            safetensors_path_tensor_headers(source)
                .unwrap()
                .into_iter()
                .map(|h| (h.name.clone(), h))
                .collect::<std::collections::BTreeMap<_, _>>()
        };
        let raw_base = read(&root.join("transformer"));
        let raw_control = read(control);
        let base_schema = public_raw_schema(false);
        let control_schema = public_raw_schema(true);
        let mut failures = public_raw_header_failures("base", &raw_base, &base_schema);
        failures.extend(public_raw_header_failures(
            "control",
            &raw_control,
            &control_schema,
        ));
        // The exported graph entry is the existing native-free route to the REAL crate-private
        // remap: ios_pricing.rs::configured_overlay_semantic_facts_for constructs a separate
        // owned header map from the same serialized source, calls remap_transformer_header_keys,
        // then adapter_header_targets/linear_header_facts and the actual Control target resolver.
        // The immutable raw_base/raw_control maps above remain the authored-source inventory;
        // aliases in the constructor view are never serialized or charged as extra raw roots.
        for provider in [
            crate::zimage::Z_IMAGE_TURBO_ID,
            crate::zimage::Z_IMAGE_TURBO_CONTROL_ID,
        ] {
            let mut spec = crate::zimage::z_image_turbo_load_spec(
                WeightsSource::Dir(root.to_owned()),
                (provider == crate::zimage::Z_IMAGE_TURBO_CONTROL_ID)
                    .then(|| WeightsSource::File(control.to_owned())),
            );
            spec.quantize = Some(Quant::Q8);
            match mlx_gen_z_image::ios_pricing::configured_overlay_semantic_facts_for(
                provider, &spec,
            ) {
                Err(error) => failures.push(format!("constructor-view {provider}: {error}")),
                Ok(graph) if provider == crate::zimage::Z_IMAGE_TURBO_CONTROL_ID => {
                    if graph["control"]["main_places"]
                        != serde_json::json!([
                            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28
                        ])
                    {
                        failures.push(
                            "constructor-view control main placement differs from raw schema"
                                .into(),
                        );
                    }
                    if graph["control"]["refiner_places"] != serde_json::json!([0, 1]) {
                        failures.push(
                            "constructor-view control refiner placement differs from raw schema"
                                .into(),
                        );
                    }
                }
                Ok(_) => {}
            }
        }
        assert!(
            failures.is_empty(),
            "PUBLIC_AGGREGATE_RAW_AND_ACTUAL_CONSTRUCTOR_SCHEMA:\n{}",
            failures.join("\n")
        );
        // Check diagnostic aggregation independently on a cloned header map. This never writes
        // a source file, changes a source-policy selector, seals payload, or enters native code.
        let mut malformed = raw_base.clone();
        malformed.remove("t_embedder.mlp.0.scales");
        malformed.remove("t_embedder.mlp.2.bias");
        malformed.get_mut("t_embedder.mlp.0.weight").unwrap().dtype =
            mlx_gen::gen_core::weightsmeta::Dtype::BF16;
        malformed.get_mut("t_embedder.mlp.0.weight").unwrap().shape[0] += 1;
        malformed.get_mut("t_embedder.mlp.0.biases").unwrap().shape[1] += 1;
        let mut alias = raw_base["t_embedder.mlp.0.weight"].clone();
        alias.name = "t_embedder.linear1.weight".into();
        malformed.insert(alias.name.clone(), alias);
        let diagnostics = public_raw_header_failures("base", &malformed, &base_schema);
        for fragment in [
            "missing companion/group64 t_embedder.mlp.0.scales",
            "missing bias t_embedder.mlp.2.bias",
            "dtype t_embedder.mlp.0.weight",
            "tensor shape t_embedder.mlp.0.weight",
            "companion/group64 shape t_embedder.mlp.0.biases",
            "unexpected raw key t_embedder.linear1.weight",
            "internal raw-alias forbidden t_embedder.linear1.weight",
        ] {
            assert!(
                diagnostics.iter().any(|line| line.contains(fragment)),
                "PUBLIC_AGGREGATE_DIAGNOSTIC_MISSING {fragment}"
            );
        }
        let mut malformed_control = raw_control.clone();
        malformed_control.remove("control_layers.14.after_proj.bias");
        let mut misplaced = raw_control["control_layers.0.before_proj.weight"].clone();
        misplaced.name = "control_layers.1.before_proj.weight".into();
        malformed_control.insert(misplaced.name.clone(), misplaced);
        let diagnostics =
            public_raw_header_failures("control", &malformed_control, &control_schema);
        for fragment in [
            "missing bias control_layers.14.after_proj.bias",
            "unexpected raw key control_layers.1.before_proj.weight",
        ] {
            assert!(
                diagnostics.iter().any(|line| line.contains(fragment)),
                "PUBLIC_CONTROL_AGGREGATE_DIAGNOSTIC_MISSING {fragment}"
            );
        }
        eprintln!("PUBLIC_AGGREGATE_RAW_SCHEMA_V1 base_headers={} control_headers={} base_packed_linears={} production_constructor_views=2",
            raw_base.len(),raw_control.len(),base_schema.iter().filter(|(name,(dtype,_))| name.ends_with(".weight") && *dtype == "U32").count());
    }

    // Two deliberately different fixtures: only public_key describes the admitted-size HOST
    // transport request. The other fields bind the historical PRIVATE 32x32 program, which is
    // not claimed to be publicly admissible or a native/catalog execution-path witness.
    struct FixturePurposes {
        public_key: EvidenceKey,
        private_core_binding: [u8; 32],
        private_binding: [u8; 32],
        private_program: serde_json::Value,
    }

    fn classified(
        registry: &ProviderRegistry,
        root: &std::path::Path,
        control: &std::path::Path,
        mode: ManualMode,
        endpoint: Option<AnchorKind>,
    ) -> FixturePurposes {
        let has_control = matches!(mode, ManualMode::Control | ManualMode::ControlReference);
        let has_reference = matches!(mode, ManualMode::Reference | ManualMode::ControlReference);
        let provider = if has_control {
            crate::zimage::Z_IMAGE_TURBO_CONTROL_ID
        } else {
            crate::zimage::Z_IMAGE_TURBO_ID
        };
        let mut base = crate::zimage::z_image_turbo_load_spec(
            WeightsSource::Dir(root.to_owned()),
            has_control.then(|| WeightsSource::File(control.to_owned())),
        );
        base.quantize = Some(Quant::Q8);
        let (mut spec, strategy, position) = match endpoint {
            Some(kind) => {
                let (spec, strategy) =
                    configured_endpoint(registry, provider, &base, kind).unwrap();
                (
                    spec,
                    strategy,
                    if kind == AnchorKind::Resident {
                        Position::ResidentEndpoint
                    } else {
                        Position::OptimizedEndpoint
                    },
                )
            }
            None => (base, MemoryStrategy::Resident, Position::Intermediate),
        };
        let contract = registry
            .memory_strategy_contract(provider, &spec)
            .unwrap()
            .unwrap();
        let selection = *contract
            .selection_space(strategy, numeric_tier(&spec), false)
            .unwrap()
            .last()
            .unwrap();
        let execution = execution_for(provider, &contract, &selection);
        // Match the legacy CALLER fact: the original DTO omits control, and the actual manual
        // argument supplies it. project() describes that argument without altering execution.
        spec.control = None;
        let projection = shared::project(Inputs {
            provider,
            mode,
            position,
            original_spec: &spec,
            actual_text_source: &root.join("text_encoder"),
            actual_control_source: control,
            width: 32,
            height: 32,
            count: 1,
            execution,
            hint: image(),
            init: has_reference.then(image),
            control_kind: ControlKind::Pose,
            // These are explicitly HEADER-TRANSPORT fixtures, not native Snapshot configuration
            // or a claim that sparse headers can execute. Z's separate metadata-only unit binds
            // the actual private Snapshot fields and SAME native-free window selectors.
            configurations: std::array::from_fn(|_| {
                "catalog sparse-header metadata transport only".into()
            }),
            original_factory_source: include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../mlx-gen-z-image/src/ios_pricing_tests.rs"
            )),
            actual_vocab_size: i32::try_from(mlx_gen_z_image::ENCODER_CONTRACT.vocab_size).unwrap(),
            te_explicit_window: execution
                .transformer_window_size
                .map(|window| usize::try_from(window).expect("u32 window fits usize on macOS")),
            te_full_cover_fallback: None,
            dit_selector_window: execution
                .transformer_window_size
                .map(|window| usize::try_from(window).expect("u32 window fits usize on macOS")),
            actual_te_layers: mlx_gen_z_image::ENCODER_CONTRACT.num_hidden_layers,
            actual_base_blocks: mlx_gen_z_image::ZImageTransformerConfig::turbo().n_layers,
            actual_control_blocks: None,
        })
        .unwrap();
        assert_eq!(projection.provider, provider);
        assert_eq!(
            (projection.request.width, projection.request.height),
            (32, 32),
            "PRIVATE_ORIGINAL_GEOMETRY_NOT_RESCALED"
        );
        assert_eq!(
            projection.program["geometry"],
            serde_json::json!({"width":32,"height":32,"count":1}),
            "PRIVATE_PROGRAM_GEOMETRY_NOT_PUBLIC_FIXTURE"
        );

        // A separate valid PUBLIC metadata transport request, at this actual descriptor's minimum.
        // It shares the declared roles/sampling, but is NOT the historical manual execution. Never
        // attach the private program/digest to it or bypass validate_actual/evidence_key.
        let descriptor = registry
            .generators()
            .map(|entry| (entry.descriptor)())
            .find(|descriptor| descriptor.id == provider)
            .unwrap();
        let public_edge = descriptor.capabilities.min_size;
        let public_request = mlx_gen::gen_core::GenerationRequest {
            width: public_edge,
            height: public_edge,
            ..projection.request.clone()
        };
        let private_work = IosWork::Generate(projection.request);
        let core_binding = work_binding(&private_work);
        let sidecar_binding = private_execution_binding(core_binding, &projection.program);
        let public_work = IosWork::Generate(public_request);
        let public_binding = work_binding(&public_work);
        assert_ne!(
            public_binding, core_binding,
            "PUBLIC_AND_PRIVATE_WORK_ARE_DISTINCT"
        );
        let wrong_public_default = IosWork::Generate(mlx_gen::gen_core::GenerationRequest {
            width: public_edge,
            height: public_edge,
            count: 1,
            memory: Some(mlx_gen::gen_core::GenerationMemory::default()),
            ..Default::default()
        });
        assert_ne!(
            public_binding,
            work_binding(&wrong_public_default),
            "VALID_PUBLIC_ROLES_SAMPLING_AND_EXECUTION_ARE_BOUND"
        );
        let candidate = build_candidate(
            registry,
            provider,
            projection.described_spec,
            Some(selection),
            strategy,
            public_work,
        )
        .unwrap();
        assert_eq!(candidate.execution, execution);
        assert_eq!(
            work_binding(&candidate.work),
            public_binding,
            "VALID_PUBLIC_TYPED_WORK_BINDING"
        );
        // Resolve EVERY actual production target/schema before the positive EvidenceKey. This
        // is metadata validation only; it loads no model and is not catalog execution coverage.
        let _configured_graph =
            mlx_gen_z_image::ios_pricing::configured_overlay_semantic_facts_for(
                provider,
                &candidate.spec,
            )
            .expect("PUBLIC_COMPLETE_ACTUAL_CONFIGURED_PROJECTION_SCHEMA");
        let key = evidence_key(
            registry,
            &candidate,
            "valid-public-metadata-transport-NOT-original-execution",
        )
        .unwrap();
        assert_eq!(
            (key.width, key.height),
            (public_edge, public_edge),
            "VALID_PUBLIC_DESCRIPTOR_MINIMUM_GEOMETRY"
        );
        // Exercise the actual classifier separately too, not a test-local copy of its policy.
        let class =
            crate::sampling::from_token(provider, key.sampling_recipe.as_deref().unwrap()).unwrap();
        let variant = crate::pricing::anchor_variant_for_roles(
            provider,
            &key.mode,
            &key.tier,
            Some(class.allocation_class()),
            key.reference_count,
            key.mask_count,
            key.control_count,
        )
        .unwrap();
        assert!(key.family().unwrap().variant.starts_with(variant.as_str()));
        FixturePurposes {
            public_key: key,
            private_core_binding: core_binding,
            private_binding: sidecar_binding,
            private_program: projection.program,
        }
    }

    #[test]
    fn original_sidecar_uses_real_family_endpoints_and_exact_work_binding() {
        // Sparse header/config/tokenizer fixtures only. Real endpoint/contract/candidate source
        // acquisition MAY STREAM-HASH the TE sparse payload for identity (64KiB buffer); layout,
        // configured graph and EvidenceKey inspect headers/configs. No semantic tensor/scalar value
        // pricing, load/admit/pricer, native Snapshot/tiers/Model/view, Array/device/counter runs.
        let dir = tempfile::tempdir().unwrap();
        let control = dir.path().join("actual-control.safetensors");
        complete_public_metadata_fixture(dir.path(), &control);
        // Bounded discovery/actual configured-graph checks FIRST, without content sealing. They
        // validate this fixture, but NEVER substitute for the later real public contract/candidate.
        // Authorization needs BOTH the lexical loader root and its canonical target (macOS
        // /var may resolve through /private/var). The existing bounded root builder declares
        // exactly this OWN fixture location; it does not rewrite any literal configured path,
        // private original-program binding, or the later real source seal.
        let discovery_roots = mlx_gen::gen_core::hf_cache_discovery_roots(dir.path()).unwrap();
        let _discovery = mlx_gen_z_image::ENCODER_CONTRACT
            .validate_source_for_discovery(
                &WeightsSource::Dir(dir.path().join("text_encoder")),
                &discovery_roots,
            )
            .expect("PUBLIC_COMPLETE_TE_HEADER_CONFIG_DISCOVERY");
        validate_public_raw_and_constructor_schemas(dir.path(), &control);
        let te_epoch = public_te_epoch(dir.path());
        let registry = crate::provider_registry().unwrap();
        let FixturePurposes {
            public_key: resident,
            private_core_binding: resident_binding,
            private_program: resident_program,
            ..
        } = classified(
            &registry,
            dir.path(),
            &control,
            ManualMode::Control,
            Some(AnchorKind::Resident),
        );
        let FixturePurposes {
            public_key: optimized,
            private_core_binding: optimized_binding,
            private_program: optimized_program,
            ..
        } = classified(
            &registry,
            dir.path(),
            &control,
            ManualMode::Control,
            Some(AnchorKind::Optimized),
        );
        let FixturePurposes {
            public_key: intermediate,
            private_program: intermediate_program,
            ..
        } = classified(&registry, dir.path(), &control, ManualMode::Control, None);
        assert_eq!(resident.family().unwrap(), optimized.family().unwrap());
        assert_eq!(optimized.family().unwrap(), intermediate.family().unwrap());
        assert_ne!(resident.load_form, intermediate.load_form);
        assert_ne!(optimized.execution, intermediate.execution);
        assert_ne!(resident_binding, optimized_binding);
        assert_eq!(resident_program["position"], "resident_endpoint");
        assert_eq!(optimized_program["position"], "optimized_endpoint");
        assert_eq!(
            intermediate_program["position"],
            "intermediate_deferred_full_cover_unstaged"
        );
        let mut families = Vec::new();
        for mode in [
            ManualMode::Text,
            ManualMode::Reference,
            ManualMode::Control,
            ManualMode::ControlReference,
        ] {
            let FixturePurposes {
                public_key: key,
                private_core_binding: binding,
                private_binding: sidecar_binding,
                private_program: program,
            } = classified(&registry, dir.path(), &control, mode, None);
            families.push(key.family().unwrap());
            let wrong_default = IosWork::Generate(mlx_gen::gen_core::GenerationRequest {
                width: 32,
                height: 32,
                count: 1,
                memory: Some(mlx_gen::gen_core::GenerationMemory::default()),
                ..Default::default()
            });
            assert_ne!(
                binding,
                work_binding(&wrong_default),
                "default DTO is not actual manual execution"
            );
            // Manual start is NOT silently manufactured as an authored strength or lost from the
            // private full binding. It remains additional program context, not catalog approval.
            let mut changed = program.clone();
            changed["manual_flow"]["start"] = serde_json::json!(99);
            assert_ne!(
                sidecar_binding,
                private_execution_binding(binding, &changed)
            );
            // HOST source-description variants only: no original constant/factory, Array or
            // runtime request is changed. A native capture or source-eligible alternate execution
            // is NOT claimed by these private digest-discrimination assertions.
            let source = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../mlx-gen-z-image/src/ios_pricing_tests.rs"
            ));
            let window = i32::try_from(program["tokens"]["window"].as_i64().unwrap()).unwrap();
            let valid = i32::try_from(program["tokens"]["valid_tokens"].as_i64().unwrap()).unwrap();
            let vocab =
                i32::try_from(program["tokens"]["vocabulary"]["size"].as_i64().unwrap()).unwrap();
            for (name, old, new) in [("VALID", valid, valid - 1), ("WINDOW", window, window + 1)] {
                let old = format!("const {name}: i32 = {old};");
                let new = format!("const {name}: i32 = {new};");
                assert_eq!(source.matches(old.as_str()).count(), 1);
                let hypothetical_source = source.replacen(old.as_str(), new.as_str(), 1);
                let mut changed = program.clone();
                changed["tokens"] =
                    shared::original_token_program(&hypothetical_source, vocab).unwrap();
                assert_ne!(
                    sidecar_binding,
                    private_execution_binding(binding, &changed),
                    "valid-token/window description must be privately bound"
                );
            }
            let mut changed = program.clone();
            let mut mask: Vec<i32> =
                serde_json::from_value(changed["tokens"]["attention_mask"]["values"].clone())
                    .unwrap();
            mask.swap(0, usize::try_from(valid).unwrap()); // same valid count, different mask positions
            changed["tokens"]["attention_mask"]["values_sha256"] =
                serde_json::json!(shared::source_i32_values_hash(&mask));
            changed["tokens"]["attention_mask"]["values"] = serde_json::json!(mask);
            assert_ne!(
                sidecar_binding,
                private_execution_binding(binding, &changed),
                "attention mask values/positions must be privately bound"
            );
            let mut changed = program.clone();
            changed["tokens"] =
                shared::original_token_program(source, vocab.checked_add(1).unwrap()).unwrap();
            assert_ne!(
                sidecar_binding,
                private_execution_binding(binding, &changed),
                "actual embedding vocabulary dependency must be privately bound"
            );
        }
        for (i, family) in families.iter().enumerate() {
            assert!(
                families[..i].iter().all(|other| other != family),
                "actual role families must not borrow text evidence"
            );
        }
        assert_eq!(
            public_te_epoch(dir.path()),
            te_epoch,
            "PUBLIC_ONE_IMMUTABLE_TE_EPOCH_ACROSS_CASES"
        );
    }
}

#[cfg(test)]
mod pre_key_cfg_tests {
    use super::*;

    #[test]
    fn key_count_uses_resolved_cfg_and_actual_negative_including_empty() {
        let mut request = GenerationRequest {
            prompt: "positive".into(),
            negative_prompt: Some("the actual negative".into()),
            guidance: Some(1.0),
            ..Default::default()
        };
        assert_eq!(key_negative_prompt(&request).unwrap(), None);
        request.guidance = Some(7.5);
        assert_eq!(
            key_negative_prompt(&request).unwrap(),
            Some("the actual negative")
        );
        request.negative_prompt = None;
        assert_eq!(key_negative_prompt(&request).unwrap(), Some(""));
        request.negative_prompt = Some(String::new());
        assert_eq!(key_negative_prompt(&request).unwrap(), Some(""));
        request.guidance = Some(0.0);
        assert_eq!(key_negative_prompt(&request).unwrap(), None);
    }
}

#[cfg(test)]
mod source_capacity_transport_tests {
    use super::*;

    #[test]
    fn swallowed_refusal_cannot_become_native_success_or_no_evidence() {
        // Finite test check, NOT a production byte cap or certificate.
        let mut seen = Vec::new();
        let mut capacity = |bytes| {
            seen.push(bytes);
            if bytes == 31 {
                Err(mlx_gen::Error::Msg("fixture allocation refused".into()))
            } else {
                Ok(())
            }
        };
        let result = with_source_metadata_capacity(&mut capacity, |next| {
            next(17)?;
            // A provider can preserve native bounds with an incomplete-host gap.
            // The real boundary must not swallow the failed allocation check.
            let _refused_analysis = next(31);
            Ok(7u64)
        });
        match result {
            Err(AdmissionError::Refused(refusal)) => {
                assert_eq!(refusal.kind, RefusalKind::Capacity)
            }
            _ => panic!("failed real check must remain Capacity"),
        }
        assert_eq!(seen, vec![17, 31]);
    }

    #[test]
    fn same_borrowed_check_sees_each_serial_pricing_request() {
        let mut seen = Vec::new();
        let mut capacity = |bytes| {
            seen.push(bytes);
            if bytes > 100 {
                Err(mlx_gen::Error::Msg("fixture allocation refused".into()))
            } else {
                Ok(())
            }
        };
        let first = with_source_metadata_capacity(&mut capacity, |next| {
            next(13)?;
            Ok(1u64)
        })
        .unwrap();
        let second = with_source_metadata_capacity(&mut capacity, |next| {
            next(17)?;
            Ok(2u64)
        })
        .unwrap();
        assert_eq!((first, second), (1, 2));
        assert_eq!(seen, vec![13, 17]);
    }
}

#[cfg(test)]
mod source_gap_classification_tests {
    use super::*;

    #[test]
    fn only_exact_five_source_law_gaps_are_no_evidence() {
        for reason in [
            SourceInventory::READDIR_METADATA_GAP,
            SourceInventory::PIN_VERIFY_METADATA_GAP,
            SourceInventory::PIN_CAPTURE_METADATA_GAP,
            SourceInventory::ABSOLUTE_PATH_METADATA_GAP,
            SourceInventory::MISSING_INVENTORY_METADATA_GAP,
        ] {
            let mapped = source_metadata_gap_refusal(AdmissionError::Failed(Error::Unsupported(
                reason.into(),
            )));
            assert!(
                matches!(mapped, AdmissionError::Refused(refusal) if refusal.kind == RefusalKind::NoEvidence && refusal.reason == reason)
            );
        }
        for reason in [
            "malformed model configuration",
            "READDIR_METADATA_GAP",
            "unknown metadata",
        ] {
            let mapped = source_metadata_gap_refusal(AdmissionError::Failed(Error::Unsupported(
                reason.into(),
            )));
            assert!(
                matches!(mapped, AdmissionError::Failed(Error::Unsupported(actual)) if actual == reason)
            );
        }
        let mapped =
            source_metadata_gap_refusal(AdmissionError::Failed(Error::Msg("read failed".into())));
        assert!(
            matches!(mapped, AdmissionError::Failed(Error::Msg(reason)) if reason == "read failed")
        );
        let mapped = source_metadata_gap_refusal(AdmissionError::refused(
            RefusalKind::Capacity,
            "actual callback refused",
        ));
        assert!(
            matches!(mapped, AdmissionError::Refused(refusal) if refusal.kind == RefusalKind::Capacity)
        );
    }
}
