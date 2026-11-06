//! Provider safety gates, selected-parameter validation, and safety-decision admission.

use super::*;

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

pub(super) fn validate_ranges(capability: &MemoryStrategyCapability, errors: &mut Vec<String>) {
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

pub(super) fn validate_owned_parameter_domain(
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

pub(super) fn validate_selected_parameters(
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

/// Defense-in-depth result. It can accept or reject; it cannot replace the worker's selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemorySafetyDecision {
    Accept,
    Reject { reason: String },
}

impl MemorySafetyDecision {
    /// A rejection whose reason is `error`'s rendering — the decision a provider's fallible
    /// validation step produces when it fails (sc-27322).
    pub fn rejected(error: impl std::fmt::Display) -> Self {
        Self::Reject {
            reason: error.to_string(),
        }
    }

    /// The decision a provider's whole fallible validation produces: accept on `Ok`, reject with
    /// the error's rendering otherwise. The inverse of [`admit_memory_safety`] (sc-27322).
    pub fn from_validation<E: std::fmt::Display>(result: std::result::Result<(), E>) -> Self {
        match result {
            Ok(()) => Self::Accept,
            Err(error) => Self::rejected(error),
        }
    }
}

/// Validate a request's selected bounded-decode tile against a provider's published domain: both
/// `edge` and `overlap` must be selected, `edge` must be one of `edges`, and `overlap` must equal
/// `expected_overlap` (sc-27295: the shared body of the per-provider `validate_decode` copies). Each
/// failure is an [`Error::Unsupported`] prefixed with `model_id`.
pub fn validate_bounded_decode_params(
    model_id: &str,
    edges: &[u32],
    expected_overlap: u32,
    edge: Option<u32>,
    overlap: Option<u32>,
) -> Result<()> {
    let edge = edge.ok_or_else(|| {
        Error::Unsupported(format!("{model_id}: bounded decode requires a tile edge"))
    })?;
    let overlap = overlap.ok_or_else(|| {
        Error::Unsupported(format!(
            "{model_id}: bounded decode requires a tile overlap"
        ))
    })?;
    if !edges.contains(&edge) || overlap != expected_overlap {
        return Err(Error::Unsupported(format!(
            "{model_id}: decode tile {edge}/{overlap} is outside the published domain \
             {edges:?} at overlap {expected_overlap}"
        )));
    }
    Ok(())
}

/// The one place a provider's refused safety decision becomes a typed error.
///
/// [`MemorySafetyDecision::Reject`] carries an already-rendered reason, so it is wrapped in
/// [`Error::Unsupported`] exactly once, here, for every backend's request-scope admission
/// (sc-27257, sc-27341).
pub fn admit_memory_safety(decision: MemorySafetyDecision) -> Result<()> {
    match decision {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(Error::Unsupported(reason)),
    }
}
