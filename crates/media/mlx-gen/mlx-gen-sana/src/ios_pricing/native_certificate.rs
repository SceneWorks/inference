//! Source/native certificate assembly. No physical-transfer or retirement premise.
use super::native_alloc::NativeBackend;
use super::native_graph::{cumulative_upper, NativeSampling};
use super::*;
use mlx_gen::ios_pricing::{
    IosAnchorObservation, IosBoundRequest, IosCertifiedBounds, IosExecutionFacts,
    IosMemoryCoordinate, IosMemoryDomain, IosObservedCheck, IosWindowBound,
};

fn invalid(reason: &str) -> Error {
    Error::Unsupported(format!("SANA native certificate: {reason}"))
}
fn same_profile(expected: &IosPricing, supplied: &IosPricing) -> bool {
    expected.phases == supplied.phases
        && expected.windows == supplied.windows
        && expected.absent_phases == supplied.absent_phases
}
fn context<'a>(
    capture: &'a IosAnchorObservation,
    check: &IosObservedCheck,
    provider: &str,
    kind: &str,
) -> Result<&'a BTreeMap<String, u64>> {
    let mut found = None;
    for &index in &check.context {
        let mark = capture
            .metadata
            .get(index as usize)
            .ok_or_else(|| invalid("observation context is out of range"))?;
        if mark.provider == provider && mark.kind == kind && found.replace(&mark.fields).is_some() {
            return Err(invalid("duplicate actual source context"));
        }
    }
    found.ok_or_else(|| invalid(&format!("missing actual {provider}/{kind} source mark")))
}
fn fields_match(fields: &BTreeMap<String, u64>, expected: &[(&str, u64)]) -> bool {
    expected
        .iter()
        .all(|&(key, value)| fields.get(key) == Some(&value))
}
/// Check actual source roots at their actual checks. No replay/planned metadata
/// is generated here. This binds the page law and closed graph to captured
/// Gemma/trunk/DC-AE execution; stale/changed contexts refuse rather than giving
/// the raw logical inventory native proof authority.
fn observed_page(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    pricing: &IosPricing,
) -> Result<u64> {
    let mut page = None;
    let mut seen = [false; 4];
    let embedding = facts
        .text_encoder
        .0
        .get("model.embed_tokens.scales")
        .or_else(|| facts.text_encoder.0.get("model.embed_tokens.weight"))
        .ok_or_else(|| invalid("missing actual Gemma embedding dtype"))?;
    let g = &facts.gemma;
    let t = &facts.transformer;
    let d = &facts.dc_ae;
    let patch = u32::try_from(t.patch_size)
        .ok()
        .filter(|p| *p > 0)
        .ok_or_else(|| invalid("invalid source patch dimension"))?;
    let tokens = u64::from(plan.height / 32 / patch)
        .checked_mul(u64::from(plan.width / 32 / patch))
        .and_then(|n| i32::try_from(n).ok())
        .ok_or_else(|| invalid("source token extent exceeds native axis"))?;
    let caption_tokens = i32::try_from(facts.caption_tokens)
        .map_err(|_| invalid("source caption extent exceeds native axis"))?;
    let latent_elements = u64::from(plan.height / 32)
        .checked_mul(u64::from(plan.width / 32))
        .and_then(|n| n.checked_mul(u64::try_from(d.latent_channels).ok()?))
        .ok_or_else(|| invalid("source unscaled latent extent overflow"))?;
    for capture in &pricing.observations {
        if capture.domain != IosMemoryDomain::NativeAllocatorGrowth
            || capture.coordinate != IosMemoryCoordinate::BeforeModelLoad
        {
            return Err(invalid("incompatible observation domain/baseline"));
        }
        for check in &capture.checks {
            let expected = match check.label.as_str() {
                label if label == GEMMA_LABELS[0] => Some((
                    0,
                    "gemma2",
                    "native-forward",
                    vec![
                        ("schema", 1),
                        ("batch", 1),
                        ("tokens", facts.gemma_tokens),
                        ("hidden", g.hidden_size as u64),
                        ("layers", g.num_layers as u64),
                        ("heads", g.num_heads as u64),
                        ("kv_heads", g.num_kv_heads as u64),
                        ("head_dim", g.head_dim as u64),
                        ("intermediate", g.intermediate_size as u64),
                        (
                            "dtype",
                            mlx_gen::sampler::sampling_dtype_code(embedding.dtype),
                        ),
                        ("pad_mask", 1),
                    ],
                )),
                labels::ENTRY => Some((
                    1,
                    "sana",
                    "native-trunk",
                    vec![
                        ("schema", 1),
                        ("batch", 1),
                        ("height", u64::from(plan.height / 32)),
                        ("width", u64::from(plan.width / 32)),
                        ("channels", t.in_channels as u64),
                        ("caption_tokens", facts.caption_tokens),
                        ("caption_width", g.hidden_size as u64),
                        ("layers", t.num_layers as u64),
                        ("dim", t.inner_dim() as u64),
                        ("patch", t.patch_size as u64),
                        ("qk_norm", u64::from(t.qk_norm)),
                        ("guidance_embeds", u64::from(t.guidance_embeds)),
                        (
                            "dtype",
                            mlx_gen::sampler::sampling_dtype_code(Dtype::Float32),
                        ),
                        ("caption_mask", 1),
                        ("streamed", 0),
                        (
                            "query_rows",
                            plan.attention.query_block(
                                1,
                                t.num_cross_attention_heads,
                                tokens,
                                caption_tokens,
                            ) as u64,
                        ),
                    ],
                )),
                labels::DECODER_INPUT => Some((
                    2,
                    "sana",
                    "native-decoder",
                    vec![
                        ("schema", 1),
                        ("batch", 1),
                        ("height", u64::from(plan.height / 32)),
                        ("width", u64::from(plan.width / 32)),
                        ("channels", d.latent_channels as u64),
                        ("stages", d.num_stages() as u64),
                        (
                            "tail_stages",
                            d.block_types
                                .iter()
                                .take_while(|t| **t == BlockType::Res)
                                .count() as u64,
                        ),
                        (
                            "input_width",
                            *d.block_out_channels
                                .last()
                                .ok_or_else(|| invalid("empty DC-AE stages"))?
                                as u64,
                        ),
                        (
                            "dtype",
                            mlx_gen::sampler::sampling_dtype_code(Dtype::Float32),
                        ),
                    ],
                )),
                labels::DECODED_PIXELS => Some((
                    3,
                    "sana",
                    "native-pixels",
                    vec![
                        ("schema", 1),
                        ("batch", 1),
                        ("channels", 3),
                        ("height", u64::from(plan.height)),
                        ("width", u64::from(plan.width)),
                        (
                            "dtype",
                            mlx_gen::sampler::sampling_dtype_code(Dtype::Float32),
                        ),
                        ("nchw_view_of_decoder", 1),
                        ("held_unscaled_elements", latent_elements),
                    ],
                )),
                _ => None,
            };
            let Some((slot, provider, kind, expected)) = expected else {
                continue;
            };
            if check.phase != ios_quantum_phase(&check.label) {
                return Err(invalid("actual source mark phase mismatch"));
            }
            if !fields_match(context(capture, check, provider, kind)?, &expected) {
                return Err(invalid("actual source graph shape/dtype/regime changed"));
            }
            let scope = context(capture, check, "mlx", "allocation-scope")?;
            if !fields_match(
                scope,
                &[
                    ("schema", 1),
                    ("setter_succeeded", 1),
                    ("cache_query_succeeded", 1),
                    ("restoring", 0),
                ],
            ) {
                return Err(invalid(
                    "native allocation scope was not observed successfully",
                ));
            }
            let current = *scope
                .get("page_bytes")
                .filter(|p| p.is_power_of_two())
                .ok_or_else(|| invalid("missing actual native page basis"))?;
            if page.replace(current).is_some_and(|old| old != current) {
                return Err(invalid("inconsistent actual native page basis"));
            }
            seen[slot] = true;
        }
    }
    for (slot, label) in [
        GEMMA_LABELS[0],
        labels::ENTRY,
        labels::DECODER_INPUT,
        labels::DECODED_PIXELS,
    ]
    .iter()
    .enumerate()
    {
        if pricing.windows.iter().any(|w| w.label == *label) && !seen[slot] {
            return Err(invalid(&format!(
                "no source-correlated actual {label} check"
            )));
        }
    }
    page.ok_or_else(|| invalid("no actual native allocation basis"))
}

fn validate_actual_generation(
    req: &IosPricingRequest<'_>,
    generation: &mlx_gen::GenerationRequest,
) -> Result<()> {
    if req.geometry.width != generation.width
        || req.geometry.height != generation.height
        || req.geometry.batch != generation.count
        || req.geometry.frames != 1
    {
        return Err(invalid(
            "actual generation geometry/count differs from pricing transport",
        ));
    }
    if generation
        .conditioning
        .iter()
        .any(|c| !matches!(c, mlx_gen::Conditioning::Reference { .. }))
    {
        return Err(invalid(
            "conditioning role is not served by Sana generation",
        ));
    }
    let reference = crate::model::resolve_reference(generation, req.provider)?;
    let has_reference = reference.is_some();
    let mode_matches = matches!(
        (req.mode, has_reference),
        (MemoryMode::TextToImage, false) | (MemoryMode::ImageToImage, true)
    );
    if !mode_matches || req.geometry.reference_count != u32::from(has_reference) {
        return Err(invalid(
            "actual reference role differs from pricing mode/geometry",
        ));
    }
    if generation
        .memory
        .is_some_and(|actual| actual != *req.execution)
    {
        return Err(invalid(
            "actual generation memory levers differ from pricing execution",
        ));
    }
    Ok(())
}

fn controls(side: &IosBoundRequest<'_>) -> Result<(PricingSampling, u32)> {
    let req = side.pricing;
    req.validate_work_context()?;
    if req.transform.is_some() || req.depth.is_some() {
        return Err(invalid("Sana certificate requires image-generation work"));
    }
    if let Some(generation) = req.generation {
        validate_actual_generation(req, generation)?;
    }
    let has_init = matches!(req.mode, MemoryMode::ImageToImage);
    match (req.generation, side.replay) {
        (Some(generation), None) => Ok((
            match req.provider {
                crate::MODEL_ID => PricingSampling::Flow(crate::sampling::resolve_base_with_init(
                    generation, has_init,
                )?),
                crate::SPRINT_MODEL_ID => PricingSampling::Scm(
                    crate::sampling::resolve_sprint_with_init(generation, has_init)?,
                ),
                _ => return Err(invalid("unknown provider")),
            },
            generation.count,
        )),
        (None, Some(replay)) => {
            if replay.true_reference_count != u32::from(has_init)
                || replay.mask_count != 0
                || replay.control_count != 0
                || replay.source.is_some()
            {
                return Err(invalid(
                    "replay conditioning/source roles do not match Sana's served route",
                ));
            }
            let sampling = match req.provider {
                crate::MODEL_ID => PricingSampling::Flow(
                    mlx_gen::sampler::flow_sampling::ResolvedFlowControls::from_recipe(
                        &mlx_gen::sampler::flow_sampling::FlowSamplingRecipe::from_token(
                            replay.token,
                        )?,
                        crate::sampling::base_defaults(),
                        mlx_gen::TimestepConvention::Sigma,
                    )?,
                ),
                crate::SPRINT_MODEL_ID => {
                    PricingSampling::Scm(crate::sampling::ResolvedSprintControls::from_recipe(
                        &crate::sampling::ScmSamplingRecipe::from_token(replay.token)?,
                        has_init,
                    )?)
                }
                _ => return Err(invalid("unknown replay provider")),
            };
            Ok((sampling, req.geometry.batch))
        }
        _ => Err(invalid(
            "needs exactly one actual generation or recorded controls replay",
        )),
    }
}
fn materialize(sampling: PricingSampling, has_init: bool, budget: u64) -> Result<NativeSampling> {
    match sampling {
        PricingSampling::Flow(controls) => {
            controls.check_capture_schedule_bytes(budget)?;
            let steps = controls.steps();
            let strength = controls.recipe().effective_strength();
            let native =
                mlx_gen::FlowMatchEuler::for_static_shift(steps, crate::pipeline::SCHEDULE_SHIFT);
            let mut resolved = controls
                .materialize_schedule(&native.sigmas, crate::pipeline::SCHEDULE_SHIFT.ln())?;
            if has_init {
                let strength = mlx_gen::img2img::resolve_strength(None, strength);
                let start = mlx_gen::img2img::init_time_step(steps, Some(strength));
                resolved.window(start, resolved.sigmas().len())?;
            }
            Ok(NativeSampling::Flow(resolved))
        }
        PricingSampling::Scm(controls) => {
            Ok(NativeSampling::Scm(controls.materialize_capture(budget)?))
        }
    }
}

#[allow(clippy::too_many_arguments)] // Source facts and supplied profiles are independently checked.
pub(super) fn certify_profiles(
    anchor_facts: &SanaPricingFacts,
    anchor_plan: &SanaPricingPlan,
    expected_anchor: &IosPricing,
    anchor: &IosPricing,
    request_facts: &SanaPricingFacts,
    request_plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    expected_request: &IosPricing,
    request: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    if !same_profile(expected_anchor, anchor) || !same_profile(expected_request, request) {
        return Err(invalid(
            "pricing undercounts or changes the source-generated owner/window profile",
        ));
    }
    let page = observed_page(anchor_facts, anchor_plan, anchor)?;
    if !request.observations.is_empty()
        && observed_page(request_facts, request_plan, request)? != page
    {
        return Err(invalid(
            "actual request and anchor allocator page domains differ",
        ));
    }
    let cpu = cumulative_upper(
        request_facts,
        request_plan,
        sampling,
        page,
        NativeBackend::Cpu,
    )?;
    let metal = cumulative_upper(
        request_facts,
        request_plan,
        sampling,
        page,
        NativeBackend::Metal,
    )?;
    let upper = cpu.max(metal);
    let mut seen = std::collections::BTreeSet::new();
    let mut bounds = Vec::new();
    for window in &request.windows {
        let class = window
            .profile_class()
            .ok_or_else(|| invalid("request has an unclassified source window"))?;
        if seen.insert(class) {
            bounds.push(IosWindowBound::RawPhysical {
                domain: IosMemoryDomain::NativeAllocatorGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                request_class: class.into(),
                growth_upper: upper,
            });
        }
    }
    if bounds.is_empty() {
        return Err(invalid("empty native source request"));
    }
    Ok(bounds)
}

/// Complete source-native request upper, with no eval/drop/synchronization
/// retirement credit and no native-to-process claim. The explicit execution
/// analysis budget is checked before either canonical schedule is materialized.
/// Every served solver uses its actual factory and actual schedule tail.
pub fn ios_certified_bounder(
    anchor_request: &IosBoundRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosBoundRequest<'_>,
    request_pricing: &IosPricing,
    execution: &IosExecutionFacts,
) -> Result<IosCertifiedBounds> {
    if !execution.ios_quantum_scope {
        return Err(invalid("execution is outside the iOS quantum source graph"));
    }
    let (anchor_controls, anchor_count) = controls(anchor_request)?;
    let (request_controls, request_count) = controls(request)?;
    // The execution owner supplies this budget. Fail before opening snapshots
    // or allocating any requested-N schedule, never substitute a private cap.
    match &request_controls {
        PricingSampling::Flow(c) => {
            c.check_capture_schedule_bytes(execution.analysis_budget_bytes)?
        }
        PricingSampling::Scm(c) => c.check_capture_budget(execution.analysis_budget_bytes)?,
    }
    let (anchor_facts, anchor_plan) = resolve_source_plan(anchor_request.pricing, anchor_count)?;
    let (request_facts, request_plan) = resolve_source_plan(request.pricing, request_count)?;
    let expected_anchor =
        price_request_controls(anchor_request.pricing, anchor_count, anchor_controls)?;
    // Resolve source pricing and native schedule from the same canonical controls.
    let expected_request = price_request_controls(
        request.pricing,
        request_count,
        match &request_controls {
            PricingSampling::Flow(c) => PricingSampling::Flow(c.clone()),
            PricingSampling::Scm(c) => PricingSampling::Scm(c.clone()),
        },
    )?;
    let sampling = materialize(
        request_controls,
        request_plan.img2img,
        execution.analysis_budget_bytes,
    )?;
    certify_profiles(
        &anchor_facts,
        &anchor_plan,
        &expected_anchor,
        anchor_pricing,
        &request_facts,
        &request_plan,
        &sampling,
        &expected_request,
        request_pricing,
    )
    .map(IosCertifiedBounds::proven)
}

#[cfg(test)]
pub(super) fn fixture_sampling(
    generation: &mlx_gen::GenerationRequest,
    sprint: bool,
    has_init: bool,
    budget: u64,
) -> Result<NativeSampling> {
    let controls = if sprint {
        PricingSampling::Scm(crate::sampling::resolve_sprint_with_init(
            generation, has_init,
        )?)
    } else {
        PricingSampling::Flow(crate::sampling::resolve_base_with_init(
            generation, has_init,
        )?)
    };
    materialize(controls, has_init, budget)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::ios_pricing::IosObservedMetadata;

    // Protocol tests only: these synthetic marks are NEVER passed to the
    // certificate builder or represented as observed production execution.
    #[test]
    fn context_rejects_dangling_duplicate_and_missing_source_marks() {
        let capture = IosAnchorObservation {
            metadata: vec![IosObservedMetadata {
                provider: "sana".into(),
                kind: "native-trunk".into(),
                fields: [("schema".into(), 1), ("streamed".into(), 0)].into(),
            }],
            ..Default::default()
        };
        let mut check = IosObservedCheck {
            phase: Some(IosPhase::Denoise),
            label: labels::ENTRY.into(),
            context: vec![0],
        };
        let mark = context(&capture, &check, "sana", "native-trunk").unwrap();
        assert!(fields_match(mark, &[("schema", 1), ("streamed", 0)]));
        assert!(!fields_match(mark, &[("streamed", 1)]));
        assert!(!fields_match(mark, &[("caption_tokens", 300)]));
        check.context.push(0);
        assert!(context(&capture, &check, "sana", "native-trunk").is_err());
        check.context = vec![u32::MAX];
        assert!(context(&capture, &check, "sana", "native-trunk").is_err());
        check.context.clear();
        assert!(context(&capture, &check, "sana", "native-trunk").is_err());
    }
    #[test]
    fn every_count_and_absence_belongs_to_the_source_profile() {
        let mut expected = IosPricing::default();
        expected.take(IosPhase::Denoise, 32, 64, labels::BLOCK);
        expected.phase_not_run(IosPhase::Encode, "no reference");
        let mut changed = expected.clone();
        changed.phases[0].resident_bytes -= 1;
        assert!(!same_profile(&expected, &changed));
        let mut changed = expected.clone();
        changed.absent_phases.clear();
        assert!(!same_profile(&expected, &changed));
    }
}
