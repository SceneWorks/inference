//! Iris native-active-counter certificates for the correlated anchor law (epic 16774).
//!
//! Every returned bound is `NativeAllocatorGrowth/BeforeModelLoad`, so it can never authorize
//! process growth or task-footprint admission (activity 26569 remains separate). The transfer is
//! deliberately not a correlated subtraction: each request window's delta is the request's own
//! cumulative native allocation upper from `native_ledger` (the larger of its CPU-stream and
//! Metal-stream source graphs, each lowered by its own backend's law). It charges `U(n)` for every
//! source allocator operand and never credits eval, lexical drop, donation or barrier retirement,
//! so it bounds the request's native growth at every instant without any anchor term. The law adds
//! the anchor's measured nonnegative growth on top, which only makes the result more conservative.
//!
//! Restoration returns no certificate: its Rust host planes are outside MLX's active counter, so
//! no native certificate exists for it and the law refuses it with its exact coverage reason.

use mlx_gen::ios_pricing::{
    IosAnchorCandidate, IosAnchorWitness, IosMemoryCoordinate, IosMemoryDomain, IosPricing,
    IosPricingRequest, IosWindowBound,
};
use mlx_gen::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};

use super::native_ledger::cumulative_native_upper_any;

/// A payload-free catalog anchor's recorded native replay token, the exact one its token-replay
/// pricing (`ios_pricing_with_token` / `pricing_facts_with_*_token`) consumes.
#[derive(Clone, Copy, Debug)]
pub enum IrisReplay<'a> {
    Generation(&'a super::SamplingToken),
    Restoration(&'a super::RestorationToken),
    Depth(&'a super::DepthToken),
}

/// Source-owned bounder for `iris_3b` generation and `iris_3b_depth`. Its signature is
/// [`mlx_gen::ios_pricing::IosBounder`]; the anchor must carry its actual work context.
pub fn ios_bounder(
    anchor_request: &IosPricingRequest<'_>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    ios_bounder_with_replay(
        anchor_request,
        None,
        anchor_pricing,
        request,
        request_pricing,
    )
}

/// [`ios_bounder`] for an anchor that is either an actual request or a payload-free token replay:
/// the anchor's execution plan, source profile and witness identity resolve from its recorded
/// token exactly as its token-replay pricing does. The request side is always actual work (its
/// cumulative native upper is the certificate delta). An anchor with neither refuses exactly.
pub fn ios_bounder_with_replay(
    anchor_request: &IosPricingRequest<'_>,
    anchor_replay: Option<IrisReplay<'_>>,
    anchor_pricing: &IosPricing,
    request: &IosPricingRequest<'_>,
    request_pricing: &IosPricing,
) -> Result<Vec<IosWindowBound>> {
    if anchor_request.provider != request.provider || anchor_request.mode != request.mode {
        return Err(Error::Unsupported(
            "Iris native transfer needs one provider family and mode".into(),
        ));
    }
    match request.provider {
        crate::model::MODEL_ID | crate::depth::DEPTH_MODEL_ID => {}
        // Host planes invisible to MLX's active counter: no native certificate exists.
        crate::restoration::MODEL_ID => return Ok(Vec::new()),
        other => {
            return Err(Error::Unsupported(format!(
                "{other}: not an Iris native-certified family"
            )))
        }
    }
    // Profiles must be exactly the source-generated ones: a caller-edited window set must not
    // borrow this certificate.
    let matches_source = |expected: IosPricing, supplied: &IosPricing| {
        expected.phases == supplied.phases
            && expected.windows == supplied.windows
            && expected.absent_phases == supplied.absent_phases
    };
    if !matches_source(
        anchor_source_pricing(anchor_request, anchor_replay)?,
        anchor_pricing,
    ) || !matches_source(super::ios_pricing(request)?, request_pricing)
    {
        return Err(Error::Unsupported(
            "Iris transfer pricing is not the source-generated profile".into(),
        ));
    }
    let identity = AnchorIdentity::of(anchor_request, anchor_replay)?;
    // Every source-matched anchor check, keyed by its actual label and phase.
    let mut witnesses: BTreeMap<(String, Option<mlx_gen::ios_pricing::IosPhase>), Vec<_>> =
        BTreeMap::new();
    let mut pages = BTreeSet::new();
    for (capture_index, capture) in anchor_pricing.observations.iter().enumerate() {
        if capture.domain != IosMemoryDomain::NativeAllocatorGrowth
            || capture.coordinate != IosMemoryCoordinate::BeforeModelLoad
        {
            continue;
        }
        for (check_index, check) in capture.checks.iter().enumerate() {
            let Some(page) = witnessed_page(capture, check, &identity) else {
                continue;
            };
            pages.insert(page);
            witnesses
                .entry((check.label.clone(), check.phase))
                .or_default()
                .push(IosAnchorWitness {
                    capture: u32::try_from(capture_index).map_err(|_| overflow())?,
                    check: u32::try_from(check_index).map_err(|_| overflow())?,
                });
        }
    }
    // One allocator page operand for the whole certificate; disagreement is not averaged.
    let page = match pages.into_iter().collect::<Vec<_>>()[..] {
        [page] => page,
        [] => {
            return Err(Error::Unsupported(
                "Iris native transfer has no actual source-matched anchor witness".into(),
            ))
        }
        _ => {
            return Err(Error::Unsupported(
                "Iris anchor witnesses disagree on the native allocator page".into(),
            ))
        }
    };
    // The actual marks do not attest the executing stream's device, so the delta is the MAX of the
    // CPU-stream and Metal-stream source ledgers; each is computed from its OWN backend lowering.
    let delta = i128::from(cumulative_native_upper_any(request, page)?);
    // Keep every observed anchor correspondence: the public law takes the MAX over candidates.
    let mut candidates = Vec::new();
    let mut anchor_classes = BTreeSet::new();
    for window in &anchor_pricing.windows {
        let Some(class) = window.profile_class() else {
            continue;
        };
        if !anchor_classes.insert(class) {
            continue;
        }
        let matched: Vec<_> = witnesses
            .iter()
            .filter(|((label, phase), _)| {
                label == &window.label && phase.is_none_or(|phase| phase == window.phase)
            })
            .flat_map(|(_, found)| found.iter().cloned())
            .collect();
        for witness in matched {
            candidates.push(IosAnchorCandidate {
                anchor_class: class.into(),
                delta_growth_upper: delta,
                physical_anchor_lower_bound: None,
                witnesses: vec![witness],
            });
        }
    }
    if candidates.is_empty() {
        return Err(Error::Unsupported(
            "Iris native transfer has no actual anchor window witness".into(),
        ));
    }
    let mut request_classes = BTreeSet::new();
    Ok(request_pricing
        .windows
        .iter()
        .filter_map(|window| window.profile_class())
        .filter(|class| request_classes.insert(*class))
        .map(|class| IosWindowBound::Transfer {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            request_class: class.into(),
            anchor_candidates: candidates.clone(),
        })
        .collect())
}

fn overflow() -> Error {
    Error::Unsupported("Iris native witness index overflow".into())
}

/// The anchor's source-generated profile: its actual work context, else its recorded token
/// through the SAME token-replay pricing the catalog priced it with.
fn anchor_source_pricing(
    req: &IosPricingRequest<'_>,
    replay: Option<IrisReplay<'_>>,
) -> Result<IosPricing> {
    match (req.provider, replay) {
        (_, None) => super::ios_pricing(req),
        (crate::model::MODEL_ID, Some(IrisReplay::Generation(token))) => {
            super::ios_pricing_with_token(req, token)
        }
        (crate::depth::DEPTH_MODEL_ID, Some(IrisReplay::Depth(token))) => {
            Ok(super::pricing_facts_with_depth_token(req, token)?.pricing)
        }
        (crate::restoration::MODEL_ID, Some(IrisReplay::Restoration(token))) => {
            Ok(super::pricing_facts_with_restoration_token(req, token)?.pricing)
        }
        (provider, Some(_)) => Err(Error::Unsupported(format!(
            "{provider}: Iris replay token belongs to another task"
        ))),
    }
}

/// The anchor run's actual execution identity, as its production code emits it.
enum AnchorIdentity {
    Generation {
        serial_count: u64,
        prompt_rows: u64,
        steps: u64,
    },
    Depth {
        source_width: u64,
        source_height: u64,
        model_width: u64,
        model_height: u64,
    },
}

impl AnchorIdentity {
    fn of(req: &IosPricingRequest<'_>, replay: Option<IrisReplay<'_>>) -> Result<Self> {
        if req.provider == crate::model::MODEL_ID {
            let token = match replay {
                None => None,
                Some(IrisReplay::Generation(token)) => Some(token),
                Some(_) => {
                    return Err(Error::Unsupported(
                        "iris_3b anchor replay is not a sampling token".into(),
                    ))
                }
            };
            let resources = mlx_gen::gen_core::iris::GenerationResources::from_spec(
                req.spec,
                crate::model::MODEL_ID,
            )
            .map_err(Error::from)?;
            // The SAME side resolution as the anchor's pricing (actual request or token alone).
            let work = super::generation_work(
                req,
                token,
                &resources,
                req.spec.load_shape == mlx_gen::gen_core::LoadShape::DeferredMaterialization,
            )?;
            Ok(Self::Generation {
                serial_count: u64::from(work.envelope.count),
                prompt_rows: work.envelope.rows,
                steps: work.envelope.plan.len() as u64,
            })
        } else {
            let plan =
                match (req.depth, replay) {
                    (Some(actual), None) => super::depth_input_plan(req.spec, actual)?,
                    (None, Some(IrisReplay::Depth(token))) => token.wire()?.plan()?,
                    (Some(_), Some(_)) => {
                        return Err(Error::Unsupported(
                            "Iris depth anchor carries both an actual request and a replay".into(),
                        ))
                    }
                    _ => return Err(Error::Unsupported(
                        "Iris depth anchor needs its actual depth request or depth replay token"
                            .into(),
                    )),
                };
            Ok(Self::Depth {
                source_width: u64::from(plan.source_width),
                source_height: u64::from(plan.source_height),
                model_width: u64::from(plan.model_width),
                model_height: u64::from(plan.model_height),
            })
        }
    }

    fn matches(&self, kind: &str, fields: &BTreeMap<String, u64>) -> bool {
        let is = |key: &str, want: u64| fields.get(key) == Some(&want);
        match *self {
            Self::Generation {
                serial_count,
                prompt_rows,
                steps,
            } => {
                kind == "request"
                    && is("serial_count", serial_count)
                    && is("prompt_rows", prompt_rows)
                    && is("steps", steps)
            }
            Self::Depth {
                source_width,
                source_height,
                model_width,
                model_height,
            } => {
                kind == "depth_request"
                    && is("source_width", source_width)
                    && is("source_height", source_height)
                    && is("model_width", model_width)
                    && is("model_height", model_height)
            }
        }
    }
}

/// An actual check witnessed inside the anchor's own production request scope: the latest Iris
/// request mark matches the anchor request, and the latest MLX allocation scope reports a
/// successful zero-cache-limit setup with an empty cache and a power-of-two allocator page.
fn witnessed_page(
    capture: &mlx_gen::ios_pricing::IosAnchorObservation,
    check: &mlx_gen::ios_pricing::IosObservedCheck,
    identity: &AnchorIdentity,
) -> Option<u64> {
    let provider = match identity {
        AnchorIdentity::Generation { .. } => crate::model::MODEL_ID,
        AnchorIdentity::Depth { .. } => crate::depth::DEPTH_MODEL_ID,
    };
    let marks = || {
        check
            .context
            .iter()
            .filter_map(|&index| capture.metadata.get(index as usize))
    };
    if !marks().any(|mark| mark.provider == provider && identity.matches(&mark.kind, &mark.fields))
    {
        return None;
    }
    let scope = marks().find(|mark| mark.provider == "mlx" && mark.kind == "allocation-scope")?;
    let field = |key: &str| scope.fields.get(key).copied();
    (field("setter_succeeded") == Some(1)
        && field("cache_query_succeeded") == Some(1)
        && field("cache_limit_requested_bytes") == Some(0)
        && field("cache_bytes") == Some(0)
        && field("restoring") == Some(0))
    .then(|| field("page_bytes"))
    .flatten()
    .filter(|page| page.is_power_of_two())
}
