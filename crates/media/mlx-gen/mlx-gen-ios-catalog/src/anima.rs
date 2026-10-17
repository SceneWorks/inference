//! **Anima Turbo** (`anima_turbo`) on iOS: the load defaults and the memory-ladder selection a phone
//! renders it with (sc-26265).
//!
//! **PLANNING ONLY.** Product admission of Anima Turbo is the single evidence-table seam in
//! [`crate::ios_admission`], exactly like SANA and SDXL: device-captured, artifact/device-bound
//! rows, the live budget, the limits-bound precondition and the provider's registered safety
//! check. Nothing in this module admits a load. It chooses WHICH selection to try (and to capture
//! first), from host numbers that are not device evidence.
//!
//! Nothing here restates the provider's ladder. The rungs, their parameter domains and their
//! prerequisites are `mlx-gen-anima`'s own contract; the planning decision runs the safety check
//! the provider registers — `standard_memory_strategy_safety_check` plus Anima's route gate — with
//! a host-derived predicted peak.
//!
//! # Planning data
//!
//! There is no device measurement of Anima yet. What exists is the provider's host sweep
//! ([`ANIMA_HOST_REQUEST_PEAKS`]): `anima_base`, bf16, 1024², a CFG pair, MLX allocator high-water.
//! It bounds `anima_turbo` because the two share one architecture, one DiT size and one VAE and differ
//! only in the DiT checkpoint and the step count — and Turbo is CFG-free, one denoise forward per step
//! against Base's two. A quantized tier shrinks only weight bytes, so the bf16 rows bound q8/q4 too.
//!
//! A row is either an [`AnimaPeakEvidence::UpperBound`] — a measured request peak, the worse of the
//! recorded runs — or an [`AnimaPeakEvidence::Floor`] — a measured *phase* peak the request cannot
//! cost less than. Only an upper bound admits; a floor can only refuse. A candidate with no row is
//! refused: an unmeasured fit is not a fit.
//!
//! The predicted peak is the row's live peak plus [`IOS_MLX_CACHE_LIMIT_CAP_MIB`], for the reason
//! [`sana_footprint_bound`](crate::sana_footprint_bound) gives: jetsam charges `phys_footprint`,
//! which counts MLX's reuse cache, and the host figure is live allocation only. The cache term is
//! bounded only after `runtime-ios`'s `bound_mlx_to_platform_limits` has run.
//!
//! # What gets selected at 1024² under 4 GiB
//!
//! Staged residency with a tiled decode at **192**, plus the rungs that measurement ran with: chunked
//! attention and the one-block DiT window (rung 4, which needs a
//! [`DeferredMaterialization`](LoadShape::DeferredMaterialization) load — [`anima_turbo_load_spec`]).
//! Rung 4 is not optional here. Staged + tiled *without* the window keeps the whole DiT resident
//! through the denoise, which the provider measured at 5.378 GiB (rung 2) and 5.229 GiB (rung 3) at
//! bf16 — over the budget at every tile edge, since the tile does not touch the denoise phase.
//!
//! Among admitted candidates the one with the **most headroom** wins (ties to the larger tile). That
//! is not SANA's largest-tile rule, on purpose: the provider documents that its request peaks below
//! 384 do not rank (0.36 GiB run-to-run spread on one row, larger than the whole 192→96 range), and on
//! iOS overshoot is a kill, not a slowdown. 192 is also the only small edge whose two runs agreed
//! exactly (2.680 / 2.680 GiB), and it is inside the provider's admitted quality ladder.

use mlx_gen::gen_core::{
    standard_memory_behavior_context, Error, LoadShape, LoadSpec, MemoryBehaviorRoute,
    MemoryBudget, MemoryMode, MemoryNumericTier, MemoryOptimizationAuthority,
    MemoryProviderContract, MemorySafetyDecision, MemorySelection, MemoryStrategy,
    MemoryStrategyParameters, ProviderRegistry, Result, WeightsSource,
};

use crate::{IOS_MEMORY_BUDGET_MIB, IOS_MLX_CACHE_LIMIT_CAP_MIB, IOS_OFFLOAD_POLICY};

/// The Anima variant the iOS catalog ships.
pub const ANIMA_TURBO_ID: &str = "anima_turbo";

/// What one host measurement says about a request's peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnimaPeakEvidence {
    /// A measured request peak (the worse run): the request costs at most this. May admit.
    UpperBound,
    /// A measured phase peak: the request costs at least this. May only refuse.
    Floor,
}

/// One measured Anima request at a square edge, in the provider's units: thousandths of a GiB of
/// MLX live allocation, exactly as `mlx-gen-anima`'s `memory_strategy` tables publish them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnimaHostRequestPeak {
    /// The selection's strategy. Every optimized row ran staged (rung 1 engaged).
    pub strategy: MemoryStrategy,
    /// The decode tile edge, px; `None` for an untiled decode.
    pub decode_tile_edge: Option<u32>,
    /// Square output edge, px.
    pub image_edge: u32,
    /// Live peak, thousandths of a GiB.
    pub live_peak_milli_gib: u32,
    pub evidence: AnimaPeakEvidence,
}

impl AnimaHostRequestPeak {
    /// The live peak in bytes, rounded up.
    pub fn live_peak_bytes(self) -> u64 {
        (u64::from(self.live_peak_milli_gib) << 30).div_ceil(1000)
    }

    /// The iOS footprint bound: live peak plus the capped reuse cache.
    pub fn footprint_bound_bytes(self) -> u64 {
        self.live_peak_bytes() + mib(IOS_MLX_CACHE_LIMIT_CAP_MIB)
    }
}

const fn row(
    strategy: MemoryStrategy,
    decode_tile_edge: Option<u32>,
    live_peak_milli_gib: u32,
    evidence: AnimaPeakEvidence,
) -> AnimaHostRequestPeak {
    AnimaHostRequestPeak {
        strategy,
        decode_tile_edge,
        image_edge: 1024,
        live_peak_milli_gib,
        evidence,
    }
}

/// `anima_base` bf16 at 1024² on Apple Silicon, from `mlx-gen-anima/src/memory_strategy.rs`
/// (`DECODE_TILE_EDGES`, `ATTENTION_CHUNK_SIZE` and `TRANSFORMER_WINDOW_SIZES` docs; measured by
/// `tests/shared_memory_ladder_real_weights.rs`).
///
/// | selection | tile | live peak GiB | kind | source |
/// |---|---:|---:|---|---|
/// | resident | untiled | ≥ 7.921 | floor | untiled decode inside the staged envelope; a resident request runs the same decode with more held |
/// | staged | untiled | ≥ 7.921 | floor | the same decode |
/// | staged + tiled (rung 2) | 768 … 192 | ≥ 5.378 | floor | its denoise phase, measured at 640, which no tile edge moves |
/// | + chunked attention (rung 3) | 640 | 5.229 | request | rung 3's request peak |
/// | + chunked attention (rung 3) | 768, 512 … 192 | ≥ 5.229 | floor | rung 3's denoise phase, again tile-independent |
/// | + DiT window 1 (rung 4) | 768 … 192 | 4.649 … 2.680 | request | the tile sweep inside the full ladder, worse of two runs |
pub const ANIMA_HOST_REQUEST_PEAKS: &[AnimaHostRequestPeak] = {
    use AnimaPeakEvidence::{Floor, UpperBound};
    use MemoryStrategy::{
        BoundedAttention, BoundedDecode, BoundedTransformerResidency, Resident, StagedResidency,
    };
    &[
        row(Resident, None, 7921, Floor),
        row(StagedResidency, None, 7921, Floor),
        row(BoundedDecode, Some(768), 5378, Floor),
        row(BoundedDecode, Some(640), 5378, Floor),
        row(BoundedDecode, Some(512), 5378, Floor),
        row(BoundedDecode, Some(384), 5378, Floor),
        row(BoundedDecode, Some(320), 5378, Floor),
        row(BoundedDecode, Some(256), 5378, Floor),
        row(BoundedDecode, Some(192), 5378, Floor),
        row(BoundedAttention, Some(768), 5229, Floor),
        row(BoundedAttention, Some(640), 5229, UpperBound),
        row(BoundedAttention, Some(512), 5229, Floor),
        row(BoundedAttention, Some(384), 5229, Floor),
        row(BoundedAttention, Some(320), 5229, Floor),
        row(BoundedAttention, Some(256), 5229, Floor),
        row(BoundedAttention, Some(192), 5229, Floor),
        row(BoundedTransformerResidency, Some(768), 4649, UpperBound),
        row(BoundedTransformerResidency, Some(640), 4153, UpperBound),
        row(BoundedTransformerResidency, Some(512), 4261, UpperBound),
        row(BoundedTransformerResidency, Some(384), 3176, UpperBound),
        row(BoundedTransformerResidency, Some(320), 2891, UpperBound),
        row(BoundedTransformerResidency, Some(256), 3036, UpperBound),
        row(BoundedTransformerResidency, Some(192), 2680, UpperBound),
    ]
};

fn mib(value: u32) -> u64 {
    u64::from(value) << 20
}

/// The iOS budget the defaults are chosen against: [`IOS_MEMORY_BUDGET_MIB`], nothing committed, no
/// reclaim credited. The measured cap of the target device replaces it at admission time (E4).
pub fn ios_memory_budget() -> MemoryBudget {
    MemoryBudget {
        total_bytes: mib(IOS_MEMORY_BUDGET_MIB),
        committed_bytes: 0,
        reclaimable_bytes: 0,
        reserved_headroom_bytes: 0,
    }
}

/// An `anima_turbo` load spec with the iOS defaults: the sequential load SANA uses, and a deferred
/// materialization so the DiT window (rung 4) is available — without it the window is `Missing` from
/// the provider's contract and nothing fits ([module docs](self)).
pub fn anima_turbo_load_spec(source: WeightsSource) -> LoadSpec {
    LoadSpec::new(source)
        .with_offload_policy(IOS_OFFLOAD_POLICY)
        .with_load_shape(LoadShape::DeferredMaterialization)
}

/// The host measurement for `selection` at `image_edge`², if there is one.
pub fn anima_host_request_peak(
    selection: &MemorySelection,
    image_edge: u32,
) -> Option<AnimaHostRequestPeak> {
    ANIMA_HOST_REQUEST_PEAKS.iter().copied().find(|record| {
        record.strategy == selection.strategy
            && record.image_edge == image_edge
            && record.decode_tile_edge == selection.parameters.decode_tile_edge
    })
}

/// The tier an Anima load runs, as its safety check compares it.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

/// PLANNING decision for `selection` at `image_edge`² under `budget`: would the host-derived bound
/// fit? This never admits a load (see [`crate::ios_admission::admit`]); it ranks what to try and
/// capture first.
///
/// The decision is the provider's registered safety check (`standard_memory_strategy_safety_check`
/// and Anima's route gate), fed the candidate's host-derived bound as its predicted peak. On top of
/// it: no host row refuses, and a floor that fits still refuses, because a floor says nothing about
/// how much more the request costs.
pub fn anima_turbo_planning_decision(
    registry: &ProviderRegistry,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    selection: &MemorySelection,
    image_edge: u32,
    budget: MemoryBudget,
) -> Result<MemorySafetyDecision> {
    let registration = registry
        .memory_strategy_registrations()
        .find(|registration| registration.provider_id == ANIMA_TURBO_ID)
        .ok_or_else(|| {
            Error::Unsupported(format!("{ANIMA_TURBO_ID} registers no memory strategy"))
        })?;
    let Some(record) = anima_host_request_peak(selection, image_edge) else {
        return Ok(MemorySafetyDecision::Reject {
            reason: format!(
                "{ANIMA_TURBO_ID}: no measurement of {:?} at tile {:?}, {image_edge}²; an \
                 unmeasured fit is not a fit",
                selection.strategy, selection.parameters.decode_tile_edge
            ),
        });
    };
    // The shared weights-free context every provider's behavior fixture is built from, so the
    // route facts (mode, phases, PID, overlay, calibration identity, load shape) cannot drift from
    // the provider's own; only what this admission knows and the fixture does not is overridden.
    let mut context = standard_memory_behavior_context(
        contract,
        selection.strategy,
        loaded_tier(spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            // Anima's phase release is request-scoped, so every load has phases to shed.
            has_phases: true,
            overlay: None,
        },
    )?;
    context.selection = *selection;
    // A host measurement carried to the device under a stated bound: a planning estimate, not a
    // calibration of this device and never admission authority.
    context.optimization_authority = if selection.strategy.is_optimized() {
        MemoryOptimizationAuthority::Estimated
    } else {
        MemoryOptimizationAuthority::Resident
    };
    context.geometry.width = image_edge;
    context.geometry.height = image_edge;
    context.budget = budget;
    context.predicted_peak_bytes = record.footprint_bound_bytes();
    context.evidence_revision = format!(
        "ios-host-bound:anima_base-bf16-{image_edge}:{:?}:{:?}",
        record.strategy, record.decode_tile_edge
    );
    let decision = (registration.safety_check)(spec, contract, &context);
    Ok(match (decision, record.evidence) {
        (MemorySafetyDecision::Accept, AnimaPeakEvidence::Floor) => MemorySafetyDecision::Reject {
            reason: format!(
                "{ANIMA_TURBO_ID}: {:?} at tile {:?} has only a measured floor ({} bytes); it may \
                 cost more than the budget",
                selection.strategy,
                selection.parameters.decode_tile_edge,
                record.footprint_bound_bytes()
            ),
        },
        (decision, _) => decision,
    })
}

/// Every selection the provider's contract publishes for this load: each strategy, at each of its
/// decode tile edges when it engages the decode rung, with the rest of its parameters from the
/// contract's own representative selection. Optimized selections are staged, as measured.
fn candidate_selections(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> Vec<MemorySelection> {
    let tier = loaded_tier(spec);
    let mut candidates = Vec::new();
    for strategy in MemoryStrategy::ALL {
        let Ok(representative) = contract.representative_selection(strategy, tier, false) else {
            // Not implemented on this load (rung 4 on an eager load): nothing to offer.
            continue;
        };
        let mut parameters = representative.parameters;
        if strategy.is_optimized() && strategy != MemoryStrategy::StagedResidency {
            parameters.stage_residency = Some(true);
        }
        let edges = match parameters.decode_tile_edge {
            Some(_) => contract
                .capability(MemoryStrategy::BoundedDecode)
                .map(|capability| capability.parameters.decode_tile_edges.clone())
                .unwrap_or_default(),
            None => Vec::new(),
        };
        if edges.is_empty() {
            candidates.push(MemorySelection {
                parameters,
                ..representative
            });
        } else {
            candidates.extend(edges.into_iter().map(|edge| MemorySelection {
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(edge),
                    ..parameters
                },
                ..representative
            }));
        }
    }
    candidates
}

/// The PLANNED iOS memory selection for an `anima_turbo` render of `image_edge`² under `budget`: of
/// every selection the provider publishes for this load, the one the planning decision
/// ([`anima_turbo_planning_decision`]) accepts with the most headroom, ties to the larger decode
/// tile. An error names why nothing fits the plan. Admission of the planned selection is separate.
pub fn anima_turbo_memory_selection(
    registry: &ProviderRegistry,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    image_edge: u32,
    budget: MemoryBudget,
) -> Result<MemorySelection> {
    let mut admitted = Vec::new();
    let mut refusals = Vec::new();
    for selection in candidate_selections(spec, contract) {
        match anima_turbo_planning_decision(
            registry, spec, contract, &selection, image_edge, budget,
        )? {
            MemorySafetyDecision::Accept => {
                let bound = anima_host_request_peak(&selection, image_edge)
                    .map(AnimaHostRequestPeak::footprint_bound_bytes)
                    .unwrap_or(u64::MAX);
                admitted.push((bound, selection));
            }
            MemorySafetyDecision::Reject { reason } => refusals.push(reason),
        }
    }
    admitted
        .into_iter()
        .min_by_key(|(bound, selection)| {
            (
                *bound,
                std::cmp::Reverse(selection.parameters.decode_tile_edge),
            )
        })
        .map(|(_, selection)| selection)
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "{ANIMA_TURBO_ID}: nothing fits {} bytes at {image_edge}²: {}",
                budget.effective_bytes(),
                refusals.join("; ")
            ))
        })
}

/// [`anima_turbo_memory_selection`] at the iOS defaults: [`ios_memory_budget`] at 1024².
pub fn anima_turbo_ios_selection(
    registry: &ProviderRegistry,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> Result<MemorySelection> {
    anima_turbo_memory_selection(registry, spec, contract, 1024, ios_memory_budget())
}

#[cfg(test)]
mod tests {
    use mlx_gen::gen_core::{
        LoadSpec, MemoryProviderContract, MemorySafetyDecision, MemorySelection, MemoryStrategy,
        WeightsSource,
    };

    use super::*;

    fn spec() -> LoadSpec {
        anima_turbo_load_spec(WeightsSource::Dir(
            "/nonexistent/ios-anima-turbo-fixture".into(),
        ))
    }

    /// The provider's weights-free contract for this load — the declaration the real contract
    /// shares, with no asset facts. Resolving it needs no weights.
    fn contract(registry: &ProviderRegistry, spec: &LoadSpec) -> MemoryProviderContract {
        let fixture = registry
            .memory_contract_fixture_registrations()
            .find(|fixture| fixture.provider_id == ANIMA_TURBO_ID)
            .expect("anima_turbo registers a weights-free contract fixture");
        (fixture.contract)(spec).expect("anima_turbo contract resolves weights-free")
    }

    fn engages(
        contract: &MemoryProviderContract,
        selection: &MemorySelection,
        rung: MemoryStrategy,
    ) -> bool {
        contract.engages_selection(selection, rung)
    }

    /// Under the 4 GiB fixture budget at 1024², the selection is staged with a tiled decode at 192
    /// (plus the window and chunked attention it was measured with), bounded at 2.680 GiB live +
    /// 1 GiB cache, and the provider's own resolver turns it into exactly that request memory.
    #[test]
    fn anima_turbo_selects_staged_tiled_192_under_4_gib_at_1024() {
        let registry = crate::provider_registry().expect("iOS media catalog builds");
        let spec = spec();
        let contract = contract(&registry, &spec);

        assert_eq!(crate::IOS_MEMORY_BUDGET_MIB, 4096);
        let selection = anima_turbo_ios_selection(&registry, &spec, &contract)
            .expect("an iOS Anima Turbo selection fits 4 GiB at 1024²");

        assert!(engages(
            &contract,
            &selection,
            MemoryStrategy::StagedResidency
        ));
        assert!(engages(
            &contract,
            &selection,
            MemoryStrategy::BoundedDecode
        ));
        assert_eq!(selection.parameters.decode_tile_edge, Some(192));
        assert_eq!(
            selection.strategy,
            MemoryStrategy::BoundedTransformerResidency
        );

        let memory = contract
            .generation_memory(&selection)
            .expect("an optimized selection carries request memory");
        assert!(memory.stage_residency, "phases must be shed on iOS");
        assert!(memory.tile_vae_decode, "the decode must be tiled on iOS");
        assert_eq!(memory.decode_tile_edge, Some(192));
        assert!(memory.stream_transformer_blocks);

        let bound = anima_host_request_peak(&selection, 1024)
            .expect("the selection is measured")
            .footprint_bound_bytes();
        assert!(bound <= ios_memory_budget().effective_bytes());
    }

    /// A resident-everything request is refused before load by the provider's safety check — on
    /// the budget, not for want of evidence: its measured floor alone (7.921 GiB) is twice the cap.
    #[test]
    fn anima_turbo_refuses_a_resident_request_before_load() {
        let registry = crate::provider_registry().expect("iOS media catalog builds");
        let spec = spec();
        let contract = contract(&registry, &spec);
        let resident = contract
            .representative_selection(MemoryStrategy::Resident, loaded_tier(&spec), false)
            .expect("resident is always a selection");

        let decision = anima_turbo_planning_decision(
            &registry,
            &spec,
            &contract,
            &resident,
            1024,
            ios_memory_budget(),
        )
        .expect("admission runs weights-free");
        let MemorySafetyDecision::Reject { reason } = decision else {
            panic!("a resident Anima Turbo request was admitted under 4 GiB");
        };
        assert!(
            reason.contains("exceeds effective budget"),
            "expected the budget refusal, got: {reason}"
        );
    }

    /// Staged + tiled without the DiT window keeps the DiT resident through the denoise; its
    /// measured floor is over the cap at every tile edge, so it is refused on the budget — never
    /// for want of a measurement. The window is what makes 4 GiB fit.
    #[test]
    fn staged_tiled_without_the_window_is_refused_under_4_gib() {
        let registry = crate::provider_registry().expect("iOS media catalog builds");
        let spec = spec();
        let contract = contract(&registry, &spec);
        let mut tiled = 0;
        for candidate in candidate_selections(&spec, &contract)
            .into_iter()
            .filter(|candidate| candidate.strategy != MemoryStrategy::BoundedTransformerResidency)
        {
            tiled += usize::from(candidate.parameters.decode_tile_edge.is_some());
            let decision = anima_turbo_planning_decision(
                &registry,
                &spec,
                &contract,
                &candidate,
                1024,
                ios_memory_budget(),
            )
            .unwrap();
            let MemorySafetyDecision::Reject { reason } = decision else {
                panic!(
                    "{:?} at tile {:?} was admitted under 4 GiB",
                    candidate.strategy, candidate.parameters.decode_tile_edge
                );
            };
            assert!(
                !reason.contains("unmeasured"),
                "{:?} at tile {:?} is unmeasured: {reason}",
                candidate.strategy,
                candidate.parameters.decode_tile_edge
            );
            assert!(
                reason.contains("exceeds effective budget"),
                "{:?} at tile {:?} must be refused on the budget, got: {reason}",
                candidate.strategy,
                candidate.parameters.decode_tile_edge
            );
        }
        // Rungs 2 and 3, each at every published tile edge.
        assert_eq!(tiled, 2 * 7);
    }

    /// On an eager load the provider declares rung 4 `Missing`, so nothing is offered that fits and
    /// the selector says so instead of picking an unmeasured rung.
    #[test]
    fn an_eager_load_has_no_ios_selection() {
        let registry = crate::provider_registry().expect("iOS media catalog builds");
        let spec = spec().with_load_shape(LoadShape::EagerMaterialization);
        let contract = contract(&registry, &spec);
        let error = anima_turbo_ios_selection(&registry, &spec, &contract)
            .expect_err("nothing measured fits 4 GiB without the DiT window");
        assert!(error.to_string().contains("nothing fits"), "{error}");
    }

    /// Finding 5: a selection the host plan accepts is still REFUSED by product admission until a
    /// reviewed device row exists; host numbers never admit. The refusal is typed `no_evidence`.
    #[test]
    fn the_planned_selection_is_refused_by_admission_without_a_device_row() {
        let registry = crate::provider_registry().expect("iOS media catalog builds");
        let root = tempfile::tempdir().unwrap();
        for component in ["diffusion_models", "text_encoders", "vae"] {
            crate::sdxl::tests::write_tensor(
                &root
                    .path()
                    .join("split_files")
                    .join(component)
                    .join("fixture.safetensors"),
                "F16",
                8,
            );
        }
        let spec = anima_turbo_load_spec(WeightsSource::Dir(root.path().to_owned()));
        let contract = contract(&registry, &spec);
        let planned = anima_turbo_ios_selection(&registry, &spec, &contract).unwrap();
        assert_eq!(
            anima_turbo_planning_decision(
                &registry,
                &spec,
                &contract,
                &planned,
                1024,
                ios_memory_budget()
            )
            .unwrap(),
            MemorySafetyDecision::Accept,
            "the host plan accepts it"
        );
        let candidate = crate::ios_admission::build_candidate(
            &registry,
            ANIMA_TURBO_ID,
            spec,
            Some(planned),
            planned.strategy,
            crate::ios_admission::IosWork::Generate(mlx_gen::gen_core::GenerationRequest {
                prompt: "p".into(),
                width: 1024,
                height: 1024,
                count: 1,
                ..Default::default()
            }),
        )
        .unwrap();
        let refused = crate::ios_admission::admit(
            &registry,
            std::slice::from_ref(&candidate),
            "iPhone17,2",
            ios_memory_budget(),
        )
        .expect_err("host numbers must never admit a device load");
        assert_eq!(
            refused.refusal_kind(),
            Some(crate::RefusalKind::NoEvidence),
            "{refused}"
        );
    }

    /// The table converts the provider's published GiB figures exactly and in the right direction.
    #[test]
    fn host_peaks_round_up_to_bytes() {
        let tile_192 = ANIMA_HOST_REQUEST_PEAKS
            .iter()
            .find(|record| {
                record.strategy == MemoryStrategy::BoundedTransformerResidency
                    && record.decode_tile_edge == Some(192)
            })
            .unwrap();
        assert_eq!(tile_192.live_peak_bytes(), 2_877_628_089);
        assert!(tile_192.live_peak_bytes() as f64 >= 2.680 * f64::from(1u32 << 30));
        assert_eq!(
            tile_192.footprint_bound_bytes(),
            tile_192.live_peak_bytes() + (1024 << 20)
        );
    }
}
