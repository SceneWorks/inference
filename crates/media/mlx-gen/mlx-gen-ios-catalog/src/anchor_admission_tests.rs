//! Admission through the anchor law (epic 16774): exact anchors, derivation, the two-anchor
//! self-check, and the typed refusals — on the SDXL fixture with a synthetic counted pricing.
use super::*;
use crate::evidence::fixture::capture;
use crate::evidence::{AnchorOutcome, AnchorSelection, RefusedQuantum, DEVICE_CAPTURE_CLASS};
use crate::sdxl::tests::{
    admit_fixture_rows as admit_with_rows, candidate, fixture, DEVICE, FIXTURE_BML_BYTES,
};
use mlx_gen::gen_core::{MemoryStrategy, Quant};
use mlx_gen::ios_pricing::IosPhase::{self, *};

const MIB: u64 = 1 << 20;

/// A synthetic counted pricing: text encoders 600 MiB resident (+10 scratch); the U-Net 2000 MiB
/// resident (1500 when staged) + a resnet scratch of 300 MiB per megapixel; the VAE 200 MiB
/// resident, decoding untiled at 1600 MiB per megapixel or tiled at a flat 400 MiB.
fn synthetic(req: &IosPricingRequest<'_>) -> mlx_gen::Result<IosPricing> {
    let mp = |bytes: u64| {
        bytes * u64::from(req.geometry.width) * u64::from(req.geometry.height) / (1024 * 1024)
    };
    let staged = req.selection.strategy != MemoryStrategy::Resident;
    use mlx_gen::gen_core::{LoadShape, OffloadPolicy};
    assert_eq!(
        (req.spec.offload_policy, req.spec.load_shape),
        if staged {
            (
                OffloadPolicy::Sequential,
                LoadShape::DeferredMaterialization,
            )
        } else {
            (OffloadPolicy::Resident, LoadShape::EagerMaterialization)
        },
        "anchor re-pricing must use the ANCHOR load form, not the requesting endpoint's"
    );
    let mut p = IosPricing::default();
    classed(&mut p, Conditioning, 600 * MIB, 10 * MIB, "SDXL CLIP layer");
    classed(
        &mut p,
        Denoise,
        if staged { 1500 } else { 2000 } * MIB,
        mp(300 * MIB),
        "SDXL UNet resnet",
    );
    match req.selection.parameters.decode_tile_edge {
        Some(_) => classed(&mut p, Decode, 200 * MIB, 400 * MIB, "SDXL VAE tiled stage"),
        None => classed(&mut p, Decode, 200 * MIB, mp(1600 * MIB), "SDXL VAE resnet"),
    }
    // img2img adds an ENCODE phase the text anchor never ran.
    if *req.mode == mlx_gen::gen_core::MemoryMode::ImageToImage {
        classed(
            &mut p,
            Encode,
            1500 * MIB,
            1500 * MIB,
            "SDXL VAE encode resnet",
        );
    }
    // The synthetic fixture prices no load phase; text modes run no encode.
    p.phase_not_run(Load, "synthetic fixture prices no load windows");
    if *req.mode != mlx_gen::gen_core::MemoryMode::ImageToImage {
        p.phase_not_run(Encode, "text mode runs no encode");
    }
    Ok(p)
}

fn phase_of(_: &str) -> Option<IosPhase> {
    Some(Denoise)
}

fn priced<T>(body: impl FnOnce() -> T) -> T {
    crate::pricing::with_test_pricing(Some((synthetic, phase_of)), body)
}

/// Every synthetic window is a CLASSED profile (class, basis, owner domain), as the certified law
/// requires; the classes name the synthetic operation per phase.
fn classed(p: &mut IosPricing, phase: IosPhase, resident: u64, scratch: u64, label: &str) {
    p.take_window(mlx_gen::ios_pricing::IosWindowPrice {
        phase,
        class: Some(format!("synthetic/{}/{label}", phase.as_key())),
        label: label.into(),
        resident_bytes: resident,
        held_bytes: 0,
        scratch_bytes: scratch,
        basis: Some(mlx_gen::ios_pricing::IosWorkBasis::default()),
        owner_domain: "synthetic-fixture".into(),
    });
}

/// The synthetic labels' actual-check stand-ins: every capture records one check per synthetic
/// (phase, label), carrying the fixture's real metadata mark as its context.
const SYNTHETIC_CHECKS: [(IosPhase, &str); 5] = [
    (Conditioning, "SDXL CLIP layer"),
    (Denoise, "SDXL UNet resnet"),
    (Decode, "SDXL VAE tiled stage"),
    (Decode, "SDXL VAE resnet"),
    (Encode, "SDXL VAE encode resnet"),
];

fn witnessed(mut row: AnchorRow) -> AnchorRow {
    let roles = [
        ("reference_count", u64::from(row.key.reference_count)),
        ("mask_count", u64::from(row.key.mask_count)),
        ("control_count", u64::from(row.key.control_count)),
    ];
    if let AnchorOutcome::Completed { captures } = &mut row.outcome {
        for capture in captures {
            // completed() already owns this schema witness; raw capture() does not. Keep
            // exactly one, and check any retained witness against this fixture's own key.
            // Neither branch is native execution or physical evidence.
            let metadata = capture.sampling_coverage.as_mut().unwrap();
            let mut existing = metadata
                .marks
                .iter()
                .filter(|mark| mark.provider == "ios_capture" && mark.kind == "conditioning-roles");
            if let Some(mark) = existing.next() {
                assert!(existing.next().is_none(), "duplicate fixture role witness");
                assert_eq!(mark.fields.len(), roles.len());
                for (name, value) in roles {
                    assert_eq!(mark.fields.get(name), Some(&value));
                }
            } else {
                metadata
                    .record("ios_capture", "conditioning-roles", &roles)
                    .unwrap();
            }
            let coverage = capture.quantum_coverage.as_mut().unwrap();
            for (phase, label) in SYNTHETIC_CHECKS {
                coverage.record(Some(phase), label, vec![0]);
            }
        }
    }
    evidence::validate_rows(std::slice::from_ref(&row))
        .expect("synthetic window witness must preserve a valid anchor row");
    row
}

fn window_total(window: &mlx_gen::ios_pricing::IosWindowPrice) -> i128 {
    i128::from(window.resident_bytes)
        + i128::from(window.held_bytes)
        + i128::from(window.scratch_bytes)
}

/// A synthetic NATIVE-ALLOCATOR bounder for the law/admission fixtures: each request class is
/// bounded relative to the anchor's same-phase class by the difference of their synthetic window
/// totals, witnessed by the anchor's actual check of that label. A phase the anchor never ran
/// refuses the whole request, as a provider bounder must.
fn synthetic_bounder(
    _anchor: &IosBoundRequest<'_>,
    at_anchor: &IosPricing,
    _request: &IosBoundRequest<'_>,
    at_request: &IosPricing,
    _execution: &IosExecutionFacts,
) -> mlx_gen::Result<IosCertifiedBounds> {
    use mlx_gen::ios_pricing::{IosAnchorCandidate, IosAnchorWitness};
    let mut bounds = Vec::new();
    for window in &at_request.windows {
        let class = window.class.clone().expect("classed synthetic window");
        let anchor_window = at_anchor
            .windows
            .iter()
            .find(|candidate| candidate.phase == window.phase)
            .ok_or_else(|| {
                mlx_gen::Error::Unsupported(format!("anchor never ran {}", window.phase.as_key()))
            })?;
        let mut witnesses = Vec::new();
        for (capture, observation) in at_anchor.observations.iter().enumerate() {
            for (check, observed) in observation.checks.iter().enumerate() {
                if observed.label == anchor_window.label && observed.phase == Some(window.phase) {
                    witnesses.push(IosAnchorWitness {
                        capture: capture as u32,
                        check: check as u32,
                    });
                }
            }
        }
        bounds.push(IosWindowBound::Transfer {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            request_class: class,
            anchor_candidates: vec![IosAnchorCandidate {
                anchor_class: anchor_window.class.clone().unwrap(),
                delta_growth_upper: window_total(window) - window_total(anchor_window),
                physical_anchor_lower_bound: None,
                witnesses,
            }],
        });
    }
    Ok(IosCertifiedBounds::proven(bounds))
}

fn any_execution(_: &str, _: &IosExecutionFacts) -> mlx_gen::Result<()> {
    Ok(())
}

/// Synthetic pricing AND the synthetic native bounder.
fn certified<T>(body: impl FnOnce() -> T) -> T {
    priced(|| crate::pricing::with_test_bounder(Some((synthetic_bounder, any_execution)), body))
}

/// The stated engineering overhead of the fixtures' approved native-to-process model.
const MODEL_OVERHEAD_MIB: u64 = 8;

fn model() -> IosNativeToProcessModel {
    IosNativeToProcessModel {
        overhead_bytes: MODEL_OVERHEAD_MIB * MIB,
        provenance: "synthetic fixture model, validated by the fixture's two endpoint anchors"
            .into(),
    }
}

/// The explicit, user-approved policy: the model AND its premise id.
fn approved() -> PremisePolicy {
    PremisePolicy::approving([NATIVE_TO_PROCESS_PREMISE_ID.to_owned()])
        .with_native_to_process(model())
}

fn cpu_scoped() -> IosExecutionFacts {
    IosExecutionFacts {
        native_retirement: mlx_gen::ios_pricing::IosNativeRetirementExecution::Ordinary,
        stream: mlx_gen::ios_pricing::IosExecutionStream::Cpu,
        ios_quantum_scope: true,
        analysis_budget_bytes: 1 << 20,
        callback_host: Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
    }
}

fn admit_under(
    registry: &ProviderRegistry,
    c: &IosCandidate,
    available_mib: u64,
    rows: &[AnchorRow],
    policy: &PremisePolicy,
) -> AdmissionResult<Admission> {
    let baseline = IosMemoryBaseline {
        domain: IosMemoryDomain::ProcessGrowth,
        coordinate: IosMemoryCoordinate::BeforeModelLoad,
        bytes: FIXTURE_BML_BYTES,
        source: "synthetic-test-sdk-before-model-load".into(),
    };
    admit_rows_with_facts(
        registry,
        std::slice::from_ref(c),
        DEVICE,
        budget(available_mib),
        rows,
        AdmissionEvidence {
            request_baseline: Some(&baseline),
            execution_facts: Some(&cpu_scoped()),
            premise_policy: policy,
        },
    )
}

/// The default boundary (no execution facts) and the default policy (no premise approved) both
/// refuse a derived request exactly; returns the approved admission for further assertions.
fn admitted_only_with_the_approved_premise(
    registry: &ProviderRegistry,
    c: &IosCandidate,
    available_mib: u64,
    rows: &[AnchorRow],
) -> Admission {
    let refusal = |result: AdmissionResult<Admission>| match result {
        Err(AdmissionError::Refused(refusal)) => {
            assert_eq!(refusal.kind, RefusalKind::NoEvidence, "{}", refusal.reason);
            refusal.reason
        }
        Err(AdmissionError::Failed(error)) => panic!("failed: {error}"),
        Ok(admitted) => panic!("admitted without approval: basis {:?}", admitted.basis()),
    };
    assert!(refusal(admit_one_of(registry, c, available_mib, rows))
        .contains("certified execution facts"));
    assert!(refusal(admit_under(
        registry,
        c,
        available_mib,
        rows,
        &PremisePolicy::default()
    ))
    .contains(NATIVE_TO_PROCESS_PREMISE_ID));
    // Supplying the model WITHOUT approving its premise approves nothing.
    assert!(refusal(admit_under(
        registry,
        c,
        available_mib,
        rows,
        &PremisePolicy::default().with_native_to_process(model()),
    ))
    .contains(NATIVE_TO_PROCESS_PREMISE_ID));
    let admitted = admit_under(registry, c, available_mib, rows, &approved()).unwrap();
    assert_eq!(admitted.certificate_premises(), &[model().premise()]);
    assert!(admitted
        .context()
        .evidence_revision
        .ends_with(&format!(";premises={NATIVE_TO_PROCESS_PREMISE_ID}")));
    admitted
}

/// The resident endpoint anchor at 1024², measured at `demand_mib`.
fn resident_anchor(registry: &ProviderRegistry, res: &IosCandidate, demand_mib: u64) -> AnchorRow {
    witnessed(evidence::fixture::completed(
        crate::source_work::fixture_key(registry, res, DEVICE),
        AnchorKind::Resident,
        AnchorSelection::from_parts(
            res.context.selection.strategy,
            &res.context.selection.parameters,
        ),
        demand_mib * MIB,
    ))
}

fn at(
    registry: &ProviderRegistry,
    root: &std::path::Path,
    edge: u32,
    strategy: MemoryStrategy,
    mode: &str,
) -> IosCandidate {
    let c = candidate(registry, root, "sdxl", edge, Some(Quant::Q4), mode);
    let mut spec = c.spec.clone();
    (if strategy == MemoryStrategy::Resident {
        AnchorKind::Resident
    } else {
        AnchorKind::Optimized
    })
    .apply_load_form(&mut spec);
    build_candidate(registry, crate::sdxl::SDXL_ID, spec, None, strategy, c.work).unwrap()
}

/// The optimized anchor (at the iOS reference geometry, 1024²): three repeats over a 300 MiB
/// baseline measuring 2480/2500/2490 MiB of growth, with phase peaks conditioning 700, denoise =
/// the run's growth, decode 900 MiB. Against the synthetic pricing at 1024² (U-Net 1500 + 300
/// scratch, tiled decode 200 + 400, text 600 + 10) its unpriced overheads are denoise 700,
/// decode 300 and conditioning 90 MiB.
fn optimized_anchor(registry: &ProviderRegistry, c: &IosCandidate) -> AnchorRow {
    witnessed(optimized_anchor_measurement(registry, c))
}

fn optimized_anchor_measurement(registry: &ProviderRegistry, c: &IosCandidate) -> AnchorRow {
    let base = 300 * MIB;
    let phases = |denoise: u64| {
        [
            (Conditioning, base + 700 * MIB),
            (Denoise, base + denoise * MIB),
            (Decode, base + 900 * MIB),
        ]
    };
    AnchorRow {
        key: crate::source_work::fixture_key(registry, c, DEVICE),
        anchor_kind: AnchorKind::Optimized,
        selection: AnchorSelection::from_parts(
            c.context.selection.strategy,
            &c.context.selection.parameters,
        ),
        outcome: AnchorOutcome::Completed {
            captures: vec![
                capture("a", base, base + 2480 * MIB, &phases(2480)),
                {
                    // The device saw the tiled decode formula at half its price.
                    let mut c = capture("b", base, base + 2500 * MIB, &phases(2500));
                    c.quanta.insert(
                        "SDXL VAE tiled stage".into(),
                        evidence::QuantumObservation {
                            priced_scratch_bytes: 400 * MIB,
                            observed_bytes: 200 * MIB,
                            exact: true,
                            preceding_sampling_mark: None,
                        },
                    );
                    c
                },
                capture("c", base, base + 2490 * MIB, &phases(2490)),
            ],
        },
        evidence_class: DEVICE_CAPTURE_CLASS.into(),
    }
}

fn budget(available_mib: u64) -> MemoryBudget {
    capacity_budget(64 << 30, FIXTURE_BML_BYTES, Some(available_mib * MIB))
}

fn admit_one_of(
    registry: &ProviderRegistry,
    c: &IosCandidate,
    available_mib: u64,
    rows: &[AnchorRow],
) -> AdmissionResult<Admission> {
    admit_with_rows(
        registry,
        std::slice::from_ref(c),
        DEVICE,
        budget(available_mib),
        rows,
    )
}

fn kind(r: AdmissionResult<Admission>) -> Option<RefusalKind> {
    match r {
        Err(e) => e.refusal_kind(),
        Ok(a) => panic!(
            "admitted: demand {} basis {:?}",
            a.demand_bytes(),
            a.basis()
        ),
    }
}

/// LOW END: at the optimized anchor's own key, admission IS the measurement — the largest repeat
/// (2500 MiB) plus the stated margin (the 32 MiB floor; the repeats spread 20 MiB) — with no
/// counted-formula inflation and no 512 MiB formula reserve.
#[test]
fn the_optimized_anchor_admits_its_own_key_at_the_measurement_plus_the_stated_margin() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let rows = [optimized_anchor(&registry, &opt)];
    priced(|| {
        let admitted = admit_one_of(&registry, &opt, 2532, &rows).unwrap();
        assert_eq!(admitted.demand_bytes(), 2500 * MIB);
        assert_eq!(admitted.margin_bytes(), evidence::ANCHOR_MARGIN_FLOOR_BYTES);
        assert_eq!(admitted.required_bytes(), 2532 * MIB);
        assert_eq!(admitted.context().predicted_peak_bytes, 2532 * MIB);
        assert_eq!(
            admitted.basis(),
            AdmissionBasis::Measured(AnchorKind::Optimized)
        );
        // The admitted work's guards carry the anchor's per-quantum device calibration.
        assert_eq!(
            admitted.quantum_calibration()["SDXL VAE tiled stage"],
            mlx_gen::ios_quantum::QuantumCalibration {
                priced_bytes: 400 * MIB,
                observed_bytes: 200 * MIB,
            }
        );
        // One MiB less of live headroom than measurement + margin: a typed capacity refusal.
        assert_eq!(
            kind(admit_one_of(&registry, &opt, 2531, &rows)),
            Some(RefusalKind::Capacity)
        );
    });
}

/// Other geometries and rungs are DERIVED only by the certified law: refused exactly at the default
/// boundary/policy, and admitted — upward from the optimized anchor, or from a same-regime
/// resident anchor — only with the approved native-to-process premise, validated against BOTH
/// measured endpoints. A request needing more than the live headroom is a capacity refusal.
#[test]
fn other_geometries_and_rungs_are_derived_and_refused_on_capacity_when_they_exceed_it() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let res = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::Resident,
        "text_to_image",
    );
    let rows = [
        optimized_anchor(&registry, &opt),
        resident_anchor(&registry, &res, 3100),
    ];
    certified(|| {
        // 512², same selection: the measured optimized growth (2500) plus the largest certified
        // class delta (conditioning/decode 0, denoise -225) plus the model's stated overhead.
        let small = at(
            &registry,
            root.path(),
            512,
            MemoryStrategy::BoundedDecode,
            "text_to_image",
        );
        let admitted = admitted_only_with_the_approved_premise(&registry, &small, 4096, &rows);
        assert_eq!(admitted.demand_bytes(), (2500 + MODEL_OVERHEAD_MIB) * MIB);
        assert_eq!(
            admitted.basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Optimized
            }
        );
        // StagedResidency is this fixture's supported NON-endpoint mechanism. SDXL's
        // measured-and-rejected attention input stays covered in the separate test below.
        let intermediate = at(
            &registry,
            root.path(),
            512,
            MemoryStrategy::StagedResidency,
            "text_to_image",
        );
        let admitted =
            admitted_only_with_the_approved_premise(&registry, &intermediate, 4096, &rows);
        assert_eq!(
            admitted.basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Optimized
            }
        );
        assert_eq!(admitted.context().load_shape, intermediate.spec.load_shape);
        // Resident at 512²: from the same-regime resident anchor (3100), all deltas <= 0.
        let resident = at(
            &registry,
            root.path(),
            512,
            MemoryStrategy::Resident,
            "text_to_image",
        );
        let derived = admitted_only_with_the_approved_premise(&registry, &resident, 8192, &rows);
        assert_eq!(derived.demand_bytes(), (3100 + MODEL_OVERHEAD_MIB) * MIB);
        assert_eq!(
            derived.basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Resident
            }
        );
        // Live headroom one MiB below demand + margin: capacity, typed, even when approved.
        assert_eq!(
            kind(admit_under(
                &registry,
                &resident,
                derived.required_mib() as u64 - 1,
                &rows,
                &approved()
            )),
            Some(RefusalKind::Capacity)
        );
        // An approved model cannot be validated by ONE endpoint: no resident anchor, no admission.
        let one_endpoint = [optimized_anchor(&registry, &opt)];
        assert!(no_evidence_reason(admit_under(
            &registry,
            &small,
            4096,
            &one_endpoint,
            &approved()
        ))
        .contains("BOTH"));
    });
}

/// The historical SDXL attention capability is deliberately Missing: it failed the measured
/// memory/bit-exactness bar, not the anchor law. Keep that original input as a precise negative
/// obligation, and verify the REAL endpoint selector never promotes it or stages the resident end.
#[test]
fn sdxl_missing_attention_is_rejected_and_literal_endpoints_select_only_supported_mechanisms() {
    use mlx_gen::gen_core::{LoadShape, OffloadPolicy};
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let c = candidate(
        &registry,
        root.path(),
        "sdxl",
        512,
        Some(Quant::Q4),
        "text_to_image",
    );
    let mut spec = c.spec.clone();
    AnchorKind::Optimized.apply_load_form(&mut spec);
    let contract = registry
        .memory_strategy_contract(crate::sdxl::SDXL_ID, &spec)
        .unwrap()
        .unwrap();
    assert_eq!(
        contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap()
            .support,
        MemoryStrategySupport::Missing
    );
    match build_candidate(
        &registry,
        crate::sdxl::SDXL_ID,
        spec.clone(),
        None,
        MemoryStrategy::BoundedAttention,
        c.work,
    ) {
        Err(AdmissionError::Failed(Error::Unsupported(reason))) => {
            assert_eq!(reason, "sdxl cannot execute BoundedAttention: Missing");
        }
        Err(error) => panic!("wrong bounded-attention rejection: {error}"),
        Ok(_) => panic!("SDXL admitted its explicitly Missing attention strategy"),
    }
    let (resident, resident_strategy) =
        configured_endpoint(&registry, crate::sdxl::SDXL_ID, &spec, AnchorKind::Resident).unwrap();
    assert_eq!(resident_strategy, MemoryStrategy::Resident);
    assert_eq!(
        (resident.offload_policy, resident.load_shape),
        (OffloadPolicy::Resident, LoadShape::EagerMaterialization)
    );
    let (optimized, optimized_strategy) = configured_endpoint(
        &registry,
        crate::sdxl::SDXL_ID,
        &spec,
        AnchorKind::Optimized,
    )
    .unwrap();
    // This q4-on-dense source fixture cannot leave quantized blocks lazy, so rung 4 is Missing;
    // the selector must pass both Missing rungs and retain the actual fully optimized decode end.
    assert_eq!(optimized_strategy, MemoryStrategy::BoundedDecode);
    assert_eq!(
        (optimized.offload_policy, optimized.load_shape),
        (
            OffloadPolicy::Sequential,
            LoadShape::DeferredMaterialization
        )
    );
    let optimized_contract = registry
        .memory_strategy_contract(crate::sdxl::SDXL_ID, &optimized)
        .unwrap()
        .unwrap();
    assert_eq!(
        optimized_contract
            .capability(optimized_strategy)
            .unwrap()
            .support,
        MemoryStrategySupport::Implemented
    );
}

#[test]
fn matching_sampling_class_derives_without_a_float_grid_and_reprices_the_anchor_recipe() {
    use mlx_gen::gen_core::CfgBatching;
    fn context_price(req: &IosPricingRequest<'_>) -> mlx_gen::Result<IosPricing> {
        // Anchors are payload-free token replays; only the actual request carries generation.
        if let Some(actual) = req.generation {
            let resolved = crate::sdxl::sampling::ResolvedSampling::resolve(actual).unwrap();
            assert_eq!(resolved.sampler_name(), "euler");
            assert_eq!(resolved.cfg_batching(), CfgBatching::Batched);
            assert_eq!(
                actual.memory.unwrap().cfg_batching,
                req.execution.cfg_batching
            );
            assert_eq!(resolved.guidance(), 8.25, "the request's own recipe");
        } else {
            assert_eq!(
                req.geometry.width, 1024,
                "the ANCHOR is replayed at anchor geometry"
            );
        }
        synthetic(req)
    }
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let make = |edge, guidance, strategy| {
        let c = at(&registry, root.path(), edge, strategy, "text_to_image");
        let IosWork::Generate(mut request) = c.work else {
            unreachable!()
        };
        request.sampler = Some("euler".into());
        request.guidance = Some(guidance);
        request.memory.as_mut().unwrap().cfg_batching = Some(CfgBatching::Batched);
        build_candidate(
            &registry,
            crate::sdxl::SDXL_ID,
            c.spec,
            None,
            strategy,
            IosWork::Generate(request),
        )
        .unwrap()
    };
    let anchor = make(1024, 7.0, MemoryStrategy::BoundedDecode);
    let resident = make(1024, 7.0, MemoryStrategy::Resident);
    let request = make(512, 8.25, MemoryStrategy::BoundedDecode);
    let rows = [
        optimized_anchor(&registry, &anchor),
        resident_anchor(&registry, &resident, 3100),
    ];
    assert_eq!(
        evidence_key(&registry, &anchor, DEVICE)
            .unwrap()
            .family()
            .unwrap(),
        evidence_key(&registry, &request, DEVICE)
            .unwrap()
            .family()
            .unwrap()
    );
    crate::pricing::with_test_pricing(Some((context_price, phase_of)), || {
        crate::pricing::with_test_bounder(Some((synthetic_bounder, any_execution)), || {
            let admitted =
                admitted_only_with_the_approved_premise(&registry, &request, 4096, &rows);
            assert_eq!(
                admitted.basis(),
                AdmissionBasis::Derived {
                    from: AnchorKind::Optimized
                }
            );
            assert_eq!(
                admitted.context().selection.strategy,
                MemoryStrategy::BoundedDecode
            );
        })
    });
}

/// An unproved mode or a header-different architecture is another family and never derived;
/// a provider without counted pricing admits only its anchors' own keys.
#[test]
fn another_family_or_a_provider_without_pricing_is_no_evidence() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let rows = [optimized_anchor(&registry, &opt)];
    let img2img = at(
        &registry,
        root.path(),
        512,
        MemoryStrategy::BoundedDecode,
        "image_to_image",
    );
    priced(|| {
        // A synthetic formula is not an accepted actual per-class component-additivity proof.
        // The unproved reference mode keeps its own two-endpoint family.
        assert_eq!(
            kind(admit_one_of(&registry, &img2img, 8192, &rows)),
            Some(RefusalKind::NoEvidence)
        );
        assert_ne!(
            evidence_key(&registry, &img2img, DEVICE)
                .unwrap()
                .family()
                .unwrap(),
            rows[0].key.family().unwrap()
        );
    });
    // Another architecture (a U-Net with different headers) is another family: never derived.
    let other_root = fixture();
    crate::sdxl::tests::write_tensor(
        &other_root
            .path()
            .join("unet/diffusion_pytorch_model.safetensors"),
        "F16",
        16,
    );
    let other = at(
        &registry,
        other_root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    assert_ne!(
        evidence_key(&registry, &other, DEVICE)
            .unwrap()
            .architecture_sha256,
        rows[0].key.architecture_sha256
    );
    priced(|| {
        assert_eq!(
            kind(admit_one_of(&registry, &other, 8192, &rows)),
            Some(RefusalKind::NoEvidence)
        );
    });
    // Another route of the SAME architecture (a fine-tune: identical headers) shares the family.
    let realvis = {
        let mut c = candidate(
            &registry,
            root.path(),
            "realvisxl",
            1024,
            Some(Quant::Q4),
            "text_to_image",
        );
        if c.context.selection.strategy != MemoryStrategy::BoundedDecode {
            c = at(
                &registry,
                root.path(),
                1024,
                MemoryStrategy::BoundedDecode,
                "text_to_image",
            );
        }
        c
    };
    let key = evidence_key(&registry, &realvis, DEVICE).unwrap();
    assert_eq!(key.architecture_sha256, rows[0].key.architecture_sha256);
    assert_eq!(key.family(), rows[0].key.family());
    // Derived only by the certified law with the approved premise and BOTH endpoints measured.
    let res = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::Resident,
        "text_to_image",
    );
    let endpoints = [
        optimized_anchor(&registry, &opt),
        resident_anchor(&registry, &res, 3100),
    ];
    certified(|| {
        let admitted =
            admitted_only_with_the_approved_premise(&registry, &realvis, 8192, &endpoints);
        assert_eq!(
            admitted.basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Optimized
            }
        );
        assert_eq!(admitted.demand_bytes(), (2500 + MODEL_OVERHEAD_MIB) * MIB);
    });
    // No pricing: the anchor's own key still admits (a measurement needs no scaling) …
    let big = at(
        &registry,
        root.path(),
        512,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    crate::pricing::with_test_pricing(None, || {
        admit_one_of(&registry, &opt, 8192, &rows).unwrap();
        // … but nothing else of the family does.
        assert_eq!(
            kind(admit_one_of(&registry, &big, 8192, &rows)),
            Some(RefusalKind::NoEvidence)
        );
    });
    // Batched CFG is another resolved allocation class: the sequential anchor cannot license it.
    // A compatible batched anchor is exercised separately above.
    let mut batched = big.clone();
    batched.execution.cfg_batching = Some(mlx_gen::gen_core::CfgBatching::Batched);
    if let IosWork::Generate(r) = &mut batched.work {
        r.memory = Some(batched.execution);
    }
    priced(|| {
        assert_eq!(
            kind(admit_one_of(&registry, &batched, 8192, &rows)),
            Some(RefusalKind::NoEvidence)
        );
    });
    // A family with only a resident anchor has nothing to derive from.
    let mut resident_only = optimized_anchor(&registry, &opt);
    resident_only.anchor_kind = AnchorKind::Resident;
    priced(|| {
        assert_eq!(
            kind(admit_one_of(&registry, &big, 8192, &[resident_only])),
            Some(RefusalKind::NoEvidence)
        );
    });
}

/// The two anchors check each other through the certified law and the SAME approved model: a
/// resident anchor the optimized anchor under-predicts fails the family's DERIVATIONS closed; the
/// measured anchors still admit at their own keys. A refused resident anchor is capacity evidence
/// at its own key, and leaves the model with only one endpoint, so nothing is derived.
#[test]
fn the_two_anchor_self_check_fails_derivations_closed_and_a_refused_resident_is_capacity() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let res = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::Resident,
        "text_to_image",
    );
    let big = at(
        &registry,
        root.path(),
        512,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    certified(|| {
        // The certified law predicts the resident endpoint at 2500 + 1200 (decode) + 8 = 3708 MiB.
        // Measured 3100: covered, so derivation proceeds.
        let rows = [
            optimized_anchor(&registry, &opt),
            resident_anchor(&registry, &res, 3100),
        ];
        assert_eq!(
            admitted_only_with_the_approved_premise(&registry, &big, 8192, &rows).demand_bytes(),
            (2500 + MODEL_OVERHEAD_MIB) * MIB
        );
        // A resident request at another geometry derives from the same-regime resident anchor.
        let res_big = at(
            &registry,
            root.path(),
            512,
            MemoryStrategy::Resident,
            "text_to_image",
        );
        assert_eq!(
            admitted_only_with_the_approved_premise(&registry, &res_big, 8192, &rows).basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Resident
            }
        );
        // Measured 3800 > 3708 predicted: the approved model fails its endpoint validation, so
        // every derivation fails closed even when approved; own keys still admit.
        let rows = [
            optimized_anchor(&registry, &opt),
            resident_anchor(&registry, &res, 3800),
        ];
        assert!(
            no_evidence_reason(admit_under(&registry, &big, 8192, &rows, &approved()))
                .contains("under-predicts")
        );
        assert_eq!(
            admit_one_of(&registry, &opt, 8192, &rows)
                .unwrap()
                .demand_bytes(),
            2500 * MIB
        );
        assert_eq!(
            admit_one_of(&registry, &res, 8192, &rows)
                .unwrap()
                .demand_bytes(),
            3800 * MIB
        );
        // A resident anchor refused before allocation: its own key is a typed capacity refusal,
        // and with one completed endpoint the approved model cannot be validated.
        let refused = AnchorRow {
            outcome: AnchorOutcome::Refused {
                refused_quantum: RefusedQuantum {
                    label: "SDXL UNet load floor".into(),
                    scratch_bytes: 5000 * MIB,
                    reserve_bytes: 512 * MIB,
                    available_bytes: Some(4000 * MIB),
                },
                run_id: "r".into(),
                build_sha: "s".into(),
                build_identity: None,
                os_build: None,
            },
            // A refused endpoint has no completed capture or demand. Construct it directly;
            // a zero-growth completed intermediate is invalid even if its outcome is replaced.
            key: crate::source_work::fixture_key(&registry, &res, DEVICE),
            anchor_kind: AnchorKind::Resident,
            selection: AnchorSelection::from_parts(
                res.context.selection.strategy,
                &res.context.selection.parameters,
            ),
            evidence_class: DEVICE_CAPTURE_CLASS.into(),
        };
        let rows = [optimized_anchor(&registry, &opt), refused];
        assert_eq!(
            kind(admit_one_of(&registry, &res, 8192, &rows)),
            Some(RefusalKind::Capacity)
        );
        assert!(
            no_evidence_reason(admit_under(&registry, &big, 8192, &rows, &approved()))
                .contains("BOTH")
        );
    });
}

fn admit_with_facts(
    registry: &ProviderRegistry,
    c: &IosCandidate,
    rows: &[AnchorRow],
    facts: Option<&IosExecutionFacts>,
) -> AdmissionResult<Admission> {
    let baseline = IosMemoryBaseline {
        domain: IosMemoryDomain::ProcessGrowth,
        coordinate: IosMemoryCoordinate::BeforeModelLoad,
        bytes: FIXTURE_BML_BYTES,
        source: "synthetic-test-sdk-before-model-load".into(),
    };
    admit_rows_with_facts(
        registry,
        std::slice::from_ref(c),
        DEVICE,
        budget(4096),
        rows,
        AdmissionEvidence {
            request_baseline: Some(&baseline),
            execution_facts: facts,
            premise_policy: &PremisePolicy::default(),
        },
    )
}

fn no_evidence_reason(result: AdmissionResult<Admission>) -> String {
    match result {
        Err(AdmissionError::Refused(refusal)) => {
            assert_eq!(refusal.kind, RefusalKind::NoEvidence, "{}", refusal.reason);
            refusal.reason
        }
        Err(AdmissionError::Failed(error)) => panic!("failed instead of refusing: {error}"),
        Ok(admitted) => panic!("admitted: basis {:?}", admitted.basis()),
    }
}

/// HOST: derivation away from an anchor's own key consumes ONLY a registered source bounder for
/// the STATED execution through the certified law. Missing facts, an unaudited stream, an
/// unscoped execution and an uncertifiable transfer each refuse exactly; the anchor's own key is
/// still admitted from its measurement without any execution fact.
#[test]
fn certified_derivation_refuses_exactly_without_facts_audited_stream_or_certificate() {
    use mlx_gen::ios_pricing::IosExecutionStream;
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let rows = [optimized_anchor(&registry, &opt)];
    priced(|| {
        let small = at(
            &registry,
            root.path(),
            512,
            MemoryStrategy::BoundedDecode,
            "text_to_image",
        );
        let facts = |stream, ios_quantum_scope| IosExecutionFacts {
            native_retirement: mlx_gen::ios_pricing::IosNativeRetirementExecution::Ordinary,
            stream,
            ios_quantum_scope,
            analysis_budget_bytes: 1 << 20,
            callback_host: Some(mlx_gen::ios_pricing::IosCallbackHost::NonAllocating),
        };
        assert!(
            no_evidence_reason(admit_with_facts(&registry, &small, &rows, None))
                .contains("certified execution facts")
        );
        for stream in [IosExecutionStream::Cpu, IosExecutionStream::Metal] {
            let unscoped = facts(stream, false);
            assert!(
                no_evidence_reason(admit_with_facts(&registry, &small, &rows, Some(&unscoped)))
                    .contains("unscoped execution is not certified"),
                "{stream:?}"
            );
            // SDXL's ledger covers BOTH stream families, so neither is refused for its stream.
            // A synthetic counted price is still not SDXL's source-generated profile: no
            // certificate (or the anchor's real SDXL replay over this header-only fixture
            // refuses first).
            let scoped = facts(stream, true);
            let reason =
                no_evidence_reason(admit_with_facts(&registry, &small, &rows, Some(&scoped)));
            assert!(
                !reason.contains("no audited native-allocation law")
                    && (reason.contains("no transfer certificate")
                        || reason.contains("not derivable")),
                "{stream:?}: {reason}"
            );
        }
        let exact = admit_with_facts(&registry, &opt, &rows, None).unwrap();
        assert_eq!(
            exact.basis(),
            AdmissionBasis::Measured(AnchorKind::Optimized)
        );
    });
}

/// HOST: premise-dependent bounds are refused by default, naming the premise; only an explicit
/// policy approving every named premise accepts them.
#[test]
fn premise_dependent_bounds_need_explicit_approval_of_every_premise() {
    let premise = |id: &str| IosCertificatePremise {
        id: id.into(),
        statement: format!("{id} is asserted, not source-proved"),
    };
    let premises = [premise("metal-synchronize-retirement"), premise("other")];
    let refusal = PremisePolicy::default().accept(&premises).unwrap_err();
    assert!(refusal.contains("metal-synchronize-retirement (") && refusal.contains("other ("));
    let partial = PremisePolicy::approving(["metal-synchronize-retirement".to_owned()]);
    let refusal = partial.accept(&premises).unwrap_err();
    assert!(!refusal.contains("metal-synchronize-retirement") && refusal.contains("other ("));
    PremisePolicy::approving([
        "metal-synchronize-retirement".to_owned(),
        "other".to_owned(),
    ])
    .accept(&premises)
    .unwrap();
    PremisePolicy::default().accept(&[]).unwrap();
}

/// HOST: the bounder registry names exactly the providers that expose source-owned bounders.
#[test]
fn only_providers_exposing_source_bounders_are_registered() {
    for provider in [
        "sana_1600m",
        "sana_sprint_1600m",
        crate::sdxl::SDXL_ID,
        crate::zimage::Z_IMAGE_TURBO_ID,
        crate::zimage::Z_IMAGE_TURBO_CONTROL_ID,
        "iris_3b",
        IRIS_RESTORE_ID,
        IRIS_DEPTH_ID,
    ] {
        assert!(
            crate::pricing::provider_bounder(provider).is_some(),
            "{provider}"
        );
    }
    let provider = crate::anima::ANIMA_TURBO_ID;
    assert!(
        crate::pricing::provider_bounder(provider).is_none(),
        "{provider}"
    );
}

/// The synthetic native bounder plus a DECLARED source-derived host-growth upper per class.
const HOST_GROWTH_MIB: u64 = 64;

fn host_declaring_bounder(
    anchor: &IosBoundRequest<'_>,
    at_anchor: &IosPricing,
    request: &IosBoundRequest<'_>,
    at_request: &IosPricing,
    execution: &IosExecutionFacts,
) -> mlx_gen::Result<IosCertifiedBounds> {
    let native = synthetic_bounder(anchor, at_anchor, request, at_request, execution)?;
    Ok(IosCertifiedBounds::proven_with_host(
        native.bounds,
        HOST_GROWTH_MIB * MIB,
        0,
    ))
}

/// The synthetic native bounder with host growth that has no finite bound yet.
fn host_unproved_bounder(
    anchor: &IosBoundRequest<'_>,
    at_anchor: &IosPricing,
    request: &IosBoundRequest<'_>,
    at_request: &IosPricing,
    execution: &IosExecutionFacts,
) -> mlx_gen::Result<IosCertifiedBounds> {
    let mut native = synthetic_bounder(anchor, at_anchor, request, at_request, execution)?;
    native.host = mlx_gen::ios_pricing::IosHostGrowth::Unproved("synthetic tokenizer gap".into());
    Ok(native)
}

/// HOST: a declared host-growth term is consumed ONLY under the separately approved
/// host-extended premise: approving the original native-to-process premise alone refuses,
/// naming the host premise; approving both ADDS the host upper to the derived delta (native
/// delta and stated overhead retained) and records both premises. The default approves nothing.
#[test]
fn declared_host_growth_needs_its_own_premise_and_is_added_not_subtracted() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let res = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::Resident,
        "text_to_image",
    );
    let rows = [
        optimized_anchor(&registry, &opt),
        resident_anchor(&registry, &res, 3100),
    ];
    priced(|| {
        crate::pricing::with_test_bounder(Some((host_declaring_bounder, any_execution)), || {
            let small = at(
                &registry,
                root.path(),
                512,
                MemoryStrategy::BoundedDecode,
                "text_to_image",
            );
            let refused = |policy: &PremisePolicy| match admit_under(
                &registry, &small, 4096, &rows, policy,
            ) {
                Err(AdmissionError::Refused(refusal)) => refusal.reason,
                Err(AdmissionError::Failed(error)) => panic!("failed: {error}"),
                Ok(admitted) => panic!("admitted: basis {:?}", admitted.basis()),
            };
            assert!(refused(&PremisePolicy::default()).contains(NATIVE_TO_PROCESS_PREMISE_ID));
            assert!(refused(&approved()).contains(NATIVE_TO_PROCESS_HOST_PREMISE_ID));
            let both = PremisePolicy::approving([
                NATIVE_TO_PROCESS_PREMISE_ID.to_owned(),
                NATIVE_TO_PROCESS_HOST_PREMISE_ID.to_owned(),
            ])
            .with_native_to_process(model());
            let admitted = admit_under(&registry, &small, 4096, &rows, &both).unwrap();
            assert_eq!(
                admitted.demand_bytes(),
                (2500 + MODEL_OVERHEAD_MIB + HOST_GROWTH_MIB) * MIB
            );
            assert_eq!(
                admitted.certificate_premises(),
                &[model().premise(), model().host_premise()]
            );
        });
        // Unproved host growth is never zero: even fully approved, the mapping refuses naming it.
        crate::pricing::with_test_bounder(Some((host_unproved_bounder, any_execution)), || {
            let small = at(
                &registry,
                root.path(),
                512,
                MemoryStrategy::BoundedDecode,
                "text_to_image",
            );
            let both = PremisePolicy::approving([
                NATIVE_TO_PROCESS_PREMISE_ID.to_owned(),
                NATIVE_TO_PROCESS_HOST_PREMISE_ID.to_owned(),
            ])
            .with_native_to_process(model());
            match admit_under(&registry, &small, 4096, &rows, &both) {
                Err(AdmissionError::Refused(refusal)) => {
                    assert!(
                        refusal.reason.contains("synthetic tokenizer gap"),
                        "{}",
                        refusal.reason
                    )
                }
                Err(AdmissionError::Failed(error)) => panic!("failed: {error}"),
                Ok(admitted) => panic!("unproved host growth admitted: {:?}", admitted.basis()),
            }
        });
    });
}

/// Transport a catalog capture through the shared production witness selector. The observation
/// stays PROCESS and the certificate stays NATIVE; source metadata acceptance grants no mapping.
#[test]
fn catalog_process_metadata_reaches_native_witness_selectors_without_numeric_conversion() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let c = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let row = optimized_anchor(&registry, &c);
    let mut pricing = IosPricing::default();
    anchor::attach_observations(&mut pricing, &row).unwrap();
    assert!(!pricing.observations.is_empty());
    for observation in &mut pricing.observations {
        assert_eq!(observation.domain, IosMemoryDomain::ProcessGrowth);
        assert!(observation.has_source_witness_coordinate());
        assert!(!observation.checks.is_empty());
        for domain in [
            IosMemoryDomain::Unspecified,
            IosMemoryDomain::NativeAllocatorAbsolute,
        ] {
            observation.domain = domain;
            assert!(!observation.has_source_witness_coordinate());
        }
        observation.domain = IosMemoryDomain::NativeAllocatorGrowth;
        assert!(observation.has_source_witness_coordinate());
        observation.coordinate = IosMemoryCoordinate::BeforeRequestConstruction;
        assert!(!observation.has_source_witness_coordinate());
    }
    let native = IosCertifiedBounds::proven(vec![IosWindowBound::RawPhysical {
        domain: IosMemoryDomain::NativeAllocatorGrowth,
        coordinate: IosMemoryCoordinate::BeforeModelLoad,
        request_class: "source-native-only".into(),
        growth_upper: 1,
    }]);
    assert!(in_anchor_domain(
        native,
        &anchor::measured_anchor(&row).unwrap(),
        &PremisePolicy::default()
    )
    .unwrap_err()
    .contains(NATIVE_TO_PROCESS_PREMISE_ID));
}

fn engineering_policy() -> PremisePolicy {
    use crate::engineering::*;
    PremisePolicy::approving([
        PROCESS_PRESSURE_PREMISE_ID.into(),
        PROCESS_RESIDUAL_PREMISE_ID.into(),
        PROCESS_PREPARATION_PREMISE_ID.into(),
    ])
    .with_engineering_process(EngineeringProcessModel {
        machine: DEVICE.into(),
        os_build: "fixture-os".into(),
        build_identity: crate::engineering::test_identity(),
        approval_reference: "synthetic policy test, not target evidence".into(),
    })
}
fn engineering_source(mut row: AnchorRow) -> AnchorRow {
    match &mut row.outcome {
        AnchorOutcome::Completed { captures } => {
            for capture in captures {
                capture.build_sha = "a".repeat(40);
                capture.build_identity = Some(crate::engineering::test_identity());
                capture.os_build = Some("fixture-os".into());
            }
        }
        AnchorOutcome::Refused {
            build_sha,
            build_identity,
            os_build,
            ..
        } => {
            *build_sha = "a".repeat(40);
            *build_identity = Some(Box::new(crate::engineering::test_identity()));
            *os_build = Some("fixture-os".into());
        }
    }
    row
}

fn admit_under_engineering(
    registry: &ProviderRegistry,
    c: &IosCandidate,
    available_mib: u64,
    rows: &[AnchorRow],
    policy: &PremisePolicy,
) -> AdmissionResult<Admission> {
    let baseline = IosMemoryBaseline {
        domain: IosMemoryDomain::ProcessGrowth,
        coordinate: IosMemoryCoordinate::BeforeModelLoad,
        bytes: FIXTURE_BML_BYTES,
        source: "synthetic-estimated-sdk-before-model-load".into(),
    };
    let mut next = |_| Ok(());
    admit_rows_with_facts_using_key_capacity(
        registry,
        std::slice::from_ref(c),
        DEVICE,
        budget(available_mib),
        rows,
        AdmissionEvidence {
            request_baseline: Some(&baseline),
            execution_facts: Some(&cpu_scoped()),
            premise_policy: policy,
        },
        Some(&mut next),
    )
}

/// An explicit empirical estimate accepts a refused resident as endpoint provenance, without
/// manufacturing its demand or source HOST bound. Ordinary certified derivation remains closed.
#[test]
fn engineering_estimate_preserves_refused_resident_and_unknown_host_provenance() {
    use crate::engineering::*;
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let res = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::Resident,
        "text_to_image",
    );
    let small = at(
        &registry,
        root.path(),
        512,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let mut refused = resident_anchor(&registry, &res, 3100);
    refused.outcome = AnchorOutcome::Refused {
        refused_quantum: RefusedQuantum {
            label: "source resident load".into(),
            scratch_bytes: 5000 * MIB,
            reserve_bytes: 512 * MIB,
            available_bytes: Some(4000 * MIB),
        },
        run_id: "fixture-refused".into(),
        build_sha: "a".repeat(40),
        build_identity: Some(Box::new(crate::engineering::test_identity())),
        os_build: Some("fixture-os".into()),
    };
    let rows = [
        engineering_source(optimized_anchor(&registry, &opt)),
        refused,
    ];
    priced(|| {
        crate::pricing::with_test_bounder(None, || {
            assert_eq!(
                kind(admit_under_engineering(
                    &registry,
                    &res,
                    8192,
                    &rows,
                    &engineering_policy()
                )),
                Some(RefusalKind::Capacity)
            );
            let admitted =
                admit_under_engineering(&registry, &small, 4096, &rows, &engineering_policy())
                    .unwrap();
            assert_eq!(admitted.basis(), AdmissionBasis::EngineeringEstimate);
            assert_eq!(admitted.demand_bytes(), 2500 * MIB);
            assert_eq!(admitted.margin_bytes(), 32 * MIB);
            let IosWork::Generate(mut other_factory) = small.work.clone() else {
                unreachable!()
            };
            other_factory.sampler = Some("heun".into());
            let other_factory = build_candidate(
                &registry,
                crate::sdxl::SDXL_ID,
                small.spec.clone(),
                Some(small.context.selection),
                small.context.selection.strategy,
                IosWork::Generate(other_factory),
            )
            .unwrap();
            let other_key = evidence_key(&registry, &other_factory, DEVICE).unwrap();
            assert_ne!(
                rows[0].key.family().unwrap(),
                other_key.family().unwrap(),
                "native factory families remain distinct"
            );
            assert!(crate::engineering::same_pressure_family(
                &rows[0].key,
                &other_key
            ));
            let other_admitted = admit_under_engineering(
                &registry,
                &other_factory,
                4096,
                &rows,
                &engineering_policy(),
            )
            .unwrap();
            assert_eq!(other_admitted.basis(), AdmissionBasis::EngineeringEstimate);
            assert_eq!(other_admitted.demand_bytes(), 2500 * MIB);
            let premises = admitted.certificate_premises();
            assert!(premises.iter().any(|p| p.id == PROCESS_PRESSURE_PREMISE_ID
                && p.statement.contains("no second numeric endpoint")));
            assert!(premises.iter().any(|p| p.id == PROCESS_RESIDUAL_PREMISE_ID
                && p.statement
                    .contains("no finite source HOST/driver certificate claim")));
            let unapproved = PremisePolicy::default().with_engineering_process(
                engineering_policy().engineering_process().unwrap().clone(),
            );
            assert!(no_evidence_reason(admit_under_engineering(
                &registry,
                &small,
                4096,
                &rows,
                &unapproved
            ))
            .contains(PROCESS_PREPARATION_PREMISE_ID));
            let mut masked_preparation = rows[0].clone();
            if let AnchorOutcome::Completed { captures } = &mut masked_preparation.outcome {
                for capture in captures {
                    let masked = capture.baseline_footprint_bytes + 900 * MIB;
                    capture
                        .before_request_snapshot
                        .as_mut()
                        .unwrap()
                        .lifetime_peak_bytes = Some(masked);
                    capture
                        .before_model_load_snapshot
                        .as_mut()
                        .unwrap()
                        .lifetime_peak_bytes = Some(masked);
                }
            }
            assert_eq!(engineering_policy().engineering_process().unwrap().preparation_estimate(&masked_preparation).unwrap(),932*MIB,"masked preparation is a labelled interval upper, not rejected or called exact demand");
            let baseline = IosMemoryBaseline {
                domain: IosMemoryDomain::ProcessGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                bytes: FIXTURE_BML_BYTES,
                source: "source-preparation refusal fixture".into(),
            };
            let calls = std::cell::Cell::new(0);
            let mut denied = |_| {
                calls.set(calls.get() + 1);
                Err(mlx_gen::Error::Unsupported(
                    "source preparation refused".into(),
                ))
            };
            let refusal = admit_rows_with_facts_using_key_capacity(
                &registry,
                std::slice::from_ref(&small),
                DEVICE,
                budget(4096),
                &rows,
                AdmissionEvidence {
                    request_baseline: Some(&baseline),
                    execution_facts: Some(&cpu_scoped()),
                    premise_policy: &engineering_policy(),
                },
                Some(&mut denied),
            );
            assert_eq!(kind(refusal), Some(RefusalKind::Capacity));
            assert_eq!(
                calls.get(),
                1,
                "denial precedes any guarded source/key/pricing reads"
            );
            let mut provenance_only = rows.clone();
            if let AnchorOutcome::Completed { captures } = &mut provenance_only[0].outcome {
                for capture in captures {
                    capture.build_sha = "b".repeat(40);
                }
            }
            assert_eq!(
                admit_under_engineering(
                    &registry,
                    &small,
                    4096,
                    &provenance_only,
                    &engineering_policy()
                )
                .unwrap()
                .demand_bytes(),
                2500 * MIB
            );
            let mut wrong_source = rows.clone();
            if let AnchorOutcome::Completed { captures } = &mut wrong_source[0].outcome {
                captures[0]
                    .build_identity
                    .as_mut()
                    .unwrap()
                    .source_closure_sha256 = "b".repeat(64);
            }
            assert!(no_evidence_reason(admit_under_engineering(
                &registry,
                &small,
                4096,
                &wrong_source,
                &engineering_policy()
            ))
            .contains("source/native identity"));

            let mut wrong_inventory = rows.clone();
            wrong_inventory[0].key.layout_sha256 = "b".repeat(64);
            assert!(no_evidence_reason(admit_under_engineering(
                &registry,
                &small,
                4096,
                &wrong_inventory,
                &engineering_policy()
            ))
            .contains("captured header/config/tokenizer/adapter inventory"));
            let underpredicted = [
                rows[0].clone(),
                engineering_source(resident_anchor(&registry, &res, 8000)),
            ];
            assert!(no_evidence_reason(admit_under_engineering(
                &registry,
                &small,
                16384,
                &underpredicted,
                &engineering_policy()
            ))
            .contains("under-predicts completed resident"));
            assert!(no_evidence_reason(admit_under(
                &registry,
                &small,
                4096,
                &rows,
                &engineering_policy()
            ))
            .contains("capacity callback"));
        })
    });
}

/// Public file-backed candidates arrive unbound; admission must retain the guarded inventory
/// before the real Iris request and optimized-anchor metadata pricers consume it.
#[test]
fn engineering_unbound_iris_candidates_reach_real_metadata_pricing() {
    use mlx_gen::gen_core::{Image, WeightsSource};
    let registry = crate::provider_registry().unwrap();
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../mlx-gen-iris/tests/fixtures");
    for (provider, directory) in [
        ("iris_3b", "tiny-snapshot/iris"),
        (IRIS_RESTORE_ID, "tiny-snapshot/upscaler"),
        (IRIS_DEPTH_ID, "tiny-depth"),
    ] {
        let mut source = LoadSpec::new(WeightsSource::Dir(fixtures.join(directory)));
        if provider == "iris_3b" {
            source.components.insert(
                "text_encoder".into(),
                WeightsSource::Dir(fixtures.join("tiny-snapshot/text_encoder")),
            );
        }
        let work = |edge| {
            let image = || Image {
                width: edge,
                height: edge,
                pixels: vec![0; (edge * edge * 3) as usize],
            };
            match provider {
                IRIS_RESTORE_ID => IosWork::Restore(mlx_gen::gen_core::TransformRequest {
                    image: image(),
                    ..Default::default()
                }),
                IRIS_DEPTH_ID => IosWork::Depth(gen_core_iris::depth::DepthRequest {
                    image: image(),
                    ..Default::default()
                }),
                _ => IosWork::Generate(GenerationRequest {
                    prompt: "host metadata".into(),
                    width: edge,
                    height: edge,
                    steps: Some(1),
                    guidance: Some(1.0),
                    ..Default::default()
                }),
            }
        };
        let opt_spec = configured_load_form(
            &registry,
            provider,
            &source,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap();
        let opt = build_candidate(
            &registry,
            provider,
            opt_spec,
            None,
            MemoryStrategy::BoundedTransformerResidency,
            work(16),
        )
        .unwrap();
        let resident = build_candidate(
            &registry,
            provider,
            source.clone(),
            None,
            MemoryStrategy::Resident,
            work(16),
        )
        .unwrap();
        let request = build_candidate(
            &registry,
            provider,
            source,
            None,
            MemoryStrategy::Resident,
            work(32),
        )
        .unwrap();
        assert!(request.spec.source_inventory().is_none());
        let mut optimized = evidence::fixture::completed(
            crate::source_work::fixture_key(&registry, &opt, DEVICE),
            AnchorKind::Optimized,
            AnchorSelection::from_parts(
                opt.context.selection.strategy,
                &opt.context.selection.parameters,
            ),
            16 * MIB,
        );
        optimized = engineering_source(optimized);
        let resident = AnchorRow {
            key: crate::source_work::fixture_key(&registry, &resident, DEVICE),
            anchor_kind: AnchorKind::Resident,
            selection: AnchorSelection::from_parts(
                resident.context.selection.strategy,
                &resident.context.selection.parameters,
            ),
            outcome: AnchorOutcome::Refused {
                refused_quantum: RefusedQuantum {
                    label: "synthetic resident provenance".into(),
                    scratch_bytes: 1,
                    reserve_bytes: 0,
                    available_bytes: Some(0),
                },
                run_id: "synthetic-resident".into(),
                build_sha: "a".repeat(40),
                build_identity: Some(Box::new(crate::engineering::test_identity())),
                os_build: Some("fixture-os".into()),
            },
            evidence_class: DEVICE_CAPTURE_CLASS.into(),
        };
        let rows = [optimized, resident];
        evidence::validate_rows(&rows).unwrap();
        // No test-pricing override: tokenizer/config/header source pricing and replay are real.
        let admission =
            admit_under_engineering(&registry, &request, 4096, &rows, &engineering_policy())
                .unwrap_or_else(|error| panic!("{provider}: {error}"));
        assert_eq!(admission.basis(), AdmissionBasis::EngineeringEstimate);
        assert!(admission.spec.source_inventory().is_some());
        assert!(request.spec.source_inventory().is_none());
    }
}

#[test]
fn engineering_exact_resident_rejects_missing_or_foreign_execution_receipts() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let res = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::Resident,
        "text_to_image",
    );
    let optimized = engineering_source(optimized_anchor(&registry, &opt));
    let resident = engineering_source(resident_anchor(&registry, &res, 3100));
    priced(|| {
        for missing in [false, true] {
            for refused in [false, true] {
                let mut row = resident.clone();
                if refused {
                    row.outcome = AnchorOutcome::Refused {
                        refused_quantum: RefusedQuantum {
                            label: "synthetic".into(),
                            scratch_bytes: 1,
                            reserve_bytes: 0,
                            available_bytes: Some(0),
                        },
                        run_id: "synthetic".into(),
                        build_sha: "a".repeat(40),
                        build_identity: Some(Box::new(crate::engineering::test_identity())),
                        os_build: Some("fixture-os".into()),
                    };
                }
                match &mut row.outcome {
                    AnchorOutcome::Completed { captures } => {
                        for capture in captures {
                            if missing {
                                capture.build_identity = None;
                            } else {
                                capture.os_build = Some("foreign-os".into());
                            }
                        }
                    }
                    AnchorOutcome::Refused {
                        build_identity,
                        os_build,
                        ..
                    } => {
                        if missing {
                            *build_identity = None;
                        } else {
                            *os_build = Some("foreign-os".into());
                        }
                    }
                }
                let rows = [optimized.clone(), row];
                let result =
                    admit_under_engineering(&registry, &res, 8192, &rows, &engineering_policy());
                assert!(no_evidence_reason(result).contains("source/native identity"));
            }
        }
    });
}

#[test]
fn engineering_preparation_preflight_uses_conservative_reviewed_route_tier_scope() {
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let opt = at(
        &registry,
        root.path(),
        1024,
        MemoryStrategy::BoundedDecode,
        "text_to_image",
    );
    let first = engineering_source(optimized_anchor(&registry, &opt));
    let mut larger_preparation = first.clone();
    // A distinct synthetic configured graph shares only the pre-parser route/tier scope.
    larger_preparation.key.configured_variant_sha256 = Some("f".repeat(64));
    if let AnchorOutcome::Completed { captures } = &mut larger_preparation.outcome {
        for capture in captures {
            let snapshot = capture.before_model_load_snapshot.as_mut().unwrap();
            snapshot.current_bytes = Some(500 * MIB);
            snapshot.lifetime_peak_bytes = Some(500 * MIB);
        }
    }
    let rows = [first, larger_preparation];
    evidence::validate_rows(&rows).unwrap();
    let scope = crate::engineering::PreparationScope {
        machine: DEVICE,
        provider: &rows[0].key.provider,
        route: &rows[0].key.route,
        mode: &rows[0].key.mode,
        tier: &rows[0].key.tier,
    };
    let policy = engineering_policy();
    let expected = policy
        .engineering_process()
        .unwrap()
        .preparation_estimate(&rows[1])
        .unwrap();
    assert_eq!(
        policy.preparation_envelope(&rows, scope).unwrap(),
        Some(expected)
    );
    assert!(policy
        .preparation_envelope(
            &rows,
            crate::engineering::PreparationScope {
                route: "different-route",
                ..scope
            }
        )
        .is_err());
    assert!(policy
        .preparation_envelope(
            &rows,
            crate::engineering::PreparationScope {
                tier: "different-tier",
                ..scope
            }
        )
        .is_err());
    let unapproved = PremisePolicy::default()
        .with_engineering_process(policy.engineering_process().unwrap().clone());
    assert!(unapproved
        .preparation_envelope(&rows, scope)
        .unwrap_err()
        .contains("unapproved premise"));
    assert_eq!(
        PremisePolicy::default()
            .preparation_envelope(&rows, scope)
            .unwrap(),
        None
    );
}
