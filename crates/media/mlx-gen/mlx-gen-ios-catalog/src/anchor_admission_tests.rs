//! Admission through the anchor law (epic 16774): exact anchors, derivation, the two-anchor
//! self-check, and the typed refusals — on the SDXL fixture with a synthetic counted pricing.
use super::*;
use crate::evidence::fixture::capture;
use crate::evidence::{AnchorOutcome, AnchorSelection, RefusedQuantum, DEVICE_CAPTURE_CLASS};
use crate::sdxl::tests::{candidate, fixture, DEVICE};
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
    let mut p = IosPricing::default();
    p.take(Conditioning, 600 * MIB, 10 * MIB, "SDXL CLIP layer");
    p.take(
        Denoise,
        if staged { 1500 } else { 2000 } * MIB,
        mp(300 * MIB),
        "SDXL UNet resnet",
    );
    match req.selection.parameters.decode_tile_edge {
        Some(_) => p.take(Decode, 200 * MIB, 400 * MIB, "SDXL VAE tiled stage"),
        None => p.take(Decode, 200 * MIB, mp(1600 * MIB), "SDXL VAE resnet"),
    }
    Ok(p)
}

fn phase_of(_: &str) -> Option<IosPhase> {
    Some(Denoise)
}

fn priced<T>(body: impl FnOnce() -> T) -> T {
    crate::pricing::with_test_pricing(Some((synthetic, phase_of)), body)
}

fn at(
    registry: &ProviderRegistry,
    root: &std::path::Path,
    edge: u32,
    strategy: MemoryStrategy,
    mode: &str,
) -> IosCandidate {
    let mut c = candidate(registry, root, "sdxl", edge, Some(Quant::Q4), mode);
    if strategy != c.context.selection.strategy {
        let IosWork::Generate(request) = c.work.clone() else {
            unreachable!()
        };
        c = build_candidate(
            registry,
            crate::sdxl::SDXL_ID,
            c.spec.clone(),
            None,
            strategy,
            IosWork::Generate(request),
        )
        .unwrap();
    }
    c
}

/// The optimized anchor (at the iOS reference geometry, 1024²): three repeats over a 300 MiB
/// baseline measuring 2480/2500/2490 MiB of growth, with phase peaks conditioning 700, denoise =
/// the run's growth, decode 900 MiB. Against the synthetic pricing at 1024² (U-Net 1500 + 300
/// scratch, tiled decode 200 + 400, text 600 + 10) its unpriced overheads are denoise 700,
/// decode 300 and conditioning 90 MiB.
fn optimized_anchor(registry: &ProviderRegistry, c: &IosCandidate) -> AnchorRow {
    let base = 300 * MIB;
    let phases = |denoise: u64| {
        [
            (Conditioning, base + 700 * MIB),
            (Denoise, base + denoise * MIB),
            (Decode, base + 900 * MIB),
        ]
    };
    AnchorRow {
        key: evidence_key(registry, c, DEVICE).unwrap(),
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
    capacity_budget(64 << 30, 0, Some(available_mib * MIB))
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

/// Other geometries and rungs are DERIVED by the law (exact arithmetic), upward from the
/// optimized anchor; a request needing more than the live headroom is a capacity refusal.
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
    let rows = [optimized_anchor(&registry, &opt)];
    priced(|| {
        // 512², same selection (derived DOWN, the overhead floored): denoise 1500 + 700 + 300 x
        // (75/300 of the calibrated formula) = 2275; decode 200 + 300 + 400 = 900; conditioning
        // 600 + 90 + 10 = 700. Demand 2275.
        let small = at(
            &registry,
            root.path(),
            512,
            MemoryStrategy::BoundedDecode,
            "text_to_image",
        );
        let admitted = admit_one_of(&registry, &small, 4096, &rows).unwrap();
        assert_eq!(admitted.demand_bytes(), 2275 * MIB);
        assert_eq!(
            admitted.basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Optimized
            }
        );
        // Resident (rung 0) at 1024²: the U-Net 2000 resident + 700 + 300 = 3000; the untiled
        // decode is another quantum (κ = 1): 200 + 300 + 1600 = 2100. Demand 3000: upward from the
        // optimized anchor.
        let resident = at(
            &registry,
            root.path(),
            1024,
            MemoryStrategy::Resident,
            "text_to_image",
        );
        let derived = admit_one_of(&registry, &resident, 4096, &rows).unwrap();
        assert_eq!(derived.demand_bytes(), 3000 * MIB);
        assert!(derived.demand_bytes() > 2500 * MIB);
        // Live headroom below the derived demand + margin: capacity, typed.
        assert_eq!(
            kind(admit_one_of(&registry, &resident, 3000 + 31, &rows)),
            Some(RefusalKind::Capacity)
        );
    });
}

/// A component-additive mode (img2img) of the same ARCHITECTURE is derived from the text anchor; a
/// header-different architecture is another family and never derived; a provider without counted
/// pricing admits only its anchors' own keys.
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
        let admitted = admit_one_of(&registry, &img2img, 8192, &rows).unwrap();
        assert_eq!(
            admitted.basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Optimized
            }
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
    // An execution lever outside the ladder (batched CFG; iOS runs SDXL CFG sequentially) is not
    // derivable either.
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

/// The two anchors check each other. A resident anchor the law (from the optimized one)
/// under-predicts beyond the tolerance fails the family's DERIVATIONS closed; the measured anchors
/// still admit at their own keys. A refused resident anchor is capacity evidence at its own key.
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
    let resident_row = |demand_mib: u64| {
        evidence::fixture::completed(
            evidence_key(&registry, &res, DEVICE).unwrap(),
            AnchorKind::Resident,
            AnchorSelection::from_parts(
                res.context.selection.strategy,
                &res.context.selection.parameters,
            ),
            demand_mib * MIB,
        )
    };
    priced(|| {
        // The law predicts the resident anchor at 3000 MiB. Measured 3100: within 12.6%.
        let rows = [optimized_anchor(&registry, &opt), resident_row(3100)];
        assert_eq!(
            admit_one_of(&registry, &big, 8192, &rows)
                .unwrap()
                .demand_bytes(),
            2275 * MIB
        );
        // A resident request at another geometry is derived from the same-regime resident anchor.
        let res_big = at(
            &registry,
            root.path(),
            512,
            MemoryStrategy::Resident,
            "text_to_image",
        );
        assert_eq!(
            admit_one_of(&registry, &res_big, 8192, &rows)
                .unwrap()
                .basis(),
            AdmissionBasis::Derived {
                from: AnchorKind::Resident
            }
        );
        // Measured 3600: the law under-predicts by 16.7% → every derivation fails closed.
        let rows = [optimized_anchor(&registry, &opt), resident_row(3600)];
        assert_eq!(
            kind(admit_one_of(&registry, &big, 8192, &rows)),
            Some(RefusalKind::NoEvidence)
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
            3600 * MIB
        );
        // A resident anchor refused before allocation: its own key is a typed capacity refusal,
        // and the family keeps deriving from the optimized anchor.
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
            },
            ..resident_row(0)
        };
        let rows = [optimized_anchor(&registry, &opt), refused];
        assert_eq!(
            kind(admit_one_of(&registry, &res, 8192, &rows)),
            Some(RefusalKind::Capacity)
        );
        assert_eq!(
            admit_one_of(&registry, &big, 8192, &rows)
                .unwrap()
                .demand_bytes(),
            2275 * MIB
        );
    });
}
