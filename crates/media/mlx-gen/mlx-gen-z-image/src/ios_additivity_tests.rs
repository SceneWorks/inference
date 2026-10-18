//! The iOS anchor law's ADDITIVITY proofs for Z-Image (epic 16774): iOS admission anchors on the
//! TEXT mode of the base architecture (`z_image_turbo`), and every other request this crate serves
//! is DERIVED from that anchor by [`mlx_gen::ios_anchor_law::derive`] — its extra components
//! re-added through the request's own counted residency, its extra phase (the VAE / control-hint
//! encode) priced by its counted formula plus the anchor's largest unpriced overhead:
//!
//! * `z_image_turbo` reference (img2img);
//! * the `z_image_turbo_control` route — control-only and control + reference, for each control
//!   kind (pose / canny / depth) — from the `z_image_turbo` TEXT anchor, whose spec carries no
//!   control source.
//!
//! Each case runs on the tiny production-structured fixture (`ios_pricing_tests::snapshot`, dense
//! and with its DiT stored packed at q4 / q8) at two selections of the iOS load spec (`Sequential`,
//! `DeferredMaterialization`): the FULLY OPTIMIZED one (the deepest rung the contract declares
//! `Implemented`, the last selection of its `selection_space`) and rung 0 `Resident`. The text
//! anchor and the mode are each MEASURED ([`mlx_gen::ios_anchor_law::measure_phases`]: MLX
//! active-memory growth per phase over a fresh load + generate, the worst of [`RUNS`]); the mode's
//! demand is derived from the text measurement by the law over the counted pricing
//! ([`crate::ios_pricing`]) and must cover the mode's measurement, overall and in every phase both
//! ran ([`bound`]). Every test re-executes itself in a CPU child with Metal made fatal.
//!
//! RESULT. Every phase but the extra ENCODE derives, for every mode, selection and tier. The encode
//! does not: it is always the bounded, layer-tiled encode on iOS, whose CPU tile quanta measure
//! above their counted price (`mlx_gen::ios_quantum_kernels::tile_bytes`: the first tile of each
//! tiled convolution grows ~5 crop-sized buffers past its formula; with the first convolution's
//! output live beside the second's, the resnet's `live + stage + tile` is exceeded by one
//! full-resolution activation). A rung-0 text anchor runs no tiled stage (its decode is whole), so
//! it measured no such overhead for the law to carry into the extra phase: the encode is
//! under-derived by ~10 MB (5%). The optimized anchor's tiled decode measures a comparable overhead,
//! which covers the encode on the dense tier by a few KB and falls a few KB short on q4 / q8. Hence
//! [`crate::ios_pricing::IOS_NON_ADDITIVE_MODES`].
//!
//! The control KIND is not a pricing input (the Fun-Controlnet-Union is a union ControlNet; the
//! kinds share one VAE-encoded control path), but each kind is measured with its own hint image.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, LoadSpec, MemoryNumericTier, MemoryStrategy,
    MemoryStrategySupport, OffloadPolicy, Quant, WeightsSource,
};
use mlx_gen::ios_anchor_law::{area, derive, measure_phases, MeasuredRun};
use mlx_gen::ios_pricing::{IosPhase, IosPricing};
use mlx_gen::ios_quantum::{with_cpu_diagnostic_headroom, Backend};

use crate::ios_pricing::{fold, ios_quantum_phase, Mode, Model, Route};
use crate::ios_pricing_tests::{
    image, packed_transformer, run_request, snapshot, Snapshot, AMPLE, EDGE,
};

const CHILD: &str = "ZIMAGE_IOS_ADDITIVITY_CPU_CHILD";
/// Measured runs per configuration (each a fresh load); the worst per phase is the measurement, as
/// a device anchor capture keeps the largest across its repeats.
const RUNS: usize = 2;

fn in_cpu_child(name: &str) -> bool {
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("ios_additivity_tests::{name}"),
                "--nocapture",
                "--test-threads=1",
                "--include-ignored",
            ])
            .env(CHILD, "1")
            .status()
            .unwrap();
        assert!(status.success(), "CPU child for {name} failed");
        return false;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert!(crate::test_metal_guard::forbid_metal());
    assert!(crate::test_metal_guard::metal_forbidden());
    true
}

/// A control kind's hint image, as the request's preprocessing hands it over: a sparse edge map
/// (canny), a smooth depth ramp (depth), a skeleton render on black (pose).
fn control_image(kind: &str, edge: u32) -> mlx_gen::Image {
    let pixel = |x: u32, y: u32| -> u8 {
        match kind {
            "canny" => {
                if (x * 7 + y * 3).is_multiple_of(23) {
                    255
                } else {
                    0
                }
            }
            "depth" => ((x + y) * 255 / (2 * edge)) as u8,
            "pose" => {
                if x.abs_diff(y) < 2 || x.abs_diff(edge - y) < 2 || x.abs_diff(edge / 2) < 2 {
                    200
                } else {
                    0
                }
            }
            other => panic!("unknown control kind {other}"),
        }
    };
    mlx_gen::Image {
        width: edge,
        height: edge,
        pixels: (0..edge * edge)
            .flat_map(|i| {
                let v = pixel(i % edge, i / edge);
                [v, v, v]
            })
            .collect(),
    }
}

/// One tier of the fixture: the snapshot root its DiT loads from and the load's quant bits.
struct Tier {
    name: &'static str,
    bits: Option<i32>,
    root: std::path::PathBuf,
    _packed: Option<tempfile::TempDir>,
}

fn tiers(snap: &Snapshot) -> Vec<Tier> {
    let mut out = vec![Tier {
        name: "dense",
        bits: None,
        root: snap.root.clone(),
        _packed: None,
    }];
    for (name, bits) in [("q4", 4), ("q8", 8)] {
        let dir = tempfile::tempdir().unwrap();
        packed_transformer(snap, dir.path(), bits);
        for component in ["text_encoder", "vae"] {
            std::os::unix::fs::symlink(snap.root.join(component), dir.path().join(component))
                .unwrap();
        }
        out.push(Tier {
            name,
            bits: Some(bits),
            root: dir.path().to_path_buf(),
            _packed: Some(dir),
        });
    }
    out
}

/// The two selections of `provider`'s iOS load spec (mirrors `mlx_gen_ios_catalog::zimage`'s
/// `z_image_turbo_load_spec`: `Sequential`, deferred materialization, the stored tier requested),
/// as `ios_admission::build_candidate` builds each: (fully optimized, rung 0).
fn selections(provider: &str, root: &Path, bits: Option<i32>) -> [(String, GenerationMemory); 2] {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    spec.quantize = bits.map(|b| if b == 4 { Quant::Q4 } else { Quant::Q8 });
    let contract =
        crate::memory_strategy::weights_free_memory_strategy_contract(provider, &spec).unwrap();
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let deepest = MemoryStrategy::ALL
        .into_iter()
        .rev()
        .find(|s| {
            contract
                .capability(*s)
                .is_some_and(|c| c.support == MemoryStrategySupport::Implemented)
        })
        .unwrap();
    [deepest, MemoryStrategy::Resident].map(|strategy| {
        let selection = *contract
            .selection_space(strategy, tier, false)
            .unwrap()
            .last()
            .unwrap();
        let execution = contract.generation_memory(&selection).unwrap_or_default();
        (format!("{strategy:?}"), execution)
    })
}

/// The counted pricing of `route` on the fixture's tier (the CPU backend the run executes on).
fn pricing(snap: &Snapshot, tier: &Tier, route: &Route) -> IosPricing {
    let mut facts = snap.facts(route.mode.control(), tier.bits);
    facts.dit =
        crate::ios_pricing::Component::read(&WeightsSource::Dir(tier.root.join("transformer")))
            .unwrap();
    fold(
        &Model::new(facts)
            .unwrap()
            .quanta(route, Backend::Cpu)
            .unwrap(),
    )
}

/// One configuration's measurement: the worst of [`RUNS`] fresh runs per phase, and how far the
/// runs disagreed per phase (the capture's own spread).
struct Measured {
    worst: MeasuredRun,
    spread: BTreeMap<IosPhase, u64>,
}

/// [`RUNS`] measured runs of `route` (each a fresh load + generate).
fn measure(snap: &Snapshot, tier: &Tier, route: &Route, hint: &mlx_gen::Image) -> Measured {
    let mut worst = MeasuredRun {
        demand: 0,
        phases: BTreeMap::new(),
    };
    let mut least: BTreeMap<IosPhase, u64> = BTreeMap::new();
    for _ in 0..RUNS {
        let ((), run) = with_cpu_diagnostic_headroom(AMPLE, || {
            measure_phases(ios_quantum_phase, || {
                run_request(snap, &tier.root, route, tier.bits, hint)
            })
        })
        .unwrap();
        worst.demand = worst.demand.max(run.demand);
        for (phase, bytes) in run.phases {
            let slot = worst.phases.entry(phase).or_insert(0);
            *slot = (*slot).max(bytes);
            let low = least.entry(phase).or_insert(u64::MAX);
            *low = (*low).min(bytes);
        }
    }
    let spread = worst
        .phases
        .iter()
        .map(|(p, &hi)| (*p, hi - least[p]))
        .collect();
    Measured { worst, spread }
}

/// The route of `mode` at the fixture geometry under `execution`.
fn route(mode: Mode, execution: GenerationMemory) -> Route {
    Route {
        mode,
        width: EDGE,
        height: EDGE,
        count: 1,
        execution,
    }
}

/// The bound a mode's measured phase must stay under, and how it was formed.
///
/// * A phase the mode prices IDENTICALLY to the anchor (same resident bytes, scratch and binding
///   quantum) runs the same work: the law returns the anchor's own measurement for it, so the
///   mode's capture is a RECAPTURE of the anchor and may differ from it only by capture noise —
///   bounded by the law's measured same-cell recapture spread
///   ([`mlx_gen::ios_anchor_law::CONSISTENCY_TOLERANCE`]).
/// * A phase the mode CHANGES is held to the derivation, up to the larger of the two captures' own
///   run-to-run spread in that phase (MLX's CPU executor releases buffers asynchronously, so
///   identical runs differ by up to ~1% of a phase; a derivation cannot be asked to beat the noise
///   of the measurement it is derived from).
fn bound(
    at_anchor: &IosPricing,
    request: &IosPricing,
    anchor: &Measured,
    mode: &Measured,
    d: &mlx_gen::ios_anchor_law::DerivedPhase,
) -> (u64, &'static str) {
    if at_anchor.phase(d.phase).is_some() && at_anchor.phase(d.phase) == request.phase(d.phase) {
        let spread = d.bytes as f64 * mlx_gen::ios_anchor_law::CONSISTENCY_TOLERANCE;
        (d.bytes + spread.ceil() as u64, " (recapture)")
    } else {
        let noise = |m: &Measured| m.spread.get(&d.phase).copied().unwrap_or(0);
        (d.bytes + noise(anchor).max(noise(mode)), "")
    }
}

/// One derived case's verdict: the table row and the phases (or `demand`) that broke.
struct Verdict {
    line: String,
    failures: Vec<String>,
    failed_phases: Vec<String>,
}

/// Derive `mode` (priced `request`, measured `measured`) from the measured text `anchor` (priced
/// `at_anchor`) and compare.
#[allow(clippy::too_many_arguments)]
fn verdict(
    case: &str,
    selection: &str,
    tier: &str,
    anchor: &Measured,
    at_anchor: &IosPricing,
    request: &IosPricing,
    measured: &Measured,
) -> Verdict {
    let a = area(EDGE, EDGE, 1);
    let derived = derive(&anchor.worst.anchor(a), at_anchor, request, a)
        .unwrap_or_else(|e| panic!("{case} {selection} {tier}: not derivable: {e}"));
    let (mut failures, mut failed_phases, mut phases) = (Vec::new(), Vec::new(), Vec::new());
    // The whole run is bounded by its worst phase's bound (phases run in sequence).
    let demand_bound = derived
        .phases
        .iter()
        .map(|d| bound(at_anchor, request, anchor, measured, d).0)
        .max()
        .unwrap_or(0);
    for d in &derived.phases {
        let Some(&m) = measured.worst.phases.get(&d.phase) else {
            continue;
        };
        let (limit, tag) = bound(at_anchor, request, anchor, measured, d);
        let ok = m <= limit;
        phases.push(format!(
            "{} {m}/{}{tag}{}",
            d.phase.as_key(),
            d.bytes,
            if ok { "" } else { " FAIL" }
        ));
        if !ok {
            failed_phases.push(d.phase.as_key().to_string());
            failures.push(format!(
                "{case} {selection} {tier}: phase {} measured {m} B > {limit} B (derived {} B{tag})",
                d.phase.as_key(),
                d.bytes
            ));
        }
    }
    if measured.worst.demand > demand_bound {
        failed_phases.push("demand".into());
        failures.push(format!(
            "{case} {selection} {tier}: demand measured {} B > {demand_bound} B (derived {} B)",
            measured.worst.demand, derived.demand
        ));
    }
    Verdict {
        line: format!(
            "| {case} | {selection} | {tier} | {} | {} | {} | {} | {} |",
            anchor.worst.demand,
            measured.worst.demand,
            derived.demand,
            if failures.is_empty() { "PASS" } else { "FAIL" },
            phases.join(", ")
        ),
        failures,
        failed_phases,
    }
}

/// Every derived case: the reference mode and both control modes for each control kind.
fn cases() -> Vec<(String, Mode, mlx_gen::Image)> {
    let mut cases = vec![(
        "z_image_turbo reference".to_string(),
        Mode::Reference,
        image(EDGE, 1),
    )];
    for kind in ["pose", "canny", "depth"] {
        cases.push((
            format!("z_image_turbo_control control {kind}"),
            Mode::Control,
            control_image(kind, EDGE),
        ));
        cases.push((
            format!("z_image_turbo_control control+reference {kind}"),
            Mode::ControlReference,
            control_image(kind, EDGE),
        ));
    }
    cases
}

/// The fully optimized selection (index 0 of [`selections`]) and rung 0 (index 1).
const OPTIMIZED: usize = 0;
const RUNG_0: usize = 1;

/// Derive every case in `cases` from the `z_image_turbo` text anchor (measured once per tier) at
/// selection `selection` on each tier named in `tiers`, printing the table. `tamper` edits each
/// derived request's pricing (the mutation check). Returns each case's verdict.
fn prove(
    selection: usize,
    tier_names: &[&str],
    cases: &[(String, Mode, mlx_gen::Image)],
    tamper: impl Fn(Mode, &mut IosPricing),
) -> Vec<(String, Verdict)> {
    let snap = snapshot();
    println!(
        "| case | selection | tier | text anchor B | mode measured B | mode derived B | verdict | per phase measured/derived |"
    );
    println!("|---|---|---|---:|---:|---:|---|---|");
    let mut out = Vec::new();
    for tier in tiers(&snap).iter().filter(|t| tier_names.contains(&t.name)) {
        let anchor = selections(crate::model::MODEL_ID, &tier.root, tier.bits);
        let control = selections(crate::model_control::MODEL_ID, &tier.root, tier.bits);
        let (name, anchor_execution) = &anchor[selection];
        let text_route = route(Mode::Text, *anchor_execution);
        let measured_anchor = measure(&snap, tier, &text_route, &image(EDGE, 1));
        let at_anchor = pricing(&snap, tier, &text_route);
        for (case, mode, hint) in cases {
            let execution = if mode.control() {
                control[selection].1
            } else {
                *anchor_execution
            };
            let mode_route = route(*mode, execution);
            let mut request = pricing(&snap, tier, &mode_route);
            tamper(*mode, &mut request);
            let v = verdict(
                case,
                name,
                tier.name,
                &measured_anchor,
                &at_anchor,
                &request,
                &measure(&snap, tier, &mode_route, hint),
            );
            println!("{}", v.line);
            out.push((case.clone(), v));
        }
    }
    out
}

fn failures(verdicts: &[(String, Verdict)]) -> Vec<String> {
    verdicts
        .iter()
        .flat_map(|(_, v)| v.failures.clone())
        .collect()
}

/// The verdicts' shape (see the module docs): every phase but the extra ENCODE derives (the whole
/// run's `demand` may break only with it, the encode being the run's largest phase), and from a
/// rung-0 anchor every case's encode is under-derived.
fn assert_only_the_encode_breaks(selection: usize, verdicts: &[(String, Verdict)]) {
    for (case, v) in verdicts {
        let broken: Vec<&str> = v.failed_phases.iter().map(String::as_str).collect();
        assert!(
            matches!(broken[..], [] | ["encode"] | ["encode", "demand"]),
            "{case}: a phase other than the encode is under-derived: {:#?}",
            v.failures
        );
        if selection == RUNG_0 {
            assert!(
                broken.first() == Some(&"encode"),
                "{case}: from a rung-0 text anchor the encode must stay under-derived — if it now \
                 derives, re-examine `IOS_NON_ADDITIVE_MODES`"
            );
        }
    }
}

/// THE PROOF, and the failure [`crate::ios_pricing::IOS_NON_ADDITIVE_MODES`] cites. Every Z-Image
/// mode (reference; control and control + reference for each kind) is derived from the
/// `z_image_turbo` text anchor at the fully optimized selection and at rung 0: every phase but the
/// extra encode is covered by its derivation, and the encode is not — from a rung-0 anchor it is
/// under-derived by about one full-resolution activation in every case, and at the optimized
/// selection it is covered only by the anchor's tiled decode measuring a comparable overhead (dense
/// by 2.6–7.7 KB; the packed tiers fall 8–21 KB short). Dense tier; the packed tiers are
/// [`packed_tier_matrix`].
#[cfg(target_os = "macos")]
#[test]
fn only_the_encode_is_not_derivable_from_the_text_anchor() {
    if !in_cpu_child("only_the_encode_is_not_derivable_from_the_text_anchor") {
        return;
    }
    assert_eq!(
        crate::ios_pricing::IOS_NON_ADDITIVE_MODES,
        &["image_to_image", "control"]
    );
    for selection in [OPTIMIZED, RUNG_0] {
        assert_only_the_encode_breaks(
            selection,
            &prove(selection, &["dense"], &cases(), |_, _| {}),
        );
    }
}

/// [`only_the_encode_is_not_derivable_from_the_text_anchor`] on the packed tiers (the DiT stored at
/// q4 and at q8). ~15 min on the CPU fixture.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "the q4/q8 additivity matrix (~15 min of CPU): run with --ignored"]
fn packed_tier_matrix() {
    if !in_cpu_child("packed_tier_matrix") {
        return;
    }
    for selection in [OPTIMIZED, RUNG_0] {
        assert_only_the_encode_breaks(
            selection,
            &prove(selection, &["q4", "q8"], &cases(), |_, _| {}),
        );
    }
}

/// The proof has teeth: a derivation that forgets the control route's ControlNet (the `C_req` re-add
/// of the extra component: its resident bytes in the phases a staged request holds it — the encode
/// and the denoise) goes red at the optimized selection. Dense tier.
#[cfg(target_os = "macos")]
#[test]
fn dropping_the_controlnet_residency_breaks_the_proof() {
    if !in_cpu_child("dropping_the_controlnet_residency_breaks_the_proof") {
        return;
    }
    let snap = snapshot();
    let controlnet = Model::new(snap.facts(true, None))
        .unwrap()
        .control_resident();
    assert!(controlnet > 0);
    let pose: Vec<_> = cases()
        .into_iter()
        .filter(|(case, _, _)| case.ends_with("pose"))
        .collect();
    let failures = failures(&prove(OPTIMIZED, &["dense"], &pose, |mode, pricing| {
        if !mode.control() {
            return;
        }
        for phase in &mut pricing.phases {
            if matches!(phase.phase, IosPhase::Encode | IosPhase::Denoise) {
                phase.resident_bytes -= controlnet;
            }
        }
    }));
    println!("{failures:#?}");
    for case in ["control pose", "control+reference pose"] {
        assert!(
            failures
                .iter()
                .any(|f| f.contains(case) && f.contains("phase encode")),
            "{case}: dropping the ControlNet's residency must break the derivation"
        );
    }
}
