//! Legacy Z-Image additivity regression targets (epic 16774), NOT accepted transfer proofs.
//! Each repeat now retains its actual baseline, ordered metadata/checks and native intervals.
//! Every verdict is computed by the certified law (`ios_anchor_law::derive_certified`) from the
//! provider's source-proved window certificates. A missing certificate is a failing positive
//! obligation (`NotDerivable`), never an accepted refusal or passing regression.
//! Counted logical VAE/control scratch does not certify physical demand or transferable overhead.
//! Runtime/capture grouping therefore preserves distinct reference/control/control+init families.
//!
//! Tiny dense/q4/q8 fixtures retain two literal endpoint load forms: fully optimized deferred
//! and fully resident eager. Earlier resident-labelled runs used deferred full-cover streams.
//! Corrected bounded native endpoint checks do not resolve the original conditioning discrepancy,
//! establish full additivity, or establish target-device/ProcessGrowth completion.
//! CPU children make Metal fatal. Control-kind aliases share the actual union-control carrier;
//! reference presence still changes its source role graph and cannot borrow control-only proof.

use std::collections::BTreeMap;
use std::path::Path;

use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, LoadSpec, MemoryNumericTier, MemoryStrategy,
    MemoryStrategySupport, OffloadPolicy,
};
use mlx_gen::ios_anchor_law::{derive_certified, MeasuredAnchor, MeasuredRun};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosWindowBound};
use mlx_gen::ios_quantum::{with_cpu_diagnostic_headroom, Backend};

use crate::ios_pricing::{fold, ios_quantum_phase, Mode, Model, Route};
use crate::ios_pricing_tests::{
    image, packed_transformer, run_request, snapshot, Snapshot, AMPLE, EDGE,
};

const CHILD: &str = "ZIMAGE_IOS_ADDITIVITY_CPU_CHILD";

const DONE: &str = "ZIMAGE_IOS_ADDITIVITY_CPU_CHILD_DONE";

// Existing CPU-child isolation, with a normal-return marker after every fixture has dropped.
struct CpuChildCompletion;
impl Drop for CpuChildCompletion {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            std::fs::write(
                std::env::var(DONE).expect("CPU child completion path"),
                b"ok",
            )
            .expect("write CPU child completion marker");
        }
    }
}

/// Measured runs per configuration (each a fresh load); the worst per phase is the measurement, as
/// a device anchor capture keeps the largest across its repeats.
const RUNS: usize = 2;

fn in_cpu_child(name: &str) -> bool {
    if std::env::var_os(CHILD).is_none() {
        let done = tempfile::tempdir().unwrap();
        let marker = done.path().join("done");
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("ios_additivity_tests::{name}"),
                "--nocapture",
                "--test-threads=1",
                "--include-ignored",
            ])
            .env(CHILD, "1")
            .env(DONE, &marker)
            .status()
            .unwrap();
        assert!(status.success(), "CPU child for {name} failed");
        assert!(
            marker.is_file(),
            "CPU child for {name} ran no completed body"
        );
        return false;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    assert_eq!(
        mlx_rs::Device::try_default().expect("CPU default-device query failed"),
        mlx_rs::Device::cpu()
    );
    assert_eq!(
        unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
        0
    );
    assert_eq!(
        mlx_rs::Stream::new().get_index().unwrap(),
        mlx_rs::Stream::cpu().get_index().unwrap()
    );
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

/// The actual capture load forms, each with its OWN contract: (fully optimized, literal resident).
/// Ordinary catalog defaults are deliberately not changed by this endpoint fixture.
fn selections(
    provider: &str,
    root: &Path,
    bits: Option<i32>,
) -> [(String, GenerationMemory, LoadSpec); 2] {
    [false, true].map(|resident| {
        let spec = crate::ios_pricing_tests::endpoint_spec(root, bits, resident);
        assert_eq!(
            spec.offload_policy,
            if resident {
                OffloadPolicy::Resident
            } else {
                OffloadPolicy::Sequential
            }
        );
        assert_eq!(
            spec.load_shape,
            if resident {
                LoadShape::EagerMaterialization
            } else {
                LoadShape::DeferredMaterialization
            }
        );
        let contract =
            crate::memory_strategy::weights_free_memory_strategy_contract(provider, &spec).unwrap();
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let strategy = if resident {
            MemoryStrategy::Resident
        } else {
            MemoryStrategy::ALL
                .into_iter()
                .rev()
                .find(|s| {
                    contract
                        .capability(*s)
                        .is_some_and(|c| c.support == MemoryStrategySupport::Implemented)
                })
                .unwrap()
        };
        let selection = *contract
            .selection_space(strategy, tier, false)
            .unwrap()
            .last()
            .unwrap();
        let execution = match contract.generation_memory(&selection) {
            Some(execution) => execution,
            None if resident
                && contract.resident_request_memory
                    == mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults =>
            {
                // Z-Image explicitly preserves resident LOAD defaults. Those defaults are literal
                // only because this endpoint spec is Resident+Eager (asserted independently above).
                GenerationMemory::default()
            }
            None => panic!("missing optimized endpoint execution"),
        };
        contract.validate_selection(&selection).unwrap();
        crate::pipeline::resolve_block_window_for(
            Some(execution),
            spec.load_shape == LoadShape::DeferredMaterialization,
            provider,
        )
        .unwrap();
        if resident {
            assert!(!execution.stage_residency && !execution.stream_transformer_blocks);
        } else {
            assert!(execution.stage_residency && execution.stream_transformer_blocks);
            assert!(execution.tile_vae_decode && execution.chunk_attention);
            assert_eq!(
                execution.decode_tile_edge,
                crate::memory_strategy::DECODE_TILE_EDGES
                    .iter()
                    .copied()
                    .min()
            );
            assert_eq!(
                execution.attention_chunk_size,
                Some(crate::memory_strategy::ATTENTION_CHUNK_SIZE)
            );
            assert_eq!(execution.transformer_window_size, Some(1));
            assert_eq!(
                execution.transformer_window_component,
                Some(mlx_gen::gen_core::TransformerComponent::Both)
            );
        }
        (format!("{strategy:?}"), execution, spec)
    })
}

/// The counted pricing of `route` on the fixture's tier (the CPU backend the run executes on).
fn pricing(snap: &Snapshot, spec: &LoadSpec, route: &Route) -> IosPricing {
    let facts = snap.facts_for_spec(route.mode.control(), spec);
    fold(
        &Model::new(facts)
            .unwrap()
            .quanta(route, Backend::Cpu)
            .unwrap(),
    )
}

/// One actual native-counter boundary, never a source-certified physical lower/transfer.
#[derive(Debug)]
struct NativeBoundary {
    label: String,
    synchronized: bool,
    check: Option<u32>,
    label_phase: Option<IosPhase>,
    lifecycle_phase: Option<IosPhase>,
    active_absolute: u64,
    interval_peak_absolute: u64,
    scratch: Option<u64>,
    context: Vec<u32>,
}

/// Keep EVERY repeat's real baseline, ordered checks/contexts and native intervals. Native
/// allocator counters exclude Rust/host/process demand; Rust handle presence is NOT Data ownership.
#[derive(Debug)]
struct NativeRepeat {
    baseline: mlx_gen::ios_pricing::IosMemoryBaseline,
    observed: mlx_gen::ios_pricing::IosAnchorObservation,
    boundaries: Vec<NativeBoundary>,
    native: MeasuredRun,
}

#[derive(Default)]
struct CaptureState {
    baseline: Option<mlx_gen::ios_pricing::IosMemoryBaseline>,
    observed: mlx_gen::ios_pricing::IosAnchorObservation,
    latest: BTreeMap<(String, String), u32>,
    boundaries: Vec<NativeBoundary>,
    phase: Option<IosPhase>,
    phase_peaks: BTreeMap<IosPhase, u64>,
    whole_peak: u64,
    unmapped_check: bool,
}

impl CaptureState {
    fn boundary(
        &mut self,
        label: &str,
        synchronized: bool,
        check: Option<u32>,
        scratch: Option<u64>,
    ) {
        assert!(
            self.baseline.is_some(),
            "native check preceded actual before-model-load boundary"
        );
        let active = mlx_rs::memory::get_active_memory() as u64;
        let peak = (mlx_rs::memory::get_peak_memory() as u64).max(active);
        self.whole_peak = self.whole_peak.max(peak);
        if let Some(phase) = self.phase {
            let slot = self.phase_peaks.entry(phase).or_insert(0);
            *slot = (*slot).max(peak);
        }
        self.boundaries.push(NativeBoundary {
            label: label.into(),
            synchronized,
            check,
            label_phase: ios_quantum_phase(label),
            lifecycle_phase: self.phase,
            active_absolute: active,
            interval_peak_absolute: peak,
            scratch,
            context: self.latest.values().copied().collect(),
        });
        // Only this private capture owns peak resets. It is NOT nested inside measure_phases.
        mlx_rs::memory::reset_peak_memory();
    }
}

fn capture_repeat(body: impl FnOnce() -> mlx_gen::Result<()>) -> mlx_gen::Result<NativeRepeat> {
    use mlx_gen::ios_pricing::{
        IosMemoryCoordinate, IosMemoryDomain, IosObservedCheck, IosObservedMetadata,
    };
    use mlx_gen::ios_quantum::{set_quantum_listener, QuantumEvent, QuantumMark};
    use std::sync::{Arc, Mutex};
    // Match the old capture's pre-body cache drain, never change fixture operation order.
    mlx_gen::residency::drain_allocator_cache();
    let state = Arc::new(Mutex::new(CaptureState::default()));
    let metadata = Arc::clone(&state);
    struct ClearListener;
    impl Drop for ClearListener {
        fn drop(&mut self) {
            set_quantum_listener(None);
        }
    }
    let cleanup = ClearListener;
    // This is an isolated exact CPU child. A preexisting process listener is an invalid fixture
    // composition, not evidence which may be silently replaced or synthesized from pricing.
    let replaced = set_quantum_listener(Some(Box::new(move |mark| {
        if let QuantumMark::Metadata {
            provider,
            kind,
            fields,
        } = mark
        {
            let mut state = metadata.lock().unwrap();
            if provider == "z_image_fixture" && kind == "before_model_load" {
                assert!(
                    state.baseline.is_none(),
                    "duplicate actual fixture boundary"
                );
                let bytes = mlx_rs::memory::get_active_memory() as u64;
                state.baseline = Some(mlx_gen::ios_pricing::IosMemoryBaseline {
                    domain: IosMemoryDomain::NativeAllocatorGrowth,
                    coordinate: IosMemoryCoordinate::BeforeModelLoad,
                    bytes,
                    source: "actual run_request pre-TE-load boundary / mlx_rs::memory::get_active_memory / CPU child".into(),
                });
                state.observed.domain = IosMemoryDomain::NativeAllocatorGrowth;
                state.observed.coordinate = IosMemoryCoordinate::BeforeModelLoad;
                state.observed.baseline_bytes = bytes;
                state.whole_peak = bytes;
                state.phase = Some(IosPhase::Load);
                mlx_rs::memory::reset_peak_memory();
            }
            if provider == "z_image_fixture" && kind == "phase" {
                state.boundary("actual fixture phase transition", false, None, None);
                let code = fields.iter().find(|(key, _)| *key == "phase").unwrap().1;
                state.phase = Some(match code {
                    0 => IosPhase::Load,
                    1 => IosPhase::Conditioning,
                    2 => IosPhase::Encode,
                    3 => IosPhase::Denoise,
                    4 => IosPhase::Decode,
                    _ => panic!("unknown actual fixture phase"),
                });
            }
            let index = u32::try_from(state.observed.metadata.len()).unwrap();
            state.observed.metadata.push(IosObservedMetadata {
                provider: provider.into(),
                kind: kind.into(),
                fields: fields
                    .iter()
                    .map(|(key, value)| ((*key).into(), *value))
                    .collect(),
            });
            state.latest.insert((provider.into(), kind.into()), index);
        }
    })));
    assert!(
        !replaced,
        "private native capture replaced an enclosing listener"
    );
    let checks = Arc::clone(&state);
    let result = mlx_gen::ios_quantum::with_cpu_diagnostic_observer(
        move |event| {
            let mut state = checks.lock().unwrap();
            match event {
                QuantumEvent::Checked { label, scratch } => {
                    let index = u32::try_from(state.observed.checks.len()).unwrap();
                    let phase = state.phase;
                    let context = state.latest.values().copied().collect();
                    state.unmapped_check |= ios_quantum_phase(label).is_none();
                    state.observed.checks.push(IosObservedCheck {
                        phase,
                        label: label.clone(),
                        context,
                    });
                    state.boundary(label, false, Some(index), Some(*scratch));
                }
                QuantumEvent::Synchronized { label, .. } => {
                    state.boundary(label, true, None, None);
                }
            }
        },
        body,
    );
    drop(cleanup);
    result?;
    let mut state = Arc::try_unwrap(state)
        .ok()
        .expect("native capture observer survived scope")
        .into_inner()
        .unwrap();
    state.boundary(
        "actual fixture return (counter query, not synchronization)",
        false,
        None,
        None,
    );
    let baseline = state
        .baseline
        .take()
        .expect("actual before-model-load boundary was not observed");
    // Absolute readings are retained above, including intervals below baseline. Nonnegative
    // growth normalization here is a COUNTER definition, not baseline-owner cancellation proof.
    let growth = |absolute: u64| absolute.saturating_sub(baseline.bytes);
    let native = MeasuredRun {
        demand: growth(state.whole_peak),
        phases: state
            .phase_peaks
            .iter()
            .map(|(phase, peak)| (*phase, growth(*peak)))
            .collect(),
    };
    state.observed.telemetry_complete = !state.unmapped_check
        && !state.observed.checks.is_empty()
        && state
            .observed
            .checks
            .iter()
            .all(|check| !check.context.is_empty());
    Ok(NativeRepeat {
        baseline,
        observed: state.observed,
        boundaries: state.boundaries,
        native,
    })
}

/// No independent maxima are merged across repeats. Each one is a separate physical-counter
/// observation and baseline inventory obligation for a genuine source-owned transfer callback.
struct Measured {
    repeats: Vec<NativeRepeat>,
}

fn measure(snap: &Snapshot, spec: &LoadSpec, route: &Route, hint: &mlx_gen::Image) -> Measured {
    let repeats = (0..RUNS).map(|_| {
        with_cpu_diagnostic_headroom(AMPLE, || capture_repeat(|| run_request(snap, spec, route, hint))).unwrap()
    }).enumerate().map(|(repeat, capture)| {
        println!("ACTUAL_NATIVE_REPEAT mode={:?} repeat={repeat} domain={:?} coordinate={:?} baseline={} source={} whole_growth={} phase_growth={:?}", route.mode, capture.baseline.domain, capture.baseline.coordinate, capture.baseline.bytes, capture.baseline.source, capture.native.demand, capture.native.phases);
        println!("ACTUAL_NATIVE_CONTEXT mode={:?} repeat={repeat} domain={:?} coordinate={:?} baseline={} complete={} metadata={:?} checks={:?}", route.mode, capture.observed.domain, capture.observed.coordinate, capture.observed.baseline_bytes, capture.observed.telemetry_complete, capture.observed.metadata, capture.observed.checks);
        for (point, boundary) in capture.boundaries.iter().enumerate() {
            println!("ACTUAL_NATIVE_BOUNDARY mode={:?} repeat={repeat} point={point} label={} synchronized={} check={:?} label_phase={:?} lifecycle_phase={:?} active_absolute={} interval_peak_absolute={} guard_scratch={:?} context={:?}", route.mode, boundary.label, boundary.synchronized, boundary.check, boundary.label_phase, boundary.lifecycle_phase, boundary.active_absolute, boundary.interval_peak_absolute, boundary.scratch, boundary.context);
        }
        capture
    }).collect();
    Measured { repeats }
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

/// One required derivation obligation. Incompleteness is a FAILURE of the positive obligation,
/// not an expected-refusal success, a source certificate, or permission to omit a case.
struct Verdict {
    line: String,
    failures: Vec<String>,
    proof_complete: bool,
}

/// The law's anchor is ONE actual repeat, never a maximum merged across repeats. Its counter
/// domain and baseline coordinate are the capture's own (NativeAllocatorGrowth/BeforeModelLoad).
fn law_anchor(repeat: &NativeRepeat) -> MeasuredAnchor {
    MeasuredAnchor {
        domain: repeat.baseline.domain,
        coordinate: repeat.baseline.coordinate,
        demand: repeat.native.demand,
        phases: repeat.native.phases.clone(),
        area: mlx_gen::ios_anchor_law::area(EDGE, EDGE, 1),
    }
}

/// Attach the actual anchor captures so certificate witnesses index real observed checks. The
/// anchor's pricing ledger itself is unchanged; observations are evidence, not a certificate.
fn with_observations(pricing: &IosPricing, captures: &Measured) -> IosPricing {
    let mut observed = pricing.clone();
    observed.observations = captures
        .repeats
        .iter()
        .map(|repeat| repeat.observed.clone())
        .collect();
    observed
}

/// The certified law decides. `bounds` are the provider's source-proved window certificates; a
/// missing or invalid certificate is `NotDerivable`, i.e. an INCOMPLETE positive obligation, never a
/// pass. A complete derivation must cover EVERY request repeat overall and in every phase, with no
/// tolerance and no merging of repeats: any shortfall is a strict numeric underprediction (the
/// historical 20,480-byte class of failure). Logical price uppers, equal labels, legacy derive,
/// measured spread, phase winners, or a synthetic bounder are never substituted for a certificate.
/// One anchor/request certificate comparison: the measured anchor, its priced windows, the priced
/// request, the request's measured repeats and the provider certificates.
struct VerdictInputs<'a> {
    anchor: &'a Measured,
    at_anchor: &'a IosPricing,
    request: &'a IosPricing,
    measured: &'a Measured,
    bounds: &'a [IosWindowBound],
}

fn verdict(case: &str, selection: &str, tier: &str, inputs: VerdictInputs<'_>) -> Verdict {
    let VerdictInputs {
        anchor,
        at_anchor,
        request,
        measured,
        bounds,
    } = inputs;
    let counts = |captures: &Measured| {
        captures
            .repeats
            .iter()
            .map(|repeat| {
                (
                    repeat.baseline.bytes,
                    repeat.native.demand,
                    repeat.observed.checks.len(),
                    repeat.observed.metadata.len(),
                )
            })
            .collect::<Vec<_>>()
    };
    // Actual role metadata is a compatibility check, NOT native owner/correspondence proof.
    let roles = |captures: &Measured| {
        captures
            .repeats
            .iter()
            .map(|capture| {
                capture
                    .observed
                    .metadata
                    .iter()
                    .find(|m| m.provider == "z_image_fixture" && m.kind == "before_model_load")
                    .map(|m| {
                        (
                            m.fields.get("control").copied(),
                            m.fields.get("reference").copied(),
                        )
                    })
            })
            .collect::<Vec<_>>()
    };
    let anchor_roles = roles(anchor);
    let request_roles = roles(measured);
    let id = format!("{case} {selection} {tier}");
    let mut failures = Vec::new();
    let mut complete = !anchor.repeats.is_empty() && !measured.repeats.is_empty();
    if !complete {
        failures.push(format!("{id}: missing an actual anchor or request repeat"));
    }
    if anchor_roles != request_roles || !anchor_roles.iter().all(Option::is_some) {
        complete = false;
        failures.push(format!(
            "{id}: REJECTED cross-family/missing actual-role anchor (no borrowing); anchor roles={anchor_roles:?}, request roles={request_roles:?}"
        ));
    }
    let at_anchor = with_observations(at_anchor, anchor);
    let mut derived = Vec::new();
    for (a, repeat) in anchor.repeats.iter().enumerate() {
        let derivation = match derive_certified(&law_anchor(repeat), &at_anchor, request, bounds) {
            Ok(derivation) => derivation,
            Err(error) => {
                complete = false;
                failures.push(format!(
                    "{id}: anchor repeat {a}: NOT DERIVABLE under the certified law: {error}"
                ));
                continue;
            }
        };
        derived.push(derivation.demand);
        for (r, actual) in measured.repeats.iter().enumerate() {
            if actual.baseline.domain != derivation.domain
                || actual.baseline.coordinate != derivation.coordinate
            {
                complete = false;
                failures.push(format!(
                    "{id}: request repeat {r} counter {:?}/{:?} is not the derivation's {:?}/{:?}",
                    actual.baseline.domain,
                    actual.baseline.coordinate,
                    derivation.domain,
                    derivation.coordinate,
                ));
                continue;
            }
            if derivation.demand < actual.native.demand {
                failures.push(format!(
                    "{id}: UNDERPREDICTED overall: anchor repeat {a} derived {} B < request repeat {r} measured {} B (deficit {} B)",
                    derivation.demand, actual.native.demand, actual.native.demand - derivation.demand,
                ));
            }
            for phase in &derivation.phases {
                let measured_phase = actual.native.phases.get(&phase.phase).copied().unwrap_or(0);
                if phase.bytes < measured_phase {
                    failures.push(format!(
                        "{id}: UNDERPREDICTED phase {}: anchor repeat {a} derived {} B < request repeat {r} measured {} B (deficit {} B)",
                        phase.phase.as_key(), phase.bytes, measured_phase, measured_phase - phase.bytes,
                    ));
                }
            }
        }
    }
    let status = if !complete {
        "INCOMPLETE"
    } else if failures.is_empty() {
        "COVERED"
    } else {
        "UNDERPREDICTED"
    };
    Verdict {
        line: format!(
            "| {case} | {selection} | {tier} | {:?} | {:?} | {derived:?} | {status} | per-repeat (actual baseline, native growth, checks, metadata); certified law, no merged maxima or allowance |",
            counts(anchor), counts(measured),
        ),
        failures,
        proof_complete: complete,
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

/// Derive every case in `cases` from its SAME actual source-family anchor (captured once per family
/// and tier) at selection `selection` on each tier named in `tiers`, printing the table. `tamper`
/// edits each derived request's source map (the mutation check). Returns each case's verdict, which
/// the certified law computes from the provider's window certificates.
fn prove(
    selection: usize,
    tier_names: &[&str],
    cases: &[(String, Mode, mlx_gen::Image)],
    tamper: impl Fn(
        Mode,
        &three_root_point_lower::source_correspondence::SourceMap,
        &mut three_root_point_lower::source_correspondence::SourceMap,
    ),
) -> Vec<(String, Verdict)> {
    use crate::ios_pricing_tests::original_execution_sidecar::Position;
    use three_root_point_lower::source_correspondence as source;
    let snap = snapshot();
    println!("| case | selection | tier | same-family anchor B | mode measured B | mode derived B | verdict | source coverage |");
    println!("|---|---|---|---:|---:|---:|---|---|");
    let mut out = Vec::new();
    for tier in tiers(&snap).iter().filter(|t| tier_names.contains(&t.name)) {
        // One capture per actual family/extreme, not one Text anchor borrowed by all roles or a
        // grid of anchors per hint/geometry. The original cases() population remains unchanged.
        let mut anchors: BTreeMap<source::Family, (Measured, IosPricing, source::SourceMap)> =
            BTreeMap::new();
        for (case, mode, hint) in cases {
            let provider = if mode.control() {
                crate::model_control::MODEL_ID
            } else {
                crate::model::MODEL_ID
            };
            let endpoints = selections(provider, &tier.root, tier.bits);
            let (name, execution, spec) = &endpoints[selection];
            let position = if selection == RUNG_0 {
                Position::ResidentEndpoint
            } else {
                Position::OptimizedEndpoint
            };
            let actual_route = route(*mode, *execution);
            let family = source::identity(&snap, spec, &actual_route, hint, position).unwrap();
            if !anchors.contains_key(&family) {
                let mut inventory =
                    source::prepare(&snap, spec, &actual_route, hint, position).unwrap();
                let captured = measure(&snap, spec, &actual_route, hint);
                inventory
                    .finish_actual(&snap, spec, &actual_route, hint, position, &captured)
                    .unwrap();
                let priced = pricing(&snap, spec, &actual_route);
                println!("ACTUAL_FAMILY_SOURCE_ANCHOR {}", inventory.value().unwrap());
                anchors.insert(family.clone(), (captured, priced, inventory));
            }
            let (measured_anchor, at_anchor, anchor_map) = anchors.get(&family).unwrap();
            let mut request_map =
                source::prepare(&snap, spec, &actual_route, hint, position).unwrap();
            let actual = measure(&snap, spec, &actual_route, hint);
            request_map
                .finish_actual(&snap, spec, &actual_route, hint, position, &actual)
                .unwrap();
            let request = pricing(&snap, spec, &actual_route);
            request_map.same_family(anchor_map).unwrap();
            tamper(*mode, anchor_map, &mut request_map);
            println!(
                "ACTUAL_FAMILY_SOURCE_REQUEST {}",
                request_map.value().unwrap()
            );
            // This is explicitly POINT-only U_request_source - L_anchor_source. No logical upper
            // is subtracted, no independent native maxima are differenced, and these comparisons
            // cannot complete any interval/phase/request or provide a physical certificate.
            println!(
                "ACTUAL_FAMILY_SOURCE_POINT_COMPARISONS {}",
                request_map.point_comparisons(anchor_map).unwrap()
            );
            let certificates = provider_certificates(
                &snap,
                spec,
                &actual_route,
                measured_anchor,
                at_anchor,
                &request,
            );
            let mut value = verdict(
                case,
                name,
                tier.name,
                VerdictInputs {
                    anchor: measured_anchor,
                    at_anchor,
                    request: &request,
                    measured: &actual,
                    bounds: &certificates.bounds,
                },
            );
            value
                .failures
                .extend(certificates.blocked_failures(case, name, tier.name));
            value.proof_complete &= certificates.blocked.is_empty();
            value
                .failures
                .extend(request_map.outstanding(anchor_map).unwrap());
            println!("{}", value.line);
            out.push((case.clone(), value));
        }
    }
    out
}

/// The provider's certificates for deriving `request` from this actual anchor. The anchor's real
/// observations are attached so every witness indexes an actual check. A class the provider cannot
/// source-prove is returned as BLOCKED with its exact missing premise; it is never certified from a
/// logical upper, eval/drop, RSS, or an empirical residual.
fn provider_certificates(
    snap: &Snapshot,
    spec: &LoadSpec,
    route: &Route,
    anchor: &Measured,
    at_anchor: &IosPricing,
    request: &IosPricing,
) -> crate::ios_pricing::certificates::WindowCertificates {
    let facts = snap.facts_for_spec(route.mode.control(), spec);
    let model = Model::new(facts).unwrap();
    // The EXACT original run_request sampling program: three-step static-shift curated Euler and
    // its manual init tail. Never production strength resolution, route defaults or a step bound.
    let mut sampling = crate::sampling::resolve_full(&mlx_gen::GenerationRequest {
        steps: Some(3),
        sampler: Some(mlx_gen::Solver::Euler.name().into()),
        ..Default::default()
    })
    .unwrap();
    let end = sampling.sigmas().len();
    sampling
        .window(usize::from(route.mode.img2img()), end)
        .unwrap();
    crate::ios_pricing::certificates::window_certificates(
        &model,
        route,
        &sampling,
        crate::ios_pricing::certificates::CertifiedExecution {
            // The original fixture quantizes the composed transformer BEFORE loading the VAE.
            load_order: crate::ios_pricing::certificates::HeavyLoadOrder::QuantizeBeforeVae,
            // measure() runs run_request inside with_cpu_diagnostic_headroom: the iOS scope.
            ios_quantum_scope: true,
            // in_cpu_child() pins the global default device/stream to CPU and installs no
            // task-local override; run_request executes in that child.
            cpu_stream: true,
        },
        &with_observations(at_anchor, anchor),
        request,
    )
    .unwrap()
}

fn failures(verdicts: &[(String, Verdict)]) -> Vec<String> {
    verdicts
        .iter()
        .flat_map(|(_, v)| v.failures.clone())
        .collect()
}

/// REQUIRED positive obligation: every Z-Image mode (reference; control and control + reference for
/// each kind) is covered by its certified derivation from its SAME actual source-family anchor
/// (the historical name says "text anchor"; Text borrowing is rejected), overall and in every
/// phase, at the fully optimized selection and at rung 0. Dense tier; the packed tiers are
/// [`packed_tier_matrix`].
#[cfg(target_os = "macos")]
#[test]
fn every_mode_is_derivable_from_the_text_anchor() {
    if !in_cpu_child("every_mode_is_derivable_from_the_text_anchor") {
        return;
    }
    let _completion = CpuChildCompletion;
    // The legacy inventory is not proof authority. Keep the real positive bound assertions below;
    // an Uncertified result must not be converted into a passing refusal.
    for selection in [OPTIMIZED, RUNG_0] {
        let failures = failures(&prove(selection, &["dense"], &cases(), |_, _, _| {}));
        assert!(failures.is_empty(), "not additive: {failures:#?}");
    }
}

/// [`every_mode_is_derivable_from_the_text_anchor`] on the packed tiers (the DiT stored at q4 and at
/// q8). ~15 min on the CPU fixture.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "the q4/q8 additivity matrix (~15 min of CPU): run with --ignored"]
fn packed_tier_matrix() {
    if !in_cpu_child("packed_tier_matrix") {
        return;
    }
    let _completion = CpuChildCompletion;
    for selection in [OPTIMIZED, RUNG_0] {
        let failures = failures(&prove(selection, &["q4", "q8"], &cases(), |_, _, _| {}));
        assert!(failures.is_empty(), "not additive: {failures:#?}");
    }
}

/// The proof has teeth: a derivation that forgets the control route's extra components in its
/// denoise (the `C_req` re-add: the ControlNet, its control context and hint caches — the request
/// priced at the text anchor's denoise residency) goes red at the optimized selection. Dense tier.
#[cfg(target_os = "macos")]
#[test]
fn dropping_the_control_components_breaks_the_proof() {
    if !in_cpu_child("dropping_the_control_components_breaks_the_proof") {
        return;
    }
    let _completion = CpuChildCompletion;
    let pose: Vec<_> = cases()
        .into_iter()
        .filter(|(case, _, _)| case.ends_with("pose"))
        .collect();
    let verdicts = prove(
        OPTIMIZED,
        &["dense"],
        &pose,
        |mode, _same_family_anchor, source| {
            if !mode.control() {
                return;
            }
            assert!(source.control_numeric_failures().unwrap().is_empty(),
                "baseline Control source-event producer upper must cover its required constructor event");
            source.omit_control_upper_terms().unwrap();
            let numeric = source.control_numeric_failures().unwrap();
            assert!(numeric.iter().any(|failure|failure.contains("phase denoise: Control source-event upper")),
                "the actual Control producer term, not a phase DTO or missing certificate, must become numerically invalid");
        },
    );
    assert!(
        verdicts.iter().all(|(_, verdict)| verdict.proof_complete),
        "the Control-component counterexample requires a genuine complete native transfer; incomplete source proof is not a discriminating failure"
    );
    let failures = failures(&verdicts);
    println!("{failures:#?}");
    for case in ["control pose", "control+reference pose"] {
        assert!(
            failures
                .iter()
                .any(|f| f.contains(case) && f.contains("phase denoise")),
            "{case}: dropping the control components must break the derivation"
        );
    }
}

/// The ORIGINAL legal intermediate, NOT a resident endpoint: Sequential+Deferred, explicit no
/// staging/window selection, with the production deferred fallback running full-cover TE/DiT windows.
fn deferred_full_cover_unstaged(provider: &str, tier: &Tier) -> (GenerationMemory, LoadSpec) {
    let spec = crate::ios_pricing_tests::endpoint_spec(&tier.root, tier.bits, false);
    assert_eq!(spec.offload_policy, OffloadPolicy::Sequential);
    assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
    let contract =
        crate::memory_strategy::weights_free_memory_strategy_contract(provider, &spec).unwrap();
    assert!(
        contract.lifecycle.transformer_window_materialization,
        "the intermediate must really arm deferred windows"
    );
    let numeric = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let selection = *contract
        .selection_space(MemoryStrategy::Resident, numeric, false)
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(
        contract.resident_request_memory,
        mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults
    );
    assert!(
        contract.generation_memory(&selection).is_none(),
        "this exact intermediate uses the declared no-override resident selection"
    );
    let execution = GenerationMemory::default();
    assert!(
        !execution.stage_residency
            && !execution.stream_transformer_blocks
            && !execution.tile_vae_decode
    );
    contract.validate_selection(&selection).unwrap();
    crate::pipeline::resolve_block_window_for(
        Some(execution),
        spec.load_shape == LoadShape::DeferredMaterialization,
        provider,
    )
    .unwrap();
    (execution, spec)
}

/// Preserve the unresolved original q8 Control conditioning case (the historical 20,480-byte
/// underprediction). The request stays the ORIGINAL Sequential+Deferred full-cover unstaged
/// INTERMEDIATE; it is derived from BOTH actual endpoints of its OWN Control family (optimized and
/// literal resident), never from a Text anchor. Each derivation must be complete under the certified
/// law and cover every request repeat overall and in every phase. One CPU intermediate regression,
/// not a third device anchor or a measured geometry grid.
#[test]
#[ignore = "CPU intermediate regression; run explicitly after native-test admission"]
fn q8_deferred_full_cover_unstaged_control_conditioning_regression() {
    if !in_cpu_child("q8_deferred_full_cover_unstaged_control_conditioning_regression") {
        return;
    }
    let _completion = CpuChildCompletion;
    let snap = snapshot();
    let all_tiers = tiers(&snap);
    let tier = all_tiers.iter().find(|tier| tier.name == "q8").unwrap();
    let (control_execution, control_spec) =
        deferred_full_cover_unstaged(crate::model_control::MODEL_ID, tier);
    let hint = control_image("pose", EDGE);
    let control = route(Mode::Control, control_execution);
    let actual = measure(&snap, &control_spec, &control, &hint);
    let request = pricing(&snap, &control_spec, &control);
    let endpoints = selections(crate::model_control::MODEL_ID, &tier.root, tier.bits);
    let mut failures = Vec::new();
    for (index, (name, execution, spec)) in endpoints.iter().enumerate() {
        let position = if index == RUNG_0 {
            "resident"
        } else {
            "optimized"
        };
        let anchor_route = route(Mode::Control, *execution);
        let anchor = measure(&snap, spec, &anchor_route, &hint);
        let at_anchor = pricing(&snap, spec, &anchor_route);
        let certificates = provider_certificates(
            &snap,
            &control_spec,
            &control,
            &anchor,
            &at_anchor,
            &request,
        );
        let selection =
            format!("INTERMEDIATE (not resident) from Control-family {position} endpoint {name}");
        let mut verdict = verdict(
            "q8 deferred/full-cover unstaged control",
            &selection,
            "q8",
            VerdictInputs {
                anchor: &anchor,
                at_anchor: &at_anchor,
                request: &request,
                measured: &actual,
                bounds: &certificates.bounds,
            },
        );
        verdict.failures.extend(certificates.blocked_failures(
            "q8 deferred/full-cover unstaged control",
            &selection,
            "q8",
        ));
        println!("{}", verdict.line);
        failures.extend(verdict.failures);
    }
    assert!(
        failures.is_empty(),
        "original intermediate remains underderived: {failures:#?}"
    );
}

/// Allocation diagnostics for the ORIGINAL q8 deferred/full-cover unstaged Control discrepancy.
/// This is a legal intermediate, not a literal resident endpoint. No verdict or
/// tolerance: report raw active growth and interval peaks around every load/conditioning quantum,
/// beside the header-counted residency/scratch. Repeated fresh loads distinguish a missing resident
/// term (already visible before TE_EMBED) from a transient within a layer. Runs the production
/// quantum observer but does not nest `measure_phases` (that would replace this observer).
#[test]
#[ignore = "CPU allocation diagnostic; run explicitly with --ignored --nocapture"]
fn q8_deferred_full_cover_unstaged_conditioning_allocations() {
    if !in_cpu_child("q8_deferred_full_cover_unstaged_conditioning_allocations") {
        return;
    }
    let _completion = CpuChildCompletion;
    use mlx_gen::ios_quantum::QuantumEvent;
    let snap = snapshot();
    let tiers = tiers(&snap);
    let tier = tiers.iter().find(|t| t.name == "q8").unwrap();
    let base = deferred_full_cover_unstaged(crate::model::MODEL_ID, tier);
    let control = deferred_full_cover_unstaged(crate::model_control::MODEL_ID, tier);
    for repeat in 0..5 {
        for mode in [
            Mode::Text,
            Mode::Reference,
            Mode::Control,
            Mode::ControlReference,
        ] {
            let execution = if mode.control() { control.0 } else { base.0 };
            let route = route(mode, execution);
            let spec = if mode.control() { &control.1 } else { &base.1 };
            let facts = snap.facts_for_spec(mode.control(), spec);
            let prices = Model::new(facts)
                .unwrap()
                .quanta(&route, Backend::Cpu)
                .unwrap();
            println!("INTERMEDIATE Sequential+Deferred/full-cover/unstaged repeat={repeat} mode={mode:?}");
            for q in &prices {
                if matches!(
                    ios_quantum_phase(q.label),
                    Some(IosPhase::Load | IosPhase::Conditioning)
                ) {
                    println!(
                        "PRICE {} resident={} scratch={}",
                        q.label, q.resident, q.scratch
                    );
                }
            }
            mlx_gen::residency::drain_allocator_cache();
            let baseline = mlx_rs::memory::get_active_memory() as u64;
            mlx_rs::memory::reset_peak_memory();
            let state = std::rc::Rc::new(std::cell::RefCell::new((String::new(), 0u64)));
            let sink = state.clone();
            with_cpu_diagnostic_headroom(AMPLE, || {
                mlx_gen::ios_quantum::with_cpu_diagnostic_observer(
                    move |event| {
                        let active = mlx_rs::memory::get_active_memory() as u64;
                        let peak = (mlx_rs::memory::get_peak_memory() as u64).max(active);
                        mlx_rs::memory::reset_peak_memory();
                        let (label, mark) = match event {
                            QuantumEvent::Checked { label, .. } => (label, "CHECK"),
                            QuantumEvent::Synchronized { label, .. } => (label, "SYNC"),
                        };
                        let mut previous = sink.borrow_mut();
                        if matches!(ios_quantum_phase(label), Some(IosPhase::Load | IosPhase::Conditioning)) {
                            println!(
                                "OBS {mark} {label} active={} peak={} delta_from_previous={} previous={}",
                                active.saturating_sub(baseline),
                                peak.saturating_sub(baseline),
                                i128::from(active) - i128::from(previous.1),
                                previous.0,
                            );
                        }
                        *previous = (format!("{mark} {label}"), active);
                    },
                    || run_request(&snap, spec, &route, &control_image("pose", EDGE)),
                )
            })
            .unwrap();
            drop(state);
            mlx_gen::residency::drain_allocator_cache();
            println!(
                "END active={} baseline={baseline}",
                mlx_rs::memory::get_active_memory()
            );
        }
    }
}

/// Positive allocation-event capacity for the ORIGINAL pre-TE_EMBED fixture prefix only.
/// This is not an IosBounder, a process bound, or completion of the all-window verdict above.
/// Source premises: fresh exact CPU child, ordinary raw loaders, the installed task-local CPU
/// stream, blocking bootstrap saves, and the preceding run's final explicit CPU synchronize.
/// Retained Data and late frees are allowed: no owner is subtracted. No Array is built or evaluated.
mod original_pre_embed_capacity {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use crate::ios_pricing::{Component, Facts};
    use mlx_gen::gen_core::weightsmeta::{Dtype, SafetensorsTensorHeader as Header};
    use mlx_gen::ios_pricing::{IosMemoryCoordinate, IosMemoryDomain};
    use mlx_gen::{Error, Result};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        PossibleRawLoad,
        LinearCast,
        QuantCode,
        QuantScale,
        QuantBias,
        PossibleQuantInputCopy,
        RopeTable,
        Input,
    }

    #[derive(Debug)]
    struct Event {
        epoch: &'static str,
        owner: String,
        kind: Kind,
        request_bytes: u64,
        charge_upper: u64,
    }

    #[derive(Debug)]
    struct Capacity {
        domain: IosMemoryDomain,
        coordinate: IosMemoryCoordinate,
        sources: BTreeMap<&'static str, Vec<PathBuf>>,
        constructor_origins: BTreeMap<String, String>,
        events: Vec<Event>,
        total: u64,
    }

    fn invalid(reason: impl std::fmt::Display) -> Error {
        Error::Msg(format!("original pre-TE_EMBED capacity: {reason}"))
    }

    /// Pinned MetalAllocator::malloc rounds the request, then BufferCache accepts only
    /// length < min(2*s, s+2*P). Fresh buffer logical length is s. This is NOT allocatedSize.
    fn charge_upper(request: u64, page: u64) -> Result<u64> {
        let rounded = mlx_gen::ios_quantum::fresh_metal_allocation_bytes(request, page)
            .ok_or_else(|| invalid("invalid page or rounded request overflow"))?;
        usize::try_from(rounded).map_err(invalid)?;
        if request == 0 {
            // Native malloc returns null BEFORE cache lookup or active-memory increment.
            return Ok(0);
        }
        let twice = rounded
            .checked_mul(2)
            .ok_or_else(|| invalid("cache threshold overflow"))?;
        let slack = page
            .checked_mul(2)
            .and_then(|extra| rounded.checked_add(extra))
            .ok_or_else(|| invalid("cache slack overflow"))?;
        usize::try_from(twice).map_err(invalid)?;
        usize::try_from(slack).map_err(invalid)?;
        twice
            .min(slack)
            .checked_sub(1)
            .ok_or_else(|| invalid("empty cache threshold"))
    }

    fn bytes(shape: &[usize], width: usize) -> Result<u64> {
        let mut result = u64::try_from(width).map_err(invalid)?;
        for &dim in shape {
            i32::try_from(dim).map_err(invalid)?;
            result = result
                .checked_mul(u64::try_from(dim).map_err(invalid)?)
                .ok_or_else(|| invalid("shape byte overflow"))?;
        }
        usize::try_from(result).map_err(invalid)?;
        Ok(result)
    }

    fn headers(component: &Component) -> Result<BTreeMap<String, Header>> {
        if component.files.is_empty() {
            return Err(invalid("raw source has no file inventory"));
        }
        let mut result = BTreeMap::new();
        for header in &component.headers {
            if matches!(
                header.dtype,
                Dtype::F8_E4M3 | Dtype::F8_E5M2 | Dtype::F8_E8M0
            ) {
                return Err(invalid("ordinary MLX loader has no fp8 storage descriptor"));
            }
            if bytes(&header.shape, header.dtype.size())? != header.data_bytes {
                return Err(invalid(format!(
                    "raw byte/shape mismatch for {}",
                    header.name
                )));
            }
            if result.insert(header.name.clone(), header.clone()).is_some() {
                return Err(invalid(format!("duplicate raw descriptor {}", header.name)));
            }
        }
        Ok(result)
    }

    impl Capacity {
        fn new() -> Self {
            Self {
                domain: IosMemoryDomain::NativeAllocatorGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                sources: BTreeMap::new(),
                constructor_origins: BTreeMap::new(),
                events: Vec::new(),
                total: 0,
            }
        }

        fn push(
            &mut self,
            epoch: &'static str,
            owner: String,
            kind: Kind,
            request_bytes: u64,
            page: u64,
        ) -> Result<()> {
            let charge_upper = charge_upper(request_bytes, page)?;
            let total = self
                .total
                .checked_add(charge_upper)
                .ok_or_else(|| invalid("prefix charge sum overflow"))?;
            self.events.push(Event {
                epoch,
                owner,
                kind,
                request_bytes,
                charge_upper,
            });
            self.total = total;
            Ok(())
        }

        fn raw(
            &mut self,
            epoch: &'static str,
            component: &Component,
            keep: impl Fn(&str) -> bool,
            page: u64,
        ) -> Result<BTreeMap<String, Header>> {
            let raw = headers(component)?;
            if self
                .sources
                .insert(epoch, component.files.clone())
                .is_some()
            {
                return Err(invalid("duplicate source-open epoch"));
            }
            for (name, header) in &raw {
                if keep(name) {
                    // A SUPERSET of read candidates, not an assertion that all were materialized.
                    // Copies/aliases preserve ArrayDesc ID/status; a reopened view is a NEW epoch.
                    self.push(
                        epoch,
                        name.clone(),
                        Kind::PossibleRawLoad,
                        header.data_bytes,
                        page,
                    )?;
                }
            }
            Ok(raw)
        }

        fn quant(
            &mut self,
            epoch: &'static str,
            raw: &BTreeMap<String, Header>,
            path: &str,
            has_bias: bool,
            page: u64,
        ) -> Result<()> {
            let weight = raw
                .get(&format!("{path}.weight"))
                .ok_or_else(|| invalid(format!("missing projection {path}")))?;
            if raw.contains_key(&format!("{path}.scales")) {
                mlx_gen::adapters::facts::linear_header_facts(
                    raw,
                    path,
                    mlx_gen::quant::DEFAULT_GROUP_SIZE,
                    Some(8),
                )
                .map_err(invalid)?;
                // The actual native quantizer's already-packed no-op, not a release credit.
                return Ok(());
            }
            if weight.shape.len() != 2
                || !matches!(
                    weight.dtype,
                    Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64
                )
            {
                return Err(invalid(format!("non-matrix floating projection {path}")));
            }
            let (rows, cols) = (weight.shape[0], weight.shape[1]);
            let group = usize::try_from(mlx_gen::quant::DEFAULT_GROUP_SIZE).map_err(invalid)?;
            if cols % group != 0 {
                return Err(invalid(format!("projection group mismatch {path}")));
            }
            // Each Linear casts its raw, row-contiguous source weight/bias to bf16. The native
            // AsType body returns the same descriptor at equal dtype, otherwise ONE copy graph.
            if weight.dtype != Dtype::BF16 {
                self.push(
                    epoch,
                    format!("{path}.weight"),
                    Kind::LinearCast,
                    bytes(&weight.shape, 2)?,
                    page,
                )?;
            }
            if has_bias {
                if let Some(bias) = raw.get(&format!("{path}.bias")) {
                    if bias.dtype != Dtype::BF16 {
                        self.push(
                            epoch,
                            format!("{path}.bias"),
                            Kind::LinearCast,
                            bytes(&bias.shape, 2)?,
                            page,
                        )?;
                    }
                }
            }
            // Count each ACTUAL module call, not each source key: distinct modules could read the
            // same raw descriptor and still create distinct quant graphs. Three outputs share ONE
            // primitive; core eval marks all siblings evaluated, so they do not each replay it.
            let packed_cols = cols
                .checked_mul(8)
                .and_then(|n| n.checked_div(32))
                .ok_or_else(|| invalid("packed shape overflow"))?;
            self.push(
                epoch,
                path.into(),
                Kind::QuantCode,
                bytes(&[rows, packed_cols], 4)?,
                page,
            )?;
            self.push(
                epoch,
                path.into(),
                Kind::QuantScale,
                bytes(&[rows, cols / group], 2)?,
                page,
            )?;
            self.push(
                epoch,
                path.into(),
                Kind::QuantBias,
                bytes(&[rows, cols / group], 2)?,
                page,
            )?;
            // The proven backend may make ONE row-contiguous input copy. Including its capacity
            // does not assert that the already-contiguous original fixture actually makes it.
            self.push(
                epoch,
                path.into(),
                Kind::PossibleQuantInputCopy,
                bytes(&weight.shape, 2)?,
                page,
            )
        }
    }

    /// Inputs MUST be the original fixture's raw facts, not Facts::read's converted components.
    /// `actual_vae` is read at the literal spec.weights/vae path, not canonicalized to snap.root.
    /// This emits a source capacity, never a certificate or a replacement all-window verdict.
    fn compose(raw: &Facts, actual_vae: &Component, page: u64) -> Result<Capacity> {
        if raw.load_components.is_some()
            || raw.te_quant_bits.is_some()
            || raw
                .quant_conversions
                .iter()
                .any(|entries| !entries.is_empty())
            || raw.quant_bits != Some(8)
            || !raw.streamable
        {
            return Err(invalid("requires original raw q8 deferred fixture facts"));
        }
        // Also validate page when all candidate byte counts happen to be zero.
        charge_upper(0, page)?;
        let mut out = Capacity::new();
        let te = headers(&raw.te)?;
        out.sources.insert("te-load", raw.te.files.clone());
        let embedding = format!("model.{}", crate::text_encoder::encoder::EMBEDDING_PATH);
        let mut embedding_keys = vec![format!("{embedding}.weight")];
        if te.contains_key(&format!("{embedding}.scales")) {
            embedding_keys.extend([format!("{embedding}.scales"), format!("{embedding}.biases")]);
        }
        for key in embedding_keys {
            let header = te
                .get(&key)
                .ok_or_else(|| invalid(format!("missing embedding {key}")))?;
            out.push(
                "te-load",
                key,
                Kind::PossibleRawLoad,
                header.data_bytes,
                page,
            )?;
        }
        let dit = out.raw(
            "dit-load",
            &raw.dit,
            |name| crate::loader::dit_load_keeps(true, name),
            page,
        )?;
        let origins = crate::loader::transformer_header_origins(&dit);
        let mut modules = dit.clone();
        crate::loader::remap_transformer_header_keys(&mut modules);
        for (_, target) in crate::transformer::adapter_header_targets(&raw.dit_cfg, true) {
            let module = crate::loader::TRANSFORMER_KEY_ALIASES
                .iter()
                .find_map(|(from, to)| (target == *from || target == *to).then_some(*to))
                .unwrap_or(target.as_str());
            if !modules.contains_key(&format!("{module}.scales")) {
                return Err(invalid(format!(
                    "base projection is not stored packed: {module}"
                )));
            }
            mlx_gen::adapters::facts::linear_header_facts(
                &modules,
                module,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
                Some(8),
            )
            .map_err(invalid)?;
            for suffix in [".weight", ".scales", ".biases", ".bias"] {
                if suffix == ".bias" && !crate::quant::dit_quantization_has_bias(module) {
                    // Authored ignored siblings remain raw candidates, not module operands.
                    continue;
                }
                let key = format!("{module}{suffix}");
                if let Some(source) = origins.get(&key) {
                    out.constructor_origins.insert(key, source.clone());
                }
            }
        }
        if raw.dit_cfg.axes_dims.len() != raw.dit_cfg.axes_lens.len() {
            return Err(invalid("RoPE axis inventory mismatch"));
        }
        for (index, (&dim, &len)) in raw
            .dit_cfg
            .axes_dims
            .iter()
            .zip(&raw.dit_cfg.axes_lens)
            .enumerate()
        {
            let dim = usize::try_from(dim).map_err(invalid)?;
            let len = usize::try_from(len).map_err(invalid)?;
            out.push(
                "dit-load",
                format!("rope.tables[{index}]"),
                Kind::RopeTable,
                bytes(&[len, dim / 2, 2], 4)?,
                page,
            )?;
        }
        if let Some(control) = &raw.control {
            let control = out.raw(
                "control-load",
                control,
                |name| crate::loader::control_load_keeps(true, name),
                page,
            )?;
            for (target, _) in crate::ios_pricing::control_quantization_targets() {
                if crate::loader::control_load_keeps(true, &target) {
                    out.quant(
                        "control-load",
                        &control,
                        &target,
                        crate::quant::dit_quantization_has_bias(&target),
                        page,
                    )?;
                }
            }
        }
        let vae = out.raw("vae-load", actual_vae, |_| true, page)?;
        // The ORIGINAL load_vae always executes with_encoder. Each attention's own target
        // constants drive its quantizer; no fixed 8/19/population count is a proof condition.
        for side in ["encoder", "decoder"] {
            for leaf in crate::vae::attention::QUANTIZATION_LINEAR_PATHS {
                out.quant(
                    "vae-load",
                    &vae,
                    &format!("{side}.mid_block.attentions.0.{leaf}"),
                    true,
                    page,
                )?;
            }
        }
        for input in ["input_ids", "attention_mask"] {
            out.push(
                "conditioning-inputs",
                input.into(),
                Kind::Input,
                bytes(&[usize::try_from(raw.window_tokens).map_err(invalid)?], 4)?,
                page,
            )?;
        }
        Ok(out)
    }

    // Private transfer of REAL producer terms, not a new provider/root API. The first-embed
    // inventory retains compose()'s exact Q8/deferred scope; other source operations are mapped
    // separately and do not inherit a prefix certificate. Raw Control terms are allocation-event
    // uppers, not a declaration that all source dictionary candidates remain simultaneously live.
    pub(super) fn control_source_terms(
        raw: &Component,
        embed_key: &str,
        page: u64,
    ) -> Result<super::three_root_point_lower::source_correspondence::AllocationInventory> {
        use super::three_root_point_lower::source_correspondence::{
            AllocationInventory, AllocationTerm,
        };
        let mut capacity = Capacity::new();
        let headers = capacity.raw("control-source-open", raw, |_| true, page)?;
        let key = format!("control_all_x_embedder.{embed_key}.weight");
        let embedding = headers.get(&key).ok_or_else(|| {
            invalid(format!(
                "actual Control patch constructor source absent: {key}"
            ))
        })?;
        if embedding.shape.len() != 2 || embedding.shape.contains(&0) {
            return Err(invalid(
                "actual Control patch constructor is not a positive matrix",
            ));
        }
        let control = capacity
            .events
            .into_iter()
            .map(|event| AllocationTerm {
                component: "Control",
                epoch: event.epoch.into(),
                files: raw.files.clone(),
                owner: event.owner,
                constructor: format!("{:?}", event.kind),
                request: event.request_bytes,
                charge_upper: event.charge_upper,
            })
            .collect();
        Ok(AllocationInventory {
            control,
            control_event_request: Some(embedding.data_bytes),
            prefix: Err(
                "raw Control source events alone are not a whole prefix/native-window upper".into(),
            ),
        })
    }
    pub(super) fn source_allocation_terms(
        raw: &Facts,
        actual_vae: &Component,
        page: u64,
    ) -> Result<super::three_root_point_lower::source_correspondence::AllocationInventory> {
        use super::three_root_point_lower::source_correspondence::{
            AllocationInventory, AllocationTerm,
        };
        let mut inventory = match &raw.control {
            Some(control) => control_source_terms(
                control,
                &format!("{}-{}", raw.dit_cfg.patch_size, raw.dit_cfg.f_patch_size),
                page,
            )?,
            None => AllocationInventory {
                control: Vec::new(),
                control_event_request: None,
                prefix: Err("not composed".into()),
            },
        };
        inventory.prefix = compose(raw, actual_vae, page)
            .map(|capacity| {
                capacity
                    .events
                    .into_iter()
                    .map(|event| AllocationTerm {
                        component: match event.epoch {
                            "te-load" => "TE",
                            "dit-load" => "DiT",
                            "control-load" => "Control",
                            "vae-load" => "VAE",
                            _ => "typed-inputs",
                        },
                        epoch: event.epoch.into(),
                        files: capacity
                            .sources
                            .get(event.epoch)
                            .cloned()
                            .unwrap_or_default(),
                        owner: event.owner,
                        constructor: format!("{:?}", event.kind),
                        request: event.request_bytes,
                        charge_upper: event.charge_upper,
                    })
                    .collect()
            })
            .map_err(|error| error.to_string());
        Ok(inventory)
    }

    #[test]
    fn checked_cache_charge_envelope_is_not_the_fresh_request_rounding() {
        // Pure HOST algebra with a toy page operand, never a machine-dependent golden.
        assert_eq!(charge_upper(0, 16).unwrap(), 0);
        assert_eq!(charge_upper(8, 16).unwrap(), 15);
        assert_eq!(charge_upper(16, 16).unwrap(), 31);
        assert_eq!(charge_upper(17, 16).unwrap(), 63);
        assert_eq!(charge_upper(37, 16).unwrap(), 79);
        assert!(charge_upper(1, 0).is_err());
        assert!(charge_upper(1, 3).is_err());
        assert!(charge_upper(u64::MAX, 16).is_err());
        let mut sum = Capacity::new();
        sum.total = u64::MAX;
        assert!(sum
            .push("host", "overflow".into(), Kind::Input, 1, 16)
            .is_err());
        assert!(sum.events.is_empty());
    }

    #[test]
    fn raw_header_events_preserve_extra_sources_and_literal_vae_epoch() {
        // HOST metadata only. These synthetic headers test the emitter, not native provider
        // adequacy. The real-point oracle must use actual Snapshot raw reads after CPU admission.
        fn header(name: String, dtype: Dtype, shape: Vec<usize>) -> Header {
            Header {
                data_bytes: bytes(&shape, dtype.size()).unwrap(),
                name,
                dtype,
                shape,
            }
        }
        fn component(path: &str, headers: Vec<Header>) -> Component {
            Component {
                files: vec![PathBuf::from(path)],
                headers,
            }
        }
        let cfg = crate::ios_pricing_tests::dit_cfg();
        let mut packed = BTreeMap::new();
        for (_, target) in crate::transformer::adapter_header_targets(&cfg, true) {
            for (suffix, dtype, shape) in [
                (".weight", Dtype::U32, vec![64, 16]),
                (".scales", Dtype::BF16, vec![64, 1]),
                (".biases", Dtype::BF16, vec![64, 1]),
            ] {
                let key = format!("{target}{suffix}");
                packed.insert(key.clone(), header(key, dtype, shape));
            }
        }
        let ignored_bias = format!(
            "noise_refiner.0.attention.{}.bias",
            crate::attention::ADAPTER_TARGET_PATHS[0]
        );
        packed.insert(
            ignored_bias.clone(),
            header(ignored_bias.clone(), Dtype::F32, vec![64]),
        );
        let mut vae = Vec::new();
        for side in ["encoder", "decoder"] {
            for leaf in crate::vae::attention::QUANTIZATION_LINEAR_PATHS {
                let target = format!("{side}.mid_block.attentions.0.{leaf}");
                vae.push(header(
                    format!("{target}.weight"),
                    Dtype::BF16,
                    vec![64, 64],
                ));
                vae.push(header(format!("{target}.bias"), Dtype::BF16, vec![64]));
            }
        }
        let actual_vae = component("tier/vae/model.safetensors", vae);
        let raw = Facts {
            te_cfg: crate::text_encoder::ZTextEncoderConfig::z_image(),
            dit_cfg: cfg,
            te: component(
                "snapshot/text_encoder/model.safetensors",
                vec![header(
                    "model.embed_tokens.weight".into(),
                    Dtype::BF16,
                    vec![2, 64],
                )],
            ),
            dit: component(
                "tier/transformer/model.safetensors",
                packed.into_values().collect(),
            ),
            vae: component("snapshot/vae/model.safetensors", Vec::new()),
            control: None,
            window_tokens: 32,
            caption_tokens: 20,
            quant_bits: Some(8),
            streamable: true,
            load_components: None,
            quant_conversions: std::array::from_fn(|_| Vec::new()),
            te_quant_bits: None,
        };
        let before = compose(&raw, &actual_vae, 16).unwrap();
        let extra = header("encoder.extra.weight".into(), Dtype::F32, vec![7]);
        let mut with_extra = actual_vae.clone();
        with_extra.headers.push(extra.clone());
        let after = compose(&raw, &with_extra, 16).unwrap();
        assert_eq!(
            after.total.checked_sub(before.total).unwrap(),
            charge_upper(extra.data_bytes, 16).unwrap()
        );
        assert_eq!(after.domain, IosMemoryDomain::NativeAllocatorGrowth);
        assert_eq!(after.coordinate, IosMemoryCoordinate::BeforeModelLoad);
        assert_eq!(after.sources["vae-load"], actual_vae.files);
        assert!(!after.constructor_origins.is_empty());
        assert!(!after.constructor_origins.contains_key(&ignored_bias));
        assert!(after
            .events
            .iter()
            .any(|event| event.kind == Kind::PossibleRawLoad && event.owner == ignored_bias));
        assert_eq!(
            after
                .events
                .iter()
                .try_fold(0u64, |sum, event| sum.checked_add(event.charge_upper))
                .unwrap(),
            after.total
        );
        assert!(after.events.iter().all(|event| !event.epoch.is_empty()
            && !event.owner.is_empty()
            && event.charge_upper >= event.request_bytes));
        assert!(after
            .events
            .iter()
            .any(|event| event.kind == Kind::RopeTable));
        assert!(after.events.iter().any(|event| event.kind == Kind::Input));
        let mut duplicate = with_extra;
        duplicate.headers.push(extra);
        assert!(compose(&raw, &duplicate, 16).is_err());
    }

    /// A point-only oracle over the unchanged original program. No new production hook, Array,
    /// eval, model metadata factory, request bounder, or physical-memory certificate is introduced.
    #[cfg(target_os = "macos")]
    mod actual_native_point {
        use super::*;
        use crate::ios_additivity_tests::{
            capture_repeat, control_image, deferred_full_cover_unstaged, image, route, run_request,
            snapshot, tiers, Mode, NativeRepeat, AMPLE, EDGE, RUNS,
        };
        use mlx_gen::gen_core::{LoadShape, OffloadPolicy, WeightsSource};

        const CHILD: &str = "ZIMAGE_ORIGINAL_PREFIX_CPU_CHILD";
        const DONE: &str = "ZIMAGE_ORIGINAL_PREFIX_CPU_DONE";
        const FQN: &str = "ios_additivity_tests::original_pre_embed_capacity::actual_native_point::original_q8_before_te_embed_matches_source_prefix_capacity";

        /// Declared before all fixture owners, armed only at the true final statement. The native
        /// synchronize is AFTER the original request/Scope and fixture Rust owners have returned;
        /// it does not assert last native Data release, active=0, or physical-footprint release.
        struct Completion;
        impl Drop for Completion {
            fn drop(&mut self) {
                if !std::thread::panicking() {
                    assert_eq!(unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) }, 0,
                        "original prefix CPU synchronization after fixture owners/Scope drop failed");
                    eprintln!("ZIMAGE_ORIGINAL_PREFIX_BODY_COMPLETED_V1 points=4 fixture_owners_dropped=1 cpu_sync=0");
                    std::fs::write(
                        std::env::var(DONE).expect("original prefix parent DONE"),
                        b"ok",
                    )
                    .expect("write original prefix completion");
                }
            }
        }

        fn in_strict_cpu_child() -> bool {
            if std::env::var_os(CHILD).is_none() {
                let done = tempfile::tempdir().unwrap();
                let marker = done.path().join("done");
                assert!(marker.is_absolute() && !marker.exists());
                let status = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        FQN,
                        "--include-ignored",
                        "--nocapture",
                        "--test-threads=1",
                    ])
                    .env(CHILD, "1")
                    .env(DONE, &marker)
                    .status()
                    .unwrap();
                assert!(status.success(), "original prefix CPU child failed");
                assert!(
                    marker.is_file(),
                    "original prefix child ran no complete body"
                );
                assert_eq!(
                    std::fs::read(&marker).unwrap().as_slice(),
                    b"ok",
                    "original prefix child did not write current completion"
                );
                return false;
            }
            let marker =
                PathBuf::from(std::env::var(DONE).expect("original prefix requires parent DONE"));
            assert!(
                marker.is_absolute(),
                "original prefix DONE must be absolute"
            );
            assert!(!marker.exists(), "original prefix DONE must be fresh");
            mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
            assert_eq!(
                mlx_rs::Device::try_default().unwrap(),
                mlx_rs::Device::cpu()
            );
            assert_eq!(
                unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
                0
            );
            assert_eq!(
                mlx_rs::Stream::new().get_index().unwrap(),
                mlx_rs::Stream::cpu().get_index().unwrap()
            );
            assert!(crate::test_metal_guard::forbid_metal());
            assert!(crate::test_metal_guard::metal_forbidden());
            eprintln!("ZIMAGE_ORIGINAL_PREFIX_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1");
            true
        }

        /// This new point is scoped to the ORIGINAL unambiguous bf16 VAE fixture. General alias
        /// worlds belong to the integrated VAE-history consumer, not an assumed raw/effective flag.
        /// Check the actual two-phase remapper's dependencies without constructing its Arrays:
        /// no distinct-source overwrite/read dependency, decoder-generated encoder key, or bad
        /// transpose rank is silently admitted. Self-target encoder operations execute once.
        fn original_vae_origins(component: &Component) -> Result<BTreeMap<String, String>> {
            let raw = headers(component)?;
            if raw.values().any(|header| header.dtype != Dtype::BF16) {
                return Err(invalid(
                    "original VAE point requires its actual bf16 raw descriptors",
                ));
            }
            let mut current = raw
                .iter()
                .map(|(key, header)| (key.clone(), (key.clone(), header.clone())))
                .collect::<BTreeMap<_, _>>();
            for encoder in [false, true] {
                let steps = current
                    .keys()
                    .filter_map(|key| crate::loader::vae_remap_step(key, encoder))
                    .collect::<Vec<_>>();
                let mut targets = BTreeMap::new();
                for step in &steps {
                    if !encoder && step.target.starts_with("encoder.") {
                        return Err(invalid("original VAE decoder generates an encoder key"));
                    }
                    if step.target != step.source && current.contains_key(&step.target) {
                        return Err(invalid(
                            "original VAE point has an order-dependent remap overwrite",
                        ));
                    }
                    if targets
                        .insert(step.target.clone(), step.source.clone())
                        .is_some()
                    {
                        return Err(invalid("original VAE point has colliding remap targets"));
                    }
                }
                // The checked disjoint supports make this metadata order equivalent to every
                // native unsorted snapshot order, not a captured native iteration-order claim.
                for step in steps {
                    let (origin, mut value) = current
                        .get(&step.source)
                        .cloned()
                        .ok_or_else(|| invalid("missing original VAE remap source"))?;
                    if step.transpose {
                        let s = &value.shape;
                        if s.len() != 4 {
                            return Err(invalid("original VAE transpose rank"));
                        }
                        value.shape = vec![s[0], s[2], s[3], s[1]];
                    }
                    value.name = step.target.clone();
                    current.insert(step.target, (origin, value));
                }
            }
            let mut origins = BTreeMap::new();
            for side in ["encoder", "decoder"] {
                for leaf in crate::vae::attention::QUANTIZATION_LINEAR_PATHS {
                    let raw_path = format!("{side}.mid_block.attentions.0.{leaf}");
                    let module = if side == "encoder" {
                        raw_path.clone()
                    } else {
                        format!("mid_block.attentions.0.{leaf}")
                    };
                    for suffix in ["weight", "bias"] {
                        let raw_key = format!("{raw_path}.{suffix}");
                        let target = format!("{module}.{suffix}");
                        let source = raw
                            .get(&raw_key)
                            .ok_or_else(|| invalid("missing original VAE projection"))?;
                        let (origin, selected) = current
                            .get(&target)
                            .ok_or_else(|| invalid("missing remapped original VAE projection"))?;
                        if origin != &raw_key
                            || selected.dtype != source.dtype
                            || selected.shape != source.shape
                            || selected.data_bytes != source.data_bytes
                        {
                            return Err(invalid(
                                "original VAE projection lost its raw correspondence",
                            ));
                        }
                        origins.insert(format!("vae::{target}"), origin.clone());
                    }
                }
            }
            Ok(origins)
        }

        /// Independent of the emitter's traversal: each admitted raw descriptor has exactly one
        /// event in its ACTUAL open epoch. This is source-event coverage, not native demand proof.
        fn require_raw_events(
            capacity: &Capacity,
            epoch: &str,
            component: &Component,
            keep: impl Fn(&str) -> bool,
            page: u64,
        ) -> Result<()> {
            let expected = headers(component)?;
            let actual = capacity
                .events
                .iter()
                .filter(|event| event.epoch == epoch && event.kind == Kind::PossibleRawLoad)
                .collect::<Vec<_>>();
            let wanted = expected
                .iter()
                .filter(|(key, _)| keep(key))
                .collect::<Vec<_>>();
            if actual.len() != wanted.len() {
                return Err(invalid("raw source event inventory cardinality"));
            }
            for (key, header) in wanted {
                let matches = actual
                    .iter()
                    .filter(|event| event.owner == *key)
                    .collect::<Vec<_>>();
                if matches.len() != 1
                    || matches[0].request_bytes != header.data_bytes
                    || matches[0].charge_upper != charge_upper(header.data_bytes, page)?
                {
                    return Err(invalid(format!(
                        "missing exact raw source event {epoch}/{key}"
                    )));
                }
            }
            Ok(())
        }

        // Source closure is TEST-program scoped, not a new admission rule: bootstrap random/save
        // work completes before capture; each original loader opens one raw dictionary epoch;
        // same-descriptor aliases do not reopen Loads; pending-load eval waits its reader futures;
        // the packed base does not repack; only resident Control/VAE attention quantizers can run
        // before this point; thin other constructors wrap supplied handles. Their possible casts,
        // one contig copy and three sibling outputs are emitted by compose. Three actual DiT RoPE
        // tables and the two run_request input uploads are the remaining positive writers. No
        // later embedding gather/RoPE/mask, TE layer, sampler or image producer is charged here.
        // with_cpu_diagnostic_headroom installs actual task-local CPU; each prior request reaches
        // decode readback and finish's explicit CPU synchronization. Retained/late-freed Data are
        // allowed in the next actual baseline, not subtracted or declared released. Sole-FQN child
        // source has no foreign MLX caller; neither cache-zero Scope nor this code establishes
        // runtime-wide foreign exclusion. A new producer/order/retained graph requires reclosure.
        fn actual_capacity(raw: &Facts, literal_vae: &Component, page: u64) -> Result<Capacity> {
            let origins = original_vae_origins(literal_vae)?;
            let mut capacity = compose(raw, literal_vae, page)?;
            capacity.constructor_origins.extend(origins);
            require_raw_events(
                &capacity,
                "dit-load",
                &raw.dit,
                |name| crate::loader::dit_load_keeps(true, name),
                page,
            )?;
            if let Some(control) = &raw.control {
                require_raw_events(
                    &capacity,
                    "control-load",
                    control,
                    |name| crate::loader::control_load_keeps(true, name),
                    page,
                )?;
            }
            require_raw_events(&capacity, "vae-load", literal_vae, |_| true, page)?;
            // The embedding selector accesses only this packed-or-dense tuple before the first
            // TE_EMBED. Future layer/normalization descriptors are not materialized by this loader.
            let embedding = format!("model.{}", crate::text_encoder::encoder::EMBEDDING_PATH);
            require_raw_events(
                &capacity,
                "te-load",
                &raw.te,
                |key| {
                    key == format!("{embedding}.weight")
                        || key == format!("{embedding}.scales")
                        || key == format!("{embedding}.biases")
                },
                page,
            )?;
            if capacity.sources.get("vae-load") != Some(&literal_vae.files) {
                return Err(invalid("VAE epoch is not the literal actual spec source"));
            }
            Ok(capacity)
        }

        /// Use only the actual BeforeModelLoad baseline and boundaries no later than the FIRST
        /// TE_EMBED Checked callback. That callback precedes gather, TE RoPE and mask construction.
        /// Later checks/results remain in the original capture but cannot contribute to this bound.
        fn assert_point(capture: &NativeRepeat, capacity: &Capacity, control: bool) {
            use mlx_gen::ios_pricing::{IosMemoryCoordinate, IosMemoryDomain};
            assert_eq!(
                capture.baseline.domain,
                IosMemoryDomain::NativeAllocatorGrowth
            );
            assert_eq!(
                capture.baseline.coordinate,
                IosMemoryCoordinate::BeforeModelLoad
            );
            let baseline = capture
                .observed
                .metadata
                .iter()
                .filter(|m| m.provider == "z_image_fixture" && m.kind == "before_model_load")
                .collect::<Vec<_>>();
            assert_eq!(
                baseline.len(),
                1,
                "exact actual before-model-load metadata required"
            );
            for (key, expected) in [
                ("schema", 1),
                ("width", u64::from(EDGE)),
                ("height", u64::from(EDGE)),
                ("serial_count", 1),
                ("control", u64::from(control)),
                ("reference", 0),
                ("streamable", 1),
                ("staged", 0),
                ("requested_bits", 8),
            ] {
                assert_eq!(
                    baseline[0]
                        .fields
                        .iter()
                        .find(|(name, _)| name.as_str() == key)
                        .map(|(_, value)| *value),
                    Some(expected),
                    "actual original prefix context {key}"
                );
            }
            let point = capture
                .boundaries
                .iter()
                .position(|boundary| {
                    boundary.label == crate::quantum::TE_EMBED
                        && boundary.check.is_some()
                        && !boundary.synchronized
                })
                .expect("first actual TE_EMBED Checked point missing");
            assert!(
                !capture.boundaries[..point]
                    .iter()
                    .any(|b| b.label == crate::quantum::TE_LAYER),
                "a future TE layer cannot precede this point"
            );
            let at = &capture.boundaries[point];
            let earlier_peak = capture.boundaries[..=point]
                .iter()
                .map(|b| b.interval_peak_absolute)
                .max()
                .unwrap();
            let upper = capture
                .baseline
                .bytes
                .checked_add(capacity.total)
                .expect("prefix absolute upper overflow");
            assert!(at.active_absolute <= upper,
                "ORIGINAL_PREFIX_NATIVE_ACTIVE_ABOVE_SOURCE_CAPACITY: active={} baseline={} capacity={}",
                at.active_absolute, capture.baseline.bytes, capacity.total);
            assert!(earlier_peak <= upper,
                "ORIGINAL_PREFIX_EARLIER_INTERVAL_ABOVE_SOURCE_CAPACITY: earlier_peak={} baseline={} capacity={}",
                earlier_peak, capture.baseline.bytes, capacity.total);
            eprintln!("ZIMAGE_ORIGINAL_PREFIX_POINT_V1 control={control} baseline={} active_absolute={} earlier_interval_peak_absolute={} source_event_capacity={} page_coordinate=BeforeModelLoad domain=NativeAllocatorGrowth",
                capture.baseline.bytes, at.active_absolute, earlier_peak, capacity.total);
        }

        #[test]
        fn raw_epoch_inventory_rejects_a_missing_actual_vae_source_event() {
            // The original fixture's encoder conv is really loaded before conditioning. Only its
            // header is needed here; no snapshot, Array, eval, device or counter is constructed.
            let component = Component {
                files: vec![PathBuf::from("literal-tier/vae/model.safetensors")],
                headers: vec![Header {
                    name: "encoder.conv_in.weight".into(),
                    dtype: Dtype::BF16,
                    shape: vec![32, 3, 3, 3],
                    data_bytes: bytes(&[32, 3, 3, 3], 2).unwrap(),
                }],
            };
            let mut capacity = Capacity::new();
            capacity.raw("vae-load", &component, |_| true, 16).unwrap();
            require_raw_events(&capacity, "vae-load", &component, |_| true, 16)
                .expect("actual raw VAE load event must be emitted");
            capacity.events.clear();
            assert!(require_raw_events(&capacity, "vae-load", &component, |_| true, 16).is_err());
            // Current VAE history support is not silently replaced by raw flags: a real overwrite
            // would require that consumer's joint histories, and cannot enter this original point.
            let mut alias = component.clone();
            alias.headers.push(Header {
                name: "encoder.conv_in.conv.weight".into(),
                dtype: Dtype::BF16,
                shape: vec![32, 3, 3, 3],
                data_bytes: bytes(&[32, 3, 3, 3], 2).unwrap(),
            });
            assert!(original_vae_origins(&alias).is_err());
        }

        #[test]
        #[ignore = "original q8 point-only native baseline: explicit CPU admission required"]
        fn original_q8_before_te_embed_matches_source_prefix_capacity() {
            if !in_strict_cpu_child() {
                return;
            }
            let mut completion: Option<Completion> = None;
            let mut full_mask = 0u8;
            {
                let page = mlx_gen::ios_quantum::native_metal_page_bytes()
                    .expect("actual MetalAllocator vm_page_size unavailable");
                let snap = snapshot();
                let all_tiers = tiers(&snap);
                let tier = all_tiers.iter().find(|tier| tier.name == "q8").unwrap();
                let text = deferred_full_cover_unstaged(crate::model::MODEL_ID, tier);
                let control = deferred_full_cover_unstaged(crate::model_control::MODEL_ID, tier);
                let text_route = route(Mode::Text, text.0);
                let control_route = route(Mode::Control, control.0);
                let text_hint = image(EDGE, 1);
                let control_hint = control_image("pose", EDGE);
                for (case, (route, spec, hint)) in [
                    (&text_route, &text.1, &text_hint),
                    (&control_route, &control.1, &control_hint),
                ]
                .into_iter()
                .enumerate()
                {
                    assert_eq!(spec.offload_policy, OffloadPolicy::Sequential);
                    assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
                    let raw = snap.facts_for_spec(route.mode.control(), spec);
                    let WeightsSource::Dir(root) = &spec.weights else {
                        panic!("original literal directory required")
                    };
                    let literal_vae =
                        Component::read(&WeightsSource::Dir(root.join("vae"))).unwrap();
                    let capacity = actual_capacity(&raw, &literal_vae, page).unwrap();
                    eprintln!("ZIMAGE_ORIGINAL_PREFIX_SOURCE_V1 case={case} page={page} epochs={:?} constructor_origins={:?} events={:?}",
                        capacity.sources, capacity.constructor_origins, capacity.events);
                    for repeat in 0..RUNS {
                        // EXACT original Text2 then Control2 request bodies and capture ordering.
                        // No new Scope/cache policy, Array clone, eval or mid-request synchronization.
                        let capture =
                            mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(AMPLE, || {
                                capture_repeat(|| run_request(&snap, spec, route, hint))
                            })
                            .unwrap();
                        assert_point(&capture, &capacity, route.mode.control());
                        full_mask |= 1 << (case * RUNS + repeat);
                        eprintln!(
                            "ZIMAGE_ORIGINAL_PREFIX_CASE_COMPLETED_V1 case={case} repeat={repeat}"
                        );
                    }
                }
            }
            assert_eq!(
                full_mask, 0b1111,
                "original point requires actual Text2 then Control2 completion"
            );
            let _ = completion.replace(Completion);
        }
    }
}

/// Private producer for THREE source-proved allocation events at ONE original TEST callback.
/// This is not an IosWindowBound/certificate/public provider API or an all-window/physical proof.
#[cfg(target_os = "macos")]
mod three_root_point_lower {
    use super::*;
    use crate::ios_pricing_tests::original_execution_sidecar::{self, Position};
    use mlx_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype as Stored};
    use mlx_gen::gen_core::{ControlKind, Quant, WeightsSource};
    use mlx_gen::ios_pricing::{IosMemoryBaseline, IosMemoryCoordinate, IosMemoryDomain};
    use sha2::{Digest, Sha256};

    fn unsupported(reason: impl Into<String>) -> mlx_gen::Error {
        mlx_gen::Error::Msg(format!("private three-root TEST point: {}", reason.into()))
    }
    #[derive(Clone, Debug)]
    struct DenseSource {
        path: std::path::PathBuf,
        shape: [u64; 2],
        dtype: Stored,
        data_bytes: u64,
        vocab: u64,
        hidden: u64,
        config: String,
    }
    fn read_actual_dense_source(snap: &Snapshot) -> mlx_gen::Result<DenseSource> {
        let dir = snap.root.join("text_encoder");
        let path = dir.join("model.safetensors");
        let mut files = std::fs::read_dir(&dir)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.retain(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"));
        files.sort();
        if files != [path.clone()] {
            return Err(unsupported("not the original single-file TE source epoch"));
        }
        let headers =
            safetensors_path_tensor_headers(&path).map_err(|e| unsupported(e.to_string()))?;
        if headers
            .iter()
            .any(|h| h.name == "model.embed_tokens.scales" || h.name == "model.embed_tokens.biases")
        {
            return Err(unsupported(
                "packed embedding is outside the proved dense Load constructor",
            ));
        }
        let weights = headers
            .iter()
            .filter(|h| h.name == "model.embed_tokens.weight")
            .collect::<Vec<_>>();
        if weights.len() != 1 || weights[0].shape.len() != 2 {
            return Err(unsupported("not one raw rank2 embedding"));
        }
        let h = weights[0];
        let config = std::fs::read_to_string(dir.join("config.json"))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&config).map_err(|e| unsupported(e.to_string()))?;
        let vocab = cfg["vocab_size"]
            .as_u64()
            .ok_or_else(|| unsupported("missing actual vocabulary"))?;
        let hidden = cfg["hidden_size"]
            .as_u64()
            .ok_or_else(|| unsupported("missing actual hidden width"))?;
        Ok(DenseSource {
            path,
            shape: [
                u64::try_from(h.shape[0]).map_err(|_| unsupported("vocab overflow"))?,
                u64::try_from(h.shape[1]).map_err(|_| unsupported("hidden overflow"))?,
            ],
            dtype: h.dtype,
            data_bytes: h.data_bytes,
            vocab,
            hidden,
            config,
        })
    }
    fn same_epoch(a: &DenseSource, b: &DenseSource) -> bool {
        a.path == b.path
            && a.shape == b.shape
            && a.dtype == b.dtype
            && a.data_bytes == b.data_bytes
            && a.vocab == b.vocab
            && a.hidden == b.hidden
            && a.config == b.config
    }
    fn request_bytes(raw: &DenseSource, tokens: &serde_json::Value) -> mlx_gen::Result<[u64; 3]> {
        // These are the actual source constructor's arguments, NOT arbitrary resident-capacity keys.
        // The admitted source proof is BF16 Load + TWO owning typed i32 copies, not dtype aliases.
        if raw.dtype != Stored::BF16
            || raw.shape != [raw.vocab, raw.hidden]
            || raw.vocab == 0
            || raw.hidden == 0
        {
            return Err(unsupported(
                "raw dtype/config/embedding shape outside dense BF16 proof",
            ));
        }
        let elements = raw
            .vocab
            .checked_mul(raw.hidden)
            .ok_or_else(|| unsupported("embedding element overflow"))?;
        let stored = elements
            .checked_mul(2)
            .ok_or_else(|| unsupported("raw BF16 extent overflow"))?;
        if raw.data_bytes != stored {
            return Err(unsupported(
                "raw header offsets disagree with actual BF16 extent",
            ));
        }
        let embedding_element_width = 2u64; // Actual PRE-CAST BF16 Load item width; numeric mutation site.
        let embedding = elements
            .checked_mul(embedding_element_width)
            .ok_or_else(|| unsupported("embedding request overflow"))?;
        let vocab = i32::try_from(raw.vocab)
            .map_err(|_| unsupported("original factory vocabulary exceeds i32"))?;
        let expected = original_execution_sidecar::original_token_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/ios_pricing_tests.rs"
            )),
            vocab,
        )?;
        if tokens != &expected
            || tokens["window"] != 32
            || tokens["valid_tokens"] != 20
            || tokens["shape"] != serde_json::json!([1, 32])
            || tokens["layout"] != "row-major contiguous"
            || tokens["input_ids"]["dtype"] != "i32"
            || tokens["attention_mask"]["dtype"] != "i32"
        {
            return Err(unsupported(
                "not the exact already-proved original typed token factory/layout",
            ));
        }
        let ids: Vec<i32> = serde_json::from_value(tokens["input_ids"]["values"].clone())
            .map_err(|e| unsupported(e.to_string()))?;
        let mask: Vec<i32> = serde_json::from_value(tokens["attention_mask"]["values"].clone())
            .map_err(|e| unsupported(e.to_string()))?;
        let bytes = |n: usize| {
            u64::try_from(n)
                .ok()
                .and_then(|n| n.checked_mul(std::mem::size_of::<i32>() as u64))
                .ok_or_else(|| unsupported("typed input request overflow"))
        };
        Ok([embedding, bytes(ids.len())?, bytes(mask.len())?])
    }
    fn fresh_request(request: u64, page: u64) -> mlx_gen::Result<u64> {
        mlx_gen::ios_quantum::fresh_metal_allocation_bytes(request, page)
            .ok_or_else(|| unsupported("invalid actual page or native fresh rounding overflow"))
    }
    #[derive(Debug)]
    struct Arithmetic {
        requests: [u64; 3],
        fresh: [u64; 3],
        sum: u64,
        growth_lower: u64,
    }
    fn arithmetic(requests: [u64; 3], page: u64, baseline: u64) -> mlx_gen::Result<Arithmetic> {
        let mut fresh = [0; 3];
        for (i, request) in requests.into_iter().enumerate() {
            fresh[i] = fresh_request(request, page)?;
        }
        let sum = fresh
            .iter()
            .try_fold(0u64, |n, r| n.checked_add(*r))
            .ok_or_else(|| unsupported("three-root sum overflow"))?;
        let growth_lower = sum.saturating_sub(baseline); // Actual anchor baseline, NEVER defaulted to zero.
        Ok(Arithmetic {
            requests,
            fresh,
            sum,
            growth_lower,
        })
    }
    fn typed_baseline(b: &IosMemoryBaseline) -> mlx_gen::Result<()> {
        if b.domain != IosMemoryDomain::NativeAllocatorGrowth
            || b.coordinate != IosMemoryCoordinate::BeforeModelLoad
        {
            return Err(unsupported(
                "not NativeAllocatorGrowth at actual BeforeModelLoad",
            ));
        }
        Ok(())
    }
    fn caller_position(spec: &LoadSpec, route: &Route) -> mlx_gen::Result<Position> {
        if spec.offload_policy != OffloadPolicy::Sequential
            || spec.load_shape != LoadShape::DeferredMaterialization
            || spec.quantize != Some(Quant::Q8)
            || !spec.adapters.is_empty()
            || spec.control.is_some()
            || spec.ip_adapter.is_some()
            || spec.pid.is_some()
            || !spec.extra_controls.is_empty()
            || spec.text_encoder.is_some()
            || !spec.components.is_empty()
            || route.width != EDGE
            || route.height != EDGE
            || route.count != 1
            || !matches!(route.mode, Mode::Text | Mode::Control)
        {
            return Err(unsupported("not an original q8 TEST caller/role/load form"));
        }
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(unsupported("not a literal original tier directory"));
        };
        let expected_spec = crate::ios_pricing_tests::endpoint_spec(root, Some(8), false);
        if spec.precision != expected_spec.precision {
            return Err(unsupported("not original fixture precision"));
        }
        if route.execution == GenerationMemory::default() {
            return Ok(Position::Intermediate);
        }
        if !route.mode.control() {
            return Err(unsupported(
                "optimized eligibility is this same Control family only",
            ));
        }
        let contract = crate::memory_strategy::weights_free_memory_strategy_contract(
            crate::model_control::MODEL_ID,
            spec,
        )?;
        let strategy = MemoryStrategy::ALL
            .into_iter()
            .rev()
            .find(|s| {
                contract
                    .capability(*s)
                    .is_some_and(|c| c.support == MemoryStrategySupport::Implemented)
            })
            .ok_or_else(|| unsupported("no actual optimized strategy"))?;
        let choices = contract
            .selection_space(
                strategy,
                MemoryNumericTier {
                    precision: spec.precision,
                    quant: spec.quantize,
                    component_precision_floors: &[],
                },
                false,
            )
            .map_err(|e| unsupported(e.to_string()))?;
        let selection = choices
            .last()
            .ok_or_else(|| unsupported("empty optimized selection"))?;
        let actual = contract
            .generation_memory(selection)
            .ok_or_else(|| unsupported("optimized execution missing"))?;
        if route.execution != actual {
            return Err(unsupported(
                "not the actual selected Control optimized execution",
            ));
        }
        Ok(Position::OptimizedEndpoint)
    }
    fn point_index(capture: &NativeRepeat, route: &Route) -> mlx_gen::Result<usize> {
        typed_baseline(&capture.baseline)?;
        if capture.observed.domain != capture.baseline.domain
            || capture.observed.coordinate != capture.baseline.coordinate
            || capture.observed.baseline_bytes != capture.baseline.bytes
        {
            return Err(unsupported("capture baseline coordinate mismatch"));
        }
        let baselines = capture
            .observed
            .metadata
            .iter()
            .enumerate()
            .filter(|(_, m)| m.provider == "z_image_fixture" && m.kind == "before_model_load")
            .collect::<Vec<_>>();
        if baselines.len() != 1 {
            return Err(unsupported("not one actual TEST source-epoch boundary"));
        }
        let (baseline_index, metadata) = baselines[0];
        for (key, expected) in [
            ("schema", 1),
            ("width", u64::from(EDGE)),
            ("height", u64::from(EDGE)),
            ("serial_count", 1),
            ("control", u64::from(route.mode.control())),
            ("reference", 0),
            ("streamable", 1),
            ("staged", u64::from(route.execution.stage_residency)),
            ("requested_bits", 8),
            ("te_rust_handle_present", 0),
            ("dit_rust_handle_present", 0),
            ("vae_rust_handle_present", 0),
        ] {
            if metadata.fields.get(key).copied() != Some(expected) {
                return Err(unsupported(format!("actual source epoch field {key}")));
            }
        }
        let i = capture
            .boundaries
            .iter()
            .position(|b| {
                b.label == crate::quantum::TE_EMBED && b.check.is_some() && !b.synchronized
            })
            .ok_or_else(|| unsupported("first actual TE_EMBED Checked is absent"))?;
        if capture.boundaries[..i]
            .iter()
            .any(|b| b.label == crate::quantum::TE_LAYER)
            || !capture.boundaries[..i]
                .iter()
                .any(|b| b.label == crate::quantum::TE_LOAD && b.check.is_some())
        {
            return Err(unsupported(
                "TE Load/first-embed chronology differs from proved caller",
            ));
        }
        let point = &capture.boundaries[i];
        let check = point
            .check
            .and_then(|c| usize::try_from(c).ok())
            .and_then(|c| capture.observed.checks.get(c))
            .ok_or_else(|| unsupported("actual point check witness missing"))?;
        if check.label != crate::quantum::TE_EMBED
            || check.phase != Some(IosPhase::Conditioning)
            || point.lifecycle_phase != Some(IosPhase::Conditioning)
            || !check.context.contains(
                &u32::try_from(baseline_index).map_err(|_| unsupported("epoch index overflow"))?,
            )
        {
            return Err(unsupported(
                "not the actual first Conditioning TE_EMBED context",
            ));
        }
        let phase = check
            .context
            .iter()
            .filter_map(|index| {
                usize::try_from(*index)
                    .ok()
                    .and_then(|n| capture.observed.metadata.get(n))
            })
            .find(|m| m.provider == "z_image_fixture" && m.kind == "phase")
            .ok_or_else(|| unsupported("actual phase context absent"))?;
        let heavy = u64::from(!route.execution.stage_residency);
        for (key, expected) in [
            ("phase", 1),
            ("te_rust_handle_present", 1),
            ("dit_rust_handle_present", heavy),
            ("vae_rust_handle_present", heavy),
        ] {
            if phase.fields.get(key).copied() != Some(expected) {
                return Err(unsupported(format!("point owner-phase field {key}")));
            }
        }
        // Handle metadata is corroborating CALLER context, not Data identity: distinct/co-resident
        // charge follows the three actual Load/owning-copy constructors and held borrows in CODE.
        Ok(i)
    }
    #[derive(Debug)]
    struct PointCheckpoint {
        arithmetic: Arithmetic,
        page: u64,
        baseline: u64,
        active: u64,
        actual_growth: u64,
        point: usize,
        binding: String,
    }
    /// The ONLY producer of PointCheckpoint owns the exact capture closure. No arbitrary body,
    /// capacity dictionary, certificate flag, runtime Array handle or generic public API is accepted.
    fn capture_actual_point(
        snap: &Snapshot,
        spec: &LoadSpec,
        route: &Route,
        hint: &mlx_gen::Image,
    ) -> mlx_gen::Result<PointCheckpoint> {
        let position = caller_position(spec, route)?;
        let raw = read_actual_dense_source(snap)?;
        let projection =
            snap.original_execution_sidecar(spec, route, hint, ControlKind::Pose, position)?;
        if !projection.program["actual_load_form"]["TE_load_quant"].is_null()
            || projection.program["geometry"]
                != serde_json::json!({"width":EDGE,"height":EDGE,"count":1})
            || projection.program["windows"]["actual_TE_layers"] != 3
        {
            return Err(unsupported(
                "not the already-proved dense unquantized original TE caller",
            ));
        }
        let requests = request_bytes(&raw, &projection.program["tokens"])?;
        let page = mlx_gen::ios_quantum::native_metal_page_bytes()
            .ok_or_else(|| unsupported("actual native page unavailable"))?;
        // EXACT existing collector/body/Scopes/evals/cache chronology; no callback or owner edits.
        let capture = with_cpu_diagnostic_headroom(AMPLE, || {
            capture_repeat(|| run_request(snap, spec, route, hint))
        })?;
        let point = point_index(&capture, route)?;
        let after = read_actual_dense_source(snap)?;
        if !same_epoch(&raw, &after) {
            return Err(unsupported(
                "raw dense source epoch changed around actual caller",
            ));
        }
        let baseline = capture.baseline.bytes;
        let arithmetic = arithmetic(requests, page, baseline)?;
        let active = capture.boundaries[point].active_absolute;
        let actual_growth = active.saturating_sub(baseline);
        let mut binding = Sha256::new();
        binding.update(b"private-actual-run-request-first-TE-embed-three-root-point-v1\0");
        binding.update(
            serde_json::to_vec(&projection.program).map_err(|e| unsupported(e.to_string()))?,
        );
        binding.update(raw.path.to_string_lossy().as_bytes());
        for value in [
            raw.vocab,
            raw.hidden,
            raw.data_bytes,
            page,
            baseline,
            u64::try_from(point).map_err(|_| unsupported("point overflow"))?,
        ] {
            binding.update(value.to_le_bytes());
        }
        Ok(PointCheckpoint {
            arithmetic,
            page,
            baseline,
            active,
            actual_growth,
            point,
            binding: format!("{:x}", binding.finalize()),
        })
    }

    fn host_metadata() -> (DenseSource, serde_json::Value) {
        let raw = DenseSource {
            path: "HOST-metadata-only/model.safetensors".into(),
            shape: [151_936, 64],
            dtype: Stored::BF16,
            data_bytes: 151_936 * 64 * 2,
            vocab: 151_936,
            hidden: 64,
            config: "HOST arithmetic only".into(),
        };
        let tokens = original_execution_sidecar::original_token_program(
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/ios_pricing_tests.rs"
            )),
            151_936,
        )
        .unwrap();
        (raw, tokens)
    }
    #[test]
    fn source_three_root_lower_arithmetic_is_checked_and_baseline_relative() {
        let (raw, tokens) = host_metadata();
        let requests = request_bytes(&raw, &tokens).unwrap();
        // Independent raw-byte and typed source counts, NOT another call to the producer formula.
        let expected = raw.data_bytes + 32 * 4 + 32 * 4;
        assert_eq!(
            requests,
            [raw.data_bytes, 128, 128],
            "THREE_ROOT_ACTUAL_PRECAST_WIDTH_AND_TYPED_COUNTS"
        );
        for (baseline, lower) in [
            (0, expected),
            (17, expected - 17),
            (expected - 1, 1),
            (expected, 0),
            (expected + 1, 0),
            (u64::MAX, 0),
        ] {
            let result = arithmetic(requests, 16_384, baseline).unwrap();
            assert_eq!(
                result.fresh, requests,
                "THREE_ROOT_NATIVE_FRESH_PER_ALLOCATION"
            );
            assert_eq!(result.sum, expected, "THREE_ROOT_DISTINCT_LIVE_REQUEST_SUM");
            assert_eq!(
                result.growth_lower, lower,
                "THREE_ROOT_POSITIVE_BASELINE_SUBTRACTION_NOT_ZERO"
            );
        }
        for (request, page, expected) in [
            (0, 16_384, 0),
            (1, 16_384, 1),
            (16_384, 16_384, 16_384),
            (16_385, 16_384, 32_768),
            (4097, 4096, 8192),
        ] {
            assert_eq!(
                fresh_request(request, page).unwrap(),
                expected,
                "THREE_ROOT_NATIVE_ROUNDING_BOUNDARY"
            );
        }
        assert!(
            fresh_request(u64::MAX, 16_384).is_err(),
            "THREE_ROOT_ROUNDING_OVERFLOW_ERROR"
        );
        assert!(fresh_request(1, 0).is_err(), "THREE_ROOT_ZERO_PAGE_ERROR");
        assert!(
            fresh_request(4, 3).is_err(),
            "THREE_ROOT_NON_NATIVE_PAGE_ERROR"
        );
        assert!(
            arithmetic([u64::MAX, 1, 1], 1, 0).is_err(),
            "THREE_ROOT_SUM_OVERFLOW_ERROR"
        );
        let mut overflow = raw.clone();
        overflow.shape = [u64::MAX, 2];
        overflow.vocab = u64::MAX;
        overflow.hidden = 2;
        assert!(
            request_bytes(&overflow, &tokens).is_err(),
            "THREE_ROOT_HEADER_PRODUCT_OVERFLOW_ERROR"
        );
    }
    #[test]
    fn unsupported_three_root_context_is_error_not_window_evidence() {
        let (raw, tokens) = host_metadata();
        let mut wrong = raw.clone();
        wrong.dtype = Stored::F32;
        assert!(
            request_bytes(&wrong, &tokens).is_err(),
            "THREE_ROOT_UNPROVED_DTYPE_NOT_EVIDENCE"
        );
        let mut wrong = raw.clone();
        wrong.shape[1] += 1;
        assert!(
            request_bytes(&wrong, &tokens).is_err(),
            "THREE_ROOT_CONFIG_SHAPE_MISMATCH"
        );
        let mut wrong = raw.clone();
        wrong.data_bytes += 2;
        assert!(
            request_bytes(&wrong, &tokens).is_err(),
            "THREE_ROOT_RAW_OFFSET_MISMATCH"
        );
        for window in [0, 31, 33] {
            let mut wrong = tokens.clone();
            wrong["window"] = serde_json::json!(window);
            assert!(
                request_bytes(&raw, &wrong).is_err(),
                "THREE_ROOT_UNPROVED_WINDOW_NOT_EVIDENCE"
            );
        }
        let mut wrong = tokens.clone();
        wrong["attention_mask"]["dtype"] = serde_json::json!("bool");
        assert!(
            request_bytes(&raw, &wrong).is_err(),
            "THREE_ROOT_MASK_NOT_BITPACKED"
        );
        let mut wrong = tokens.clone();
        wrong["layout"] = serde_json::json!("view");
        assert!(
            request_bytes(&raw, &wrong).is_err(),
            "THREE_ROOT_UNPROVED_VIEW_NOT_EVIDENCE"
        );
        let mut spec =
            crate::ios_pricing_tests::endpoint_spec(Path::new("HOST-shape-only"), Some(8), false);
        assert!(
            caller_position(&spec, &route(Mode::Reference, GenerationMemory::default())).is_err(),
            "THREE_ROOT_OTHER_ROLE_NOT_EVIDENCE"
        );
        spec.load_shape = LoadShape::EagerMaterialization;
        assert!(
            caller_position(&spec, &route(Mode::Text, GenerationMemory::default())).is_err(),
            "THREE_ROOT_OTHER_LOAD_EPOCH_NOT_EVIDENCE"
        );
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec.quantize = Some(Quant::Q4);
        assert!(
            caller_position(&spec, &route(Mode::Text, GenerationMemory::default())).is_err(),
            "THREE_ROOT_OTHER_TIER_NOT_EVIDENCE"
        );
        let baseline = IosMemoryBaseline {
            domain: IosMemoryDomain::ProcessGrowth,
            coordinate: IosMemoryCoordinate::BeforeModelLoad,
            bytes: 17,
            source: "HOST domain negative".into(),
        };
        assert!(
            typed_baseline(&baseline).is_err(),
            "THREE_ROOT_PROCESS_DOMAIN_NOT_EVIDENCE"
        );
        let baseline = IosMemoryBaseline {
            domain: IosMemoryDomain::NativeAllocatorGrowth,
            coordinate: IosMemoryCoordinate::WindowEntry,
            bytes: 17,
            source: "HOST coordinate negative".into(),
        };
        assert!(
            typed_baseline(&baseline).is_err(),
            "THREE_ROOT_OTHER_WINDOW_NOT_EVIDENCE"
        );
    }

    const LOWER_CHILD: &str = "ZIMAGE_THREE_ROOT_LOWER_CPU_CHILD";
    const LOWER_DONE: &str = "ZIMAGE_THREE_ROOT_LOWER_CPU_DONE";
    const LOWER_FQN: &str="ios_additivity_tests::three_root_point_lower::actual_three_root_lower_at_original_and_control_optimized_points";
    struct LowerCompletion {
        mask: u8,
    }
    impl Drop for LowerCompletion {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                assert_eq!(self.mask, 63, "THREE_ROOT_ALL_SIX_POINT_CASES");
                assert_eq!(
                    unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
                    0,
                    "THREE_ROOT_FINAL_CPU_SYNC"
                );
                eprintln!("ZIMAGE_THREE_ROOT_LOWER_BODY_COMPLETED_V1 points=6 mask=63 fixture_owners_dropped=1 cpu_sync=0");
                std::fs::write(
                    std::env::var(LOWER_DONE).expect("three-root parent DONE"),
                    b"ok",
                )
                .unwrap();
            }
        }
    }
    fn strict_three_root_cpu_child() -> bool {
        if std::env::var_os(LOWER_CHILD).is_none() {
            let dir = tempfile::tempdir().unwrap();
            let marker = dir.path().join("done");
            assert!(marker.is_absolute() && !marker.exists());
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--ignored",
                    "--exact",
                    LOWER_FQN,
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(LOWER_CHILD, "1")
                .env(LOWER_DONE, &marker)
                .status()
                .unwrap();
            assert!(status.success(), "THREE_ROOT_NATIVE_CHILD_STATUS");
            assert!(marker.is_file(), "THREE_ROOT_NATIVE_BODY_MISSING");
            assert_eq!(
                std::fs::read(marker).unwrap().as_slice(),
                b"ok",
                "THREE_ROOT_FRESH_DONE_CONTENT"
            );
            return false;
        }
        let marker = std::path::PathBuf::from(
            std::env::var(LOWER_DONE).expect("THREE_ROOT_PARENT_DONE_REQUIRED"),
        );
        assert!(marker.is_absolute(), "THREE_ROOT_DONE_ABSOLUTE");
        assert!(!marker.exists(), "THREE_ROOT_DONE_FRESH");
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert_eq!(
            mlx_rs::Device::try_default().unwrap(),
            mlx_rs::Device::cpu()
        );
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        eprintln!("ZIMAGE_THREE_ROOT_LOWER_CPU_GUARDS_READY_V1 device=cpu native_stream=cpu fatal_metal_trap=1");
        true
    }
    #[test]
    #[ignore = "six actual CPU source-point captures; explicit finite admission required"]
    fn actual_three_root_lower_at_original_and_control_optimized_points() {
        let mut completion: Option<LowerCompletion> = None;
        if !strict_three_root_cpu_child() {
            return;
        }
        let mut mask = 0u8;
        {
            let snap = snapshot();
            let all_tiers = tiers(&snap);
            let tier = all_tiers.iter().find(|t| t.name == "q8").unwrap();
            let text = deferred_full_cover_unstaged(crate::model::MODEL_ID, tier);
            let control = deferred_full_cover_unstaged(crate::model_control::MODEL_ID, tier);
            let optimized = selections(crate::model_control::MODEL_ID, &tier.root, tier.bits)
                .into_iter()
                .next()
                .unwrap();
            let text_route = route(Mode::Text, text.0);
            let control_route = route(Mode::Control, control.0);
            let optimized_route = route(Mode::Control, optimized.1);
            let text_hint = image(EDGE, 1);
            let control_hint = control_image("pose", EDGE);
            for (case, (route, spec, hint)) in [
                (&text_route, &text.1, &text_hint),
                (&control_route, &control.1, &control_hint),
                (&optimized_route, &optimized.2, &control_hint),
            ]
            .into_iter()
            .enumerate()
            {
                for repeat in 0..RUNS {
                    let point = capture_actual_point(&snap, spec, route, hint).unwrap();
                    assert!(
                        point.arithmetic.growth_lower <= point.actual_growth,
                        "THREE_ROOT_LOWER_ABOVE_ACTUAL_POINT_GROWTH: {point:?}"
                    );
                    assert!(
                        point.arithmetic.sum <= point.active,
                        "THREE_ROOT_DISTINCT_CHARGES_ABOVE_ACTUAL_POINT_ACTIVE: {point:?}"
                    );
                    eprintln!("ZIMAGE_THREE_ROOT_LOWER_POINT_V1 case={case} repeat={repeat} page={} baseline={} requests={:?} rounded={:?} sum={} lower={} actual_growth={} active={} point={} source_point_binding={}",
                        point.page,point.baseline,point.arithmetic.requests,point.arithmetic.fresh,point.arithmetic.sum,
                        point.arithmetic.growth_lower,point.actual_growth,point.active,point.point,point.binding);
                    mask |= 1 << (case * RUNS + repeat);
                }
            }
        }
        let _ = completion.replace(LowerCompletion { mask });
    }

    /// Private original-caller mechanism. A source inventory/correspondence candidate is not an
    /// IosWindowBound. In particular, a first-embed point floor never certifies its preceding gap,
    /// a caption cast, a whole phase, a whole request, or ProcessGrowth.
    pub(in crate::ios_additivity_tests) mod source_correspondence {
        use super::*;
        use crate::ios_pricing::Component;
        use crate::ios_pricing_tests::original_execution_sidecar::{Position, Projection};
        use mlx_gen::ios_pricing::{IosAnchorObservation, IosWindowPrice};
        use std::collections::BTreeMap;
        use std::path::PathBuf;

        fn invalid(message: impl std::fmt::Display) -> mlx_gen::Error {
            mlx_gen::Error::Msg(format!("private source correspondence: {message}"))
        }
        fn digest(value: &serde_json::Value) -> mlx_gen::Result<String> {
            let mut hash = Sha256::new();
            hash.update(b"original-caller-family-source-op-v1\0");
            hash.update(serde_json::to_vec(value).map_err(invalid)?);
            Ok(format!("{:x}", hash.finalize()))
        }
        fn execution_binding(projection: &Projection) -> mlx_gen::Result<String> {
            use mlx_gen::gen_core::Conditioning;
            let mut carriers = Vec::new();
            for carrier in &projection.request.conditioning {
                let (image, kind, scale, strength) = match carrier {
                    Conditioning::Control { image, kind, scale } => (
                        image,
                        format!("Control/{kind:?}"),
                        scale.map(f32::to_bits),
                        None,
                    ),
                    Conditioning::Reference { image, strength } => {
                        (image, "Reference".into(), None, strength.map(f32::to_bits))
                    }
                    _ => return Err(invalid("unknown actual original-caller carrier")),
                };
                let mut pixels = Sha256::new();
                pixels.update(&image.pixels);
                carriers.push(serde_json::json!({"kind":kind,"width":image.width,"height":image.height,
                    "pixels_sha256":format!("{:x}",pixels.finalize()),"scale_bits":scale,"strength_bits":strength}));
            }
            digest(
                &serde_json::json!({"actual_program":projection.program,"typed_carriers":carriers,
                "width":projection.request.width,"height":projection.request.height,"count":projection.request.count,
                "seed":projection.request.seed,"steps":projection.request.steps,"sampler":projection.request.sampler,
                "actual_memory":format!("{:?}",projection.request.memory)}),
            )
        }
        fn role(mode: Mode) -> &'static str {
            match mode {
                Mode::Text => "text",
                Mode::Reference => "reference",
                Mode::Control => "control",
                Mode::ControlReference => "control-reference",
            }
        }

        /// Private exact-CPU-child family, NOT an EvidenceKey or device calibration family.
        /// Geometry, endpoint load form and window choice deliberately do not manufacture new
        /// families. Actual constructor graph, numeric tier, roles and allocation class do.
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub(in crate::ios_additivity_tests) struct Family {
            session: u32,
            architecture: String,
            tier: String,
            role: String,
            sampling: String,
            configured_graph: String,
        }
        impl Family {
            fn from_projection(
                projection: &Projection,
                spec: &LoadSpec,
                mode: Mode,
            ) -> mlx_gen::Result<Self> {
                let program = &projection.program;
                let mut counts = [0u32; 3];
                for item in &projection.request.conditioning {
                    let index = match item {
                        mlx_gen::gen_core::Conditioning::Reference { .. } => 0,
                        mlx_gen::gen_core::Conditioning::Mask { .. } => 1,
                        mlx_gen::gen_core::Conditioning::Control { .. } => 2,
                        _ => return Err(invalid("unsupported actual conditioning carrier")),
                    };
                    counts[index] = counts[index]
                        .checked_add(1)
                        .ok_or_else(|| invalid("role count overflow"))?;
                }
                crate::ios_pricing::validate_pricing_roles(
                    &projection.provider,
                    projection.request.memory_reference_count(),
                    counts[0],
                    counts[1],
                    counts[2],
                )?;
                if counts[0] != u32::from(mode.img2img()) || counts[2] != u32::from(mode.control())
                {
                    return Err(invalid("actual role graph disagrees with mode"));
                }
                let sampling = crate::sampling::resolve(&projection.request)?.allocation_class();
                let start = program["manual_flow"]["start"]
                    .as_u64()
                    .ok_or_else(|| invalid("actual manual start absent"))?;
                let configurations = program["actual_constructor_configurations"]
                    .as_array()
                    .filter(|v| v.len() == 4)
                    .ok_or_else(|| invalid("actual constructor configurations absent"))?;
                Ok(Self {
                    session: std::process::id(),
                    architecture: digest(&serde_json::json!(configurations))?,
                    tier: format!("{:?}/{:?}", spec.precision, spec.quantize),
                    role: role(mode).into(),
                    // The manual start is an actual caller fact, not an invented DTO strength.
                    sampling: format!(
                        "{sampling}/manual-start={start}/shift={}",
                        program["manual_flow"]["shift_bits"]
                    ),
                    configured_graph: digest(&serde_json::json!({"provider":projection.provider,
                        "control":mode.control(),"reference":mode.img2img(),
                        "te_layers":program["windows"]["actual_TE_layers"],
                        "dit_blocks":program["windows"]["actual_base_blocks"],
                        "control_blocks":program["windows"]["actual_control_blocks"]}))?,
                })
            }
            fn value(&self) -> serde_json::Value {
                serde_json::json!({"private_cpu_session":self.session,"architecture":self.architecture,
                    "tier":self.tier,"role":self.role,"allocation":self.sampling,"graph":self.configured_graph})
            }
        }

        /// One positive allocation-event term from the existing native allocator envelope.
        /// Its sum is an event budget, never a claim that every candidate is simultaneously live.
        #[derive(Clone, Debug)]
        pub(in crate::ios_additivity_tests) struct AllocationTerm {
            pub(in crate::ios_additivity_tests) component: &'static str,
            pub(in crate::ios_additivity_tests) epoch: String,
            pub(in crate::ios_additivity_tests) files: Vec<PathBuf>,
            pub(in crate::ios_additivity_tests) owner: String,
            pub(in crate::ios_additivity_tests) constructor: String,
            pub(in crate::ios_additivity_tests) request: u64,
            pub(in crate::ios_additivity_tests) charge_upper: u64,
        }
        impl AllocationTerm {
            fn value(&self) -> serde_json::Value {
                serde_json::json!({"component":self.component,"source_candidate_open_epoch":self.epoch,
                    "literal_files":self.files,"raw_owner":self.owner,"constructor":self.constructor,
                    "request_bytes":self.request,"charged_allocation_upper":self.charge_upper})
            }
        }
        pub(in crate::ios_additivity_tests) struct AllocationInventory {
            pub(in crate::ios_additivity_tests) control: Vec<AllocationTerm>,
            pub(in crate::ios_additivity_tests) control_event_request: Option<u64>,
            pub(in crate::ios_additivity_tests) prefix: Result<Vec<AllocationTerm>, String>,
        }
        fn total(terms: &[AllocationTerm]) -> mlx_gen::Result<u64> {
            terms.iter().try_fold(0u64, |n, t| {
                n.checked_add(t.charge_upper)
                    .ok_or_else(|| invalid("source allocation-event upper overflow"))
            })
        }

        #[derive(Clone, Debug, PartialEq, Eq)]
        enum InputBorrowScope {
            DuringEncodeTokens,
            AfterInputTemporaries,
            OutsideEncodeTokens,
        }
        #[derive(Clone, Debug)]
        struct SourceHistory {
            source_op: usize,
            quant_first_use: bool,
            epoch: Option<&'static str>,
            decode_late: Option<bool>,
        }
        fn source_history(window: &IosWindowPrice) -> mlx_gen::Result<SourceHistory> {
            let class = window
                .class
                .as_deref()
                .ok_or_else(|| invalid("missing source class"))?;
            let source_op = class
                .split("/source-op-")
                .nth(1)
                .and_then(|tail| tail.split('/').next())
                .and_then(|number| number.parse::<usize>().ok())
                .ok_or_else(|| invalid("missing actual source-op index"))?;
            let request = class.contains("/request-first");
            let recurrent = class.contains("/denoise-recurrent");
            if request && recurrent {
                return Err(invalid(
                    "source class mixes request-first and recurrent history",
                ));
            }
            let late = class.contains("/decode-late-true");
            let early = class.contains("/decode-late-false");
            if late && early {
                return Err(invalid("source class mixes late-decode histories"));
            }
            Ok(SourceHistory {
                source_op,
                quant_first_use: class.contains("/source-first-use"),
                epoch: if request {
                    Some("request-first")
                } else if recurrent {
                    Some("denoise-recurrent")
                } else if class.contains("/no-model-forward") {
                    Some("no-model-forward")
                } else {
                    None
                },
                decode_late: if late {
                    Some(true)
                } else if early {
                    Some(false)
                } else {
                    None
                },
            })
        }
        fn input_borrow_scope(
            window: &IosWindowPrice,
            history: &SourceHistory,
        ) -> mlx_gen::Result<InputBorrowScope> {
            if window.phase != IosPhase::Conditioning {
                return Ok(InputBorrowScope::OutsideEncodeTokens);
            }
            if window.label == crate::quantum::CAPTION_FEATURES {
                return match history.source_op {
                    0 => Ok(InputBorrowScope::DuringEncodeTokens),
                    1 => Ok(InputBorrowScope::AfterInputTemporaries),
                    _ => Err(invalid("unknown original caption source operation")),
                };
            }
            Ok(InputBorrowScope::DuringEncodeTokens)
        }
        #[derive(Clone, Debug)]
        struct Operation {
            source_ordinal: usize,
            window: IosWindowPrice,
            history: SourceHistory,
            input_borrow_scope: InputBorrowScope,
            possible_owners: Vec<&'static str>,
            native_interval_obligations: Vec<&'static str>,
            // A real producer-owned ADDITIVE TERM, not phase.resident_bytes and not a complete
            // native upper. Removing it alters the emitter consumed by both diagnostics/proofs.
            control_allocation_upper: Vec<AllocationTerm>,
        }
        fn owners(phase: IosPhase, control: bool, reference: bool) -> Vec<&'static str> {
            let mut out = vec![
                "actual-baseline-roots-uninventoried",
                "late-native-frees-unproved",
            ];
            match phase {
                IosPhase::Load => out.extend([
                    "raw-source-loads",
                    "constructor-casts",
                    "quantize-sibling-outputs",
                    "load-workspace",
                ]),
                IosPhase::Conditioning => out.extend([
                    "TE-embedding",
                    "typed-ids-mask-during-encode_tokens",
                    "RoPE-mask-hidden",
                    "caption-valid-gather",
                    "caption-cast-after-input-temporaries",
                ]),
                IosPhase::Encode => out.extend([
                    "VAE-encoder",
                    "host-image-upload",
                    "encoder-carries-and-tile-assembly",
                    "caption",
                    "latent-context",
                ]),
                IosPhase::Denoise => out.extend([
                    "DiT-raw-and-effective-weights",
                    "caption",
                    "latents",
                    "solver-initial-and-recurrent-state",
                    "sampler-callback-output",
                    "serial-output-carries",
                ]),
                IosPhase::Decode => out.extend([
                    "VAE-decoder",
                    "latent",
                    "decode-carries-and-tile-assembly",
                    "RGB-conversion",
                    "serial-completed-outputs",
                ]),
            }
            if control {
                out.extend([
                    "Control-raw-and-effective-weights",
                    "Control-context-hint-caches",
                ]);
            }
            if reference {
                out.extend(["reference-init", "reference-encode-result"]);
            }
            out
        }
        fn operations(
            windows: &[IosWindowPrice],
            allocation: &AllocationInventory,
            control: bool,
            reference: bool,
        ) -> mlx_gen::Result<Vec<Operation>> {
            let mut out = Vec::new();
            for (index, window) in windows.iter().enumerate() {
                window
                    .class
                    .as_deref()
                    .filter(|c| !c.is_empty())
                    .ok_or_else(|| {
                        invalid(format!(
                            "unclassified actual source operation {}",
                            window.label
                        ))
                    })?;
                let history = source_history(window)?;
                let input_borrow_scope = input_borrow_scope(window, &history)?;
                let mut control_upper = if control && window.phase == IosPhase::Denoise {
                    allocation.control.clone()
                } else {
                    Vec::new()
                };
                for term in &mut control_upper {
                    // Separate candidate opens for separate source ops; no same-name/root alias
                    // cancellation across a streamed reopen, callback or late-quant history.
                    term.epoch = format!(
                        "{}/class={}/occurrence={index}",
                        term.epoch,
                        window.class.as_deref().unwrap()
                    );
                }
                out.push(Operation {
                    source_ordinal: index,
                    window: window.clone(),
                    history,
                    input_borrow_scope,
                    possible_owners: owners(window.phase, control, reference),
                    native_interval_obligations: vec![
                        "per-source-op allocation/workspace upper",
                        "live-Data/co-residency-or-no-credit",
                        "baseline inventory",
                        "callback/quantum-gap correspondence",
                    ],
                    control_allocation_upper: control_upper,
                });
            }
            Ok(out)
        }

        #[derive(Clone, Debug)]
        struct PointFloor {
            repeat: usize,
            boundary: usize,
            actual_baseline: u64,
            actual_active: u64,
            root_sum: u64,
            growth_lower: u64,
        }
        #[derive(Clone, Debug)]
        struct TraceInterval {
            repeat: usize,
            end_boundary: usize,
            label: String,
            synchronized: bool,
            check: Option<u32>,
            lifecycle_phase: Option<IosPhase>,
            label_phase: Option<IosPhase>,
            baseline: u64,
            end_active: u64,
            interval_peak: u64,
            guard_scratch: Option<u64>,
            metadata: Vec<u32>,
            candidates: Vec<usize>,
        }
        fn intervals(
            operations: &[Operation],
            measured: &Measured,
        ) -> mlx_gen::Result<Vec<TraceInterval>> {
            let mut out = Vec::new();
            for (repeat, capture) in measured.repeats.iter().enumerate() {
                typed_baseline(&capture.baseline)?;
                if capture.observed.domain != capture.baseline.domain
                    || capture.observed.coordinate != capture.baseline.coordinate
                    || capture.observed.baseline_bytes != capture.baseline.bytes
                {
                    return Err(invalid("actual baseline/observation coordinate mismatch"));
                }
                let starts = capture
                    .observed
                    .metadata
                    .iter()
                    .filter(|m| m.provider == "z_image_fixture" && m.kind == "before_model_load")
                    .collect::<Vec<_>>();
                if starts.len() != 1 {
                    return Err(invalid("expected ONE actual baseline inventory mark"));
                }
                for name in [
                    "te_rust_handle_present",
                    "dit_rust_handle_present",
                    "vae_rust_handle_present",
                ] {
                    if starts[0].fields.get(name).copied() != Some(0) {
                        return Err(invalid("original pre-load handle chronology changed"));
                    }
                }
                // Keep EVERY interval, including pre-first-check, phase transitions, synchronization
                // and final return. Labels produce candidates only; they NEVER certify mapping.
                for (index, boundary) in capture.boundaries.iter().enumerate() {
                    for metadata in &boundary.context {
                        if capture.observed.metadata.get(*metadata as usize).is_none() {
                            return Err(invalid("out-of-range actual context"));
                        }
                    }
                    if let Some(index) = boundary.check {
                        let check = capture
                            .observed
                            .checks
                            .get(index as usize)
                            .ok_or_else(|| invalid("actual check index out of range"))?;
                        if check.label != boundary.label
                            || check.phase != boundary.lifecycle_phase
                            || check.context != boundary.context
                        {
                            return Err(invalid("actual boundary/check/context mismatch"));
                        }
                    }
                    if boundary.interval_peak_absolute < boundary.active_absolute {
                        return Err(invalid("invalid actual native interval peak"));
                    }
                    let candidates = operations
                        .iter()
                        .enumerate()
                        .filter(|(_, op)| {
                            op.window.label == boundary.label
                                && boundary
                                    .lifecycle_phase
                                    .is_none_or(|p| p == op.window.phase)
                                && boundary.label_phase.is_none_or(|p| p == op.window.phase)
                        })
                        .map(|(index, _)| index)
                        .collect();
                    out.push(TraceInterval {
                        repeat,
                        end_boundary: index,
                        label: boundary.label.clone(),
                        synchronized: boundary.synchronized,
                        check: boundary.check,
                        lifecycle_phase: boundary.lifecycle_phase,
                        label_phase: boundary.label_phase,
                        baseline: capture.baseline.bytes,
                        end_active: boundary.active_absolute,
                        interval_peak: boundary.interval_peak_absolute,
                        guard_scratch: boundary.scratch,
                        metadata: boundary.context.clone(),
                        candidates,
                    });
                }
            }
            Ok(out)
        }

        struct CaptureInventory {
            baseline: IosMemoryBaseline,
            observed: IosAnchorObservation,
            whole_upper: u64,
            phase_uppers: BTreeMap<IosPhase, u64>,
        }
        type FileEpoch = (PathBuf, u64, u64, u64, i64, i64, u64, i64, i64);
        fn source_epochs(
            files: impl IntoIterator<Item = PathBuf>,
        ) -> mlx_gen::Result<Vec<FileEpoch>> {
            use std::os::unix::fs::MetadataExt;
            let mut paths = files.into_iter().collect::<Vec<_>>();
            paths.sort();
            paths.dedup();
            paths
                .into_iter()
                .map(|path| {
                    let target = std::fs::metadata(&path)?;
                    let literal = std::fs::symlink_metadata(&path)?;
                    Ok((
                        path,
                        target.dev(),
                        target.ino(),
                        target.len(),
                        target.mtime(),
                        target.mtime_nsec(),
                        literal.ino(),
                        literal.mtime(),
                        literal.mtime_nsec(),
                    ))
                })
                .collect()
        }
        pub(in crate::ios_additivity_tests) struct SourceMap {
            family: Family,
            literal_program: serde_json::Value,
            program_binding: String,
            work_binding: String,
            operations: Vec<Operation>,
            allocation: AllocationInventory,
            intervals: Vec<TraceInterval>,
            captures: Vec<CaptureInventory>,
            point_floors: Vec<PointFloor>,
            lower_scope_reason: Option<String>,
            actual_page: u64,
            source_files: BTreeMap<String, Vec<PathBuf>>,
            constructor_origins: BTreeMap<String, String>,
            source_epochs: Vec<FileEpoch>,
            dense_before: Option<DenseSource>,
        }
        fn raw_graph(components: &[(&str, Component)]) -> mlx_gen::Result<String> {
            let mut graph = Vec::new();
            for (component, raw) in components {
                let mut headers = BTreeMap::new();
                for header in &raw.headers {
                    let expected = header
                        .shape
                        .iter()
                        .try_fold(header.dtype.size() as u64, |n, &dim| {
                            n.checked_mul(dim as u64)
                        })
                        .ok_or_else(|| invalid("raw architecture header overflow"))?;
                    if expected!=header.data_bytes || headers.insert(header.name.clone(),serde_json::json!({
                        "dtype":format!("{:?}",header.dtype),"shape":header.shape,"data_bytes":header.data_bytes})).is_some() {
                        return Err(invalid("raw constructor graph has inconsistent/duplicate header"));
                    }
                }
                graph.push(
                    serde_json::json!({"component":component,"raw_header_namespace":headers}),
                );
            }
            digest(
                &serde_json::json!({"scope":"conservative private raw constructor graph, no literal-path folding","components":graph}),
            )
        }
        fn family_with_sources(
            mut family: Family,
            raw: &crate::ios_pricing::Facts,
        ) -> mlx_gen::Result<Family> {
            let mut components = vec![
                ("TE", raw.te.clone()),
                ("DiT", raw.dit.clone()),
                ("VAE", raw.vae.clone()),
            ];
            if let Some(control) = &raw.control {
                components.push(("Control", control.clone()));
            }
            let actual = raw_graph(&components)?;
            family.architecture = digest(
                &serde_json::json!({"declared":family.architecture,"actual_headers":actual}),
            )?;
            family.configured_graph = digest(
                &serde_json::json!({"declared":family.configured_graph,"actual_headers":actual}),
            )?;
            Ok(family)
        }
        pub(in crate::ios_additivity_tests) fn identity(
            snap: &Snapshot,
            spec: &LoadSpec,
            route: &Route,
            hint: &mlx_gen::Image,
            position: Position,
        ) -> mlx_gen::Result<Family> {
            let projection =
                snap.original_execution_sidecar(spec, route, hint, ControlKind::Pose, position)?;
            let WeightsSource::Dir(root) = &spec.weights else {
                return Err(invalid("actual source root absent"));
            };
            let mut raw = snap.facts_for_spec(route.mode.control(), spec);
            raw.vae = Component::read(&WeightsSource::Dir(root.join("vae")))?;
            family_with_sources(
                Family::from_projection(&projection, spec, route.mode)?,
                &raw,
            )
        }
        pub(in crate::ios_additivity_tests) fn prepare(
            snap: &Snapshot,
            spec: &LoadSpec,
            route: &Route,
            hint: &mlx_gen::Image,
            position: Position,
        ) -> mlx_gen::Result<SourceMap> {
            let projection =
                snap.original_execution_sidecar(spec, route, hint, ControlKind::Pose, position)?;
            let family = Family::from_projection(&projection, spec, route.mode)?;
            let WeightsSource::Dir(root) = &spec.weights else {
                return Err(invalid("original source directory absent"));
            };
            let actual_vae = Component::read(&WeightsSource::Dir(root.join("vae")))?;
            let mut raw = snap.facts_for_spec(route.mode.control(), spec);
            raw.vae = actual_vae.clone(); // literal LOAD source, never canonicalized to snap.root.
            let family = family_with_sources(family, &raw)?;
            let page = mlx_gen::ios_quantum::native_metal_page_bytes()
                .ok_or_else(|| invalid("actual native page unavailable"))?;
            let allocation = super::super::original_pre_embed_capacity::source_allocation_terms(
                &raw,
                &actual_vae,
                page,
            )?;
            let source_files: BTreeMap<String, Vec<PathBuf>> = [
                ("TE", raw.te.files.clone()),
                ("DiT", raw.dit.files.clone()),
                ("VAE", actual_vae.files.clone()),
                (
                    "Control",
                    raw.control
                        .as_ref()
                        .map(|c| c.files.clone())
                        .unwrap_or_default(),
                ),
            ]
            .into_iter()
            .map(|(name, files)| (name.to_string(), files))
            .collect();
            let model = Model::new(raw)?;
            let constructor_origins = model.dit_constructor_origins().clone();
            let priced = model.quanta(route, Backend::Cpu)?;
            let windows = priced.iter().map(|p| p.window.clone()).collect::<Vec<_>>();
            let operations = operations(
                &windows,
                &allocation,
                route.mode.control(),
                route.mode.img2img(),
            )?;
            let (dense_before, lower_scope_reason) = match caller_position(spec, route) {
                Ok(_) => (Some(read_actual_dense_source(snap)?), None),
                Err(error) => (None, Some(error.to_string())),
            };
            let mut epoch_paths = source_files.values().flatten().cloned().collect::<Vec<_>>();
            epoch_paths.push(snap.root.join("text_encoder/config.json"));
            let epochs = source_epochs(epoch_paths)?;
            Ok(SourceMap {
                family,
                program_binding: digest(&projection.program)?,
                work_binding: execution_binding(&projection)?,
                literal_program: projection.program,
                operations,
                allocation,
                intervals: Vec::new(),
                captures: Vec::new(),
                point_floors: Vec::new(),
                lower_scope_reason,
                actual_page: page,
                source_files,
                constructor_origins,
                source_epochs: epochs,
                dense_before,
            })
        }
        impl SourceMap {
            pub(in crate::ios_additivity_tests) fn finish_actual(
                &mut self,
                snap: &Snapshot,
                spec: &LoadSpec,
                route: &Route,
                hint: &mlx_gen::Image,
                position: Position,
                measured: &Measured,
            ) -> mlx_gen::Result<()> {
                if !self.captures.is_empty() {
                    return Err(invalid("actual observation map already bound"));
                }
                let projection = snap.original_execution_sidecar(
                    spec,
                    route,
                    hint,
                    ControlKind::Pose,
                    position,
                )?;
                if digest(&projection.program)? != self.program_binding
                    || execution_binding(&projection)? != self.work_binding
                {
                    return Err(invalid(
                        "actual literal program/typed work changed across capture",
                    ));
                }
                let paths = self
                    .source_epochs
                    .iter()
                    .map(|epoch| epoch.0.clone())
                    .collect::<Vec<_>>();
                if source_epochs(paths)? != self.source_epochs {
                    return Err(invalid(
                        "literal source identity epoch changed across capture",
                    ));
                }
                if mlx_gen::ios_quantum::native_metal_page_bytes() != Some(self.actual_page) {
                    return Err(invalid("native page changed across capture"));
                }
                self.intervals = intervals(&self.operations, measured)?;
                if let Some(before) = &self.dense_before {
                    let after = read_actual_dense_source(snap)?;
                    if !same_epoch(before, &after) {
                        return Err(invalid(
                            "raw dense header/config epoch changed across capture",
                        ));
                    }
                    let requests = request_bytes(before, &projection.program["tokens"])?;
                    for (repeat, capture) in measured.repeats.iter().enumerate() {
                        let boundary = point_index(capture, route)?;
                        let value = arithmetic(requests, self.actual_page, capture.baseline.bytes)?;
                        let active = capture.boundaries[boundary].active_absolute;
                        if value.sum > active
                            || value.growth_lower > active.saturating_sub(capture.baseline.bytes)
                        {
                            return Err(invalid(
                                "source first-point lower exceeds actual native observation",
                            ));
                        }
                        self.point_floors.push(PointFloor {
                            repeat,
                            boundary,
                            actual_baseline: capture.baseline.bytes,
                            actual_active: active,
                            root_sum: value.sum,
                            growth_lower: value.growth_lower,
                        });
                    }
                }
                self.captures = measured
                    .repeats
                    .iter()
                    .map(|capture| CaptureInventory {
                        baseline: capture.baseline.clone(),
                        observed: capture.observed.clone(),
                        whole_upper: capture.native.demand,
                        phase_uppers: capture.native.phases.clone(),
                    })
                    .collect();
                Ok(())
            }
            /// Actual numeric producer mutation seam. No phase DTO, source eligibility, label,
            /// certificate flag, oracle, baseline or ownership claim is changed by this operation.
            pub(in crate::ios_additivity_tests) fn omit_control_upper_terms(
                &mut self,
            ) -> mlx_gen::Result<()> {
                let mut before = 0u64;
                for op in &mut self.operations {
                    before = before
                        .checked_add(total(&op.control_allocation_upper)?)
                        .ok_or_else(|| invalid("Control upper sum overflow"))?;
                    op.control_allocation_upper.clear();
                }
                if let Ok(prefix) = &mut self.allocation.prefix {
                    prefix.retain(|term| term.component != "Control");
                }
                if before == 0 {
                    return Err(invalid(
                        "negative requires actual nonzero Control producer terms",
                    ));
                }
                Ok(())
            }
            pub(in crate::ios_additivity_tests) fn control_numeric_failures(
                &self,
            ) -> mlx_gen::Result<Vec<String>> {
                let Some(request) = self.allocation.control_event_request else {
                    return Ok(Vec::new());
                };
                // This is the required patch-embedder Load EVENT's fresh charge, not a whole-request
                // physical/native lower. A bound on its source-event budget cannot be below it.
                let floor = fresh_request(request, self.actual_page)?;
                let mut failures = Vec::new();
                for op in self
                    .operations
                    .iter()
                    .filter(|op| op.window.phase == IosPhase::Denoise)
                {
                    let upper = total(&op.control_allocation_upper)?;
                    if upper < floor {
                        failures.push(format!("phase denoise: Control source-event upper {upper} < required constructor-event floor {floor} at {}",op.window.class.as_deref().unwrap_or("unclassified")));
                    }
                }
                Ok(failures)
            }
            pub(in crate::ios_additivity_tests) fn same_family(
                &self,
                other: &Self,
            ) -> mlx_gen::Result<()> {
                if self.family != other.family {
                    return Err(invalid(
                        "actual family mismatch; no Text/reference/Control borrowing",
                    ));
                }
                Ok(())
            }
            pub(in crate::ios_additivity_tests) fn point_comparisons(
                &self,
                anchor: &Self,
            ) -> mlx_gen::Result<serde_json::Value> {
                self.same_family(anchor)?;
                let Ok(prefix) = &self.allocation.prefix else {
                    return Ok(
                        serde_json::json!({"scope":"point only","missing":"source prefix capacity is outside its scope"}),
                    );
                };
                let request_upper = total(prefix)?;
                let mut comparisons = Vec::new();
                let mut maxima = BTreeMap::<usize, u64>::new();
                for request in &self.point_floors {
                    for candidate in &anchor.point_floors {
                        let whole = anchor
                            .captures
                            .get(candidate.repeat)
                            .ok_or_else(|| invalid("anchor repeat absent"))?
                            .whole_upper;
                        let derived = point_upper(whole, request_upper, candidate.growth_lower)?;
                        let actual = request
                            .actual_active
                            .saturating_sub(request.actual_baseline);
                        if actual > derived {
                            return Err(invalid(
                                "source point transfer underbounds the actual first-embed point",
                            ));
                        }
                        let maximum = maxima.entry(request.repeat).or_insert(derived);
                        *maximum = (*maximum).max(derived);
                        comparisons.push(serde_json::json!({"request_repeat":request.repeat,"anchor_repeat":candidate.repeat,
                            "anchor_whole_upper":whole,"request_source_prefix_upper":request_upper,
                            "anchor_source_point_lower":candidate.growth_lower,
                            "signed_delta":(i128::from(request_upper)-i128::from(candidate.growth_lower)).to_string(),
                            "derived_point_upper":derived,"actual_request_point_growth":actual,
                            "scope":"FIRST_TE_EMBED_INSTANT_ONLY_NOT_ANY_WINDOW"}));
                    }
                }
                Ok(
                    serde_json::json!({"scope":"point only, not full phase or request","comparisons":comparisons,
                    "max_over_ALL_plausible_anchor_repeats":maxima}),
                )
            }
            pub(in crate::ios_additivity_tests) fn outstanding(
                &self,
                anchor: &Self,
            ) -> mlx_gen::Result<Vec<String>> {
                self.same_family(anchor)?;
                let mut issues = self.control_numeric_failures()?;
                for op in &self.operations {
                    issues.push(format!(
                        "{} {:?}: unproved {}",
                        op.window.class.as_deref().unwrap_or("unclassified"),
                        op.window.phase,
                        op.native_interval_obligations.join(", ")
                    ));
                }
                for interval in &self.intervals {
                    issues.push(format!("repeat{} interval baseline/previous→boundary{} ({}, {:?}, check{:?}): {} source candidates, NOT source-proved gap coverage",
                        interval.repeat,interval.end_boundary,interval.label,interval.lifecycle_phase,interval.check,interval.candidates.len()));
                }
                // A source point floor is deliberately not copied onto a window or interval.
                issues.push(format!("{} source point floors do not cover {} operation inventories or {} quantum intervals",
                    self.point_floors.len(),self.operations.len(),self.intervals.len()));
                Ok(issues)
            }
            pub(in crate::ios_additivity_tests) fn value(
                &self,
            ) -> mlx_gen::Result<serde_json::Value> {
                let ops=self.operations.iter().map(|op|->mlx_gen::Result<serde_json::Value>{Ok(serde_json::json!({"ordinal":op.source_ordinal,
                    "source_class":op.window.class,"label":op.window.label,"phase":format!("{:?}",op.window.phase),
                    "source_history":{"source_op":op.history.source_op,"quant_first_use":op.history.quant_first_use,"epoch":op.history.epoch,"decode_late":op.history.decode_late},
                    "Rust_input_borrow_scope_NOT_Data_proof":format!("{:?}",op.input_borrow_scope),
                    "logical_only":{"resident":op.window.resident_bytes,"held":op.window.held_bytes,"scratch":op.window.scratch_bytes},
                    "source_work_basis_NOT_native_proof":format!("{:?}",op.window.basis),"logical_owner_domain_NOT_native_proof":format!("{:?}",op.window.owner_domain),
                    "possible_owners_NOT_Data_proof":op.possible_owners,"native_interval_obligations":op.native_interval_obligations,
                    "Control_allocation_event_upper_term":total(&op.control_allocation_upper)?,
                    "Control_source_terms":op.control_allocation_upper.iter().map(AllocationTerm::value).collect::<Vec<_>>()}))}).collect::<mlx_gen::Result<Vec<_>>>()?;
                let traces=self.intervals.iter().map(|t|serde_json::json!({"repeat":t.repeat,"end_boundary":t.end_boundary,
                    "label":t.label,"synchronized":t.synchronized,"check":t.check,"lifecycle_phase":format!("{:?}",t.lifecycle_phase),"label_phase":format!("{:?}",t.label_phase),
                    "actual_B":t.baseline,"actual_active":t.end_active,"actual_interval_peak":t.interval_peak,"guard_scratch_source_only":t.guard_scratch,
                    "metadata_indices":t.metadata,"candidate_source_ops_NOT_correspondence_proof":t.candidates})).collect::<Vec<_>>();
                let floors=self.point_floors.iter().map(|p|serde_json::json!({"repeat":p.repeat,"boundary":p.boundary,
                    "actual_B":p.actual_baseline,"actual_active":p.actual_active,"three_root_sum":p.root_sum,
                    "growth_lower":p.growth_lower,"scope":"FIRST_TE_EMBED_INSTANT_ONLY"})).collect::<Vec<_>>();
                let captures=self.captures.iter().map(|c|serde_json::json!({"domain":format!("{:?}",c.baseline.domain),
                    "coordinate":format!("{:?}",c.baseline.coordinate),"actual_B":c.baseline.bytes,"baseline_query_source":c.baseline.source,
                    "whole_run_native_upper_NOT_point_active":c.whole_upper,"phase_uppers_NOT_independent_subtrahends":format!("{:?}",c.phase_uppers),
                    "telemetry_complete":c.observed.telemetry_complete,
                    "actual_metadata":c.observed.metadata.iter().map(|m|serde_json::json!({"provider":m.provider,"kind":m.kind,"fields":m.fields})).collect::<Vec<_>>(),
                    "actual_checks":c.observed.checks.iter().map(|check|serde_json::json!({"label":check.label,"phase":format!("{:?}",check.phase),"context":check.context})).collect::<Vec<_>>()
                })).collect::<Vec<_>>();
                Ok(
                    serde_json::json!({"scope":"private actual source inventory, not certified all-window bound",
                    "family":self.family.value(),"literal_program_binding":self.program_binding,"literal_typed_work_binding":self.work_binding,"actual_program":self.literal_program,
                    "literal_source_open_inventories":self.source_files,"source_epochs_before_after_checked":self.source_epochs,"DiT_constructor_raw_origins":self.constructor_origins,
                    "actual_page":self.actual_page,"operations":ops,"actual_intervals":traces,"actual_baselines_captures":captures,"first_point_floors":floors,
                    "point_lower_ineligibility":self.lower_scope_reason,
                    "prefix_capacity":match &self.allocation.prefix {Ok(v)=>serde_json::json!({"allocation_event_upper":total(v)?,"terms":v.iter().map(AllocationTerm::value).collect::<Vec<_>>(),"scope":"pre-first-TE_EMBED source prefix ONLY"}),Err(e)=>serde_json::json!({"unavailable":e})}}),
                )
            }
        }

        fn point_upper(
            anchor_whole: u64,
            request_source_upper: u64,
            anchor_source_lower: u64,
        ) -> mlx_gen::Result<u64> {
            if anchor_source_lower > anchor_whole {
                return Err(invalid("source lower exceeds measured whole anchor upper"));
            }
            let delta = i128::from(request_source_upper)
                .checked_sub(i128::from(anchor_source_lower))
                .ok_or_else(|| invalid("point delta overflow"))?;
            u64::try_from(
                i128::from(anchor_whole)
                    .checked_add(delta)
                    .ok_or_else(|| invalid("point transfer overflow"))?,
            )
            .map_err(invalid)
        }
        #[test]
        fn source_point_transfer_uses_a_real_lower_not_an_independent_upper() {
            assert_eq!(point_upper(100, 37, 80).unwrap(), 57);
            assert_eq!(point_upper(100, 37, 0).unwrap(), 137);
            assert!(point_upper(79, 37, 80).is_err());
            assert!(point_upper(u64::MAX, u64::MAX, 0).is_err());
            // Positive baseline changes the SOURCE lower. Do not force the observed baseline to0.
            let lower = arithmetic([32, 4, 4], 16, 17).unwrap().growth_lower;
            assert_eq!(lower, 23);
            assert_eq!(point_upper(100, 37, lower).unwrap(), 114);
        }
        #[test]
        fn raw_control_upper_is_numeric_and_negative_omits_the_actual_producer_term() {
            use mlx_gen::gen_core::weightsmeta::{Dtype, SafetensorsTensorHeader as Header};
            let component = Component {
                files: vec![PathBuf::from("literal-control/model.safetensors")],
                headers: vec![
                    Header {
                        name: "control_all_x_embedder.2-1.weight".into(),
                        dtype: Dtype::BF16,
                        shape: vec![2, 132],
                        data_bytes: 528,
                    },
                    Header {
                        name: "control_all_x_embedder.2-1.bias".into(),
                        dtype: Dtype::BF16,
                        shape: vec![2],
                        data_bytes: 4,
                    },
                ],
            };
            let graph = raw_graph(&[("Control", component.clone())]).unwrap();
            let mut changed = component.clone();
            changed.headers[0].shape[0] = 3;
            changed.headers[0].data_bytes = 792;
            assert_ne!(
                graph,
                raw_graph(&[("Control", changed.clone())]).unwrap(),
                "ACTUAL_ARCHITECTURE_NOT_DECLARED_CONFIG_ONLY"
            );
            changed.headers[0].data_bytes += 1;
            assert!(
                raw_graph(&[("Control", changed)]).is_err(),
                "HEADER_OFFSETS_NOT_INVENTED_FOR_FAMILY"
            );
            let allocation = super::super::original_pre_embed_capacity::control_source_terms(
                &component, "2-1", 16,
            )
            .unwrap();
            // Independent literal native cache-envelope answers: ceil(528/16)*16+32-1,
            // and min(2*4,4+32)-1. Neither expectation calls the emitter or its helper.
            assert_eq!(
                total(&allocation.control).unwrap(),
                559 + 7,
                "CONTROL_REAL_PRODUCER_EVENT_UPPER"
            );
            assert_eq!(allocation.control_event_request, Some(528));
            let windows = vec![window(
                IosPhase::Denoise,
                "z-image/control/denoise/control/source-op-0/request-first/decode-late-false",
                "Control",
            )];
            let mut map = host_map(
                operations(&windows, &allocation, true, false).unwrap(),
                allocation,
            );
            assert!(map.control_numeric_failures().unwrap().is_empty());
            let phase_dto_before = map.operations[0].window.resident_bytes;
            map.omit_control_upper_terms().unwrap();
            assert_eq!(
                map.operations[0].window.resident_bytes, phase_dto_before,
                "NEGATIVE_IS_NOT_PHASE_DTO_TAMPER"
            );
            let errors = map.control_numeric_failures().unwrap();
            assert_eq!(errors.len(), 1);
            assert!(errors[0].contains("phase denoise: Control source-event upper 0 < required constructor-event floor 528"));
            assert!(
                !errors[0].contains("INCOMPLETE"),
                "EVENT_NUMERIC_RED_NOT_A_MISSING_PROOF"
            );
        }
        fn window(phase: IosPhase, class: &str, label: &str) -> IosWindowPrice {
            IosWindowPrice {
                phase,
                class: Some(class.into()),
                label: label.into(),
                resident_bytes: 17,
                held_bytes: 3,
                scratch_bytes: 5,
                basis: None,
                owner_domain: "HOST-source-owner-not-native-backing".into(),
            }
        }
        fn host_map(operations: Vec<Operation>, allocation: AllocationInventory) -> SourceMap {
            SourceMap {
                family: Family {
                    session: std::process::id(),
                    architecture: "HOST-shape".into(),
                    tier: "BF16/Q8".into(),
                    role: "control".into(),
                    sampling: "Euler/native/manual-start0".into(),
                    configured_graph: "actual-Control-graph".into(),
                },
                literal_program: serde_json::json!({"geometry":{"count":1}}),
                program_binding: "HOST-program-only".into(),
                work_binding: "HOST-typed-work-only".into(),
                operations,
                allocation,
                intervals: Vec::new(),
                captures: Vec::new(),
                point_floors: Vec::new(),
                lower_scope_reason: Some("HOST has no native point capture".into()),
                actual_page: 16,
                source_files: BTreeMap::new(),
                constructor_origins: BTreeMap::new(),
                source_epochs: Vec::new(),
                dense_before: None,
            }
        }
        #[test]
        fn all_source_ops_histories_and_gaps_remain_distinct_and_uncertified() {
            let allocation = AllocationInventory {
                control: Vec::new(),
                control_event_request: None,
                prefix: Err("HOST no real raw prefix".into()),
            };
            let windows=vec![window(IosPhase::Load,"z-image/control/load/TE/source-op-0","TE Load"),
                window(IosPhase::Conditioning,"z-image/control/conditioning/caption/source-op-0",crate::quantum::CAPTION_FEATURES),
                window(IosPhase::Conditioning,"z-image/control/conditioning/caption/source-op-1",crate::quantum::CAPTION_FEATURES),
                window(IosPhase::Encode,"z-image/control/encode/VAE/source-op-0","Encode"),
                window(IosPhase::Denoise,"z-image/control/denoise/DIT/source-op-0/source-first-use/request-first/decode-late-false","DiT"),
                window(IosPhase::Denoise,"z-image/control/denoise/DIT/source-op-0/denoise-recurrent/decode-late-true","DiT"),
                window(IosPhase::Denoise,"z-image/control/denoise/sampler/source-op-0/denoise-recurrent/decode-late-true","Sampler"),
                window(IosPhase::Decode,"z-image/control/decode/output/source-op-0/request-first/decode-late-true","RGB")];
            let ops = operations(&windows, &allocation, true, true).unwrap();
            assert_eq!(ops.len(), windows.len());
            assert_ne!(
                ops[1].window.class, ops[2].window.class,
                "CAPTION_GATHER_AND_POST_INPUT_DROP_CAST_DIFFER"
            );
            assert_eq!(
                ops[1].input_borrow_scope,
                InputBorrowScope::DuringEncodeTokens
            );
            assert_eq!(
                ops[2].input_borrow_scope,
                InputBorrowScope::AfterInputTemporaries
            );
            assert_eq!(ops[4].history.epoch, Some("request-first"));
            assert!(ops[4].history.quant_first_use);
            assert_eq!(ops[5].history.epoch, Some("denoise-recurrent"));
            assert_eq!(ops[5].history.decode_late, Some(true));
            assert_ne!(
                ops[4].window.class, ops[5].window.class,
                "FIRST_RECURRENT_LATE_QUANT_HISTORY_NOT_CANCELLED"
            );
            assert!(ops
                .iter()
                .all(|op| !op.native_interval_obligations.is_empty()));
            let map = host_map(ops, allocation);
            assert!(
                !map.outstanding(&map).unwrap().is_empty(),
                "INVENTORY_IS_NOT_A_COMPLETE_PROOF"
            );
            let value = map.value().unwrap();
            assert_eq!(value["operations"].as_array().unwrap().len(), windows.len());
            assert!(value["first_point_floors"].as_array().unwrap().is_empty());
        }
        #[test]
        fn same_family_rejects_role_tier_graph_sampling_and_session_borrowing() {
            let empty = || AllocationInventory {
                control: Vec::new(),
                control_event_request: None,
                prefix: Err("HOST".into()),
            };
            let anchor = host_map(Vec::new(), empty());
            let mut request = host_map(Vec::new(), empty());
            assert!(request.same_family(&anchor).is_ok());
            for field in 0..6 {
                request.family = anchor.family.clone();
                match field {
                    0 => request.family.session = request.family.session.wrapping_add(1),
                    1 => request.family.architecture.push('x'),
                    2 => request.family.tier.push('x'),
                    3 => request.family.role = "text".into(),
                    4 => request.family.sampling.push('x'),
                    _ => request.family.configured_graph.push('x'),
                }
                assert!(
                    request.same_family(&anchor).is_err(),
                    "SAME_FAMILY_DIMENSION_{field}"
                );
            }
        }
        #[test]
        fn quantum_gaps_and_actual_baseline_metadata_are_not_erased_by_label_candidates() {
            use mlx_gen::ios_pricing::{IosObservedCheck, IosObservedMetadata};
            let allocation = AllocationInventory {
                control: Vec::new(),
                control_event_request: None,
                prefix: Err("HOST".into()),
            };
            let windows = vec![
                window(
                    IosPhase::Conditioning,
                    "z-image/text/conditioning/caption/source-op-0",
                    "Caption",
                ),
                window(
                    IosPhase::Conditioning,
                    "z-image/text/conditioning/caption/source-op-1",
                    "Caption",
                ),
            ];
            let ops = operations(&windows, &allocation, false, false).unwrap();
            let metadata = vec![IosObservedMetadata {
                provider: "z_image_fixture".into(),
                kind: "before_model_load".into(),
                fields: [
                    ("te_rust_handle_present".into(), 0),
                    ("dit_rust_handle_present".into(), 0),
                    ("vae_rust_handle_present".into(), 0),
                ]
                .into_iter()
                .collect(),
            }];
            let make = |label: &str, check: Option<u32>, sync: bool| NativeBoundary {
                label: label.into(),
                synchronized: sync,
                check,
                label_phase: Some(IosPhase::Conditioning),
                lifecycle_phase: Some(IosPhase::Conditioning),
                active_absolute: 37,
                interval_peak_absolute: 41,
                scratch: Some(5),
                context: vec![0],
            };
            let capture = NativeRepeat {
                baseline: IosMemoryBaseline {
                    domain: IosMemoryDomain::NativeAllocatorGrowth,
                    coordinate: IosMemoryCoordinate::BeforeModelLoad,
                    bytes: 17,
                    source: "HOST synthetic positive baseline".into(),
                },
                observed: IosAnchorObservation {
                    domain: IosMemoryDomain::NativeAllocatorGrowth,
                    coordinate: IosMemoryCoordinate::BeforeModelLoad,
                    baseline_bytes: 17,
                    metadata,
                    checks: vec![IosObservedCheck {
                        phase: Some(IosPhase::Conditioning),
                        label: "Caption".into(),
                        context: vec![0],
                    }],
                    telemetry_complete: true,
                },
                boundaries: vec![
                    make("Caption", Some(0), false),
                    make("Caption", None, true),
                    make("actual phase transition", None, false),
                    make("actual return", None, false),
                ],
                native: MeasuredRun {
                    demand: 24,
                    phases: [(IosPhase::Conditioning, 24)].into_iter().collect(),
                },
            };
            let mut measured = Measured {
                repeats: vec![capture],
            };
            measured.repeats[0].boundaries[3].active_absolute = 3;
            measured.repeats[0].boundaries[3].interval_peak_absolute = 7;
            let trace = intervals(&ops, &measured).unwrap();
            assert_eq!(trace.len(), 4, "PREFIX_SYNC_PHASE_AND_RETURN_GAPS_RETAINED");
            assert_eq!(trace[0].baseline, 17, "BASELINE_NOT_DEFAULTED_TO_ZERO");
            assert_eq!(
                trace[0].candidates,
                vec![0, 1],
                "ALL_PLAUSIBLE_SOURCE_OPS_NOT_A_LABEL_WINNER"
            );
            assert_eq!(trace[1].check, None);
            assert!(trace[1].synchronized);
            assert!(
                trace[2].candidates.is_empty() && trace[3].candidates.is_empty(),
                "GAPS_NOT_DISCARDED_AS_UNOBSERVED_ZERO"
            );
            assert_eq!(
                (trace[3].end_active, trace[3].interval_peak),
                (3, 7),
                "BELOW_BASELINE_ABSOLUTES_RETAINED"
            );
            measured.repeats[0].boundaries[0].context.push(99);
            assert!(
                intervals(&ops, &measured).is_err(),
                "ACTUAL_METADATA_INDEX_REQUIRED"
            );
            measured.repeats[0].boundaries[0].context.pop();
            measured.repeats[0].observed.baseline_bytes = 0;
            assert!(
                intervals(&ops, &measured).is_err(),
                "DIFFERENT_BASELINE_INVENTORY_NOT_COMPATIBLE_BY_DOMAIN_STRING"
            );
        }
        #[test]
        fn actual_typed_projection_family_ignores_endpoint_execution_but_not_roles() {
            use mlx_gen::gen_core::{Conditioning, GenerationRequest};
            let spec = crate::ios_pricing_tests::endpoint_spec(
                Path::new("HOST-literal-source"),
                Some(8),
                false,
            );
            let make = |mode: Mode| {
                let mut conditioning = Vec::new();
                if mode.control() {
                    conditioning.push(Conditioning::Control {
                        image: image(2, 1),
                        kind: ControlKind::Pose,
                        scale: Some(0.8),
                    });
                }
                if mode.img2img() {
                    conditioning.push(Conditioning::Reference {
                        image: image(2, 2),
                        strength: None,
                    });
                }
                Projection {
                    provider: if mode.control() {
                        crate::model_control::MODEL_ID
                    } else {
                        crate::model::MODEL_ID
                    }
                    .into(),
                    described_spec: spec.clone(),
                    request: GenerationRequest {
                        width: 288,
                        height: 288,
                        count: 1,
                        steps: Some(3),
                        seed: Some(7),
                        sampler: Some(mlx_gen::Solver::Euler.name().into()),
                        conditioning,
                        ..Default::default()
                    },
                    program: serde_json::json!({"actual_constructor_configurations":["TE3","tinyDiT4","VAEencoder","VAEdecoder"],
                        "manual_flow":{"start":usize::from(mode.img2img()),"shift_bits":3.0f32.to_bits()},
                        "windows":{"actual_TE_layers":3,"actual_base_blocks":4,"actual_control_blocks":mode.control().then_some(15)}}),
                }
            };
            let projection = make(Mode::Control);
            let control = Family::from_projection(&projection, &spec, Mode::Control).unwrap();
            let mut resident = spec.clone();
            resident.offload_policy = OffloadPolicy::Resident;
            resident.load_shape = LoadShape::EagerMaterialization;
            assert_eq!(
                control,
                Family::from_projection(&projection, &resident, Mode::Control).unwrap(),
                "FAMILY_NOT_REPARTITIONED_BY_ENDPOINT_LOAD_FORM"
            );
            let mut changed = make(Mode::Control);
            changed.request.width = 1024;
            changed.request.height = 512;
            assert_eq!(
                control,
                Family::from_projection(&changed, &spec, Mode::Control).unwrap(),
                "GEOMETRY_IS_WORK_NOT_FAMILY"
            );
            assert_ne!(
                execution_binding(&projection).unwrap(),
                execution_binding(&changed).unwrap(),
                "ACTUAL_TYPED_GEOMETRY_STILL_WORK_BOUND"
            );
            let mut different_hint = make(Mode::Control);
            if let Conditioning::Control { image, .. } = &mut different_hint.request.conditioning[0]
            {
                image.pixels[0] ^= 1;
            }
            assert_eq!(
                control,
                Family::from_projection(&different_hint, &spec, Mode::Control).unwrap()
            );
            assert_ne!(
                execution_binding(&projection).unwrap(),
                execution_binding(&different_hint).unwrap(),
                "ACTUAL_HINT_PIXELS_NOT_DTO_DEFAULT_OR_FAMILY_ID"
            );
            let families = [
                Mode::Text,
                Mode::Reference,
                Mode::Control,
                Mode::ControlReference,
            ]
            .map(|mode| Family::from_projection(&make(mode), &spec, mode).unwrap());
            for (index, family) in families.iter().enumerate() {
                assert!(
                    !families[..index].contains(family),
                    "ACTUAL_ROLE_FAMILIES_NEVER_BORROW_TEXT"
                );
            }
            assert!(
                Family::from_projection(&projection, &spec, Mode::Text).is_err(),
                "ROLE_ENUM_CANNOT_RELABEL_ACTUAL_CONTROL_CARRIER"
            );
        }
        #[test]
        fn literal_source_epoch_changes_are_errors_not_new_anchor_identity() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("literal-source.bin");
            std::fs::write(&path, b"before").unwrap();
            let before = source_epochs([path.clone()]).unwrap();
            assert_eq!(
                before[0].0, path,
                "LITERAL_PATH_NOT_CANONICALIZED_FOR_BINDING"
            );
            std::fs::write(&path, b"changed-longer").unwrap();
            let after = source_epochs([path.clone()]).unwrap();
            assert_ne!(before, after, "SOURCE_EPOCH_MUST_BRACKET_ACTUAL_CAPTURE");
            assert!(source_epochs([dir.path().join("missing")]).is_err());
        }
    }
}
