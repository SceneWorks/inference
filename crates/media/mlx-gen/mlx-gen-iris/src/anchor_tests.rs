//! CPU-only Iris anchor-law proofs on tiny dense and packed q4/q8 exports (epic 16774).
//! Every measurement is a fresh load + run inside diagnostic headroom with Metal trapped.
//!
//! Every derivation goes through `derive_certified` with the source-owned
//! `ios_pricing::ios_bounder` certificates, witnessed by the anchor run's own actual quantum checks.
//! Those certificates are native-allocator-counter only (never task footprint). Prompt batches and
//! serial counts are certified at their ACTUAL rows/count, never scaled from batch 1
//! (`IOS_NON_ADDITIVE_BATCH` stays true). Single-prompt 2x-area, sampling-epoch and depth
//! second-size checks assert measured ≤ derived, whole run and per phase. Restoration remains its
//! own single-mode family: its host planes are outside MLX active memory, so it has no native
//! certificate and `derive_certified` refuses it with its exact first-uncovered-class reason.

use mlx_gen::gen_core::iris::{IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, MemoryGeometry, MemoryMode, MemoryNumericTier, MemorySelection,
    MemoryStrategy, MemoryStrategySupport,
};
use mlx_gen::ios_anchor_law::{area, derive_certified, measure_phases, MeasuredRun, NotDerivable};
use mlx_gen::ios_pricing::{IosAnchorObservation, IosPhase, IosPricing, IosPricingRequest};
use mlx_gen::ios_quantum::{
    with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
};
use mlx_gen::{
    GenerationRequest, Generator, Image, LoadSpec, Quant, TargetSize, Transform, TransformRequest,
    WeightsSource,
};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;

use crate::ios_pricing::{ios_pricing, ios_quantum_phase};
use crate::packed::ConvertKind;
use crate::stream_tests::{
    backbone_dir, fixtures, generation_spec, in_cpu_child, restore_request, AMPLE,
};

/// The fixture geometry of every generation run.
const W: u32 = 32;
const H: u32 = 16;

/// The two actual endpoints and their matching specs: optimized (the last selection of the
/// deepest legal implemented rung, streaming only when that rung engages it) and fully resident
/// (eager whole components, no streaming or staging). A Resident selection over a deferred spec
/// is NOT the resident endpoint.
fn proved_selections(
    provider: &str,
    spec: &LoadSpec,
) -> Vec<(
    &'static str,
    bool,
    bool,
    LoadSpec,
    MemorySelection,
    GenerationMemory,
)> {
    let contract = crate::memory_strategy::memory_strategy_contract(provider, spec).unwrap();
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let optimized = MemoryStrategy::ALL
        .into_iter()
        .rev()
        .filter(|s| {
            contract
                .capability(*s)
                .is_some_and(|c| matches!(c.support, MemoryStrategySupport::Implemented))
        })
        .find_map(|s| {
            contract
                .selection_space(s, tier, false)
                .ok()?
                .last()
                .copied()
        })
        .unwrap();
    let rung0 = *contract
        .selection_space(MemoryStrategy::Resident, tier, false)
        .unwrap()
        .last()
        .unwrap();
    [("optimized", false, optimized), ("rung 0", true, rung0)]
        .into_iter()
        .map(|(name, resident, s)| {
            assert_eq!(s.strategy == MemoryStrategy::Resident, resident);
            let memory = contract.generation_memory(&s).unwrap_or_else(|| {
                assert!(resident, "optimized execution controls are missing");
                assert_eq!(
                    contract.resident_request_memory,
                    mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults,
                    "only the declared Resident preserve-defaults contract permits None"
                );
                GenerationMemory::default()
            });
            if resident {
                assert_eq!(
                    memory,
                    GenerationMemory::default(),
                    "Resident must engage no levers"
                );
            }
            let expected_stream = s.strategy == MemoryStrategy::BoundedTransformerResidency;
            assert_eq!(memory.stream_transformer_blocks, expected_stream);
            if !resident && provider == crate::model::MODEL_ID {
                assert!(
                    memory.stage_residency && memory.chunk_attention,
                    "highest legal optimized generation requires staging and bounded attention"
                );
                assert!(memory.attention_chunk_size.is_some());
                if expected_stream {
                    assert!(memory.transformer_window_size.is_some());
                }
            }
            let mut endpoint = spec.clone();
            endpoint.load_shape = if expected_stream {
                LoadShape::DeferredMaterialization
            } else {
                LoadShape::EagerMaterialization
            };
            endpoint.offload_policy = if !resident && provider == crate::model::MODEL_ID {
                mlx_gen::gen_core::OffloadPolicy::Sequential
            } else {
                mlx_gen::gen_core::OffloadPolicy::Resident
            };
            (name, resident, expected_stream, endpoint, s, memory)
        })
        .collect()
}

/// One measured run and the actual scalar quantum marks its repeats emitted. The marks are the
/// anchor's certificate witnesses; they are never Arrays or a planned/replayed observation.
struct Captured {
    run: MeasuredRun,
    observations: Vec<IosAnchorObservation>,
}

impl std::ops::Deref for Captured {
    type Target = MeasuredRun;
    fn deref(&self) -> &MeasuredRun {
        &self.run
    }
}

/// Record the actual production marks of `body` (metadata, and each check's latest mark of every
/// producer/kind) in the native-allocator domain the Iris certificates are stated in.
fn observe_marks<T>(body: impl FnOnce() -> T) -> (T, IosAnchorObservation) {
    use mlx_gen::ios_pricing::{
        IosMemoryCoordinate, IosMemoryDomain, IosObservedCheck, IosObservedMetadata,
    };
    use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};
    type State = (IosAnchorObservation, BTreeMap<(String, String), u32>);
    let recorded: std::sync::Arc<std::sync::Mutex<State>> =
        std::sync::Arc::new(std::sync::Mutex::new((
            IosAnchorObservation {
                domain: IosMemoryDomain::NativeAllocatorGrowth,
                coordinate: IosMemoryCoordinate::BeforeModelLoad,
                ..Default::default()
            },
            BTreeMap::new(),
        )));
    let sink = std::sync::Arc::clone(&recorded);
    struct RemoveListener;
    impl Drop for RemoveListener {
        fn drop(&mut self) {
            set_quantum_listener(None);
        }
    }
    assert!(
        !set_quantum_listener(Some(Box::new(move |mark| {
            let mut recorded = sink.lock().expect("actual mark collector lock");
            let (observation, latest) = &mut *recorded;
            match mark {
                QuantumMark::Metadata {
                    provider,
                    kind,
                    fields,
                } => {
                    let index = u32::try_from(observation.metadata.len())
                        .expect("bounded fixture metadata");
                    observation.metadata.push(IosObservedMetadata {
                        provider: provider.into(),
                        kind: kind.into(),
                        fields: fields
                            .iter()
                            .map(|&(key, value)| (key.into(), value))
                            .collect(),
                    });
                    latest.insert((provider.into(), kind.into()), index);
                }
                QuantumMark::Checked { label, .. } => observation.checks.push(IosObservedCheck {
                    phase: ios_quantum_phase(label),
                    label: label.into(),
                    context: latest.values().copied().collect(),
                }),
                QuantumMark::Synchronized { .. } => {}
            }
        }))),
        "an isolated anchor child must not replace another capture listener"
    );
    let guard = RemoveListener;
    let value = body();
    drop(guard);
    let observation = recorded
        .lock()
        .expect("actual mark collector lock")
        .0
        .clone();
    (value, observation)
}

/// Use the same repeat-max capture policy for anchor and request. CPU graph scheduling can vary
/// a phase peak between identical runs; never hide a discrepancy behind an assertion tolerance.
/// Each repeat loads afresh, and the largest measured demand and each phase are retained, exactly
/// as `MeasuredAnchor` defines its capture semantics. Every repeat's actual marks are kept.
fn measure(mut body: impl FnMut() -> mlx_gen::Result<()>) -> Captured {
    let mut worst = MeasuredRun {
        demand: 0,
        phases: BTreeMap::new(),
    };
    let mut observations = Vec::new();
    for repeat in 0..3 {
        let (((), run), observation) = observe_marks(|| {
            with_cpu_diagnostic_headroom(AMPLE, || measure_phases(ios_quantum_phase, &mut body))
                .unwrap()
        });
        eprintln!("CAPTURE repeat {repeat}: {} {:?}", run.demand, run.phases);
        assert!(
            !observation.checks.is_empty(),
            "a measured run must observe its actual quantum checks"
        );
        observations.push(observation);
        worst.demand = worst.demand.max(run.demand);
        for (phase, bytes) in run.phases {
            let slot = worst.phases.entry(phase).or_insert(0);
            *slot = (*slot).max(bytes);
        }
    }
    Captured {
        run: worst,
        observations,
    }
}

/// A missing measured or priced phase must never turn into a zero or an omitted assertion.
fn assert_phase_coverage(name: &str, measured: &BTreeMap<IosPhase, u64>, pricing: &IosPricing) {
    let priced: BTreeSet<_> = pricing.phases.iter().map(|p| p.phase).collect();
    assert!(!priced.is_empty(), "{name}: no priced phases");
    assert_eq!(
        priced.len(),
        pricing.phases.len(),
        "{name}: duplicate priced phase"
    );
    assert_eq!(
        measured.keys().copied().collect::<BTreeSet<_>>(),
        priced,
        "{name}: missing measured or priced phase"
    );
}

#[test]
fn missing_phase_is_not_a_zero_or_a_skipped_assertion() {
    let mut pricing = IosPricing::default();
    pricing.take(IosPhase::Denoise, 1, 1, "denoise");
    pricing.take(IosPhase::Decode, 1, 1, "decode");
    let measured = BTreeMap::from([(IosPhase::Denoise, 1), (IosPhase::Decode, 1)]);
    assert_phase_coverage("control", &measured, &pricing);
    let mut missing_measurement = measured.clone();
    missing_measurement.remove(&IosPhase::Decode);
    assert!(std::panic::catch_unwind(|| {
        assert_phase_coverage("missing measurement", &missing_measurement, &pricing)
    })
    .is_err());
    let mut missing_price = pricing.clone();
    missing_price.phases.retain(|p| p.phase != IosPhase::Decode);
    assert!(std::panic::catch_unwind(|| {
        assert_phase_coverage("missing price", &measured, &missing_price)
    })
    .is_err());
}

/// One derivation checked against a measured run: measured ≤ derived, whole run and in every
/// phase both state.
struct Check {
    name: String,
    /// The anchor's own measured demand.
    anchor: u64,
    measured: u64,
    derived: Result<u64, NotDerivable>,
    /// `(phase, measured, derived)` of every phase measured above its derivation.
    over: Vec<(&'static str, u64, u64)>,
}

impl Check {
    fn passed(&self) -> bool {
        self.derived
            .as_ref()
            .is_ok_and(|&d| self.measured <= d && self.over.is_empty())
    }
}

/// The certified derivation of `request` from `anchor`: the anchor pricing carries the anchor
/// run's actual marks, and the Iris source bounder supplies every certificate. A bounder error is a
/// test failure, never a silent refusal.
fn certify(
    name: &str,
    anchor: (&Captured, &IosPricingRequest<'_>, &IosPricing),
    request: (&IosPricingRequest<'_>, &IosPricing),
) -> Result<mlx_gen::ios_anchor_law::Derivation, NotDerivable> {
    let (anchor_run, anchor_ctx, at_anchor) = anchor;
    let (request_ctx, priced) = request;
    let mut witnessed = at_anchor.clone();
    witnessed.observations = anchor_run.observations.clone();
    let bounds = crate::ios_pricing::ios_bounder(anchor_ctx, &witnessed, request_ctx, priced)
        .unwrap_or_else(|e| panic!("{name}: the Iris source bounder failed: {e}"));
    let g = anchor_ctx.geometry;
    derive_certified(
        &anchor_run.anchor(area(g.width, g.height, g.batch)),
        &witnessed,
        priced,
        &bounds,
    )
}

/// A payload-free token-replay anchor (the catalog form: no work carrier, only its recorded
/// token) must price the same source profile as its actual-request anchor and receive the SAME
/// certificates from the same captured marks; without its token it refuses with `missing`.
fn assert_replay_anchor_certifies(
    name: &str,
    anchor_run: &Captured,
    actual: (&IosPricingRequest<'_>, &IosPricing),
    free: (
        &IosPricingRequest<'_>,
        crate::ios_pricing::IrisReplay<'_>,
        &IosPricing,
    ),
    request: (&IosPricingRequest<'_>, &IosPricing),
    missing: &str,
) {
    let (actual_ctx, at_anchor) = actual;
    let (free_ctx, replay, replayed) = free;
    let (request_ctx, priced) = request;
    assert!(
        replayed.phases == at_anchor.phases
            && replayed.windows == at_anchor.windows
            && replayed.absent_phases == at_anchor.absent_phases,
        "{name}: the token replay priced a different anchor profile"
    );
    let witness = |pricing: &IosPricing| {
        let mut witnessed = pricing.clone();
        witnessed.observations = anchor_run.observations.clone();
        witnessed
    };
    let (witnessed, replay_witnessed) = (witness(at_anchor), witness(replayed));
    let actual_bounds =
        crate::ios_pricing::ios_bounder(actual_ctx, &witnessed, request_ctx, priced)
            .unwrap_or_else(|e| panic!("{name}: actual anchor refused: {e}"));
    let replay_bounds = crate::ios_pricing::ios_bounder_with_replay(
        free_ctx,
        Some(replay),
        &replay_witnessed,
        request_ctx,
        priced,
    )
    .unwrap_or_else(|e| panic!("{name}: token-replay anchor refused: {e}"));
    assert!(!replay_bounds.is_empty(), "{name}: no certificate");
    assert_eq!(
        replay_bounds, actual_bounds,
        "{name}: the token-replay anchor certified differently"
    );
    match crate::ios_pricing::ios_bounder(free_ctx, &replay_witnessed, request_ctx, priced) {
        Err(mlx_gen::Error::Unsupported(reason)) => assert_eq!(reason, missing, "{name}"),
        other => panic!("{name}: a carrier-free anchor without its token was {other:?}"),
    }
}

fn check(
    name: String,
    anchor: (&Captured, &IosPricingRequest<'_>, &IosPricing),
    request: (&Captured, &IosPricingRequest<'_>, &IosPricing),
) -> Check {
    let (anchor_run, anchor_ctx, at_anchor) = anchor;
    let (measured, request_ctx, priced) = request;
    let mut c = Check {
        name,
        anchor: anchor_run.demand,
        measured: measured.demand,
        derived: Ok(0),
        over: Vec::new(),
    };
    for (side, run, pricing) in [
        ("anchor", &anchor_run.run, at_anchor),
        ("request", &measured.run, priced),
    ] {
        assert_phase_coverage(&format!("{} {side}", c.name), &run.phases, pricing);
        mlx_gen::ios_quantum::assert_tracks_measured(
            &format!("{} {side} whole request", c.name),
            pricing.peak_bytes(),
            run.demand,
            1.5,
        );
        for p in &pricing.phases {
            mlx_gen::ios_quantum::assert_tracks_measured(
                &format!("{} {side} {}", c.name, p.phase.as_key()),
                p.peak_bytes(),
                run.phases[&p.phase],
                1.5,
            );
            eprintln!(
                "  {} {side:<7} {:<12} measured {:>9} = resident {:>9} + scratch {:>9} ({})",
                c.name,
                p.phase.as_key(),
                run.phases[&p.phase],
                p.resident_bytes,
                p.scratch_bytes,
                p.binding
            );
        }
    }
    match certify(
        &c.name,
        (anchor_run, anchor_ctx, at_anchor),
        (request_ctx, priced),
    ) {
        Ok(derived) => {
            let phases: BTreeSet<_> = derived.phases.iter().map(|p| p.phase).collect();
            for phase in measured.phases.keys() {
                assert!(
                    phases.contains(phase),
                    "{}: derivation omitted measured phase {}",
                    c.name,
                    phase.as_key()
                );
            }
            // A derived phase the run never measures must be an actual absence priced at zero.
            for phase in &derived.phases {
                if !measured.phases.contains_key(&phase.phase) {
                    assert!(
                        priced.absent_phases.contains_key(&phase.phase) && phase.bytes == 0,
                        "{}: derived unmeasured phase {} is not an actual absence",
                        c.name,
                        phase.phase.as_key()
                    );
                }
            }
            for phase in derived
                .phases
                .iter()
                .filter(|p| measured.phases.contains_key(&p.phase))
            {
                let got = measured.phases[&phase.phase];
                eprintln!(
                    "  {} derived {:<12} measured {:>9} derived {:>9}{}",
                    c.name,
                    phase.phase.as_key(),
                    got,
                    phase.bytes,
                    if phase.calibrated { " (κ)" } else { "" }
                );
                if got > phase.bytes {
                    c.over.push((phase.phase.as_key(), got, phase.bytes));
                }
            }
            c.derived = Ok(derived.demand);
            eprintln!(
                "ANCHOR {}: measured {} vs derived {} ({:+.1}%) {}",
                c.name,
                c.measured,
                derived.demand,
                100.0 * (derived.demand as f64 - c.measured as f64) / c.measured.max(1) as f64,
                if c.passed() { "PASS" } else { "FAIL" }
            );
        }
        Err(e) => {
            eprintln!("ANCHOR {}: NOT DERIVABLE: {e}", c.name);
            c.derived = Err(e);
        }
    }
    c
}

fn failures(checks: &[Check]) -> Vec<String> {
    checks
        .iter()
        .filter(|c| !c.passed())
        .map(|c| format!("{}: {} vs {:?} {:?}", c.name, c.measured, c.derived, c.over))
        .collect()
}

// ---- iris_3b: the prompt batch derived from the text anchor ------------------------------------

fn generation_request(prompts: &[&str], memory: GenerationMemory) -> GenerationRequest {
    let (prompt, prompt_batch) = match prompts {
        [one] => ((*one).to_owned(), Vec::new()),
        many => (
            String::new(),
            many.iter().map(|p| (*p).to_owned()).collect(),
        ),
    };
    GenerationRequest {
        prompt,
        prompt_batch,
        width: W,
        height: H,
        steps: Some(3),
        seed: Some(42),
        memory: (memory != GenerationMemory::default()).then_some(memory),
        ..Default::default()
    }
}

/// The iOS pricing admission hands the law: mode `text_to_image`, batch = the prompts, the
/// artifact-determined window (`token_window` 0, as `ios_admission::token_window` sends for Iris).
static TEXT_TO_IMAGE: MemoryMode = MemoryMode::TextToImage;

fn generation_ctx<'a>(
    spec: &'a LoadSpec,
    selection: &'a MemorySelection,
    memory: &'a GenerationMemory,
    width: u32,
    batch: u32,
    actual: &'a GenerationRequest,
) -> IosPricingRequest<'a> {
    IosPricingRequest {
        provider: crate::model::MODEL_ID,
        spec,
        mode: &TEXT_TO_IMAGE,
        geometry: MemoryGeometry {
            width,
            height: H,
            batch,
            frames: 1,
            reference_count: 0,
        },
        selection,
        execution: memory,
        generation: Some(actual),
        transform: None,
        depth: None,
        token_window: 0,
    }
}

fn generation_pricing(
    spec: &LoadSpec,
    selection: &MemorySelection,
    memory: &GenerationMemory,
    width: u32,
    batch: u32,
    actual: &GenerationRequest,
) -> IosPricing {
    ios_pricing(&generation_ctx(
        spec, selection, memory, width, batch, actual,
    ))
    .unwrap()
}

thread_local! {
    static GENERATION_ENDPOINT: Cell<Option<bool>> = const { Cell::new(None) };
    static ENDPOINT_COMPONENTS: RefCell<BTreeMap<&'static str, usize>> = const { RefCell::new(BTreeMap::new()) };
    static GENERATION_EPOCHS: RefCell<VecDeque<crate::ios_pricing::GenerationEpoch>> = const { RefCell::new(VecDeque::new()) };
    static CURRENT_EPOCH: Cell<Option<crate::ios_pricing::GenerationEpoch>> = const { Cell::new(None) };
    static GENERATION_SHAPE: Cell<Option<[i32; 4]>> = const { Cell::new(None) };
}

/// Inspect storage without evaluating: a lazy placeholder must fail rather than become resident
/// because a proof materialized it. Native availability is checked before reading the data pointer.
pub(crate) fn materialized_addresses<'a>(
    arrays: impl IntoIterator<Item = &'a mlx_rs::Array>,
) -> Vec<usize> {
    arrays
        .into_iter()
        .map(|array| {
            let mut available = false;
            assert_eq!(
                unsafe { mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr()) },
                0,
                "native parameter availability query failed"
            );
            assert!(
                available,
                "endpoint owns an unevaluated parameter, not resident storage"
            );
            let address = mlx_llm::primitives::weights::data_address(array);
            assert_ne!(address, 0, "materialized parameter has no storage");
            address
        })
        .collect()
}

/// Called at the actual production loader entry, not at a proof wrapper invocation.
pub(crate) fn record_generation_loader_entry(component: &'static str) {
    if GENERATION_ENDPOINT.with(Cell::get).is_some() {
        ENDPOINT_COMPONENTS.with(|seen| *seen.borrow_mut().entry(component).or_insert(0) += 1);
    }
}

/// Called on the REAL loaded component, before its owner can discard it during staging.
pub(crate) fn observe_generation_tower(tower: &crate::text_encoder::IrisTextEncoder) {
    if let Some(streamed) = GENERATION_ENDPOINT.with(Cell::get) {
        tower.assert_endpoint_ownership(streamed);
    }
}

pub(crate) fn observe_generation_backbone(dit: &crate::dit::IrisDiT) {
    if let Some(streamed) = GENERATION_ENDPOINT.with(Cell::get) {
        dit.assert_endpoint_ownership(streamed);
    }
}

/// Witnesses run at actual solver/DiT/serial-loop sites. They inspect dtype, shape and already-live
/// storage only; they never evaluate a lazy input to manufacture a retention claim.
pub(crate) fn observe_solver_epoch(
    index: usize,
    step: &mlx_gen::gen_core::iris::DpmStep,
    x: &mlx_rs::Array,
    previous: Option<&mlx_rs::Array>,
) {
    if GENERATION_ENDPOINT.with(Cell::get).is_none() {
        return;
    }
    assert!(
        CURRENT_EPOCH.with(Cell::get).is_none(),
        "previous model evaluation never ran"
    );
    let expected = GENERATION_EPOCHS
        .with(|epochs| epochs.borrow_mut().pop_front())
        .expect("actual solver evaluated beyond its canonical plan");
    assert_eq!(index, expected.index);
    assert_eq!(*step, expected.step);
    let shape = GENERATION_SHAPE
        .with(Cell::get)
        .expect("missing expected solver geometry");
    assert_eq!(x.shape(), &shape);
    assert_eq!(
        x.dtype(),
        mlx_rs::Dtype::Float32,
        "solver state must remain f32"
    );
    assert_eq!(
        previous.is_some(),
        expected.history_rows != 0,
        "actual previous x0 retention differs from the epoch law"
    );
    if let Some(previous) = previous {
        assert_eq!(previous.shape(), &shape);
        assert_eq!(previous.dtype(), mlx_rs::Dtype::Float32);
        // Dynamic state may still be a pending dependency at this callback. Its actual peak is
        // measured through the production quantum barrier; never test-side-evaluate it here.
    }
    CURRENT_EPOCH.with(|current| current.set(Some(expected)));
}

pub(crate) fn observe_generation_forward(x: &mlx_rs::Array, conditioning: &mlx_rs::Array) {
    if GENERATION_ENDPOINT.with(Cell::get).is_none() {
        return;
    }
    let expected = CURRENT_EPOCH
        .with(Cell::take)
        .expect("model forward has no solver epoch");
    let mut shape = GENERATION_SHAPE
        .with(Cell::get)
        .expect("missing forward geometry");
    shape[0] = expected.forward_rows as i32;
    assert_eq!(x.shape(), &shape, "actual CFG/interval forward rows");
    assert_eq!(
        x.dtype(),
        mlx_rs::Dtype::Float32,
        "flow input is f32 before projection-local casts"
    );
    assert_eq!(conditioning.shape()[0], shape[0]);
}

pub(crate) fn observe_serial_outputs(serial: u32, outputs: &[mlx_rs::Array]) {
    if GENERATION_ENDPOINT.with(Cell::get).is_none() {
        return;
    }
    let expected =
        GENERATION_EPOCHS.with(|epochs| *epochs.borrow().front().expect("extra serial solve"));
    assert_eq!(serial, expected.serial);
    assert_eq!(
        outputs.len(),
        serial as usize,
        "earlier solved batches were not retained"
    );
    let shape = GENERATION_SHAPE
        .with(Cell::get)
        .expect("missing serial geometry");
    assert_eq!(
        outputs.len() as u64 * shape[0] as u64,
        expected.completed_rows
    );
    for output in outputs {
        assert_eq!(output.shape(), &shape);
        assert_eq!(output.dtype(), mlx_rs::Dtype::Float32);
    }
    let _storage = materialized_addresses(outputs);
}

/// Finite native-fixture witness only: huge-N metadata/refusal tests must not collect this
/// executable expected epoch queue before the production plan-payload guard.
fn run_generation_endpoint(
    spec: &LoadSpec,
    request: &GenerationRequest,
    resident: bool,
    expected_stream: bool,
) -> mlx_gen::Result<Vec<Vec<u8>>> {
    let memory = match request.memory {
        Some(memory) => memory,
        None if resident => {
            let contract =
                crate::memory_strategy::memory_strategy_contract(crate::model::MODEL_ID, spec)
                    .unwrap();
            assert_eq!(
                contract.resident_request_memory,
                mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults
            );
            GenerationMemory::default()
        }
        None => panic!("optimized request lost its execution controls"),
    };
    let streamed = expected_stream;
    let staged = !resident;
    assert_eq!(memory.stream_transformer_blocks, streamed);
    assert_eq!(memory.stage_residency, staged);
    assert_eq!(memory.chunk_attention, !resident);
    let resources =
        mlx_gen::gen_core::iris::GenerationResources::from_spec(spec, crate::model::MODEL_ID)
            .map_err(mlx_gen::Error::from)?;
    let config = mlx_gen::gen_core::iris::IrisConfig::from_dir(&resources.backbone_dir)
        .map_err(mlx_gen::Error::from)?;
    let recipe = crate::ios_pricing::resolved_generation_recipe(&config, request, 42)?;
    let shape = [
        recipe.params.prompts.len() as i32,
        config.model.in_channels as i32,
        recipe.params.height as i32,
        recipe.params.width as i32,
    ];
    struct Restore(
        Option<bool>,
        BTreeMap<&'static str, usize>,
        VecDeque<crate::ios_pricing::GenerationEpoch>,
        Option<crate::ios_pricing::GenerationEpoch>,
        Option<[i32; 4]>,
    );
    impl Drop for Restore {
        fn drop(&mut self) {
            GENERATION_ENDPOINT.with(|expected| expected.set(self.0));
            ENDPOINT_COMPONENTS.with(|seen| *seen.borrow_mut() = std::mem::take(&mut self.1));
            GENERATION_EPOCHS.with(|epochs| *epochs.borrow_mut() = std::mem::take(&mut self.2));
            CURRENT_EPOCH.with(|current| current.set(self.3));
            GENERATION_SHAPE.with(|shape| shape.set(self.4));
        }
    }
    let _restore = Restore(
        GENERATION_ENDPOINT.with(|expected| expected.replace(Some(streamed))),
        ENDPOINT_COMPONENTS.with(|seen| std::mem::take(&mut *seen.borrow_mut())),
        GENERATION_EPOCHS
            .with(|epochs| std::mem::replace(&mut *epochs.borrow_mut(), recipe.epochs().collect())),
        CURRENT_EPOCH.with(|current| current.replace(None)),
        GENERATION_SHAPE.with(|expected| expected.replace(Some(shape))),
    );
    let generator = crate::model::load_model(spec)?;
    let before = generator.assert_retained_endpoint(!staged);
    let output = generator
        .generate(request, &mut |_| {})
        .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
    let expected_images = request.prompt_batch.len().max(1) * request.count as usize;
    let pixels = match output {
        mlx_gen::GenerationOutput::Images(images) => {
            assert_eq!(
                images.len(),
                expected_images,
                "prompt_batch * count outputs must all be retained and decoded"
            );
            images.into_iter().map(|image| image.pixels).collect()
        }
        _ => panic!("Iris generation returned non-image output"),
    };
    let after = generator.assert_retained_endpoint(!staged);
    assert_eq!(
        before, after,
        "resident parameter storage was replaced during the request"
    );
    assert!(
        GENERATION_EPOCHS.with(|epochs| epochs.borrow().is_empty()),
        "canonical epochs were never evaluated"
    );
    assert!(
        CURRENT_EPOCH.with(Cell::get).is_none(),
        "last model evaluation never ran"
    );
    ENDPOINT_COMPONENTS.with(|seen| {
        assert_eq!(
            *seen.borrow(),
            BTreeMap::from([("backbone", 1), ("tower", 1)]),
            "both actual components must load once, without request-time rematerialization"
        )
    });
    Ok(pixels)
}

fn measure_generation(
    spec: &LoadSpec,
    request: &GenerationRequest,
    resident: bool,
    expected_stream: bool,
) -> Captured {
    measure(|| run_generation_endpoint(spec, request, resident, expected_stream).map(|_| ()))
}

/// The tiers proved on the iOS deferred load: dense, and the packed q4 / q8 exports.
pub(crate) fn generation_tiers(root: &std::path::Path) -> Vec<(&'static str, LoadSpec)> {
    let mut tiers = vec![("dense", generation_spec(LoadShape::DeferredMaterialization))];
    let aligned = crate::packed_tests::aligned_tiny_text_encoder(root);
    for (name, quant) in [("q4 packed", Quant::Q4), ("q8 packed", Quant::Q8)] {
        let backbone = crate::packed_tests::convert(
            ConvertKind::Export { depth: false },
            &backbone_dir(),
            root,
            quant,
        );
        let tower = crate::packed_tests::convert(ConvertKind::TextEncoder, &aligned, root, quant);
        let mut spec = LoadSpec::new(WeightsSource::Dir(backbone))
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_quant(quant);
        spec.components
            .insert(TEXT_ENCODER_COMPONENT.into(), WeightsSource::Dir(tower));
        tiers.push((name, spec));
    }
    tiers
}

/// The batch the request is priced at (the proof's mutation seam).
#[derive(Clone, Copy)]
struct BatchScaling {
    priced_batch: u32,
}

const BATCH: BatchScaling = BatchScaling { priced_batch: 2 };

/// Per tier × selection: the 2-prompt batch and the serial count 2 (each with the anchor's own
/// self-certified single-prompt derivation beside it), and (the comparator) one prompt at twice
/// the width, each derived from the measured single-prompt text anchor.
struct PromptBatchChecks {
    batches: Vec<(Check, u64)>,
    wides: Vec<Check>,
}

fn prompt_batch_checks(scaling: BatchScaling) -> PromptBatchChecks {
    let root = tempfile::tempdir().unwrap();
    let text = ["a lighthouse on a cliff at dusk"];
    let batch = ["a red fox in the snow", "golden hour"];
    let mut out = PromptBatchChecks {
        batches: Vec::new(),
        wides: Vec::new(),
    };
    for (tier, spec) in generation_tiers(root.path()) {
        for (rung, resident, expected_stream, spec, selection, memory) in
            proved_selections(crate::model::MODEL_ID, &spec)
        {
            let anchor_request = generation_request(&text, memory);
            let anchor = measure_generation(&spec, &anchor_request, resident, expected_stream);
            let anchor_ctx = generation_ctx(&spec, &selection, &memory, W, 1, &anchor_request);
            let at_anchor = ios_pricing(&anchor_ctx).unwrap();
            let anchor_side = (&anchor, &anchor_ctx, &at_anchor);
            // The single-prompt request certified from its own anchor: what a batch would price
            // at if it were derived or priced as batch size 1.
            let single = certify(
                &format!("iris_3b text→text {tier} {rung}"),
                anchor_side,
                (&anchor_ctx, &at_anchor),
            )
            .unwrap()
            .demand;
            let batched = generation_request(&batch, memory);
            let batched_ctx = generation_ctx(
                &spec,
                &selection,
                &memory,
                W,
                scaling.priced_batch,
                &batched,
            );
            let batched_priced = ios_pricing(&batched_ctx).unwrap();
            // The catalog's payload-free anchor: the same context with no carrier, plus the
            // anchor's recorded canonical sampling token.
            let token = crate::ios_pricing::canonical_token(
                &crate::ios_pricing::generation_metadata_config(&spec).unwrap(),
                &anchor_request,
            )
            .unwrap();
            let free_ctx = IosPricingRequest {
                generation: None,
                ..anchor_ctx
            };
            assert_replay_anchor_certifies(
                &format!("iris_3b replay anchor→batch2 {tier} {rung}"),
                &anchor,
                (&anchor_ctx, &at_anchor),
                (
                    &free_ctx,
                    crate::ios_pricing::IrisReplay::Generation(&token),
                    &crate::ios_pricing::ios_pricing_with_token(&free_ctx, &token).unwrap(),
                ),
                (&batched_ctx, &batched_priced),
                "iris_3b pricing requires the actual generation request or its recorded sampling \
                 token (sampling, CFG and serial count)",
            );
            out.batches.push((
                check(
                    format!("iris_3b text→batch2 {tier} {rung}"),
                    anchor_side,
                    (
                        &measure_generation(&spec, &batched, resident, expected_stream),
                        &batched_ctx,
                        &batched_priced,
                    ),
                ),
                single,
            ));
            // `count` retains each solved f32 batch until the common RGB conversion. It is not
            // the same execution as a prompt batch, although admission counts both as total images.
            let mut counted = generation_request(&text, memory);
            counted.count = 2;
            let counted_ctx = generation_ctx(
                &spec,
                &selection,
                &memory,
                W,
                scaling.priced_batch,
                &counted,
            );
            out.batches.push((
                check(
                    format!("iris_3b text→count2 {tier} {rung}"),
                    anchor_side,
                    (
                        &measure_generation(&spec, &counted, resident, expected_stream),
                        &counted_ctx,
                        &ios_pricing(&counted_ctx).unwrap(),
                    ),
                ),
                single,
            ));
            let mut wide = generation_request(&text, memory);
            wide.width = 2 * W;
            let wide_ctx = generation_ctx(&spec, &selection, &memory, 2 * W, 1, &wide);
            out.wides.push(check(
                format!("iris_3b text→text {}x{H} {tier} {rung}", 2 * W),
                anchor_side,
                (
                    &measure_generation(&spec, &wide, resident, expected_stream),
                    &wide_ctx,
                    &ios_pricing(&wide_ctx).unwrap(),
                ),
            ));
        }
    }
    out
}

fn prompt_batch_body() {
    const { assert!(crate::ios_pricing::IOS_NON_ADDITIVE_BATCH) };
    let PromptBatchChecks { batches, wides } = prompt_batch_checks(BATCH);
    assert_eq!(
        batches.len(),
        12,
        "prompt batch / count × 3 tiers × 2 endpoints"
    );
    for (c, single) in &batches {
        let derived = *c
            .derived
            .as_ref()
            .unwrap_or_else(|e| panic!("{}: batch is not certified: {e}", c.name));
        // The certificate carries the ACTUAL rows/count: it is above the text anchor it starts
        // from, and above the same anchor's own single-prompt certificate (never batch-1 priced).
        assert!(
            derived > c.anchor,
            "{}: derived {derived} does not scale the text anchor {}",
            c.name,
            c.anchor
        );
        assert!(
            derived > *single,
            "{}: derived {derived} is priced as the single-prompt certificate {single}",
            c.name
        );
    }
    let batch_checks: Vec<_> = batches.into_iter().map(|(c, _)| c).collect();
    let under = failures(&batch_checks);
    assert!(
        under.is_empty(),
        "batch/count above its certified derivation: {under:#?}"
    );
    let under = failures(&wides);
    assert!(under.is_empty(), "2x area above its derivation: {under:#?}");
}

/// Certify the prompt batch and serial count at their actual rows/count (never scaled from
/// batch 1) and the single prompt at twice the area, each from the measured text anchor, whole
/// run and per phase.
///
/// Mutation: price the batch/count as batch 1, or certify it from the batch-1 request; the
/// pricing context or the single-prompt comparison goes red.
#[test]
fn prompt_batch_derivability_is_measured_at_both_endpoints_per_phase() {
    in_cpu_child(
        "anchor_tests::prompt_batch_derivability_is_measured_at_both_endpoints_per_phase",
        prompt_batch_body,
    );
}

fn metadata_body() {
    use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};
    use std::sync::{Arc, Mutex};
    let spec = generation_spec(LoadShape::EagerMaterialization);
    let mut request =
        generation_request(&["a red fox", "golden hour"], GenerationMemory::default());
    request.sampler = Some("euler".into());
    request.steps = Some(4);
    request.count = 2;
    request.cfg_interval = Some((0.25, 0.75));
    request.scheduler_shift = Some(1.0);
    let config = crate::ios_pricing::generation_metadata_config(&spec).unwrap();
    let recipe = crate::ios_pricing::resolved_generation_recipe(&config, &request, 42).unwrap();
    type Record = (String, Vec<(String, u64)>);
    let records = Arc::new(Mutex::new((Vec::<Record>::new(), false)));
    let sink = records.clone();
    assert!(!set_quantum_listener(Some(Box::new(move |mark| {
        if let QuantumMark::Metadata {
            provider,
            kind,
            fields,
        } = mark
        {
            if provider != crate::model::MODEL_ID {
                return;
            }
            let mut records = sink.lock().expect("metadata collector poisoned");
            if records.0.len() == 128 {
                records.1 = true;
                return;
            }
            records.0.push((
                kind.into(),
                fields
                    .iter()
                    .map(|&(key, value)| (key.into(), value))
                    .collect(),
            ));
        }
    }))));
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            set_quantum_listener(None);
        }
    }
    let restore = Restore;
    with_cpu_diagnostic_headroom(AMPLE, || {
        run_generation_endpoint(&spec, &request, true, false)
    })
    .unwrap();
    drop(restore);
    let records = records.lock().unwrap();
    assert!(!records.1, "actual epoch metadata was truncated");
    let by_kind = |kind: &str| {
        records
            .0
            .iter()
            .filter(|record| record.0 == kind)
            .map(|record| &record.1)
            .collect::<Vec<_>>()
    };
    let field = |fields: &Vec<(String, u64)>, key: &str| {
        fields
            .iter()
            .find(|(name, _)| name == key)
            .unwrap_or_else(|| panic!("missing metadata field {key}"))
            .1
    };
    assert_eq!(by_kind("request").len(), 1);
    let serials = by_kind("serial");
    assert_eq!(serials.len(), request.count as usize);
    for (index, serial) in serials.iter().enumerate() {
        assert_eq!(field(serial, "serial"), index as u64);
        assert_eq!(field(serial, "completed_batches"), index as u64);
        assert_eq!(field(serial, "completed_rows"), 2 * index as u64);
    }
    let solvers = by_kind("solver");
    let forwards = by_kind("forward");
    let expected: Vec<_> = recipe.epochs().collect();
    assert_eq!(
        solvers.len(),
        expected.len(),
        "actual private solver loop coverage"
    );
    assert_eq!(
        forwards.len(),
        expected.len(),
        "actual model evaluation coverage"
    );
    for ((solver, forward), epoch) in solvers.iter().zip(forwards).zip(expected) {
        assert_eq!(field(solver, "step"), epoch.index as u64);
        assert_eq!(
            field(solver, "history_present"),
            u64::from(epoch.history_rows != 0)
        );
        assert_eq!(field(solver, "history_rows"), epoch.history_rows);
        assert_eq!(
            field(solver, "update_order"),
            1,
            "actual Euler uses First updates but retains history"
        );
        assert_eq!(field(solver, "first"), u64::from(epoch.index == 0));
        assert_eq!(
            field(solver, "terminal"),
            u64::from(epoch.index + 1 == recipe.plan.len())
        );
        assert_eq!(field(solver, "prior_completed_steps"), epoch.index as u64);
        assert_eq!(field(forward, "evaluation"), epoch.index as u64 + 1);
        assert_eq!(field(forward, "rows"), epoch.forward_rows);
        assert_eq!(field(forward, "conditioning_rows"), epoch.forward_rows);
        assert_eq!(field(forward, "guided"), u64::from(epoch.guided));
        assert_eq!(field(forward, "input_f32"), 1);
    }
    let completed = by_kind("solve_complete");
    assert_eq!(completed.len(), request.count as usize);
    for complete in completed {
        assert_eq!(field(complete, "evaluations"), recipe.plan.len() as u64);
    }
}

#[test]
fn actual_runtime_metadata_matches_recipe_epochs() {
    in_cpu_child(
        "anchor_tests::actual_runtime_metadata_matches_recipe_epochs",
        metadata_body,
    );
}

// ---- actual Iris solver / CFG / serial epochs ---------------------------------------------

fn sampling_body() {
    let root = tempfile::tempdir().unwrap();
    let mut checked = Vec::new();
    for (tier, source) in generation_tiers(root.path()) {
        for (rung, resident, expected_stream, spec, selection, memory) in
            proved_selections(crate::model::MODEL_ID, &source)
        {
            let mut anchor_request =
                generation_request(&["a lighthouse on a cliff at dusk"], memory);
            anchor_request.steps = Some(4);
            let anchor = measure_generation(&spec, &anchor_request, resident, expected_stream);
            let anchor_ctx = generation_ctx(&spec, &selection, &memory, W, 1, &anchor_request);
            let at_anchor = ios_pricing(&anchor_ctx).unwrap();
            // Mechanism classes, not a float grid: both actual solver graphs; no/all/mixed CFG;
            // first-only/last-only guidance; one/two/multistep; prompt rows vs serial count.
            for (name, sampler, steps, guidance, interval, rows, count, width) in [
                ("default order2", None, 4, 3.0, None, 1, 1, W),
                ("Euler full CFG", Some("euler"), 4, 3.0, None, 1, 1, W),
                (
                    "Euler unguided serial",
                    Some("euler"),
                    4,
                    1.0,
                    None,
                    2,
                    2,
                    W,
                ),
                (
                    "order2 unguided serial",
                    Some("dpmpp_2m"),
                    4,
                    1.0,
                    None,
                    1,
                    2,
                    W,
                ),
                (
                    "Euler mixed CFG",
                    Some("euler"),
                    4,
                    3.0,
                    Some((0.25, 0.75)),
                    2,
                    2,
                    2 * W,
                ),
                (
                    "order2 mixed CFG",
                    Some("dpmpp_2m"),
                    4,
                    3.0,
                    Some((0.25, 0.75)),
                    2,
                    2,
                    2 * W,
                ),
                ("one terminal step", Some("dpmpp_2m"), 1, 3.0, None, 1, 1, W),
                (
                    "two first-order steps",
                    Some("dpmpp_2m"),
                    2,
                    1.0,
                    None,
                    1,
                    1,
                    W,
                ),
                (
                    "Euler first-only CFG",
                    Some("euler"),
                    4,
                    3.0,
                    Some((0.9, 1.0)),
                    1,
                    1,
                    W,
                ),
                (
                    "order2 last-only CFG",
                    Some("dpmpp_2m"),
                    4,
                    3.0,
                    Some((0.0, 0.3)),
                    1,
                    2,
                    W,
                ),
            ] {
                let prompts: &[&str] = if rows == 1 {
                    &["a lighthouse on a cliff at dusk"]
                } else {
                    &["a red fox", "golden hour"]
                };
                let mut request = generation_request(prompts, memory);
                request.sampler = sampler.map(str::to_owned);
                request.steps = Some(steps);
                request.guidance = Some(guidance);
                request.cfg_interval = interval;
                request.scheduler_shift = interval.map(|_| 1.0);
                request.count = count;
                request.width = width;
                let mut guarded = None;
                let measured = measure(|| {
                    guarded = Some(run_generation_endpoint(
                        &spec,
                        &request,
                        resident,
                        expected_stream,
                    )?);
                    Ok(())
                });
                // Same actual endpoint and request, guards disabled: unchanged byte tolerance.
                let unguarded = match crate::model::load(&spec)
                    .unwrap()
                    .generate(&request, &mut |_| {})
                    .unwrap()
                {
                    mlx_gen::GenerationOutput::Images(images) => {
                        images.into_iter().map(|i| i.pixels).collect::<Vec<_>>()
                    }
                    _ => panic!("expected Iris RGB images"),
                };
                assert_eq!(
                    guarded.expect("the measured body must produce RGB"),
                    unguarded,
                    "{tier} {rung} {name}: guarded sampling changed RGB"
                );
                let request_ctx =
                    generation_ctx(&spec, &selection, &memory, width, rows * count, &request);
                checked.push(check(
                    format!("Iris sampling {tier} {rung} {name}"),
                    (&anchor, &anchor_ctx, &at_anchor),
                    (&measured, &request_ctx, &ios_pricing(&request_ctx).unwrap()),
                ));
            }
        }
    }
    let under = failures(&checked);
    assert!(
        under.is_empty(),
        "actual Iris sampling epochs above their derivation: {under:#?}"
    );
}

#[test]
fn actual_sampling_epochs_bound_both_solvers_cfg_intervals_and_serial_outputs() {
    in_cpu_child(
        "anchor_tests::actual_sampling_epochs_bound_both_solvers_cfg_intervals_and_serial_outputs",
        sampling_body,
    );
}

// ---- iris_3b_restore / iris_3b_depth: self-consistency across geometry -----------------------

pub(crate) fn downstream_tiers(
    root: &std::path::Path,
    src: &std::path::Path,
    depth: bool,
) -> Vec<(&'static str, LoadSpec)> {
    let deferred = |dir| {
        LoadSpec::new(WeightsSource::Dir(dir)).with_load_shape(LoadShape::DeferredMaterialization)
    };
    let mut tiers = vec![("dense", deferred(src.to_path_buf()))];
    for (name, quant) in [("q4 packed", Quant::Q4), ("q8 packed", Quant::Q8)] {
        let packed = crate::packed_tests::convert(ConvertKind::Export { depth }, src, root, quant);
        tiers.push((name, deferred(packed).with_quant(quant)));
    }
    tiers
}

fn downstream_pricing(
    provider: &str,
    task: IrisTask,
    spec: &LoadSpec,
    selection: &MemorySelection,
    memory: &GenerationMemory,
    geometry: MemoryGeometry,
    actual: (
        Option<&TransformRequest>,
        Option<&mlx_gen::gen_core::iris::depth::DepthRequest>,
    ),
) -> IosPricing {
    ios_pricing(&IosPricingRequest {
        provider,
        spec,
        mode: &crate::memory_strategy::memory_mode(task),
        geometry,
        selection,
        execution: memory,
        generation: None,
        transform: actual.0,
        depth: actual.1,
        token_window: 0,
    })
    .unwrap()
}

/// The downstream task's actual work context: a restoration transform or a depth request.
struct DownstreamWork<'a> {
    transform: Option<&'a TransformRequest>,
    depth: Option<&'a mlx_gen::gen_core::iris::depth::DepthRequest>,
}

fn downstream_ctx<'a>(
    provider: &'a str,
    mode: &'a MemoryMode,
    spec: &'a LoadSpec,
    selection: &'a MemorySelection,
    memory: &'a GenerationMemory,
    geometry: MemoryGeometry,
    work: DownstreamWork<'a>,
) -> IosPricingRequest<'a> {
    let DownstreamWork { transform, depth } = work;
    IosPricingRequest {
        provider,
        spec,
        mode,
        geometry,
        selection,
        execution: memory,
        generation: None,
        transform,
        depth,
        token_window: 0,
    }
}

fn restore_at(width: u32, height: u32) -> TransformRequest {
    TransformRequest {
        image: Image {
            width,
            height,
            pixels: (0..width * height * 3)
                .map(|i| (i * 53 % 256) as u8)
                .collect(),
        },
        target: TargetSize::Scale(4.0),
        ..restore_request()
    }
}

/// A restoration run measured, and the processing geometry its plan ran.
fn measure_restoration(
    spec: &LoadSpec,
    req: &TransformRequest,
    streamed: bool,
) -> (Captured, MemoryGeometry) {
    let mut geometry = None;
    let run = measure(|| {
        let restorer = crate::restoration::IrisRestorer::load(spec)?;
        restorer.assert_endpoint_ownership(streamed);
        let before = restorer.endpoint_parameter_addresses();
        restorer.plan(req)?;
        geometry = Some(MemoryGeometry {
            width: req.image.width,
            height: req.image.height,
            batch: 1,
            frames: 1,
            reference_count: 0,
        });
        restorer
            .apply(req, &mut |_| {})
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
        assert_eq!(
            before,
            restorer.endpoint_parameter_addresses(),
            "restoration replaced resident parameter storage during the request"
        );
        Ok(())
    });
    (run, geometry.unwrap())
}

fn depth_request(width: u32, height: u32) -> mlx_gen::gen_core::iris::depth::DepthRequest {
    mlx_gen::gen_core::iris::depth::DepthRequest {
        image: Image {
            width,
            height,
            pixels: (0..width * height * 3)
                .map(|i| (i * 31 % 256) as u8)
                .collect(),
        },
        resolution: mlx_gen::gen_core::iris::depth::DepthResolution::Native,
        ..Default::default()
    }
}

fn measure_depth(
    spec: &LoadSpec,
    width: u32,
    height: u32,
    streamed: bool,
) -> (Captured, MemoryGeometry) {
    let actual = depth_request(width, height);
    let run = measure(|| {
        let depth = crate::depth::load_depth(spec)?;
        depth.assert_endpoint_ownership(streamed);
        let before = depth.endpoint_parameter_addresses();
        let output = mlx_gen::gen_core::iris::depth::IrisDepthEstimator::estimate(
            &depth,
            &actual,
            &mut |_| {},
        )
        .map_err(mlx_gen::Error::from)?;
        assert_eq!((output.map.width, output.map.height), (width, height));
        assert_eq!(
            before,
            depth.endpoint_parameter_addresses(),
            "depth replaced resident parameter storage during the request"
        );
        Ok(())
    });
    let geometry = MemoryGeometry {
        width,
        height,
        batch: 1,
        frames: 1,
        reference_count: 0,
    };
    (run, geometry)
}

/// The first restoration class `derive_certified` reaches uncovered: Load runs first and its
/// classes are visited in order. Restoration has no native certificate, so this is its exact
/// current refusal.
const RESTORATION_REFUSAL_CLASS: &str =
    "iris.restoration.backbone_load/Iris backbone load/operator0";

fn downstream_body() {
    let root = tempfile::tempdir().unwrap();
    let mut checks = Vec::new();
    let mut restore_host = Vec::new();
    let upscaler = fixtures().join("tiny-snapshot/upscaler");
    let restore_mode = crate::memory_strategy::memory_mode(IrisTask::Restoration);
    for (tier, spec) in downstream_tiers(root.path(), &upscaler, false) {
        let provider = crate::restoration::MODEL_ID;
        for (rung, _resident, expected_stream, spec, selection, memory) in
            proved_selections(provider, &spec)
        {
            let streamed = expected_stream;
            let (first_request, second_request) = (restore_request(), restore_at(24, 20));
            let (anchor, at) = measure_restoration(&spec, &first_request, streamed);
            let (measured, second) = measure_restoration(&spec, &second_request, streamed);
            assert!(area(second.width, second.height, 1) > area(at.width, at.height, 1));
            let ctx = |g, actual| {
                downstream_ctx(
                    provider,
                    &restore_mode,
                    &spec,
                    &selection,
                    &memory,
                    g,
                    DownstreamWork {
                        transform: Some(actual),
                        depth: None,
                    },
                )
            };
            let (anchor_ctx, request_ctx) = (ctx(at, &first_request), ctx(second, &second_request));
            restore_host
                .push(crate::ios_pricing::restoration_decode_host_bytes(&anchor_ctx).unwrap());
            checks.push(check(
                format!(
                    "iris_3b_restore {}x{}→{}x{} {tier} {rung}",
                    at.width, at.height, second.width, second.height
                ),
                (&anchor, &anchor_ctx, &ios_pricing(&anchor_ctx).unwrap()),
                (&measured, &request_ctx, &ios_pricing(&request_ctx).unwrap()),
            ));
        }
    }
    let depth_src = fixtures().join("tiny-depth");
    let depth_mode = crate::memory_strategy::memory_mode(IrisTask::Depth);
    for (tier, spec) in downstream_tiers(root.path(), &depth_src, true) {
        let provider = crate::depth::DEPTH_MODEL_ID;
        for (rung, _resident, expected_stream, spec, selection, memory) in
            proved_selections(provider, &spec)
        {
            let streamed = expected_stream;
            let (anchor, at) = measure_depth(&spec, 24, 16, streamed);
            let (measured, second) = measure_depth(&spec, 48, 32, streamed);
            let (first_request, second_request) = (
                depth_request(at.width, at.height),
                depth_request(second.width, second.height),
            );
            let ctx = |g, actual| {
                downstream_ctx(
                    provider,
                    &depth_mode,
                    &spec,
                    &selection,
                    &memory,
                    g,
                    DownstreamWork {
                        transform: None,
                        depth: Some(actual),
                    },
                )
            };
            let (anchor_ctx, request_ctx) = (ctx(at, &first_request), ctx(second, &second_request));
            let (at_anchor, priced) = (
                ios_pricing(&anchor_ctx).unwrap(),
                ios_pricing(&request_ctx).unwrap(),
            );
            // Without the anchor run's actual marks there is no witness and no certificate.
            assert!(
                crate::ios_pricing::ios_bounder(&anchor_ctx, &at_anchor, &request_ctx, &priced)
                    .is_err(),
                "{tier} {rung}: an unwitnessed depth anchor was certified"
            );
            // The catalog's payload-free depth anchor: no DepthRequest, only its recorded token.
            let token = crate::ios_pricing::canonical_depth_token(&spec, &first_request).unwrap();
            let free_ctx = IosPricingRequest {
                depth: None,
                ..anchor_ctx
            };
            assert_replay_anchor_certifies(
                &format!("iris_3b_depth replay anchor 24x16→48x32 {tier} {rung}"),
                &anchor,
                (&anchor_ctx, &at_anchor),
                (
                    &free_ctx,
                    crate::ios_pricing::IrisReplay::Depth(&token),
                    &crate::ios_pricing::pricing_facts_with_depth_token(&free_ctx, &token)
                        .unwrap()
                        .pricing,
                ),
                (&request_ctx, &priced),
                "Iris depth pricing requires actual DepthRequest source/resolution",
            );
            checks.push(check(
                format!("iris_3b_depth 24x16→48x32 {tier} {rung}"),
                (&anchor, &anchor_ctx, &at_anchor),
                (&measured, &request_ctx, &priced),
            ));
        }
    }
    assert_eq!(checks.len(), 12, "2 routes × 3 tiers × 2 selections");
    let (restore, depth) = checks.split_at(6);
    // Restoration's intended NEGATIVE: invisible Rust host planes have no native certificate, so
    // the certified law refuses at its first uncovered class rather than pricing host payload.
    for (c, host) in restore.iter().zip(&restore_host) {
        assert!(
            *host > 0,
            "{}: fixture must actually exercise Rust host planes",
            c.name
        );
        assert_eq!(
            c.derived,
            Err(NotDerivable::Coverage(RESTORATION_REFUSAL_CLASS.into())),
            "{}: host payload upper {host} gained transfer authority",
            c.name
        );
    }
    // Depth: certified at the second size, above its own anchor, and measured ≤ derived whole
    // run and per phase.
    for c in depth {
        let derived = *c
            .derived
            .as_ref()
            .unwrap_or_else(|e| panic!("{}: depth second size is not certified: {e}", c.name));
        assert!(
            derived > c.anchor,
            "{}: derived {derived} ≤ anchor {}",
            c.name,
            c.anchor
        );
    }
    let under = failures(depth);
    assert!(under.is_empty(), "depth above its derivation: {under:#?}");
}

/// Restoration and depth, each its own family with one mode. Depth is certified from its own
/// measured run at a second input size and stays within it, whole run and per phase. Restoration's
/// host planes are invisible to MLX's active memory, so it has no native certificate: its exact
/// `derive_certified` refusal is asserted.
#[test]
fn depth_derives_a_second_size_and_restoration_is_host_unmeasurable() {
    in_cpu_child(
        "anchor_tests::depth_derives_a_second_size_and_restoration_is_host_unmeasurable",
        downstream_body,
    );
}

// ---- the priced resident bounds the live set ----------------------------------------------

/// Per phase, the largest MLX active memory (over the run's pre-load baseline) at any of its
/// quantum checks — the live set the phase's priced resident bytes stand for.
fn live_at_checks(body: impl FnOnce() -> mlx_gen::Result<()>) -> BTreeMap<IosPhase, u64> {
    let live: Rc<RefCell<BTreeMap<IosPhase, u64>>> = Rc::default();
    let sink = live.clone();
    mlx_gen::residency::drain_allocator_cache();
    let base = mlx_rs::memory::get_active_memory() as u64;
    with_cpu_diagnostic_headroom(AMPLE, || {
        with_cpu_diagnostic_observer(
            move |event| {
                if let QuantumEvent::Checked { label, .. } = event {
                    let active = (mlx_rs::memory::get_active_memory() as u64).saturating_sub(base);
                    let phase = ios_quantum_phase(label).unwrap();
                    let mut live = sink.borrow_mut();
                    let slot = live.entry(phase).or_insert(0);
                    *slot = (*slot).max(active);
                }
            },
            body,
        )
    })
    .unwrap();
    live.take()
}

/// Per phase, the live bytes at its checks that its priced resident does not account for.
fn unpriced(
    name: &str,
    live: &BTreeMap<IosPhase, u64>,
    pricing: &IosPricing,
) -> BTreeMap<IosPhase, u64> {
    assert_phase_coverage(name, live, pricing);
    pricing
        .phases
        .iter()
        .map(|p| {
            let at = live[&p.phase];
            eprintln!(
                "RESIDENT {name} {:<12} live {at:>9} priced {:>9}",
                p.phase.as_key(),
                p.resident_bytes
            );
            (p.phase, at.saturating_sub(p.resident_bytes))
        })
        .collect()
}

/// The unpriced live bytes of a family's runs (the first is its smallest work): each phase's stays
/// under one allocator page and never grows with the work.
fn assert_unpriced_is_fixed(family: &str, runs: &[BTreeMap<IosPhase, u64>]) {
    let page = crate::retention::PAGE_BYTES;
    let first = runs
        .first()
        .expect("a live-set family must have measured runs");
    assert!(!first.is_empty(), "{family}: no measured phases");
    for run in runs {
        assert_eq!(
            run.keys().collect::<Vec<_>>(),
            first.keys().collect::<Vec<_>>(),
            "{family}: missing live-set phase"
        );
        for (phase, &gap) in run {
            assert!(
                gap < page,
                "{family} {phase:?}: {gap} B live beyond its priced resident"
            );
            let base = first[phase];
            assert!(
                gap <= base,
                "{family} {phase:?}: the unpriced live set grows with the work ({base} → {gap} B)"
            );
        }
    }
}

fn resident_body() {
    let root = tempfile::tempdir().unwrap();
    for (tier, spec) in generation_tiers(root.path()) {
        for (rung, resident, expected_stream, spec, selection, memory) in
            proved_selections(crate::model::MODEL_ID, &spec)
        {
            let family = format!("iris_3b {tier} {rung}");
            let mut runs = Vec::new();
            for prompts in [
                &["a lighthouse on a cliff at dusk"][..],
                &["a red fox", "golden hour"],
            ] {
                for width in [W, 2 * W] {
                    let mut request = generation_request(prompts, memory);
                    request.width = width;
                    let live = live_at_checks(|| {
                        run_generation_endpoint(&spec, &request, resident, expected_stream)
                            .map(|_| ())
                    });
                    let batch = prompts.len() as u32;
                    runs.push(unpriced(
                        &format!("{family} {width}x{H} batch {batch}"),
                        &live,
                        &generation_pricing(&spec, &selection, &memory, width, batch, &request),
                    ));
                }
            }
            assert_unpriced_is_fixed(&family, &runs);
        }
    }
    let upscaler = fixtures().join("tiny-snapshot/upscaler");
    for (tier, spec) in downstream_tiers(root.path(), &upscaler, false) {
        let provider = crate::restoration::MODEL_ID;
        for (rung, _resident, expected_stream, spec, selection, memory) in
            proved_selections(provider, &spec)
        {
            let family = format!("iris_3b_restore {tier} {rung}");
            let mut runs = Vec::new();
            for req in [restore_request(), restore_at(24, 20)] {
                let mut geometry = None;
                let live = live_at_checks(|| {
                    let restorer = crate::restoration::IrisRestorer::load(&spec)?;
                    restorer.assert_endpoint_ownership(expected_stream);
                    restorer.plan(&req)?;
                    geometry = Some(MemoryGeometry {
                        width: req.image.width,
                        height: req.image.height,
                        batch: 1,
                        frames: 1,
                        reference_count: 0,
                    });
                    restorer
                        .apply(&req, &mut |_| {})
                        .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
                    Ok(())
                });
                let g = geometry.unwrap();
                runs.push(unpriced(
                    &format!("{family} {}x{}", g.width, g.height),
                    &live,
                    &downstream_pricing(
                        provider,
                        IrisTask::Restoration,
                        &spec,
                        &selection,
                        &memory,
                        g,
                        (Some(&req), None),
                    ),
                ));
            }
            assert_unpriced_is_fixed(&family, &runs);
        }
    }
    let depth_src = fixtures().join("tiny-depth");
    for (tier, spec) in downstream_tiers(root.path(), &depth_src, true) {
        let provider = crate::depth::DEPTH_MODEL_ID;
        for (rung, _resident, expected_stream, spec, selection, memory) in
            proved_selections(provider, &spec)
        {
            let family = format!("iris_3b_depth {tier} {rung}");
            let mut runs = Vec::new();
            for (width, height) in [(24u32, 16u32), (48, 32)] {
                let actual_depth = depth_request(width, height);
                let live = live_at_checks(|| {
                    let depth = crate::depth::load_depth(&spec)?;
                    depth.assert_endpoint_ownership(expected_stream);
                    mlx_gen::gen_core::iris::depth::IrisDepthEstimator::estimate(
                        &depth,
                        &actual_depth,
                        &mut |_| {},
                    )
                    .map_err(mlx_gen::Error::from)?;
                    Ok(())
                });
                let g = MemoryGeometry {
                    width,
                    height,
                    batch: 1,
                    frames: 1,
                    reference_count: 0,
                };
                runs.push(unpriced(
                    &format!("{family} {width}x{height}"),
                    &live,
                    &downstream_pricing(
                        provider,
                        IrisTask::Depth,
                        &spec,
                        &selection,
                        &memory,
                        g,
                        (None, Some(&actual_depth)),
                    ),
                ));
            }
            assert_unpriced_is_fixed(&family, &runs);
        }
    }
}

/// Every phase's priced resident bytes account for the MLX live set at each of its quantum checks
/// up to a few sub-page allocations that do not grow with the work — the pixel embedder's cached
/// position table, the forward's held RoPE tables, text stream and modulation cores, the per-row PiT
/// stage's batched input, the solve's text batches, the depth forward's held input and every
/// skeleton tensor at its allocated size included — for generation (optimized and rung 0, dense /
/// q4 / q8, text and batch at two widths), restoration and depth (dense / q4 / q8, two sizes).
///
/// *Mutation that reds this:* dropping the `pixel_table` term from a denoise or decode resident in
/// `ios_pricing`.
#[test]
fn the_priced_resident_accounts_for_the_live_set_at_every_check() {
    in_cpu_child(
        "anchor_tests::the_priced_resident_accounts_for_the_live_set_at_every_check",
        resident_body,
    );
}

// ---- installed generation adapter: native CPU anchor → twice the area -----------------------

/// Real PEFT file for the tiny backbone's patch projection, following adapter_facts_tests' native
/// parent-embed fixture. Constructed only AFTER in_cpu_child selects CPU and traps Metal; no fixture
/// Arrays survive into the before-model-load measurement coordinate.
fn generation_anchor_adapter(root: &std::path::Path) -> mlx_gen::AdapterSpec {
    use mlx_gen::gen_core::iris::adapter_provenance;
    use mlx_rs::{transforms::eval, Array};

    let path = root.join("anchor-patch-lora.safetensors");
    // Nonzero, modest factors keep the solver finite while exercising a large real rank term.
    let a = Array::from_slice(&vec![0.001f32; 1536 * 48], &[1536, 48]);
    let b = Array::from_slice(&vec![0.001f32; 64 * 1536], &[64, 1536]);
    eval([&a, &b]).unwrap();
    let metadata: std::collections::HashMap<_, _> =
        adapter_provenance(IrisTask::Generation, "iris_3b")
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect();
    Array::save_safetensors(
        [
            ("s_embedder.proj.lora_A.weight", &a),
            ("s_embedder.proj.lora_B.weight", &b),
        ],
        Some(&metadata),
        &path,
    )
    .unwrap();
    mlx_gen::AdapterSpec::new(path, 0.25, mlx_gen::AdapterKind::Lora)
}

fn installed_adapter_anchor_body() {
    let root = tempfile::tempdir().unwrap();
    let adapter = generation_anchor_adapter(root.path());
    let mut checked = Vec::new();
    for (tier, mut source) in generation_tiers(root.path()) {
        // Adapters attach to materialized projections; the streamed-load refusal is intentional.
        // Ask the contract for the deepest LEGAL eager endpoint, never force rung 4 with adapters.
        source.load_shape = LoadShape::EagerMaterialization;
        source.adapters = vec![adapter.clone()];
        for (rung, resident, expected_stream, spec, selection, memory) in
            proved_selections(crate::model::MODEL_ID, &source)
        {
            assert!(!expected_stream, "installed adapters cannot use streaming");
            let name = format!("iris_3b installed adapter→2x area {tier} {rung}");
            let anchor_request = generation_request(&["a lighthouse at dusk"], memory);
            let mut wide = anchor_request.clone();
            wide.width = 2 * W;
            let anchor = measure_generation(&spec, &anchor_request, resident, expected_stream);
            let measured = measure_generation(&spec, &wide, resident, expected_stream);
            // A header-only/mock overlay cannot satisfy these real install/materialize witnesses.
            // Require every repeat on BOTH sides, not just one mark from a previous capture.
            for captured in [&anchor, &measured] {
                for observation in &captured.observations {
                    for label in [
                        crate::adapters::ADAPTER_HOST_QUANTUM,
                        crate::adapters::ADAPTER_LOAD_QUANTUM,
                        crate::adapters::ADAPTER_MATERIALIZE_QUANTUM,
                    ] {
                        assert!(
                            observation.checks.iter().any(|check| check.label == label),
                            "{name}: actual installed-adapter boundary missing: {label}"
                        );
                    }
                }
            }
            let anchor_ctx = generation_ctx(&spec, &selection, &memory, W, 1, &anchor_request);
            let wide_ctx = generation_ctx(&spec, &selection, &memory, 2 * W, 1, &wide);
            let at_anchor = ios_pricing(&anchor_ctx).unwrap();
            let priced = ios_pricing(&wide_ctx).unwrap();
            let result = check(
                name.clone(),
                (&anchor, &anchor_ctx, &at_anchor),
                (&measured, &wide_ctx, &priced),
            );
            let adapted = *result
                .derived
                .as_ref()
                .unwrap_or_else(|e| panic!("{name}: installed adapter is not certified: {e}"));

            // Same request, tier and execution endpoint, with ONLY the installed stack removed.
            // Its own actual bare anchor witnesses the counterfactual certificate; do not feed
            // adapted marks to a bare bounder or compare a certificate against an unrelated price.
            let mut bare_spec = spec.clone();
            bare_spec.adapters.clear();
            let bare_anchor =
                measure_generation(&bare_spec, &anchor_request, resident, expected_stream);
            let bare_anchor_ctx =
                generation_ctx(&bare_spec, &selection, &memory, W, 1, &anchor_request);
            let bare_wide_ctx = generation_ctx(&bare_spec, &selection, &memory, 2 * W, 1, &wide);
            let bare_at_anchor = ios_pricing(&bare_anchor_ctx).unwrap();
            let bare_priced = ios_pricing(&bare_wide_ctx).unwrap();
            let bare = certify(
                &format!("{name} no-adapter control"),
                (&bare_anchor, &bare_anchor_ctx, &bare_at_anchor),
                (&bare_wide_ctx, &bare_priced),
            )
            .unwrap()
            .demand;
            assert!(
                adapted > bare,
                "{name}: adapter certificate {adapted} must exceed the same no-adapter request certificate {bare}"
            );
            checked.push(result);
        }
    }
    assert!(
        !checked.is_empty(),
        "no installed-adapter endpoint was exercised"
    );
    let under = failures(&checked);
    assert!(
        under.is_empty(),
        "installed adapter 2x area above certified derivation, whole run or per phase: {under:#?}"
    );
}

/// NATIVE macOS CPU-child proof (not host-only): real installed LoRA generation, dense/q4/q8,
/// resident and deepest legal eager optimized endpoints. Both sizes load/run afresh; the source
/// bounder and derive_certified must cover the run and every measured phase, with no tolerance.
/// Mutation: omit installed adapter load/retention/forward terms from the native upper ledger;
/// the native lower-bound checks or the no-adapter certificate discriminator must fail.
#[test]
fn installed_adapter_derivability_is_measured_at_both_endpoints_per_phase() {
    in_cpu_child(
        "anchor_tests::installed_adapter_derivability_is_measured_at_both_endpoints_per_phase",
        installed_adapter_anchor_body,
    );
}
