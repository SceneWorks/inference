//! CPU-only Iris anchor-law proofs on tiny dense and packed q4/q8 exports (epic 16774).
//! Every measurement is a fresh load + run inside diagnostic headroom with Metal trapped.
//!
//! The earlier batch-2 Decode counterexample predates the distinct final-clamp Denoise quantum
//! and is not an established classification of this tree. Collect every tier and endpoint before
//! classifying batch derivability; keep `IOS_NON_ADDITIVE_BATCH` true until whole-run and every
//! phase pass. Single-prompt 2x-area and depth second-size checks assert measured ≤ derived.
//! Restoration remains its own single-mode family: its host planes are visible to the device's
//! footprint capture but not MLX active memory, so the CPU stand-in cannot prove its derivation.
//! These assertions and the revised pricing are pending verification after the disk-space stop.

use mlx_gen::gen_core::iris::{IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::gen_core::{
    GenerationMemory, LoadShape, MemoryGeometry, MemoryMode, MemoryNumericTier, MemorySelection,
    MemoryStrategy, MemoryStrategySupport,
};
use mlx_gen::ios_anchor_law::{area, derive, measure_phases, MeasuredRun, NotDerivable};
use mlx_gen::ios_pricing::{IosPhase, IosPricing, IosPricingRequest};
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

/// Use the same repeat-max capture policy for anchor and request. CPU graph scheduling can vary
/// a phase peak between identical runs; never hide a discrepancy behind an assertion tolerance.
/// Each repeat loads afresh, and the largest measured demand and each phase are retained, exactly
/// as `MeasuredAnchor` defines its capture semantics.
fn measure(mut body: impl FnMut() -> mlx_gen::Result<()>) -> MeasuredRun {
    let mut worst = MeasuredRun {
        demand: 0,
        phases: BTreeMap::new(),
    };
    for repeat in 0..3 {
        let ((), run) =
            with_cpu_diagnostic_headroom(AMPLE, || measure_phases(ios_quantum_phase, &mut body))
                .unwrap();
        eprintln!("CAPTURE repeat {repeat}: {} {:?}", run.demand, run.phases);
        worst.demand = worst.demand.max(run.demand);
        for (phase, bytes) in run.phases {
            let slot = worst.phases.entry(phase).or_insert(0);
            *slot = (*slot).max(bytes);
        }
    }
    worst
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

fn check(
    name: String,
    anchor: (&MeasuredRun, u64, &IosPricing),
    request: (&MeasuredRun, u64, &IosPricing),
) -> Check {
    let (anchor_run, anchor_area, at_anchor) = anchor;
    let (measured, request_area, priced) = request;
    let mut c = Check {
        name,
        anchor: anchor_run.demand,
        measured: measured.demand,
        derived: Ok(0),
        over: Vec::new(),
    };
    for (side, run, pricing) in [
        ("anchor", anchor_run, at_anchor),
        ("request", measured, priced),
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
    match derive(
        &anchor_run.anchor(anchor_area),
        at_anchor,
        priced,
        request_area,
    ) {
        Ok(derived) => {
            assert_eq!(
                derived
                    .phases
                    .iter()
                    .map(|p| p.phase)
                    .collect::<BTreeSet<_>>(),
                measured.phases.keys().copied().collect::<BTreeSet<_>>(),
                "{}: derivation omitted a measured phase",
                c.name
            );
            for phase in &derived.phases {
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
fn generation_pricing(
    spec: &LoadSpec,
    selection: &MemorySelection,
    memory: &GenerationMemory,
    width: u32,
    batch: u32,
    actual: &GenerationRequest,
) -> IosPricing {
    ios_pricing(&IosPricingRequest {
        provider: crate::model::MODEL_ID,
        spec,
        mode: &MemoryMode::TextToImage,
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
    })
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
) -> MeasuredRun {
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

/// How the batch request is priced and scaled for the derivation (the proof's mutation seam).
#[derive(Clone, Copy)]
struct BatchScaling {
    /// The batch the request is priced at.
    priced_batch: u32,
    /// The batch the request's area counts.
    area_batch: u32,
}

const BATCH: BatchScaling = BatchScaling {
    priced_batch: 2,
    area_batch: 2,
};

/// Per tier × selection: the 2-prompt batch, and (the comparator) one prompt at twice the width,
/// each derived from the measured single-prompt text anchor.
fn prompt_batch_checks(scaling: BatchScaling) -> (Vec<Check>, Vec<Check>) {
    let root = tempfile::tempdir().unwrap();
    let text = ["a lighthouse on a cliff at dusk"];
    let batch = ["a red fox in the snow", "golden hour"];
    let (mut batches, mut wides) = (Vec::new(), Vec::new());
    for (tier, spec) in generation_tiers(root.path()) {
        for (rung, resident, expected_stream, spec, selection, memory) in
            proved_selections(crate::model::MODEL_ID, &spec)
        {
            let anchor = measure_generation(
                &spec,
                &generation_request(&text, memory),
                resident,
                expected_stream,
            );
            let at_anchor = generation_pricing(
                &spec,
                &selection,
                &memory,
                W,
                1,
                &generation_request(&text, memory),
            );
            let measured = measure_generation(
                &spec,
                &generation_request(&batch, memory),
                resident,
                expected_stream,
            );
            batches.push(check(
                format!("iris_3b text→batch2 {tier} {rung}"),
                (&anchor, area(W, H, 1), &at_anchor),
                (
                    &measured,
                    area(W, H, scaling.area_batch),
                    &generation_pricing(
                        &spec,
                        &selection,
                        &memory,
                        W,
                        scaling.priced_batch,
                        &generation_request(&batch, memory),
                    ),
                ),
            ));
            // `count` retains each solved f32 batch until the common RGB conversion. It is not
            // the same execution as a prompt batch, although admission counts both as total images.
            let mut counted = generation_request(&text, memory);
            counted.count = 2;
            batches.push(check(
                format!("iris_3b text→count2 {tier} {rung}"),
                (&anchor, area(W, H, 1), &at_anchor),
                (
                    &measure_generation(&spec, &counted, resident, expected_stream),
                    area(W, H, scaling.area_batch),
                    &generation_pricing(
                        &spec,
                        &selection,
                        &memory,
                        W,
                        scaling.priced_batch,
                        &counted,
                    ),
                ),
            ));
            let mut wide = generation_request(&text, memory);
            wide.width = 2 * W;
            wides.push(check(
                format!("iris_3b text→text {}x{H} {tier} {rung}", 2 * W),
                (&anchor, area(W, H, 1), &at_anchor),
                (
                    &measure_generation(&spec, &wide, resident, expected_stream),
                    area(2 * W, H, 1),
                    &generation_pricing(&spec, &selection, &memory, 2 * W, 1, &wide),
                ),
            ));
        }
    }
    (batches, wides)
}

fn prompt_batch_body() {
    const { assert!(crate::ios_pricing::IOS_NON_ADDITIVE_BATCH) };
    let (batches, wides) = prompt_batch_checks(BATCH);
    assert_eq!(
        batches.len(),
        12,
        "prompt batch / count × 3 tiers × 2 endpoints"
    );
    // The derivation carries the batch: every derivation is above the text anchor it starts from.
    for c in &batches {
        let derived = *c.derived.as_ref().unwrap();
        assert!(
            derived > c.anchor,
            "{}: derived {derived} does not scale the text anchor {}",
            c.name,
            c.anchor
        );
    }
    // Report the CURRENT whole-run and per-phase evidence, without demanding a stale failure.
    // The production nonadditivity flag stays true until this complete matrix is validated.
    let under = failures(&batches);
    eprintln!(
        "BATCH CLASSIFICATION: {} / {} whole-run + per-phase checks pass; failures {under:#?}",
        batches.iter().filter(|c| c.passed()).count(),
        batches.len()
    );
    for c in &batches {
        assert!(
            c.measured <= *c.derived.as_ref().unwrap(),
            "{}: batch whole-run demand exceeds its derivation",
            c.name
        );
    }
    // Single-prompt geometry remains a passing claim, without a measurement tolerance.
    let under = failures(&wides);
    assert!(under.is_empty(), "2x area above its derivation: {under:#?}");
}

/// Collect batch whole-run and per-phase classification on the current tree; a prior Decode
/// counterexample is not required to survive pricing repairs. A single prompt at twice the area
/// is checked against the same text anchor as well.
///
/// Mutation: price/count the batch as batch 1; the derived-demand growth assertion goes red.
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
            let at_anchor = generation_pricing(&spec, &selection, &memory, W, 1, &anchor_request);
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
                checked.push(check(
                    format!("Iris sampling {tier} {rung} {name}"),
                    (&anchor, area(W, H, 1), &at_anchor),
                    (
                        &measured,
                        area(width, H, rows * count),
                        &generation_pricing(
                            &spec,
                            &selection,
                            &memory,
                            width,
                            rows * count,
                            &request,
                        ),
                    ),
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
) -> (MeasuredRun, MemoryGeometry) {
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
) -> (MeasuredRun, MemoryGeometry) {
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

fn downstream_body() {
    let root = tempfile::tempdir().unwrap();
    let mut checks = Vec::new();
    let mut restore_host = Vec::new();
    let upscaler = fixtures().join("tiny-snapshot/upscaler");
    for (tier, spec) in downstream_tiers(root.path(), &upscaler, false) {
        let provider = crate::restoration::MODEL_ID;
        for (rung, _resident, expected_stream, spec, selection, memory) in
            proved_selections(provider, &spec)
        {
            let streamed = expected_stream;
            let (anchor, at) = measure_restoration(&spec, &restore_request(), streamed);
            let (measured, second) = measure_restoration(&spec, &restore_at(24, 20), streamed);
            assert!(area(second.width, second.height, 1) > area(at.width, at.height, 1));
            let price = |g, actual: &TransformRequest| {
                downstream_pricing(
                    provider,
                    IrisTask::Restoration,
                    &spec,
                    &selection,
                    &memory,
                    g,
                    (Some(actual), None),
                )
            };
            restore_host.push(
                crate::ios_pricing::restoration_decode_host_bytes(&IosPricingRequest {
                    provider,
                    spec: &spec,
                    mode: &crate::memory_strategy::memory_mode(IrisTask::Restoration),
                    geometry: at,
                    selection: &selection,
                    execution: &memory,
                    generation: None,
                    transform: Some(&restore_request()),
                    depth: None,
                    token_window: 0,
                })
                .unwrap(),
            );
            checks.push(check(
                format!(
                    "iris_3b_restore {}x{}→{}x{} {tier} {rung}",
                    at.width, at.height, second.width, second.height
                ),
                (
                    &anchor,
                    area(at.width, at.height, 1),
                    &price(at, &restore_request()),
                ),
                (
                    &measured,
                    area(second.width, second.height, 1),
                    &price(second, &restore_at(24, 20)),
                ),
            ));
        }
    }
    let depth_src = fixtures().join("tiny-depth");
    for (tier, spec) in downstream_tiers(root.path(), &depth_src, true) {
        let provider = crate::depth::DEPTH_MODEL_ID;
        for (rung, _resident, expected_stream, spec, selection, memory) in
            proved_selections(provider, &spec)
        {
            let streamed = expected_stream;
            let (anchor, at) = measure_depth(&spec, 24, 16, streamed);
            let (measured, second) = measure_depth(&spec, 48, 32, streamed);
            let price = |g, actual: &mlx_gen::gen_core::iris::depth::DepthRequest| {
                downstream_pricing(
                    provider,
                    IrisTask::Depth,
                    &spec,
                    &selection,
                    &memory,
                    g,
                    (None, Some(actual)),
                )
            };
            checks.push(check(
                format!("iris_3b_depth 24x16→48x32 {tier} {rung}"),
                (
                    &anchor,
                    area(at.width, at.height, 1),
                    &price(at, &depth_request(at.width, at.height)),
                ),
                (
                    &measured,
                    area(second.width, second.height, 1),
                    &price(second, &depth_request(second.width, second.height)),
                ),
            ));
        }
    }
    assert_eq!(checks.len(), 12, "2 routes × 3 tiers × 2 selections");
    let (restore, depth) = checks.split_at(6);
    // Restoration's intended NEGATIVE: invisible Rust host planes do not become process evidence
    // by subtracting a logical resident/host upper from the native CPU reading. No provider source
    // certificate is supplied by this legacy call. The separate depth/generation POSITIVE transfer
    // obligations remain below/in their own tests; this rejection does not satisfy them.
    for (c, host) in restore.iter().zip(&restore_host) {
        assert!(
            *host > 0,
            "{}: fixture must actually exercise Rust host planes",
            c.name
        );
        assert!(
            matches!(c.derived, Err(NotDerivable::Uncertified)),
            "{}: uncertified host payload upper {host} gained transfer authority: {:?}",
            c.name,
            c.derived
        );
    }
    // Depth: measured ≤ derived at the second size, whole run and per phase.
    let under = failures(depth);
    assert!(under.is_empty(), "depth above its derivation: {under:#?}");
}

/// Restoration and depth, each its own family with one mode. Depth is derived from its own measured
/// run at a second input size and stays within it (the law's self-consistency on that route).
/// Restoration is not measurable by this host stand-in (its host planes are priced resident but
/// invisible to MLX's active memory; a device capture's `phys_footprint` sees them), so only that
/// is asserted.
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
