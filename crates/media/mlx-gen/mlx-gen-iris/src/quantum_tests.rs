//! The iOS execution quanta of every Iris inference route, exercised on the committed miniature
//! exports in a CPU-only child process (the physical-iOS barriers enabled through
//! `mlx_gen::ios_quantum`'s CPU diagnostic fixture, Metal trapped):
//!
//! * guarded execution is BYTE-IDENTICAL to the unguarded host path;
//! * every block / layer / tile is checked before it synchronizes, once per block / layer / tile;
//! * a refused quantum refuses before its graph is built, and nothing after it runs;
//! * the scratch figures are read from the loaded tensors and the request geometry.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use mlx_gen::gen_core::iris::depth::{DepthRequest, DepthResolution, IrisDepthEstimator};
use mlx_gen::gen_core::iris::downstream::TaskExport;
use mlx_gen::gen_core::iris::{IrisConfig, IrisTask, TEXT_ENCODER_COMPONENT};
use mlx_gen::ios_quantum::{
    attention_block_bytes, take_refusal, with_cpu_diagnostic_headroom,
    with_cpu_diagnostic_observer, AttentionBlockGeometry, QuantumEvent, PROCESS_RESERVE_BYTES,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    AttentionBudget, AttentionPlan, GenerationOutput, GenerationRequest, Image, InputSizing,
    LoadSpec, Precision, Result, TargetSize, Transform, TransformRequest, WeightsSource,
};
use mlx_rs::{random, Array, Dtype};

use crate::dit::{quanta as dit_q, IrisDiT, TextBatch};
use crate::nn::diagnostics;

#[path = "../tests/common/metal_guard.rs"]
mod metal_guard;

/// Headroom every miniature quantum fits in.
const AMPLE: u64 = 1 << 34;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn backbone_dir() -> PathBuf {
    fixtures().join("tiny-snapshot/iris")
}

fn text_encoder_dir() -> PathBuf {
    fixtures().join("tiny-snapshot/text_encoder")
}

/// Run `body` in a child of this test binary whose MLX default device is the CPU, with Metal
/// trapped; the parent fails unless the child ran the body to the end.
fn in_cpu_child(test: &str, body: fn()) {
    const CHILD: &str = "IRIS_QUANTUM_CPU_CHILD";
    const DONE: &str = "IRIS_QUANTUM_CPU_CHILD_DONE";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(metal_guard::forbid_metal());
        assert!(metal_guard::metal_forbidden());
        body();
        std::fs::write(std::env::var(DONE).unwrap(), b"ok").unwrap();
        return;
    }
    let done = tempfile::tempdir().unwrap();
    let marker = done.path().join("done");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads", "1"])
        .env(CHILD, test)
        .env(DONE, &marker)
        .status()
        .unwrap();
    assert!(status.success(), "{test} failed in its CPU child");
    assert!(marker.is_file(), "{test}: the CPU child ran no body");
}

/// One observed quantum event, the attention calls executed before it, and whether MLX's cache
/// limit was zero (an `ios_quantum::Scope` held) when it fired.
type Observed = Vec<(QuantumEvent, usize, bool)>;

/// Run `body` with the barriers enabled at `headroom`, recording every event and how many
/// attention calls had executed when it fired.
fn observed<T>(headroom: u64, body: impl FnOnce() -> Result<T>) -> (Result<T>, Observed) {
    diagnostics::reset();
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = events.clone();
    let result = with_cpu_diagnostic_headroom(headroom, || {
        with_cpu_diagnostic_observer(
            move |event| {
                // Reading the limit means setting one; zero is restored when it was zero.
                let previous = mlx_rs::memory::set_cache_limit(0);
                mlx_rs::memory::set_cache_limit(previous);
                sink.borrow_mut().push((
                    event.clone(),
                    diagnostics::observations().len(),
                    previous == 0,
                ))
            },
            body,
        )
    });
    let events = events.take();
    (result, events)
}

fn is_checked(event: &QuantumEvent, want: &str) -> bool {
    matches!(event, QuantumEvent::Checked { label, .. } if label == want)
}

/// Each `label` quantum is checked, then synchronized, `count` times, never interleaved with
/// itself.
fn assert_paired(events: &Observed, label: &str, count: usize) {
    let mut sequence = Vec::new();
    for (event, _, _) in events {
        match event {
            QuantumEvent::Checked { label: l, .. } if l == label => sequence.push('C'),
            QuantumEvent::Synchronized { label: l, .. } if l == label => sequence.push('S'),
            _ => {}
        }
    }
    let want: String = "CS".repeat(count);
    assert_eq!(
        sequence.iter().collect::<String>(),
        want,
        "{label}: expected {count} checked-then-synchronized quanta"
    );
}

/// Every quantum ran under an entry point's `Scope` (MLX cache limit zero).
fn assert_scoped(events: &Observed) {
    for (event, _, scoped) in events {
        assert!(scoped, "{event:?} ran outside an ios_quantum::Scope");
    }
}

fn checks(events: &Observed) -> Vec<(String, u64, usize)> {
    events
        .iter()
        .filter_map(|(event, attention, _)| match event {
            QuantumEvent::Checked { label, scratch } => Some((label.clone(), *scratch, *attention)),
            QuantumEvent::Synchronized { .. } => None,
        })
        .collect()
}

/// The refusal contract of one route, from its passing run's events:
///
/// * zero headroom refuses the FIRST quantum, before any quantum or attention call ran;
/// * for EVERY later quantum that needs more than each quantum before it, a headroom affording
///   exactly those refuses THAT quantum: the events are the passing run's prefix up to its check,
///   and no attention ran inside or after it (its graph was never built).
///
/// Returns the labels refused mid-route, in route order.
fn assert_refusals<T>(
    passing: &Observed,
    run: impl Fn(u64) -> (Result<T>, Observed),
) -> Vec<String> {
    let checked = checks(passing);
    assert!(!checked.is_empty(), "the route ran no quantum");
    take_refusal();
    let (result, events) = run(0);
    assert!(result.is_err(), "zero headroom must refuse");
    let refusal = take_refusal().expect("the refusal is recorded");
    assert_eq!(refusal.label, checked[0].0, "the first quantum refuses");
    assert_eq!(refusal.scratch_bytes, checked[0].1);
    assert!(
        events.is_empty(),
        "nothing ran before the refusal: {events:?}"
    );
    assert!(diagnostics::observations().is_empty(), "no attention ran");

    let check_positions: Vec<usize> = passing
        .iter()
        .enumerate()
        .filter(|(_, (e, _, _))| matches!(e, QuantumEvent::Checked { .. }))
        .map(|(i, _)| i)
        .collect();
    let mut refused = Vec::new();
    let mut affordable = checked[0].1;
    for k in 1..checked.len() {
        if checked[k].1 <= affordable {
            continue;
        }
        let (result, events) = run(PROCESS_RESERVE_BYTES + affordable);
        assert!(result.is_err(), "{} must refuse", checked[k].0);
        let refusal = take_refusal().expect("the refusal is recorded");
        assert_eq!(refusal.label, checked[k].0);
        assert_eq!(refusal.scratch_bytes, checked[k].1);
        let prefix: Vec<&QuantumEvent> = passing[..check_positions[k]]
            .iter()
            .map(|(e, _, _)| e)
            .collect();
        let got: Vec<&QuantumEvent> = events.iter().map(|(e, _, _)| e).collect();
        assert_eq!(got, prefix, "only the quanta before {} ran", checked[k].0);
        assert_eq!(
            diagnostics::observations().len(),
            checked[k].2,
            "no attention ran inside or after the refused {}",
            checked[k].0
        );
        refused.push(checked[k].0.clone());
        affordable = checked[k].1;
    }
    refused
}

fn host(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    a.eval().unwrap();
    a.as_slice::<f32>().to_vec()
}

fn rand(shape: &[i32], seed: u64) -> Array {
    random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap()
}

fn tiny_backbone() -> (IrisDiT, IrisConfig) {
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let dit = crate::model::load_backbone(&backbone_dir(), &config, Dtype::Float32).unwrap();
    (dit, config)
}

/// A batch-2 (prompt-batch) forward over a 16x24 image: inputs and the text batch.
struct DitCase {
    x: Array,
    t: Array,
    states: Array,
    mask: Vec<Vec<i32>>,
}

impl DitCase {
    fn new(config: &IrisConfig) -> Self {
        let m = &config.model;
        let layers = config.text_encoder.hidden_layers.len() as i32;
        let mut mask = vec![vec![1; m.text_len]; 2];
        mask[1][m.text_len - 3..].fill(0);
        Self {
            x: rand(&[2, 3, 16, 24], 7),
            t: Array::from_slice(&[500.0f32, 250.0], &[2]),
            states: rand(&[2, m.text_len as i32, layers, m.text_dim as i32], 8),
            mask,
        }
    }

    fn forward(&self, dit: &IrisDiT, budget: AttentionBudget) -> Result<Array> {
        dit.forward_plan(
            &self.x,
            &self.t,
            &TextBatch {
                states: &self.states,
                mask: &self.mask,
            },
            AttentionPlan::budgeted(budget),
        )
    }
}

/// A budget that genuinely chunks the miniature attention (its rows split below `Sq`).
const CHUNKING: AttentionBudget = AttentionBudget::from_score_elements(300, true);

fn dit_forward_guarded_body() {
    let (dit, config) = tiny_backbone();
    let case = DitCase::new(&config);
    for budget in [AttentionBudget::UNBOUNDED, CHUNKING] {
        let unguarded = case.forward(&dit, budget).unwrap();
        let (guarded, events) = observed(AMPLE, || case.forward(&dit, budget));
        assert_scoped(&events);
        assert_eq!(
            host(&guarded.unwrap()),
            host(&unguarded),
            "guarded execution changed the backbone output ({budget:?})"
        );
        for (label, count) in [
            (dit_q::EMBED, 1),
            (dit_q::TEXT_LAYER_BLOCK, 2),
            (dit_q::TEXT_POOL, 1),
            (dit_q::TEXT_REFINER_BLOCK, 2),
            (dit_q::TEXT_OUTPUT, 1),
            (dit_q::TRUNK_BLOCK, dit.trunk_depth()),
            (dit_q::PIXEL_EMBED, 1),
            (dit_q::PIXEL_BLOCK, dit.pixel_depth()),
            (dit_q::OUTPUT, 1),
        ] {
            assert_paired(&events, label, count);
        }
        assert_eq!(
            checks(&events).len(),
            9 + dit.trunk_depth() + dit.pixel_depth()
        );
        let mid = assert_refusals(&events, |headroom| {
            observed(headroom, || case.forward(&dit, budget))
        });
        eprintln!("backbone ({budget:?}) refuses mid-forward at {mid:?}");
        assert!(
            mid.iter().any(|label| label == dit_q::TRUNK_BLOCK),
            "a trunk block must be refusable on its own: {mid:?}"
        );
    }
}

#[test]
fn backbone_blocks_are_guarded_byte_identical_and_refuse_before_construction() {
    in_cpu_child(
        "quantum_tests::backbone_blocks_are_guarded_byte_identical_and_refuse_before_construction",
        dit_forward_guarded_body,
    );
}

/// Σ elements of the checkpoint's tensors under `prefix` — the loaded block's weights, read
/// independently of the module tree.
fn checkpoint_elements(w: &Weights, prefix: &str) -> u64 {
    w.keys()
        .filter(|k| k.starts_with(prefix))
        .map(|k| w.require(k).unwrap().size() as u64)
        .sum()
}

fn checkpoint_rows(w: &Weights, key: &str) -> u64 {
    w.require(key).unwrap().shape()[0] as u64
}

fn scratch_geometry_body() {
    let (dit, config) = tiny_backbone();
    let m = &config.model;
    let w = Weights::from_file(backbone_dir().join("model.safetensors")).unwrap();
    let unbounded = AttentionPlan::UNBOUNDED;
    let tokens = |hp: i32, wp: i32| hp * wp + m.text_len as i32;
    for i in 0..dit.trunk_depth() {
        let mlp = if i < m.dual_depth { "mlp_x" } else { "mlp" };
        let expected = attention_block_bytes(AttentionBlockGeometry {
            batch: 2,
            tokens: tokens(4, 6) as u64,
            dim: m.hidden_size as u64,
            heads: m.num_heads as u64,
            context_tokens: 0,
            ff_hidden: checkpoint_rows(&w, &format!("blocks.{i}.{mlp}.w1.weight")),
            query_rows: tokens(4, 6) as u64,
            weight_elements: checkpoint_elements(&w, &format!("blocks.{i}.")),
        });
        assert_eq!(
            dit.trunk_block_scratch(i, 2, 24, m.text_len as i32, unbounded),
            expected,
            "trunk block {i} is priced from its loaded tensors"
        );
        // More tokens, more scratch; a chunked plan prices only its query rows.
        assert!(
            dit.trunk_block_scratch(i, 2, 64, m.text_len as i32, unbounded)
                > dit.trunk_block_scratch(i, 2, 24, m.text_len as i32, unbounded)
        );
        assert!(
            dit.trunk_block_scratch(
                i,
                2,
                64,
                m.text_len as i32,
                AttentionPlan::budgeted(CHUNKING)
            ) < dit.trunk_block_scratch(i, 2, 64, m.text_len as i32, unbounded)
        );
    }
    for i in 0..dit.pixel_depth() {
        let p = &format!("pixel_blocks.{i}.");
        let ppp = (m.patch_size * m.patch_size) as u64;
        let expected = attention_block_bytes(AttentionBlockGeometry {
            batch: 1,
            tokens: 24,
            dim: (ppp * m.pixel.hidden_size as u64)
                .max(checkpoint_rows(&w, &format!("{p}compress.weight"))),
            heads: m.pixel.num_heads as u64,
            context_tokens: 0,
            ff_hidden: ppp * checkpoint_rows(&w, &format!("{p}mlp.fc1.weight")),
            query_rows: 24,
            weight_elements: checkpoint_elements(&w, p),
        });
        assert_eq!(dit.pixel_block_scratch(i, 1, 24, unbounded), expected);
        assert!(dit.pixel_block_scratch(i, 1, 64, unbounded) > expected);
    }
    for i in 0..2 {
        let p = format!("y_embedder.layer_blocks.{i}.");
        let layers = config.text_encoder.hidden_layers.len() as u64;
        let expected = attention_block_bytes(AttentionBlockGeometry {
            batch: 2 * m.text_len as u64,
            tokens: layers,
            dim: m.text_dim as u64,
            heads: m.text_lap_num_heads as u64,
            context_tokens: 0,
            ff_hidden: checkpoint_rows(&w, &format!("{p}mlp.0.weight")),
            query_rows: layers,
            weight_elements: checkpoint_elements(&w, &p),
        });
        assert_eq!(
            dit.text_layer_block_scratch(i, 2 * m.text_len as i32, layers as i32, unbounded),
            expected
        );
        let p = format!("y_embedder.refiner.blocks.{i}.");
        let expected = attention_block_bytes(AttentionBlockGeometry {
            batch: 2,
            tokens: m.text_len as u64,
            dim: m.hidden_size as u64,
            heads: m.num_heads as u64,
            context_tokens: 0,
            ff_hidden: checkpoint_rows(&w, &format!("{p}mlp.w1.weight")),
            query_rows: m.text_len as u64,
            weight_elements: checkpoint_elements(&w, &p),
        });
        assert_eq!(
            dit.text_refiner_block_scratch(i, 2, m.text_len as i32, unbounded),
            expected
        );
    }
    assert!(dit.embed_scratch(1, 64) > dit.embed_scratch(1, 24));
}

#[test]
fn backbone_scratch_is_read_from_loaded_tensors_and_grows_with_geometry() {
    in_cpu_child(
        "quantum_tests::backbone_scratch_is_read_from_loaded_tensors_and_grows_with_geometry",
        scratch_geometry_body,
    );
}

fn text_encoder_body() {
    use crate::text_encoder::{quanta as te_q, IrisTextEncoder};
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let load = || IrisTextEncoder::load(&text_encoder_dir(), &config.text_encoder);
    let (loaded, load_events) = observed(AMPLE, load);
    let te = loaded.unwrap();
    assert_eq!(
        checks(&load_events)
            .iter()
            .map(|c| c.0.as_str())
            .collect::<Vec<_>>(),
        [te_q::LOAD]
    );
    take_refusal();
    let (refused, _) = observed(0, load);
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, te_q::LOAD);

    let prompt = "a red fox in the snow";
    let unguarded = te.encode(prompt).unwrap();
    let (guarded, events) = observed(AMPLE, || te.encode(prompt));
    assert_scoped(&events);
    let guarded = guarded.unwrap();
    assert_eq!(host(&guarded.states), host(&unguarded.states));
    assert_eq!(guarded.mask, unguarded.mask);
    let deepest = *config.text_encoder.hidden_layers.last().unwrap();
    assert_paired(&events, te_q::TOWER, 1);
    assert_paired(&events, te_q::EMBEDDING, 1);
    assert_paired(&events, te_q::LAYER, deepest);
    assert_paired(&events, te_q::CONDITIONING, 1);
    // The tower's check precedes every layer's; each layer's scratch holds one more state.
    assert!(is_checked(&events[0].0, te_q::TOWER));
    let layer_scratch: Vec<u64> = checks(&events)
        .into_iter()
        .filter(|c| c.0 == te_q::LAYER || c.0 == te_q::EMBEDDING)
        .map(|c| c.1)
        .collect();
    assert!(layer_scratch.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(checks(&events)[0].1, *layer_scratch.last().unwrap());
    let window = te.window(prompt).unwrap().input_ids.len() as i32;
    assert!(te.layer_scratch(2 * window, 0) > te.layer_scratch(window, 0));
    // Every decoder layer is ALSO checked from inside the shared stack before it is built, holding
    // one more state each time.
    let decoder: Vec<u64> = checks(&events)
        .into_iter()
        .filter(|c| c.0 == te_q::DECODER_LAYER)
        .map(|c| c.1)
        .collect();
    assert_eq!(decoder.len(), te.lm_config_layers());
    assert!(decoder.windows(2).all(|w| w[0] < w[1]));
    // Any mid-walk refusal is a decoder layer refused before its graph was built.
    let refused = assert_refusals(&events, |headroom| observed(headroom, || te.encode(prompt)));
    assert!(
        refused.iter().all(|label| label == te_q::DECODER_LAYER),
        "{refused:?}"
    );
}

#[test]
fn text_encoder_layers_are_guarded_byte_identical_and_refuse_before_construction() {
    in_cpu_child(
        "quantum_tests::text_encoder_layers_are_guarded_byte_identical_and_refuse_before_construction",
        text_encoder_body,
    );
}

fn depth_body() {
    use crate::depth::{load_depth_export, quanta as depth_q, DEPTH_MODEL_ID};
    let export = || {
        TaskExport::from_dir(
            &fixtures().join("tiny-depth"),
            IrisTask::Depth,
            DEPTH_MODEL_ID,
        )
    };
    let (loaded, load_events) = observed(AMPLE, || {
        load_depth_export(export().unwrap(), Dtype::Float32)
    });
    let depth = loaded.unwrap();
    assert_eq!(
        checks(&load_events)
            .iter()
            .map(|c| c.0.as_str())
            .collect::<Vec<_>>(),
        [depth_q::LOAD]
    );
    take_refusal();
    let (refused, _) = observed(0, || load_depth_export(export().unwrap(), Dtype::Float32));
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, depth_q::LOAD);

    let rgb = rand(&[1, 3, 16, 24], 11);
    let unguarded = depth.forward(&rgb).unwrap();
    let (guarded, events) = observed(AMPLE, || depth.forward(&rgb));
    assert_eq!(host(&guarded.unwrap()), host(&unguarded));
    assert_paired(&events, depth_q::INPUT, 1);
    assert_paired(&events, depth_q::REDUCER, 1);
    assert_paired(&events, dit_q::TRUNK_BLOCK, 3);
    assert_paired(&events, dit_q::PIXEL_BLOCK, 2);
    assert!(is_checked(&events[0].0, depth_q::INPUT));
    let reducer = checks(&events).len() - 1;
    assert_eq!(checks(&events)[reducer].0, depth_q::REDUCER);
    let mid = assert_refusals(&events, |headroom| {
        observed(headroom, || depth.forward(&rgb))
    });
    eprintln!("depth refuses mid-forward at {mid:?}");
    assert!(!mid.is_empty());

    // The estimator entry point (its own scope) is byte-identical too.
    let image = Image {
        width: 20,
        height: 12,
        pixels: (0..20 * 12 * 3).map(|i| (i * 37 % 256) as u8).collect(),
    };
    let request = DepthRequest {
        image,
        resolution: DepthResolution::Native,
        ..Default::default()
    };
    let unguarded = depth.estimate(&request, &mut |_| {}).unwrap();
    let (guarded, events) = observed(AMPLE, || {
        depth
            .estimate(&request, &mut |_| {})
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))
    });
    assert_eq!(guarded.unwrap().map.values, unguarded.map.values);
    assert_scoped(&events);
    assert_paired(&events, depth_q::REDUCER, 1);
}

#[test]
fn depth_forward_is_guarded_byte_identical_and_refuses_before_construction() {
    in_cpu_child(
        "quantum_tests::depth_forward_is_guarded_byte_identical_and_refuses_before_construction",
        depth_body,
    );
}

fn restoration_body() {
    use crate::restoration::{quanta as restore_q, IrisRestorer};
    let mut spec = LoadSpec::new(WeightsSource::Dir(
        fixtures().join("tiny-snapshot/upscaler"),
    ));
    spec.precision = Precision::Fp32;
    let (loaded, load_events) = observed(AMPLE, || IrisRestorer::load(&spec));
    let restorer = loaded.unwrap();
    assert_eq!(
        checks(&load_events)
            .iter()
            .map(|c| c.0.as_str())
            .collect::<Vec<_>>(),
        [restore_q::LOAD, crate::model::BACKBONE_LOAD_QUANTUM]
    );
    take_refusal();
    let (refused, _) = observed(0, || IrisRestorer::load(&spec));
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, restore_q::LOAD);

    let request = TransformRequest {
        image: Image {
            width: 10,
            height: 6,
            pixels: (0..10 * 6 * 3).map(|i| (i * 53 % 256) as u8).collect(),
        },
        target: TargetSize::Scale(4.0),
        input_sizing: InputSizing::Original,
        ..Default::default()
    };
    let tiles = restorer.plan(&request).unwrap().forward_count();
    assert!(tiles > 1, "the fixture must genuinely tile");
    let apply = || {
        restorer
            .apply(&request, &mut |_| {})
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))
    };
    let unguarded = apply().unwrap();
    let (guarded, events) = observed(AMPLE, apply);
    assert_scoped(&events);
    assert_eq!(guarded.unwrap().pixels, unguarded.pixels);
    assert_paired(&events, restore_q::TILE, tiles);
    assert_paired(&events, dit_q::TRUNK_BLOCK, 3 * tiles);
    assert_paired(&events, dit_q::PIXEL_BLOCK, 2 * tiles);
    assert!(is_checked(&events[0].0, restore_q::TILE));
    let mid = assert_refusals(&events, |headroom| observed(headroom, apply));
    eprintln!("restoration refuses mid-apply at {mid:?}");
    assert!(!mid.is_empty());
}

#[test]
fn restoration_tiles_are_guarded_byte_identical_and_refuse_before_construction() {
    in_cpu_child(
        "quantum_tests::restoration_tiles_are_guarded_byte_identical_and_refuse_before_construction",
        restoration_body,
    );
}

fn generation_body() {
    use crate::text_encoder::quanta as te_q;
    let mut spec = LoadSpec::new(WeightsSource::Dir(backbone_dir()));
    spec.precision = Precision::Fp32;
    spec.components.insert(
        TEXT_ENCODER_COMPONENT.into(),
        WeightsSource::Dir(text_encoder_dir()),
    );
    let (loaded, load_events) = observed(AMPLE, || crate::model::load(&spec));
    let generator = loaded.unwrap();
    let mut loads: Vec<String> = checks(&load_events).into_iter().map(|c| c.0).collect();
    loads.sort();
    assert_eq!(loads, [crate::model::BACKBONE_LOAD_QUANTUM, te_q::LOAD]);
    take_refusal();
    let (refused, _) = observed(0, || crate::model::load(&spec));
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, checks(&load_events)[0].0);

    let steps = 2;
    let request = GenerationRequest {
        prompt: String::new(),
        prompt_batch: vec!["a red fox in the snow".into(), "golden hour".into()],
        width: 32,
        height: 16,
        steps: Some(steps),
        seed: Some(42),
        ..Default::default()
    };
    let generate = || {
        generator
            .generate(&request, &mut |_| {})
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))
    };
    let pixels = |output: GenerationOutput| match output {
        GenerationOutput::Images(images) => images.into_iter().map(|i| i.pixels).collect(),
        _ => panic!("expected images"),
    };
    let unguarded: Vec<Vec<u8>> = pixels(generate().unwrap());
    let (guarded, events) = observed(AMPLE, generate);
    assert_scoped(&events);
    assert_eq!(pixels(guarded.unwrap()), unguarded);
    assert_eq!(unguarded.len(), 2, "one image per prompt");
    // Two prompts and the CFG negative are encoded; each step is one batched CFG forward.
    assert_paired(&events, te_q::TOWER, 3);
    assert_paired(&events, crate::solver::STEP_QUANTUM, steps as usize);
    assert_paired(&events, dit_q::EMBED, steps as usize);
    assert_paired(&events, dit_q::TRUNK_BLOCK, 3 * steps as usize);
    assert_paired(&events, crate::model::IMAGE_QUANTUM, 1);
    assert!(is_checked(&events[0].0, te_q::TOWER));
    let mid = assert_refusals(&events, |headroom| observed(headroom, generate));
    eprintln!("generation refuses mid-request at {mid:?}");
}

#[test]
fn generation_is_guarded_byte_identical_and_refuses_before_construction() {
    in_cpu_child(
        "quantum_tests::generation_is_guarded_byte_identical_and_refuses_before_construction",
        generation_body,
    );
}
