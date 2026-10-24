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
    take_refusal, with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
    PROCESS_RESERVE_BYTES,
};
use mlx_gen::weights::Weights;
use mlx_gen::{
    AttentionBudget, AttentionPlan, GenerationOutput, GenerationRequest, Image, InputSizing,
    LoadSpec, Precision, Result, TargetSize, Transform, TransformRequest, WeightsSource,
};
use mlx_rs::{random, Array, Dtype};

use crate::dit::{quanta as dit_q, IrisDiT, TextBatch};
use crate::nn::diagnostics;

use crate::test_metal_guard as metal_guard;

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
        assert_eq!(
            mlx_rs::Device::try_default().expect("CPU default-device query failed"),
            mlx_rs::Device::cpu()
        );
        assert!(metal_guard::forbid_metal());
        assert!(metal_guard::metal_forbidden());
        // Select the native CPU stream before constructing any fixture, not just in scopes.
        assert_eq!(
            unsafe { mlx_sys::mlx_set_default_stream(mlx_rs::Stream::cpu().as_ptr()) },
            0
        );
        assert_eq!(
            mlx_rs::Stream::new().get_index().unwrap(),
            mlx_rs::Stream::cpu().get_index().unwrap()
        );
        body();
        std::fs::write(std::env::var(DONE).unwrap(), b"ok").unwrap();
        return;
    }
    let done = tempfile::tempdir().unwrap();
    let marker = done.path().join("done");
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--include-ignored",
            "--exact",
            test,
            "--nocapture",
            "--test-threads",
            "1",
        ])
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
        let expected = [
            (dit_q::EMBED, 1),
            (dit_q::TEXT_LAYER_BLOCK, 2),
            (dit_q::TEXT_POOL, 1),
            (dit_q::TEXT_MASK, 1),
            (dit_q::TEXT_REFINER_BLOCK, 2),
            (dit_q::TEXT_OUTPUT, 1),
            (dit_q::TRUNK_POSITIONS, 1),
            (dit_q::TRUNK_BLOCK, dit.trunk_depth()),
            (dit_q::PIXEL_EMBED, 1),
            (dit_q::PIXEL_POSITIONS, 1),
            (dit_q::PIXEL_BLOCK, dit.pixel_depth()),
            (dit_q::OUTPUT, 1),
        ];
        for &(label, count) in &expected {
            assert_paired(&events, label, count);
        }
        assert_eq!(
            checks(&events).len(),
            expected.iter().map(|(_, count)| count).sum::<usize>()
        );
        // The embed guard carries the previous lazy guidance/solver allowance. Positional
        // constructors have their own EARLY quantum, not a late allowance on the first trunk.
        let m = &config.model;
        let p = m.patch_size as i32;
        let (b, n_patches, n_txt) = (2, (16 / p) * (24 / p), m.text_len as i32);
        let plan = AttentionPlan::budgeted(budget);
        let head_dim = (m.hidden_size / m.num_heads) as u64;
        let of = |label: &str| -> Vec<u64> {
            checks(&events)
                .into_iter()
                .filter(|c| c.0 == label)
                .map(|c| c.1)
                .collect()
        };
        assert_eq!(
            of(dit_q::EMBED),
            [dit.embed_scratch(b, n_patches) + 16 * case.x.size() as u64 * 4]
        );
        let entry = 4 * n_patches as u64 * head_dim * 4
            + 4 * n_txt as u64 * head_dim * 4
            + 3 * (b * n_txt) as u64 * m.hidden_size as u64 * 4;
        assert_eq!(
            of(dit_q::TRUNK_POSITIONS),
            [entry],
            "positions are checked before their constructors"
        );
        let trunk = of(dit_q::TRUNK_BLOCK);
        for (i, scratch) in trunk.iter().enumerate() {
            let floor = dit.trunk_block_scratch(i, b, n_patches, n_txt, plan);
            assert_eq!(
                *scratch, floor,
                "trunk {i}: positional allocations already completed in their own quantum"
            );
        }
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

fn allocation_region_body() {
    use mlx_gen::ios_quantum::{set_quantum_listener, QuantumMark};
    use std::sync::{Arc, Mutex};
    let (dit, config) = tiny_backbone();
    let case = DitCase::new(&config);
    for (role, label) in [
        (0, dit_q::TEXT_MASK),
        (1, dit_q::TRUNK_POSITIONS),
        (2, dit_q::PIXEL_POSITIONS),
    ] {
        let baseline = Arc::new(Mutex::new(None));
        let sink = baseline.clone();
        assert!(!set_quantum_listener(Some(Box::new(move |mark| {
            if let QuantumMark::Metadata {
                provider,
                kind,
                fields,
            } = mark
            {
                if provider == crate::model::MODEL_ID && kind == "allocation_region" {
                    let actual = fields
                        .iter()
                        .find(|(key, _)| *key == "role")
                        .expect("actual allocation role missing")
                        .1;
                    if actual == role {
                        let mut active = 0;
                        assert_eq!(
                            unsafe { mlx_sys::mlx_get_active_memory(&mut active) },
                            0,
                            "native active-memory query failed"
                        );
                        assert!(
                            active > 0,
                            "the witness must own real materialized fixtures"
                        );
                        assert_eq!(
                            unsafe { mlx_sys::mlx_reset_peak_memory() },
                            0,
                            "native peak reset failed"
                        );
                        *sink.lock().expect("allocation witness poisoned") = Some(active);
                    }
                }
            }
        }))));
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                set_quantum_listener(None);
            }
        }
        let restore = Restore;
        take_refusal();
        let (result, events) = observed(AMPLE, || {
            mlx_gen::ios_quantum::with_cpu_diagnostic_short_at(label, || {
                case.forward(&dit, AttentionBudget::UNBOUNDED)
            })
        });
        let mut peak = 0;
        assert_eq!(
            unsafe { mlx_sys::mlx_get_peak_memory(&mut peak) },
            0,
            "native peak query failed"
        );
        drop(restore);
        assert!(
            result.is_err(),
            "{label} must refuse at its own early guard"
        );
        assert_eq!(
            take_refusal().expect("constructor refusal recorded").label,
            label
        );
        let before = baseline
            .lock()
            .unwrap()
            .expect("actual constructor region was never entered");
        assert!(before > 0);
        // Native reset_peak resets to zero, not active memory. With no later allocation it stays
        // zero; any constructor allocation updates it to the then-active total, even if freed.
        assert_eq!(
            peak, 0,
            "{label}: native constructor allocation occurred before refusal"
        );
        assert!(
            !events.iter().any(|(event, _, _)| is_checked(event, label)),
            "a refused constructor must not pass its check"
        );
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn allocation_regions_refuse_before_native_constructor_buffers() {
    in_cpu_child(
        "quantum_tests::allocation_regions_refuse_before_native_constructor_buffers",
        || {
            allocation_region_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::allocation_regions_refuse_before_native_constructor_buffers");
        },
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn backbone_blocks_are_guarded_byte_identical_and_refuse_before_construction() {
    in_cpu_child(
        "quantum_tests::backbone_blocks_are_guarded_byte_identical_and_refuse_before_construction",
        || {
            dit_forward_guarded_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::backbone_blocks_are_guarded_byte_identical_and_refuse_before_construction");
        },
    );
}

fn checkpoint_rows(w: &Weights, key: &str) -> u64 {
    w.require(key).unwrap().shape()[0] as u64
}

fn scratch_geometry_body() {
    use crate::dit::{TextBlockGeometry, TrunkGeometry};
    let (dit, config) = tiny_backbone();
    let m = &config.model;
    let w = Weights::from_file(backbone_dir().join("model.safetensors")).unwrap();
    let unbounded = AttentionPlan::UNBOUNDED;
    let unfused = crate::retention::unfused();
    let kv_heads = m.num_kv_heads.unwrap() as u64;
    for i in 0..dit.trunk_depth() {
        let dual = i < m.dual_depth;
        let mlp = if dual { "mlp_x" } else { "mlp" };
        // The price is the block's counts over the CHECKPOINT's widths and the request geometry.
        let g = TrunkGeometry {
            compute_bytes: 4,
            batch: 2,
            img_tokens: 24,
            txt_tokens: m.text_len as u64,
            dim: m.hidden_size as u64,
            heads: m.num_heads as u64,
            kv_heads,
            ff: checkpoint_rows(&w, &format!("blocks.{i}.{mlp}.w1.weight")),
            query_rows: 24 + m.text_len as u64,
            eval_per_chunk: false,
            split_qkv: kv_heads != m.num_heads as u64,
            text: true,
            down_pad: 0,
            // f32 compute: the first block's compute-width image stream is f32 as well.
            img_stream_bytes: 4,
        };
        let expected = if dual {
            IrisDiT::dual_block_bytes(&g, unfused)
        } else {
            IrisDiT::single_block_bytes(&g, unfused)
        };
        assert_eq!(
            dit.trunk_block_scratch(i, 2, 24, m.text_len as i32, unbounded),
            expected,
            "trunk block {i}"
        );
        // More tokens, more scratch.
        assert!(
            dit.trunk_block_scratch(i, 2, 64, m.text_len as i32, unbounded)
                > dit.trunk_block_scratch(i, 2, 24, m.text_len as i32, unbounded)
        );
    }
    for i in 0..dit.pixel_depth() {
        assert!(
            dit.pixel_block_scratch(i, 1, 64, unbounded)
                > dit.pixel_block_scratch(i, 1, 24, unbounded)
        );
    }
    let layers = config.text_encoder.hidden_layers.len() as u64;
    for i in 0..2 {
        let expected = IrisDiT::text_block_bytes(
            TextBlockGeometry {
                compute_bytes: 4,
                batch: 2 * m.text_len as u64,
                tokens: layers,
                dim: m.text_dim as u64,
                heads: m.text_lap_num_heads as u64,
                ff: checkpoint_rows(&w, &format!("y_embedder.layer_blocks.{i}.mlp.0.weight")),
                query_rows: layers,
                eval_per_chunk: false,
                refiner: false,
                down_pad: 0,
                stream_bytes: 4,
            },
            unfused,
        );
        assert_eq!(
            dit.text_layer_block_scratch(i, 2 * m.text_len as i32, layers as i32, unbounded),
            expected
        );
        let t = m.text_len as u64;
        let expected = IrisDiT::text_block_bytes(
            TextBlockGeometry {
                compute_bytes: 4,
                batch: 2,
                tokens: t,
                dim: m.hidden_size as u64,
                heads: m.num_heads as u64,
                ff: checkpoint_rows(&w, &format!("y_embedder.refiner.blocks.{i}.mlp.w1.weight")),
                query_rows: t,
                eval_per_chunk: false,
                refiner: true,
                down_pad: 0,
                stream_bytes: 4,
            },
            unfused,
        );
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
        [te_q::LOAD, crate::stream::LOAD_TARGET_QUANTUM]
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
#[ignore = "model inference; explicit model-proof workflow"]
fn text_encoder_layers_are_guarded_byte_identical_and_refuse_before_construction() {
    in_cpu_child(
        "quantum_tests::text_encoder_layers_are_guarded_byte_identical_and_refuse_before_construction", || { text_encoder_body(); eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::text_encoder_layers_are_guarded_byte_identical_and_refuse_before_construction"); },
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
        [
            depth_q::LOAD,
            crate::stream::LOAD_TARGET_QUANTUM,
            crate::stream::EXPORT_TARGET_QUANTUM
        ]
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
#[ignore = "model inference; explicit model-proof workflow"]
fn depth_forward_is_guarded_byte_identical_and_refuses_before_construction() {
    in_cpu_child(
        "quantum_tests::depth_forward_is_guarded_byte_identical_and_refuses_before_construction",
        || {
            depth_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::depth_forward_is_guarded_byte_identical_and_refuses_before_construction");
        },
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
        [
            restore_q::LOAD,
            crate::stream::LOAD_TARGET_QUANTUM,
            crate::model::BACKBONE_LOAD_QUANTUM,
            crate::stream::LOAD_TARGET_QUANTUM,
            crate::stream::EXPORT_TARGET_QUANTUM
        ]
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
    // The driver's host planes are checked before the first tile, and the post-tile planes once
    // the last tile has returned (after its synchronization, before the driver allocates them).
    let plan = restorer.plan(&request).unwrap();
    let host = crate::restoration::host_plane_bytes(&plan).unwrap();
    let checked = checks(&events);
    assert_eq!(
        checked[0],
        (
            restore_q::PLAN.to_string(),
            plan.metadata().axis_bytes().unwrap(),
            0
        )
    );
    let host_position = checked
        .iter()
        .position(|event| event.0 == restore_q::HOST_PLANES)
        .expect("host planes checked");
    assert_eq!(
        checked[host_position],
        (restore_q::HOST_PLANES.to_string(), host.peak(), 0)
    );
    let first_tile = checked
        .iter()
        .position(|event| event.0 == restore_q::TILE)
        .expect("tile checked");
    assert!(
        host_position < first_tile,
        "host construction is guarded before any tile"
    );
    let last = checks(&events).last().cloned().unwrap();
    assert_eq!(
        (last.0.as_str(), last.1),
        (restore_q::POST_TILE, host.post_tile_new())
    );
    let mid = assert_refusals(&events, |headroom| observed(headroom, apply));
    eprintln!("restoration refuses mid-apply at {mid:?}");
    assert!(!mid.is_empty());
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn restoration_tiles_are_guarded_byte_identical_and_refuse_before_construction() {
    in_cpu_child(
        "quantum_tests::restoration_tiles_are_guarded_byte_identical_and_refuse_before_construction", || { restoration_body(); eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::restoration_tiles_are_guarded_byte_identical_and_refuse_before_construction"); },
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
    let loads: Vec<String> = checks(&load_events).into_iter().map(|c| c.0).collect();
    assert_eq!(
        loads,
        [
            te_q::LOAD,
            crate::stream::LOAD_TARGET_QUANTUM,
            crate::model::BACKBONE_LOAD_QUANTUM,
            crate::stream::LOAD_TARGET_QUANTUM
        ]
    );
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
    assert_paired(&events, crate::solver::FINALIZE_QUANTUM, 1);
    assert_paired(&events, dit_q::EMBED, steps as usize);
    assert_paired(&events, dit_q::TRUNK_BLOCK, 3 * steps as usize);
    assert_paired(&events, crate::model::IMAGE_QUANTUM, 1);
    assert!(is_checked(&events[0].0, te_q::TOWER));
    let mid = assert_refusals(&events, |headroom| observed(headroom, generate));
    eprintln!("generation refuses mid-request at {mid:?}");

    // Refuse the final clamp of the f32 solver state [prompts, 3, H, W] specifically.
    // Its price need not exceed earlier quanta, so the monotone-headroom walk above is not proof.
    let final_label = crate::solver::FINALIZE_QUANTUM;
    let final_position = events
        .iter()
        .position(|(event, _, _)| is_checked(event, final_label))
        .expect("the passing route checked its final clamp");
    let final_scratch = checks(&events)
        .into_iter()
        .find(|(label, _, _)| label == final_label)
        .unwrap()
        .1;
    assert_eq!(
        crate::ios_pricing::ios_quantum_phase(final_label),
        Some(mlx_gen::ios_pricing::IosPhase::Denoise)
    );
    assert!(events[final_position + 1..]
        .iter()
        .any(|(event, _, _)| is_checked(event, crate::model::IMAGE_QUANTUM)));
    take_refusal();
    let (refused, refused_events) = observed(AMPLE, || {
        mlx_gen::ios_quantum::with_cpu_diagnostic_short_at(final_label, generate)
    });
    assert!(refused.is_err(), "the final-clamp guard must refuse");
    let refusal = take_refusal().expect("final-clamp refusal must be recorded");
    assert_eq!(refusal.label, final_label);
    assert_eq!(refusal.scratch_bytes, final_scratch);
    let prefix: Vec<_> = events[..final_position].iter().map(|e| &e.0).collect();
    let got: Vec<_> = refused_events.iter().map(|e| &e.0).collect();
    assert_eq!(got, prefix, "only quanta before the final clamp ran");
    assert_eq!(diagnostics::observations().len(), events[final_position].1);
    assert!(
        refused_events.iter().all(|(event, _, _)| {
            let label = match event {
                QuantumEvent::Checked { label, .. } | QuantumEvent::Synchronized { label, .. } => {
                    label
                }
            };
            crate::ios_pricing::ios_quantum_phase(label)
                != Some(mlx_gen::ios_pricing::IosPhase::Decode)
        }),
        "Decode must not be entered after a final-clamp refusal"
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn generation_is_guarded_byte_identical_and_refuses_before_construction() {
    in_cpu_child(
        "quantum_tests::generation_is_guarded_byte_identical_and_refuses_before_construction",
        || {
            generation_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::generation_is_guarded_byte_identical_and_refuses_before_construction");
        },
    );
}

/// Write the miniature tower as a two-shard snapshot shaped like the release's: shard 1 holds every
/// language-tower tensor AND some vision tensors; shard 2 holds only vision tensors and is NOT a
/// readable safetensors file at all, so opening it would fail the load. Returns
/// `(dir, language-tower bytes and largest tensor in shard 1)`.
fn sharded_text_encoder_with_a_vision_shard(root: &std::path::Path) -> (PathBuf, u64, u64) {
    let source = text_encoder_dir();
    let dir = root.join("text_encoder");
    std::fs::create_dir_all(&dir).unwrap();
    for entry in std::fs::read_dir(&source).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_owned();
        if name != "model.safetensors" {
            std::fs::copy(&path, dir.join(name)).unwrap();
        }
    }
    let tensors = Array::load_safetensors(source.join("model.safetensors")).unwrap();
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let (mut first, mut map) = (Vec::new(), serde_json::Map::new());
    let (mut tower_bytes, mut largest) = (0u64, 0u64);
    let mut vision_seen = 0;
    for name in names {
        let array = &tensors[name];
        let shard = if name.starts_with("model.language_model.") {
            array.eval().unwrap();
            let bytes = array.nbytes() as u64;
            tower_bytes += bytes;
            largest = largest.max(bytes);
            "model-00001-of-00002.safetensors"
        } else {
            vision_seen += 1;
            if vision_seen % 2 == 0 {
                "model-00001-of-00002.safetensors"
            } else {
                "model-00002-of-00002.safetensors"
            }
        };
        if shard.starts_with("model-00001") {
            first.push((name.as_str(), array));
        }
        map.insert(name.clone(), shard.into());
    }
    assert!(vision_seen >= 2, "the fixture must carry vision tensors");
    Array::save_safetensors(
        first,
        None::<&std::collections::HashMap<String, String>>,
        dir.join("model-00001-of-00002.safetensors"),
    )
    .unwrap();
    std::fs::write(
        dir.join("model-00002-of-00002.safetensors"),
        b"not a safetensors file: the vision tower must never be opened",
    )
    .unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::json!({ "metadata": {}, "weight_map": map }).to_string(),
    )
    .unwrap();
    (dir, tower_bytes, largest)
}

fn text_only_shards_body() {
    use crate::text_encoder::{quanta as te_q, IrisTextEncoder};
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let (dir, tower_bytes, _largest) = sharded_text_encoder_with_a_vision_shard(root.path());
    // Only the shard that holds tower tensors is opened: the vision-only shard is not even a
    // readable file, and the load succeeds.
    let (loaded, events) = observed(AMPLE, || IrisTextEncoder::load(&dir, &config.text_encoder));
    let te = loaded.unwrap();
    // The load floor prices the tower's tensors only — not the vision tensors that share its
    // shard, not the vision-only shard.
    let model_cfg = mlx_llm::ModelConfig::from_dir(&dir).unwrap();
    let mut selected = Vec::new();
    for file in crate::text_encoder::weight_shards(&dir, !model_cfg.tie_word_embeddings).unwrap() {
        selected.extend(
            crate::stream::headers(&file)
                .unwrap()
                .into_iter()
                .filter(|h| {
                    crate::text_encoder::is_text_tower_tensor(
                        &h.name,
                        !model_cfg.tie_word_embeddings,
                    )
                }),
        );
    }
    let (logical, additional) = crate::stream::tower_load_output_bytes(
        &selected,
        |_| true,
        crate::stream::tower_rope_bytes(&model_cfg).unwrap(),
    )
    .unwrap();
    assert_eq!(
        logical, tower_bytes,
        "original raw tower tuple excludes every vision tensor"
    );
    assert_eq!(
        checks(&events),
        [
            (te_q::LOAD.to_string(), tower_bytes, 0),
            (
                crate::stream::LOAD_TARGET_QUANTUM.to_string(),
                logical + additional,
                0
            ),
        ]
    );
    // Host output is the unsharded fixture's, bit for bit.
    let reference = IrisTextEncoder::load(&text_encoder_dir(), &config.text_encoder).unwrap();
    for prompt in ["a red fox in the snow", ""] {
        let (a, b) = (
            te.encode(prompt).unwrap(),
            reference.encode(prompt).unwrap(),
        );
        assert_eq!(host(&a.states), host(&b.states));
        assert_eq!(a.mask, b.mask);
    }
    // The memory contract's text-encoder pricing reads the same file list (it would fail on the
    // unreadable vision shard) and prices the tower's tensors only.
    let shards = crate::text_encoder::weight_shards(&dir, false).unwrap();
    assert_eq!(shards, [dir.join("model-00001-of-00002.safetensors")]);
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn text_only_loads_neither_read_nor_price_the_vision_tower() {
    in_cpu_child(
        "quantum_tests::text_only_loads_neither_read_nor_price_the_vision_tower",
        || {
            text_only_shards_body();
            eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::text_only_loads_neither_read_nor_price_the_vision_tower");
        },
    );
}

fn adapter_read_body() {
    use mlx_gen::{AdapterKind, AdapterSpec};
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let file = fixtures().join("iris_lora.safetensors");
    let specs = [AdapterSpec::new(file.clone(), 0.75, AdapterKind::Lora)];
    let load = || {
        crate::model::load_backbone_with_adapters(
            &backbone_dir(),
            &config,
            Dtype::Float32,
            &specs,
            IrisTask::Generation,
            crate::model::MODEL_ID,
        )
    };
    let unguarded = load().unwrap();
    let (guarded, events) = observed(AMPLE, load);
    let guarded = guarded.unwrap();
    assert_eq!(guarded.1.len(), unguarded.1.len());
    // The adapter file is priced from its headers (its stored bytes), after the
    // backbone's own floor and before the file is first opened.
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file).unwrap();
    let stored: u64 = headers.iter().map(|h| h.data_bytes).sum();
    let labels: Vec<(String, u64)> = checks(&events).into_iter().map(|c| (c.0, c.1)).collect();
    let base_headers = crate::stream::headers(&backbone_dir().join("model.safetensors")).unwrap();
    let raw = crate::stream::materialized_load_bytes(&base_headers, |_| true).unwrap();
    let view = base_headers.iter().map(|h| (h.name.as_str(), h)).collect();
    let joint = crate::stream::backbone_load_target_bytes(&view, Dtype::Float32, |_| true).unwrap();
    assert_eq!(
        labels[0],
        (crate::model::BACKBONE_LOAD_QUANTUM.to_string(), raw)
    );
    assert_eq!(
        labels[1],
        (crate::stream::LOAD_TARGET_QUANTUM.to_string(), joint)
    );
    assert_eq!(
        labels[2],
        (crate::adapters::ADAPTER_LOAD_QUANTUM.to_string(), stored)
    );
    // With no headroom the install refuses at that check, before reading or installing anything.
    take_refusal();
    let mut weights = Weights::from_file(backbone_dir().join("model.safetensors")).unwrap();
    let (refused, _) = observed(0, || {
        crate::adapters::install(
            &mut weights,
            Dtype::Float32,
            &specs,
            IrisTask::Generation,
            crate::model::MODEL_ID,
        )
    });
    assert!(refused.is_err());
    assert_eq!(
        take_refusal().unwrap().label,
        crate::adapters::ADAPTER_LOAD_QUANTUM
    );
}

#[test]
fn adapter_files_are_priced_before_they_are_read() {
    in_cpu_child(
        "quantum_tests::adapter_files_are_priced_before_they_are_read",
        adapter_read_body,
    );
}

fn packed_adapter_host_guard_case(dtype: Dtype, marker_shape: &[i32]) {
    use mlx_gen::{AdapterKind, AdapterSpec};
    let mut weights = Weights::from_file(backbone_dir().join("model.safetensors")).unwrap();
    weights.materialize().unwrap();
    let key = weights
        .keys()
        .find(|name| {
            name.ends_with(".weight")
                && weights
                    .peek(name)
                    .is_some_and(|a| a.ndim() == 2 && a.shape()[1] >= 64 && a.shape()[1] % 64 == 0)
        })
        .unwrap()
        .to_owned();
    let shape = weights.peek(&key).unwrap().shape().to_vec();
    let (out, input) = (shape[0] as usize, shape[1] as usize);
    let base = key.strip_suffix(".weight").unwrap();
    let codes = Array::from_slice(
        &vec![0u32; out * input / 4],
        &[out as i32, (input / 4) as i32],
    );
    let scales = Array::from_slice(
        &vec![1f32; out * input / 64],
        &[out as i32, (input / 64) as i32],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let biases = Array::from_slice(
        &vec![0f32; out * input / 64],
        &[out as i32, (input / 64) as i32],
    )
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    let marker = |value: usize| match dtype {
        Dtype::Int64 => Array::from_slice(&[value as i64], marker_shape),
        Dtype::Float32 => Array::from_slice(&[value as f32], marker_shape),
        Dtype::Float64 => Array::from_slice_f64(&[value as f64], marker_shape),
        Dtype::Int32 => Array::from_slice(&[value as i32], marker_shape),
        _ => unreachable!("the test's explicit valid marker formats"),
    };
    let left = marker(input);
    let right = marker(0);
    for array in [&codes, &scales, &biases, &left, &right] {
        array.eval().unwrap();
    }
    let width = mlx_rs::ops::add(&left, &right).unwrap(); // deliberately unevaluated I32 scalar
    weights.insert(key.clone(), codes);
    weights.insert(format!("{base}.scales"), scales);
    weights.insert(format!("{base}.biases"), biases);
    weights.insert(
        format!("{base}{}", crate::packed::IN_FEATURES_SUFFIX),
        width.clone(),
    );
    let specs = [AdapterSpec::new(
        fixtures().join("iris_lora.safetensors"),
        0.75,
        AdapterKind::Lora,
    )];
    let baseline = mlx_rs::memory::get_active_memory();
    take_refusal();
    let (denied, _) = observed(0, || {
        crate::adapters::install(
            &mut weights,
            Dtype::Float32,
            &specs,
            IrisTask::Generation,
            crate::model::MODEL_ID,
        )
    });
    assert!(denied.is_err());
    assert_eq!(
        take_refusal().unwrap().label,
        crate::adapters::ADAPTER_WIDTH_QUANTUM
    );
    assert_eq!(
        mlx_rs::memory::get_active_memory(),
        baseline,
        "denial must precede width.try_item, not merely the later placeholder constructor"
    );

    let active = Rc::new(RefCell::new(Vec::new()));
    let sink = active.clone();
    let installed = with_cpu_diagnostic_headroom(AMPLE, || with_cpu_diagnostic_observer(
        move |event| {
            if matches!(event, QuantumEvent::Checked { label, .. } if label == crate::adapters::ADAPTER_HOST_QUANTUM || label == crate::adapters::ADAPTER_WIDTH_QUANTUM) {
                sink.borrow_mut().push(mlx_rs::memory::get_active_memory() - baseline);
            }
        },
        || crate::adapters::install(&mut weights, Dtype::Float32,
            &specs, IrisTask::Generation, crate::model::MODEL_ID),
    )).unwrap();
    assert!(installed.is_some());
    if dtype == Dtype::Int32 {
        assert_eq!(
            &*active.borrow(),
            &[0, std::mem::size_of::<i32>()],
            "the original I32 fixture still checks before width allocation and before placeholder"
        );
    } else {
        // A converted temporary may live until the backend releases it; do not assert a guessed
        // cast-retirement instant. The no-evaluation denial above remains exact for every dtype.
        assert_eq!(active.borrow().len(), 2);
        assert_eq!(active.borrow()[0], 0);
    }
    drop(installed);
    take_refusal();
    let (denied, events) = observed(AMPLE, || {
        mlx_gen::ios_quantum::with_cpu_diagnostic_short_at(
            crate::adapters::ADAPTER_HOST_QUANTUM,
            || {
                crate::adapters::install(
                    &mut weights,
                    Dtype::Float32,
                    &specs,
                    IrisTask::Generation,
                    crate::model::MODEL_ID,
                )
            },
        )
    });
    assert!(denied.is_err());
    assert_eq!(
        take_refusal().unwrap().label,
        crate::adapters::ADAPTER_HOST_QUANTUM
    );
    let checked = checks(&events);
    assert_eq!(
        checked.len(),
        1,
        "the placeholder refusal must prevent all adapter file reads"
    );
    let source = width.item_size() as u64;
    let cast = if dtype == Dtype::Int32 { 0 } else { 4 };
    assert_eq!(
        (checked[0].0.as_str(), checked[0].1),
        (crate::adapters::ADAPTER_WIDTH_QUANTUM, source + cast)
    );
}

fn packed_adapter_host_guard_body() {
    with_cpu_diagnostic_headroom(AMPLE, || {
        // This fixture's visible scalar requests must be fresh too; an unrelated cached 6-byte
        // buffer is allowed to back a 4-byte request outside the existing allocation scope.
        let _scope = mlx_gen::ios_quantum::Scope::try_new()?;
        packed_adapter_host_guard_case(Dtype::Int32, &[]);
        packed_adapter_host_guard_case(Dtype::Int32, &[1]);
        packed_adapter_host_guard_case(Dtype::Int64, &[1]);
        packed_adapter_host_guard_case(Dtype::Float32, &[1]);
        packed_adapter_host_guard_case(Dtype::Float64, &[1]);
        Ok(())
    })
    .unwrap();
}

#[test]
fn packed_adapter_host_guards_both_width_reads_and_placeholders() {
    in_cpu_child(
        "quantum_tests::packed_adapter_host_guards_both_width_reads_and_placeholders",
        packed_adapter_host_guard_body,
    );
}

fn seed_larger_reusable_cache() {
    let page = crate::retention::PAGE_BYTES as usize;
    // BufferCache accepts this capacity for a 2-page request: 3p < min(4p, 2p+2p).
    let request = 2 * page;
    let seed = 3 * page;
    assert!(seed < (2 * request).min(request + 2 * page));
    assert_eq!(
        unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
        0
    );
    mlx_rs::memory::clear_cache();
    let array = Array::from_slice(&vec![0u8; seed], &[seed as i32]);
    array.eval().unwrap();
    drop(array);
    assert_eq!(
        unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
        0
    );
    assert!(
        mlx_rs::memory::get_cache_memory() >= seed,
        "the discriminator must start with reusable larger backing"
    );
}

fn load_after_larger_cache<T>(limit: usize, body: impl FnOnce() -> Result<T>) -> (T, Observed) {
    seed_larger_reusable_cache();
    let (result, events) = observed(AMPLE, body);
    let result = result.unwrap();
    assert!(
        !checks(&events).is_empty(),
        "an actual component load must reach its pre-read check"
    );
    assert_scoped(&events);
    assert_eq!(
        mlx_rs::memory::set_cache_limit(limit),
        limit,
        "every nested load scope must restore the caller policy"
    );
    (result, events)
}

fn component_load_scope_body() {
    struct RestoreCache(usize);
    impl Drop for RestoreCache {
        fn drop(&mut self) {
            mlx_rs::memory::clear_cache();
            mlx_rs::memory::set_cache_limit(self.0);
        }
    }
    let limit = 16 * crate::retention::PAGE_BYTES as usize;
    let _restore = RestoreCache(mlx_rs::memory::set_cache_limit(limit));
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let baseline = mlx_rs::memory::get_active_memory();
    let (backbone, _) = load_after_larger_cache(limit, || {
        crate::model::load_backbone(&backbone_dir(), &config, Dtype::Bfloat16)
    });
    let expected = crate::memory_strategy::backbone_bytes(
        &backbone_dir().join("model.safetensors"),
        Precision::Bf16,
        None,
    )
    .unwrap();
    assert_eq!(
        unsafe { mlx_sys::mlx_synchronize(mlx_rs::Stream::cpu().as_ptr()) },
        0
    );
    let actual = (mlx_rs::memory::get_active_memory() - baseline) as u64;
    assert!(actual > 0 && actual <= expected,
        "fresh retained model buffers {actual} exceed source per-buffer A {expected}; cached capacity is not a logical payload");
    drop(backbone);

    let (streamed, events) = load_after_larger_cache(limit, || {
        crate::model::load_backbone_streamed(
            &backbone_dir(),
            &config,
            Dtype::Bfloat16,
            None,
            Some(1),
        )
    });
    assert!(
        !checks(&events).iter().any(|c| c.0 == dit_q::BLOCK_READ),
        "scope setup must not materialize deferred blocks"
    );
    drop(streamed);
    let (tower, _) = load_after_larger_cache(limit, || {
        crate::text_encoder::IrisTextEncoder::load_quantized(
            &text_encoder_dir(),
            &config.text_encoder,
            None,
        )
    });
    drop(tower);
    let (tower, events) = load_after_larger_cache(limit, || {
        crate::text_encoder::IrisTextEncoder::load_streamed(
            &text_encoder_dir(),
            &config.text_encoder,
            None,
        )
    });
    assert!(
        !checks(&events)
            .iter()
            .any(|c| c.0 == crate::text_encoder::quanta::LAYER_READ),
        "deferred decoder layers must stay unread"
    );
    drop(tower);

    let spec =
        crate::stream_tests::generation_spec(mlx_gen::gen_core::LoadShape::EagerMaterialization);
    let (nested, _) = load_after_larger_cache(limit, || crate::model::load_model(&spec));
    drop(nested); // outer model scope plus both nested public component scopes
    let (depth, _) = load_after_larger_cache(limit, || {
        crate::depth::load_depth_export(
            TaskExport::from_dir(
                &fixtures().join("tiny-depth"),
                IrisTask::Depth,
                crate::depth::DEPTH_MODEL_ID,
            )
            .unwrap(),
            Dtype::Bfloat16,
        )
    });
    drop(depth);
    let spec = LoadSpec::new(WeightsSource::Dir(
        fixtures().join("tiny-snapshot/upscaler"),
    ));
    let (restorer, _) =
        load_after_larger_cache(limit, || crate::restoration::IrisRestorer::load(&spec));
    drop(restorer);

    let missing = tempfile::tempdir().unwrap();
    seed_larger_reusable_cache();
    let cached = mlx_rs::memory::get_cache_memory();
    assert!(!mlx_gen::ios_quantum::enabled());
    assert!(crate::model::load_backbone(missing.path(), &config, Dtype::Bfloat16).is_err());
    assert_eq!(
        mlx_rs::memory::get_cache_memory(),
        cached,
        "disabled Scope::None must not clear caches"
    );
    assert_eq!(mlx_rs::memory::set_cache_limit(limit), limit);
    let (failed, _) = observed(AMPLE, || {
        crate::model::load_backbone(missing.path(), &config, Dtype::Bfloat16)
    });
    assert!(failed.is_err());
    assert_eq!(
        mlx_rs::memory::set_cache_limit(limit),
        limit,
        "early load error must restore policy too"
    );
}

#[test]
fn component_load_scopes_establish_fresh_backing_before_materialization() {
    in_cpu_child(
        "quantum_tests::component_load_scopes_establish_fresh_backing_before_materialization",
        component_load_scope_body,
    );
}

// The constructor guard must deny AFTER the unchanged raw floor passes, and BEFORE the native
// file load/read casts/build/internal eval. These are source-authored CPU-child tests, not a
// replacement for the original raw-floor, adapter-7440, geometry or measured<=derived tests.
fn load_target_denial_body() {
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let headers = crate::stream::headers(&backbone_dir().join("model.safetensors")).unwrap();
    let raw = crate::stream::materialized_load_bytes(&headers, |_| true).unwrap();
    let view = headers.iter().map(|h| (h.name.as_str(), h)).collect();
    let joint =
        crate::stream::backbone_load_target_bytes(&view, Dtype::Bfloat16, |_| true).unwrap();
    assert!(joint > raw);
    take_refusal();
    let active = mlx_rs::memory::get_active_memory();
    let (denied, events) = observed(raw + PROCESS_RESERVE_BYTES, || {
        crate::model::load_backbone(&backbone_dir(), &config, Dtype::Bfloat16)
    });
    assert!(
        denied.is_err(),
        "a raw-affordable but target-unaffordable model must not build"
    );
    assert_eq!(
        checks(&events),
        [(crate::model::BACKBONE_LOAD_QUANTUM.to_string(), raw, 0)]
    );
    let refusal = take_refusal().expect("the actual load-target check must refuse");
    assert_eq!(refusal.label, crate::stream::LOAD_TARGET_QUANTUM);
    assert_eq!(refusal.scratch_bytes, joint);
    assert_eq!(
        mlx_rs::memory::get_active_memory(),
        active,
        "no native model target was allocated"
    );

    let cfg = mlx_llm::ModelConfig::from_dir(text_encoder_dir()).unwrap();
    let mut tower_headers = Vec::new();
    for file in
        crate::text_encoder::weight_shards(&text_encoder_dir(), !cfg.tie_word_embeddings).unwrap()
    {
        tower_headers.extend(
            crate::stream::headers(&file)
                .unwrap()
                .into_iter()
                .filter(|h| {
                    crate::text_encoder::is_text_tower_tensor(&h.name, !cfg.tie_word_embeddings)
                }),
        );
    }
    let (logical, additional) = crate::stream::tower_load_output_bytes(
        &tower_headers,
        |_| true,
        crate::stream::tower_rope_bytes(&cfg).unwrap(),
    )
    .unwrap();
    assert!(additional > 0);
    take_refusal();
    let active = mlx_rs::memory::get_active_memory();
    let (denied, events) = observed(logical + PROCESS_RESERVE_BYTES, || {
        crate::text_encoder::IrisTextEncoder::load(&text_encoder_dir(), &config.text_encoder)
    });
    assert!(
        denied.is_err(),
        "target check precedes CausalLm build_rope and internal eval"
    );
    assert_eq!(
        checks(&events),
        [(crate::text_encoder::quanta::LOAD.to_string(), logical, 0)]
    );
    let refusal = take_refusal().expect("the actual tower load-target check must refuse");
    assert_eq!(refusal.label, crate::stream::LOAD_TARGET_QUANTUM);
    assert_eq!(refusal.scratch_bytes, logical + additional);
    assert_eq!(
        mlx_rs::memory::get_active_memory(),
        active,
        "no tower target/frequency row was created"
    );
}

#[test]
fn load_target_guard_denies_after_raw_pass_before_first_native_target() {
    in_cpu_child(
        "quantum_tests::load_target_guard_denies_after_raw_pass_before_first_native_target",
        load_target_denial_body,
    );
}

/// Source plan for one actual eager backbone load: the PURE plan the runtime guard also uses.
fn backbone_plan(quant: Option<mlx_gen::Quant>, compute: Dtype) -> crate::stream::PendingBackbone {
    let headers = crate::stream::headers(&backbone_dir().join("model.safetensors")).unwrap();
    let quant = crate::nn::BackboneQuant::for_load(quant).unwrap();
    crate::stream::backbone_pending_plan(&vec![headers], "", compute, quant)
        .unwrap()
        .1
}

fn refused_scratch(label: &str, load: impl FnOnce() -> Result<()>) -> (u64, Vec<String>) {
    let (scratch, labels, available) = refused_capacity(label, load);
    // short_at offers exactly the stage's own price plus reserve, minus one byte.
    assert_eq!(
        available,
        scratch + PROCESS_RESERVE_BYTES - 1,
        "{label}: refused at its own price"
    );
    (scratch, labels)
}

fn refused_capacity(label: &str, load: impl FnOnce() -> Result<()>) -> (u64, Vec<String>, u64) {
    take_refusal();
    let (denied, events) = observed(AMPLE, || {
        mlx_gen::ios_quantum::with_cpu_diagnostic_short_at(label, load)
    });
    assert!(denied.is_err(), "{label}: the real stage must refuse");
    let labels: Vec<String> = checks(&events).into_iter().map(|check| check.0).collect();
    assert!(
        !labels.iter().any(|l| l == label),
        "{label}: a refused check emits no Checked mark"
    );
    let refusal = take_refusal().unwrap();
    assert_eq!(refusal.label, label);
    (
        refusal.scratch_bytes,
        labels,
        refusal.available_bytes.expect("fixture headroom is known"),
    )
}

/// The refused headroom admits the constructor-only law but not the real pending plan: dropping
/// the stored-grid graph would PASS this budget, so the refusal discriminates the quant term.
fn assert_discriminates(label: &str, available: u64, constructor_only: u64, pending: u64) {
    assert!(
        available >= constructor_only + PROCESS_RESERVE_BYTES,
        "{label}: a constructor-only law would admit this headroom"
    );
    assert!(
        available < pending + PROCESS_RESERVE_BYTES,
        "{label}: the real pending plan is refused"
    );
}

fn post_adapter_denial_body() {
    use mlx_gen::{AdapterKind, AdapterSpec, Quant};
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let specs = [AdapterSpec::new(
        fixtures().join("iris_lora.safetensors"),
        0.75,
        AdapterKind::Lora,
    )];
    let compute = Dtype::Bfloat16;
    let unquantized = backbone_plan(None, compute);
    for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
        for adapted in [false, true] {
            if quant.is_none() && !adapted {
                continue;
            } // bare dense has no pending stage
            let plan = backbone_plan(quant, compute);
            let adapters: &[AdapterSpec] = if adapted { &specs } else { &[] };
            let load = || {
                crate::model::load_backbone_quantized(
                    &backbone_dir(),
                    &config,
                    compute,
                    adapters,
                    IrisTask::Generation,
                    crate::model::MODEL_ID,
                    quant,
                )
                .map(|_| ())
            };
            let (scratch, labels, available) =
                refused_capacity(crate::nn::PENDING_CONSTRUCTION_QUANTUM, load);
            if quant.is_some() {
                let quant_only = plan.constructors - unquantized.constructors;
                assert_discriminates(
                    crate::nn::PENDING_CONSTRUCTION_QUANTUM,
                    available,
                    scratch - quant_only,
                    scratch,
                );
            }
            assert_eq!(
                labels[0],
                crate::model::BACKBONE_LOAD_QUANTUM,
                "original raw check first"
            );
            assert_eq!(
                labels[1],
                crate::stream::LOAD_TARGET_QUANTUM,
                "early joint check second"
            );
            if adapted {
                let adapter_file = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(
                    fixtures().join("iris_lora.safetensors"),
                )
                .unwrap()
                .iter()
                .map(|h| h.data_bytes)
                .sum::<u64>();
                let first_adapter = labels
                    .iter()
                    .position(|l| l == crate::adapters::ADAPTER_LOAD_QUANTUM)
                    .unwrap();
                assert!(
                    first_adapter > 1,
                    "adapter reads follow raw and early joint checks"
                );
                let _ = adapter_file; // exact 7440 tuple remains asserted by adapter_read_body
                                      // Residual materialization is added on top of the source constructor plan.
                assert!(scratch >= plan.constructors);
            } else {
                assert_eq!(
                    scratch, plan.constructors,
                    "{quant:?}: exact pending build plan"
                );
            }
            if quant.is_some() {
                // A constructor-only law (no stored-grid graph) is strictly smaller and would
                // have admitted a budget this check correctly refuses.
                assert!(plan.constructors > unquantized.constructors, "{quant:?}");
            }
            let (scratch, labels, available) =
                refused_capacity(crate::nn::ADAPTED_BASE_EVAL_QUANTUM, load);
            if quant.is_some() {
                assert_discriminates(
                    crate::nn::ADAPTED_BASE_EVAL_QUANTUM,
                    available,
                    unquantized.evaluation,
                    scratch,
                );
            }
            assert!(labels
                .iter()
                .any(|l| l == crate::nn::PENDING_CONSTRUCTION_QUANTUM));
            assert_eq!(
                scratch, plan.evaluation,
                "{quant:?} adapted={adapted}: post-build pending only"
            );
            assert!(
                plan.evaluation <= plan.constructors,
                "already-built scalars are not recharged"
            );
            if quant.is_some() {
                assert!(
                    plan.evaluation > unquantized.evaluation,
                    "{quant:?}: quant eval graph counted"
                );
            }
        }
    }
}

fn export_denial_body() {
    let elements = |dir: &std::path::Path| {
        let header = crate::stream::headers(&dir.join("empty_prompt.safetensors"))
            .unwrap()
            .into_iter()
            .find(|h| h.name == "embeddings")
            .unwrap();
        header.shape.iter().product::<usize>() as u64
    };
    let upscaler = fixtures().join("tiny-snapshot/upscaler");
    let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(upscaler.clone()));
    let upload = crate::stream::empty_prompt_upload_bytes(elements(&upscaler)).unwrap();
    let (scratch, labels) = refused_scratch(crate::stream::EXPORT_TARGET_QUANTUM, || {
        crate::restoration::IrisRestorer::load(&spec).map(|_| ())
    });
    assert_eq!(
        scratch, upload,
        "restoration: only the fresh F32 prompt upload remains"
    );
    assert!(
        labels
            .iter()
            .filter(|l| *l == crate::stream::LOAD_TARGET_QUANTUM)
            .count()
            >= 2,
        "outer and inner backbone joint checks both passed first"
    );

    let depth = fixtures().join("tiny-depth");
    let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(depth.clone()));
    let model = crate::stream::headers(&depth.join("model.safetensors")).unwrap();
    let prompt = crate::stream::headers(&depth.join("empty_prompt.safetensors")).unwrap();
    let plan =
        crate::stream::backbone_pending_plan(&vec![model, prompt], "pixel.", Dtype::Bfloat16, None)
            .unwrap()
            .1;
    let upload = crate::stream::empty_prompt_upload_bytes(elements(&depth)).unwrap();
    let (scratch, labels) = refused_scratch(crate::stream::EXPORT_TARGET_QUANTUM, || {
        crate::depth::load_depth(&spec).map(|_| ())
    });
    assert_eq!(
        scratch,
        plan.after_build_targets().unwrap() + upload,
        "depth: only post-build pending targets plus F32 upload"
    );
    assert!(plan.constructors > 0);
    assert!(labels
        .iter()
        .any(|l| l == crate::stream::LOAD_TARGET_QUANTUM));
}

#[test]
fn export_target_guards_refuse_before_late_restoration_and_depth_uploads() {
    in_cpu_child(
        "quantum_tests::export_target_guards_refuse_before_late_restoration_and_depth_uploads",
        export_denial_body,
    );
}

#[test]
fn adapted_and_dense_quantized_pending_guards_follow_real_intervening_growth() {
    in_cpu_child(
        "quantum_tests::adapted_and_dense_quantized_pending_guards_follow_real_intervening_growth",
        post_adapter_denial_body,
    );
}

/// Keep the packed bases and their metadata unchanged; only encode their existing logical widths
/// in another accepted native numeric representation. Every projection gets an explicit marker,
/// including skeleton and deferred-stack projections, so selection is independently observable.
fn numeric_depth_width_markers(dir: &std::path::Path, dtype: Dtype, shape: &[i32]) {
    let path = dir.join("model.safetensors");
    let metadata: std::collections::HashMap<_, _> =
        mlx_gen::gen_core::weightsmeta::safetensors_file_metadata(&path)
            .unwrap()
            .into_iter()
            .collect();
    let mut tensors = Array::load_safetensors(&path).unwrap();
    let projections: Vec<_> = tensors
        .keys()
        .filter_map(|name| {
            let base = name.strip_suffix(".scales")?;
            let weight = tensors.get(&format!("{base}.weight"))?;
            (weight.dtype() == Dtype::Uint32).then(|| base.to_owned())
        })
        .collect();
    assert!(
        !projections.is_empty(),
        "the packed fixture must contain real projections"
    );
    for base in projections {
        let marker = format!("{base}{}", crate::packed::IN_FEATURES_SUFFIX);
        let value = match tensors.get(&marker) {
            Some(array) => array.try_item::<i32>().unwrap(),
            None => {
                tensors[&format!("{base}.scales")].shape()[1] * crate::nn::QUANT_GROUP_SIZE as i32
            }
        };
        // The matrix-marker case intentionally preserves the loader's existing BF16 read
        // policy. All this fixture's logical widths are representable, so no numeric knob moves.
        let round_trip = Array::from_slice(&[value], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap()
            .try_item::<i32>()
            .unwrap();
        assert_eq!(
            round_trip, value,
            "fixture width must survive the unchanged read policy"
        );
        let array = match dtype {
            Dtype::Int64 => Array::from_slice(&[i64::from(value)], shape),
            Dtype::Float32 => Array::from_slice(&[value as f32], shape),
            Dtype::Int32 => Array::from_slice(&[value], shape),
            _ => unreachable!("the test names its supported numeric marker formats"),
        };
        tensors.insert(marker, array);
    }
    for array in tensors.values() {
        array.eval().unwrap();
    }
    let mut entries: Vec<_> = tensors
        .iter()
        .map(|(name, array)| (name.as_str(), array))
        .collect();
    entries.sort_unstable_by_key(|(name, _)| *name);
    // Do not overwrite a file whose lazy/mapped source arrays are still owned by the writer.
    let output = dir.join("numeric-width-model.safetensors");
    Array::save_safetensors(entries, Some(&metadata), &output).unwrap();
    std::fs::rename(output, path).unwrap();
}

fn depth_completed_width_body() {
    use mlx_gen::gen_core::LoadShape;
    for (dtype, shape) in [
        (Dtype::Int64, Vec::new()),
        (Dtype::Float32, vec![1]),
        (Dtype::Int32, vec![1, 1]),
    ] {
        let root = tempfile::tempdir().unwrap();
        let source = fixtures().join("tiny-depth");
        for quant in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let dir = crate::packed_tests::convert(
                crate::packed::ConvertKind::Export { depth: true },
                &source,
                root.path(),
                quant,
            );
            numeric_depth_width_markers(&dir, dtype, &shape);
            let full_model = crate::stream::headers(&dir.join("model.safetensors")).unwrap();
            let prompt = crate::stream::headers(&dir.join("empty_prompt.safetensors")).unwrap();
            let embeddings = prompt
                .iter()
                .find(|header| header.name == "embeddings")
                .unwrap();
            let elements = embeddings.shape.iter().product::<usize>() as u64;
            let upload = crate::stream::empty_prompt_upload_bytes(elements).unwrap();
            let all = crate::stream::backbone_pending_plan(
                &vec![full_model.clone(), prompt.clone()],
                "pixel.",
                Dtype::Bfloat16,
                None,
            )
            .unwrap()
            .1;
            for load_shape in [
                LoadShape::EagerMaterialization,
                LoadShape::DeferredMaterialization,
            ] {
                let selected: Vec<_> = full_model
                    .iter()
                    .filter(|header| {
                        load_shape != LoadShape::DeferredMaterialization
                            || header
                                .name
                                .strip_prefix("pixel.")
                                .is_none_or(|name| crate::dit::Stack::of(name).is_none())
                    })
                    .cloned()
                    .collect();
                let plan = crate::stream::backbone_pending_plan(
                    &vec![selected, prompt.clone()],
                    "pixel.",
                    Dtype::Bfloat16,
                    None,
                )
                .unwrap()
                .1;
                assert!(
                    plan.width_outputs > 0,
                    "the selected skeleton marker conversion must run"
                );
                if load_shape == LoadShape::DeferredMaterialization {
                    assert!(
                        all.width_outputs > plan.width_outputs,
                        "unloaded stack marker getters must not grant completion credit"
                    );
                }
                let expected = plan.after_build_targets().unwrap() + upload;
                assert_eq!(
                    expected + plan.width_outputs,
                    plan.constructors + upload,
                    "exact same getter operations move out of future allocation scratch"
                );
                let spec = LoadSpec::new(WeightsSource::Dir(dir.clone()))
                    .with_quant(quant)
                    .with_load_shape(load_shape);
                let (scratch, labels) =
                    refused_scratch(crate::stream::EXPORT_TARGET_QUANTUM, || {
                        crate::depth::load_depth(&spec).map(|_| ())
                    });
                assert_eq!(scratch, expected, "{dtype:?} {quant:?} {load_shape:?}");
                assert!(
                    scratch < plan.constructors + upload,
                    "completed getter casts must not be allocated a second time"
                );
                assert!(labels
                    .iter()
                    .any(|label| label == crate::stream::LOAD_TARGET_QUANTUM));
                let (loaded, events) = observed(AMPLE, || crate::depth::load_depth(&spec));
                assert!(
                    loaded.is_ok(),
                    "accepted marker format must still load: {dtype:?} {quant:?}"
                );
                assert_eq!(
                    checks(&events)
                        .iter()
                        .filter(|check| check.0 == crate::stream::EXPORT_TARGET_QUANTUM)
                        .count(),
                    1
                );
                drop(loaded);
            }
        }
    }
}

#[test]
fn depth_late_targets_use_only_completed_selected_numeric_width_operations() {
    in_cpu_child(
        "quantum_tests::depth_late_targets_use_only_completed_selected_numeric_width_operations",
        depth_completed_width_body,
    );
}
