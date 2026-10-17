//! The measured proof of every counted Iris iOS quantum price (epic 16774, round 3).
//!
//! Each body below builds the REAL module (the committed miniature exports), materializes the
//! quantum's inputs, then measures MLX's peak allocation while exactly that quantum's body builds
//! and evaluates ([`mlx_gen::ios_quantum::measured_peak_bytes`]) and asserts the counted formula is
//! at least the measurement and at most [`PROOF_MAX`] times it
//! ([`mlx_gen::ios_quantum::assert_tracks_measured`]). Shapes are small, so the CPU's own attention
//! fallback (materialized logits and softmax) is priced as such through
//! [`crate::nn::sdpa_scores`]; the iOS GPU runs those attentions fused. Every body runs at the f32
//! and the bf16 compute width, and the attention sites under both an unbounded and a chunking plan.
//! All in a CPU child with Metal trapped.

use std::path::PathBuf;

use mlx_gen::gen_core::iris::downstream::TaskExport;
use mlx_gen::gen_core::iris::{DpmStep, DpmUpdate, IrisConfig, IrisTask, Prediction};
use mlx_gen::ios_quantum::{assert_tracks_measured, measured_peak_bytes};
use mlx_gen::{AttentionBudget, AttentionPlan};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use crate::nn::key_padding_mask;

use crate::test_metal_guard as metal_guard;

/// The largest formula / measurement ratio a proof accepts.
const PROOF_MAX: f64 = 1.5;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn backbone_dir() -> PathBuf {
    fixtures().join("tiny-snapshot/iris")
}

fn in_cpu_child(test: &str, body: fn()) {
    const CHILD: &str = "IRIS_QUANTUM_PROOF_CHILD";
    const DONE: &str = "IRIS_QUANTUM_PROOF_CHILD_DONE";
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

/// Assert `formula` tracks `measured` within [`PROOF_MAX`], printing both.
fn prove(label: &str, formula: u64, measured: u64) {
    prove_within(label, formula, measured, PROOF_MAX);
}

/// [`prove`] with a stated, site-specific bound (never above 2.0).
fn prove_within(label: &str, formula: u64, measured: u64, max: f64) {
    assert!(max <= 2.0);
    eprintln!(
        "PROOF {label}: formula {formula} B, measured {measured} B, ratio {:.3}",
        formula as f64 / measured.max(1) as f64
    );
    if std::env::var_os("IRIS_PROOF_PRINT_ONLY").is_none() {
        assert_tracks_measured(label, formula, measured, max);
    }
}

/// The solver step's bound: one count per update order serves every batch, and MLX's buffer
/// donation leaves a second-order step at 4 live states for one image and 5⅓ for two (measured),
/// so the six-state count sits at 1.5x of the former plus its scalar coefficients.
const SOLVER_PROOF_MAX: f64 = 1.55;

fn rand(shape: &[i32], seed: u64, dtype: Dtype) -> Array {
    let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
        .unwrap()
        .as_dtype(dtype)
        .unwrap();
    eval([&a]).unwrap();
    a
}

/// Runs of every measurement; a proof holds against the largest.
const RUNS: usize = 5;

/// Measure `run`'s peak (the largest of [`RUNS`] runs) and hand back its outputs for the next
/// quantum.
///
/// MLX's CPU scheduler encodes ahead of execution (an op's output is allocated when it is encoded,
/// its inputs released when it has run), so an unconstrained measurement swings with thread timing.
/// Each run is measured with MLX's memory limit at one byte, under which MLX waits for the work in
/// flight before it encodes the next op: the peak is then the graph's LIVE set — what the counted
/// formulas price. Buffer donation still depends on timing, hence the largest of several runs.
fn measure(run: impl Fn() -> mlx_gen::Result<Vec<Array>>) -> (u64, Vec<Array>) {
    let mut kept = Vec::new();
    let mut worst = 0;
    for _ in 0..RUNS {
        let previous = mlx_rs::memory::set_memory_limit(1);
        let peak = measured_peak_bytes(|| {
            let outputs = run()?;
            kept = outputs.clone();
            Ok(outputs)
        });
        mlx_rs::memory::set_memory_limit(previous);
        worst = worst.max(peak.unwrap());
    }
    (worst, kept)
}

const CHUNKING: AttentionBudget = AttentionBudget::from_score_elements(300, true);

fn backbone_body() {
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let m = &config.model;
    let layers = config.text_encoder.hidden_layers.len() as i32;
    let (t_len, dt) = (m.text_len as i32, m.text_dim as i32);
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        for (b, h, w) in [(2, 16, 24), (1, 32, 32)] {
            for budget in [AttentionBudget::UNBOUNDED, CHUNKING] {
                // A fresh backbone per shape: the pixel position table is built inside the first
                // pixel-embed quantum of a size, which its price carries.
                let dit = crate::model::load_backbone(&backbone_dir(), &config, dtype).unwrap();
                let plan = AttentionPlan::budgeted(budget);
                let tag = format!("{dtype:?} b{b} {h}x{w} {budget:?}");
                let p = m.patch_size as i32;
                let (hp, wp) = (h / p, w / p);
                let n = hp * wp;
                let x = rand(&[b, 3, h, w], 1, Dtype::Float32);
                let t = Array::from_slice(&vec![500.0f32; b as usize], &[b]);
                let states = rand(&[b, t_len, layers, dt], 2, Dtype::Float32);
                let mut mask = vec![vec![1; t_len as usize]; b as usize];
                mask[0][t_len as usize - 3..].fill(0);
                eval([&t]).unwrap();

                let (peak, out) = measure(|| {
                    let e = dit.embed_body(&x, &t)?;
                    let mut v = vec![e.s, e.t_emb, e.core_img];
                    v.extend(e.core_txt);
                    Ok(v)
                });
                prove(&format!("embed {tag}"), dit.embed_scratch(b, n), peak);
                let (mut s, t_emb, core_img) = (out[0].clone(), out[1].clone(), out[2].clone());
                let core_txt = out.get(3).cloned();

                let mut y = states.reshape(&[b * t_len, layers, dt]).unwrap();
                for i in 0..2 {
                    let (peak, out) = measure(|| Ok(vec![dit.text_layer_body(i, &y, plan)?]));
                    prove(
                        &format!("text layer block {i} {tag}"),
                        dit.text_layer_block_scratch(i, b * t_len, layers, plan),
                        peak,
                    );
                    y = out[0].clone();
                }
                let (peak, out) = measure(|| Ok(vec![dit.text_pool_body(&y, b, t_len, dt)?]));
                prove(
                    &format!("text pool {tag}"),
                    dit.text_pool_scratch(b, t_len, layers, dt),
                    peak,
                );
                let mut y = out[0].clone();
                let key_mask = key_padding_mask(&mask, t_len as usize);
                eval([&key_mask]).unwrap();
                for i in 0..2 {
                    let (peak, out) =
                        measure(|| Ok(vec![dit.text_refiner_body(i, &y, &key_mask, plan)?]));
                    prove(
                        &format!("text refiner block {i} {tag}"),
                        dit.text_refiner_block_scratch(i, b, t_len, plan),
                        peak,
                    );
                    y = out[0].clone();
                }
                let (peak, out) = measure(|| Ok(vec![dit.text_output_body(&y)?]));
                prove(
                    &format!("text output {tag}"),
                    dit.text_output_scratch(b, t_len),
                    peak,
                );
                let mut y = dit.text_positions(&out[0]).unwrap();
                eval([&y]).unwrap();

                let ctx = dit.trunk_ctx(hp, wp, t_len, plan);
                for i in 0..dit.trunk_depth() {
                    let (peak, out) = measure(|| {
                        let (s, y) =
                            dit.run_trunk_block(i, &ctx, &s, &y, &core_img, core_txt.as_ref())?;
                        Ok(vec![s, y])
                    });
                    prove(
                        &format!("trunk block {i} {tag}"),
                        dit.trunk_block_scratch(i, b, n, t_len, plan),
                        peak,
                    );
                    s = out[0].clone();
                    y = out[1].clone();
                }

                let (peak, out) = measure(|| {
                    let (s_cond, pixels) = dit.pixel_embed_body(&x, &t_emb, &s)?;
                    Ok(vec![s_cond, pixels])
                });
                prove(
                    &format!("pixel embed {tag}"),
                    dit.pixel_embed_scratch(b, h, w),
                    peak,
                );
                let (s_cond, mut pixels) = (out[0].clone(), out[1].clone());
                let pctx = dit.pixel_ctx(b, hp, wp, plan);
                for i in 0..dit.pixel_depth() {
                    let (peak, out) =
                        measure(|| Ok(vec![dit.run_pixel_block(i, &pctx, &pixels, &s_cond)?]));
                    prove(
                        &format!("pixel block {i} {tag}"),
                        dit.pixel_block_scratch(i, b, n, plan),
                        peak,
                    );
                    pixels = out[0].clone();
                }
                let (peak, _) = measure(|| Ok(vec![dit.output_body(&pixels, b, h, w)?]));
                prove(&format!("output {tag}"), dit.output_scratch(b, n), peak);
            }
        }
    }
}

#[test]
fn every_backbone_quantum_price_tracks_its_measured_peak() {
    in_cpu_child(
        "quantum_proofs::every_backbone_quantum_price_tracks_its_measured_peak",
        backbone_body,
    );
}

fn text_encoder_body() {
    use crate::text_encoder::IrisTextEncoder;
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let te = IrisTextEncoder::load(
        &fixtures().join("tiny-snapshot/text_encoder"),
        &config.text_encoder,
    )
    .unwrap();
    let vocab = te.lm().config().vocab_size;
    for n in [48, 96] {
        let ids: Vec<i32> = (0..n).map(|i| (i * 7 + 3) % vocab).collect();
        let ids = Array::from_slice(&ids, &[1, n]);
        for k in [0usize, 1, 5] {
            // One decoder layer: the states before it (and their K/V) materialized first, then the
            // next state alone — a fresh graph per run.
            let mut worst = 0;
            for _ in 0..RUNS {
                let mut cache = te.lm().new_cache();
                let states = te.lm().hidden_states(&ids, &mut cache, 0).unwrap();
                for state in &states[..=k] {
                    eval([state]).unwrap();
                }
                let next = states[k + 1].clone();
                drop(states);
                let previous = mlx_rs::memory::set_memory_limit(1);
                let peak = measured_peak_bytes(|| Ok(vec![next])).unwrap();
                mlx_rs::memory::set_memory_limit(previous);
                worst = worst.max(peak);
            }
            let (layer, _held) = te.layer_scratch_parts(n, k + 1);
            prove(&format!("text decoder layer {k} n{n}"), layer, worst);
        }
    }
    // The conditioning quantum over a real window, short (padded) and the full budget.
    for prompt in ["a fox", "a red fox in the snow at golden hour"] {
        let window = te.window(prompt).unwrap();
        let n = window.input_ids.len() as i32;
        let ids = Array::from_slice(&window.input_ids, &[1, n]);
        let mut cache = te.lm().new_cache();
        let states = te.lm().hidden_states(&ids, &mut cache, 0).unwrap();
        for state in &states {
            eval([state]).unwrap();
        }
        let start = window.prefix_len as i32;
        let real = window.window_tokens() as i32;
        let (peak, _) = measure(|| Ok(vec![te.conditioning_body(&states, start, real)?]));
        prove(
            &format!("text conditioning {prompt:?}"),
            te.conditioning_scratch(real),
            peak,
        );
    }
}

#[test]
fn every_text_encoder_quantum_price_tracks_its_measured_peak() {
    in_cpu_child(
        "quantum_proofs::every_text_encoder_quantum_price_tracks_its_measured_peak",
        text_encoder_body,
    );
}

fn small_steps_body() {
    // Solver step: the lazy CFG combine of a batched forward output plus the update, both orders.
    for (b, h, w) in [(1, 64, 64), (2, 32, 48)] {
        let x = rand(&[b, 3, h, w], 3, Dtype::Float32);
        let forward = rand(&[2 * b, 3, h, w], 4, Dtype::Float32);
        let prev = rand(&[b, 3, h, w], 5, Dtype::Float32);
        for update in [
            DpmUpdate::First { cx: 0.9, c0: 0.1 },
            DpmUpdate::Second {
                cx: 0.9,
                c0: 0.1,
                c1: 0.05,
                r0: 0.5,
            },
        ] {
            let step = DpmStep {
                s: 0.8,
                t: 800.0,
                update,
            };
            let (peak, _) = measure(|| {
                let halves = forward.split(2, 0)?;
                let out = crate::solver::cfg_combine(&halves[0], &halves[1], 3.0)?;
                let (next, x0) =
                    crate::solver::step_update(&x, &out, &step, Prediction::Velocity, Some(&prev))?;
                Ok(vec![next, x0])
            });
            prove_within(
                &format!("solver step {update:?} b{b} {h}x{w}"),
                crate::solver::step_scratch(x.size() as u64, &update),
                peak,
                SOLVER_PROOF_MAX,
            );
        }
    }
    // The RGB8 conversion (its MLX part; the u8 host pixels are the formula's last term).
    for (b, h, w) in [(1, 64, 64), (2, 32, 48)] {
        let x = rand(&[b, 3, h, w], 6, Dtype::Float32);
        let elements = x.size() as u64;
        let (peak, _) = measure(|| {
            crate::pipeline::to_images(&x)?;
            Ok(Vec::new())
        });
        prove(
            &format!("image conversion b{b} {h}x{w}"),
            crate::model::image_scratch(elements) - elements,
            peak,
        );
    }
    // Depth input and reducer, at both compute widths.
    use crate::depth::{
        depth_input_body, depth_input_scratch, depth_reducer_body, depth_reducer_scratch,
        load_depth_export, DepthParts, DEPTH_MODEL_ID,
    };
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let export = TaskExport::from_dir(
            &fixtures().join("tiny-depth"),
            IrisTask::Depth,
            DEPTH_MODEL_ID,
        )
        .unwrap();
        let depth = load_depth_export(export, dtype).unwrap();
        let parts: DepthParts = depth.parts();
        for (b, h, w) in [(1, 32, 32), (2, 16, 24)] {
            let rgb = rand(&[b, 3, h, w], 7, Dtype::Float32);
            let (peak, _) = measure(|| {
                let (x, t, states) = depth_input_body(&parts, &rgb)?;
                Ok(vec![x, t, states])
            });
            prove(
                &format!("depth input {dtype:?} b{b} {h}x{w}"),
                depth_input_scratch(b, h, w),
                peak,
            );
            let out = rand(&[b, 3, h, w], 8, Dtype::Float32);
            let (peak, _) = measure(|| Ok(vec![depth_reducer_body(&parts, &out)?]));
            prove(
                &format!("depth reducer {dtype:?} b{b} {h}x{w}"),
                depth_reducer_scratch(&out, dtype),
                peak,
            );
        }
    }
}

#[test]
fn every_small_step_quantum_price_tracks_its_measured_peak() {
    in_cpu_child(
        "quantum_proofs::every_small_step_quantum_price_tracks_its_measured_peak",
        small_steps_body,
    );
}
