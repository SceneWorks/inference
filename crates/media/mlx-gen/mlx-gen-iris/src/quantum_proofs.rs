//! The measured proof of every counted Iris iOS quantum price (epic 16774, rounds 3–4).
//!
//! Each body below builds the REAL module (the committed miniature exports), materializes the
//! quantum's inputs, then measures MLX's peak allocation while exactly that quantum's body builds
//! and evaluates — UNTHROTTLED, natural and stalled, the worst of several runs ([`measure`]) — and
//! asserts the counted formula ([`crate::retention`]) is at least the measurement, reporting ratios
//! above [`PROOF_MAX`] ([`mlx_gen::ios_quantum::assert_tracks_measured`]). Shapes are small, so
//! the CPU's unfused fallbacks (RMSNorm, attention with materialized scores) are priced as such
//! ([`crate::retention::unfused`]); the iOS GPU runs those fused. Every body runs at the f32 and
//! the bf16 compute width, and the attention sites under both an unbounded and a chunking plan.
//! All in a CPU child with Metal trapped.

use std::path::PathBuf;

use mlx_gen::gen_core::iris::downstream::TaskExport;
use mlx_gen::gen_core::iris::{DpmStep, DpmUpdate, IrisConfig, IrisTask, Prediction};
use mlx_gen::ios_quantum::{
    assert_tracks_measured, measured_peak_bytes, with_cpu_diagnostic_headroom,
};
use mlx_gen::{AttentionBudget, AttentionPlan};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use crate::nn::key_padding_mask;

use crate::test_metal_guard as metal_guard;

/// Report-only formula / measurement ratio, shared by decoder and other quanta.
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
        .args(["--exact", test, "--nocapture", "--test-threads", "1"])
        .env(CHILD, test)
        .env(DONE, &marker)
        .status()
        .unwrap();
    assert!(status.success(), "{test} failed in its CPU child");
    assert!(marker.is_file(), "{test}: the CPU child ran no body");
}

/// Enforce `formula >= measured` in every mode, printing both and reporting ratios above
/// [`PROOF_MAX`]. `IRIS_PROOF_PRINT_ONLY` no longer disables safety assertions.
fn prove(label: &str, formula: u64, measured: u64) {
    eprintln!(
        "PROOF {label}: formula {formula} B, measured {measured} B, ratio {:.3}",
        formula as f64 / measured.max(1) as f64
    );
    assert_tracks_measured(label, formula, measured, PROOF_MAX);
}

#[test]
fn proof_policy_cannot_be_disabled_by_print_only() {
    in_cpu_child(
        "quantum_proofs::proof_policy_cannot_be_disabled_by_print_only",
        || {
            // This isolated, single-test child cannot affect the parent or another test.
            std::env::set_var("IRIS_PROOF_PRINT_ONLY", "1");
            prove("report-only control", 151, 100);
            prove("decoder report-only control", 170, 100);
            assert!(
                std::panic::catch_unwind(|| prove("underpriced discriminator", 99, 100)).is_err(),
                "print-only must not disable the measured lower bound"
            );
        },
    );
}

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

/// The CPU fixture's headroom: every guard passes.
const PLENTY: u64 = 1 << 34;

/// Queue a long CPU kernel ahead of the measured body on the same stream and hand it back (the
/// caller holds it until the body has evaluated). See [`measure`].
fn stall() -> Array {
    thread_local! {
        static OPERAND: Array = {
            let a = random::normal::<f32>(&[STALL_EDGE, STALL_EDGE], None, None, Some(&random::key(99).unwrap())).unwrap();
            eval([&a]).unwrap();
            a
        };
    }
    OPERAND.with(|a| {
        let s = a.matmul(a).unwrap();
        mlx_rs::transforms::async_eval([&s]).unwrap();
        s
    })
}

/// The stall kernel's edge: one `[E, E]` f32 product (~137 GFLOP at 4096, its one output allocated
/// before the measurement starts) keeps the CPU stream's worker busy far longer than building and
/// encoding any quantum's graph takes.
const STALL_EDGE: i32 = 4096;

/// Measure `run`'s peak, unthrottled, as the largest of [`RUNS`] natural runs and [`RUNS`] stalled
/// ones ([`measure_once`]), and hand back its outputs for the next quantum. No warm-up: the first
/// run of a size builds what the device's first quantum of that size builds (the pixel position
/// table), which its price carries.
fn measure(run: impl Fn() -> mlx_gen::Result<Vec<Array>>) -> (u64, Vec<Array>) {
    let mut kept = Vec::new();
    let mut worst = 0;
    for stalled in [false, true] {
        for _ in 0..RUNS {
            // The previous run's outputs are released BEFORE the measurement: released during it,
            // they would deflate the peak.
            kept.clear();
            let (peak, outputs) = measure_once(&run, stalled);
            kept = outputs;
            worst = worst.max(peak);
        }
    }
    (worst, kept)
}

/// One measurement of `run` on the iOS path — the CPU fixture enables every guard and sub-layer
/// barrier the device runs, and the device's quantum scope (no buffer cache, so a freed buffer is
/// never handed back larger than asked for).
///
/// MLX's scheduler encodes ahead of execution: an op's output is allocated when it is encoded
/// (on the encoding thread), its inputs stay referenced until its queued task (CPU: up to ten tasks
/// of ten dispatches in flight) or command buffer (Metal) completes, and an operand a queued task
/// still references cannot be donated. A sub-layer's intermediates can therefore all be live at
/// once — the retention model every counted price follows ([`crate::retention`]). A natural run
/// reaches it only as far as the worker happens to lag. A STALLED run queues a long kernel
/// ([`stall`]) on the same stream ahead of every sub-layer (at the start, and after each barrier
/// through [`crate::nn::BARRIER_HOOK`]), so each sub-layer is encoded — its buffers allocated, none
/// released — before its first op runs: the retention worst case, realized on the CPU. Each
/// sub-layer's peak is read from its own start, net of the stall kernels' outputs. Never measured
/// under a memory limit, which would drain the work in flight before each op and measure only the
/// minimal live set.
fn measure_once(run: &dyn Fn() -> mlx_gen::Result<Vec<Array>>, stalled: bool) -> (u64, Vec<Array>) {
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct Stalls {
        ahead: Vec<Array>,
        /// Bytes the stall kernels' outputs hold.
        offset: u64,
        base: u64,
        worst: u64,
    }
    impl Stalls {
        fn issue(&mut self) {
            let before = mlx_rs::memory::get_active_memory() as u64;
            self.ahead.push(stall());
            self.offset += (mlx_rs::memory::get_active_memory() as u64).saturating_sub(before);
            mlx_rs::memory::reset_peak_memory();
        }
        fn record(&mut self) {
            let peak = mlx_rs::memory::get_peak_memory() as u64;
            self.worst = self.worst.max(peak.saturating_sub(self.base + self.offset));
        }
    }

    with_cpu_diagnostic_headroom(PLENTY, || {
        let _scope = mlx_gen::ios_quantum::Scope::new();
        if !stalled {
            let mut outputs = Vec::new();
            let peak = measured_peak_bytes(|| {
                outputs = run()?;
                Ok(outputs.clone())
            })?;
            return Ok((peak, outputs));
        }
        mlx_rs::memory::clear_cache();
        let state = Rc::new(RefCell::new(Stalls {
            base: mlx_rs::memory::get_active_memory() as u64,
            ..Stalls::default()
        }));
        state.borrow_mut().issue();
        let hooked = Rc::clone(&state);
        crate::nn::BARRIER_HOOK.with(|hook| {
            *hook.borrow_mut() = Some(Box::new(move || {
                let mut stalls = hooked.borrow_mut();
                stalls.record();
                stalls.issue();
            }));
        });
        let outputs = run();
        let evaluated = outputs
            .as_ref()
            .map_err(|e| mlx_gen::Error::Msg(e.to_string()))
            .and_then(|o| Ok(eval(o.iter())?));
        crate::nn::BARRIER_HOOK.with(|hook| hook.borrow_mut().take());
        let outputs = outputs?;
        evaluated?;
        state.borrow_mut().record();
        let worst = state.borrow().worst;
        Ok((worst, outputs))
    })
    .unwrap()
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
            // next state alone — a fresh graph per run, natural and stalled (see [`measure`]).
            let mut worst = 0;
            for stalled in [false, true] {
                // Twice the runs: how much of the layer the CPU encode-ahead window retains
                // swings with timing. Excess conservatism is reported at the common threshold.
                for _ in 0..2 * RUNS {
                    let peak = with_cpu_diagnostic_headroom(PLENTY, || {
                        let _scope = mlx_gen::ios_quantum::Scope::new();
                        let mut cache = te.lm().new_cache();
                        let states = te.lm().hidden_states(&ids, &mut cache, 0).unwrap();
                        for state in &states[..=k] {
                            eval([state]).unwrap();
                        }
                        let next = states[k + 1].clone();
                        drop(states);
                        let ahead = stalled.then(stall);
                        let peak = measured_peak_bytes(|| Ok(vec![next]));
                        drop(ahead);
                        peak
                    })
                    .unwrap();
                    worst = worst.max(peak);
                }
            }
            let (layer, _held) = te.layer_scratch_parts(n, k + 1);
            prove(&format!("text decoder layer {k} n{n}"), layer, worst);
        }
    }
    // The decoder layer's counted components, each a graph short enough for the CPU to retain
    // whole (each uses the common reporting threshold): its bf16 RMSNorm, its NeoX RoPE and its causal
    // grouped-query attention, through `mlx_llm`'s own functions.
    use crate::retention::Retention;
    use crate::text_encoder::{decoder_attention, decoder_norm, decoder_rope};
    let llm = |e: mlx_llm::Error| mlx_gen::Error::Msg(e.to_string());
    let cfg = te.lm().config();
    let (hidden, heads, kv_heads, hd) = (
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
    );
    for n in [48, 96] {
        let price = |f: &dyn Fn(&mut Retention)| {
            let mut r = Retention::on(2, true);
            f(&mut r);
            r.price()
        };
        let x = rand(&[1, n, hidden], 11, Dtype::Bfloat16);
        let w = rand(&[hidden], 12, Dtype::Bfloat16);
        let (peak, _) = measure(|| {
            Ok(vec![
                mlx_llm::primitives::nn::rms_norm(&x, &w, 1e-6).map_err(llm)?
            ])
        });
        let (n64, rows) = (n as u64, n as u64);
        prove(
            &format!("text decoder norm n{n}"),
            price(&|r| decoder_norm(r, n64 * hidden as u64, rows, false)),
            peak,
        );
        let q = rand(&[1, n, heads, hd], 13, Dtype::Bfloat16);
        let cos = rand(&[1, n, hd], 14, Dtype::Bfloat16);
        let sin = rand(&[1, n, hd], 15, Dtype::Bfloat16);
        let (peak, _) = measure(|| {
            Ok(vec![mlx_llm::primitives::rope::apply_rope(
                &q, &cos, &sin, false,
            )
            .map_err(llm)?])
        });
        prove(
            &format!("text decoder rope n{n}"),
            price(&|r| decoder_rope(r, n64 * (heads * hd) as u64)),
            peak,
        );
        // q is the roped query's transposed view; k / v are views of the layer's cache block.
        let qt = q.transpose_axes(&[0, 2, 1, 3]).unwrap();
        let block = mlx_llm::primitives::kv_cache::KV_BLOCK_TOKENS;
        let cache_k = rand(&[1, kv_heads, block, hd], 16, Dtype::Bfloat16);
        let cache_v = rand(&[1, kv_heads, block, hd], 17, Dtype::Bfloat16);
        let kt = cache_k.split_axis(&[n], 2).unwrap().swap_remove(0);
        let vt = cache_v.split_axis(&[n], 2).unwrap().swap_remove(0);
        let (peak, _) = measure(|| {
            let out = mlx_llm::primitives::attention::sdpa(
                &qt,
                &kt,
                &vt,
                1.0 / (hd as f32).sqrt(),
                mlx_llm::primitives::attention::AttnMask::Causal,
            )
            .map_err(llm)?;
            Ok(vec![out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
                1,
                n,
                heads * hd,
            ])?])
        });
        prove(
            &format!("text decoder attention n{n}"),
            price(&|r| decoder_attention(r, n64, heads as u64, hd as u64)),
            peak,
        );
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
        let states: Vec<Option<Array>> = states.into_iter().map(Some).collect();
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
            prove(
                &format!("solver step {update:?} b{b} {h}x{w}"),
                crate::solver::step_scratch(x.size() as u64, &update),
                peak,
            );
        }
    }
    // The final solve clamp is its own guarded denoise quantum, not deferred decode work.
    for (b, h, w) in [(1, 64, 64), (2, 32, 48)] {
        let x = rand(&[b, 3, h, w], 6, Dtype::Float32);
        let (peak, _) = measure(|| Ok(vec![mlx_rs::ops::clip(&x, (-1.0f32, 1.0f32))?]));
        prove(
            "final solver clamp",
            crate::solver::final_clamp_scratch(x.size() as u64),
            peak,
        );
    }
    // RGB8 conversion starts from an evaluated solved batch; the host pixels are the last term.
    for (b, h, w) in [(1, 64, 64), (2, 32, 48)] {
        let x = rand(&[b, 3, h, w], 6, Dtype::Float32);
        let elements = x.size() as u64;
        let (peak, _) = measure(|| {
            crate::pipeline::to_images(&x)?;
            Ok(Vec::new())
        });
        prove(
            &format!("image conversion b{b} {h}x{w}"),
            crate::model::image_scratch(elements, b as u64) - elements,
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

/// The streamed (rung-4) quanta: a backbone window's read (every block of every stack, dense at
/// both compute widths and packed at both tiers), the tower's layer read (f32-stored, packed), its
/// cache-free decoder layer and its embedding + RoPE quantum, and the per-row output concatenation.
fn streamed_body() {
    use crate::dit::Stack;
    use crate::packed::ConvertKind;
    use mlx_gen::weights::Weights;
    use mlx_gen::Quant;
    let config = IrisConfig::from_dir(&backbone_dir()).unwrap();
    let root = tempfile::tempdir().unwrap();
    let mut backbones = vec![
        (backbone_dir(), Dtype::Float32, None),
        (backbone_dir(), Dtype::Bfloat16, None),
    ];
    for quant in [Quant::Q4, Quant::Q8] {
        let packed = crate::packed_tests::convert(
            ConvertKind::Export { depth: false },
            &backbone_dir(),
            root.path(),
            quant,
        );
        backbones.push((packed, Dtype::Bfloat16, Some(quant)));
    }
    for (dir, dtype, quant) in backbones {
        let dit =
            crate::model::load_backbone_streamed(&dir, &config, dtype, quant, Some(1)).unwrap();
        let stream = dit.block_stream().unwrap();
        for stack in Stack::ALL {
            for i in 0..stream.len(stack) {
                let (peak, _) = measure(|| {
                    let mut view = Weights::from_file(stream.path())?;
                    let built = stream.build(&mut view, &config.model, stack, i)?;
                    Ok(built.arrays().into_iter().cloned().collect())
                });
                prove(
                    &format!("backbone block read {stack:?} {i} {dtype:?} {quant:?}"),
                    crate::stream::window_read_bytes(&[stream.read(stack, i)]),
                    peak,
                );
            }
        }
    }

    use crate::text_encoder::IrisTextEncoder;
    let aligned = crate::packed_tests::aligned_tiny_text_encoder(root.path());
    let mut towers = vec![(fixtures().join("tiny-snapshot/text_encoder"), None)];
    for quant in [Quant::Q4, Quant::Q8] {
        let packed =
            crate::packed_tests::convert(ConvertKind::TextEncoder, &aligned, root.path(), quant);
        towers.push((packed, Some(quant)));
    }
    for (dir, quant) in towers {
        let te = IrisTextEncoder::load_streamed(&dir, &config.text_encoder, quant).unwrap();
        let tower = te.streamed().unwrap();
        let cfg = tower.config();
        for i in [0, 1, cfg.num_layers - 1] {
            let (peak, _) = measure(|| tower.read_layer(i));
            prove(
                &format!("text layer read {i} {quant:?}"),
                crate::stream::window_read_bytes(&[tower.read_price(i)]),
                peak,
            );
        }
        for n in [48, 96] {
            let ids: Vec<i32> = (0..n).map(|i| (i * 7 + 3) % cfg.vocab_size).collect();
            let (peak, out) = measure(|| {
                let ids = Array::from_slice(&ids, &[1, n]);
                let h = mlx_llm::primitives::nn::embed(tower.embed_table(), &ids)
                    .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
                let (cos, sin) = tower
                    .rope()
                    .cos_sin(n, 0, Dtype::Bfloat16)
                    .map_err(|e| mlx_gen::Error::Msg(e.to_string()))?;
                Ok(vec![h, cos, sin])
            });
            prove(
                &format!("text streamed embedding n{n} {quant:?}"),
                crate::text_encoder::streamed_embedding_bytes(cfg, n as u64),
                peak,
            );
            let (h, cos, sin) = (out[0].clone(), out[1].clone(), out[2].clone());
            for i in [0, 1] {
                let layer = tower.build_layer(i).unwrap();
                // One cache-free decoder layer, with the same hard lower/report-only upper policy.
                let mut worst = 0;
                for stalled in [false, true] {
                    for _ in 0..2 * RUNS {
                        let peak = with_cpu_diagnostic_headroom(PLENTY, || {
                            let _scope = mlx_gen::ios_quantum::Scope::new();
                            let ahead = stalled.then(stall);
                            let peak = measured_peak_bytes(|| {
                                Ok(vec![layer.forward(cfg, &h, &cos, &sin)?])
                            });
                            drop(ahead);
                            peak
                        })
                        .unwrap();
                        worst = worst.max(peak);
                    }
                }
                prove(
                    &format!("text streamed layer {i} n{n} {quant:?}"),
                    crate::text_encoder::streamed_layer_bytes(cfg, n as u64),
                    worst,
                );
            }
        }
    }

    // The per-row PiT stage's output rows concatenated into the batch.
    for (b, h, w) in [(2, 16, 24), (4, 32, 32)] {
        let rows: Vec<Array> = (0..b)
            .map(|r| rand(&[1, 3, h, w], 20 + r as u64, Dtype::Float32))
            .collect();
        let (peak, _) = measure(|| {
            let refs: Vec<&Array> = rows.iter().collect();
            Ok(vec![mlx_rs::ops::concatenate_axis(&refs, 0)?])
        });
        prove(
            &format!("output rows b{b} {h}x{w}"),
            crate::dit::IrisDiT::output_rows_bytes(b, 3, h, w),
            peak,
        );
    }
}

#[test]
fn every_streamed_quantum_price_tracks_its_measured_peak() {
    in_cpu_child(
        "quantum_proofs::every_streamed_quantum_price_tracks_its_measured_peak",
        streamed_body,
    );
}
