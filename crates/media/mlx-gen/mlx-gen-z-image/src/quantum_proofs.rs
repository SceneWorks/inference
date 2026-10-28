//! The measured proof of every counted Z-Image iOS quantum price (epic 16774, sc-26433).
//!
//! Each proof builds the REAL module (random weights at proportions where the activations dominate),
//! materializes the quantum's inputs and resident weights, then measures MLX's peak allocation while
//! exactly that guarded quantum builds and evaluates — UNTHROTTLED (no memory limit is ever set; the
//! CPU fixture only enables the iOS guards and barriers), the worst of [`RUNS`] runs after one warm-up
//! ([`measure`]) — and asserts the counted formula is at least the measurement (hard), reporting a
//! ratio above [`MAX_RATIO`] ([`mlx_gen::ios_quantum::assert_tracks_measured`]: report-only; low-end
//! accuracy is calibrated on the device). The proofs price on
//! the backend they measure ([`Backend::Cpu`]: composed RMSNorm, unfused SDPA, explicit-GEMM
//! convolution); the device prices the same counts with Metal's kernels. Activations run at the
//! widths the device runs them: the text encoder and DiT stream f32 (the f32 RoPE tables and pad
//! tokens promote), the VAE f32 (the f32 latent de-normalization promotes); the weight widths cover
//! every tier (dense f32/bf16 storage, packed Q4). All in a CPU child with Metal trapped.

use std::collections::HashMap;

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::ios_quantum::{
    assert_tracks_measured, measured_peak_bytes, with_cpu_diagnostic_headroom, Backend,
};
use mlx_gen::weights::Weights;
use mlx_gen::{Result, WeightsSource};
use mlx_rs::{random, Array, Dtype};

use crate::control_transformer::ZImageControlTransformer;

/// The stated bound: a counted price may exceed the measurement by at most 1.5x.
const MAX_RATIO: f64 = 1.5;
/// Measured runs per proof; the WORST (largest) peak is the one the formula must cover.
const RUNS: usize = 5;
/// The CPU fixture's headroom: every guard passes.
const PLENTY: u64 = 1 << 34;

const CHILD: &str = "ZIMAGE_IOS_QUANTUM_PROOF_CHILD";

fn in_cpu_child(name: &str) -> bool {
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("quantum_proofs::{name}"),
                "--nocapture",
                "--include-ignored",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .env("IOS_CAPTURE_IN_CHILD", "1")
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

/// Measure `run`'s peak, unthrottled, as the largest of [`RUNS`] natural runs and [`RUNS`] stalled
/// ones ([`measure_once`]), after one warm-up run that materializes lazy weights and kernel state.
fn measure(run: impl Fn() -> Result<Vec<Array>>) -> u64 {
    with_cpu_diagnostic_headroom(PLENTY, || {
        let warm = run()?;
        mlx_rs::transforms::eval(warm.iter())?;
        Ok(())
    })
    .unwrap();
    let mut worst = 0;
    for stalled in [false, true] {
        for _ in 0..RUNS {
            worst = worst.max(measure_once(&run, stalled));
        }
    }
    worst
}

/// Queue a long CPU kernel ahead of the measured body on the same stream and hand it back (the
/// caller holds it until the body has evaluated).
fn stall() -> Array {
    thread_local! {
        static OPERAND: Array = {
            let a = random::normal::<f32>(&[STALL_EDGE, STALL_EDGE], None, None, Some(&random::key(99).unwrap())).unwrap();
            a.eval().unwrap();
            a
        };
    }
    OPERAND.with(|a| {
        let s = a.matmul(a).unwrap();
        mlx_rs::transforms::async_eval([&s]).unwrap();
        s
    })
}

/// The stall kernel's edge: one `[E, E]` f32 product keeps the CPU stream's worker busy far longer
/// than building and encoding any quantum's graph takes.
const STALL_EDGE: i32 = 2048;

/// One measurement of `run` on the iOS path (the CPU fixture enables every guard and sub-quantum
/// barrier the device runs, and the device's quantum scope: no buffer cache).
///
/// MLX's scheduler encodes ahead of execution: an op's output is allocated when it is encoded, its
/// inputs stay referenced until its queued task (CPU) or command buffer (Metal) completes, so a
/// phase's intermediates can all be live at once — the retention every counted price follows. The
/// plain run is [`measured_peak_bytes`], which stalls the stream ahead of the body only. A STALLED
/// run here also queues a long
/// kernel ([`stall`]) on the same stream ahead of the body and after every sub-quantum barrier
/// ([`crate::quantum::BARRIER_HOOK`]), so each phase is encoded — its buffers allocated, none
/// released — before its first op runs: the retention worst case, realized on the CPU. Each phase's
/// peak is read from its own start, net of the stall kernels' outputs. Never measured under a
/// memory limit, which would drain the work in flight before each op.
fn measure_once(run: &dyn Fn() -> Result<Vec<Array>>, stalled: bool) -> u64 {
    use std::cell::RefCell;
    use std::rc::Rc;

    #[derive(Default)]
    struct Stalls {
        ahead: Vec<Array>,
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
            return measured_peak_bytes(run);
        }
        mlx_rs::memory::clear_cache();
        let state = Rc::new(RefCell::new(Stalls {
            base: mlx_rs::memory::get_active_memory() as u64,
            ..Stalls::default()
        }));
        state.borrow_mut().issue();
        let hooked = Rc::clone(&state);
        crate::quantum::BARRIER_HOOK.with(|hook| {
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
            .and_then(|o| Ok(mlx_rs::transforms::eval(o.iter())?));
        crate::quantum::BARRIER_HOOK.with(|hook| hook.borrow_mut().take());
        let outputs = outputs?;
        evaluated?;
        state.borrow_mut().record();
        drop(outputs);
        let worst = state.borrow().worst;
        Ok(worst)
    })
    .unwrap()
}

/// Assert `formula` tracks the worst measured peak of `run`, printing both — and that the guarded
/// code under measurement CHECKS exactly `formula` (its first quantum's price), so the proven
/// formula is the one the device enforces, not a parallel copy.
fn prove(label: &str, formula: u64, run: impl Fn() -> Result<Vec<Array>>) {
    let checked = first_check(&run);
    assert_eq!(
        checked,
        Some(formula),
        "{label}: the guarded quantum checks a different price than the proven formula"
    );
    prove_composite(label, formula, run);
}

/// [`prove`] for a run of several quanta whose formula is their composite (a tiled stage and its
/// tiles; a prepare quantum and the embed quantum after it).
fn prove_composite(label: &str, formula: u64, run: impl Fn() -> Result<Vec<Array>>) {
    let measured = measure(run);
    eprintln!(
        "PROOF {label}: formula {formula} B, worst measured {measured} B, ratio {:.3}",
        formula as f64 / measured.max(1) as f64
    );
    if std::env::var_os("ZIMAGE_PROOF_PRINT_ONLY").is_none() {
        assert_tracks_measured(label, formula, measured, MAX_RATIO);
    }
}

/// The price of the first quantum `run` checks, under the CPU fixture.
fn first_check(run: &dyn Fn() -> Result<Vec<Array>>) -> Option<u64> {
    let seen = std::rc::Rc::new(std::cell::RefCell::new(None));
    let sink = seen.clone();
    with_cpu_diagnostic_headroom(PLENTY, || {
        mlx_gen::ios_quantum::with_cpu_diagnostic_observer(
            move |e| {
                if let mlx_gen::ios_quantum::QuantumEvent::Checked { scratch, .. } = e {
                    sink.borrow_mut().get_or_insert(*scratch);
                }
            },
            run,
        )
    })
    .unwrap();
    let first = *seen.borrow();
    first
}

fn rnd(shape: &[i32], seed: u64, scale: f32) -> Array {
    let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
        .unwrap()
        .multiply(Array::from_f32(scale))
        .unwrap();
    a.eval().unwrap();
    a
}

/// Seeded random module weights, keyed as the checkpoints store them.
pub(crate) struct Gen {
    map: HashMap<String, Array>,
    seed: u64,
}

impl Gen {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            map: HashMap::new(),
            seed,
        }
    }
    fn put(&mut self, key: String, shape: &[i32], scale: f32) {
        self.seed += 1;
        self.map.insert(key, rnd(shape, self.seed, scale));
    }
    fn ones(&mut self, key: String, c: i32) {
        self.map.insert(key, Array::ones::<f32>(&[c]).unwrap());
    }
    fn linear(&mut self, p: &str, out: i32, inp: i32, bias: bool) {
        self.put(
            format!("{p}.weight"),
            &[out, inp],
            (1.0 / inp as f32).sqrt(),
        );
        if bias {
            self.put(format!("{p}.bias"), &[out], 0.02);
        }
    }
    /// A Z-Image DiT block (adaLN when `adaln`, a context block otherwise).
    pub(crate) fn dit_block(&mut self, p: &str, dim: i32, heads: i32, ff: i32, adaln: bool) {
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            self.linear(&format!("{p}.attention.{n}"), dim, dim, false);
        }
        for n in ["norm_q", "norm_k"] {
            self.ones(format!("{p}.attention.{n}.weight"), dim / heads);
        }
        for n in [
            "attention_norm1",
            "attention_norm2",
            "ffn_norm1",
            "ffn_norm2",
        ] {
            self.ones(format!("{p}.{n}.weight"), dim);
        }
        self.linear(&format!("{p}.feed_forward.w1"), ff, dim, false);
        self.linear(&format!("{p}.feed_forward.w3"), ff, dim, false);
        self.linear(&format!("{p}.feed_forward.w2"), dim, ff, false);
        if adaln {
            self.linear(
                &format!("{p}.adaLN_modulation.0"),
                4 * dim,
                dim.min(256),
                true,
            );
        }
    }
    /// A Qwen3 decoder layer.
    fn te_layer(&mut self, p: &str, hidden: i32, heads: i32, kv: i32, hd: i32, inter: i32) {
        self.ones(format!("{p}.input_layernorm.weight"), hidden);
        self.ones(format!("{p}.post_attention_layernorm.weight"), hidden);
        self.linear(&format!("{p}.self_attn.q_proj"), heads * hd, hidden, false);
        self.linear(&format!("{p}.self_attn.k_proj"), kv * hd, hidden, false);
        self.linear(&format!("{p}.self_attn.v_proj"), kv * hd, hidden, false);
        self.linear(&format!("{p}.self_attn.o_proj"), hidden, heads * hd, false);
        self.ones(format!("{p}.self_attn.q_norm.weight"), hd);
        self.ones(format!("{p}.self_attn.k_norm.weight"), hd);
        self.linear(&format!("{p}.mlp.gate_proj"), inter, hidden, false);
        self.linear(&format!("{p}.mlp.up_proj"), inter, hidden, false);
        self.linear(&format!("{p}.mlp.down_proj"), hidden, inter, false);
    }
    /// An NHWC conv `[out, k, k, in]` + bias (as the VAE loader leaves it after its remap).
    fn conv(&mut self, p: &str, out: i32, inp: i32, k: i32) {
        self.put(
            format!("{p}.weight"),
            &[out, k, k, inp],
            (1.0 / (inp * k * k) as f32).sqrt(),
        );
        self.put(format!("{p}.bias"), &[out], 0.02);
    }
    fn norm(&mut self, p: &str, c: i32) {
        self.ones(format!("{p}.weight"), c);
        self.map
            .insert(format!("{p}.bias"), Array::zeros::<f32>(&[c]).unwrap());
    }
    fn resnet(&mut self, p: &str, cin: i32, cout: i32) {
        self.norm(&format!("{p}.norm1"), cin);
        self.conv(&format!("{p}.conv1"), cout, cin, 3);
        self.norm(&format!("{p}.norm2"), cout);
        self.conv(&format!("{p}.conv2"), cout, cout, 3);
        if cin != cout {
            self.conv(&format!("{p}.conv_shortcut"), cout, cin, 1);
        }
    }
    fn vae_attention(&mut self, p: &str, c: i32) {
        self.norm(&format!("{p}.group_norm"), c);
        for n in ["to_q", "to_k", "to_v", "to_out.0"] {
            self.linear(&format!("{p}.{n}"), c, c, true);
        }
    }
    /// Cast every tensor to `dtype` (a bf16-stored tier).
    fn cast(&mut self, dtype: Dtype) {
        for a in self.map.values_mut() {
            *a = a.as_dtype(dtype).unwrap();
            a.eval().unwrap();
        }
    }
    pub(crate) fn weights(&self) -> Weights {
        Weights::from_map(self.map.clone())
    }
}

/// A DiT block's RoPE table `[T, hd/2, 2]` (f32, as `RopeEmbedder` builds it).
fn freqs(tokens: i32, head_dim: i32, seed: u64) -> Array {
    rnd(&[tokens, head_dim / 2, 2], seed, 1.0)
}

/// The weight storages every tier runs: dense f32, dense bf16, packed Q4 (quantized in memory and
/// materialized — the layout a packed snapshot loads).
#[derive(Clone, Copy, Debug)]
enum Tier {
    F32,
    Bf16,
    Q4,
}
const TIERS: [Tier; 3] = [Tier::F32, Tier::Bf16, Tier::Q4];

#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn dit_block_formulas_track_the_measured_peak() {
    if !in_cpu_child("dit_block_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_proofs::dit_block_formulas_track_the_measured_peak");
        return;
    }
    let (dim, heads, ff, t) = (128, 2, 384, 256);
    let backend = Backend::current();
    assert_eq!(backend, Backend::Cpu);
    for tier in TIERS {
        for compiled in [false, true] {
            let _glue = compiled.then(crate::CompileGlueGuard::enable);
            let mut g = Gen::new(10);
            g.dit_block("b", dim, heads, ff, true);
            g.dit_block("c", dim, heads, ff, false);
            if matches!(tier, Tier::Bf16) {
                g.cast(Dtype::Bfloat16);
            }
            let w = g.weights();
            let cfg = crate::transformer_block::ZImageBlockConfig {
                dim,
                n_heads: heads,
                norm_eps: 1e-5,
            };
            let mut block =
                crate::transformer_block::ZImageTransformerBlock::from_weights(&w, "b", cfg)
                    .unwrap();
            let mut context =
                crate::context_block::ZImageContextBlock::from_weights(&w, "c", dim, heads, 1e-5)
                    .unwrap();
            if matches!(tier, Tier::Q4) {
                block.quantize(4).unwrap();
                context.quantize(4).unwrap();
            }
            for lin in block.projections() {
                lin.materialize_weights().unwrap();
            }
            for lin in [
                &context.attention.to_q,
                &context.attention.to_k,
                &context.attention.to_v,
                &context.attention.to_out,
                &context.feed_forward.w1,
                &context.feed_forward.w2,
                &context.feed_forward.w3,
            ] {
                lin.materialize_weights().unwrap();
            }
            let x = rnd(&[1, t, dim], 1, 1.0);
            let fc = freqs(t, dim / heads, 2);
            let t_emb = rnd(&[1, dim], 3, 1.0);
            for (plan_name, plan) in [
                ("unbounded", AttentionPlan::UNBOUNDED),
                (
                    "chunked",
                    AttentionPlan::budgeted(AttentionBudget::from_score_elements(
                        (heads * 64 * t) as u64,
                        true,
                    )),
                ),
            ] {
                let tag = format!("{tier:?} compiled={compiled} {plan_name}");
                prove(
                    &format!("DiT block {tag}"),
                    block.quantum_scratch(&x, &fc, plan, backend),
                    || Ok(vec![block.forward_quantum(&x, &fc, &t_emb, plan, 0, Ok)?]),
                );
                prove(
                    &format!("DiT context block {tag}"),
                    context.quantum_scratch(&x, &fc, plan, backend),
                    || Ok(vec![context.forward_quantum(&x, &fc, plan)?]),
                );
            }
        }
    }

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_proofs::dit_block_formulas_track_the_measured_peak");
    }
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_proofs::dit_block_formulas_track_the_measured_peak");
}

/// A DiT at the shipped proportions in miniature: dim 128 (2 heads x 64), FFN x3, a 64-wide
/// caption, 16 latent channels, patch 2 — streamable from `path` when given.
pub(crate) fn proof_dit_cfg() -> crate::transformer::ZImageTransformerConfig {
    crate::transformer::ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 16,
        dim: 128,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 2,
        norm_eps: 1e-5,
        cap_feat_dim: 64,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![16, 24, 24],
        axes_lens: vec![256, 64, 64],
        frequency_embedding_size: 256,
    }
}

impl Gen {
    /// Every DiT tensor of [`proof_dit_cfg`] under `w.` (plus a 15+2 block ControlNet when
    /// `control`), keyed as the checkpoints store them after the loader's remap.
    pub(crate) fn dit(&mut self, control: bool) {
        let c = proof_dit_cfg();
        let (d, h, ff) = (c.dim, c.n_heads, 3 * c.dim);
        let patch = c.patch_size * c.patch_size * c.f_patch_size * c.in_channels;
        self.linear("w.all_x_embedder.2-1", d, patch, true);
        self.ones("w.cap_embedder.0.weight".into(), c.cap_feat_dim);
        self.linear("w.cap_embedder.1", d, c.cap_feat_dim, true);
        self.put("w.cap_pad_token".into(), &[1, d], 0.1);
        self.put("w.x_pad_token".into(), &[1, d], 0.1);
        self.linear(
            "w.t_embedder.linear1",
            512,
            c.frequency_embedding_size,
            true,
        );
        self.linear("w.t_embedder.linear2", d.min(256), 512, true);
        for i in 0..c.n_refiner_layers {
            self.dit_block(&format!("w.noise_refiner.{i}"), d, h, ff, true);
            self.dit_block(&format!("w.context_refiner.{i}"), d, h, ff, false);
        }
        for i in 0..c.n_layers {
            self.dit_block(&format!("w.layers.{i}"), d, h, ff, true);
        }
        self.linear(
            "w.all_final_layer.2-1.adaLN_modulation.0",
            d,
            d.min(256),
            true,
        );
        self.linear("w.all_final_layer.2-1.linear", patch, d, true);
        if control {
            let cpatch = c.patch_size * c.patch_size * c.f_patch_size * crate::CONTROL_IN_DIM;
            self.linear("w.control_all_x_embedder.2-1", d, cpatch, true);
            for (stack, n) in [("control_layers", 15), ("control_noise_refiner", 2)] {
                for i in 0..n {
                    let p = format!("w.{stack}.{i}");
                    self.dit_block(&p, d, h, ff, true);
                    self.linear(&format!("{p}.after_proj"), d, d, true);
                    if i == 0 {
                        self.linear(&format!("{p}.before_proj"), d, d, true);
                    }
                }
            }
        }
    }
}

/// `n` evaluated caption features `[n, cap_dim]`.
fn caption(n: i32, seed: u64) -> Array {
    rnd(&[n, proof_dit_cfg().cap_feat_dim], seed, 1.0)
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn dit_stage_formulas_track_the_measured_peak() {
    if !in_cpu_child("dit_stage_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_proofs::dit_stage_formulas_track_the_measured_peak");
        return;
    }
    let backend = Backend::current();
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let mut g = Gen::new(20);
        g.dit(false);
        if dtype == Dtype::Bfloat16 {
            g.cast(Dtype::Bfloat16);
        }
        let w = g.weights();
        let dit =
            crate::transformer::ZImageTransformer::from_weights(&w, "w", proof_dit_cfg()).unwrap();
        w.materialize().unwrap();
        // 32x32 latent → 256 image tokens; a 20-token caption (padded to 32).
        let x = rnd(&[16, 1, 32, 32], 3, 1.0)
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        x.eval().unwrap();
        let cap = caption(20, 4);
        let dims = (16, 1, 32, 32);
        let tag = format!("{dtype:?}");
        prove(
            &format!("DiT prepare {tag}"),
            dit.prepare_bytes(dims, &cap),
            || {
                let p = dit.prepare(dims, &cap)?;
                Ok(vec![
                    p.cap_tokens,
                    p.x_keep,
                    p.cap_keep,
                    p.x_freqs,
                    p.cap_freqs,
                    p.unified_freqs,
                ])
            },
        );
        let prep = dit.prepare(dims, &cap).unwrap();
        prove(
            &format!("DiT image embed {tag}"),
            dit.embed_bytes(&x, prep.img_pad, backend),
            || {
                let (t, e) = dit.embed_quantum(&prep, &x, 0.6)?;
                Ok(vec![t, e])
            },
        );
        prove(
            &format!("DiT caption embed {tag}"),
            dit.caption_bytes(&prep.cap_tokens, backend),
            || Ok(vec![dit.caption_quantum(&prep)?]),
        );
        let (t_emb, x_emb) = dit.embed_quantum(&prep, &x, 0.6).unwrap();
        let cap_emb = dit.caption_quantum(&prep).unwrap();
        mlx_rs::transforms::eval([&t_emb, &x_emb, &cap_emb]).unwrap();
        prove(
            &format!("DiT unify {tag}"),
            crate::transformer::unify_bytes(&x_emb, &cap_emb),
            || Ok(vec![dit.unify_quantum(&x_emb, &cap_emb)?]),
        );
        let unified = dit.unify_quantum(&x_emb, &cap_emb).unwrap();
        unified.eval().unwrap();
        let x_len = x_emb.shape()[1];
        prove(
            &format!("DiT output {tag}"),
            dit.output_bytes(&unified, backend),
            || {
                Ok(vec![dit.output_quantum(
                    &unified,
                    &t_emb,
                    x_len,
                    prep.x_size,
                )?])
            },
        );
    }

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_proofs::dit_stage_formulas_track_the_measured_peak");
    }
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_proofs::dit_stage_formulas_track_the_measured_peak");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn text_encoder_formulas_track_the_measured_peak() {
    if !in_cpu_child("text_encoder_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_proofs::text_encoder_formulas_track_the_measured_peak");
        return;
    }
    let backend = Backend::current();
    // Qwen3-4B proportions in miniature: hidden 64, q 8x16 = 128 (2x), kv 2x16 = 32, MLP 256 (4x).
    let (hidden, heads, kv, hd, inter, t) = (64, 8, 2, 16, 256, 128);
    for tier in TIERS {
        let mut g = Gen::new(30);
        g.te_layer("l", hidden, heads, kv, hd, inter);
        if matches!(tier, Tier::Bf16) {
            g.cast(Dtype::Bfloat16);
        }
        let w = g.weights();
        let mut layer =
            crate::text_encoder::EncoderLayer::from_weights(&w, "l", heads, kv, hd, 1e-6).unwrap();
        if matches!(tier, Tier::Q4) {
            layer.quantize(4).unwrap();
        }
        w.materialize().unwrap();
        let x = rnd(&[1, t, hidden], 1, 1.0);
        let rope = mlx_gen::nn::TextRope::new(hd, 1_000_000.0);
        let (cos, sin) = rope.forward(t).unwrap();
        let mask = rnd(&[1, 1, t, t], 2, 1.0);
        mlx_rs::transforms::eval([&cos, &sin]).unwrap();
        prove(
            &format!("Qwen3 layer {tier:?}"),
            layer.quantum_scratch(1, t, 4, backend),
            || {
                Ok(vec![crate::text_encoder::encoder::layer_quantum(
                    &layer, &x, &cos, &sin, &mask, 0,
                )?])
            },
        );
    }
    // The embed quantum at the shipped proportion of hidden to prompt length (2560 : 512), over a
    // dense f32, a dense bf16 and a packed Q4 table.
    let (vocab, hidden, s_len) = (1024, 320, 64);
    for tier in TIERS {
        let mut g = Gen::new(40);
        g.put("model.embed_tokens.weight".into(), &[vocab, hidden], 0.05);
        g.te_layer("model.layers.0", hidden, 4, 2, 80, 2 * hidden);
        if matches!(tier, Tier::Bf16) {
            g.cast(Dtype::Bfloat16);
        }
        let w = g.weights();
        let cfg = crate::text_encoder::ZTextEncoderConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            n_layers: 1,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 80,
            intermediate_size: 2 * hidden,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        };
        let mut te = crate::text_encoder::TextEncoder::from_weights(&w, "model", &cfg).unwrap();
        if matches!(tier, Tier::Q4) {
            te.quantize(4).unwrap();
        }
        w.materialize().unwrap();
        let ids = Array::from_slice(
            &(0..s_len).map(|i| (i * 7) % vocab).collect::<Vec<i32>>(),
            &[1, s_len],
        );
        let am = Array::from_slice(&vec![1i32; s_len as usize], &[1, s_len]);
        let (e, ..) = te.embed_quantum(&ids, &am).unwrap();
        e.eval().unwrap();
        prove(
            &format!("Qwen3 embed {tier:?}"),
            te.embed_bytes(1, s_len),
            || {
                let (e, c, s, m) = te.embed_quantum(&ids, &am)?;
                Ok(vec![e, c, s, m])
            },
        );
    }

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_proofs::text_encoder_formulas_track_the_measured_peak");
    }
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_proofs::text_encoder_formulas_track_the_measured_peak");
}

/// Save `map` (keys sorted) as one `.safetensors` file at `path`.
fn save(map: &HashMap<String, Array>, path: &std::path::Path) {
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    Array::save_safetensors(
        keys.iter()
            .map(|k| (k.as_str(), &map[*k]))
            .collect::<Vec<_>>(),
        None,
        path,
    )
    .unwrap();
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn control_and_streamed_formulas_track_the_measured_peak() {
    if !in_cpu_child("control_and_streamed_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_proofs::control_and_streamed_formulas_track_the_measured_peak");
        return;
    }
    let backend = Backend::current();
    let cfg = proof_dit_cfg();
    let (d, h) = (cfg.dim, cfg.n_heads);
    // The published ControlNet is dense bf16; the base is a packed tier.
    let mut g = Gen::new(50);
    g.dit(true);
    g.cast(Dtype::Bfloat16);
    let dir = tempfile::tempdir().unwrap();
    let control_file = dir.path().join("control.safetensors");
    save(&g.map, &control_file);
    let w = g.weights();
    let base =
        crate::transformer::ZImageTransformer::from_weights(&w, "w", proof_dit_cfg()).unwrap();
    let control = ZImageControlTransformer::from_weights(base, &w, "w").unwrap();
    w.materialize().unwrap();

    // The control embed quantum, measured inside `prepare_control` (whose DiT prepare quantum
    // precedes it; its outputs are live while the embed runs).
    let cap = caption(20, 4);
    let cc = rnd(&[crate::CONTROL_IN_DIM, 1, 32, 32], 5, 1.0);
    let dims = (16, 1, 32, 32);
    let pad = 0;
    prove_composite(
        "ControlNet prepare + embed",
        control.base.prepare_bytes(dims, &cap) + control.control_embed_bytes(&cc, pad),
        || {
            let p = control.prepare_control(dims, &cap, &cc)?;
            Ok(vec![p.c_emb])
        },
    );

    // The control blocks (block 0 seeds the branch with `before_proj`), f32 control state over a
    // bf16-stored dense branch, and the same blocks streamed from the checkpoint with the load-time
    // Q4 packing replayed inside the quantum.
    let t = 256 + 32;
    let c = rnd(&[1, t, d], 6, 1.0);
    let x_base = rnd(&[1, t, d], 7, 1.0);
    let fc = freqs(t, d / h, 8);
    let t_emb = rnd(&[1, d], 9, 1.0);
    let plan = AttentionPlan::UNBOUNDED;
    let blocks = [&control.control_layers[0], &control.control_layers[1]];
    for (i, block) in blocks.into_iter().enumerate() {
        prove(
            &format!("ControlNet block {i} resident bf16"),
            block.quantum_scratch(&c, &x_base, &fc, plan, backend),
            || {
                let (c, hint) = crate::control_transformer::control_block_quantum(
                    block,
                    i,
                    c.clone(),
                    &x_base,
                    &fc,
                    &t_emb,
                    plan,
                    0,
                )?;
                Ok(vec![c, hint])
            },
        );
    }
    let mut stream = crate::block_stream::ZImageBlockStream::new(
        WeightsSource::File(control_file.clone()),
        "w.control_layers",
        cfg.block_cfg(),
        15,
        true,
    );
    stream.set_quant_bits(4);
    for i in 0..2 {
        let mut view = stream.open().unwrap();
        let (block, extra) = stream.materialize_control(&mut view, i).unwrap();
        let formula = block.quantum_scratch(&c, &x_base, &fc, plan, backend) + extra;
        drop((block, view));
        prove(
            &format!("ControlNet block {i} streamed Q4"),
            formula,
            || {
                let mut view = stream.open()?;
                let (block, extra) = stream.materialize_control(&mut view, i)?;
                let (c, hint) = crate::control_transformer::control_block_quantum(
                    &block,
                    i,
                    c.clone(),
                    &x_base,
                    &fc,
                    &t_emb,
                    plan,
                    extra,
                )?;
                Ok(vec![c, hint])
            },
        );
    }

    // A resident dense ControlNet block's load-time Q4 packing.
    let mut probe = crate::control_transformer_block::ZImageControlBlock::from_weights(
        &w,
        "w.control_layers.0",
        cfg.block_cfg(),
        true,
    )
    .unwrap();
    let formula = probe.quantize_bytes(4);
    probe.quantize_quantum(4).unwrap();
    prove("ControlNet load-time Q4 packing", formula, || {
        let mut block = crate::control_transformer_block::ZImageControlBlock::from_weights(
            &w,
            "w.control_layers.0",
            cfg.block_cfg(),
            true,
        )?;
        block.quantize_quantum(4)?;
        Ok(block.weight_arrays().into_iter().cloned().collect())
    });

    // A base block streamed from a packed Q4 file: its stored bytes are read inside its quantum.
    let mut packed = HashMap::new();
    for (k, a) in &g.map {
        if !k.starts_with("w.layers.0.") {
            continue;
        }
        let k = k.clone();
        if k.ends_with(".weight") && a.ndim() == 2 && a.shape()[1] % 64 == 0 {
            let (wq, scales, biases) = mlx_rs::ops::quantize(a, 64, 4).unwrap();
            let base = k.trim_end_matches(".weight");
            packed.insert(format!("{base}.weight"), wq);
            packed.insert(format!("{base}.scales"), scales);
            packed.insert(format!("{base}.biases"), biases);
        } else {
            packed.insert(k, a.clone());
        }
    }
    let packed_file = dir.path().join("packed.safetensors");
    save(&packed, &packed_file);
    let stream = crate::block_stream::ZImageBlockStream::new(
        WeightsSource::File(packed_file),
        "w.layers",
        cfg.block_cfg(),
        1,
        false,
    );
    let mut view = stream.open().unwrap();
    let (block, extra) = stream.materialize_counted(&mut view, 0).unwrap();
    assert!(
        block.projections()[0].is_quantized(),
        "the stream must load packed"
    );
    let formula = block.quantum_scratch(&c, &fc, plan, backend) + extra;
    drop((block, view));
    prove("DiT block streamed packed Q4", formula, || {
        let mut view = stream.open()?;
        let (block, extra) = stream.materialize_counted(&mut view, 0)?;
        Ok(vec![
            block.forward_quantum(&c, &fc, &t_emb, plan, extra, Ok)?
        ])
    });

    // A LoRA-adapted block: the forward-time residuals ride inside the block's quantum.
    let mut lora_block = crate::transformer_block::ZImageTransformerBlock::from_weights(
        &g.weights(),
        "w.layers.1",
        cfg.block_cfg(),
    )
    .unwrap();
    for path in [["attention", "to_q"], ["feed_forward", "w1"]] {
        let lin = mlx_gen::adapters::AdaptableHost::adaptable_mut(&mut lora_block, &path).unwrap();
        let out = lin.base_shape()[0];
        let inp = lin.base_shape()[1];
        lin.push(mlx_gen::adapters::Adapter::Lora {
            a: rnd(&[inp, 16], 11, 0.1),
            b: rnd(&[16, out], 12, 0.1),
            scale: 0.8,
        });
    }
    for lin in lora_block.projections() {
        lin.materialize_weights().unwrap();
    }
    prove(
        "DiT block LoRA bf16",
        lora_block.quantum_scratch(&c, &fc, plan, backend),
        || {
            Ok(vec![
                lora_block.forward_quantum(&c, &fc, &t_emb, plan, 0, Ok)?
            ])
        },
    );

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_proofs::control_and_streamed_formulas_track_the_measured_peak");
    }
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_proofs::control_and_streamed_formulas_track_the_measured_peak");
}

/// Every adapter kind's forward-time residual chain ([`crate::quantum::adapter_bytes`]) on one adapted
/// projection: LoRA, a dense LoKr whose bf16 delta casts to the f32 stream, and a structured LoKr
/// with bf16 factors — each over the projection's own output.
#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn adapter_formulas_track_the_measured_peak() {
    if !in_cpu_child("adapter_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_proofs::adapter_formulas_track_the_measured_peak");
        return;
    }
    use mlx_gen::adapters::{AdaptableLinear, Adapter, LokrFactors};
    let (rows, width) = (1024, 256);
    let x = rnd(&[rows, width], 70, 1.0);
    let bf16 = |a: Array| {
        let a = a.as_dtype(Dtype::Bfloat16).unwrap();
        a.eval().unwrap();
        a
    };
    let adapters = [
        (
            "LoRA",
            Adapter::Lora {
                a: rnd(&[width, 16], 71, 0.1),
                b: rnd(&[16, width], 72, 0.1),
                scale: 0.8,
            },
        ),
        (
            "dense LoKr (bf16 delta)",
            Adapter::Lokr {
                delta: bf16(rnd(&[width, width], 73, 0.1)),
                scale: 0.8,
            },
        ),
        (
            "structured LoKr (bf16 factors)",
            Adapter::LokrStructured {
                factors: LokrFactors {
                    w1: bf16(rnd(&[16, 16], 74, 0.1)),
                    w2: bf16(rnd(&[16, 16], 75, 0.1)),
                    a: 16,
                    b: 16,
                    c: 16,
                    d: 16,
                    scale: 0.8,
                },
            },
        ),
    ];
    for (label, adapter) in adapters {
        let mut lin = AdaptableLinear::dense(rnd(&[width, width], 76, 0.1), None);
        lin.push(adapter);
        lin.materialize_weights().unwrap();
        let lins = [&lin];
        let formula = crate::quantum::ab(4, &[rows as u64, width as u64])
            + crate::quantum::promoted_weight_bytes(&lins, 4)
            + crate::quantum::adapter_bytes(&lins, rows as u64, 4);
        prove_composite(&format!("adapter {label}"), formula, || {
            Ok(vec![lin.forward(&x)?])
        });
    }

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!(
            "\nIOS_CAPTURE_BODY_COMPLETED quantum_proofs::adapter_formulas_track_the_measured_peak"
        );
    }
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_proofs::adapter_formulas_track_the_measured_peak"
    );
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn vae_formulas_track_the_measured_peak() {
    if !in_cpu_child("vae_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_proofs::vae_formulas_track_the_measured_peak");
        return;
    }
    let backend = Backend::current();
    let act = 4u64;
    let mut g = Gen::new(60);
    g.resnet("r0", 64, 64);
    g.resnet("r1", 64, 32);
    g.vae_attention("a", 32);
    g.conv("up.conv", 64, 64, 3);
    g.conv("down.conv", 64, 64, 3);
    g.conv("in.conv", 32, 3, 3);
    g.conv("down32.conv", 32, 32, 3);
    let w = g.weights();
    let r0 = crate::vae::ResnetBlock2D::from_weights(&w, "r0").unwrap();
    let r1 = crate::vae::ResnetBlock2D::from_weights(&w, "r1").unwrap();
    let attn = crate::vae::VaeAttention::from_weights(&w, "a").unwrap();
    let up = crate::vae::UpSampler::from_weights(&w, "up").unwrap();
    let down = crate::vae::DownSampler::from_weights(&w, "down").unwrap();
    let conv_in = crate::vae::conv_layers::ConvLayer::from_weights(&w, "in").unwrap();
    let down32 = crate::vae::DownSampler::from_weights(&w, "down32").unwrap();
    w.materialize().unwrap();
    let x = rnd(&[1, 64, 64, 64], 1, 1.0);
    prove("VAE resnet", r0.quantum_scratch(&x, backend), || {
        Ok(vec![r0.forward_quantum(&x)?])
    });
    prove(
        "VAE resnet + shortcut",
        r1.quantum_scratch(&x, backend),
        || Ok(vec![r1.forward_quantum(&x)?]),
    );
    prove(
        "VAE upsample",
        {
            let call =
                mlx_gen::ios_quantum::ConvCall::new(&[1, 128, 128, 64], act, up.weight(), 1, 1);
            mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, backend)
        },
        || Ok(vec![up.forward_quantum(&x)?]),
    );
    prove(
        "VAE downsample",
        {
            let call =
                mlx_gen::ios_quantum::ConvCall::new(&[1, 65, 65, 64], act, down.weight(), 2, 0);
            mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, backend)
        },
        || Ok(vec![down.forward_quantum(&x)?]),
    );
    // The mid attention unchunked (N = 1024) and, on iOS, query-chunked (N = 8192: 16 Mi score
    // elements per chunk).
    for (hh, ww) in [(32, 32), (64, 128)] {
        let xa = rnd(&[1, 32, hh, ww], 2, 1.0);
        let n = hh * ww;
        let plan = crate::vae::VaeAttention::bounded_plan();
        let rows = plan.budget.query_block(1, 1, n, n);
        prove(
            &format!("VAE attention {n} tokens, {rows} rows per call"),
            mlx_gen::ios_quantum::vae_attention_bytes(1, n, 32, rows, act, backend),
            || Ok(vec![attn.forward_quantum(&xa)?]),
        );
    }
    // The layer-tiled resnet: its stage quantum plus the largest tile it checks (the stage's buffers
    // are live while each tile runs).
    let edge = 24;
    let stage = mlx_gen::ios_quantum::tiled_stage_bytes(&[1, 64, 64, 64], 64, 32, act, true);
    let resnet_tile = mlx_gen::ios_quantum::TileSource {
        full_width: 64,
        normalized: true,
    };
    let tile = mlx_gen::ios_quantum::tile_bytes(
        &[1, edge, edge, 64],
        resnet_tile,
        act,
        &w.require("r1.conv1.weight").unwrap().clone(),
        false,
        backend,
    )
    .max(mlx_gen::ios_quantum::tile_bytes(
        &[1, edge, edge, 32],
        resnet_tile,
        act,
        &w.require("r1.conv2.weight").unwrap().clone(),
        false,
        backend,
    ));
    prove_composite("VAE tiled resnet stage + tile", stage + tile, || {
        Ok(vec![r1.forward_tiled(&x, edge, None)?])
    });
    // The bounded encode (sc-26433): the tiled `conv_in` over a full-resolution image, and the
    // tiled stride-2 downsample — each stage plus the largest tile it checks (an exact-halo crop:
    // `2·core + 1` input pixels a side for the downsample).
    let image = rnd(&[1, 3, 128, 128], 3, 1.0);
    let edge = 32;
    prove_composite(
        "VAE tiled encode conv_in stage + tile",
        mlx_gen::ios_quantum::tiled_stage_bytes(&[1, 128, 128, 3], 3, 32, act, false)
            + mlx_gen::ios_quantum::tile_bytes(
                &[1, edge, edge, 3],
                mlx_gen::ios_quantum::TileSource {
                    full_width: 128,
                    normalized: false,
                },
                act,
                conv_in.weight(),
                false,
                backend,
            ),
        || Ok(vec![conv_in.forward_tiled(&image, edge, None)?]),
    );
    // The downsample at a stage-dominated shape (8-pixel output cores over a 128x128x32 input: the
    // row strips and the assembly carry the peak) and a tile-dominated one (one 32-pixel core: its
    // 65x65 padded crop, output and kernel transient) — the shapes SDXL proves the shared prices at.
    let wide = rnd(&[1, 32, 128, 128], 4, 1.0);
    prove_composite(
        "VAE tiled downsample stage + tile (many tiles)",
        mlx_gen::ios_quantum::downsample_stage_bytes(&[1, 64, 64, 32], act)
            + mlx_gen::ios_quantum::downsample_tile_bytes(
                &[1, 17, 17, 32],
                act,
                down32.weight(),
                backend,
            ),
        || Ok(vec![down32.forward_tiled(&wide, 8, None)?]),
    );
    prove_composite(
        "VAE tiled downsample stage + tile (one tile)",
        mlx_gen::ios_quantum::downsample_stage_bytes(&[1, 32, 32, 64], act)
            + mlx_gen::ios_quantum::downsample_tile_bytes(
                &[1, 65, 65, 64],
                act,
                down.weight(),
                backend,
            ),
        || Ok(vec![down.forward_tiled(&x, 32, None)?]),
    );

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!(
            "\nIOS_CAPTURE_BODY_COMPLETED quantum_proofs::vae_formulas_track_the_measured_peak"
        );
    }
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_proofs::vae_formulas_track_the_measured_peak"
    );
}

/// A decoder that returns a fixed (materialized) decode, so `decode_batch`'s own pixel-conversion
/// quantum is what a measurement sees.
struct FixedDecode(Array);
impl mlx_gen::LatentDecoder for FixedDecode {
    fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
        Some(&mlx_gen::gen_core::FLUX1_LATENT_SPACE)
    }
    fn decode(&self, _latents: &Array) -> Result<Array> {
        Ok(self.0.clone())
    }
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn pipeline_and_vae_head_formulas_track_the_measured_peak() {
    if !in_cpu_child("pipeline_and_vae_head_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_proofs::pipeline_and_vae_head_formulas_track_the_measured_peak");
        return;
    }
    let backend = Backend::current();
    // Latents: noise, bf16 cast and the img2img blend at a 512² request.
    let req = mlx_gen::GenerationRequest {
        prompt: "p".into(),
        width: 512,
        height: 512,
        count: 1,
        ..Default::default()
    };
    let scheduler = mlx_gen::FlowMatchEuler::for_static_shift(4, 3.0);
    let clean = rnd(&[16, 1, 64, 64], 1, 1.0);
    let elements = 16 * 64 * 64;
    for (name, clean) in [("txt2img", None), ("img2img", Some(&clean))] {
        let latents = crate::quantum::latents_bytes(elements, clean.is_some());
        prove(&format!("latents {name}"), latents, || {
            crate::pipeline::denoise_batch(&scheduler, clean, 1, 7, &req, &mut |_| {}, |x, _, _| {
                Ok(x)
            })
        });
    }
    // The decoded image's f32 cast and pixel conversion (f32 and bf16 decodes).
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let decoded = rnd(&[1, 3, 1, 256, 256], 2, 0.5).as_dtype(dtype).unwrap();
        decoded.eval().unwrap();
        let formula = crate::quantum::zimage_pixel_conversion_bytes(
            decoded.size() as u64,
            decoded.item_size() as u64,
        );
        let dec = FixedDecode(decoded);
        let lat = rnd(&[16, 1, 32, 32], 3, 1.0);
        prove(&format!("decoded to image {dtype:?}"), formula, || {
            let images = crate::pipeline::decode_batch(
                &dec,
                None,
                vec![lat.clone()],
                &mlx_gen::CancelFlag::default(),
                &mut |_| {},
            )?;
            assert_eq!(images.len(), 1);
            Ok(vec![])
        });
    }
    // The init image upload (a resize from 300² to 256²).
    let image = mlx_gen::Image {
        width: 300,
        height: 300,
        pixels: (0..300 * 300 * 3).map(|i| ((i * 37) % 251) as u8).collect(),
    };
    prove("init image", crate::quantum::ab(4, &[256, 256, 3]), || {
        Ok(vec![crate::pipeline::init_image_quantum(&image, 256, 256)?])
    });
    // The control context's mask/inpaint channels and concatenation.
    let control_latents = rnd(&[16, 1, 64, 64], 4, 1.0);
    prove(
        "control context",
        crate::quantum::ab(4, &[1, 1, 64, 64])
            + crate::quantum::ab(4, &[16, 1, 64, 64])
            + crate::quantum::ab(4, &[33, 1, 64, 64]),
        || {
            Ok(vec![crate::pipeline::control_context_quantum(
                &control_latents,
            )?])
        },
    );
    // The caption features' valid-token gather and bf16 cast at Qwen3's 2560 width.
    let enc = rnd(&[1, 64, 2560], 5, 1.0);
    prove(
        "caption features",
        crate::quantum::ab(4, &[40]) + crate::quantum::ab(4, &[40, 2560]),
        || {
            Ok(vec![crate::pipeline::caption_quantum(&enc, 40, |e| {
                crate::pipeline::slice_valid(e, 40)
            })?])
        },
    );
    let cap = rnd(&[40, 2560], 6, 1.0);
    prove(
        "caption cast",
        crate::quantum::ab(2, &[cap.size() as u64]),
        || Ok(vec![crate::pipeline::caption_cast_quantum(&cap)?]),
    );

    // The VAE decode head (de-normalization + conv_in) and output head, and the encoder's conv_in
    // and output head + latent mean, over the tiny VAE fixture (decoder conv_in 16 → 64, conv_out
    // 32 → 3; encoder conv_in 3 → 32, conv_out 64 → 32).
    let vae = crate::quantum_tests::vae();
    let latent = rnd(&[1, 16, 64, 64], 7, 1.0)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    latent.eval().unwrap();
    let conv_in = |w: &str| {
        Weights::from_file(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/vae_decoder.safetensors"),
        )
        .unwrap()
        .require(w)
        .unwrap()
        .clone()
    };
    let call = mlx_gen::ios_quantum::ConvCall::new(
        &[1, 64, 64, 16],
        4,
        &conv_in("conv_in.conv.weight"),
        1,
        1,
    );
    prove(
        "VAE decode conv_in + de-normalization",
        mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, backend)
            + 2 * crate::quantum::tb(4, &[latent.size() as u64]),
        || Ok(vec![vae.decode_in_quantum(&latent)?]),
    );
    let h = rnd(&[1, 32, 128, 128], 8, 1.0);
    let call = mlx_gen::ios_quantum::ConvCall::new(
        &[1, 128, 128, 32],
        4,
        &conv_in("conv_out.conv.weight"),
        1,
        1,
    );
    prove(
        "VAE decode conv_out",
        mlx_gen::ios_quantum::norm_conv_bytes(&call, backend),
        || Ok(vec![vae.decoder().out_quantum(&h)?]),
    );
    let encoder = vae.encoder().unwrap();
    let image = rnd(&[1, 3, 256, 256], 9, 1.0);
    prove(
        "VAE encode conv_in",
        {
            let w = encoder_conv_weight(encoder, true);
            let call = mlx_gen::ios_quantum::ConvCall::new(&[1, 256, 256, 3], 4, &w, 1, 1);
            mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, backend)
        },
        || Ok(vec![encoder.conv_in_quantum(&image)?]),
    );
    let h = rnd(&[1, 64, 32, 32], 10, 1.0);
    prove(
        "VAE encode conv_out + latent mean",
        {
            let w = encoder_conv_weight(encoder, false);
            let call = mlx_gen::ios_quantum::ConvCall::new(&[1, 32, 32, 64], 4, &w, 1, 1);
            mlx_gen::ios_quantum::norm_conv_bytes(&call, backend)
                + crate::vae::Vae::latent_mean_bytes(&image)
        },
        || {
            Ok(vec![encoder.head(
                &h,
                crate::vae::Vae::latent_mean_bytes(&image),
                |o| vae.latent_mean(o),
            )?])
        },
    );

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_proofs::pipeline_and_vae_head_formulas_track_the_measured_peak");
    }
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_proofs::pipeline_and_vae_head_formulas_track_the_measured_peak");
}

/// The encoder's `conv_in` (`first`) or `conv_out` weight.
fn encoder_conv_weight(encoder: &crate::vae::Encoder, first: bool) -> Array {
    encoder.conv_weight(first).clone()
}
