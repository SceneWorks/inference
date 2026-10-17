//! sc-26270: SDXL iOS bounded-quantum guards, exercised on tiny random-weight models on MLX's CPU
//! device inside a child process that makes any Metal use fatal.
//!
//! For every guarded path family (U-Net plain / IP / ControlNet / rung-4 streamed, CLIP text, CLIP
//! vision + IP Resampler, VAE decode dense / evaluated / tiled, VAE encode, component loads) the
//! tests prove: guarded execution is byte-identical to the unguarded host path; every quantum is
//! checked before it is synchronized, quanta never interleave, and there is exactly one checked
//! quantum per block / layer / tile of the fixture; a refusal happens before the refused quantum
//! is constructed (nothing after it is checked or synchronized, and the typed record names it);
//! and the scratch is derived from the loaded tensors and grows with the request geometry.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use mlx_gen::ios_quantum::{
    take_refusal, with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
    PROCESS_RESERVE_BYTES,
};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::{random, Array, Dtype};

use crate::config::{ClipActivation, ClipTextConfig, UNetConfig};
use crate::quantum::Backend;
use crate::quantum::*;
use crate::unet::UNet2DConditionModel;

/// Headroom no tiny fixture can exhaust.
const PLENTY: u64 = 1 << 34;

/// Re-run test `name` in a child process pinned to the CPU device with Metal made fatal. Returns
/// `true` inside the child (run the body), `false` in the parent (the child already asserted).
fn cpu_child(name: &str) -> bool {
    const CHILD: &str = "SDXL_IOS_QUANTUM_CPU_CHILD";
    if std::env::var_os(CHILD).is_some() {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        assert!(crate::test_metal_guard::forbid_metal());
        assert!(crate::test_metal_guard::metal_forbidden());
        return true;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            &format!("quantum_tests::{name}"),
            "--nocapture",
            "--include-ignored",
        ])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success(), "CPU child for {name} failed");
    false
}

/// A quantum event with the (per-run) stream index dropped, so two runs compare.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Ev {
    Checked(String, u64),
    Synced(String),
}

fn norm(events: &[QuantumEvent]) -> Vec<Ev> {
    events
        .iter()
        .map(|e| match e {
            QuantumEvent::Checked { label, scratch } => Ev::Checked(label.clone(), *scratch),
            QuantumEvent::Synchronized { label, .. } => Ev::Synced(label.clone()),
        })
        .collect()
}

/// Run `body` with the CPU fixture enabled at `headroom`, recording every quantum event.
fn observe<T>(headroom: u64, body: impl FnOnce() -> Result<T>) -> (Result<T>, Vec<Ev>) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = events.clone();
    let result = with_cpu_diagnostic_headroom(headroom, || {
        with_cpu_diagnostic_observer(move |e| sink.borrow_mut().push(e.clone()), body)
    });
    let events = norm(&events.borrow());
    (result, events)
}

fn bits(a: &Array) -> Vec<u32> {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    a.eval().unwrap();
    a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
}

fn checked(events: &[Ev], label: &str) -> Vec<u64> {
    events
        .iter()
        .filter_map(|e| match e {
            Ev::Checked(l, s) if l == label => Some(*s),
            _ => None,
        })
        .collect()
}

fn synced(events: &[Ev], label: &str) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, Ev::Synced(l) if l == label))
        .count()
}

/// The per-core synchronization label of `mlx_gen::vae_tiling`'s tiled convolutions.
const TILED_CORE: &str = "tiled conv core";

/// Every synchronized quantum was checked first, quanta never interleave, every tile check lies
/// inside an open tiled VAE stage, and nothing is left open.
fn assert_well_ordered(events: &[Ev]) {
    assert!(!events.is_empty(), "the fixture must run guarded quanta");
    let mut open: Option<&str> = None;
    for (i, e) in events.iter().enumerate() {
        match e {
            Ev::Checked(label, _) if label == VAE_TILE => assert_eq!(
                open,
                Some(VAE_TILED_STAGE),
                "tile check #{i} outside a tiled stage: {events:?}"
            ),
            Ev::Checked(label, _) => {
                assert!(
                    open.is_none(),
                    "{label} checked while {open:?} is still open: {events:?}"
                );
                open = Some(label);
            }
            // `mlx_gen::vae_tiling` synchronizes and drains each tiled core itself, inside the
            // open tiled stage, after that tile's own check.
            Ev::Synced(label) if label == TILED_CORE => assert_eq!(
                open,
                Some(VAE_TILED_STAGE),
                "tiled core synchronized outside a tiled stage: {events:?}"
            ),
            Ev::Synced(label) => {
                assert_eq!(
                    open,
                    Some(label.as_str()),
                    "{label} synchronized without its own preceding check: {events:?}"
                );
                open = None;
            }
        }
    }
    assert!(open.is_none(), "{open:?} never synchronized");
}

/// Refusal before construction, twice: (1) with zero headroom the FIRST quantum refuses and no
/// event at all is observed; (2) at a headroom that admits every earlier quantum but not the first
/// one needing more scratch than all before it, the refusal names THAT quantum, carries its scratch,
/// and the observed events are exactly the full run's prefix up to (not including) its check — so
/// nothing after it was checked, built or synchronized. (2) needs such a quantum to exist (a path
/// whose first quantum is its widest has none); returns whether (2) ran.
fn assert_refusals<T>(full: &[Ev], run: impl Fn(u64) -> (Result<T>, Vec<Ev>)) -> bool {
    let _ = take_refusal();
    let first = full
        .iter()
        .find_map(|e| match e {
            Ev::Checked(l, _) => Some(l.clone()),
            Ev::Synced(_) => None,
        })
        .expect("a checked quantum");
    let (result, events) = run(0);
    assert!(result.is_err(), "zero headroom must refuse");
    assert_eq!(take_refusal().map(|r| r.label), Some(first));
    assert!(
        events.is_empty(),
        "nothing may run past a first-quantum refusal: {events:?}"
    );

    let checks: Vec<(usize, &str, u64)> = full
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Ev::Checked(l, s) => Some((i, l.as_str(), *s)),
            Ev::Synced(_) => None,
        })
        .collect();
    let Some((k, max_prev)) = (1..checks.len()).find_map(|k| {
        let max_prev = checks[..k].iter().map(|c| c.2).max().unwrap();
        (checks[k].2 > max_prev).then_some((k, max_prev))
    }) else {
        return false;
    };
    let (index, label, scratch) = checks[k];
    let (result, events) = run(PROCESS_RESERVE_BYTES + max_prev);
    assert!(result.is_err(), "{label} must refuse");
    let record = take_refusal().expect("a typed refusal record");
    assert_eq!(record.label, label);
    assert_eq!(record.scratch_bytes, scratch);
    assert_eq!(
        record.available_bytes,
        Some(PROCESS_RESERVE_BYTES + max_prev)
    );
    assert_eq!(
        events,
        full[..index],
        "the run must stop exactly before the refused {label}"
    );
    true
}

fn rnd(shape: &[i32], seed: u64, scale: f32) -> Array {
    let key = random::key(seed).unwrap();
    mlx_rs::ops::multiply(
        random::normal::<f32>(shape, None, None, Some(&key)).unwrap(),
        Array::from_f32(scale),
    )
    .unwrap()
}

/// `(latent, text memory, pooled, time ids)` for the tiny U-Net at `batch` rows of `h x w`.
fn unet_inputs(batch: i32, h: i32, w: i32) -> (Array, Array, Array, Array) {
    use crate::test_support::{TINY_CONTEXT_DIM, TINY_POOLED_DIM};
    let out = (
        rnd(&[batch, h, w, 4], 11, 1.0),
        rnd(&[batch, 5, TINY_CONTEXT_DIM], 12, 1.0),
        rnd(&[batch, TINY_POOLED_DIM], 13, 1.0),
        crate::pipeline::text_time_ids(batch),
    );
    mlx_rs::transforms::eval([&out.0, &out.1, &out.2, &out.3]).unwrap();
    out
}

fn resnet_count(cfg: &UNetConfig) -> usize {
    let l: i32 = cfg.layers_per_block.iter().sum();
    (l + 2 + l + cfg.num_blocks() as i32) as usize
}

/// The structural quantum counts of one U-Net forward.
fn assert_unet_counts(events: &[Ev], unet: &UNet2DConditionModel, cfg: &UNetConfig) {
    let depths = unet.transformer_stack_depths();
    let n = cfg.num_blocks();
    for (label, want) in [
        (UNET_STEM, 1),
        (UNET_RESNET, resnet_count(cfg)),
        (TRANSFORMER_PROJ_IN, depths.len()),
        (TRANSFORMER_BLOCK, depths.iter().sum()),
        (TRANSFORMER_PROJ_OUT, depths.len()),
        (UNET_DOWNSAMPLE, n - 1),
        (UNET_UPSAMPLE, n - 1),
        (UNET_CONV_OUT, 1),
    ] {
        assert_eq!(checked(events, label).len(), want, "{label} checks");
        assert_eq!(synced(events, label), want, "{label} synchronizations");
    }
}

#[test]
fn unet_forward_is_bounded_identical_and_refuses_before_construction() {
    if !cpu_child("unet_forward_is_bounded_identical_and_refuses_before_construction") {
        return;
    }
    let cfg = crate::test_support::tiny_unet_config();
    let unet = crate::test_support::tiny_unet(7).unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let forward = || unet.forward(&x, 500.0, &ctx, &pooled, &ids);
    let host = forward().unwrap();
    let (guarded, events) = observe(PLENTY, forward);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert_unet_counts(&events, &unet, &cfg);
    // The stem is `conv_in` (output, bias add, CPU kernel transient) + the time embedding MLPs +
    // the three latent-sized lazy buffers of the forward's input, all from the loaded weights.
    assert_eq!(
        checked(&events, UNET_STEM),
        [crate::unet::stem_bytes(
            &x,
            &ctx,
            unet.conv_in_weight(),
            None,
            0,
            Backend::Cpu
        )]
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, forward
    )));
}

#[test]
fn unet_ip_forward_prices_the_ip_tokens_and_stays_identical() {
    if !cpu_child("unet_ip_forward_prices_the_ip_tokens_and_stays_identical") {
        return;
    }
    let cfg = crate::test_support::tiny_unet_config();
    let mut unet = crate::test_support::tiny_unet(7).unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let (_, plain) = observe(PLENTY, || unet.forward(&x, 500.0, &ctx, &pooled, &ids));
    let layers: usize = unet.transformer_stack_depths().iter().sum();
    let pairs = (0..layers)
        .map(|i| {
            let c = 64; // every tiny-U-Net attention runs at the inner 64-channel level
            (
                rnd(
                    &[c, crate::test_support::TINY_CONTEXT_DIM],
                    100 + i as u64,
                    0.1,
                ),
                rnd(
                    &[c, crate::test_support::TINY_CONTEXT_DIM],
                    200 + i as u64,
                    0.1,
                ),
            )
        })
        .collect();
    unet.install_ip_adapter(pairs).unwrap();
    let tokens = rnd(&[1, 3, crate::test_support::TINY_CONTEXT_DIM], 31, 1.0);
    let forward = || unet.forward_with_ip(&x, 500.0, &ctx, &pooled, &ids, (&tokens, 0.7));
    let host = forward().unwrap();
    let (guarded, events) = observe(PLENTY, forward);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert_unet_counts(&events, &unet, &cfg);
    // The IP branch prices its three extra keys and its output in the cross-attention phase: the
    // first block (16 tokens of the 4x4 grid) at 5 text + 3 IP keys with the IP buffers — never
    // below the plain block, whose widest phase may be its FFN.
    let (with_ip, without) = (
        checked(&events, TRANSFORMER_BLOCK),
        checked(&plain, TRANSFORMER_BLOCK),
    );
    assert_eq!(with_ip.len(), without.len());
    assert_eq!(
        with_ip[0],
        transformer_block_bytes(
            1,
            1,
            16,
            64,
            2,
            32,
            5 + 3,
            4 * 64,
            16,
            true,
            4,
            Backend::Cpu
        )
    );
    assert!(
        with_ip.iter().zip(&without).all(|(a, b)| a >= b),
        "{with_ip:?} vs {without:?}"
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, forward
    )));
}

#[test]
fn controlnet_branch_and_injection_are_bounded_and_identical() {
    if !cpu_child("controlnet_branch_and_injection_are_bounded_and_identical") {
        return;
    }
    let cfg = crate::test_support::tiny_control_unet_config();
    let unet = UNet2DConditionModel::from_weights(
        &crate::test_support::tiny_unet_weights_for(&cfg, 3).unwrap(),
        &cfg,
    )
    .unwrap();
    let control = crate::unet::ControlNet::from_weights(
        &crate::test_support::tiny_controlnet_weights(&cfg, 5).unwrap(),
        &cfg,
    )
    .unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let image = rnd(&[1, 64, 64, 3], 41, 0.5);
    image.eval().unwrap();
    let run = || -> Result<Vec<Array>> {
        let cond = control.embed_cond(&image)?;
        let residuals = control.forward(&x, &cond, 500.0, &ctx, &pooled, &ids, 0.8)?;
        let eps = unet.forward_with_control(&x, 500.0, &ctx, &pooled, &ids, &residuals)?;
        let mut out = residuals.down;
        out.push(residuals.mid);
        out.push(eps);
        Ok(out)
    };
    let host = run().unwrap();
    let (guarded, events) = observe(PLENTY, run);
    let guarded = guarded.unwrap();
    assert_eq!(host.len(), guarded.len());
    for (a, b) in host.iter().zip(&guarded) {
        assert_eq!(bits(a), bits(b));
    }
    assert_well_ordered(&events);
    for (label, want) in [
        (CONTROL_COND_CONV, 8),
        (CONTROL_STEM, 1),
        (CONTROL_ZERO_CONV, 10),
        (CONTROL_DOWN_INJECTION, 1),
        (CONTROL_MID_INJECTION, 1),
        (UNET_STEM, 1),
        (UNET_CONV_OUT, 1),
        // The ControlNet runs the encoder copy's down + mid resnets; the U-Net all of its own.
        (UNET_RESNET, 6 + 2 + resnet_count(&cfg)),
    ] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    assert_refusals(&events, |headroom| observe(headroom, run));
    // The full-resolution conditioning embedding is this chain's widest quantum, so prove the
    // mid-path refusal on the U-Net-with-control half (stem → … → injections) on its own.
    let cond = control.embed_cond(&image).unwrap();
    let residuals = control
        .forward(&x, &cond, 500.0, &ctx, &pooled, &ids, 0.8)
        .unwrap();
    let inject = || unet.forward_with_control(&x, 500.0, &ctx, &pooled, &ids, &residuals);
    let (_, inject_events) = observe(PLENTY, inject);
    assert_well_ordered(&inject_events);
    assert!(assert_refusals(&inject_events, |headroom| observe(
        headroom, inject
    )));
}

#[test]
fn streamed_transformer_blocks_are_checked_before_materialization() {
    if !cpu_child("streamed_transformer_blocks_are_checked_before_materialization") {
        return;
    }
    let cfg = crate::test_support::tiny_unet_config();
    let weights = crate::test_support::tiny_unet_weights(7).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("unet.safetensors");
    let keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    let named: Vec<(&str, &Array)> = keys
        .iter()
        .map(|k| (k.as_str(), weights.get(k).unwrap()))
        .collect();
    Array::save_safetensors(named, None, &path).unwrap();
    let resident = UNet2DConditionModel::from_weights(&weights, &cfg).unwrap();
    let mut streamed =
        UNet2DConditionModel::from_weights(&Weights::from_file(&path).unwrap(), &cfg).unwrap();
    streamed.arm_block_streams(
        &mlx_gen::WeightsSource::File(path.clone()),
        Dtype::Float32,
        None,
    );
    assert!(streamed.can_stream_blocks());
    let (x, ctx, pooled, ids) = unet_inputs(1, 8, 8);
    let cancel = mlx_gen::CancelFlag::default();
    let plan =
        crate::plan::SdxlForwardPlan::UNBOUNDED.with_window(Some(crate::plan::SdxlBlockWindow {
            size: 1,
            cancel: &cancel,
        }));
    let forward = || streamed.forward_planned(&x, 500.0, &ctx, &pooled, &ids, None, None, plan);
    let host = forward().unwrap();
    assert_eq!(
        bits(&host),
        bits(&resident.forward(&x, 500.0, &ctx, &pooled, &ids).unwrap())
    );
    let (guarded, events) = observe(PLENTY, forward);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert_unet_counts(&events, &streamed, &cfg);
    // The streamed block's floor is its resident twin's: same loaded shapes, same request.
    let (_, resident_events) = observe(PLENTY, || resident.forward(&x, 500.0, &ctx, &pooled, &ids));
    assert_eq!(
        checked(&events, TRANSFORMER_BLOCK),
        checked(&resident_events, TRANSFORMER_BLOCK)
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, forward
    )));
}

/// The block floor is `attention_block_bytes` over THIS block's loaded tensors (non-production
/// widths: 64 channels, 2 heads, a 4x GEGLU, 16-wide context) and the request (tokens, context
/// tokens, chunk rows) — and it grows with the latent grid.
#[test]
fn block_and_resnet_floors_come_from_loaded_tensors_and_grow_with_geometry() {
    if !cpu_child("block_and_resnet_floors_come_from_loaded_tensors_and_grow_with_geometry") {
        return;
    }
    let unet = crate::test_support::tiny_unet(7).unwrap();
    let observe_at = |h: i32, w: i32| {
        let (x, ctx, pooled, ids) = unet_inputs(1, h, w);
        observe(PLENTY, || unet.forward(&x, 500.0, &ctx, &pooled, &ids)).1
    };
    let small = observe_at(8, 8);
    let large = observe_at(16, 16);
    // 64 channels, 2 heads of 32, a 4x GEGLU, 16 latent tokens at the 4x4 grid, 5 context tokens,
    // f32 activations, the CPU backend (unfused SDPA).
    let expected =
        transformer_block_bytes(1, 1, 16, 64, 2, 32, 5, 4 * 64, 16, false, 4, Backend::Cpu);
    // The first block is down_blocks.1's at the 4x4 grid (16 tokens) of an 8x8 latent.
    assert_eq!(checked(&small, TRANSFORMER_BLOCK)[0], expected);
    for label in [TRANSFORMER_BLOCK, UNET_RESNET, UNET_STEM, UNET_CONV_OUT] {
        let (s, l) = (checked(&small, label), checked(&large, label));
        assert!(
            s.iter().zip(&l).all(|(a, b)| b > a),
            "{label}: {s:?} vs {l:?}"
        );
    }
    // A resnet's floor uses its loaded conv channels: the first up resnet (4x4 grid) concatenates
    // the 64-channel mid output with a 64-channel skip.
    // (Weights shaped like the loaded `[out, 3, 3, in]` convs: the price reads only their shapes.)
    let conv1 = Array::zeros::<f32>(&[64, 3, 3, 128]).unwrap();
    let conv2 = Array::zeros::<f32>(&[64, 3, 3, 64]).unwrap();
    assert_eq!(
        checked(&small, UNET_RESNET)[4],
        resnet_bytes(&[1, 4, 4, 64], 4, &conv1, &conv2, Backend::Cpu)
    );
    // A bounded attention plan is priced at its chunk's query rows: at a 16x16 latent the attention
    // grid is 64 tokens; 160 scores per chunk gives 16 cross-attention (5 keys x 2 heads) rows and
    // one self-attention row, so the block prices the wider score domain's chunk (16 rows) with
    // the chunk outputs and their concatenation.
    let budget = mlx_gen::attention::AttentionBudget::from_score_elements(160, false);
    assert_eq!(budget.query_block(1, 2, 64, 5), 16);
    let (x, ctx, pooled, ids) = unet_inputs(1, 16, 16);
    let plan = crate::plan::SdxlForwardPlan::with_attention(
        mlx_gen::attention::AttentionPlan::budgeted(budget),
    );
    let (bounded_out, chunked) = observe(PLENTY, || {
        unet.forward_planned(&x, 500.0, &ctx, &pooled, &ids, None, None, plan)
    });
    bounded_out.unwrap();
    let (full, bounded) = (
        checked(&large, TRANSFORMER_BLOCK),
        checked(&chunked, TRANSFORMER_BLOCK),
    );
    assert_eq!(full.len(), bounded.len());
    assert_eq!(
        bounded[0],
        transformer_block_bytes(1, 1, 64, 64, 2, 32, 5, 4 * 64, 16, false, 4, Backend::Cpu)
    );
    assert_eq!(
        full[0],
        transformer_block_bytes(1, 1, 64, 64, 2, 32, 5, 4 * 64, 64, false, 4, Backend::Cpu)
    );
}

// ---------------------------------------------------------------------------------------------
// CLIP text encoders.
// ---------------------------------------------------------------------------------------------

struct Gen {
    map: HashMap<String, Array>,
    seed: u64,
}

impl Gen {
    fn new(seed: u64) -> Self {
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
    fn ln(&mut self, p: &str, c: i32) {
        self.ones(format!("{p}.weight"), c);
        self.put(format!("{p}.bias"), &[c], 0.02);
    }
    fn clip_layers(&mut self, p: &str, layers: i32, d: i32, mlp: i32) {
        for i in 0..layers {
            let l = format!("{p}.encoder.layers.{i}");
            self.ln(&format!("{l}.layer_norm1"), d);
            self.ln(&format!("{l}.layer_norm2"), d);
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                self.linear(&format!("{l}.self_attn.{proj}"), d, d, true);
            }
            self.linear(&format!("{l}.mlp.fc1"), mlp, d, true);
            self.linear(&format!("{l}.mlp.fc2"), d, mlp, true);
        }
    }
    fn weights(self) -> Weights {
        Weights::from_map(self.map)
    }
}

/// A tiny CLIP text encoder: `d` wide, three layers, a 3x MLP, optional projection.
fn tiny_clip(d: i32, heads: i32, projection: Option<i32>, seed: u64) -> (Weights, ClipTextConfig) {
    let cfg = ClipTextConfig {
        num_layers: 3,
        model_dims: d,
        num_heads: heads,
        max_length: 8,
        vocab_size: 40,
        projection_dim: projection,
        hidden_act: ClipActivation::Gelu,
    };
    let mut g = Gen::new(seed);
    g.put(
        "text_model.embeddings.token_embedding.weight".into(),
        &[40, d],
        0.5,
    );
    g.put(
        "text_model.embeddings.position_embedding.weight".into(),
        &[8, d],
        0.1,
    );
    g.clip_layers("text_model", 3, d, 3 * d);
    g.ln("text_model.final_layer_norm", d);
    if let Some(p) = projection {
        g.linear("text_projection", p, d, false);
    }
    (g.weights(), cfg)
}

fn clip_ids(n: i32) -> Array {
    let mut ids: Vec<i32> = (0..2 * n).map(|i| 1 + (i * 7) % 30).collect();
    ids[(n - 1) as usize] = 39; // EOS is the max id
    ids[(2 * n - 1) as usize] = 39;
    Array::from_slice(&ids, &[2, n])
}

#[test]
fn clip_text_encoders_are_bounded_per_layer_and_identical() {
    if !cpu_child("clip_text_encoders_are_bounded_per_layer_and_identical") {
        return;
    }
    let (w1, c1) = tiny_clip(24, 3, None, 300);
    let (w2, c2) = tiny_clip(32, 4, Some(12), 400);
    let te1 = crate::text_encoder::ClipTextEncoder::from_weights(&w1, "text_model", &c1).unwrap();
    let te2 = crate::text_encoder::ClipTextEncoder::from_weights(&w2, "text_model", &c2).unwrap();
    let ids = clip_ids(8);
    let encode = || crate::pipeline::encode_conditioning(&te1, &te2, &ids);
    let (cond, pooled) = encode().unwrap();
    let (guarded, events) = observe(PLENTY, encode);
    let (g_cond, g_pooled) = guarded.unwrap();
    assert_eq!(bits(&cond), bits(&g_cond));
    assert_eq!(bits(&pooled), bits(&g_pooled));
    assert_well_ordered(&events);
    for (label, want) in [(CLIP_EMBEDDINGS, 2), (CLIP_LAYER, 6), (CLIP_POOLED, 2)] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    // The layer floor is the loaded widths (3x MLP, configured heads), not CLIP-L/bigG constants.
    let layers = checked(&events, CLIP_LAYER);
    assert_eq!(
        layers[0],
        clip_layer_bytes(2, 8, 24, 72, 3, 8, true, 4, Backend::Cpu)
    );
    assert_eq!(
        layers[3],
        clip_layer_bytes(2, 8, 32, 96, 4, 8, true, 4, Backend::Cpu)
    );
    let short = clip_ids(4);
    let (_, short_events) = observe(PLENTY, || te1.forward(&short).map(|o| o.pooled));
    assert!(checked(&short_events, CLIP_LAYER)[0] < layers[0]);
    assert!(assert_refusals(&events, |headroom| observe(
        headroom, encode
    )));
}

// ---------------------------------------------------------------------------------------------
// IP-Adapter image encoder: CLIP vision tower + Resampler.
// ---------------------------------------------------------------------------------------------

fn tiny_ip_encoder() -> crate::ip_adapter::IpImageEncoder {
    let vcfg = crate::vision_encoder::VisionConfig {
        hidden: 32,
        num_layers: 3,
        num_heads: 4,
        patch: 4,
        image_size: 16,
        num_channels: 3,
        quick_gelu: false,
    };
    let mut g = Gen::new(500);
    g.put(
        "vision_model.embeddings.patch_embedding.weight".into(),
        &[32, 3, 4, 4],
        0.2,
    );
    g.put("vision_model.embeddings.class_embedding".into(), &[32], 0.1);
    g.put(
        "vision_model.embeddings.position_embedding.weight".into(),
        &[17, 32],
        0.1,
    );
    g.ln("vision_model.pre_layrnorm", 32);
    g.clip_layers("vision_model", 3, 32, 64);
    let encoder =
        crate::vision_encoder::ClipVisionEncoder::from_weights(&g.weights(), &vcfg).unwrap();
    let rcfg = crate::ip_adapter::ResamplerConfig {
        dim: 32,
        depth: 2,
        heads: 2,
        dim_head: 16,
        num_queries: 4,
        embed_dim: 32,
        output_dim: 16,
    };
    let mut g = Gen::new(600);
    g.put("image_proj.latents".into(), &[1, 4, 32], 0.5);
    g.linear("image_proj.proj_in", 32, 32, true);
    g.linear("image_proj.proj_out", 16, 32, true);
    g.ln("image_proj.norm_out", 16);
    for i in 0..2 {
        let a = format!("image_proj.layers.{i}.0");
        g.ln(&format!("{a}.norm1"), 32);
        g.ln(&format!("{a}.norm2"), 32);
        g.linear(&format!("{a}.to_q"), 32, 32, false);
        g.linear(&format!("{a}.to_kv"), 64, 32, false);
        g.linear(&format!("{a}.to_out"), 32, 32, false);
        let f = format!("image_proj.layers.{i}.1");
        g.ln(&format!("{f}.0"), 32);
        g.linear(&format!("{f}.1"), 48, 32, false);
        g.linear(&format!("{f}.3"), 32, 48, false);
    }
    let resampler =
        crate::ip_adapter::Resampler::from_weights(&g.weights(), "image_proj", &rcfg).unwrap();
    crate::ip_adapter::IpImageEncoder::with_image_size(encoder, resampler, 16)
}

#[test]
fn ip_image_encoder_is_bounded_per_layer_and_identical() {
    if !cpu_child("ip_image_encoder_is_bounded_per_layer_and_identical") {
        return;
    }
    let encoder = tiny_ip_encoder();
    let image = mlx_gen::media::Image {
        width: 20,
        height: 18,
        pixels: (0..20 * 18 * 3).map(|i| (i * 37 % 251) as u8).collect(),
    };
    let tokens = || encoder.tokens(&image);
    let host = tokens().unwrap();
    let (guarded, events) = observe(PLENTY, tokens);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    for (label, want) in [
        (VISION_EMBEDDINGS, 1),
        // `penultimate` runs every layer but the last.
        (VISION_LAYER, 2),
        (RESAMPLER_PROJ_IN, 1),
        (RESAMPLER_LAYER, 2),
        (RESAMPLER_PROJ_OUT, 1),
    ] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    // 17 tokens (16 patches + class), the loaded 64-wide MLP, four heads of 8.
    assert_eq!(
        checked(&events, VISION_LAYER)[0],
        clip_layer_bytes(1, 17, 32, 64, 4, 8, false, 4, Backend::Cpu)
    );
    // Perceiver layers (CPU backend): 8 key-side buffers over 17 image tokens + 4 queries at the
    // 32-wide inner width (6 + the fallback's k/v copies), 15 query-side buffers (5 + three
    // 3-buffer CPU LayerNorms + the scaled q) with the loaded 48-wide FFN's two, the CPU's two
    // materialized score tensors (2 heads x 4 queries x 21 keys), and one page per buffer.
    assert_eq!(
        checked(&events, RESAMPLER_LAYER)[0],
        (8 * 21 * 32 + 4 * (15 * 32 + 2 * 48) + 2 * 2 * 4 * 21) * 4 + (8 + 15 + 4) * PAGE_BYTES
    );
    // The patch-embedding quantum is this chain's widest, so only the first-quantum refusal applies.
    assert_refusals(&events, |headroom| observe(headroom, tokens));
}

// ---------------------------------------------------------------------------------------------
// VAE decode / encode.
// ---------------------------------------------------------------------------------------------

fn tiny_vae() -> crate::vae::Autoencoder {
    let (w, cfg) = crate::vae::tests::fixture();
    crate::vae::Autoencoder::from_weights(&w, &cfg).unwrap()
}

fn vae_latent() -> Array {
    let z = rnd(&[1, 8, 9, 4], 71, 1.0);
    z.eval().unwrap();
    z
}

/// The dense decode quanta of the tiny VAE: conv_in, mid resnet/attention/resnet, two up blocks of
/// two resnets, one upsample, the output conv.
fn assert_dense_decode_counts(events: &[Ev]) {
    for (label, want) in [
        (VAE_DECODE_IN, 1),
        (VAE_RESNET, 2 + 4),
        (VAE_ATTENTION, 1),
        (VAE_UPSAMPLE, 1),
        (VAE_DECODE_OUT, 1),
    ] {
        assert_eq!(checked(events, label).len(), want, "{label} checks");
        assert_eq!(synced(events, label), want, "{label} synchronizations");
    }
}

#[test]
fn vae_dense_and_evaluated_decode_are_bounded_and_identical() {
    if !cpu_child("vae_dense_and_evaluated_decode_are_bounded_and_identical") {
        return;
    }
    let vae = tiny_vae();
    let z = vae_latent();
    for evaluated in [false, true] {
        let decode = || {
            if evaluated {
                vae.decode_evaluated(&z)
            } else {
                vae.decode(&z)
            }
        };
        let host = decode().unwrap();
        let (guarded, events) = observe(PLENTY, decode);
        assert_eq!(
            bits(&host),
            bits(&guarded.unwrap()),
            "evaluated={evaluated}"
        );
        assert_well_ordered(&events);
        assert_dense_decode_counts(&events);
        // The attention is priced per query chunk at the loaded width: the fixture's 72 latent
        // tokens fit one chunk, so it scores all of them against each other.
        assert_eq!(
            checked(&events, VAE_ATTENTION),
            [vae_attention_bytes(1, 72, 32, 72, 4)]
        );
        assert!(assert_refusals(&events, |headroom| observe(
            headroom, decode
        )));
    }
}

/// The production decode route (`pipeline::decode_image_tiled` through the NCHW decoder seam, both
/// dense and tiled) ends in a guarded pixel conversion and stays byte-identical.
#[test]
fn decode_image_route_is_bounded_end_to_end() {
    if !cpu_child("decode_image_route_is_bounded_end_to_end") {
        return;
    }
    let vae = tiny_vae();
    let z = vae_latent();
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
    for tiled in [None, Some(&tiling)] {
        let decode = || crate::pipeline::decode_image_tiled(&vae, &z, None, tiled, None);
        let host = decode().unwrap();
        let (guarded, events) = observe(PLENTY, decode);
        let guarded = guarded.unwrap();
        assert_eq!((host.width, host.height), (guarded.width, guarded.height));
        assert_eq!(host.pixels, guarded.pixels);
        assert_well_ordered(&events);
        assert_eq!(
            checked(&events, DECODED_TO_IMAGE),
            vec![16 * 18 * 3 * 4 + PAGE_BYTES]
        );
        assert_eq!(synced(&events, DECODED_TO_IMAGE), 1);
        assert_refusals(&events, |headroom| observe(headroom, decode));
    }
}

/// Tiles one layer-wise tiled conv of an `h x w` input cuts at `edge` (the halo-1 core plan of
/// `mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc`).
fn tiles(h: i32, w: i32, edge: i32) -> usize {
    if h <= edge && w <= edge {
        return 1;
    }
    let core = edge - 2;
    (((h + core - 1) / core) * ((w + core - 1) / core)) as usize
}

#[test]
fn vae_tiled_decode_checks_every_tile_before_its_graph() {
    if !cpu_child("vae_tiled_decode_checks_every_tile_before_its_graph") {
        return;
    }
    let vae = tiny_vae();
    let z = vae_latent();
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
    let decode = || vae.decode_tiled(&z, &tiling, None);
    let host = decode().unwrap();
    let (guarded, events) = observe(PLENTY, decode);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    // Output edge 8 over one 2x upsample: block 0 tiles at edge 4 on the 8x9 latent grid (two
    // resnets, two convs each), its upsample conv at edge 8 on 16x18, block 1 likewise at 16x18,
    // then the output conv.
    let want_tiles =
        4 * tiles(8, 9, 4) + tiles(16, 18, 8) + 4 * tiles(16, 18, 8) + tiles(16, 18, 8);
    assert_eq!(checked(&events, VAE_TILE).len(), want_tiles);
    assert_eq!(checked(&events, VAE_TILED_STAGE).len(), 4 + 1 + 1);
    assert_eq!(synced(&events, VAE_TILED_STAGE), 4 + 1 + 1);
    // Each tile's core is synchronized and drained before the next tile's graph is built.
    assert_eq!(synced(&events, TILED_CORE), want_tiles);
    // The head is still dense and guarded.
    for (label, want) in [(VAE_DECODE_IN, 1), (VAE_RESNET, 2), (VAE_ATTENTION, 1)] {
        assert_eq!(checked(&events, label).len(), want, "{label}");
    }
    // A tile's floor is its own extent: the 4-edge tiles are smaller than the 8-edge ones.
    let tile_floors = checked(&events, VAE_TILE);
    assert!(tile_floors[0] < *tile_floors.last().unwrap());
    assert_refusals(&events, |headroom| observe(headroom, decode));
}

#[test]
fn vae_encode_is_bounded_and_identical() {
    if !cpu_child("vae_encode_is_bounded_and_identical") {
        return;
    }
    let vae = tiny_vae();
    let image = rnd(&[1, 16, 18, 3], 81, 0.5);
    image.eval().unwrap();
    let encode = || vae.encode_mean(&image);
    let host = encode().unwrap();
    let (guarded, events) = observe(PLENTY, encode);
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    for (label, want) in [
        (VAE_ENCODE_IN, 1),
        (VAE_RESNET, 2 + 2),
        (VAE_DOWNSAMPLE, 1),
        (VAE_ATTENTION, 1),
        (VAE_ENCODE_OUT, 1),
    ] {
        assert_eq!(checked(&events, label).len(), want, "{label} checks");
        assert_eq!(synced(&events, label), want, "{label} synchronizations");
    }
    assert_refusals(&events, |headroom| observe(headroom, encode));
}

// ---------------------------------------------------------------------------------------------
// Component loads.
// ---------------------------------------------------------------------------------------------

fn save(weights: &Weights, path: &std::path::Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let keys: Vec<String> = weights.keys().map(str::to_owned).collect();
    let named: Vec<(&str, &Array)> = keys
        .iter()
        .map(|k| (k.as_str(), weights.get(k).unwrap()))
        .collect();
    Array::save_safetensors(named, None, path).unwrap();
}

/// The cast-aware load floor for dense `files` loaded at `target` (`width` bytes): every tensor's
/// resident bytes after the cast (its cast copy when its stored dtype differs, its stored bytes
/// otherwise) plus the largest tensor's transient (stored + cast copy, or one stored read).
fn cast_load_floor(
    files: &[&std::path::Path],
    target: mlx_gen::gen_core::weightsmeta::Dtype,
    width: u64,
) -> u64 {
    let (mut total, mut largest) = (0u64, 0u64);
    for h in files
        .iter()
        .flat_map(|f| mlx_gen::gen_core::safetensors_path_tensor_headers(f).unwrap())
    {
        if h.dtype == target {
            total += h.data_bytes;
            largest = largest.max(h.data_bytes);
        } else {
            let cast = h.shape.iter().product::<usize>() as u64 * width;
            total += cast;
            largest = largest.max(cast + h.data_bytes);
        }
    }
    total + largest
}

/// Every component loader checks its header floor before constructing anything; with no headroom
/// it refuses with a typed record naming the component and never reaches construction. The tiny
/// fixtures cannot satisfy the SDXL-base configs those loaders build, so with ample headroom each
/// load records its check FIRST and only then fails in construction — the order under test.
#[test]
fn component_loads_check_their_header_floor_before_construction() {
    if !cpu_child("component_loads_check_their_header_floor_before_construction") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let (vae_w, _) = crate::vae::tests::fixture();
    let vae_file = root.join("vae/diffusion_pytorch_model.safetensors");
    save(&vae_w, &vae_file);
    let (clip_w, _) = tiny_clip(24, 3, None, 300);
    let clip_file = root.join("text_encoder/model.safetensors");
    save(&clip_w, &clip_file);
    let unet_file = root.join("unet/diffusion_pytorch_model.safetensors");
    save(
        &crate::test_support::tiny_unet_weights(7).unwrap(),
        &unet_file,
    );
    let fix_file = root.join("fix/vae.safetensors");
    save(&vae_w, &fix_file);

    let fix = mlx_gen::WeightsSource::File(fix_file.clone());
    let unet_source = mlx_gen::WeightsSource::File(unet_file.clone());
    type Load<'a> = Box<dyn Fn() -> Result<()> + 'a>;
    let cases: Vec<(&str, &std::path::Path, Load)> = vec![
        (
            "SDXL VAE load",
            &vae_file,
            Box::new(|| crate::loader::load_vae(root).map(drop)),
        ),
        (
            "SDXL fp16-fix VAE load",
            &fix_file,
            Box::new(|| crate::loader::load_vae_fp16_fix(&fix).map(drop)),
        ),
        (
            "SDXL text_encoder load",
            &clip_file,
            Box::new(|| crate::loader::load_text_encoder_1_dtype(root, Dtype::Float32).map(drop)),
        ),
        (
            "SDXL UNet load",
            &unet_file,
            Box::new(|| crate::loader::load_unet_dtype(root, Dtype::Float16).map(drop)),
        ),
        (
            "SDXL ControlNet load",
            &unet_file,
            Box::new(|| crate::loader::load_controlnet(&unet_source, Dtype::Float16).map(drop)),
        ),
    ];
    use mlx_gen::gen_core::weightsmeta::Dtype as Stored;
    for (label, file, load) in &cases {
        // Every loader casts after reading (`cast_for_load` / the VAEs' `cast_all`): the f32
        // fixtures loaded at f16 (the fp16-fix VAE, the U-Net, the ControlNet) price their f16
        // copies; the f32 loads (the stock VAE, CLIP here) cast nothing and price the stored bytes.
        let floor = match *label {
            "SDXL fp16-fix VAE load" | "SDXL UNet load" | "SDXL ControlNet load" => {
                cast_load_floor(&[file], Stored::F16, 2)
            }
            _ => cast_load_floor(&[file], Stored::F32, 4),
        };
        let _ = take_refusal();
        let (refused, events) = observe(0, load);
        assert!(refused.is_err(), "{label}");
        let record = take_refusal().expect("typed load refusal");
        assert_eq!(record.label, *label);
        assert_eq!(record.scratch_bytes, floor);
        assert!(events.is_empty());
        let (_, events) = observe(PLENTY, load);
        assert_eq!(
            events.first(),
            Some(&Ev::Checked(label.to_string(), floor)),
            "{label} must be checked before construction: {events:?}"
        );
    }
    // A cast load's floor carries the copies: the f16 copies of the f32 U-Net, the f32 source
    // held with the largest copy — not the stored bytes alone.
    assert!(
        cast_load_floor(&[&unet_file], Stored::F16, 2)
            != cast_load_floor(&[&unet_file], Stored::F32, 4)
    );
}

/// The fused-LDM phase floors count only the components that phase builds.
#[test]
fn fused_checkpoint_floor_counts_only_the_phase_components() {
    if !cpu_child("fused_checkpoint_floor_counts_only_the_phase_components") {
        return;
    }
    use mlx_gen::gen_core::sdxl_ldm::SdxlComponent;
    let tmp = tempfile::tempdir().unwrap();
    let file = tmp.path().join("fused.safetensors");
    let mut g = Gen::new(900);
    g.put(
        "model.diffusion_model.input_blocks.7.0.out_layers.3.weight".into(),
        &[64],
        1.0,
    );
    g.put(
        "first_stage_model.decoder.up.1.block.2.nin_shortcut.weight".into(),
        &[16],
        1.0,
    );
    g.put(
        "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight".into(),
        &[8],
        1.0,
    );
    g.put(
        "conditioner.embedders.1.model.text_projection".into(),
        &[4, 2],
        1.0,
    );
    g.put("unrelated.metadata".into(), &[1000], 1.0);
    save(&g.weights(), &file);
    let floor = |components: &[SdxlComponent]| {
        let _ = take_refusal();
        let refused = with_cpu_diagnostic_headroom(0, || {
            crate::ldm::before_component_load("LDM", &file, components)
        });
        assert!(refused.is_err());
        take_refusal().unwrap().scratch_bytes
    };
    // Resident bytes after each component's cast (the f32 U-Net / CLIP tensors become f16 copies;
    // the VAE stays f32) plus the largest transient (an f32 source held with its f16 copy).
    assert_eq!(
        floor(&[SdxlComponent::ClipL, SdxlComponent::ClipBigG]),
        (16 + 16) + (16 + 32)
    );
    assert_eq!(
        floor(&[SdxlComponent::Unet, SdxlComponent::Vae]),
        (128 + 64) + (128 + 256)
    );
    assert_eq!(floor(&[SdxlComponent::Unet]), 128 + (128 + 256));
    // Off device it reads nothing and passes.
    crate::ldm::before_component_load("LDM", &file, &[SdxlComponent::Unet]).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Conditioning window count (weights-free; iOS admission keys evidence on it).
// ---------------------------------------------------------------------------------------------

/// A two-word CLIP vocabulary under `<root>/tokenizer/` (every `a` / `b` word is one token).
fn tiny_tokenizer_root(root: &std::path::Path) {
    let dir = root.join("tokenizer");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("vocab.json"),
        r#"{"<|startoftext|>": 49406, "<|endoftext|>": 49407, "a</w>": 320, "b</w>": 321}"#,
    )
    .unwrap();
    std::fs::write(dir.join("merges.txt"), "#version: 0.2\n").unwrap();
}

#[test]
fn conditioning_window_count_matches_the_encode_path() {
    if !cpu_child("conditioning_window_count_matches_the_encode_path") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    tiny_tokenizer_root(tmp.path());
    let spec = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::Dir(tmp.path().to_owned()));
    let tokenizer = crate::loader::load_tokenizer(tmp.path()).unwrap();
    let words = |n: usize| vec!["a"; n].join(" ");
    let count = |p: &str, n: Option<&str>| crate::conditioning_window_count(&spec, p, n).unwrap();
    let encoded = |p: &str, n: Option<&str>| tokenizer.tokenize_windows(p, n).unwrap().len() as u32;
    let short = words(10);
    let long = words(80); // 80 content tokens > 75 per window
    let longer = words(160);
    assert_eq!(count(&short, None), 1);
    assert_eq!(count(&short, Some("b b")), 1);
    assert_eq!(count(&long, None), 2);
    assert_eq!(
        count(&short, Some(&long)),
        2,
        "a longer negative raises the count"
    );
    assert_eq!(count(&long, Some(&longer)), 3);
    for (p, n) in [
        (short.as_str(), None),
        (short.as_str(), Some("b")),
        (long.as_str(), None),
        (short.as_str(), Some(long.as_str())),
        (long.as_str(), Some(longer.as_str())),
        (words(75).as_str(), None),
        (words(76).as_str(), Some("")),
    ] {
        assert_eq!(count(p, n), encoded(p, n), "{p:?} / {n:?}");
    }
    // A fused single-file checkpoint resolves the caller-staged tokenizer component, like `load`.
    let mut fused = mlx_gen::LoadSpec::new(mlx_gen::WeightsSource::File(
        tmp.path().join("model.safetensors"),
    ));
    assert!(crate::conditioning_window_count(&fused, &short, None).is_err());
    fused.components.insert(
        crate::LDM_TOKENIZER_COMPONENT.to_owned(),
        mlx_gen::WeightsSource::Dir(tmp.path().to_owned()),
    );
    assert_eq!(
        crate::conditioning_window_count(&fused, &long, None).unwrap(),
        2
    );
}

/// The f16-storage upsample convolution accumulates in f32 (`tiled_conv2d_3x3_nhwc_f32_accum`): its
/// stage prices the stage activations PLUS the f32 weight and bias copies that stay live across all
/// of the stage's tiles. The f32 path has no such copies.
#[test]
fn half_upsample_stage_prices_its_f32_accumulation_copies() {
    if !cpu_child("half_upsample_stage_prices_its_f32_accumulation_copies") {
        return;
    }
    let (b, h, w, cin, cout) = (1, 6, 6, 4, 8);
    let x = Array::ones::<f32>(&[b, h, w, cin]).unwrap();
    let weight = Array::ones::<f32>(&[cout, 3, 3, cin]).unwrap();
    let bias = Array::zeros::<f32>(&[cout]).unwrap();
    let stage = |x: &Array, weight: &Array, bias: &Array| {
        let (_, events) = observe(PLENTY, || {
            crate::vae::upsample_tiled(x, weight, bias, 8, None)
        });
        checked(&events, VAE_TILED_STAGE)
    };
    // The nearest-2x copy (input width) and two output-width images (row strips + assembly), at
    // the doubled extent and the activation width.
    let up = |width: u64| (b * 2 * h * 2 * w) as u64 * width;
    let activations = |width: u64| up(width) * (cin as u64 + 2 * cout as u64);
    assert_eq!(stage(&x, &weight, &bias), [activations(4)]);
    let half = |a: &Array| a.as_dtype(Dtype::Float16).unwrap();
    let wide_copies = (cout * 9 * cin) as u64 * 4 + cout as u64 * 4;
    assert_eq!(
        stage(&half(&x), &half(&weight), &half(&bias)),
        [activations(2) + wide_copies]
    );
}

/// The bounded (tiled) encode against the whole-image encode, held to the tiled-decode parity
/// standard (`vae::tests`): 1e-4 of peak for an f32 VAE against the f32 whole-image encode, 1e-2 of
/// peak for the fp16 VAE against the f32 whole-image encode of the same f16-rounded weights — at
/// tile edges that split the image evenly, raggedly and down to the 3-pixel floor (the stride-2
/// downsample tiles reach the padding row/column at the ragged edges).
#[test]
fn tiled_encode_meets_the_tiled_parity_standard() {
    if !cpu_child("tiled_encode_meets_the_tiled_parity_standard") {
        return;
    }
    let (w, cfg) = crate::vae::tests::fixture();
    let mut rounded = w.clone();
    rounded.cast_all(Dtype::Float16).unwrap();
    let mut half = rounded.clone();
    rounded.cast_all(Dtype::Float32).unwrap();
    half.cast_all(Dtype::Float16).unwrap();
    let reference = crate::vae::Autoencoder::from_weights(&rounded, &cfg).unwrap();
    let half = crate::vae::Autoencoder::from_weights(&half, &cfg).unwrap();
    for (h, wd) in [(16, 18), (17, 23)] {
        let image = rnd(&[1, h, wd, 3], 81, 0.5);
        image.eval().unwrap();
        let dense = reference.encode_mean(&image).unwrap();
        for edge in [3, 5, 8, 9] {
            let tiled = reference.encode_tiled(&image, edge, None).unwrap();
            let delta = relative_max_delta(&dense, &tiled);
            eprintln!("{h}x{wd} edge {edge}: f32 {delta:.3e}");
            assert!(
                delta <= 1e-4,
                "f32 tiled encode at edge {edge} drifted {delta:.3e}"
            );
            let tiled16 = half.encode_tiled(&image, edge, None).unwrap();
            assert_eq!(tiled16.dtype(), Dtype::Float32, "callers receive f32");
            let delta = relative_max_delta(&dense, &tiled16);
            eprintln!("{h}x{wd} edge {edge}: f16 {delta:.3e}");
            assert!(
                delta <= 1e-2,
                "fp16 tiled encode at edge {edge} drifted {delta:.3e}"
            );
        }
        // An image that fits one tile takes the exact whole-image encode.
        assert_eq!(
            bits(&dense),
            bits(&reference.encode_tiled(&image, 32, None).unwrap())
        );
    }
}

/// The bounded encode on the iOS path: `encode_mean` selects it above one tile, every tiled stage
/// and tile is checked before its graph and refuses before construction, guarded execution equals
/// the host's run of the same bounded encode bit for bit, and its mid attention runs
/// query-chunked; off the iOS path `encode_mean` is the whole-image encode. A bounded-decode rung's
/// tiling routes the img2img init encode through the tiled encode (its cancel point proves the
/// route).
#[test]
fn bounded_encode_is_selected_on_ios_guarded_and_routed_by_the_rung() {
    if !cpu_child("bounded_encode_is_selected_on_ios_guarded_and_routed_by_the_rung") {
        return;
    }
    let vae = tiny_vae();
    let tile = crate::vae::IOS_ENCODE_TILE_PX;
    // One more than the iOS tile edge: the encoder's 2x downsample leaves a 133x133 latent
    // (17689 tokens), so the bounded attention chunks.
    let image = rnd(&[1, tile + 10, tile + 10, 3], 91, 0.5);
    image.eval().unwrap();
    let host = vae.encode_tiled(&image, tile, None).unwrap();
    assert!(
        mlx_gen::attention::last_chunk_count() > 1,
        "the bounded encode's mid attention must run query-chunked"
    );
    let (guarded, events) = observe(PLENTY, || vae.encode_mean(&image));
    assert_eq!(bits(&host), bits(&guarded.unwrap()));
    assert_well_ordered(&events);
    assert!(
        !checked(&events, VAE_TILE).is_empty(),
        "iOS encode_mean must tile"
    );
    // conv_in, block 0's resnet, its downsample, block 1's resnet: one tiled stage each.
    assert_eq!(checked(&events, VAE_TILED_STAGE).len(), 4);
    assert_eq!(synced(&events, VAE_TILED_STAGE), 4);
    assert_eq!(
        checked(&events, VAE_ENCODE_IN).len(),
        0,
        "no whole-image conv_in on iOS"
    );
    assert!(assert_refusals(&events, |headroom| observe(
        headroom,
        || vae.encode_mean(&image)
    )));

    // The rung's tiling routes the init encode through the tiled encode: a tripped cancel flag
    // stops it between tiles, while the untiled route ignores it.
    let pixels = mlx_gen::media::Image {
        width: 40,
        height: 32,
        pixels: (0..40 * 32 * 3).map(|i| (i * 31 % 251) as u8).collect(),
    };
    let tripped = mlx_gen::CancelFlag::default();
    tripped.cancel();
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(8, 0);
    let routed = crate::pipeline::encode_init_latents_tiled(
        &vae,
        &pixels,
        40,
        32,
        Some(&tiling),
        Some(&tripped),
    );
    assert!(
        matches!(routed, Err(mlx_gen::Error::Canceled)),
        "{routed:?}"
    );
    crate::pipeline::encode_init_latents_tiled(&vae, &pixels, 40, 32, None, Some(&tripped))
        .unwrap();
}

/// Items 3 and 5 (sc-26270 round 3): every ancestral denoise step's CFG combine + sampler update is
/// materialized inside its own guarded quantum (checked, synchronized, byte-identical to the host),
/// and a sequential CFG half prices its transformer blocks at the ONE latent row it runs — only the
/// text k/v it projects at the pair's batch are pair-sized.
#[test]
fn denoise_steps_are_guarded_and_sequential_cfg_prices_the_latent_batch() {
    if !cpu_child("denoise_steps_are_guarded_and_sequential_cfg_prices_the_latent_batch") {
        return;
    }
    use mlx_gen::gen_core::CfgBatching;
    let unet = crate::test_support::tiny_unet(7).unwrap();
    let euler =
        crate::sampler::EulerSampler::new(&crate::config::DiffusionConfig::sdxl_base(), true)
            .unwrap();
    let sampler = crate::sampler::AncestralEuler::new(&euler, 3, 1000.0).unwrap();
    let (_, ctx, pooled, ids) = unet_inputs(2, 8, 8);
    let latents = rnd(&[1, 8, 8, 4], 21, 1.0);
    latents.eval().unwrap();
    let cancel = mlx_gen::CancelFlag::default();
    let (unet, sampler, cancel) = (&unet, &sampler, &cancel);
    let (ctx, pooled, ids) = (&ctx, &pooled, &ids);
    let run = |batching: CfgBatching| {
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED.with_cfg_batching(batching);
        let d = crate::pipeline::Denoiser::with_plan(unet, sampler, plan);
        let latents = latents.clone();
        move || -> Result<Array> {
            mlx_rs::random::seed(5)?;
            crate::pipeline::denoise(
                &d,
                latents.clone(),
                ctx,
                pooled,
                ids,
                5.0,
                cancel,
                &mut |_| {},
            )
        }
    };
    for batching in [CfgBatching::Batched, CfgBatching::Sequential] {
        let host = run(batching)().unwrap();
        let (guarded, events) = observe(PLENTY, run(batching));
        assert_eq!(bits(&host), bits(&guarded.unwrap()), "{batching:?}");
        assert_well_ordered(&events);
        let steps = checked(&events, SAMPLER_STEP);
        assert_eq!(
            steps.len(),
            3,
            "{batching:?}: one sampler-step quantum per step"
        );
        assert_eq!(synced(&events, SAMPLER_STEP), 3);
        assert!(steps
            .iter()
            .all(|s| *s == sampler_step_bytes(&latents, false)));
    }
    let blocks = |batching| checked(&observe(PLENTY, run(batching)).1, TRANSFORMER_BLOCK);
    let (batched, sequential) = (
        blocks(CfgBatching::Batched),
        blocks(CfgBatching::Sequential),
    );
    assert_eq!(
        sequential.len(),
        2 * batched.len(),
        "two halves per guided step"
    );
    // A sequential half prices one latent row; the batched forward two.
    assert!(
        sequential.iter().max() < batched.iter().min(),
        "sequential {sequential:?} vs batched {batched:?}"
    );
}

/// Item 3 (sc-26270 round 3): the ControlNet branch keeps the request's bounded attention plan, so
/// its attention runs (and its blocks are priced) under the plan's chunking, not unbounded.
#[test]
fn controlnet_keeps_the_bounded_attention_plan() {
    if !cpu_child("controlnet_keeps_the_bounded_attention_plan") {
        return;
    }
    let cfg = crate::test_support::tiny_control_unet_config();
    let control = crate::unet::ControlNet::from_weights(
        &crate::test_support::tiny_controlnet_weights(&cfg, 5).unwrap(),
        &cfg,
    )
    .unwrap();
    let (x, ctx, pooled, ids) = unet_inputs(1, 16, 16);
    let image = rnd(&[1, 128, 128, 3], 41, 0.5);
    image.eval().unwrap();
    let cond = control.embed_cond(&image).unwrap();
    cond.eval().unwrap();
    let run = |plan: crate::plan::SdxlForwardPlan<'_>| {
        mlx_gen::attention::reset_chunk_count();
        let out = control
            .forward_planned(&x, &cond, 500.0, &ctx, &pooled, &ids, 0.8, plan)
            .unwrap();
        (out.mid, mlx_gen::attention::last_chunk_count())
    };
    let (mid, unchunked) = run(crate::plan::SdxlForwardPlan::UNBOUNDED);
    assert_eq!(unchunked, 1, "the unbounded branch runs one SDPA call");
    // 40 scores per chunk over the mid block's 4x4 grid (16 tokens, 2 heads, 5 text keys): its last
    // (cross) attention runs 4-row chunks.
    let budget = mlx_gen::attention::AttentionBudget::from_score_elements(40, true);
    assert_eq!(budget.query_block(1, 2, 16, 5), 4);
    let (bounded_mid, chunks) = run(crate::plan::SdxlForwardPlan::with_attention(
        mlx_gen::attention::AttentionPlan::budgeted(budget),
    ));
    assert!(
        chunks > 1,
        "the ControlNet must run the request's bounded plan, ran {chunks} chunk(s)"
    );
    // Chunked attention is the same attention (fused per chunk): the branch's output holds.
    assert!(relative_max_delta(&mid, &bounded_mid) <= 1e-4);
}

/// The device's convolution transients, modelled from MLX's Metal dispatch (they cannot be measured
/// on the CPU): Winograd only for a 3x3 stride-1 call with 32-aligned channels, `N·H·W ≥ 4096` and
/// `C + O ≥ 256` — the padded input, the transformed input/output tiles and the 64-tap filter, plus
/// the weight's contiguous copy every call makes; and the f32-accumulating upsample tile's promoted
/// crop.
#[test]
fn metal_conv_transients() {
    if !cpu_child("metal_conv_transients") {
        return;
    }
    let w = |o: i32, k: i32, c: i32| {
        Array::zeros::<f32>(&[o, k, k, c])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap()
    };
    // A 1024² level-0 resnet conv: [1, 128, 128, 320] -> 320 at f16.
    let call = ConvCall::new(&[1, 128, 128, 320], 2, &w(320, 3, 320), 1, 1);
    assert!(call.winograd());
    let tiles = 22 * 22; // ⌈128 / 6⌉²
    let padded = 6 * (128_u64 + 2 - 2).div_ceil(6) + 2; // 134
    assert_eq!(
        call.transient(Backend::Metal),
        320 * 9 * 320 * 2 // contiguous weight copy
            + padded * padded * 320 * 2
            + 64 * tiles * 320 * 2 * 2 // transformed input + output
            + 64 * 320 * 320 * 2 // transformed filter
    );
    // Too small (N·H·W = 1024), strided, 1x1, or misaligned: implicit GEMM, the weight copy only.
    for call in [
        ConvCall::new(&[1, 32, 32, 1280], 2, &w(1280, 3, 1280), 1, 1),
        ConvCall::new(&[1, 128, 128, 320], 2, &w(320, 3, 320), 2, 1),
        ConvCall::new(&[1, 128, 128, 320], 2, &w(320, 1, 320), 1, 0),
        ConvCall::new(&[1, 128, 128, 4], 2, &w(320, 3, 4), 1, 1),
    ] {
        assert!(!call.winograd());
        assert_eq!(
            call.transient(Backend::Metal),
            call.weight_elements_for_test() * 2
        );
    }
    // The fp16-fix VAE's f32-accumulating upsample tile prices its f32 promoted crop; the f32 path
    // has none.
    let wt = w(512, 3, 512);
    let tile = [1, 64, 64, 512];
    let crop = 64 * 64 * 512 * 2;
    let wide = ConvCall::new(&tile, 4, &wt, 1, 1);
    assert_eq!(
        tile_bytes(&tile, 2, &wt, true, Backend::Metal),
        // f16 crop (after its SiLU), the f32 promoted crop, the f32 output and core, the f32 call's
        // transients.
        crop + 64 * 64 * 512 * 4 + 2 * wide.out_bytes() + wide.transient(Backend::Metal)
    );
    let narrow = ConvCall::new(&tile, 2, &wt, 1, 1);
    assert_eq!(
        tile_bytes(&tile, 2, &wt, false, Backend::Metal),
        crop + 2 * narrow.out_bytes() + narrow.transient(Backend::Metal)
    );
}

/// `max|a - b| / max|reference|`, at f32.
fn relative_max_delta(reference: &Array, candidate: &Array) -> f32 {
    let f = |a: &Array| {
        let a = a.as_dtype(Dtype::Float32).unwrap();
        a.eval().unwrap();
        a.as_slice::<f32>().to_vec()
    };
    let (r, c) = (f(reference), f(candidate));
    let peak = r.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(peak > 0.0);
    r.iter()
        .zip(&c)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
        / peak
}

/// sc-26270 blocker fix: the VAE mid attention runs query-chunked on iOS. Each chunk's rows attend
/// to every key — the same single-head attention, row for row — but a chunk changes the score
/// GEMM's row count, so the result is equal up to float reassociation, not bit-identical (measured
/// on the CPU). It is held to the tiled-decode parity standard (`vae::tests`): 1e-4 of peak for an
/// f32 VAE against the f32 single call, 1e-2 of peak for the fp16 VAE against the f32 single call
/// on the same f16-rounded weights — at chunk sizes that divide the tokens, leave a ragged tail,
/// and degenerate to one row. The chunked plan is selected only under the iOS path: off device the
/// plan is unbounded, so the host decode is the historical single SDPA call, byte for byte.
#[test]
fn vae_attention_query_chunks_meet_the_tiled_parity_standard_and_are_ios_only() {
    if !cpu_child("vae_attention_query_chunks_meet_the_tiled_parity_standard_and_are_ios_only") {
        return;
    }
    let mut g = Gen::new(7700);
    g.ln("a.group_norm", 64);
    for p in ["to_q", "to_k", "to_v", "to_out.0"] {
        g.linear(&format!("a.{p}"), 64, 64, true);
    }
    // f16-rounded weights, so the f32 reference and the f16 attention share their storage rounding.
    let mut rounded = g.weights();
    rounded.cast_all(Dtype::Float16).unwrap();
    let mut half = rounded.clone();
    rounded.cast_all(Dtype::Float32).unwrap();
    let attn = crate::vae::VaeAttention::from_weights(&rounded, "a").unwrap();
    let attn16 = crate::vae::VaeAttention::from_weights(&half, "a").unwrap();
    half.cast_all(Dtype::Float16).unwrap();
    let x = rnd(&[1, 24, 40, 64], 7, 2.0)
        .as_dtype(Dtype::Float16)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    x.eval().unwrap();
    let x16 = x.as_dtype(Dtype::Float16).unwrap();
    let tokens = 24 * 40;
    let unbounded = mlx_gen::attention::AttentionPlan::UNBOUNDED;
    let full = attn.forward_planned(&x, unbounded).unwrap();
    let full16 = attn16.forward_planned(&x16, unbounded).unwrap();
    for rows in [1, 37, 128, 480, tokens - 1] {
        let budget = mlx_gen::attention::AttentionBudget::from_score_elements(
            rows as u64 * tokens as u64,
            true,
        );
        assert_eq!(budget.query_block(1, 1, tokens, tokens), rows);
        let plan = mlx_gen::attention::AttentionPlan::budgeted(budget);
        let chunked = attn.forward_planned(&x, plan).unwrap();
        let delta = relative_max_delta(&full, &chunked);
        eprintln!(
            "f32 {rows}-row chunks: {delta:.3e} of peak, bit-exact {}",
            bits(&full) == bits(&chunked)
        );
        assert!(
            delta <= 1e-4,
            "f32 {rows}-row chunks drifted {delta:.3e} of peak"
        );
        let chunked16 = attn16.forward_planned(&x16, plan).unwrap();
        assert_eq!(
            chunked16.dtype(),
            Dtype::Float16,
            "the chunked f16 path stays at f16"
        );
        let delta = relative_max_delta(&full, &chunked16);
        eprintln!(
            "f16 {rows}-row chunks: {delta:.3e} of peak vs f32 (single f16 call {:.3e})",
            relative_max_delta(&full, &full16)
        );
        assert!(
            delta <= 1e-2,
            "f16 {rows}-row chunks drifted {delta:.3e} of peak"
        );
    }
    // Plan selection: unbounded on the host, the bounded score budget under the iOS path.
    assert!(crate::vae::VaeAttention::plan().budget.is_unbounded());
    let ios =
        with_cpu_diagnostic_headroom(PLENTY, || Ok(crate::vae::VaeAttention::plan())).unwrap();
    assert_eq!(
        ios.budget.max_score_elements(),
        VAE_ATTENTION_SCORE_ELEMENTS
    );
    assert!(ios.budget.eval_per_chunk());
    // A 1024² decode's 128x128 latent attention chunks to 1024 rows (16 chunks), so its priced
    // f32 scores are 64 MiB per chunk, not the 1 GiB of the full 16384² score tensor.
    assert_eq!(ios.budget.query_block(1, 1, 16384, 16384), 1024);
    assert!(
        vae_attention_bytes(1, 16384, 512, 1024, 4)
            < vae_attention_bytes(1, 16384, 512, 16384, 4) / 4
    );
}

// ---------------------------------------------------------------------------------------------
// Measured-peak proofs (sc-26270 round 3): every counted SDXL quantum price, evaluated with the
// CPU backend's kernel transients, is at least MLX's measured CPU peak for that quantum's body and
// within `MAX_RATIO` of it. Each body is the guarded quantum's own code (or the one-to-three-line
// op sequence it runs, spelled with the same `mlx_gen::nn` calls), at a small shape where the
// activations dominate the weights; weights and inputs are materialized before the measurement.
// ---------------------------------------------------------------------------------------------

mod proofs {
    use super::*;
    use mlx_gen::ios_quantum::{assert_tracks_measured, measured_peak_bytes};

    /// The stated bound: a counted price may exceed the measurement by at most 1.5x.
    const MAX_RATIO: f64 = 1.5;

    /// Measure `run` (after one warm-up run that materializes lazy weights and kernel state) and
    /// prove `formula` tracks it. Prints the ratio for the record.
    fn prove(label: &str, formula: u64, run: impl Fn() -> Result<Vec<Array>>) -> f64 {
        let warm = run().unwrap();
        mlx_rs::transforms::eval(warm.iter()).unwrap();
        drop(warm);
        // Measured on the iOS path: the CPU fixture enables every guard and barrier the device runs.
        let measured = with_cpu_diagnostic_headroom(PLENTY, || measured_peak_bytes(&run)).unwrap();
        let ratio = formula as f64 / measured as f64;
        eprintln!("PROOF {label}: formula {formula} B, measured {measured} B, ratio {ratio:.3}");
        assert_tracks_measured(label, formula, measured, MAX_RATIO);
        ratio
    }

    fn evaluated(a: Array) -> Array {
        a.eval().unwrap();
        a
    }

    fn rand(shape: &[i32], seed: u64) -> Array {
        evaluated(rnd(shape, seed, 1.0))
    }

    impl Gen {
        /// An NCHW conv `[out, in, k, k]` + bias, as checkpoints store it.
        fn conv(&mut self, p: &str, out: i32, inp: i32, k: i32) {
            self.put(
                format!("{p}.weight"),
                &[out, inp, k, k],
                (1.0 / (inp * k * k) as f32).sqrt(),
            );
            self.put(format!("{p}.bias"), &[out], 0.02);
        }
        /// A diffusers `ResnetBlock2D` (U-Net with a `temb` projection, VAE without).
        fn resnet(&mut self, p: &str, cin: i32, cout: i32, temb: Option<i32>) {
            self.ln(&format!("{p}.norm1"), cin);
            self.conv(&format!("{p}.conv1"), cout, cin, 3);
            if let Some(t) = temb {
                self.linear(&format!("{p}.time_emb_proj"), cout, t, true);
            }
            self.ln(&format!("{p}.norm2"), cout);
            self.conv(&format!("{p}.conv2"), cout, cout, 3);
            if cin != cout {
                self.conv(&format!("{p}.conv_shortcut"), cout, cin, 1);
            }
        }
        /// A U-Net `BasicTransformerBlock` (self-attn, cross-attn, GEGLU FFN).
        fn block(&mut self, p: &str, dim: i32, ctx: i32, ff: i32) {
            for n in ["norm1", "norm2", "norm3"] {
                self.ln(&format!("{p}.{n}"), dim);
            }
            for (a, kv) in [("attn1", dim), ("attn2", ctx)] {
                self.linear(&format!("{p}.{a}.to_q"), dim, dim, false);
                self.linear(&format!("{p}.{a}.to_k"), dim, kv, false);
                self.linear(&format!("{p}.{a}.to_v"), dim, kv, false);
                self.linear(&format!("{p}.{a}.to_out.0"), dim, dim, true);
            }
            self.linear(&format!("{p}.ff.net.0.proj"), 2 * ff, dim, true);
            self.linear(&format!("{p}.ff.net.2"), dim, ff, true);
        }
    }

    /// A conv weight in the production layout: the NHWC transposed VIEW of an NCHW checkpoint
    /// tensor (`unet::nchw_to_nhwc`), so the kernels copy it exactly as they do in production.
    fn conv_weight(out: i32, k: i32, inp: i32, seed: u64) -> Array {
        let nchw = evaluated(rnd(
            &[out, inp, k, k],
            seed,
            (1.0 / (inp * k * k) as f32).sqrt(),
        ));
        crate::unet::nchw_to_nhwc(&nchw).unwrap()
    }

    // -- U-Net ---------------------------------------------------------------------------------

    #[test]
    fn unet_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::unet_quanta_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let plan = crate::plan::SdxlForwardPlan::UNBOUNDED;
        let mut g = Gen::new(7000);
        g.resnet("plain", 64, 64, Some(128));
        g.resnet("up", 128, 64, Some(128));
        let w = g.weights();
        let temb = rand(&[1, 128], 1);
        let x = rand(&[1, 32, 32, 64], 2);
        let skip = rand(&[1, 32, 32, 64], 3);

        // Resnet: the down/mid shape and the up path's skip concatenation.
        let plain = crate::unet::ResnetBlock2D::from_weights(&w, "plain").unwrap();
        prove(UNET_RESNET, plain.quantum_scratch(&x), || {
            Ok(vec![plain.forward_quantum(&x, None, Some(&temb), plan)?])
        });
        let up = crate::unet::ResnetBlock2D::from_weights(&w, "up").unwrap();
        prove(
            &format!("{UNET_RESNET} (skip concat)"),
            up.quantum_scratch(&x),
            || {
                Ok(vec![up.forward_quantum(
                    &x,
                    Some(skip.clone()),
                    Some(&temb),
                    plan,
                )?])
            },
        );

        // Downsample (stride-2 conv) and upsample (nearest-2x + conv), as `UNetBlock2D` runs them.
        let wd = conv_weight(64, 3, 64, 4);
        let bd = rand(&[64], 5);
        let call = ConvCall::new(x.shape(), 4, &wd, 2, 1);
        prove(UNET_DOWNSAMPLE, single_conv_bytes(&call, 0, 0, cpu), || {
            Ok(vec![mlx_gen::nn::conv2d(&x, &wd, Some(&bd), 2, 1)?])
        });
        let small = rand(&[1, 16, 16, 64], 6);
        let call = ConvCall::new(&[1, 32, 32, 64], 4, &wd, 1, 1);
        prove(UNET_UPSAMPLE, single_conv_bytes(&call, 1, 0, cpu), || {
            Ok(vec![mlx_gen::nn::conv2d(
                &mlx_gen::nn::upsample_nearest(&small, 2)?,
                &wd,
                Some(&bd),
                1,
                1,
            )?])
        });

        // conv_out head: GroupNorm + SiLU + conv to 4 channels.
        let (gw, gb) = (
            evaluated(Array::ones::<f32>(&[64]).unwrap()),
            rand(&[64], 7),
        );
        let wo = conv_weight(4, 3, 64, 8);
        let bo = rand(&[4], 9);
        let call = ConvCall::new(x.shape(), 4, &wo, 1, 1);
        prove(UNET_CONV_OUT, norm_conv_bytes(&call, cpu), || {
            let y = mlx_gen::nn::group_norm(&x, &gw, &gb, 32, 1e-5)?;
            let y = crate::silu_glue(&y)?;
            Ok(vec![mlx_gen::nn::conv2d(&y, &wo, Some(&bo), 1, 1)?])
        });

        // Stem: conv_in + the time/text_time embeddings at the guided pair's batch, over the
        // curated route's lazily scaled and CFG-concatenated latent.
        let unet = crate::test_support::tiny_unet(7).unwrap();
        let (latent, ctx, pooled, ids) = unet_inputs(2, 32, 32);
        let one = rand(&[1, 32, 32, 4], 10);
        let lazy_x = || -> Result<Array> {
            let scaled = mlx_rs::ops::multiply(&one, Array::from_f32(0.5))?;
            Ok(mlx_rs::ops::concatenate_axis(&[&scaled, &scaled], 0)?)
        };
        prove(UNET_STEM, unet.stem_scratch_for_test(&latent, &ctx), || {
            let (x, temb, _) = unet.stem(
                &lazy_x()?,
                500.0,
                &ctx,
                &pooled,
                &ids,
                plan,
                2,
                Dtype::Float32,
            )?;
            Ok(vec![x, temb])
        });
    }

    #[test]
    fn transformer_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::transformer_quanta_track_their_measured_peaks") {
            return;
        }
        let (dim, ctx_dim, ff, heads) = (128, 64, 512, 2);
        let mut g = Gen::new(7100);
        g.block("b", dim, ctx_dim, ff);
        g.ln("t.norm", dim);
        g.linear("t.proj_in", dim, dim, true);
        g.linear("t.proj_out", dim, dim, true);
        g.block("t.transformer_blocks.0", dim, ctx_dim, ff);
        let w = g.weights();
        let block =
            crate::unet::transformer::TransformerBlock::from_weights(&w, "b", dim, heads).unwrap();
        let y = rand(&[1, 256, dim], 1);
        let memory = rand(&[1, 7, ctx_dim], 2);
        let pair_memory = rand(&[2, 7, ctx_dim], 3);
        let unbounded = crate::plan::SdxlForwardPlan::UNBOUNDED;

        // Batched / plain forward: one latent row, the text memory at that row.
        prove(
            TRANSFORMER_BLOCK,
            block.quantum_scratch(1, 1, 256, 7, false, 4, unbounded.attention),
            || Ok(vec![block.forward_ip(&y, &memory, None, unbounded)?]),
        );
        // Sequential CFG half: the latent at ONE row, the text k/v projected at the pair's two.
        let half = unbounded.with_cfg_row(Some(1));
        prove(
            &format!("{TRANSFORMER_BLOCK} (sequential CFG half)"),
            block.quantum_scratch(1, 2, 256, 7, false, 4, half.attention),
            || Ok(vec![block.forward_ip(&y, &pair_memory, None, half)?]),
        );
        // Bounded attention: 64 query rows per self-attention chunk, each chunk evaluated.
        let budget = mlx_gen::attention::AttentionBudget::from_score_elements(2 * 64 * 256, true);
        let bounded = crate::plan::SdxlForwardPlan::with_attention(
            mlx_gen::attention::AttentionPlan::budgeted(budget),
        );
        assert_eq!(budget.query_block(1, heads, 256, 256), 64);
        prove(
            &format!("{TRANSFORMER_BLOCK} (bounded attention)"),
            block.quantum_scratch(1, 1, 256, 7, false, 4, bounded.attention),
            || Ok(vec![block.forward_ip(&y, &memory, None, bounded)?]),
        );

        // proj_in / proj_out of a Transformer2D over a 16x16 grid.
        let t =
            crate::unet::transformer::Transformer2D::from_weights(&w, "t", dim, heads, 1).unwrap();
        let x = rand(&[1, 16, 16, dim], 4);
        let (proj_in, proj_out) = t.projection_scratch(&x);
        prove(TRANSFORMER_PROJ_IN, proj_in, || {
            Ok(vec![t.proj_in_body(&x)?])
        });
        let hidden = rand(&[1, 256, dim], 5);
        prove(TRANSFORMER_PROJ_OUT, proj_out, || {
            Ok(vec![t.proj_out_body(&hidden, &x)?])
        });
    }

    #[test]
    fn controlnet_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::controlnet_quanta_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        // The conditioning embedding's first (full-resolution, 3 -> 16) and a stride-2 (16 -> 32)
        // convolution, each with its SiLU epilogue.
        let image = rand(&[1, 64, 64, 3], 1);
        let w_in = conv_weight(16, 3, 3, 2);
        let b_in = rand(&[16], 3);
        prove(
            CONTROL_COND_CONV,
            single_conv_bytes(&ConvCall::new(image.shape(), 4, &w_in, 1, 1), 0, 1, cpu),
            || {
                Ok(vec![mlx_gen::nn::silu(&mlx_gen::nn::conv2d(
                    &image,
                    &w_in,
                    Some(&b_in),
                    1,
                    1,
                )?)?])
            },
        );
        let e = rand(&[1, 64, 64, 16], 4);
        let w_dn = conv_weight(32, 3, 16, 5);
        let b_dn = rand(&[32], 6);
        prove(
            &format!("{CONTROL_COND_CONV} (stride 2)"),
            single_conv_bytes(&ConvCall::new(e.shape(), 4, &w_dn, 2, 1), 0, 1, cpu),
            || {
                Ok(vec![mlx_gen::nn::silu(&mlx_gen::nn::conv2d(
                    &e,
                    &w_dn,
                    Some(&b_dn),
                    2,
                    1,
                )?)?])
            },
        );
        // A 1x1 zero-conv head, scaled.
        let r = rand(&[1, 32, 32, 64], 7);
        let wz = conv_weight(64, 1, 64, 8);
        let bz = rand(&[64], 9);
        let scale = Array::from_f32(0.8);
        prove(
            CONTROL_ZERO_CONV,
            single_conv_bytes(&ConvCall::new(r.shape(), 4, &wz, 1, 0), 0, 0, cpu),
            || {
                Ok(vec![mlx_rs::ops::multiply(
                    &mlx_gen::nn::conv2d(&r, &wz, Some(&bz), 1, 0)?,
                    &scale,
                )?])
            },
        );
    }

    // -- Text / vision encoders ------------------------------------------------------------------

    #[test]
    fn encoder_layers_track_their_measured_peaks() {
        if !cpu_child("proofs::encoder_layers_track_their_measured_peaks") {
            return;
        }
        // CLIP text layer: 77 tokens, 128 wide, 2 heads, the causal additive mask.
        let cfg = ClipTextConfig {
            num_layers: 1,
            model_dims: 128,
            num_heads: 2,
            max_length: 77,
            vocab_size: 40,
            projection_dim: None,
            hidden_act: ClipActivation::Gelu,
        };
        let mut g = Gen::new(7200);
        g.put(
            "text_model.embeddings.token_embedding.weight".into(),
            &[40, 128],
            0.5,
        );
        g.put(
            "text_model.embeddings.position_embedding.weight".into(),
            &[77, 128],
            0.1,
        );
        g.clip_layers("text_model", 1, 128, 3 * 128);
        g.ln("text_model.final_layer_norm", 128);
        let te =
            crate::text_encoder::ClipTextEncoder::from_weights(&g.weights(), "text_model", &cfg)
                .unwrap();
        let x = rand(&[2, 77, 128], 1);
        let (scratch, body) = te.layer_quantum_for_test(0, &x);
        prove(CLIP_LAYER, scratch, || Ok(vec![body()?]));

        // CLIP vision layer: 257 tokens (16x16 patches + class), 128 wide, 2 heads, 2x MLP.
        let vcfg = crate::vision_encoder::VisionConfig {
            hidden: 128,
            num_layers: 1,
            num_heads: 2,
            patch: 4,
            image_size: 64,
            num_channels: 3,
            quick_gelu: false,
        };
        let mut g = Gen::new(7300);
        g.put(
            "vision_model.embeddings.patch_embedding.weight".into(),
            &[128, 3, 4, 4],
            0.2,
        );
        g.put(
            "vision_model.embeddings.class_embedding".into(),
            &[128],
            0.1,
        );
        g.put(
            "vision_model.embeddings.position_embedding.weight".into(),
            &[257, 128],
            0.1,
        );
        g.ln("vision_model.pre_layrnorm", 128);
        g.clip_layers("vision_model", 1, 128, 256);
        let vision =
            crate::vision_encoder::ClipVisionEncoder::from_weights(&g.weights(), &vcfg).unwrap();
        let v = rand(&[1, 257, 128], 2);
        let (scratch, body) = vision.layer_quantum_for_test(0, &v);
        prove(VISION_LAYER, scratch, || Ok(vec![body()?]));
        let pixels = rand(&[1, 64, 64, 3], 5);
        let (scratch, body) = vision.embed_quantum_for_test(&pixels);
        prove(VISION_EMBEDDINGS, scratch, || Ok(vec![body()?]));

        // IP Resampler perceiver layer: 257 image tokens + 16 queries, 2 heads of 64, 4x FFN.
        let rcfg = crate::ip_adapter::ResamplerConfig {
            dim: 128,
            depth: 1,
            heads: 2,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 128,
            output_dim: 64,
        };
        let mut g = Gen::new(7400);
        g.put("image_proj.latents".into(), &[1, 16, 128], 0.5);
        g.linear("image_proj.proj_in", 128, 128, true);
        g.linear("image_proj.proj_out", 64, 128, true);
        g.ln("image_proj.norm_out", 64);
        g.ln("image_proj.layers.0.0.norm1", 128);
        g.ln("image_proj.layers.0.0.norm2", 128);
        g.linear("image_proj.layers.0.0.to_q", 128, 128, false);
        g.linear("image_proj.layers.0.0.to_kv", 256, 128, false);
        g.linear("image_proj.layers.0.0.to_out", 128, 128, false);
        g.ln("image_proj.layers.0.1.0", 128);
        g.linear("image_proj.layers.0.1.1", 512, 128, false);
        g.linear("image_proj.layers.0.1.3", 128, 512, false);
        let resampler =
            crate::ip_adapter::Resampler::from_weights(&g.weights(), "image_proj", &rcfg).unwrap();
        let features = rand(&[1, 257, 128], 3);
        let latents = rand(&[1, 16, 128], 4);
        let (scratch, body) = resampler.layer_quantum_for_test(0, &features, &latents);
        prove(RESAMPLER_LAYER, scratch, || Ok(vec![body()?]));
    }

    // -- VAE -------------------------------------------------------------------------------------

    #[test]
    fn vae_quanta_track_their_measured_peaks() {
        if !cpu_child("proofs::vae_quanta_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let mut g = Gen::new(7500);
        g.resnet("r", 64, 64, None);
        g.ln("a.group_norm", 64);
        for p in ["to_q", "to_k", "to_v", "to_out.0"] {
            g.linear(&format!("a.{p}"), 64, 64, true);
        }
        let w = g.weights();

        // A temb-free resnet (`resnet_quantum`'s lazy body).
        let resnet = crate::unet::ResnetBlock2D::from_weights(&w, "r").unwrap();
        let x = rand(&[1, 32, 32, 64], 1);
        prove(VAE_RESNET, resnet.quantum_scratch(&x), || {
            Ok(vec![resnet.forward(&x, None)?])
        });

        // Mid attention, unchunked (the host path) and query-chunked (the iOS plan) — priced at
        // the chunk's rows.
        let attn = crate::vae::VaeAttention::from_weights(&w, "a").unwrap();
        let small = rand(&[1, 16, 16, 64], 2);
        prove(
            VAE_ATTENTION,
            vae_attention_bytes(1, 256, 64, 256, 4),
            || Ok(vec![attn.forward(&small)?]),
        );
        let budget = mlx_gen::attention::AttentionBudget::from_score_elements(128 * 1024, true);
        assert_eq!(budget.query_block(1, 1, 1024, 1024), 128);
        let chunked = mlx_gen::attention::AttentionPlan::budgeted(budget);
        prove(
            &format!("{VAE_ATTENTION} (query-chunked)"),
            vae_attention_bytes(1, 1024, 64, 128, 4),
            || Ok(vec![attn.forward_planned(&x, chunked)?]),
        );

        // Downsample: asymmetric pad + stride-2 conv.
        let wd = conv_weight(64, 3, 64, 3);
        let bd = rand(&[64], 4);
        let call = ConvCall::new(&[1, 33, 33, 64], 4, &wd, 2, 0);
        prove(VAE_DOWNSAMPLE, single_conv_bytes(&call, 1, 0, cpu), || {
            let padded = mlx_rs::ops::pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            Ok(vec![mlx_gen::nn::conv2d(&padded, &wd, Some(&bd), 2, 0)?])
        });

        // Decode conv_in over the lazily de-normalized latent; encode conv_in over the image.
        let z = rand(&[1, 32, 32, 4], 5);
        let wi = conv_weight(64, 3, 4, 6);
        let bi = rand(&[64], 7);
        let post = rand(&[4, 4], 8);
        let call = ConvCall::new(z.shape(), 4, &wi, 1, 1);
        prove(VAE_DECODE_IN, single_conv_bytes(&call, 2, 0, cpu), || {
            let scaled = mlx_rs::ops::multiply(&z, Array::from_f32(1.0 / 0.13025))?;
            let projected = mlx_rs::ops::matmul(&scaled, post.t())?;
            Ok(vec![mlx_gen::nn::conv2d(&projected, &wi, Some(&bi), 1, 1)?])
        });
        let image = rand(&[1, 64, 64, 3], 9);
        let we = conv_weight(64, 3, 3, 10);
        let call = ConvCall::new(image.shape(), 4, &we, 1, 1);
        prove(VAE_ENCODE_IN, single_conv_bytes(&call, 1, 0, cpu), || {
            Ok(vec![mlx_gen::nn::conv2d(&image, &we, Some(&bi), 1, 1)?])
        });

        // Decode conv_out: GroupNorm + SiLU + conv to 3 channels.
        let (gw, gb) = (
            evaluated(Array::ones::<f32>(&[64]).unwrap()),
            rand(&[64], 11),
        );
        let wo = conv_weight(3, 3, 64, 12);
        let bo = rand(&[3], 13);
        let call = ConvCall::new(x.shape(), 4, &wo, 1, 1);
        prove(VAE_DECODE_OUT, norm_conv_bytes(&call, cpu), || {
            let y = mlx_gen::nn::group_norm(&x, &gw, &gb, 32, 1e-5)?;
            Ok(vec![mlx_gen::nn::conv2d(
                &mlx_gen::nn::silu(&y)?,
                &wo,
                Some(&bo),
                1,
                1,
            )?])
        });

        // Encode out (the fixture VAE's real head) over its encoder body.
        let vae = tiny_vae();
        let (body, scratch) = vae
            .encode_body_for_test(&rand(&[1, 64, 64, 3], 14))
            .unwrap();
        let body = evaluated(body);
        prove(VAE_ENCODE_OUT, scratch, || Ok(vec![vae.encode_out(&body)?]));

        // The decoded image's pixel conversion.
        let decoded = rand(&[1, 64, 64, 3], 15);
        prove(DECODED_TO_IMAGE, decoded_to_image_bytes(&decoded), || {
            crate::pipeline::decoded_to_image(&decoded)?;
            Ok(vec![])
        });
    }

    #[test]
    fn tiled_vae_stages_and_tiles_track_their_measured_peaks() {
        if !cpu_child("proofs::tiled_vae_stages_and_tiles_track_their_measured_peaks") {
            return;
        }
        let cpu = Backend::Cpu;
        let mut g = Gen::new(7600);
        g.resnet("r", 64, 64, None);
        let resnet = crate::unet::ResnetBlock2D::from_weights(&g.weights(), "r").unwrap();
        let wu = conv_weight(64, 3, 64, 1);
        let bu = rand(&[64], 2);
        // A tiled stage plus its widest tile (the 16-edge tiles of a 32x32 extent).
        let x = rand(&[1, 32, 32, 64], 3);
        let tile = tile_bytes(&[1, 16, 16, 64], 4, &wu, false, cpu);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (resnet)"),
            tiled_stage_bytes(&[1, 32, 32, 64], 64, 64, 4, true) + tile,
            || Ok(vec![resnet.forward_tiled_vae(&x, 16, None)?]),
        );
        // The bounded encode's stride-2 downsample, at a stage-dominated shape (8-pixel output
        // cores, 17x17 exact-halo crops, over a 128x128x32 input: the row strips + assembly carry
        // the peak) and a tile-dominated one (one 32-pixel core: its 65x65 padded crop, output and
        // im2col carry it).
        let wds = conv_weight(32, 3, 32, 5);
        let bds = rand(&[32], 9);
        let wide = rand(&[1, 128, 128, 32], 10);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (encode downsample, many tiles)"),
            downsample_stage_bytes(&[1, 64, 64, 32], 4)
                + downsample_tile_bytes(&[1, 17, 17, 32], 4, &wds, cpu),
            || {
                Ok(vec![crate::vae::downsample_tiled(
                    &wide, &wds, &bds, 8, None,
                )?])
            },
        );
        let wd = conv_weight(64, 3, 64, 5);
        let sq = rand(&[1, 64, 64, 64], 11);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (encode downsample, one tile)"),
            downsample_stage_bytes(&[1, 32, 32, 64], 4)
                + downsample_tile_bytes(&[1, 65, 65, 64], 4, &wd, cpu),
            || Ok(vec![crate::vae::downsample_tiled(&sq, &wd, &bu, 32, None)?]),
        );
        // The bounded encode's conv_in: a 3-channel image to 32 channels at 16-pixel tiles.
        let image = rand(&[1, 64, 64, 3], 6);
        let wi = conv_weight(32, 3, 3, 7);
        let bi = rand(&[32], 8);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (encode conv_in)"),
            tiled_stage_bytes(&[1, 64, 64, 32], 3, 32, 4, false)
                + tile_bytes(&[1, 18, 18, 3], 4, &wi, false, cpu),
            || {
                Ok(vec![mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc(
                    &image,
                    &wi,
                    Some(&bi),
                    16,
                    None,
                    |t| Ok(t.clone()),
                )?])
            },
        );
        let small = rand(&[1, 16, 16, 64], 4);
        prove(
            &format!("{VAE_TILED_STAGE} + {VAE_TILE} (upsample)"),
            tiled_stage_bytes(&[1, 32, 32, 64], 64, 64, 4, false) + tile,
            || {
                Ok(vec![crate::vae::upsample_tiled(
                    &small, &wu, &bu, 16, None,
                )?])
            },
        );
    }

    #[test]
    fn sampler_step_tracks_its_measured_peak() {
        if !cpu_child("proofs::sampler_step_tracks_its_measured_peak") {
            return;
        }
        use mlx_gen::DiffusionSampler;
        let sampler =
            crate::sampler::EulerSampler::new(&crate::config::DiffusionConfig::sdxl_base(), true)
                .unwrap();
        let schedule = crate::sampler::AncestralEuler::new(&sampler, 20, 1000.0).unwrap();
        let latents = rand(&[1, 64, 64, 4], 1);
        let eps_text = rand(&[1, 64, 64, 4], 2);
        let eps_neg = rand(&[1, 64, 64, 4], 3);
        prove(SAMPLER_STEP, sampler_step_bytes(&latents, false), || {
            let eps =
                mlx_gen::gen_core::guidance::cfg(&mlx_gen::MlxLatentOps, &eps_text, &eps_neg, 7.0)?;
            Ok(vec![schedule.step(&eps, &latents, 3)?])
        });
    }
}

// ---------------------------------------------------------------------------------------------
// Reachable-sizes prediction (sc-26270 round 3, item 7). Host-derived PREDICTIONS, not evidence:
// resident bytes per phase from the on-disk safetensors headers under the iOS load-cast rule, the
// largest counted quantum of that phase at production shapes (Metal backend, f16 activations,
// sequential CFG, fused U-Net SDPA, the fp16-fix VAE with the chunked mid attention), and whether
// resident + scratch + the 512 MiB reserve fits the 6142 MiB entitled launch headroom.
// ---------------------------------------------------------------------------------------------

mod reachable {
    use super::*;
    use mlx_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, Dtype as Stored};

    const MIB: u64 = 1 << 20;
    const ENTITLED: u64 = 6142 * MIB;
    const HALF: u64 = 2;

    /// The caller-provisioned snapshot for `repo`, from `SDXL_PREDICTION_SNAPSHOTS` (a directory
    /// holding one sub-directory per repo name). Inference never derives a cache path.
    fn snapshot(repo: &str) -> Option<std::path::PathBuf> {
        let root = std::path::PathBuf::from(std::env::var_os("SDXL_PREDICTION_SNAPSHOTS")?);
        Some(root.join(repo)).filter(|p| p.is_dir())
    }

    /// Resident bytes of `file` after the iOS load: every float tensor not at `target` is cast to
    /// it (`width` bytes) — on a packed file only the dense leaves (never codes or `.scales` /
    /// `.biases`) — else its stored bytes.
    fn resident(file: &std::path::Path, target: Stored, width: u64) -> u64 {
        let headers = safetensors_path_tensor_headers(file).unwrap();
        let packed = headers.iter().any(|h| h.name.ends_with(".scales"));
        headers
            .iter()
            .map(|h| {
                let floating = matches!(
                    h.dtype,
                    Stored::F16 | Stored::BF16 | Stored::F32 | Stored::F64
                );
                let companion = h.name.ends_with(".scales") || h.name.ends_with(".biases");
                let casts = h.dtype != target && (!packed || (floating && !companion));
                if casts {
                    h.shape.iter().product::<usize>() as u64 * width
                } else {
                    h.data_bytes
                }
            })
            .sum()
    }

    /// A dense f16 ControlNet's resident bytes after `quantize(bits)` (its attention/FFN/embedding
    /// linears pack at `bits` + one bf16 scale and bias per 64-group; convs and norms stay f16).
    fn quantized_controlnet(file: &std::path::Path, bits: Option<u64>) -> u64 {
        safetensors_path_tensor_headers(file)
            .unwrap()
            .iter()
            .map(|h| {
                let elements = h.shape.iter().product::<usize>() as u64;
                let linear = h.shape.len() == 2
                    && h.name.ends_with(".weight")
                    && !h.name.contains("conv")
                    && !h.name.contains("norm");
                match bits {
                    Some(b) if linear => elements * b / 8 + elements / 64 * 2 * 2,
                    _ => elements * HALF,
                }
            })
            .sum()
    }

    /// The largest tensor of `file` as stored (a load's read transient; `before_cast_for_load`'s
    /// scratch beyond the resident bytes for an uncast file).
    fn largest(file: &std::path::Path) -> u64 {
        safetensors_path_tensor_headers(file)
            .unwrap()
            .iter()
            .map(|h| h.data_bytes)
            .max()
            .unwrap_or(0)
    }

    fn w(o: i32, k: i32, c: i32) -> Array {
        Array::zeros::<f32>(&[o, k, k, c])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap()
    }

    fn worst(quanta: &[(String, u64)]) -> (String, u64) {
        quanta.iter().max_by_key(|q| q.1).cloned().unwrap()
    }

    /// Every counted quantum of one SDXL U-Net forward at latent edge `l` (sequential CFG: one
    /// latent row, the pair's text memory), `ip` IP tokens.
    fn unet_quanta(l: i32, ip: i32) -> Vec<(String, u64)> {
        let m = Backend::Metal;
        let mut q = Vec::new();
        let x = Array::zeros::<f32>(&[1, l, l, 4])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        let ctx = Array::zeros::<f32>(&[2, 77, 2048])
            .unwrap()
            .as_dtype(Dtype::Float16)
            .unwrap();
        q.push((
            UNET_STEM.into(),
            crate::unet::stem_bytes(&x, &ctx, &w(320, 3, 4), None, 0, m),
        ));
        let (l1, l2) = (l / 2, l / 4);
        // (grid, pre-concat channels, conv1 in, out)
        let resnets = [
            (l, 320, 320, 320),
            (l1, 320, 320, 640),
            (l1, 640, 640, 640),
            (l2, 640, 640, 1280),
            (l2, 1280, 1280, 1280),
            (l2, 1280, 2560, 1280),
            (l2, 1280, 1920, 1280),
            (l1, 1280, 1920, 640),
            (l1, 640, 1280, 640),
            (l1, 640, 960, 640),
            (l, 640, 960, 320),
            (l, 320, 640, 320),
        ];
        for (h, cx, cin, cout) in resnets {
            q.push((
                format!("{UNET_RESNET} {cin}->{cout} @{h}"),
                resnet_bytes(&[1, h, h, cx], HALF, &w(cout, 3, cin), &w(cout, 3, cout), m),
            ));
        }
        for (h, c) in [(l, 320), (l1, 640)] {
            let call = ConvCall::new(&[1, h, h, c], HALF, &w(c, 3, c), 2, 1);
            q.push((
                format!("{UNET_DOWNSAMPLE} @{h}"),
                single_conv_bytes(&call, 0, 0, m),
            ));
        }
        for (h, c) in [(l2, 1280), (l1, 640)] {
            let call = ConvCall::new(&[1, 2 * h, 2 * h, c], HALF, &w(c, 3, c), 1, 1);
            q.push((
                format!("{UNET_UPSAMPLE} @{h}"),
                single_conv_bytes(&call, 1, 0, m),
            ));
        }
        for (h, dim, heads) in [(l1, 640u64, 10), (l2, 1280, 20)] {
            let tokens = h * h;
            q.push((
                format!("{TRANSFORMER_BLOCK} {dim} @{h}"),
                transformer_block_bytes(
                    1,
                    2,
                    tokens,
                    dim,
                    heads,
                    64,
                    77 + ip,
                    4 * dim,
                    tokens,
                    ip > 0,
                    HALF,
                    m,
                ),
            ));
            let plane = |c: u64| (tokens as u64) * c * HALF;
            q.push((
                format!("{TRANSFORMER_PROJ_IN} @{h}"),
                plane(dim) * (GROUP_NORM_BUFFERS + 1) + page_slack(GROUP_NORM_BUFFERS + 1),
            ));
        }
        let call = ConvCall::new(&[1, l, l, 320], HALF, &w(4, 3, 320), 1, 1);
        q.push((UNET_CONV_OUT.into(), norm_conv_bytes(&call, m)));
        q
    }

    /// The ControlNet branch's own quanta at latent edge `l`: its conditioning embedding at full
    /// resolution (once per request) and the zero-convs; its encoder copy is the U-Net's down/mid.
    fn controlnet_quanta(l: i32) -> Vec<(String, u64)> {
        let m = Backend::Metal;
        let mut q = Vec::new();
        let e = 8 * l;
        let convs = [
            (e, 3, 16, 1),
            (e, 16, 16, 1),
            (e, 16, 32, 2),
            (e / 2, 32, 32, 1),
            (e / 2, 32, 96, 2),
            (e / 4, 96, 96, 1),
            (e / 4, 96, 256, 2),
            (e / 8, 256, 320, 1),
        ];
        for (h, c, o, s) in convs {
            let call = ConvCall::new(&[1, h, h, c], HALF, &w(o, 3, c), s, 1);
            q.push((
                format!("{CONTROL_COND_CONV} {c}->{o} @{h}"),
                single_conv_bytes(&call, 0, 1, m),
            ));
        }
        let call = ConvCall::new(&[1, l, l, 320], HALF, &w(320, 1, 320), 1, 0);
        q.push((CONTROL_ZERO_CONV.into(), single_conv_bytes(&call, 0, 0, m)));
        q
    }

    /// The fp16-fix VAE decode at latent edge `l`: the dense head (conv_in, mid resnets, the
    /// chunked mid attention) and either the dense tail or the layer-wise tiled tail at output
    /// tile edge `tile` (each stage priced with its widest tile), and the pixel conversion.
    fn decode_quanta(l: i32, tile: Option<i32>) -> Vec<(String, u64)> {
        let m = Backend::Metal;
        let mut q = Vec::new();
        let call = ConvCall::new(&[1, l, l, 4], HALF, &w(512, 3, 4), 1, 1);
        q.push((VAE_DECODE_IN.into(), single_conv_bytes(&call, 2, 0, m)));
        q.push((
            format!("{VAE_RESNET} mid @{l}"),
            resnet_bytes(&[1, l, l, 512], HALF, &w(512, 3, 512), &w(512, 3, 512), m),
        ));
        let n = l * l;
        let rows = mlx_gen::attention::AttentionBudget::from_score_elements(
            VAE_ATTENTION_SCORE_ELEMENTS,
            true,
        )
        .query_block(1, 1, n, n);
        q.push((
            format!("{VAE_ATTENTION} ({rows}-row chunks)"),
            vae_attention_bytes(1, n, 512, rows, HALF),
        ));
        // (input grid, cin, cout, upsample after)
        let blocks = [
            (l, 512, 512, true),
            (2 * l, 512, 512, true),
            (4 * l, 512, 256, true),
            (8 * l, 256, 128, false),
        ];
        let bias = |c: i32| {
            Array::zeros::<f32>(&[c])
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        };
        for (i, (h, cin, cout, up)) in blocks.into_iter().enumerate() {
            let scale = 8 / (1 << i);
            for j in 0..3 {
                let ci = if j == 0 { cin } else { cout };
                let label = format!("{VAE_RESNET} {ci}->{cout} @{h}");
                q.push((
                    label,
                    match tile {
                        None => resnet_bytes(
                            &[1, h, h, ci],
                            HALF,
                            &w(cout, 3, ci),
                            &w(cout, 3, cout),
                            m,
                        ),
                        Some(t) => {
                            let e = ((t + scale - 1) / scale).max(3);
                            tiled_stage_bytes(&[1, h, h, cout], ci, cout, HALF, true)
                                + tile_bytes(
                                    &[1, e.min(h) + 2, e.min(h) + 2, ci],
                                    HALF,
                                    &w(cout, 3, ci),
                                    false,
                                    m,
                                )
                        }
                    },
                ));
            }
            if up {
                let label = format!("{VAE_UPSAMPLE} @{}", 2 * h);
                q.push((
                    label,
                    match tile {
                        None => single_conv_bytes(
                            &ConvCall::new(&[1, 2 * h, 2 * h, cout], HALF, &w(cout, 3, cout), 1, 1),
                            1,
                            0,
                            m,
                        ),
                        Some(t) => {
                            let e = ((t + scale / 2 - 1) / (scale / 2)).max(3);
                            tiled_stage_f32_accum(
                                &[1, 2 * h, 2 * h, cout],
                                &w(cout, 3, cout),
                                &bias(cout),
                                HALF,
                            ) + tile_bytes(
                                &[1, e.min(2 * h) + 2, e.min(2 * h) + 2, cout],
                                HALF,
                                &w(cout, 3, cout),
                                true,
                                m,
                            )
                        }
                    },
                ));
            }
        }
        let e8 = 8 * l;
        q.push((
            VAE_DECODE_OUT.into(),
            match tile {
                None => norm_conv_bytes(
                    &ConvCall::new(&[1, e8, e8, 128], HALF, &w(3, 3, 128), 1, 1),
                    m,
                ),
                Some(t) => {
                    tiled_stage_bytes(&[1, e8, e8, 3], 128, 3, HALF, false)
                        + tile_bytes(
                            &[1, t.min(e8) + 2, t.min(e8) + 2, 128],
                            HALF,
                            &w(3, 3, 128),
                            false,
                            m,
                        )
                }
            },
        ));
        let decoded = Array::zeros::<f32>(&[1, e8, e8, 3]).unwrap();
        q.push((DECODED_TO_IMAGE.into(), decoded_to_image_bytes(&decoded)));
        q
    }

    /// The fp16-fix VAE encode (img2img) at image edge `8l`: dense, every stage at its extent.
    fn encode_quanta(l: i32) -> Vec<(String, u64)> {
        let m = Backend::Metal;
        let e = 8 * l;
        let mut q = vec![(
            VAE_ENCODE_IN.to_string(),
            single_conv_bytes(
                &ConvCall::new(&[1, e, e, 3], HALF, &w(128, 3, 3), 1, 1),
                1,
                0,
                m,
            ),
        )];
        for (h, cin, cout) in [
            (e, 128, 128),
            (e / 2, 128, 256),
            (e / 4, 256, 512),
            (e / 8, 512, 512),
        ] {
            q.push((
                format!("{VAE_RESNET} {cin}->{cout} @{h}"),
                resnet_bytes(
                    &[1, h, h, cin],
                    HALF,
                    &w(cout, 3, cin),
                    &w(cout, 3, cout),
                    m,
                ),
            ));
            if h > e / 8 {
                q.push((
                    format!("{VAE_DOWNSAMPLE} @{h}"),
                    single_conv_bytes(
                        &ConvCall::new(&[1, h + 1, h + 1, cout], HALF, &w(cout, 3, cout), 2, 0),
                        1,
                        0,
                        m,
                    ),
                ));
            }
        }
        let n = l * l;
        let rows = mlx_gen::attention::AttentionBudget::from_score_elements(
            VAE_ATTENTION_SCORE_ELEMENTS,
            true,
        )
        .query_block(1, 1, n, n);
        q.push((
            VAE_ATTENTION.into(),
            vae_attention_bytes(1, n, 512, rows, HALF),
        ));
        let call = ConvCall::new(&[1, l, l, 512], HALF, &w(8, 3, 512), 1, 1);
        q.push((
            VAE_ENCODE_OUT.into(),
            norm_conv_bytes(&call, m) + 4 * call.out_bytes() * 2,
        ));
        q
    }

    /// The bounded (tiled) fp16-fix VAE encode at image edge `8l` with `tile` input pixels: each
    /// full-resolution layer a tiled stage priced with its widest tile, the latent-resolution mid
    /// block dense with the chunked attention, the output head.
    fn encode_tiled_quanta(l: i32, tile: i32) -> Vec<(String, u64)> {
        let m = Backend::Metal;
        let e = 8 * l;
        let crop = |edge: i32, h: i32| edge.min(h) + 2;
        let mut q = vec![(
            format!("{VAE_TILED_STAGE} conv_in @{e}"),
            tiled_stage_bytes(&[1, e, e, 128], 3, 128, HALF, false)
                + tile_bytes(
                    &[1, crop(tile, e), crop(tile, e), 3],
                    HALF,
                    &w(128, 3, 3),
                    false,
                    m,
                ),
        )];
        let mut scale = 1;
        for (h, cin, cout) in [
            (e, 128, 128),
            (e / 2, 128, 256),
            (e / 4, 256, 512),
            (e / 8, 512, 512),
        ] {
            let edge = ((tile + scale - 1) / scale).max(3);
            for ci in [cin, cout] {
                q.push((
                    format!("{VAE_TILED_STAGE} resnet {ci}->{cout} @{h}"),
                    tiled_stage_bytes(&[1, h, h, cout], ci, cout, HALF, true)
                        + tile_bytes(
                            &[1, crop(edge, h), crop(edge, h), ci],
                            HALF,
                            &w(cout, 3, ci),
                            false,
                            m,
                        ),
                ));
            }
            if h > e / 8 {
                scale *= 2;
                let core = ((tile + scale - 1) / scale).max(1).min(h / 2);
                q.push((
                    format!("{VAE_TILED_STAGE} downsample @{h}"),
                    downsample_stage_bytes(&[1, h / 2, h / 2, cout], HALF)
                        + downsample_tile_bytes(
                            &[1, 2 * core + 1, 2 * core + 1, cout],
                            HALF,
                            &w(cout, 3, cout),
                            m,
                        ),
                ));
            }
        }
        q.push((
            format!("{VAE_RESNET} mid @{l}"),
            resnet_bytes(&[1, l, l, 512], HALF, &w(512, 3, 512), &w(512, 3, 512), m),
        ));
        let n = l * l;
        let rows = mlx_gen::attention::AttentionBudget::from_score_elements(
            VAE_ATTENTION_SCORE_ELEMENTS,
            true,
        )
        .query_block(1, 1, n, n);
        q.push((
            VAE_ATTENTION.into(),
            vae_attention_bytes(1, n, 512, rows, HALF),
        ));
        let call = ConvCall::new(&[1, l, l, 512], HALF, &w(8, 3, 512), 1, 1);
        q.push((
            VAE_ENCODE_OUT.into(),
            norm_conv_bytes(&call, m) + 4 * call.out_bytes() * 2,
        ));
        q
    }

    fn row(
        mode: &str,
        tier: &str,
        edge: i32,
        phase: &str,
        resident: u64,
        (label, scratch): (String, u64),
    ) {
        let total = resident + scratch + PROCESS_RESERVE_BYTES;
        println!(
            "| {mode} | {tier} | {edge} | {phase} | {:.0} | {:.0} ({label}) | {:.0} + 512 = {:.0} | {} |",
            resident as f64 / MIB as f64,
            scratch as f64 / MIB as f64,
            (resident + scratch) as f64 / MIB as f64,
            total as f64 / MIB as f64,
            if total <= ENTITLED { "fits" } else { "REFUSED" }
        );
    }

    #[test]
    #[ignore = "prediction table from on-disk safetensors headers; run with --ignored --nocapture"]
    fn reachable_sizes_prediction() {
        if !cpu_child("reachable::reachable_sizes_prediction") {
            return;
        }
        let Some(base) = snapshot("models--SceneWorks--sdxl-base-mlx") else {
            println!("sdxl-base-mlx snapshot not on disk");
            return;
        };
        let fix = snapshot("models--madebyollin--sdxl-vae-fp16-fix")
            .map(|d| d.join("diffusion_pytorch_model.safetensors"));
        let cn = snapshot("models--xinsir--controlnet-tile-sdxl-1.0")
            .map(|d| d.join("diffusion_pytorch_model.safetensors"));
        let vae = fix.as_deref().map_or(0, |f| resident(f, Stored::F16, HALF));
        println!(
            "fp16-fix VAE resident at f16: {:.0} MiB",
            vae as f64 / MIB as f64
        );
        println!("| mode | tier | edge | phase | resident MiB | largest quantum MiB | resident+scratch + reserve MiB | vs 6142 MiB |");
        println!("|---|---|---|---|---|---|---|---|");
        for (tier, bits) in [("bf16", None), ("q8", Some(8u64)), ("q4", Some(4))] {
            let dir = base.join(tier);
            let file = |sub: &str, stem: &str| {
                let fp16 = dir.join(sub).join(format!("{stem}.fp16.safetensors"));
                if fp16.exists() {
                    fp16
                } else {
                    dir.join(sub).join(format!("{stem}.safetensors"))
                }
            };
            let te = resident(&file("text_encoder", "model"), Stored::F16, HALF)
                + resident(&file("text_encoder_2", "model"), Stored::F16, HALF);
            let unet = resident(&file("unet", "diffusion_pytorch_model"), Stored::F16, HALF);
            let control = cn.as_deref().map_or(0, |f| quantized_controlnet(f, bits));
            // IP-Adapter plus (not on disk; config-derived): ViT-H/14 (632M params) + the Resampler
            // (83M) at f16, and the 70 cross-attention K/V pairs (341M params) at the tier.
            let ip_pairs = 341_000_000u64 * bits.map_or(16, |b| b) / 8
                + bits.map_or(0, |_| 341_000_000 / 64 * 4);
            let ip_resident = (632_000_000 + 83_000_000) * HALF + ip_pairs;
            println!(
                "<!-- {tier}: text encoders {:.0} MiB, U-Net {:.0} MiB, ControlNet-tile {:.0} MiB, IP (config) {:.0} MiB -->",
                te as f64 / MIB as f64, unet as f64 / MIB as f64, control as f64 / MIB as f64, ip_resident as f64 / MIB as f64
            );
            let text = [
                (
                    CLIP_LAYER.to_string() + " L",
                    clip_layer_bytes(2, 77, 768, 3072, 12, 64, true, HALF, Backend::Metal),
                ),
                (
                    CLIP_LAYER.to_string() + " bigG",
                    clip_layer_bytes(2, 77, 1280, 5120, 20, 64, true, HALF, Backend::Metal),
                ),
            ];
            let unet_file = file("unet", "diffusion_pytorch_model");
            // Staged loads: the text encoders load into an empty process; the U-Net (its floor =
            // its resident bytes + its largest tensor) loads after they are shed, the VAE beside it.
            let te_floor = te + largest(&file("text_encoder_2", "model"));
            row(
                "all",
                tier,
                0,
                "text load",
                0,
                ("SDXL text_encoder_2 load floor".into(), te_floor),
            );
            let unet_floor = unet + largest(&unet_file);
            row(
                "all",
                tier,
                0,
                "U-Net load",
                0,
                ("SDXL UNet load floor".into(), unet_floor),
            );
            if let Some(f) = cn.as_deref() {
                // The ControlNet loads dense f16 (its tier quantization runs after the load).
                let floor = resident(f, Stored::F16, HALF) + largest(f);
                row(
                    "controlnet",
                    tier,
                    0,
                    "ControlNet load",
                    unet + vae,
                    ("SDXL ControlNet load floor".into(), floor),
                );
            }
            for edge in [512, 768, 1024] {
                let l = edge / 8;
                row("txt2img", tier, edge, "text", te, worst(&text));
                row(
                    "txt2img",
                    tier,
                    edge,
                    "denoise",
                    unet + vae,
                    worst(&unet_quanta(l, 0)),
                );
                row(
                    "txt2img",
                    tier,
                    edge,
                    "decode (dense)",
                    vae,
                    worst(&decode_quanta(l, None)),
                );
                row(
                    "txt2img",
                    tier,
                    edge,
                    "decode (tiled 256)",
                    vae,
                    worst(&decode_quanta(l, Some(256))),
                );
                // img2img and inpaint share the init encode, which runs in the heavy phase (the
                // U-Net is already resident); the init latent is then denoised in the txt2img
                // denoise phase (inpaint adds its blend to the sampler-step quantum).
                row(
                    "img2img/inpaint",
                    tier,
                    edge,
                    "encode (dense, host path)",
                    unet + vae,
                    worst(&encode_quanta(l)),
                );
                row(
                    "img2img/inpaint",
                    tier,
                    edge,
                    "encode (bounded, iOS 256)",
                    unet + vae,
                    worst(&encode_tiled_quanta(l, crate::vae::IOS_ENCODE_TILE_PX)),
                );
                let mut cq = unet_quanta(l, 0);
                cq.extend(controlnet_quanta(l));
                row(
                    "controlnet",
                    tier,
                    edge,
                    "denoise",
                    unet + vae + control,
                    worst(&cq),
                );
                let vision =
                    clip_layer_bytes(1, 257, 1280, 5120, 16, 80, false, HALF, Backend::Metal);
                row(
                    "ip-adapter",
                    tier,
                    edge,
                    "image encode",
                    unet + vae + ip_resident,
                    (VISION_LAYER.into(), vision),
                );
                row(
                    "ip-adapter",
                    tier,
                    edge,
                    "denoise",
                    unet + vae + ip_resident,
                    worst(&unet_quanta(l, 16)),
                );
            }
        }
    }
}
