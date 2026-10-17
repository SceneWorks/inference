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
        .args(["--exact", &format!("quantum_tests::{name}"), "--nocapture"])
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
    // The stem also materializes the previous step's CFG combine + sampler update (16 latent-sized
    // f32 buffers) and the conditioning concatenation (inputs + result: 2 context-sized buffers).
    let s = x.shape();
    let (cin, cout) = conv_channels(unet.conv_in_weight());
    assert_eq!(
        checked(&events, UNET_STEM),
        [conv_stage(s[0], s[1], s[2], &[cin, cout])
            + 16 * x.size() as u64 * 4
            + 2 * ctx.size() as u64 * 4]
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
    // The IP branch adds three keys and two projections to every block's floor.
    let (with_ip, without) = (
        checked(&events, TRANSFORMER_BLOCK),
        checked(&plain, TRANSFORMER_BLOCK),
    );
    assert_eq!(with_ip.len(), without.len());
    assert!(
        with_ip.iter().zip(&without).all(|(a, b)| a > b),
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
    let (c, ctx_dim, tokens, ctx_tokens) = (64u64, 16u64, 16u64, 5u64);
    let weight_elements = 4 * c * c + c // attn1 q/k/v/out (+out bias)
        + c * c + 2 * c * ctx_dim + c * c + c // attn2 q, k/v over the context, out (+bias)
        + 2 * (4 * c * c + 4 * c) // GEGLU value/gate halves (+biases)
        + c * 4 * c + c // ff.net.2 (+bias)
        + 6 * c; // three LayerNorms
    let expected =
        mlx_gen::ios_quantum::attention_block_bytes(mlx_gen::ios_quantum::AttentionBlockGeometry {
            batch: 1,
            tokens,
            dim: c,
            heads: 2,
            context_tokens: ctx_tokens,
            ff_hidden: 4 * c,
            query_rows: tokens,
            weight_elements,
        });
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
    let up_first = checked(&small, UNET_RESNET)[4];
    assert_eq!(
        up_first,
        mlx_gen::ios_quantum::conv_stage_bytes(1, 4, 4, 128)
    );
    // A bounded attention plan prices only its chunk's query rows: at a 16x16 latent the attention
    // grid is 64 tokens; 160 scores per chunk gives 16 cross-attention (5 keys x 2 heads) rows and
    // one self-attention row, so every block's floor drops.
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
    assert!(
        full.iter().zip(&bounded).all(|(a, b)| b < a),
        "{full:?} vs {bounded:?}"
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
        mlx_gen::ios_quantum::transformer_layer_bytes(2, 8, 24, 72, 3, 8)
    );
    assert_eq!(
        layers[3],
        mlx_gen::ios_quantum::transformer_layer_bytes(2, 8, 32, 96, 4, 8)
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
        mlx_gen::ios_quantum::transformer_layer_bytes(1, 17, 32, 64, 4, 8)
    );
    // Perceiver layers attend over image tokens + queries with the loaded 48-wide FFN.
    assert_eq!(
        checked(&events, RESAMPLER_LAYER)[0],
        mlx_gen::ios_quantum::transformer_layer_bytes(1, 17 + 4, 32, 48, 2, 16)
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
        // The attention floor scores all 72 latent tokens against each other at the loaded width.
        let attention = checked(&events, VAE_ATTENTION)[0];
        assert!(attention >= 8 * 72 * 72 * 4, "{attention}");
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
            vec![16 * 18 * 3 * 4 * 6]
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

/// `before_load`'s header-only floor for `files`: stored bytes plus the largest tensor.
fn header_floor(files: &[&std::path::Path]) -> u64 {
    let headers: Vec<u64> = files
        .iter()
        .flat_map(|f| mlx_gen::gen_core::safetensors_path_tensor_headers(f).unwrap())
        .map(|h| h.data_bytes)
        .collect();
    headers.iter().sum::<u64>() + headers.iter().copied().max().unwrap_or(0)
}

/// `before_load_cast`'s floor: stored bytes, every tensor's `width`-byte cast copy, and the largest
/// cast copy.
fn cast_floor(files: &[&std::path::Path], width: u64) -> u64 {
    let headers: Vec<(u64, u64)> = files
        .iter()
        .flat_map(|f| mlx_gen::gen_core::safetensors_path_tensor_headers(f).unwrap())
        .map(|h| {
            (
                h.data_bytes,
                h.shape.iter().product::<usize>() as u64 * width,
            )
        })
        .collect();
    headers.iter().map(|(s, c)| s + c).sum::<u64>()
        + headers.iter().map(|(_, c)| *c).max().unwrap_or(0)
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
            Box::new(|| crate::loader::load_unet_dtype(root, Dtype::Float32).map(drop)),
        ),
        (
            "SDXL ControlNet load",
            &unet_file,
            Box::new(|| crate::loader::load_controlnet(&unet_source, Dtype::Float32).map(drop)),
        ),
    ];
    for (label, file, load) in &cases {
        // The VAE loaders cast after reading (f32 / the fp16-fix dtype): the stored AND the cast
        // copies are priced, not the stored bytes alone.
        let floor = match *label {
            "SDXL VAE load" => cast_floor(&[file], 4),
            "SDXL fp16-fix VAE load" => cast_floor(&[file], 2),
            _ => header_floor(&[file]),
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
    assert!(cast_floor(&[&vae_file], 4) > header_floor(&[&vae_file]));
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
    // Stored f32 bytes plus the largest counted tensor; the VAE also carries its f32 cast copy.
    assert_eq!(
        floor(&[SdxlComponent::ClipL, SdxlComponent::ClipBigG]),
        32 + 32 + 32
    );
    assert_eq!(
        floor(&[SdxlComponent::Unet, SdxlComponent::Vae]),
        256 + 64 + 64 + 256
    );
    assert_eq!(floor(&[SdxlComponent::Unet]), 256 + 256);
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
    let activations = 4 * (b * 2 * h * 2 * w * cout) as u64 * 4;
    assert_eq!(stage(&x, &weight, &bias), [activations]);
    let half = |a: &Array| a.as_dtype(Dtype::Float16).unwrap();
    let wide_copies = (cout * 9 * cin) as u64 * 4 + cout as u64 * 4;
    assert_eq!(
        stage(&half(&x), &half(&weight), &half(&bias)),
        [activations + wide_copies]
    );
}
