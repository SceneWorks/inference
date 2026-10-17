//! iOS bounded-quantum guards on the Anima inference path, exercised on tiny CPU models through the
//! diagnostic fixture (`mlx_gen::ios_quantum::with_cpu_diagnostic_headroom`). Every test re-executes
//! itself in a child process pinned to the CPU device with Metal made fatal, so nothing here can
//! touch the GPU. The VAE decode quanta live in `mlx-gen-qwen-image` and are tested there.

use std::cell::RefCell;
use std::rc::Rc;

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::ios_quantum::{
    take_refusal, with_cpu_diagnostic_headroom, with_cpu_diagnostic_observer, QuantumEvent,
    PROCESS_RESERVE_BYTES,
};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Result, WeightsSource};
use mlx_rs::{random, Array, Dtype};

use crate::conditioner::{
    AnimaTextConditioner, CONDITIONER_BLOCK_QUANTUM, CONDITIONER_EMBED_QUANTUM,
    CONDITIONER_OUTPUT_QUANTUM,
};
use crate::config::{ConditionerConfig, DitConfig, Qwen3Config};
use crate::text_encoder::{AnimaQwen3, TE_EMBED_QUANTUM, TE_LAYER_QUANTUM, TE_NORM_QUANTUM};
use crate::transformer::{CosmosDiT, DIT_BLOCK_QUANTUM, DIT_EMBED_QUANTUM, DIT_OUTPUT_QUANTUM};

const CHILD: &str = "ANIMA_IOS_QUANTUM_CPU_CHILD";
/// Far above every tiny fixture's scratch + the 512 MiB reserve: every guard passes.
const AMPLE: u64 = 1 << 34;

/// Re-run `name` in a child pinned to the CPU with Metal made fatal. Returns `true` inside the
/// child (the caller runs its body), `false` in the parent after the child succeeded.
fn in_cpu_child(name: &str) -> bool {
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("quantum_tests::{name}"),
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
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

/// Run `body` under the CPU fixture at `headroom`, recording every quantum event.
fn observed<T>(headroom: u64, body: impl FnOnce() -> Result<T>) -> (Result<T>, Vec<QuantumEvent>) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let sink = events.clone();
    let result = with_cpu_diagnostic_headroom(headroom, || {
        with_cpu_diagnostic_observer(move |e| sink.borrow_mut().push(e.clone()), body)
    });
    let events = events.borrow().clone();
    (result, events)
}

/// Every quantum is checked BEFORE construction and synchronized before the next one is checked:
/// the events strictly alternate `Checked(l)`, `Synchronized(l)`.
fn assert_checked_before_synchronized(events: &[QuantumEvent]) {
    assert!(
        events.len().is_multiple_of(2),
        "unpaired quantum events: {events:?}"
    );
    for pair in events.chunks(2) {
        match pair {
            [QuantumEvent::Checked { label: c, .. }, QuantumEvent::Synchronized { label: s, .. }] =>
            {
                assert_eq!(c, s, "{events:?}")
            }
            _ => panic!("a quantum was not checked before it synchronized: {events:?}"),
        }
    }
}

fn checked(events: &[QuantumEvent], label: &str) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, QuantumEvent::Checked { label: l, .. } if l == label))
        .count()
}

fn scratch_of(events: &[QuantumEvent], label: &str) -> Vec<u64> {
    events
        .iter()
        .filter_map(|e| match e {
            QuantumEvent::Checked { label: l, scratch } if l == label => Some(*scratch),
            _ => None,
        })
        .collect()
}

fn synchronized(events: &[QuantumEvent]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, QuantumEvent::Synchronized { .. }))
        .count()
}

fn f32_bytes(a: &Array) -> Vec<f32> {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    a.eval().unwrap();
    a.as_slice::<f32>().to_vec()
}

/// Refuse the run with a headroom that admits every quantum up to (not including) the first one
/// whose scratch exceeds all earlier ones, and prove nothing at or past it ran.
fn assert_refuses_at_first_growth<T>(
    passing: &[QuantumEvent],
    run: impl FnOnce() -> Result<T>,
) -> String {
    let checks: Vec<(String, u64)> = passing
        .iter()
        .filter_map(|e| match e {
            QuantumEvent::Checked { label, scratch } => Some((label.clone(), *scratch)),
            QuantumEvent::Synchronized { .. } => None,
        })
        .collect();
    let index = (1..checks.len())
        .find(|&i| checks[i].1 > checks[..i].iter().map(|c| c.1).max().unwrap())
        .expect("fixture must have a quantum larger than every earlier one");
    let admitted = checks[..index].iter().map(|c| c.1).max().unwrap();
    let (result, events) = observed(admitted + PROCESS_RESERVE_BYTES, run);
    assert!(result.is_err(), "a refused quantum must fail the run");
    let refusal = take_refusal().expect("refusal recorded");
    assert_eq!(refusal.label, checks[index].0);
    assert_eq!(refusal.scratch_bytes, checks[index].1);
    assert_checked_before_synchronized(&events);
    assert_eq!(
        synchronized(&events),
        index,
        "exactly the admitted quanta ran; nothing at or past the refusal: {events:?}"
    );
    checks[index].0.clone()
}

// ---- DiT -----------------------------------------------------------------------------------------

fn dit_cfg(num_layers: usize, mlp_ratio: f32) -> DitConfig {
    DitConfig {
        in_channels: 4,
        out_channels: 4,
        num_attention_heads: 2,
        attention_head_dim: 8,
        num_layers,
        mlp_ratio,
        // ≠ hidden (16): the cross-attention k/v widths come from the loaded tensors.
        text_embed_dim: 12,
        adaln_lora_dim: 8,
        max_size: (4, 16, 16),
        patch_size: (1, 2, 2),
        rope_scale: (1.0, 4.0, 4.0),
        concat_padding_mask: true,
    }
}

/// `[1, C, 1, edge, edge]` latents, a `[1]` sigma and a `[1, ctx, 12]` conditioner output.
fn dit_inputs(cfg: &DitConfig, edge: i32, ctx: i32) -> (Array, Array, Array) {
    let key = random::key(7).unwrap();
    let latents = random::normal::<f32>(
        &[1, cfg.in_channels as i32, 1, edge, edge][..],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let sigma = Array::from_slice(&[0.7f32], &[1]);
    let key = random::key(9).unwrap();
    let encoder = random::normal::<f32>(
        &[1, ctx, cfg.text_embed_dim as i32][..],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    (latents, sigma, encoder)
}

#[test]
fn dit_resident_blocks_are_guarded_quanta() {
    if !in_cpu_child("dit_resident_blocks_are_guarded_quanta") {
        return;
    }
    let cfg = dit_cfg(3, 2.0);
    let dit = CosmosDiT::synthetic(cfg, 11);
    let (latents, sigma, encoder) = dit_inputs(&cfg, 8, 6);
    let run = || dit.forward(&latents, &sigma, &encoder, Dtype::Float32);

    let host = f32_bytes(&run().unwrap());
    let (guarded, events) = observed(AMPLE, run);
    assert_eq!(
        f32_bytes(&guarded.unwrap()),
        host,
        "guards changed numerics"
    );
    assert_checked_before_synchronized(&events);
    assert_eq!(checked(&events, DIT_BLOCK_QUANTUM), cfg.num_layers);
    assert_eq!(checked(&events, DIT_EMBED_QUANTUM), 1);
    assert_eq!(checked(&events, DIT_OUTPUT_QUANTUM), 1);
    let labels: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            QuantumEvent::Checked { label, .. } => Some(label.as_str()),
            QuantumEvent::Synchronized { .. } => None,
        })
        .collect();
    assert_eq!(labels.first(), Some(&DIT_EMBED_QUANTUM));
    assert_eq!(labels.last(), Some(&DIT_OUTPUT_QUANTUM));
    // The entry quantum also materializes the previous step's CFG combine + sampler update and this
    // grid's RoPE tables: the largest block floor + 16 latent-sized f32 buffers + 4 tables.
    let sh = latents.shape();
    let (pt, ph, pw) = cfg.patch_size;
    let tokens = (sh[2] / pt as i32) * (sh[3] / ph as i32) * (sh[4] / pw as i32);
    let block_floor = *scratch_of(&events, DIT_BLOCK_QUANTUM).iter().max().unwrap();
    assert_eq!(
        scratch_of(&events, DIT_EMBED_QUANTUM),
        [block_floor
            + 16 * latents.size() as u64 * 4
            + 4 * tokens as u64 * cfg.attention_head_dim as u64 * 4]
    );

    // A bounded attention plan is priced (fewer query rows) and still byte-identical when guarded.
    let plan = AttentionPlan::budgeted(AttentionBudget::from_score_elements(64, true));
    let bounded = || dit.forward_bounded(&latents, &sigma, &encoder, Dtype::Float32, plan, None);
    let host_bounded = f32_bytes(&bounded().unwrap());
    let (guarded_bounded, bounded_events) = observed(AMPLE, bounded);
    assert_eq!(f32_bytes(&guarded_bounded.unwrap()), host_bounded);
    let block_scratch = |events: &[QuantumEvent]| {
        events.iter().find_map(|e| match e {
            QuantumEvent::Checked { label, scratch } if label == DIT_BLOCK_QUANTUM => {
                Some(*scratch)
            }
            _ => None,
        })
    };
    assert!(
        block_scratch(&bounded_events).unwrap() < block_scratch(&events).unwrap(),
        "a chunked plan must price its chunk's query rows, not every token"
    );

    // Refusal before construction: the first guarded quantum refuses, nothing is built or synced.
    take_refusal();
    let (refused, refused_events) = observed(0, run);
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, DIT_EMBED_QUANTUM);
    assert!(
        refused_events.is_empty(),
        "nothing may run past a refusal: {refused_events:?}"
    );
}

/// The stage-4 golden hook shares `run_blocks`, so its partial stack is guarded per block too.
#[test]
fn dit_partial_stack_hook_is_guarded_per_block() {
    if !in_cpu_child("dit_partial_stack_hook_is_guarded_per_block") {
        return;
    }
    let cfg = dit_cfg(3, 2.0);
    let dit = CosmosDiT::synthetic(cfg, 5);
    let (latents, sigma, encoder) = dit_inputs(&cfg, 8, 6);
    let run = || dit.forward_hidden(&latents, &sigma, &encoder, Dtype::Float32, Some(2));
    let host = f32_bytes(&run().unwrap());
    let (guarded, events) = observed(AMPLE, run);
    assert_eq!(f32_bytes(&guarded.unwrap()), host);
    assert_checked_before_synchronized(&events);
    assert_eq!(checked(&events, DIT_BLOCK_QUANTUM), 2);
}

/// Every tensor `CosmosDiT::from_weights(.., "net", cfg)` reads, written to a temp safetensors so
/// rung 4 can re-open it per window.
fn write_dit_checkpoint(dir: &std::path::Path, cfg: &DitConfig) -> std::path::PathBuf {
    let hidden = cfg.hidden_size() as i32;
    let adaln = cfg.adaln_lora_dim as i32;
    let ctx = cfg.text_embed_dim as i32;
    let head = cfg.attention_head_dim as i32;
    let ff = (hidden as f32 * cfg.mlp_ratio) as i32;
    let (pt, ph, pw) = cfg.patch_size;
    let patch_in = (cfg.patch_in_channels() * pt * ph * pw) as i32;
    let patch_out = (pt * ph * pw * cfg.out_channels) as i32;
    let mut named: Vec<(String, Array)> = Vec::new();
    let mut seed = 100u64;
    let mut push = |name: String, shape: Vec<i32>, norm: bool| {
        seed += 1;
        let value = if norm {
            Array::ones::<f32>(&shape).unwrap()
        } else {
            let w = random::normal::<f32>(&shape, None, None, Some(&random::key(seed).unwrap()))
                .unwrap();
            mlx_rs::ops::multiply(&w, Array::from_slice(&[0.05f32], &[1])).unwrap()
        };
        named.push((name, value));
    };
    for i in 0..cfg.num_layers {
        let b = format!("net.blocks.{i}");
        for attn in ["self_attn", "cross_attn"] {
            let kv_in = if attn == "cross_attn" { ctx } else { hidden };
            push(
                format!("{b}.{attn}.q_proj.weight"),
                vec![hidden, hidden],
                false,
            );
            push(
                format!("{b}.{attn}.k_proj.weight"),
                vec![hidden, kv_in],
                false,
            );
            push(
                format!("{b}.{attn}.v_proj.weight"),
                vec![hidden, kv_in],
                false,
            );
            push(
                format!("{b}.{attn}.output_proj.weight"),
                vec![hidden, hidden],
                false,
            );
            push(format!("{b}.{attn}.q_norm.weight"), vec![head], true);
            push(format!("{b}.{attn}.k_norm.weight"), vec![head], true);
        }
        push(format!("{b}.mlp.layer1.weight"), vec![ff, hidden], false);
        push(format!("{b}.mlp.layer2.weight"), vec![hidden, ff], false);
        for norm in [
            "adaln_modulation_self_attn",
            "adaln_modulation_cross_attn",
            "adaln_modulation_mlp",
        ] {
            push(format!("{b}.{norm}.1.weight"), vec![adaln, hidden], false);
            push(
                format!("{b}.{norm}.2.weight"),
                vec![3 * hidden, adaln],
                false,
            );
        }
    }
    push(
        "net.x_embedder.proj.1.weight".into(),
        vec![hidden, patch_in],
        false,
    );
    push(
        "net.t_embedder.1.linear_1.weight".into(),
        vec![hidden, hidden],
        false,
    );
    push(
        "net.t_embedder.1.linear_2.weight".into(),
        vec![3 * hidden, hidden],
        false,
    );
    push("net.t_embedding_norm.weight".into(), vec![hidden], true);
    push(
        "net.final_layer.adaln_modulation.1.weight".into(),
        vec![adaln, hidden],
        false,
    );
    push(
        "net.final_layer.adaln_modulation.2.weight".into(),
        vec![2 * hidden, adaln],
        false,
    );
    push(
        "net.final_layer.linear.weight".into(),
        vec![patch_out, hidden],
        false,
    );
    let path = dir.join("dit.safetensors");
    let refs: Vec<(&str, &Array)> = named.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &path).unwrap();
    path
}

#[test]
fn dit_windowed_blocks_are_guarded_quanta() {
    if !in_cpu_child("dit_windowed_blocks_are_guarded_quanta") {
        return;
    }
    let cfg = dit_cfg(3, 2.0);
    let tmp = tempfile::tempdir().unwrap();
    let path = write_dit_checkpoint(tmp.path(), &cfg);
    let weights = Weights::from_file(&path).unwrap();
    let dit = CosmosDiT::from_weights(&weights, "net", cfg)
        .unwrap()
        .with_block_stream(WeightsSource::File(path.clone()), "net");
    let (latents, sigma, encoder) = dit_inputs(&cfg, 8, 6);
    let cancel = CancelFlag::default();
    let windowed = || {
        let window = dit.block_window(Some(1), &cancel)?;
        assert!(window.is_some(), "rung 4 must actually window");
        dit.forward_bounded(
            &latents,
            &sigma,
            &encoder,
            Dtype::Float32,
            AttentionPlan::UNBOUNDED,
            window,
        )
    };

    let host = f32_bytes(&windowed().unwrap());
    let resident = f32_bytes(
        &dit.forward(&latents, &sigma, &encoder, Dtype::Float32)
            .unwrap(),
    );
    assert_eq!(host, resident, "fixture: windowed must equal resident");
    let (guarded, events) = observed(AMPLE, windowed);
    assert_eq!(
        f32_bytes(&guarded.unwrap()),
        host,
        "guards changed numerics"
    );
    assert_checked_before_synchronized(&events);
    assert_eq!(checked(&events, DIT_BLOCK_QUANTUM), cfg.num_layers);

    take_refusal();
    let (refused, refused_events) = observed(0, windowed);
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, DIT_EMBED_QUANTUM);
    assert!(refused_events.is_empty(), "{refused_events:?}");
}

// ---- Conditioner ---------------------------------------------------------------------------------

fn conditioner_cfg(mlp_ratio: f32) -> ConditionerConfig {
    ConditionerConfig {
        source_dim: 12,
        target_dim: 16,
        model_dim: 16,
        num_layers: 3,
        num_attention_heads: 2,
        mlp_ratio,
        target_vocab_size: 32,
        min_sequence_length: 24,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
    }
}

fn conditioner_inputs(cfg: &ConditionerConfig, ss: i32, st: i32) -> (Array, Array) {
    let source = random::normal::<f32>(
        &[1, ss, cfg.source_dim as i32][..],
        None,
        None,
        Some(&random::key(3).unwrap()),
    )
    .unwrap();
    let ids: Vec<i32> = (0..st)
        .map(|i| (i * 5 + 1) % cfg.target_vocab_size as i32)
        .collect();
    (source, Array::from_slice(&ids, &[1, st]))
}

#[test]
fn conditioner_blocks_are_guarded_quanta() {
    if !in_cpu_child("conditioner_blocks_are_guarded_quanta") {
        return;
    }
    let cfg = conditioner_cfg(4.0);
    let cond = AnimaTextConditioner::synthetic(cfg, 21);
    let (source, ids) = conditioner_inputs(&cfg, 7, 5);
    let weights = [1.0f32, 1.25, 1.0, 0.8, 1.0];
    let run = || cond.forward_weighted(&source, &ids, Some(&weights), Dtype::Float32);

    let host = f32_bytes(&run().unwrap());
    let (guarded, events) = observed(AMPLE, run);
    let guarded = guarded.unwrap();
    assert_eq!(guarded.shape(), &[1, 24, 16]);
    assert_eq!(f32_bytes(&guarded), host, "guards changed numerics");
    assert_checked_before_synchronized(&events);
    assert_eq!(checked(&events, CONDITIONER_BLOCK_QUANTUM), cfg.num_layers);
    assert_eq!(checked(&events, CONDITIONER_EMBED_QUANTUM), 1);
    assert_eq!(checked(&events, CONDITIONER_OUTPUT_QUANTUM), 1);
    // The entry quantum materializes the caller's lazy Qwen3 mask-multiply: the largest block
    // floor + three source-sized f32 buffers (states, cast mask, product).
    let block_floor = *scratch_of(&events, CONDITIONER_BLOCK_QUANTUM)
        .iter()
        .max()
        .unwrap();
    assert_eq!(
        scratch_of(&events, CONDITIONER_EMBED_QUANTUM),
        [block_floor + 3 * source.size() as u64 * 4]
    );

    take_refusal();
    let (refused, refused_events) = observed(0, run);
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, CONDITIONER_EMBED_QUANTUM);
    assert!(refused_events.is_empty(), "{refused_events:?}");

    // The padded 512-style output is the largest quantum: admit everything before it, and prove
    // every block ran and nothing after the refusal did.
    let label = assert_refuses_at_first_growth(&events, run);
    assert_eq!(label, CONDITIONER_OUTPUT_QUANTUM);
}

// ---- Qwen3 text encoder --------------------------------------------------------------------------

/// hidden 6 ≠ heads·head_dim 8 and intermediate 18 — the real Qwen3-0.6B asymmetry, so a floor that
/// confused hidden with heads·head_dim (or read a constant) cannot match.
fn tiny_qwen3_cfg(n_layers: usize) -> Qwen3Config {
    Qwen3Config {
        vocab_size: 16,
        hidden_size: 6,
        n_layers,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    }
}

fn tiny_qwen3_weights(cfg: &Qwen3Config, inter: i32) -> Weights {
    let h = cfg.hidden_size;
    let qd = cfg.n_heads * cfg.head_dim;
    let kvd = cfg.n_kv_heads * cfg.head_dim;
    let mut w = Weights::empty();
    let mut seed = 500u64;
    let mut synth = |w: &mut Weights, key: String, shape: &[i32]| {
        seed += 1;
        let value = if key.ends_with("norm.weight") {
            Array::ones::<f32>(shape).unwrap()
        } else {
            let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
                .unwrap();
            mlx_rs::ops::multiply(&a, Array::from_slice(&[0.3f32], &[1])).unwrap()
        };
        w.insert(key, value);
    };
    synth(
        &mut w,
        "model.embed_tokens.weight".into(),
        &[cfg.vocab_size, h],
    );
    for i in 0..cfg.n_layers {
        let b = format!("model.layers.{i}");
        synth(&mut w, format!("{b}.input_layernorm.weight"), &[h]);
        synth(&mut w, format!("{b}.post_attention_layernorm.weight"), &[h]);
        synth(&mut w, format!("{b}.self_attn.q_proj.weight"), &[qd, h]);
        synth(&mut w, format!("{b}.self_attn.k_proj.weight"), &[kvd, h]);
        synth(&mut w, format!("{b}.self_attn.v_proj.weight"), &[kvd, h]);
        synth(&mut w, format!("{b}.self_attn.o_proj.weight"), &[h, qd]);
        synth(
            &mut w,
            format!("{b}.self_attn.q_norm.weight"),
            &[cfg.head_dim],
        );
        synth(
            &mut w,
            format!("{b}.self_attn.k_norm.weight"),
            &[cfg.head_dim],
        );
        synth(&mut w, format!("{b}.mlp.gate_proj.weight"), &[inter, h]);
        synth(&mut w, format!("{b}.mlp.up_proj.weight"), &[inter, h]);
        synth(&mut w, format!("{b}.mlp.down_proj.weight"), &[h, inter]);
    }
    synth(&mut w, "model.norm.weight".into(), &[h]);
    w
}

fn qwen3_inputs(len: i32) -> (Array, Array) {
    let ids: Vec<i32> = (0..len).map(|i| (i * 3 + 2) % 16).collect();
    let mut mask = vec![1i32; len as usize];
    *mask.last_mut().unwrap() = 0;
    (
        Array::from_slice(&ids, &[1, len]),
        Array::from_slice(&mask, &[1, len]),
    )
}

#[test]
fn text_encoder_layers_are_guarded_quanta() {
    if !in_cpu_child("text_encoder_layers_are_guarded_quanta") {
        return;
    }
    let cfg = tiny_qwen3_cfg(3);
    let te = AnimaQwen3::from_weights_dtype(
        &tiny_qwen3_weights(&cfg, 18),
        "model",
        &cfg,
        Dtype::Float32,
    )
    .unwrap();
    let (ids, mask) = qwen3_inputs(5);
    let run = || te.forward(&ids, &mask);

    let host = f32_bytes(&run().unwrap());
    let (guarded, events) = observed(AMPLE, run);
    assert_eq!(
        f32_bytes(&guarded.unwrap()),
        host,
        "guards changed numerics"
    );
    assert_checked_before_synchronized(&events);
    assert_eq!(checked(&events, TE_LAYER_QUANTUM), cfg.n_layers);
    assert_eq!(checked(&events, TE_EMBED_QUANTUM), 1);
    assert_eq!(checked(&events, TE_NORM_QUANTUM), 1);

    take_refusal();
    let (refused, refused_events) = observed(0, run);
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, TE_EMBED_QUANTUM);
    assert!(refused_events.is_empty(), "{refused_events:?}");
}

// ---- Scratch geometry ----------------------------------------------------------------------------

#[test]
fn scratch_derives_from_loaded_tensors_and_grows_with_geometry() {
    if !in_cpu_child("scratch_derives_from_loaded_tensors_and_grows_with_geometry") {
        return;
    }
    let scratch_of = |events: &[QuantumEvent], label: &str| {
        events
            .iter()
            .find_map(|e| match e {
                QuantumEvent::Checked { label: l, scratch } if l == label => Some(*scratch),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no {label} check in {events:?}"))
    };

    // DiT: exact floor from the loaded block (MLP width from the tensors, not `DitConfig::anima()`),
    // growing with image tokens and with the loaded MLP width.
    let dit_block = |cfg: DitConfig, edge: i32| {
        let dit = CosmosDiT::synthetic(cfg, 11);
        let (latents, sigma, encoder) = dit_inputs(&cfg, edge, 6);
        let (result, events) = observed(AMPLE, || {
            dit.forward(&latents, &sigma, &encoder, Dtype::Float32)
        });
        result.unwrap();
        scratch_of(&events, DIT_BLOCK_QUANTUM)
    };
    let base = dit_block(dit_cfg(1, 2.0), 8);
    let (h, text, ff, adaln, head_dim) = (16u64, 12u64, 32u64, 8u64, 8u64);
    // self q/k/v/out + cross q/out (h·h each), cross k/v (h·text), four q/k norms, three adaLN
    // down/up pairs, and the MLP.
    let weight_elements =
        6 * h * h + 2 * h * text + 4 * head_dim + 3 * (adaln * h + 3 * h * adaln) + 2 * ff * h;
    let (heads, tokens, ctx) = (2u64, 16u64, 6u64);
    assert_eq!(
        base,
        mlx_gen::ios_quantum::attention_block_bytes(mlx_gen::ios_quantum::AttentionBlockGeometry {
            batch: 1,
            tokens,
            dim: h,
            heads,
            context_tokens: ctx,
            ff_hidden: ff,
            query_rows: tokens,
            weight_elements,
        })
    );
    assert!(
        dit_block(dit_cfg(1, 2.0), 16) > base,
        "more tokens, more scratch"
    );
    assert!(
        dit_block(dit_cfg(1, 3.0), 8) > base,
        "a wider loaded MLP, more scratch"
    );

    // Text encoder: the layer floor reads hidden/intermediate/heads/head_dim off the tensors.
    let te_layer = |inter: i32, len: i32| {
        let cfg = tiny_qwen3_cfg(1);
        let te = AnimaQwen3::from_weights_dtype(
            &tiny_qwen3_weights(&cfg, inter),
            "model",
            &cfg,
            Dtype::Float32,
        )
        .unwrap();
        let (ids, mask) = qwen3_inputs(len);
        let (result, events) = observed(AMPLE, || te.forward(&ids, &mask));
        result.unwrap();
        scratch_of(&events, TE_LAYER_QUANTUM)
    };
    assert_eq!(
        te_layer(18, 5),
        mlx_gen::ios_quantum::transformer_layer_bytes(1, 5, 6, 18, 2, 4)
    );
    assert!(te_layer(18, 9) > te_layer(18, 5));
    assert!(te_layer(30, 5) > te_layer(18, 5));

    // Conditioner: grows with the token count and the loaded MLP width.
    let cond_block = |ratio: f32, st: i32| {
        let cfg = conditioner_cfg(ratio);
        let cond = AnimaTextConditioner::synthetic(cfg, 21);
        let (source, ids) = conditioner_inputs(&cfg, 7, st);
        let (result, events) = observed(AMPLE, || cond.forward(&source, &ids, Dtype::Float32));
        result.unwrap();
        scratch_of(&events, CONDITIONER_BLOCK_QUANTUM)
    };
    assert_eq!(
        cond_block(4.0, 5),
        mlx_gen::ios_quantum::transformer_layer_bytes(1, 7, 16, 64, 2, 8)
    );
    assert!(cond_block(4.0, 12) > cond_block(4.0, 5));
    assert!(cond_block(2.0, 5) < cond_block(4.0, 5));
}

// ---- Pre-load floors -----------------------------------------------------------------------------

/// Both phase loaders check the header-only floor (every tensor's bytes plus the largest one)
/// BEFORE constructing anything: a refusal at zero headroom names the load and prices exactly the
/// on-disk inventory of the files that phase would construct from.
#[test]
fn phase_loads_check_the_header_floor_before_construction() {
    if !in_cpu_child("phase_loads_check_the_header_floor_before_construction") {
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("split_files");
    for dir in ["diffusion_models", "text_encoders", "vae"] {
        std::fs::create_dir_all(root.join(dir)).unwrap();
    }
    let cfg = dit_cfg(2, 2.0);
    let dit = write_dit_checkpoint(&root.join("diffusion_models"), &cfg);
    let dit_path = root
        .join("diffusion_models")
        .join(crate::config::Variant::Turbo.dit_filename());
    std::fs::rename(dit, &dit_path).unwrap();
    let te_path = root.join("text_encoders/qwen_3_06b_base.safetensors");
    let te = tiny_qwen3_weights(&tiny_qwen3_cfg(1), 18);
    let te_named: Vec<(String, Array)> = te
        .keys()
        .map(|k| (k.to_string(), te.require(k).unwrap().clone()))
        .collect();
    let refs: Vec<(&str, &Array)> = te_named.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, &te_path).unwrap();
    let vae_path = root.join("vae/qwen_image_vae.safetensors");
    let vae_tensor = Array::ones::<f32>(&[4, 4]).unwrap();
    Array::save_safetensors(vec![("conv1.weight", &vae_tensor)], None, &vae_path).unwrap();

    let floor = |paths: &[&std::path::Path]| {
        let sizes: Vec<u64> = paths
            .iter()
            .flat_map(|p| {
                mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(p)
                    .unwrap()
                    .into_iter()
                    .map(|h| h.data_bytes)
            })
            .collect();
        sizes.iter().sum::<u64>() + sizes.iter().copied().max().unwrap()
    };
    let source = WeightsSource::Dir(root.clone());

    take_refusal();
    let (refused, events) = observed(0, || {
        crate::loader::load_text_phase(&source, crate::config::Variant::Turbo).map(|_| ())
    });
    assert!(refused.is_err());
    let refusal = take_refusal().unwrap();
    assert_eq!(refusal.label, crate::loader::TEXT_LOAD_QUANTUM);
    assert_eq!(refusal.scratch_bytes, floor(&[&te_path]));
    assert!(events.is_empty(), "{events:?}");

    let (refused, events) = observed(0, || {
        crate::loader::load_heavy_phase(&source, crate::config::Variant::Turbo).map(|_| ())
    });
    assert!(refused.is_err());
    let refusal = take_refusal().unwrap();
    assert_eq!(refusal.label, crate::loader::HEAVY_LOAD_QUANTUM);
    assert_eq!(refusal.scratch_bytes, floor(&[&dit_path, &vae_path]));
    assert!(events.is_empty(), "{events:?}");

    // A streamed (rung-4) load prices only what it makes resident: the DiT's non-block tensors and
    // the conditioner, plus the VAE. Block bodies are priced per window by each block's quantum.
    let resident_floor = {
        let mut sizes: Vec<u64> = Vec::new();
        let dit_headers =
            mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&dit_path).unwrap();
        let prefix = dit_headers
            .iter()
            .find_map(|h| h.name.strip_suffix(".x_embedder.proj.1.weight"))
            .unwrap()
            .to_string();
        let mut block_bytes = 0;
        for h in dit_headers {
            if h.name.starts_with(&format!("{prefix}.blocks.")) {
                block_bytes += h.data_bytes;
            } else {
                sizes.push(h.data_bytes);
            }
        }
        assert!(block_bytes > 0, "the fixture must carry block bodies");
        sizes.extend(
            mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&vae_path)
                .unwrap()
                .into_iter()
                .map(|h| h.data_bytes),
        );
        sizes.iter().sum::<u64>() + sizes.iter().copied().max().unwrap()
    };
    assert!(resident_floor < floor(&[&dit_path, &vae_path]));
    let (refused, events) = observed(0, || {
        crate::loader::load_heavy_phase_with_stream(&source, crate::config::Variant::Turbo, true)
            .map(|_| ())
    });
    assert!(refused.is_err());
    let refusal = take_refusal().unwrap();
    assert_eq!(refusal.label, crate::loader::HEAVY_LOAD_QUANTUM);
    assert_eq!(refusal.scratch_bytes, resident_floor);
    assert!(events.is_empty(), "{events:?}");

    // With headroom the floor passes and is observed as a checked boundary before the load runs.
    let (_, events) = observed(AMPLE, || {
        crate::loader::load_heavy_phase(&source, crate::config::Variant::Turbo).map(|_| ())
    });
    assert!(matches!(
        events.first(),
        Some(QuantumEvent::Checked { label, .. }) if label == crate::loader::HEAVY_LOAD_QUANTUM
    ));
}
