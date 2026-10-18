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
use crate::transformer::{
    sdpa_scores, Backend, CosmosDiT, DIT_BLOCK_QUANTUM, DIT_EMBED_QUANTUM, DIT_OUTPUT_QUANTUM,
};

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
                // An `#[ignore]`d prediction is only ever reached by an explicit `--ignored` run.
                "--include-ignored",
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
    // The entry quantum is priced at its own counted body (which carries this grid's RoPE
    // allowance) plus the previous step's lazy CFG combine + sampler update: 16 latent-sized f32
    // buffers.
    assert_eq!(
        scratch_of(&events, DIT_EMBED_QUANTUM),
        [dit.embed_body_bytes(latents.shape(), 4) + 16 * latents.size() as u64 * 4]
    );
    let sh = latents.shape();
    let (pt, ph, pw) = cfg.patch_size;
    let tokens = (sh[2] / pt as i32) * (sh[3] / ph as i32) * (sh[4] / pw as i32);
    assert!(
        dit.embed_body_bytes(latents.shape(), 4)
            >= 4 * tokens as u64 * cfg.attention_head_dim as u64 * 4,
        "the entry carries the RoPE allowance"
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
        "a chunked plan must price its synchronized attention phase, not both attentions"
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
    write_dit_checkpoint_as(dir, cfg, Dtype::Float32)
}

/// [`write_dit_checkpoint`] stored at `dtype` (bf16 = the shipped checkpoint's storage).
fn write_dit_checkpoint_as(
    dir: &std::path::Path,
    cfg: &DitConfig,
    dtype: Dtype,
) -> std::path::PathBuf {
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
        named.push((name, value.as_dtype(dtype).unwrap()));
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
    // Each streamed block's quantum reads its own stored bytes inside it, and is priced with
    // exactly them on top of the resident block's scratch.
    let (_, resident_events) = observed(AMPLE, || {
        dit.forward(&latents, &sigma, &encoder, Dtype::Float32)
    });
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path).unwrap();
    let stored: Vec<u64> = (0..cfg.num_layers)
        .map(|i| {
            headers
                .iter()
                .filter(|h| h.name.starts_with(&format!("net.blocks.{i}.")))
                .map(|h| h.data_bytes)
                .sum()
        })
        .collect();
    let streamed_scratch = scratch_of(&events, DIT_BLOCK_QUANTUM);
    let resident_scratch = scratch_of(&resident_events, DIT_BLOCK_QUANTUM);
    for i in 0..cfg.num_layers {
        assert_eq!(
            streamed_scratch[i],
            resident_scratch[i] + stored[i],
            "block {i}"
        );
    }

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
    // The entry quantum: its own counted body plus the caller's lazy Qwen3 mask-multiply, three
    // source-sized f32 buffers (states, cast mask, product).
    assert_eq!(
        scratch_of(&events, CONDITIONER_EMBED_QUANTUM),
        [cond.embed_body_bytes(1, 5, 7, 4) + 3 * source.size() as u64 * 4]
    );

    take_refusal();
    let (refused, refused_events) = observed(0, run);
    assert!(refused.is_err());
    assert_eq!(take_refusal().unwrap().label, CONDITIONER_EMBED_QUANTUM);
    assert!(refused_events.is_empty(), "{refused_events:?}");

    // The first block outgrows the entry: admit only the entry, and prove it ran and nothing at or
    // after the refused block did.
    let label = assert_refuses_at_first_growth(&events, run);
    assert_eq!(label, CONDITIONER_BLOCK_QUANTUM);
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
    let (h, ff) = (16u64, 32u64);
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
            element_bytes: 4,
            weight_bytes: 0,
            attn_hidden_live: crate::transformer::BLOCK_ATTN_HIDDEN_LIVE,
            context_live: crate::transformer::BLOCK_CONTEXT_LIVE,
            // The CPU child's SDPA is the unfused fallback: logits + softmax, f32.
            scores: mlx_gen::ios_quantum::Scores::Materialized {
                live: 2,
                element_bytes: 4,
            },
            mlp_hidden_live: crate::transformer::BLOCK_MLP_HIDDEN_LIVE,
            ff_live: crate::transformer::BLOCK_FF_LIVE,
        })
    );
    assert!(
        dit_block(dit_cfg(1, 2.0), 16) > base,
        "more tokens, more scratch"
    );
    // Where the MLP phase binds (x12 and wider here, above the attention phase with the CPU's
    // materialized scores), a wider loaded MLP costs more scratch.
    assert!(
        dit_block(dit_cfg(1, 13.0), 8) > dit_block(dit_cfg(1, 12.0), 8),
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
    // q (2 heads x 4 = 8) is 2 hidden units wide (rounded up), k/v (1 x 4) 1 unit, the MLP (18)
    // 3 units: 10 hidden + 8 q + 5 kv + 3 intermediate widths, one phase; f32 weights at the f32
    // width are not promoted.
    let te_geometry = mlx_gen::ios_quantum::AttentionBlockGeometry {
        batch: 1,
        tokens: 5,
        dim: 6,
        heads: 2,
        context_tokens: 0,
        ff_hidden: 18,
        query_rows: 5,
        element_bytes: 4,
        weight_bytes: 0,
        attn_hidden_live: 10 + 8 * 2 + 5 + 3 * 3,
        context_live: 0,
        // The CPU child's SDPA is the unfused fallback: logits + softmax, f32.
        scores: mlx_gen::ios_quantum::Scores::Materialized {
            live: 2,
            element_bytes: 4,
        },
        mlp_hidden_live: 0,
        ff_live: 0,
    };
    assert_eq!(
        te_layer(18, 5),
        mlx_gen::ios_quantum::attention_block_bytes(te_geometry)
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
    // 5 target tokens over 7 source tokens, model_dim 16, MLP x4.
    assert_eq!(
        cond_block(4.0, 5),
        mlx_gen::ios_quantum::attention_block_bytes(mlx_gen::ios_quantum::AttentionBlockGeometry {
            batch: 1,
            tokens: 5,
            dim: 16,
            heads: 2,
            context_tokens: 7,
            ff_hidden: 64,
            query_rows: 5,
            element_bytes: 4,
            weight_bytes: 0,
            attn_hidden_live: crate::conditioner::COND_ATTN_HIDDEN_LIVE,
            context_live: crate::conditioner::COND_CONTEXT_LIVE,
            // The CPU child's SDPA is the unfused fallback: logits + softmax, f32.
            scores: mlx_gen::ios_quantum::Scores::Materialized {
                live: 2,
                element_bytes: 4,
            },
            mlp_hidden_live: crate::conditioner::COND_MLP_HIDDEN_LIVE,
            ff_live: crate::conditioner::COND_FF_LIVE,
        })
    );
    assert!(cond_block(4.0, 12) > cond_block(4.0, 5));
    // Where the MLP phase binds (x9 and wider here), a wider loaded MLP costs more scratch.
    assert!(cond_block(10.0, 5) < cond_block(12.0, 5));

    // Each production call site prices the scores of the backend it RUNS on: at head dim 64, which
    // the Metal kernel serves (fused, no score term), the CPU child's checked scratch is the
    // materialized pricing — a call site hard-coding the device treatment would check less.
    let distinct = |label: &str, checked: u64, at: &dyn Fn(Backend) -> u64| {
        assert!(
            at(Backend::Metal) < at(Backend::Cpu),
            "{label}: head dim 64 is fused on Metal"
        );
        assert_eq!(checked, at(Backend::Cpu), "{label}");
    };
    {
        let cfg = DitConfig {
            num_attention_heads: 1,
            attention_head_dim: 64,
            ..dit_cfg(1, 2.0)
        };
        let dit = CosmosDiT::synthetic(cfg, 11);
        let (latents, sigma, encoder) = dit_inputs(&cfg, 8, 6);
        let (result, events) = observed(AMPLE, || {
            dit.forward(&latents, &sigma, &encoder, Dtype::Float32)
        });
        result.unwrap();
        let block = &dit.blocks()[0];
        distinct(
            DIT_BLOCK_QUANTUM,
            scratch_of(&events, DIT_BLOCK_QUANTUM),
            &|backend| block.quantum_scratch(1, 16, 6, AttentionPlan::UNBOUNDED, 4, 0, backend),
        );
    }
    {
        let cfg = ConditionerConfig {
            target_dim: 64,
            model_dim: 64,
            num_attention_heads: 1,
            ..conditioner_cfg(4.0)
        };
        let cond = AnimaTextConditioner::synthetic(cfg, 21);
        let (source, ids) = conditioner_inputs(&cfg, 7, 12);
        let (result, events) = observed(AMPLE, || cond.forward(&source, &ids, Dtype::Float32));
        result.unwrap();
        let block = &cond.blocks()[0];
        distinct(
            CONDITIONER_BLOCK_QUANTUM,
            scratch_of(&events, CONDITIONER_BLOCK_QUANTUM),
            &|backend| block.quantum_scratch(1, 12, 7, 4, backend),
        );
    }
    {
        let cfg = Qwen3Config {
            head_dim: 64,
            ..tiny_qwen3_cfg(1)
        };
        let te = AnimaQwen3::from_weights_dtype(
            &tiny_qwen3_weights(&cfg, 18),
            "model",
            &cfg,
            Dtype::Float32,
        )
        .unwrap();
        let (ids, mask) = qwen3_inputs(12);
        let (result, events) = observed(AMPLE, || te.forward(&ids, &mask));
        result.unwrap();
        distinct(
            TE_LAYER_QUANTUM,
            scratch_of(&events, TE_LAYER_QUANTUM),
            &|backend| te.layer(0, 1, 12, 4, backend).1,
        );
    }
}

// ---- Pre-load floors -----------------------------------------------------------------------------

/// Both phase loaders check the header-only floor (every tensor's stored bytes)
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
    // A bf16 VAE file, which the loader casts to f32: priced at stored + cast bytes. Large enough to
    // be the deciding floor, so a headroom that admits the DiT refuses the VAE.
    let vae_path = root.join("vae/qwen_image_vae.safetensors");
    let vae_tensor = Array::ones::<f32>(&[256, 256])
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    Array::save_safetensors(vec![("conv1.weight", &vae_tensor)], None, &vae_path).unwrap();
    let vae_cast_floor = 256 * 256 * 2 + 256 * 256 * 4;

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
        // `before_load`'s proven floor: the stored bytes (no extra per-tensor read copy).
        sizes.iter().sum::<u64>()
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
    assert_eq!(refusal.scratch_bytes, floor(&[&dit_path]));
    assert!(events.is_empty(), "{events:?}");

    // The VAE is checked separately, right before it is read (after the DiT and conditioner are
    // materialized, so the live headroom carries them), at its stored bytes + their f32 casts: a
    // headroom that admits the DiT's floor refuses it before it is read.
    assert!(vae_cast_floor > floor(&[&dit_path]));
    let (refused, events) = observed(floor(&[&dit_path]) + PROCESS_RESERVE_BYTES, || {
        crate::loader::check_vae_load(&root)
    });
    assert!(refused.is_err());
    let refusal = take_refusal().unwrap();
    assert_eq!(refusal.label, crate::loader::VAE_LOAD_QUANTUM);
    assert_eq!(refusal.scratch_bytes, vae_cast_floor);
    assert!(events.is_empty(), "{events:?}");

    // A streamed (rung-4) load prices only what it makes resident: the DiT's non-block tensors and
    // the conditioner. Block bodies are priced per window by each block's quantum.
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
        // `before_load`'s proven floor: the stored bytes (no extra per-tensor read copy).
        sizes.iter().sum::<u64>()
    };
    assert!(resident_floor < floor(&[&dit_path]));
    let (refused, events) = observed(0, || {
        crate::loader::load_heavy_phase_with_stream(&source, crate::config::Variant::Turbo, true)
            .map(|_| ())
    });
    assert!(refused.is_err());
    let refusal = take_refusal().unwrap();
    assert_eq!(refusal.label, crate::loader::HEAVY_LOAD_QUANTUM);
    assert_eq!(refusal.scratch_bytes, resident_floor);
    assert!(events.is_empty(), "{events:?}");

    // With headroom the DiT floor passes as a checked boundary before the load runs; the VAE floor
    // is NOT checked up front (the fixture checkpoint carries no conditioner, so the load stops at
    // its construction, before the VAE).
    let (loaded, events) = observed(AMPLE, || {
        crate::loader::load_heavy_phase(&source, crate::config::Variant::Turbo).map(|_| ())
    });
    assert!(loaded.is_err(), "the fixture has no conditioner");
    assert_eq!(
        events,
        [QuantumEvent::Checked {
            label: crate::loader::HEAVY_LOAD_QUANTUM.into(),
            scratch: floor(&[&dit_path]),
        }]
    );
    let (_, events) = observed(AMPLE, || crate::loader::check_vae_load(&root));
    assert_eq!(
        events,
        [QuantumEvent::Checked {
            label: crate::loader::VAE_LOAD_QUANTUM.into(),
            scratch: vae_cast_floor,
        }]
    );
}

// ---- Measured-peak proofs ------------------------------------------------------------------------
//
// Every quantum type's counted formula, proven against MLX's measured peak
// (`mlx_gen::ios_quantum::measured_peak_bytes`) on the REAL module at the shipped bf16 storage and
// compute width, with every input and resident weight materialized first, so the measurement is the
// quantum body's own allocation. Each proof asserts formula >= measured and formula <= MAX_RATIO x
// measured (`mlx_gen::ios_quantum::assert_tracks_measured`).

/// The largest ratio any counted Anima formula may hold over its measured peak.
const MAX_RATIO: f64 = 1.5;

/// Runs per measurement. MLX's CPU stream retires buffers asynchronously, so one quantum's peak
/// varies run to run — by whole buffers (the final norm's measured peak has been seen to halve) —
/// and the proof takes the worst of several: enough that the worst run reaches the counted
/// retention (3 left the ratio bound flaky).
const MEASURE_RUNS: usize = 10;

fn measured(body: impl Fn() -> Result<Array>) -> u64 {
    measured_many(|| Ok(vec![body()?]))
}

fn measured_many(body: impl Fn() -> Result<Vec<Array>>) -> u64 {
    (0..MEASURE_RUNS)
        .map(|_| mlx_gen::ios_quantum::measured_peak_bytes(&body).unwrap())
        .max()
        .unwrap()
}

fn prove(label: &str, formula: u64, measured: u64) {
    println!(
        "{label}: formula {formula} B, measured {measured} B, ratio {:.3}",
        formula as f64 / measured.max(1) as f64
    );
    mlx_gen::ios_quantum::assert_tracks_measured(label, formula, measured, MAX_RATIO);
}

fn eval_all(arrays: &[&Array]) {
    mlx_rs::transforms::eval(arrays.iter().copied()).unwrap();
}

/// A DiT at the shipped proportions in miniature: hidden 128 (2 heads x 64), MLP x4, a 64-wide
/// conditioner stream, 16 latent channels.
fn proof_dit_cfg() -> DitConfig {
    DitConfig {
        in_channels: 16,
        out_channels: 16,
        num_attention_heads: 2,
        attention_head_dim: 64,
        num_layers: 1,
        mlp_ratio: 4.0,
        text_embed_dim: 64,
        adaln_lora_dim: 32,
        max_size: (4, 64, 64),
        patch_size: (1, 2, 2),
        rope_scale: (1.0, 4.0, 4.0),
        concat_padding_mask: true,
    }
}

/// The proofs run on MLX's CPU backend, which has no fused SDPA kernel: its fallback materializes
/// each attention's scores (`[B, H, Sq, Sk]`, the logits and their softmax), which the device's
/// fused kernel never allocates. Each proof prices its formula on the backend it measures —
/// `Backend::current()`, the CPU here — so [`crate::transformer::sdpa_scores`] adds exactly those
/// scores to the device formula's counted hidden buffers (and never twice).
fn proof_backend() -> Backend {
    let backend = Backend::current();
    assert_eq!(backend, Backend::Cpu, "the proofs measure on the CPU");
    backend
}

/// The score treatment follows MLX 0.32's Metal `use_fallback`: fused only on Metal, at a
/// kernel-instantiated width (f32/bf16/f16), for head dims `sdpa_full` (more than 8 rows: 64/80/128)
/// or `sdpa_vector` (at most 8 rows, no more than the keys: 64/96/128/256) serves — for every call,
/// including a chunked attention's remainder. Pure arithmetic: no device is touched.
#[test]
fn sdpa_scores_follow_the_metal_kernel_support() {
    use mlx_gen::ios_quantum::Scores;
    let fused = |backend, head_dim, eb, attentions: &[(u64, u64, u64)]| {
        sdpa_scores(backend, head_dim, eb, attentions) == Scores::Fused
    };
    let one = |rows: u64, keys: u64| [(rows, rows, keys)];
    // sdpa_full (> 8 query rows): head dims 64/80/128 at f32 and bf16; 96/256/32/384 fall back.
    for d in [64, 80, 128] {
        for eb in [4, 2] {
            assert!(
                fused(Backend::Metal, d, eb, &one(4096, 4608)),
                "full d={d} eb={eb}"
            );
        }
    }
    for d in [32, 96, 256, 384] {
        assert!(!fused(Backend::Metal, d, 4, &one(4096, 4608)), "full d={d}");
    }
    // sdpa_vector (<= 8 query rows, rows <= keys): 64/96/128/256; 80 falls back, as do more rows
    // than keys.
    for d in [64, 96, 128, 256] {
        assert!(fused(Backend::Metal, d, 2, &one(8, 512)), "vector d={d}");
    }
    assert!(!fused(Backend::Metal, 80, 2, &one(8, 512)));
    assert!(!fused(Backend::Metal, 128, 4, &one(8, 7)));
    // A width no kernel is instantiated for (f64) falls back.
    assert!(!fused(Backend::Metal, 128, 8, &one(4096, 4096)));
    // The CPU never fuses, whatever the shape.
    assert!(!fused(Backend::Cpu, 128, 4, &one(4096, 4608)));
    // A chunked attention: 100 queries in 32-row calls leaves a 4-row remainder, which
    // sdpa_vector serves at 128 but not at 80; an even split has no remainder call.
    assert!(fused(Backend::Metal, 128, 4, &[(100, 32, 100)]));
    assert!(!fused(Backend::Metal, 80, 4, &[(100, 32, 100)]));
    assert!(fused(Backend::Metal, 80, 4, &[(96, 32, 96)]));
    // One falling-back attention of a block (cross-attention over fewer keys than its 4 queries)
    // prices the block's scores as materialized.
    assert!(!fused(Backend::Metal, 64, 4, &[(4, 4, 4), (4, 4, 3)]));
    // The materialized treatment: logits + softmax at the compute width.
    assert_eq!(
        sdpa_scores(Backend::Cpu, 128, 2, &one(16, 16)),
        Scores::Materialized {
            live: 2,
            element_bytes: 2
        }
    );
    // The shipped geometry on the device: DiT (128) and the Qwen3 tower (128) at f32 over 4096
    // and 512 tokens, the conditioner (64) over 512 T5 x 512 Qwen3 tokens — all fused.
    assert!(fused(
        Backend::Metal,
        128,
        4,
        &[(4096, 4096, 4096), (4096, 4096, 512)]
    ));
    assert!(fused(Backend::Metal, 128, 4, &one(512, 512)));
    assert!(fused(
        Backend::Metal,
        64,
        4,
        &[(512, 512, 512), (512, 512, 512)]
    ));
}

#[test]
fn dit_formulas_track_the_measured_peak() {
    if !in_cpu_child("dit_formulas_track_the_measured_peak") {
        return;
    }
    let dtype = Dtype::Bfloat16;
    // Entry: its counted body (the sampler-step allowance covers the PREVIOUS step's lazy update,
    // which a materialized input does not carry, and is arithmetic-tested separately). Proven at
    // the shipped hidden : head_dim proportion (16 heads), where the RoPE tables are the minor term
    // they are in production; the block proofs keep 2 heads so the CPU's materialized fallback
    // scores stay small.
    {
        let cfg = DitConfig {
            num_attention_heads: 16,
            attention_head_dim: 8,
            ..proof_dit_cfg()
        };
        let dir = tempfile::tempdir().unwrap();
        let path = write_dit_checkpoint_as(dir.path(), &cfg, Dtype::Bfloat16);
        let weights = Weights::from_file(&path).unwrap();
        let dit = CosmosDiT::from_weights(&weights, "net", cfg).unwrap();
        weights.materialize().unwrap();
        let (latents, sigma, _) = dit_inputs(&cfg, 16, 32);
        eval_all(&[&latents, &sigma]);
        let entry = dit.embed_body_bytes(latents.shape(), 2);
        let measured_entry = measured_many(|| {
            let (h, rope, temb, emb) = dit.embed(&latents, &sigma, dtype)?;
            Ok(vec![h, rope.cos, rope.sin, temb, emb])
        });
        prove(DIT_EMBED_QUANTUM, entry, measured_entry);
    }

    // Blocks and output, with f32 weights (activations only: nothing to promote) and with the
    // shipped bf16 weights (+ the f32 copies the f32 stream promotes them to).
    // And with a q4-packed checkpoint (the convert-at-install tier): `quantized_matmul` consumes
    // the packed projections in place, so there are no copies either.
    let cfg = proof_dit_cfg();
    for (storage, packed) in [
        (Dtype::Float32, None),
        (Dtype::Bfloat16, None),
        (Dtype::Bfloat16, Some(4)),
    ] {
        let tag = |label: &str| match packed {
            Some(bits) => format!("{label} [q{bits} weights]"),
            None => format!("{label} [{storage:?} weights]"),
        };
        let tmp = tempfile::tempdir().unwrap();
        let mut path = write_dit_checkpoint_as(tmp.path(), &cfg, storage);
        if let Some(bits) = packed {
            let dst = tmp.path().join("packed.safetensors");
            crate::convert::quantize_anima_dit(&path, &dst, bits, 64).unwrap();
            path = dst;
        }
        let weights = Weights::from_file(&path).unwrap();
        let dit = CosmosDiT::from_weights(&weights, "net", cfg).unwrap();
        weights.materialize().unwrap();
        let (latents, sigma, encoder) = dit_inputs(&cfg, 16, 32);
        eval_all(&[&latents, &sigma, &encoder]);
        let (hidden, rope, temb, embedded) = dit.embed(&latents, &sigma, dtype).unwrap();
        eval_all(&[&hidden, &rope.cos, &rope.sin, &temb, &embedded]);
        let (b, t) = (hidden.shape()[0], hidden.shape()[1]);
        let ctx = encoder.shape()[1];
        let block = &dit.blocks()[0];
        let unbounded = AttentionPlan::UNBOUNDED;
        let run = |h: &Array, plan: AttentionPlan<'_>| {
            block.forward(h, &encoder, &embedded, &temb, (&rope.cos, &rope.sin), plan)
        };
        // The first block takes the bf16 patch embedding; every later block the f32 stream the
        // RoPE promotion produced (and in production the conditioner's f32 output as encoder).
        let f32_hidden = run(&hidden, unbounded).unwrap();
        f32_hidden.eval().unwrap();
        assert_eq!(
            f32_hidden.dtype(),
            Dtype::Float32,
            "the RoPE tables promote the stream"
        );
        let eb = crate::transformer::block_element_bytes(&hidden, &rope.cos);
        assert_eq!(eb, 4);
        if storage == Dtype::Float32 || packed.is_some() {
            assert_eq!(
                block.promoted_weight_bytes(eb),
                0,
                "f32 and packed weights are not promoted"
            );
        } else {
            assert!(
                block.promoted_weight_bytes(eb) > 0,
                "bf16 weights are promoted"
            );
        }
        let cpu = proof_backend();
        let formula = block.quantum_scratch(b, t, ctx, unbounded, eb, 0, cpu);
        prove(
            &tag("Anima DiT block (bf16 in)"),
            formula,
            measured(|| run(&hidden, unbounded)),
        );
        prove(
            &tag(DIT_BLOCK_QUANTUM),
            formula,
            measured(|| run(&f32_hidden, unbounded)),
        );

        // Rung 3: a query-chunked plan (each chunk evaluated before the next).
        let chunked =
            AttentionPlan::budgeted(AttentionBudget::from_score_elements(2 * 16 * 64, true));
        let rows = block
            .quantum_geometry(b, t, ctx, chunked, eb, 0, cpu)
            .query_rows;
        assert!(rows < t as u64, "the plan must genuinely chunk: {rows}");
        prove(
            &tag("Anima DiT block (chunked attention)"),
            block.quantum_scratch(b, t, ctx, chunked, eb, 0, cpu),
            measured(|| run(&f32_hidden, chunked)),
        );

        // Rung 4: a block streamed from the checkpoint reads its stored bytes INSIDE its quantum,
        // and is priced with exactly those bytes.
        let stream = crate::block_stream::AnimaBlockStream::new(
            WeightsSource::File(path.clone()),
            "net",
            cfg,
        );
        let mut view = stream.open().unwrap();
        let (streamed, stored) = stream.materialize_counted(&mut view, 0).unwrap();
        let on_disk: u64 = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&path)
            .unwrap()
            .iter()
            .filter(|h| h.name.starts_with("net.blocks.0."))
            .map(|h| h.data_bytes)
            .sum();
        assert_eq!(
            stored, on_disk,
            "a streamed block is priced at its own stored bytes"
        );
        prove(
            &tag("Anima DiT block (streamed)"),
            streamed.quantum_scratch(b, t, ctx, unbounded, eb, stored, cpu),
            measured(|| {
                streamed.forward(
                    &f32_hidden,
                    &encoder,
                    &embedded,
                    &temb,
                    (&rope.cos, &rope.sin),
                    unbounded,
                )
            }),
        );

        // Output, over the f32 stream the last block returns.
        prove(
            &tag(DIT_OUTPUT_QUANTUM),
            dit.output_quantum_scratch(b, t, f32_hidden.item_size() as u64),
            measured(|| dit.output(&f32_hidden, &embedded, &temb, (1, 8, 8))),
        );
    }
}

/// Every tensor `AnimaTextConditioner::from_weights(.., "net.llm_adapter", cfg)` reads, at `dtype`.
fn conditioner_weights(cfg: &ConditionerConfig, dtype: Dtype) -> Weights {
    let d = cfg.model_dim as i32;
    let src = cfg.source_dim as i32;
    let hd = cfg.head_dim() as i32;
    let ff = (cfg.mlp_ratio * d as f32).round() as i32;
    let mut w = Weights::empty();
    let mut seed = 900u64;
    let mut put = |w: &mut Weights, key: String, shape: &[i32], norm: bool| {
        seed += 1;
        let value = if norm {
            Array::ones::<f32>(shape).unwrap()
        } else {
            let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
                .unwrap();
            mlx_rs::ops::multiply(&a, Array::from_slice(&[0.05f32], &[1])).unwrap()
        };
        w.insert(key, value.as_dtype(dtype).unwrap());
    };
    let p = "net.llm_adapter";
    put(
        &mut w,
        format!("{p}.embed.weight"),
        &[cfg.target_vocab_size as i32, d],
        false,
    );
    for i in 0..cfg.num_layers {
        let b = format!("{p}.blocks.{i}");
        for (attn, kv_in) in [("self_attn", d), ("cross_attn", src)] {
            put(&mut w, format!("{b}.{attn}.q_proj.weight"), &[d, d], false);
            put(
                &mut w,
                format!("{b}.{attn}.k_proj.weight"),
                &[d, kv_in],
                false,
            );
            put(
                &mut w,
                format!("{b}.{attn}.v_proj.weight"),
                &[d, kv_in],
                false,
            );
            put(&mut w, format!("{b}.{attn}.o_proj.weight"), &[d, d], false);
            put(&mut w, format!("{b}.{attn}.q_norm.weight"), &[hd], true);
            put(&mut w, format!("{b}.{attn}.k_norm.weight"), &[hd], true);
        }
        for norm in ["norm_self_attn", "norm_cross_attn", "norm_mlp"] {
            put(&mut w, format!("{b}.{norm}.weight"), &[d], true);
        }
        put(&mut w, format!("{b}.mlp.0.weight"), &[ff, d], false);
        put(&mut w, format!("{b}.mlp.0.bias"), &[ff], false);
        put(&mut w, format!("{b}.mlp.2.weight"), &[d, ff], false);
        put(&mut w, format!("{b}.mlp.2.bias"), &[d], false);
    }
    put(
        &mut w,
        format!("{p}.out_proj.weight"),
        &[cfg.target_dim as i32, d],
        false,
    );
    put(
        &mut w,
        format!("{p}.out_proj.bias"),
        &[cfg.target_dim as i32],
        false,
    );
    put(
        &mut w,
        format!("{p}.norm.weight"),
        &[cfg.target_dim as i32],
        true,
    );
    w
}

#[test]
fn conditioner_formulas_track_the_measured_peak() {
    if !in_cpu_child("conditioner_formulas_track_the_measured_peak") {
        return;
    }
    // Production feeds the conditioner the Qwen3 tower's f32 states and runs it at their dtype.
    let dtype = Dtype::Float32;
    let build = |heads: usize, storage: Dtype| {
        let cfg = ConditionerConfig {
            source_dim: 128,
            target_dim: 128,
            model_dim: 128,
            num_layers: 1,
            num_attention_heads: heads,
            mlp_ratio: 4.0,
            target_vocab_size: 64,
            min_sequence_length: 96,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        };
        let w = conditioner_weights(&cfg, storage);
        let cond = AnimaTextConditioner::from_weights(&w, "net.llm_adapter", cfg).unwrap();
        w.materialize().unwrap();
        cond
    };
    let (st, ss) = (32, 48);
    let ids: Vec<i32> = (0..st).map(|i| (i * 5 + 1) % 64).collect();
    let ids = Array::from_slice(&ids, &[1, st]);
    let source = random::normal::<f32>(
        &[1, ss, 128][..],
        None,
        None,
        Some(&random::key(3).unwrap()),
    )
    .unwrap();
    eval_all(&[&ids, &source]);

    // Entry, at the shipped model_dim : head_dim proportion (16 heads), where the RoPE tables are
    // the minor term they are in production.
    let wide = build(16, Dtype::Bfloat16);
    prove(
        CONDITIONER_EMBED_QUANTUM,
        wide.embed_body_bytes(1, st, ss, 4),
        measured_many(|| {
            let (h, (tc, ts), (sc, sn)) = wide.embed(&ids, ss, dtype)?;
            Ok(vec![h, tc, ts, sc, sn])
        }),
    );

    // Block (2 heads, so the CPU fallback's materialized scores stay small) and output, with f32
    // weights (the activation counts alone: nothing to promote) and at the shipped bf16 storage
    // (+ the f32 copies the f32 stream promotes the weights to).
    for storage in [Dtype::Float32, Dtype::Bfloat16] {
        let tag = |label: &str| format!("{label} [{storage:?} weights]");
        let cond = build(2, storage);
        let (hidden, (tc, ts), (sc, sn)) = cond.embed(&ids, ss, dtype).unwrap();
        eval_all(&[&hidden, &tc, &ts, &sc, &sn]);
        let block = &cond.blocks()[0];
        let out = block
            .forward(&hidden, &source, (&tc, &ts), (&sc, &sn))
            .unwrap();
        out.eval().unwrap();
        let eb = out.item_size() as u64;
        assert_eq!(eb, 4);
        prove(
            &tag(CONDITIONER_BLOCK_QUANTUM),
            block.quantum_scratch(1, st, ss, eb, proof_backend()),
            measured(|| block.forward(&out, &source, (&tc, &ts), (&sc, &sn))),
        );
        let weights = vec![1.25f32; st as usize];
        prove(
            &tag(CONDITIONER_OUTPUT_QUANTUM),
            cond.output_quantum_scratch(1, st, eb),
            measured(|| cond.output(&out, 1, st, Some(&weights), out.dtype())),
        );
    }
}

#[test]
fn text_encoder_formulas_track_the_measured_peak() {
    if !in_cpu_child("text_encoder_formulas_track_the_measured_peak") {
        return;
    }
    // Qwen3-0.6B's proportions in miniature: q (heads x head_dim) twice the hidden width, GQA 2:1,
    // MLP x3.
    let cfg = Qwen3Config {
        vocab_size: 64,
        hidden_size: 128,
        n_layers: 1,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 128,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    };
    let tower = |cfg: &Qwen3Config, storage: Dtype| {
        let mut w = tiny_qwen3_weights(cfg, 384);
        w.cast_all(storage).unwrap();
        let te = AnimaQwen3::from_weights_dtype(&w, "model", cfg, Dtype::Bfloat16).unwrap();
        w.materialize().unwrap();
        te
    };
    let s = 48;
    let ids: Vec<i32> = (0..s).map(|i| (i * 3 + 2) % 64).collect();
    let ids = Array::from_slice(&ids, &[1, s]);
    let mut mask = vec![1i32; s as usize];
    mask[s as usize - 1] = 0;
    let mask = Array::from_slice(&mask, &[1, s]);
    eval_all(&[&ids, &mask]);

    // The embed proof at the shipped hidden : head_dim proportion (1024 : 128), where the RoPE
    // tables are the minor term they are in production.
    let wide = tower(
        &Qwen3Config {
            n_heads: 8,
            n_kv_heads: 4,
            head_dim: 16,
            ..cfg
        },
        Dtype::Bfloat16,
    );
    prove(
        TE_EMBED_QUANTUM,
        wide.embed_body_bytes(1, s),
        measured_many(|| {
            let (h, c, sn, m) = wide.embed(&ids, &mask)?;
            Ok(vec![h, c, sn, m])
        }),
    );

    // Layers and the final norm at the shipped bf16 storage (+ the f32 copies the f32 stream
    // promotes the weights to). The f32 storage of the sc-10577 parity path is not an iOS load;
    // its counts over-price it.
    {
        let storage = Dtype::Bfloat16;
        let tag = |label: &str| format!("{label} [{storage:?} weights]");
        let te = tower(&cfg, storage);
        let (h, cos, sin, m) = te.embed(&ids, &mask).unwrap();
        eval_all(&[&h, &cos, &sin, &m]);
        let eb = crate::text_encoder::layer_element_bytes(&h, &cos);
        assert_eq!(eb, 4, "the f32 RoPE tables promote the layer to f32");
        let (layer, scratch) = te.layer(0, 1, s, eb, proof_backend());
        let out = layer.forward(&h, &cos, &sin, &m).unwrap();
        out.eval().unwrap();
        assert_eq!(out.dtype(), Dtype::Float32);
        prove(
            &tag("Anima Qwen3 layer (bf16 in)"),
            scratch,
            measured(|| layer.forward(&h, &cos, &sin, &m)),
        );
        prove(
            &tag(TE_LAYER_QUANTUM),
            scratch,
            measured(|| layer.forward(&out, &cos, &sin, &m)),
        );
        prove(
            &tag(TE_NORM_QUANTUM),
            te.norm_bytes(1, s, out.item_size() as u64),
            measured(|| te.final_norm(&out)),
        );
    }
}

// ---- Phase-boundary materializations -------------------------------------------------------------

/// The two evaluations `generate_impl` runs between phases — the conditioner inputs before the
/// Qwen3 TE is shed, and the latents before the DiT is shed — are each a guarded quantum: priced at
/// what is still lazy there (each prompt's mask-multiply; each latent's last CFG combine + sampler
/// update), checked before the graph runs, refused before it runs.
#[test]
fn phase_boundary_materializations_are_guarded() {
    if !in_cpu_child("phase_boundary_materializations_are_guarded") {
        return;
    }
    use crate::model::{
        materialize_conditioning_inputs, materialize_latents, CONDITIONING_INPUTS_QUANTUM,
        LATENTS_QUANTUM,
    };
    let lazy = |shape: &[i32]| {
        let a = Array::ones::<f32>(shape).unwrap();
        mlx_rs::ops::add(&a, &a).unwrap()
    };

    let latents = vec![lazy(&[1, 16, 1, 8, 8]), lazy(&[1, 16, 1, 8, 8])];
    let latent_scratch = 2 * mlx_gen::ios_quantum::sampler_step_bytes(16 * 8 * 8);
    take_refusal();
    let (refused, events) = observed(0, || materialize_latents(&latents));
    assert!(refused.is_err());
    let refusal = take_refusal().unwrap();
    assert_eq!(
        (refusal.label.as_str(), refusal.scratch_bytes),
        (LATENTS_QUANTUM, latent_scratch)
    );
    assert!(events.is_empty(), "{events:?}");
    let (done, events) = observed(AMPLE, || materialize_latents(&latents));
    done.unwrap();
    assert_checked_before_synchronized(&events);
    assert_eq!(scratch_of(&events, LATENTS_QUANTUM), [latent_scratch]);

    let inputs = |n: i32| crate::pipeline::AnimaCondInputs {
        source: lazy(&[1, n, 12]),
        t5_ids: Array::from_slice(&[1i32, 2, 3], &[1, 3]),
        t5_weights: vec![1.0; 3],
    };
    let (cond, uncond) = (inputs(5), inputs(7));
    let cond_scratch = mlx_gen::ios_quantum::elementwise_bytes(5 * 12, 3)
        + mlx_gen::ios_quantum::elementwise_bytes(7 * 12, 3);
    let (refused, events) = observed(0, || materialize_conditioning_inputs(&cond, Some(&uncond)));
    assert!(refused.is_err());
    let refusal = take_refusal().unwrap();
    assert_eq!(
        (refusal.label.as_str(), refusal.scratch_bytes),
        (CONDITIONING_INPUTS_QUANTUM, cond_scratch)
    );
    assert!(events.is_empty(), "{events:?}");
    let (done, events) = observed(AMPLE, || {
        materialize_conditioning_inputs(&cond, Some(&uncond))
    });
    done.unwrap();
    assert_checked_before_synchronized(&events);
    assert_eq!(
        scratch_of(&events, CONDITIONING_INPUTS_QUANTUM),
        [cond_scratch]
    );
}

// ---- Reachable-sizes prediction ------------------------------------------------------------------

/// The entitled iOS launch headroom the prediction compares against.
const ENTITLED_HEADROOM_MIB: u64 = 6142;

/// The real Anima `split_files/` snapshot, if this host has it (header reads only).
/// The caller-provisioned `split_files/` snapshot (inference never derives a cache path).
fn anima_snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("ANIMA_SPLIT_FILES").map(Into::into)
}

/// Stored bytes of a `[out, in]` `.weight` at a packed tier, by the convert-at-install rule
/// (`mlx_gen::quant::quantize_map`): `bits`-wide codes plus bf16 scales and biases per 64-group.
/// `None` for a tensor the converter leaves dense.
fn packed_bytes(
    header: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader,
    bits: u64,
) -> Option<u64> {
    let base = header.name.strip_suffix(".weight")?;
    if !crate::convert::is_dit_quant_target(base) || header.shape.len() != 2 {
        return None;
    }
    let (out, inp) = (header.shape[0] as u64, header.shape[1] as u64);
    (inp % 64 == 0 && inp >= 64).then(|| out * inp * bits / 8 + 2 * out * (inp / 64) * 2)
}

/// A HOST-DERIVED PREDICTION, not evidence: for `anima_turbo` at q4/q8/bf16 x 512/768/1024 and every
/// ladder rung, the resident bytes of each phase (from the real snapshot's safetensors headers; no
/// tensor data is read), the largest counted quantum scratch in that phase (every formula above,
/// evaluated at production geometry on modules built lazily from those headers), and whether
/// `resident + scratch + 512 MiB reserve <= 6142 MiB`. Load floors are reported as their own
/// phases. Run: `cargo test -p mlx-gen-anima --lib reachable_sizes_prediction -- --ignored --nocapture`.
#[test]
#[ignore = "prediction table over the real snapshot's headers; run with --ignored --nocapture"]
fn reachable_sizes_prediction() {
    if !in_cpu_child("reachable_sizes_prediction") {
        return;
    }
    let Some(root) = anima_snapshot() else {
        println!("no Anima snapshot on this host (set ANIMA_SPLIT_FILES); nothing to predict");
        return;
    };
    let headers = |p: &std::path::Path| {
        mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(p).unwrap()
    };
    let dit_path = root
        .join("diffusion_models")
        .join(crate::config::Variant::Turbo.dit_filename());
    let te_path = root.join("text_encoders/qwen_3_06b_base.safetensors");
    let vae_path = root.join("vae/qwen_image_vae.safetensors");
    let (dit_h, te_h, vae_h) = (headers(&dit_path), headers(&te_path), headers(&vae_path));
    let prefix = dit_h
        .iter()
        .find_map(|h| h.name.strip_suffix(".x_embedder.proj.1.weight"))
        .unwrap()
        .to_string();
    let blocks_prefix = format!("{prefix}.blocks.");
    const MIB: u64 = 1024 * 1024;
    let mib = |b: u64| b as f64 / MIB as f64;

    // Lazy modules over the real files: shapes only, nothing evaluated.
    let dit_w = Weights::from_file(&dit_path).unwrap();
    let dit = CosmosDiT::from_weights(&dit_w, &prefix, DitConfig::anima()).unwrap();
    let cond = AnimaTextConditioner::from_weights(
        &dit_w,
        &format!("{prefix}.llm_adapter"),
        ConditionerConfig::anima(),
    )
    .unwrap();
    let te = AnimaQwen3::from_weights(
        &Weights::from_file(&te_path).unwrap(),
        "model",
        &Qwen3Config::anima(),
    )
    .unwrap();
    let vae = {
        let src = Weights::from_file(&vae_path).unwrap();
        let mut diff = Weights::empty();
        let keys: Vec<String> = src.keys().map(String::from).collect();
        for k in &keys {
            diff.insert(
                crate::vae::convert_vae_key(k),
                src.require(k).unwrap().clone(),
            );
        }
        diff.cast_all(Dtype::Float32).unwrap();
        mlx_gen_qwen_image::loader::remap_vae_keys(&mut diff).unwrap();
        crate::vae::QwenVae::from_weights(&diff).unwrap()
    };

    let te_bytes: u64 = te_h.iter().map(|h| h.data_bytes).sum();
    let vae_stored: u64 = vae_h.iter().map(|h| h.data_bytes).sum();
    let vae_f32: u64 = vae_h
        .iter()
        .map(|h| h.shape.iter().product::<usize>() as u64 * 4)
        .sum();
    let vae_load_floor = vae_stored + vae_f32;
    let is_adapter = |n: &str| n.contains("llm_adapter.");
    let cond_bytes: u64 = dit_h
        .iter()
        .filter(|h| is_adapter(&h.name))
        .map(|h| h.data_bytes)
        .sum();

    println!(
        "Anima Turbo reachable-sizes PREDICTION (host-derived from {}; not evidence)",
        root.display()
    );
    println!(
        "files: DiT+conditioner {:.1} MiB (conditioner {:.1}), Qwen3 TE {:.1} MiB, VAE {:.1} MiB stored / {:.1} MiB as f32",
        mib(dit_h.iter().map(|h| h.data_bytes).sum()),
        mib(cond_bytes),
        mib(te_bytes),
        mib(vae_stored),
        mib(vae_f32)
    );
    println!(
        "budget: resident + scratch + {} MiB reserve <= {ENTITLED_HEADROOM_MIB} MiB",
        PROCESS_RESERVE_BYTES / MIB
    );
    println!("| tier | edge | rung | phase | resident MiB | largest scratch (quantum) MiB | total MiB | fits |");
    println!("|---|---:|---|---|---:|---|---:|---|");

    #[derive(Clone, Copy)]
    struct Rung {
        name: &'static str,
        staged: bool,
        tiled: bool,
        chunked: bool,
        windowed: bool,
    }
    let rungs = [
        Rung {
            name: "Resident",
            staged: false,
            tiled: false,
            chunked: false,
            windowed: false,
        },
        Rung {
            name: "StagedResidency",
            staged: true,
            tiled: false,
            chunked: false,
            windowed: false,
        },
        Rung {
            name: "BoundedDecode@192",
            staged: true,
            tiled: true,
            chunked: false,
            windowed: false,
        },
        Rung {
            name: "BoundedAttention@192",
            staged: true,
            tiled: true,
            chunked: true,
            windowed: false,
        },
        Rung {
            name: "BoundedTransformerResidency@192 (planned)",
            staged: true,
            tiled: true,
            chunked: true,
            windowed: true,
        },
    ];
    let tiling = mlx_gen::tiling::TilingConfig::spatial_only(192, 64);
    for (tier, bits) in [("q4", Some(4u64)), ("q8", Some(8)), ("bf16", None)] {
        let at_tier = |h: &mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader| {
            bits.and_then(|b| packed_bytes(h, b))
                .unwrap_or(h.data_bytes)
        };
        let dit_tensors: Vec<u64> = dit_h
            .iter()
            .filter(|h| !is_adapter(&h.name))
            .map(at_tier)
            .collect();
        let dit_bytes: u64 = dit_tensors.iter().sum();
        let nonblock_bytes: u64 = dit_h
            .iter()
            .filter(|h| !is_adapter(&h.name) && !h.name.starts_with(&blocks_prefix))
            .map(at_tier)
            .sum();
        let block_bytes = |i: usize| -> u64 {
            let p = format!("{blocks_prefix}{i}.");
            dit_h
                .iter()
                .filter(|h| h.name.starts_with(&p))
                .map(at_tier)
                .sum()
        };
        let largest_block = (0..28).map(block_bytes).max().unwrap();
        for edge in [512u64, 768, 1024] {
            let lat = edge / 8;
            let tokens = (lat / 2) * (lat / 2);
            let latent = [1u64, 16, 1, lat, lat];
            let latent_elems = 16 * lat * lat;
            let carried = 512 * 1024 * 4 + latent_elems * 4;
            for rung in rungs {
                let plan = if rung.chunked {
                    AttentionPlan::budgeted(AttentionBudget::CONSTRAINED)
                } else {
                    AttentionPlan::UNBOUNDED
                };
                let dit_res = if rung.windowed {
                    nonblock_bytes
                } else {
                    dit_bytes
                };
                let heavy = dit_res + cond_bytes + vae_f32;
                // Carried through the decode: the latent, the mid-block output the tail reads (384
                // channels at latent extent, f32) and, when tiled, the full-size f32 RGB slab and
                // output accumulators.
                let decode_carried = latent_elems * 4
                    + 384 * lat * lat * 4
                    + if rung.tiled {
                        2 * 3 * edge * edge * 4
                    } else {
                        0
                    };
                let (text_res, denoise_res, decode_res) = if rung.staged {
                    (te_bytes, heavy + carried, vae_f32 + decode_carried)
                } else {
                    let all = te_bytes + heavy + carried;
                    (all, all, all)
                };
                let max_of =
                    |v: Vec<(&'static str, u64)>| v.into_iter().max_by_key(|x| x.1).unwrap();
                // Text phase: the Qwen3 tower over 512 tokens, then the conditioning inputs.
                let s = 512i32;
                let text = max_of(vec![
                    (TE_EMBED_QUANTUM, te.embed_body_bytes(1, s)),
                    (TE_LAYER_QUANTUM, te.layer(0, 1, s, 4, Backend::Metal).1),
                    (TE_NORM_QUANTUM, te.norm_bytes(1, s, 4)),
                    (
                        crate::model::CONDITIONING_INPUTS_QUANTUM,
                        mlx_gen::ios_quantum::elementwise_bytes(512 * 1024, 3),
                    ),
                ]);
                // Denoise: the conditioner (512 T5 x 512 Qwen3 tokens) and every DiT quantum.
                let block = &dit.blocks()[0];
                let weight = if rung.windowed { largest_block } else { 0 };
                let denoise = max_of(vec![
                    (
                        CONDITIONER_EMBED_QUANTUM,
                        cond.embed_body_bytes(1, 512, 512, 4)
                            + mlx_gen::ios_quantum::elementwise_bytes(512 * 1024, 3),
                    ),
                    (
                        CONDITIONER_BLOCK_QUANTUM,
                        cond.blocks()[0].quantum_scratch(1, 512, 512, 4, Backend::Metal),
                    ),
                    (
                        CONDITIONER_OUTPUT_QUANTUM,
                        cond.output_quantum_scratch(1, 512, 4),
                    ),
                    (
                        DIT_EMBED_QUANTUM,
                        dit.embed_body_bytes(&[1, 16, 1, lat as i32, lat as i32], 2)
                            + mlx_gen::ios_quantum::sampler_step_bytes(latent_elems),
                    ),
                    (
                        DIT_BLOCK_QUANTUM,
                        // The snapshot is bf16: a packed tier's projections are consumed in
                        // place, so its blocks carry no promoted f32 copies.
                        block.quantum_scratch(
                            1,
                            tokens as i32,
                            512,
                            plan,
                            4,
                            weight,
                            Backend::Metal,
                        ) - if bits.is_some() {
                            block.promoted_weight_bytes(4)
                        } else {
                            0
                        },
                    ),
                    (
                        DIT_OUTPUT_QUANTUM,
                        dit.output_quantum_scratch(1, tokens as i32, 4),
                    ),
                    (
                        crate::model::LATENTS_QUANTUM,
                        mlx_gen::ios_quantum::sampler_step_bytes(latent_elems),
                    ),
                ]);
                // Decode: every VAE quantum (the mid-block attention query-chunked, as on iOS).
                let decode = max_of(vae.decode_quanta(
                    latent,
                    rung.tiled.then_some(&tiling),
                    crate::vae::QwenVae::ios_decode_attention_plan(),
                    4,
                ));
                // Load floors: the heavy DiT(+conditioner) file floor first (nothing resident yet:
                // staged loads it after the TE is shed, a resident load before the TE), then the
                // VAE's cast floor over the materialized DiT; a resident load reads the TE last.
                let dit_floor = if rung.windowed {
                    nonblock_bytes + cond_bytes
                } else {
                    dit_bytes + cond_bytes
                };
                let text_load_base = if rung.staged {
                    0
                } else {
                    dit_res + cond_bytes + vae_f32
                };
                let rows = [
                    (
                        "text load",
                        text_load_base,
                        (crate::loader::TEXT_LOAD_QUANTUM, te_bytes),
                    ),
                    ("text", text_res, text),
                    (
                        "heavy load",
                        0,
                        (crate::loader::HEAVY_LOAD_QUANTUM, dit_floor),
                    ),
                    (
                        "VAE load",
                        dit_res + cond_bytes,
                        (crate::loader::VAE_LOAD_QUANTUM, vae_load_floor),
                    ),
                    ("denoise", denoise_res, denoise),
                    ("decode", decode_res, decode),
                ];
                for (phase, resident, (label, scratch)) in rows {
                    let total = resident + scratch + PROCESS_RESERVE_BYTES;
                    println!(
                        "| {tier} | {edge} | {} | {phase} | {:.0} | {:.0} ({label}) | {:.0} | {} |",
                        rung.name,
                        mib(resident),
                        mib(scratch),
                        mib(total),
                        if total <= ENTITLED_HEADROOM_MIB * MIB {
                            "yes"
                        } else {
                            "NO"
                        }
                    );
                }
            }
        }
    }
}
