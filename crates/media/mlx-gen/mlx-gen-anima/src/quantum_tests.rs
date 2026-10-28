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
#[ignore = "model inference; explicit model-proof workflow"]
fn dit_resident_blocks_are_guarded_quanta() {
    if !in_cpu_child("dit_resident_blocks_are_guarded_quanta") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::dit_resident_blocks_are_guarded_quanta");
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
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::dit_resident_blocks_are_guarded_quanta"
    );
}

/// The stage-4 golden hook shares `run_blocks`, so its partial stack is guarded per block too.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn dit_partial_stack_hook_is_guarded_per_block() {
    if !in_cpu_child("dit_partial_stack_hook_is_guarded_per_block") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::dit_partial_stack_hook_is_guarded_per_block");
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
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::dit_partial_stack_hook_is_guarded_per_block");
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
#[ignore = "model inference; explicit model-proof workflow"]
fn dit_windowed_blocks_are_guarded_quanta() {
    if !in_cpu_child("dit_windowed_blocks_are_guarded_quanta") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::dit_windowed_blocks_are_guarded_quanta");
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
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::dit_windowed_blocks_are_guarded_quanta"
    );
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
#[ignore = "model inference; explicit model-proof workflow"]
fn conditioner_blocks_are_guarded_quanta() {
    if !in_cpu_child("conditioner_blocks_are_guarded_quanta") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::conditioner_blocks_are_guarded_quanta");
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
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::conditioner_blocks_are_guarded_quanta"
    );
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
#[ignore = "model inference; explicit model-proof workflow"]
fn text_encoder_layers_are_guarded_quanta() {
    if !in_cpu_child("text_encoder_layers_are_guarded_quanta") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::text_encoder_layers_are_guarded_quanta");
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
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::text_encoder_layers_are_guarded_quanta"
    );
}

// ---- Scratch geometry ----------------------------------------------------------------------------

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn scratch_derives_from_loaded_tensors_and_grows_with_geometry() {
    if !in_cpu_child("scratch_derives_from_loaded_tensors_and_grows_with_geometry") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::scratch_derives_from_loaded_tensors_and_grows_with_geometry");
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
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::scratch_derives_from_loaded_tensors_and_grows_with_geometry");
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
// quantum body's own allocation. Each proof asserts formula >= measured (hard) and reports a ratio
// above MAX_RATIO (`mlx_gen::ios_quantum::assert_tracks_measured`: report-only).

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
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn dit_formulas_track_the_measured_peak() {
    if !in_cpu_child("dit_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::dit_formulas_track_the_measured_peak");
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

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!(
            "\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::dit_formulas_track_the_measured_peak"
        );
    }
    eprintln!(
        "\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::dit_formulas_track_the_measured_peak"
    );
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
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn conditioner_formulas_track_the_measured_peak() {
    if !in_cpu_child("conditioner_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::conditioner_formulas_track_the_measured_peak");
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

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::conditioner_formulas_track_the_measured_peak");
    }
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::conditioner_formulas_track_the_measured_peak");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow; expensive fixture capture; use the fixed ios-selected-captures dispatch profile"]
fn text_encoder_formulas_track_the_measured_peak() {
    if !in_cpu_child("text_encoder_formulas_track_the_measured_peak") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::text_encoder_formulas_track_the_measured_peak");
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

    if std::env::var_os("IOS_CAPTURE_IN_CHILD").is_some() {
        eprintln!("\nIOS_CAPTURE_BODY_COMPLETED quantum_tests::text_encoder_formulas_track_the_measured_peak");
    }
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::text_encoder_formulas_track_the_measured_peak");
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

// ---- iOS pricing: label attribution --------------------------------------------------------------

/// Every quantum label this crate declares (each `*_QUANTUM` string constant in its sources), the
/// shared solver-step label its sampler emits, and every decode label the Qwen-Image VAE it decodes
/// through declares, is attributed to a phase.
#[test]
fn every_quantum_label_maps_to_a_phase() {
    use crate::ios_pricing::ios_quantum_phase;
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut sources: Vec<std::path::PathBuf> = std::fs::read_dir(manifest.join("src"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    // The Qwen-Image VAE decode this route runs (`mlx-gen-qwen-image/src/vae`).
    let qwen_vae = manifest.join("../mlx-gen-qwen-image/src/vae");
    sources.extend(["mod.rs", "blocks.rs"].map(|f| qwen_vae.join(f)));
    // The curated solver's step boundary (`mlx_gen::sampler`).
    sources.push(manifest.join("../src/sampler.rs"));
    let mut labels = Vec::new();
    for path in &sources {
        let text = std::fs::read_to_string(path).unwrap();
        for line in text.lines().filter(|l| l.contains("const ")) {
            if let Some((_, rest)) = line.split_once("_QUANTUM: &str = \"") {
                labels.push(rest.split('"').next().unwrap().to_string());
            }
        }
    }
    // 15 Anima labels + 7 Qwen-Image VAE decode labels + the solver step.
    assert_eq!(labels.len(), 23, "{labels:?}");
    for label in &labels {
        assert!(
            ios_quantum_phase(label).is_some(),
            "unattributed quantum label {label:?}"
        );
    }
    assert_eq!(ios_quantum_phase("not an Anima quantum"), None);
}

// ---- iOS pricing: proof on the tiny fixture ------------------------------------------------------

/// The tiny configs the pricing proof drives the production path with: every width distinct where a
/// confusion would hide (Qwen3 hidden 12 ≠ heads·head_dim 8; DiT context 16 ≠ hidden 24), the real
/// tokenizers' vocabularies, and a RoPE range that covers a 512² request.
fn proof_configs() -> (DitConfig, ConditionerConfig, Qwen3Config) {
    let dit = DitConfig {
        in_channels: crate::config::VAE_CHANNELS,
        out_channels: crate::config::VAE_CHANNELS,
        num_attention_heads: 2,
        attention_head_dim: 12,
        num_layers: 2,
        mlp_ratio: 2.0,
        text_embed_dim: 16,
        adaln_lora_dim: 8,
        max_size: (4, 64, 64),
        patch_size: (1, 2, 2),
        rope_scale: (1.0, 4.0, 4.0),
        concat_padding_mask: true,
    };
    let conditioner = ConditionerConfig {
        source_dim: 12,
        target_dim: 16,
        model_dim: 16,
        num_layers: 2,
        num_attention_heads: 2,
        mlp_ratio: 4.0,
        target_vocab_size: 32128,
        min_sequence_length: 24,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
    };
    let qwen3 = Qwen3Config {
        vocab_size: 151_936,
        hidden_size: 12,
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    };
    (dit, conditioner, qwen3)
}

fn save(path: &std::path::Path, tensors: &std::collections::HashMap<String, Array>) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let refs: Vec<(&str, &Array)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    Array::save_safetensors(refs, None, path).unwrap();
}

/// A tiny Qwen-Image VAE in Anima's ORIGINAL on-disk naming (the layout of the real
/// `qwen_image_vae.safetensors`, at base width `c` instead of 96), stored bf16.
fn tiny_original_vae(c: i32) -> std::collections::HashMap<String, Array> {
    let (d1, d2, d4) = (c, 2 * c, 4 * c);
    let mut t = std::collections::HashMap::new();
    let mut seed = 3000u64;
    let mut put = |t: &mut std::collections::HashMap<String, Array>, k: String, shape: &[i32]| {
        seed += 1;
        let v = if k.ends_with("gamma") {
            Array::ones::<f32>(shape).unwrap()
        } else {
            let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
                .unwrap();
            mlx_rs::ops::multiply(&a, Array::from_slice(&[0.05f32], &[1])).unwrap()
        };
        t.insert(k, v.as_dtype(Dtype::Bfloat16).unwrap());
    };
    let mut resnet = |t: &mut std::collections::HashMap<String, Array>, p: &str, i: i32, o: i32| {
        put(t, format!("{p}.residual.0.gamma"), &[i, 1, 1, 1]);
        put(t, format!("{p}.residual.2.weight"), &[o, i, 3, 3, 3]);
        put(t, format!("{p}.residual.2.bias"), &[o]);
        put(t, format!("{p}.residual.3.gamma"), &[o, 1, 1, 1]);
        put(t, format!("{p}.residual.6.weight"), &[o, o, 3, 3, 3]);
        put(t, format!("{p}.residual.6.bias"), &[o]);
        if i != o {
            put(t, format!("{p}.shortcut.weight"), &[o, i, 1, 1, 1]);
            put(t, format!("{p}.shortcut.bias"), &[o]);
        }
    };
    for side in ["encoder", "decoder"] {
        resnet(&mut t, &format!("{side}.middle.0"), d4, d4);
        resnet(&mut t, &format!("{side}.middle.2"), d4, d4);
    }
    // Encoder: conv1, downsamples 0..10, head.
    resnet(&mut t, "encoder.downsamples.0", d1, d1);
    resnet(&mut t, "encoder.downsamples.1", d1, d1);
    resnet(&mut t, "encoder.downsamples.3", d1, d2);
    resnet(&mut t, "encoder.downsamples.4", d2, d2);
    resnet(&mut t, "encoder.downsamples.6", d2, d4);
    resnet(&mut t, "encoder.downsamples.7", d4, d4);
    resnet(&mut t, "encoder.downsamples.9", d4, d4);
    resnet(&mut t, "encoder.downsamples.10", d4, d4);
    // Decoder: upsamples 0..14 (resnets; 3, 7, 11 resample).
    for (i, (a, b)) in [
        (0, (d4, d4)),
        (1, (d4, d4)),
        (2, (d4, d4)),
        (4, (d2, d4)),
        (5, (d4, d4)),
        (6, (d4, d4)),
        (8, (d2, d2)),
        (9, (d2, d2)),
        (10, (d2, d2)),
        (12, (d1, d1)),
        (13, (d1, d1)),
        (14, (d1, d1)),
    ] {
        resnet(&mut t, &format!("decoder.upsamples.{i}"), a, b);
    }
    // The non-resnet tensors (`resnet` holds the first writer).
    let mut seed = 4000u64;
    let mut put = |t: &mut std::collections::HashMap<String, Array>, k: &str, shape: &[i32]| {
        seed += 1;
        let v = if k.ends_with("gamma") {
            Array::ones::<f32>(shape).unwrap()
        } else {
            let a = random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
                .unwrap();
            mlx_rs::ops::multiply(&a, Array::from_slice(&[0.05f32], &[1])).unwrap()
        };
        t.insert(k.to_string(), v.as_dtype(Dtype::Bfloat16).unwrap());
    };
    for side in ["encoder", "decoder"] {
        put(&mut t, &format!("{side}.middle.1.norm.gamma"), &[d4, 1, 1]);
        put(
            &mut t,
            &format!("{side}.middle.1.to_qkv.weight"),
            &[3 * d4, d4, 1, 1],
        );
        put(&mut t, &format!("{side}.middle.1.to_qkv.bias"), &[3 * d4]);
        put(
            &mut t,
            &format!("{side}.middle.1.proj.weight"),
            &[d4, d4, 1, 1],
        );
        put(&mut t, &format!("{side}.middle.1.proj.bias"), &[d4]);
    }
    put(&mut t, "encoder.conv1.weight", &[d1, 3, 3, 3, 3]);
    put(&mut t, "encoder.conv1.bias", &[d1]);
    for (i, ch, time) in [(2, d1, false), (5, d2, true), (8, d4, true)] {
        put(
            &mut t,
            &format!("encoder.downsamples.{i}.resample.1.weight"),
            &[ch, ch, 3, 3],
        );
        put(
            &mut t,
            &format!("encoder.downsamples.{i}.resample.1.bias"),
            &[ch],
        );
        if time {
            put(
                &mut t,
                &format!("encoder.downsamples.{i}.time_conv.weight"),
                &[ch, ch, 3, 1, 1],
            );
            put(
                &mut t,
                &format!("encoder.downsamples.{i}.time_conv.bias"),
                &[ch],
            );
        }
    }
    put(&mut t, "encoder.head.0.gamma", &[d4, 1, 1, 1]);
    put(&mut t, "encoder.head.2.weight", &[32, d4, 3, 3, 3]);
    put(&mut t, "encoder.head.2.bias", &[32]);
    put(&mut t, "conv1.weight", &[32, 32, 1, 1, 1]);
    put(&mut t, "conv1.bias", &[32]);
    put(&mut t, "conv2.weight", &[16, 16, 1, 1, 1]);
    put(&mut t, "conv2.bias", &[16]);
    put(&mut t, "decoder.conv1.weight", &[d4, 16, 3, 3, 3]);
    put(&mut t, "decoder.conv1.bias", &[d4]);
    for (i, out, inp, time) in [(3, d2, d4, true), (7, d2, d4, true), (11, d1, d2, false)] {
        put(
            &mut t,
            &format!("decoder.upsamples.{i}.resample.1.weight"),
            &[out, inp, 3, 3],
        );
        put(
            &mut t,
            &format!("decoder.upsamples.{i}.resample.1.bias"),
            &[out],
        );
        if time {
            put(
                &mut t,
                &format!("decoder.upsamples.{i}.time_conv.weight"),
                &[2 * inp, inp, 3, 1, 1],
            );
            put(
                &mut t,
                &format!("decoder.upsamples.{i}.time_conv.bias"),
                &[2 * inp],
            );
        }
    }
    put(&mut t, "decoder.head.0.gamma", &[d1, 1, 1, 1]);
    put(&mut t, "decoder.head.2.weight", &[3, d1, 3, 3, 3]);
    put(&mut t, "decoder.head.2.bias", &[3]);
    t
}

/// Write a tiny `split_files/` snapshot of the shipped layout: the DiT + bundled conditioner, the
/// Qwen3 tower and the Qwen-Image VAE, each stored bf16 like the real files.
fn write_tiny_snapshot(
    root: &std::path::Path,
    configs: &(DitConfig, ConditionerConfig, Qwen3Config),
) {
    let (dit_cfg, cond_cfg, qwen_cfg) = configs;
    let scratch = tempfile::tempdir().unwrap();
    let dit_path = write_dit_checkpoint_as(scratch.path(), dit_cfg, Dtype::Bfloat16);
    let mut dit = Weights::from_file(&dit_path).unwrap().into_tensors();
    dit.extend(conditioner_weights(cond_cfg, Dtype::Bfloat16).into_tensors());
    save(
        &root
            .join("diffusion_models")
            .join(crate::config::Variant::Turbo.dit_filename()),
        &dit,
    );
    let te: std::collections::HashMap<String, Array> = tiny_qwen3_weights(qwen_cfg, 20)
        .into_tensors()
        .into_iter()
        .map(|(k, v)| (k, v.as_dtype(Dtype::Bfloat16).unwrap()))
        .collect();
    save(&root.join(crate::loader::TEXT_ENCODER_FILE), &te);
    save(&root.join(crate::loader::VAE_FILE), &tiny_original_vae(2));
}

/// PROOF that the iOS pricing prices what runs: the REAL production path — `load_turbo` on an iOS
/// load spec (sequential + deferred materialization) and `Generator::generate` with the request
/// memory the contract resolves — runs on a tiny bf16 snapshot under the CPU diagnostic fixture,
/// and for every quantum label it checks, the largest scratch it checked EQUALS the price's
/// (so every phase's largest counted quantum and its binding label match too), for the fully
/// optimized selection (the cheapest rung-4 selection) and for rung 0 Resident. Text-to-image is
/// Anima's only route (no encode mode exists).
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn ios_pricing_equals_the_checked_quanta_on_the_production_path() {
    if !in_cpu_child("ios_pricing_equals_the_checked_quanta_on_the_production_path") {
        eprintln!("\nMODEL_INFERENCE_PARENT_COMPLETED quantum_tests::ios_pricing_equals_the_checked_quanta_on_the_production_path");
        return;
    }
    use crate::ios_pricing::{
        fold, ios_quantum_phase, price_quanta, snapshot_facts, AnimaPricingWork,
    };
    use mlx_gen::gen_core::{MemoryNumericTier, MemoryStrategy};
    use mlx_gen::ios_pricing::IosPhase;
    use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy};

    let configs = proof_configs();
    crate::loader::FIXTURE_CONFIGS.with(|c| c.set(Some(configs)));
    let tmp = tempfile::tempdir().unwrap();
    write_tiny_snapshot(tmp.path(), &configs);
    // The iOS load (`mlx-gen-ios-catalog::anima::anima_turbo_load_spec`).
    let spec = LoadSpec::new(WeightsSource::Dir(tmp.path().to_path_buf()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_load_shape(LoadShape::DeferredMaterialization);
    let generator = crate::model::load_turbo(&spec).unwrap();
    let contract = crate::memory_strategy::memory_strategy_contract("anima_turbo", &spec).unwrap();
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let prompt = "a silver-haired girl";
    let tokenizers = crate::tokenizer::AnimaTokenizers::load().unwrap();
    let qwen_tokens = tokenizers
        .encode_qwen(&crate::prompt_weight::strip_prompt_weights(prompt))
        .unwrap()
        .0
        .shape()[1] as u32;
    let t5_tokens = tokenizers.encode_t5_weighted(prompt).unwrap().0.shape()[1] as u32;
    let facts = snapshot_facts(&spec, crate::config::Variant::Turbo).unwrap();

    for strategy in [
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategy::Resident,
    ] {
        let selection = contract
            .selection_space(strategy, tier, false)
            .unwrap()
            .pop()
            .unwrap();
        let execution = contract.generation_memory(&selection).unwrap_or_default();
        let req = GenerationRequest {
            prompt: prompt.into(),
            width: 512,
            height: 512,
            steps: Some(1),
            memory: Some(execution),
            ..Default::default()
        };
        let (result, events) = observed(AMPLE, || {
            generator
                .generate(&req, &mut |_| {})
                .map_err(|e| mlx_gen::Error::Msg(e.to_string()))
        });
        result.unwrap();
        assert_checked_before_synchronized_or_loads(&events);
        let work = AnimaPricingWork {
            width: 512,
            height: 512,
            count: 1,
            qwen_tokens,
            t5_tokens,
            backend: crate::transformer::Backend::current(),
        };
        let priced = price_quanta(&facts, work, &execution).unwrap();
        let largest = |pairs: &mut dyn Iterator<Item = (String, u64)>| {
            let mut by_label = std::collections::BTreeMap::<String, u64>::new();
            for (label, scratch) in pairs {
                let slot = by_label.entry(label).or_default();
                *slot = (*slot).max(scratch);
            }
            by_label
        };
        let checked = largest(&mut events.iter().filter_map(|e| match e {
            QuantumEvent::Checked { label, scratch } => Some((label.clone(), *scratch)),
            QuantumEvent::Synchronized { .. } => None,
        }));
        let prices = largest(
            &mut priced
                .iter()
                .map(|q| (q.label.to_string(), q.scratch_bytes)),
        );
        assert_eq!(checked, prices, "{strategy:?}: checked vs priced quanta");
        for label in checked.keys() {
            assert!(ios_quantum_phase(label).is_some(), "{label}");
        }
        // Each non-load phase's price is its largest checked quantum, bound by that quantum's label
        // (a phase's resident set is one regime); a load phase's is one of its checked loads.
        let pricing = fold(&priced);
        assert!(pricing.phase(IosPhase::Encode).is_none());
        for phase in IosPhase::ALL {
            let in_phase: Vec<(&String, &u64)> = checked
                .iter()
                .filter(|(l, _)| ios_quantum_phase(l) == Some(phase))
                .collect();
            let price = pricing.phase(phase);
            assert_eq!(
                price.is_some(),
                !in_phase.is_empty(),
                "{strategy:?} {phase:?}"
            );
            let Some(price) = price else { continue };
            if phase == IosPhase::Load {
                assert_eq!(checked[&price.binding], price.scratch_bytes);
                continue;
            }
            let (label, scratch) = in_phase.into_iter().max_by_key(|(_, s)| **s).unwrap();
            assert_eq!(
                (price.binding.as_str(), price.scratch_bytes),
                (label.as_str(), *scratch),
                "{strategy:?} {phase:?}"
            );
        }
        println!("{strategy:?}: {pricing:?}");
    }
    crate::loader::FIXTURE_CONFIGS.with(|c| c.set(None));
    eprintln!("\nMODEL_INFERENCE_BODY_COMPLETED quantum_tests::ios_pricing_equals_the_checked_quanta_on_the_production_path");
}

/// Load floors are single `Checked` events (`before_load*` checks without a synchronized graph);
/// every other quantum is checked before it synchronizes.
fn assert_checked_before_synchronized_or_loads(events: &[QuantumEvent]) {
    let loads = [
        crate::loader::TEXT_LOAD_QUANTUM,
        crate::loader::HEAVY_LOAD_QUANTUM,
        crate::loader::VAE_LOAD_QUANTUM,
    ];
    let rest: Vec<QuantumEvent> = events
        .iter()
        .filter(|e| !matches!(e, QuantumEvent::Checked { label, .. } if loads.contains(&label.as_str())))
        .cloned()
        .collect();
    assert_checked_before_synchronized(&rest);
}

// ---- Reachable-sizes prediction ------------------------------------------------------------------

/// The entitled iOS launch headroom the prediction compares against.
const ENTITLED_HEADROOM_MIB: u64 = 6142;

/// The caller-provisioned `split_files/` snapshot (inference never derives a cache path).
fn anima_snapshot() -> Option<std::path::PathBuf> {
    std::env::var_os("ANIMA_SPLIT_FILES").map(Into::into)
}

/// The DiT checkpoint's headers at a packed tier, by the convert-at-install rule
/// (`convert::quantize_anima_dit` → `mlx_gen::quant::quantize_map`): every group-aligned 2-D DiT
/// `.weight` (never the bundled conditioner) becomes `bits`-wide u32 codes plus bf16 scales and
/// biases per 64-group; every other tensor passes through.
fn packed_headers(
    headers: &[mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader],
    bits: usize,
) -> Vec<mlx_gen::gen_core::weightsmeta::SafetensorsTensorHeader> {
    use mlx_gen::gen_core::weightsmeta::{Dtype as Stored, SafetensorsTensorHeader as H};
    const GROUP: usize = 64;
    let mut out = Vec::with_capacity(headers.len());
    for h in headers {
        let base = h
            .name
            .strip_suffix(".weight")
            .filter(|b| crate::convert::is_dit_quant_target(b));
        match (base, h.shape.as_slice()) {
            (Some(base), &[o, i]) if i % GROUP == 0 && i >= GROUP => {
                out.push(H {
                    name: h.name.clone(),
                    dtype: Stored::U32,
                    shape: vec![o, i * bits / 32],
                    data_bytes: (o * i * bits / 8) as u64,
                });
                for part in ["scales", "biases"] {
                    out.push(H {
                        name: format!("{base}.{part}"),
                        dtype: Stored::BF16,
                        shape: vec![o, i / GROUP],
                        data_bytes: (o * (i / GROUP) * 2) as u64,
                    });
                }
            }
            _ => out.push(h.clone()),
        }
    }
    out
}

/// A HOST-DERIVED PREDICTION, not evidence: `anima_turbo` text-to-image (its only route) at
/// q4/q8/bf16 x 512/768/1024 on the iOS load, priced by `crate::ios_pricing` from the real
/// snapshot's headers (no tensor data is read; q4/q8 project the bf16 headers through the
/// convert-at-install rule). For each phase: resident + largest counted quantum + 512 MiB reserve
/// against 6142 MiB, at
///
/// * (A) the FULLY OPTIMIZED selection — the cheapest (last) selection of the deepest rung the iOS
///   load implements: `BoundedTransformerResidency` (rung 4, `Implemented` on a
///   DeferredMaterialization load; it engages rungs 1–3) — which must fit; and
/// * (B) rung 0 `Resident` (expected to refuse where the untiled decode or the whole warm set binds).
///
/// Run: `cargo test -p mlx-gen-anima --lib reachable_sizes_prediction -- --ignored --nocapture`.
#[test]
#[ignore = "prediction table over the real snapshot's headers; run with --ignored --nocapture"]
fn reachable_sizes_prediction() {
    if !in_cpu_child("reachable_sizes_prediction") {
        return;
    }
    use crate::ios_pricing::{fold, ios_pricing, price_quanta, snapshot_facts, AnimaPricingWork};
    use mlx_gen::gen_core::{
        MemoryGeometry, MemoryMode, MemoryNumericTier, MemoryStrategy, MemoryStrategySupport,
    };
    use mlx_gen::ios_pricing::IosPricingRequest;
    use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Quant};

    let Some(root) = anima_snapshot() else {
        println!("no Anima snapshot on this host (set ANIMA_SPLIT_FILES); nothing to predict");
        return;
    };
    const MIB: u64 = 1024 * 1024;
    let mib = |b: u64| b as f64 / MIB as f64;
    let budget = ENTITLED_HEADROOM_MIB * MIB;
    println!(
        "Anima Turbo reachable-sizes PREDICTION (host-derived from {}; not evidence)",
        root.display()
    );
    println!("mode: text (MemoryMode::TextToImage — Anima's only route; no encode phase)");
    println!(
        "budget: resident + scratch + {} MiB reserve <= {ENTITLED_HEADROOM_MIB} MiB",
        PROCESS_RESERVE_BYTES / MIB
    );
    let mut fully_optimized_misses = Vec::new();
    let mut header_printed = false;
    for (tier_name, quant) in [
        ("q4", Some(Quant::Q4)),
        ("q8", Some(Quant::Q8)),
        ("bf16", None),
    ] {
        // The iOS load, mirroring `mlx-gen-ios-catalog::anima::anima_turbo_load_spec` (that crate
        // depends on this one, so it cannot be a dependency here): `IOS_OFFLOAD_POLICY`
        // (sequential) + a deferred materialization, at the tier.
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        spec.quantize = quant;
        let contract =
            crate::memory_strategy::memory_strategy_contract("anima_turbo", &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(&MemoryStrategySupport::Implemented),
            "rung 4 is implemented on the iOS load"
        );
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let deepest = MemoryStrategy::ALL
            .into_iter()
            .filter(|s| {
                matches!(
                    contract.capability(*s).map(|c| &c.support),
                    Some(MemoryStrategySupport::Implemented)
                )
            })
            .max()
            .unwrap();
        assert_eq!(deepest, MemoryStrategy::BoundedTransformerResidency);
        let configs = [
            (
                "A",
                contract
                    .selection_space(deepest, tier, false)
                    .unwrap()
                    .pop()
                    .unwrap(),
            ),
            (
                "B",
                contract
                    .selection_space(MemoryStrategy::Resident, tier, false)
                    .unwrap()
                    .pop()
                    .unwrap(),
            ),
        ];
        let mut facts = snapshot_facts(&spec, crate::config::Variant::Turbo).unwrap();
        if let Some(q) = quant {
            let bits = if q == Quant::Q4 { 4 } else { 8 };
            facts.dit = packed_headers(&facts.dit, bits);
        }
        if !header_printed {
            header_printed = true;
            for (name, selection) in &configs {
                println!(
                    "({name}) {:?} {:?} -> execution {:?}",
                    selection.strategy,
                    selection.parameters,
                    contract.generation_memory(selection).unwrap_or_default()
                );
            }
            println!("| tier | edge | config | phase | resident MiB | largest quantum MiB (binding) | peak + reserve MiB | fits |");
            println!("|---|---:|---|---|---:|---|---:|---|");
        }
        for edge in [512u32, 768, 1024] {
            for (name, selection) in &configs {
                let execution = contract.generation_memory(selection).unwrap_or_default();
                let tokens = crate::tokenizer::MAX_LEN as u32;
                let work = AnimaPricingWork {
                    width: edge,
                    height: edge,
                    count: 1,
                    qwen_tokens: tokens,
                    t5_tokens: tokens,
                    backend: crate::transformer::Backend::Metal,
                };
                let quanta = price_quanta(&facts, work, &execution).unwrap();
                let pricing = fold(&quanta);
                if quant.is_none() {
                    // The device entry point reads the same facts off the snapshot, header-only:
                    // nothing it builds is evaluated, so MLX allocates nothing.
                    // An actual valid Turbo text-to-image request at this edge: one serial output,
                    // the provider's default FLOW sampler/scheduler/steps.
                    let generation = mlx_gen::GenerationRequest {
                        width: edge,
                        height: edge,
                        count: 1,
                        ..Default::default()
                    };
                    // Independent expected envelope from the sampler's source semantics, not the
                    // entry's output: each DiT/sampler-body quantum is owned once per possible solver
                    // callback family over the f32 (w/8)x(h/8)xC latent, holding the callback's outer
                    // owners, the caller's f32 latent handle (except inside the step itself, which
                    // owns that state), and the host sigma schedule; the sampler step additionally
                    // owns a solve-complete window holding only the schedule. Other quanta are their
                    // own windows, and Encode never runs.
                    let sampling = crate::sampling::resolve(&generation).unwrap();
                    let latent = u64::from(edge / crate::config::VAE_COMPRESSION).pow(2)
                        * crate::config::VAE_CHANNELS as u64;
                    let callbacks = mlx_gen::sampler::solver_memory::solver_callback_windows(
                        sampling.solver(),
                        sampling.convention(),
                        latent,
                        4,
                        4,
                    )
                    .unwrap();
                    let schedule = sampling.host_schedule_bytes().unwrap();
                    let class = |window: &mlx_gen::ios_pricing::IosWindowPrice, family: &str| {
                        format!(
                            "{}/{}/{family}",
                            window.class.as_deref().unwrap(),
                            sampling.allocation_class()
                        )
                    };
                    let mut expected = mlx_gen::ios_pricing::IosPricing::default();
                    for quantum in &quanta {
                        let step = quantum.label == mlx_gen::sampler::SAMPLER_STEP_QUANTUM;
                        let body = step
                            || [
                                crate::transformer::DIT_EMBED_QUANTUM,
                                crate::transformer::DIT_BLOCK_QUANTUM,
                                crate::transformer::DIT_OUTPUT_QUANTUM,
                            ]
                            .contains(&quantum.label);
                        if !body {
                            expected.take_window(quantum.window.clone());
                            continue;
                        }
                        for callback in &callbacks {
                            let mut window = quantum.window.clone();
                            window.class = Some(class(&window, &callback.family()));
                            window.held_bytes += callback.callback_outer_bytes
                                + if step { 0 } else { callback.state_bytes }
                                + schedule;
                            expected.take_window(window);
                        }
                        if step {
                            let mut window = quantum.window.clone();
                            window.class = Some(class(&window, "solve-complete"));
                            window.held_bytes += schedule;
                            expected.take_window(window);
                        }
                    }
                    expected.phase_not_run(
                        mlx_gen::ios_pricing::IosPhase::Encode,
                        "Anima serves text-to-image only",
                    );
                    assert!(
                        !callbacks.is_empty(),
                        "the Turbo FLOW sampler owns callbacks"
                    );
                    let before = mlx_rs::memory::get_active_memory();
                    let entry = ios_pricing(&IosPricingRequest {
                        provider: "anima_turbo",
                        spec: &spec,
                        mode: &MemoryMode::TextToImage,
                        geometry: MemoryGeometry {
                            width: edge,
                            height: edge,
                            batch: 1,
                            frames: 1,
                            reference_count: 0,
                        },
                        selection,
                        execution: &execution,
                        generation: Some(&generation),
                        transform: None,
                        depth: None,
                        token_window: 0,
                    })
                    .unwrap();
                    assert_eq!(mlx_rs::memory::get_active_memory(), before, "header-only");
                    assert_eq!(entry, expected);
                    assert_ne!(
                        entry, pricing,
                        "sampler callbacks are not the bare quantum fold"
                    );
                }
                for price in &pricing.phases {
                    let total = price.peak_bytes() + PROCESS_RESERVE_BYTES;
                    let fits = total <= budget;
                    if *name == "A" && !fits {
                        fully_optimized_misses
                            .push(format!("{tier_name} {edge} {:?}", price.phase));
                    }
                    println!(
                        "| {tier_name} | {edge} | {name} | {} | {:.0} | {:.0} ({}) | {:.0} | {} |",
                        price.phase.as_key(),
                        mib(price.resident_bytes),
                        mib(price.scratch_bytes),
                        price.binding,
                        mib(total),
                        if fits { "yes" } else { "NO" }
                    );
                }
            }
        }
    }
    assert!(
        fully_optimized_misses.is_empty(),
        "the fully optimized selection must fit: {fully_optimized_misses:?}"
    );
}
