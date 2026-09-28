//! Real-weight acceptance for the per-token DeltaNet checkpoint ring (sc-24131) on Qwen3.8-27B
//! (`#[ignore]`d — needs the frozen snapshot and a GPU with ~54 GB free):
//!
//! * **AC1** — after one verify forward of `K + 1` tokens (the current token plus `K` drafts),
//!   rolling the engine's step cache back to *every* position `j in 0..=K + 1` inside that
//!   forward restores each linear layer's conv tail and SSM state **exactly** (f32 max abs error
//!   `0`, well inside the `1e-6` gate) under the oracles whose arithmetic is the verify
//!   forward's own: `j = 0` against the prompt-only state, `j = K + 1` against a ring-less
//!   reference cache fed the same `K + 1`-token forward, and every interior `j` against a
//!   second ring cache fed the same first `j` tokens and a *different* suffix (the slot for `j`
//!   depends on tokens `0..j` only — the causality a rollback relies on). The rings' device
//!   addresses never change across the verify step and the rollbacks.
//!
//!   The literal oracle — a fresh **token-at-a-time** decode of `prompt + j` tokens — is
//!   measured and printed too, together with a ring-free envelope (the reference cache after
//!   one `K + 1`-token forward vs after `K + 1` single-token forwards, no ring involved). On the
//!   BF16 checkpoint the two differ by the cuBLAS row-count effect S2 documented (a
//!   `M = K + 1`-row projection GEMM rounds its rows differently from `M = 1`, sc-24130's
//!   knife-edge root cause): a last-bit bf16 change in the conv tail (the raw `in_proj_qkv`
//!   rows, ULP 0.5 at their magnitude) and its propagation into the f32 SSM state. That
//!   envelope is a property of the projections, not of the rollback, so the gates on the
//!   literal oracle are that the ring's interior error stays in the envelope's class (within an
//!   order of magnitude) and that every restored slot is closer to the fresh decode of exactly
//!   `j` tokens than to `j ± 1` tokens; the `1e-6` gate itself is met exactly under the
//!   same-arithmetic oracles above and on the f32 tiny config
//!   (`models::qwen35::tests::verify_step_rollback_to_every_position_matches_a_fresh_decode`).
//! * **AC2 (engine)** — a short greedy MTP run at every `K in 1..=5` recovers every partial
//!   rejection with a direct rollback: `replay_forwards == 0`, exactly one target forward per
//!   verify step. (The 256-token bench rows are the sealed AC2 evidence; this is the in-test
//!   gate.)
//!
//! ```text
//! BONSAI_QWEN38_SNAPSHOT=E:\...\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
//!   DELTANET_RING_DRAFTS=3 \
//!   cargo test --release --features cuda -p candle-llm --test deltanet_ring -- --ignored --nocapture
//! ```
//!
//! `BONSAI_QWEN38_SNAPSHOT` is the manifest's own environment name for this model
//! (`release/real-weight-models.toml`); it is a passed-in path, never derived.
//! `DELTANET_RING_DRAFTS` is the comma list of draft widths the AC1 sweep covers (default `3`).

mod common;

use candle_core::{DType, Tensor};
use candle_llm::decode::{
    generate_speculative, CancelFlag, GenerationConfig, LogitsScope, MtpProposer,
    SpeculativePrompt, StepModel, StepRequest, StepTokens,
};
use candle_llm::device::select_device;
use candle_llm::models::{Qwen35Cache, Qwen35Model};
use candle_llm::primitives::DecodeCache;

const SNAPSHOT_VAR: &str = "BONSAI_QWEN38_SNAPSHOT";
const PROMPT: &str = "Write a detailed, multi-paragraph explanation of how transformer language \
    models generate text. Cover tokenization, self-attention, the key/value cache, and greedy \
    versus sampled decoding, and finish with the trade-offs of speculative decoding.";

fn host(x: &Tensor) -> Vec<f32> {
    x.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "state element counts differ");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Every linear layer's `(conv, ssm)` state on the host.
fn states(cache: &Qwen35Cache) -> Vec<(Vec<f32>, Vec<f32>)> {
    cache
        .recurrent_states()
        .into_iter()
        .map(|(conv, ssm)| {
            (
                conv.map(host).unwrap_or_default(),
                ssm.map(host).unwrap_or_default(),
            )
        })
        .collect()
}

/// The greedy continuation of `prompt` through the step driver (the tokens the verify forward
/// is fed: the real continuation, so the accepted/rejected split is the model's own).
fn continuation(model: &Qwen35Model, prompt: &[i32], n: usize) -> Vec<i32> {
    let config = GenerationConfig {
        max_new_tokens: n,
        seed: Some(0),
        ..Default::default()
    };
    candle_llm::decode::generate_step(
        model,
        prompt,
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap()
    .0
    .tokens
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn ac1_verify_step_rollback_to_every_position_matches_a_fresh_decode() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (model, mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let p = prompt.len();
    let widths: Vec<usize> = std::env::var("DELTANET_RING_DRAFTS")
        .unwrap_or_else(|_| "3".to_string())
        .split(',')
        .map(|s| s.trim().parse().expect("draft width"))
        .collect();
    let tokens = continuation(&model, &prompt, 8);
    eprintln!("[ac1] prompt {p} tokens; continuation {tokens:?}");

    // A different continuation for the "same prefix, other suffix" oracle: shifted by one so no
    // position repeats its token.
    let other: Vec<i32> = tokens[1..].to_vec();

    // The prompt-only state (j = 0) and the token-at-a-time states (the literal oracle) on the
    // reference cache, computed once.
    let mut single = model.new_cache();
    model
        .forward_step(&mut single, StepRequest::last(&prompt))
        .unwrap();
    device.synchronize().unwrap();
    let mut single_states = vec![states(&single)];
    for t in &tokens[..7] {
        model
            .forward_step(&mut single, StepRequest::last(&[*t]))
            .unwrap();
        device.synchronize().unwrap();
        single_states.push(states(&single));
    }
    let err = |a: &[(Vec<f32>, Vec<f32>)], b: &[(Vec<f32>, Vec<f32>)]| -> (f32, f32) {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .fold((0.0f32, 0.0f32), |(c, s), ((ac, as_), (bc, bs))| {
                (c.max(max_abs_diff(ac, bc)), s.max(max_abs_diff(as_, bs)))
            })
    };
    let verify_forward = |cache: &mut Qwen35Cache, toks: &[i32]| {
        model
            .forward_step(
                cache,
                StepRequest {
                    tokens: StepTokens::Host(toks),
                    scope: LogitsScope::All,
                    want_hidden: false,
                },
            )
            .unwrap();
        device.synchronize().unwrap();
    };

    let mut worst_exact = 0.0f32;
    for &k in &widths {
        let verify = &tokens[..k + 1];
        // The ring-free envelope: the reference cache after one `K + 1`-token forward vs after
        // `K + 1` single-token forwards. No ring is involved; this is the projections' row-count
        // effect alone.
        let mut whole = model.new_cache();
        model
            .forward_step(&mut whole, StepRequest::last(&prompt))
            .unwrap();
        verify_forward(&mut whole, verify);
        let whole_states = states(&whole);
        let (env_conv, env_ssm) = err(&whole_states, &single_states[k + 1]);
        eprintln!(
            "[ac1] K={k}: ring-free envelope (reference M={} forward vs {} single-token forwards): conv max|err| {env_conv:e} ssm max|err| {env_ssm:e}",
            k + 1,
            k + 1
        );

        let mut cache = model.new_cache_for(p + 16, k).unwrap();
        assert_eq!(cache.max_checkpoints(), k + 1);
        let addresses = cache.recurrent_ring_addresses().unwrap();
        assert!(!addresses.is_empty(), "the step cache has rings");
        let ring_bytes = cache.recurrent_bytes();
        model
            .forward_step(&mut cache, StepRequest::last(&prompt))
            .unwrap();
        verify_forward(&mut cache, verify);
        assert_eq!(cache.len(), (p + k + 1) as i32);
        assert_eq!(cache.recurrent_ring_addresses().unwrap(), addresses);
        assert_eq!(
            cache.checkpoint_offsets(),
            (p as i32..(p + k + 1) as i32).collect::<Vec<_>>(),
            "K={k}: the step start and every verify position are restorable"
        );
        assert_eq!(cache.recurrent_bytes(), ring_bytes, "nothing grew");
        for j in (0..=k + 1).rev() {
            cache.rollback_to((p + j) as i32).unwrap();
            assert_eq!(cache.len(), (p + j) as i32);
            assert_eq!(cache.recurrent_ring_addresses().unwrap(), addresses);
            let got = states(&cache);
            // The same-arithmetic oracle for this `j`.
            let (oracle, oracle_name) = if j == 0 {
                (single_states[0].clone(), "prompt-only state".to_string())
            } else if j == k + 1 {
                (
                    whole_states.clone(),
                    format!("reference cache, same M={} forward", k + 1),
                )
            } else {
                let mut twin = model.new_cache_for(p + 16, k).unwrap();
                model
                    .forward_step(&mut twin, StepRequest::last(&prompt))
                    .unwrap();
                let mut toks: Vec<i32> = verify[..j].to_vec();
                toks.extend_from_slice(&other[j..k + 1]);
                assert_ne!(toks, verify);
                verify_forward(&mut twin, &toks);
                twin.rollback_to((p + j) as i32).unwrap();
                (
                    states(&twin),
                    format!("second ring, same first {j} tokens + other suffix"),
                )
            };
            let (exact_conv, exact_ssm) = err(&got, &oracle);
            let (lit_conv, lit_ssm) = err(&got, &single_states[j]);
            eprintln!(
                "[ac1] K={k} j={j}: rollback_to({}) vs {oracle_name}: conv max|err| {exact_conv:e} ssm max|err| {exact_ssm:e}; vs token-at-a-time decode: conv {lit_conv:e} ssm {lit_ssm:e} ({} linear layers)",
                p + j,
                got.len()
            );
            assert!(
                exact_conv <= 1e-6 && exact_ssm <= 1e-6,
                "K={k} j={j}: conv {exact_conv:e} ssm {exact_ssm:e} exceed 1e-6 under the same-arithmetic oracle"
            );
            // The literal oracle differs by the projections' row-count effect (a bf16 last-bit
            // change in the conv tail and its propagation into the f32 SSM state): the same
            // class as the ring-free envelope, never an order of magnitude beyond it — and the
            // restored slot is closer to the fresh decode of exactly `j` tokens than to `j - 1`
            // or `j + 1` tokens (a wrong slot would be closest to a neighbour).
            assert!(
                lit_conv <= 10.0 * env_conv.max(1e-6) && lit_ssm <= 10.0 * env_ssm.max(1e-6),
                "K={k} j={j}: the ring's interior error (conv {lit_conv:e} ssm {lit_ssm:e}) is not of the ring-free row-count class (conv {env_conv:e} ssm {env_ssm:e})"
            );
            for neighbour in [j.wrapping_sub(1), j + 1] {
                if let Some(other_state) = single_states.get(neighbour) {
                    let (_, off_ssm) = err(&got, other_state);
                    assert!(
                        lit_ssm < off_ssm,
                        "K={k} j={j}: closer to the fresh {neighbour}-token state ({off_ssm:e}) than to the {j}-token state ({lit_ssm:e})"
                    );
                }
            }
            worst_exact = worst_exact.max(exact_conv).max(exact_ssm);
        }
        // Older than the ring is the typed refusal, and the cache is untouched by it.
        cache.rollback_to((p + k + 1) as i32).unwrap_err();
        assert!(matches!(
            cache.rollback_to((p - 1) as i32),
            Err(candle_llm::error::Error::RollbackUnavailable { .. })
        ));
        assert_eq!(cache.len(), p as i32);
    }
    eprintln!(
        "[ac1] widths {widths:?}: worst same-arithmetic max|err| {worst_exact:e} (gate 1e-6)"
    );

    // AC2 (engine): every partial rejection is a direct rollback — no replay forward.
    let mtp = mtp.expect("the frozen Qwen3.8 snapshot carries a complete MTP head");
    let config = GenerationConfig {
        max_new_tokens: 48,
        seed: Some(0),
        ..Default::default()
    };
    for k in 1..=5usize {
        let mut proposer = MtpProposer::new(&mtp);
        let run = generate_speculative(
            &model,
            &mut proposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            k,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let r = run.record;
        eprintln!(
            "[ac2] mtp K={k}: verify_steps {} direct_rollbacks {} replay_forwards {} fwd/verify {:?} acceptance {:.3} cache live {} MiB checkpoints {} MiB",
            r.verify_steps,
            r.direct_rollbacks,
            r.replay_forwards,
            r.target_forwards_per_verify_step(),
            r.acceptance_rate().unwrap_or(0.0),
            run.memory.live_bytes >> 20,
            run.memory.checkpoint_bytes >> 20,
        );
        assert_eq!(r.replay_forwards, 0, "K={k}");
        assert_eq!(r.target_forwards_per_verify_step(), Some(1.0), "K={k}");
        assert_eq!(
            r.target_forwards,
            1 + r.verify_steps,
            "K={k}: prefill + one per verify"
        );
        assert_eq!(
            run.memory.checkpoint_bytes,
            model.recurrent_state_bytes(k + 2) - model.recurrent_state_bytes(1),
            "K={k}: the ring's checkpoint slots"
        );
    }
}
