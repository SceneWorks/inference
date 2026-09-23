//! Real-weight acceptance for the per-token DeltaNet checkpoint ring (sc-24131) on Qwen3.8-27B
//! (`#[ignore]`d — needs the frozen snapshot and a GPU with ~54 GB free):
//!
//! * **AC1** — after one verify forward of `K + 1` tokens (the current token plus `K` drafts),
//!   rolling the engine's step cache back to *every* position `j in 0..=K + 1` inside that
//!   forward leaves each linear layer's conv tail and SSM state within `1e-6` (f32 max abs error)
//!   of a fresh token-at-a-time decode of `prompt + j` tokens on the reference cache — and the
//!   rings' device addresses never change across the verify step and the rollbacks.
//! * **AC2 (engine)** — a short greedy MTP run at every `K in 1..=5` recovers every partial
//!   rejection with a direct rollback: `replay_fallbacks == 0`, exactly one target forward per
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

    let mut worst_conv = 0.0f32;
    let mut worst_ssm = 0.0f32;
    for &k in &widths {
        let mut cache = model.new_cache_for(p + 16, k).unwrap();
        assert_eq!(cache.max_checkpoints(), k + 1);
        let addresses = cache.recurrent_ring_addresses().unwrap();
        assert!(!addresses.is_empty(), "the step cache has rings");
        let ring_bytes = cache.recurrent_bytes();
        model
            .forward_step(&mut cache, StepRequest::last(&prompt))
            .unwrap();
        let verify = &tokens[..k + 1];
        model
            .forward_step(
                &mut cache,
                StepRequest {
                    tokens: StepTokens::Host(verify),
                    scope: LogitsScope::All,
                    want_hidden: false,
                },
            )
            .unwrap();
        device.synchronize().unwrap();
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
            let mut fresh = model.new_cache();
            model
                .forward_step(&mut fresh, StepRequest::last(&prompt))
                .unwrap();
            for t in &tokens[..j] {
                model
                    .forward_step(&mut fresh, StepRequest::last(&[*t]))
                    .unwrap();
            }
            device.synchronize().unwrap();
            let (got, want) = (states(&cache), states(&fresh));
            assert_eq!(got.len(), want.len());
            let (mut conv_err, mut ssm_err) = (0.0f32, 0.0f32);
            for ((gc, gs), (wc, ws)) in got.iter().zip(&want) {
                conv_err = conv_err.max(max_abs_diff(gc, wc));
                ssm_err = ssm_err.max(max_abs_diff(gs, ws));
            }
            eprintln!(
                "[ac1] K={k} j={j}: rollback_to({}) conv max|err| {conv_err:e} ssm max|err| {ssm_err:e} ({} linear layers)",
                p + j,
                got.len()
            );
            assert!(
                conv_err <= 1e-6 && ssm_err <= 1e-6,
                "K={k} j={j}: conv {conv_err:e} ssm {ssm_err:e} exceed 1e-6"
            );
            worst_conv = worst_conv.max(conv_err);
            worst_ssm = worst_ssm.max(ssm_err);
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
        "[ac1] widths {widths:?}: worst conv max|err| {worst_conv:e}, worst ssm max|err| {worst_ssm:e} (gate 1e-6)"
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
            "[ac2] mtp K={k}: verify_steps {} direct_rollbacks {} replay_fallbacks {} fwd/verify {:?} acceptance {:.3} cache live {} MiB checkpoints {} MiB",
            r.verify_steps,
            r.direct_rollbacks,
            r.replay_fallbacks,
            r.target_forwards_per_verify_step(),
            r.acceptance_rate().unwrap_or(0.0),
            run.memory.live_bytes >> 20,
            run.memory.checkpoint_bytes >> 20,
        );
        assert_eq!(r.replay_fallbacks, 0, "K={k}");
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
