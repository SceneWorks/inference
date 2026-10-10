//! Real-weight acceptance for the sc-24129 seams on Qwen3.8-27B (`#[ignore]`d — needs the frozen
//! snapshot and a GPU with ~54 GB free):
//!
//! * **AC1** — a 256-token greedy decode driven through [`StepModel`] is token-identical to the
//!   pre-change `Decode` loop on the same prompt.
//! * **AC2** — `Qwen35Cache::rollback_to(n)` after decoding past `n` and re-decoding token `n`
//!   yields the same logits as a fresh decode to `n + 1`, on real weights.
//!
//! ```text
//! BONSAI_QWEN38_SNAPSHOT=E:\...\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
//!   cargo test --release --features cuda -p candle-llm --test decode_step_parity -- --ignored --nocapture
//! ```
//!
//! `BONSAI_QWEN38_SNAPSHOT` is the manifest's own environment name for this model
//! (`release/real-weight-models.toml`); it is a passed-in path, never derived.

mod common;

use candle_core::{DType, Tensor};
use candle_llm::decode::{
    generate_step, generate_with, CancelFlag, DecodePath, GenerationConfig, StepModel, StepRequest,
};
use candle_llm::device::select_device;
use candle_llm::primitives::{input_ids, DecodeCache};

const SNAPSHOT_VAR: &str = "BONSAI_QWEN38_SNAPSHOT";
const FIXTURE_TOKENS: usize = 256;
const PROMPT: &str = "Write a detailed, multi-paragraph explanation of how transformer language \
    models generate text. Cover tokenization, self-attention, the key/value cache, and greedy \
    versus sampled decoding, and finish with the trade-offs of speculative decoding.";

fn greedy(max_new_tokens: usize) -> GenerationConfig {
    let mut config = GenerationConfig {
        max_new_tokens,
        seed: Some(0),
        stop_tokens: Vec::new(),
        ..Default::default()
    };
    config.sampling.temperature = 0.0;
    config
}

fn host(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn ac1_step_model_greedy_fixture_is_token_identical_to_reference() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (model, _mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let config = greedy(FIXTURE_TOKENS);

    let reference = generate_with(
        &model,
        &prompt,
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    let (step, record) = generate_step(
        &model,
        &prompt,
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(reference.tokens.len(), FIXTURE_TOKENS);
    let divergence = reference
        .tokens
        .iter()
        .zip(&step.tokens)
        .position(|(a, b)| a != b);
    assert_eq!(
        step.tokens, reference.tokens,
        "StepModel greedy decode diverged from the reference path at {divergence:?}"
    );
    assert_eq!(record.path, DecodePath::StepModel);
    assert_eq!(record.generated_tokens, FIXTURE_TOKENS as u64);
    assert_eq!(record.target_forwards, FIXTURE_TOKENS as u64);
    assert_eq!(record.host_syncs, FIXTURE_TOKENS as u64);
    eprintln!(
        "[ac1] {} tokens identical; record {record:?}",
        reference.tokens.len()
    );
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn ac2_rollback_then_redecode_matches_fresh_decode_on_real_weights() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (model, _mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    // Continue the prompt with a short greedy run so the rollback window holds real text.
    let (continued, _) = generate_step(
        &model,
        &prompt,
        &greedy(12),
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    let mut tokens = prompt.clone();
    tokens.extend(&continued.tokens);
    let n = prompt.len() + 4;
    let m = tokens.len();

    // Fresh: prefill tokens[..n], decode tokens[n].
    let mut fresh = model.new_cache();
    model
        .decode_logits(&input_ids(&tokens[..n], &device).unwrap(), &mut fresh, 0)
        .unwrap();
    let fresh_logits = model
        .decode_logits(
            &input_ids(&tokens[n..n + 1], &device).unwrap(),
            &mut fresh,
            n as i32,
        )
        .unwrap();

    // Rolled back: prefill tokens[..n], decode tokens[n..m] one at a time, roll back to n.
    let mut cache = StepModel::new_cache(&model);
    cache.set_max_checkpoints(16).unwrap(); // keep every position of this run (the step seam keeps two)
    model
        .forward_step(&mut cache, StepRequest::last(&tokens[..n]))
        .unwrap();
    for t in &tokens[n..m] {
        model
            .forward_step(&mut cache, StepRequest::last(&[*t]))
            .unwrap();
    }
    assert_eq!(cache.len(), m as i32);
    let before = cache.memory();
    cache.rollback_to(n as i32).unwrap();
    assert_eq!(cache.len(), n as i32);
    let after = cache.memory();
    assert!(after.total_bytes() < before.total_bytes());
    let replayed = model
        .forward_step(&mut cache, StepRequest::last(&tokens[n..n + 1]))
        .unwrap()
        .logits;
    let (a, b) = (host(&fresh_logits), host(&replayed));
    let max_diff = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        max_diff, 0.0,
        "rollback_to({n}) then re-decode must equal a fresh decode to {n} + 1"
    );
    eprintln!("[ac2] rollback from {m} to {n}: logits identical (max diff {max_diff}); cache {before:?} -> {after:?}");
}
