//! Real-weight acceptance for the static KV cache (sc-24132) on Qwen3.8-27B (`#[ignore]`d —
//! needs the frozen snapshot and a GPU with ~54 GB free):
//!
//! * **AC1** — a 256-token greedy decode on the preallocated [`StaticKvCache`] (the `StepModel`
//!   driver's default cache) is token-identical to the `AttnKv` reference path — both the
//!   pre-epic `Decode` loop and the same driver with the growing cache selected — and the record
//!   names the cache that ran.
//! * **AC3 (real weights)** — the static buffers' CUDA device pointers are unchanged across the
//!   whole 256-token run and a rollback.
//!
//! ```text
//! BONSAI_QWEN38_SNAPSHOT=E:\...\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
//!   cargo test --release --features cuda -p candle-llm --test static_kv_parity -- --ignored --nocapture
//! ```
//!
//! `BONSAI_QWEN38_SNAPSHOT` is the manifest's own environment name for this model
//! (`release/real-weight-models.toml`); it is a passed-in path, never derived.
//!
//! [`StaticKvCache`]: candle_llm::primitives::StaticKvCache

mod common;

use candle_llm::decode::{
    generate_step, generate_with, CancelFlag, DecodePath, GenerationConfig, StepModel, StepRequest,
};
use candle_llm::device::select_device;
use candle_llm::primitives::{kv_materialize_count, DecodeCache, KvCacheKind};

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

fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn ac1_static_kv_greedy_fixture_is_token_identical_to_attn_kv() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (mut model, _mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let config = greedy(FIXTURE_TOKENS);

    // The pre-epic reference loop: AttnKv slots, repeat_kv + sdpa.
    let reference = generate_with(
        &model,
        &prompt,
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(reference.tokens.len(), FIXTURE_TOKENS);

    // The step driver on the static cache (its default).
    assert_eq!(model.step_kv_cache(), KvCacheKind::Static);
    let before = kv_materialize_count();
    let (fixed, record) = generate_step(
        &model,
        &prompt,
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(
        kv_materialize_count() - before,
        0,
        "the static run must issue no cat / repeat_kv copies at all"
    );
    assert_eq!(record.path, DecodePath::StepModel);
    assert_eq!(record.kv_cache, KvCacheKind::Static);
    assert_eq!(record.generated_tokens, FIXTURE_TOKENS as u64);
    assert_eq!(record.target_forwards, FIXTURE_TOKENS as u64);
    assert_eq!(
        fixed.tokens,
        reference.tokens,
        "static KV greedy decode diverged from the AttnKv reference loop at {:?}",
        first_divergence(&reference.tokens, &fixed.tokens)
    );

    // The same driver with the growing (AttnKv) cache selected: the in-driver parity oracle.
    model.set_step_kv_cache(KvCacheKind::Growing);
    let (growing, record) = generate_step(
        &model,
        &prompt,
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(record.kv_cache, KvCacheKind::Growing);
    assert_eq!(
        growing.tokens,
        reference.tokens,
        "growing StepModel decode diverged from the reference loop at {:?}",
        first_divergence(&reference.tokens, &growing.tokens)
    );
    eprintln!(
        "[ac1] {} tokens identical across reference / static / growing; static record {record:?}",
        reference.tokens.len()
    );
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn ac3_static_kv_device_pointers_are_stable_across_the_fixture_and_a_rollback() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (model, _mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);

    let capacity = prompt.len() + FIXTURE_TOKENS;
    let mut cache = model.new_static_cache(capacity, 4).unwrap();
    let addresses = cache.static_kv_addresses().unwrap();
    assert_eq!(
        addresses.len(),
        16,
        "Qwen3.8-27B has 16 full-attention layers"
    );
    let preallocated = cache.memory().live_bytes;
    assert_eq!(preallocated, model.static_kv_bytes(capacity));

    let mut logits = model
        .forward_step(&mut cache, StepRequest::last(&prompt))
        .unwrap()
        .logits;
    for step in 0..FIXTURE_TOKENS {
        let next = logits
            .argmax(candle_core::D::Minus1)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()[0] as i32;
        logits = model
            .forward_step(&mut cache, StepRequest::last(&[next]))
            .unwrap()
            .logits;
        if step % 32 == 0 {
            assert_eq!(
                cache.static_kv_addresses().unwrap(),
                addresses,
                "step {step}"
            );
        }
    }
    device.synchronize().unwrap();
    assert_eq!(cache.static_kv_addresses().unwrap(), addresses);
    let memory = cache.memory();
    assert_eq!(
        memory.live_bytes - preallocated,
        cache.recurrent_bytes() - memory.checkpoint_bytes,
        "the attention buffers never grew: past the preallocation only the live recurrent state"
    );
    let n = *cache.checkpoint_offsets().first().unwrap();
    cache.rollback_to(n).unwrap();
    assert_eq!(cache.len(), n);
    assert_eq!(cache.static_kv_addresses().unwrap(), addresses);
    model
        .forward_step(&mut cache, StepRequest::last(&[1]))
        .unwrap();
    assert_eq!(cache.static_kv_addresses().unwrap(), addresses);
    eprintln!(
        "[ac3] {} attention buffers pinned across {FIXTURE_TOKENS} steps and a rollback to {n}; \
         preallocated {preallocated} bytes",
        addresses.len()
    );
}
