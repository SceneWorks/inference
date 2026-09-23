//! Real-weight acceptance and evidence for the CUDA-graph runner (sc-24134) on Qwen3.8-27B
//! (`#[ignore]`d — needs the frozen snapshot and a GPU with ~54 GB free):
//!
//! * **AC1** — a 256-token greedy decode with the graph runner **on** is token-identical to the
//!   same request with it **off**, for speculation off (the step driver on the static KV cache)
//!   and for MTP `K = 3` (the unified engine). On this candle revision the 27B hybrid falls
//!   back eager with the named reason `deltanet_state_unstable` (the S1 cache replaces its
//!   DeltaNet state per step), so identity holds by construction; the records say so.
//! * **Census** — what a real 27B decode step and a `K = 3` verify step are made of when
//!   recorded as a graph: kernel launches, host uploads (candle's per-op layout metadata),
//!   allocations. The number behind the story's finding and the upper bound of what a full
//!   graph could save.
//!
//! ```text
//! BONSAI_QWEN38_SNAPSHOT=E:\...\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
//!   cargo test --release --features cuda -p candle-llm --test cuda_graphs -- --ignored --nocapture
//! ```
#![cfg(feature = "cuda")]

mod common;

use candle_llm::decode::graph::{census_step, cuda_graphs_policy_guard};
use candle_llm::decode::{
    generate_speculative, generate_step, CancelFlag, GenerationConfig, GraphRunner, LogitsScope,
    MtpProposer, SpeculativePrompt, StepModel, StepRequest, StepTokens,
};
use candle_llm::device::select_device;
use candle_llm::primitives::{input_ids, DecodeCache, KvCacheKind};

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

/// The device a graphs-on deployment gets: the switch on when `select_device` runs puts the
/// model on its own stream (the legacy default cannot be captured). Both rows of each AC1
/// comparison run on it, so graphs on vs off differ only in the runner.
fn graphs_on_device() -> candle_core::Device {
    let _guard = cuda_graphs_policy_guard(Some(true));
    select_device().unwrap()
}

fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn ac1_graphs_on_is_token_identical_to_eager_for_spec_off_and_mtp_k3() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = graphs_on_device();
    let (model, mtp) = common::qwen35::load(&snapshot, &device);
    let mtp = mtp.expect("the snapshot carries a complete MTP head");
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let config = greedy(FIXTURE_TOKENS);
    assert_eq!(model.step_kv_cache(), KvCacheKind::Static);

    // ---- Speculation off: the step driver, graphs off then on. ----
    let (eager, eager_record) = {
        let _guard = cuda_graphs_policy_guard(Some(false));
        generate_step(
            &model,
            &prompt,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap()
    };
    assert_eq!(eager.tokens.len(), FIXTURE_TOKENS);
    assert_eq!(eager_record.cuda_graphs.label(), "none");
    let (graphs, graphs_record) = {
        let _guard = cuda_graphs_policy_guard(Some(true));
        let runner = GraphRunner::new(&model);
        generate_step(
            &runner,
            &prompt,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap()
    };
    eprintln!(
        "[cuda_graphs] spec off, graphs on: {} ({} tokens)",
        graphs_record.cuda_graphs.describe(),
        graphs.tokens.len()
    );
    assert_eq!(
        graphs.tokens,
        eager.tokens,
        "graphs on diverged from eager at {:?}",
        first_divergence(&eager.tokens, &graphs.tokens)
    );
    assert_eq!(
        graphs_record.cuda_graphs.eager,
        eager_record.target_forwards
    );
    assert_eq!(
        graphs_record.cuda_graphs.fallback_reason,
        Some("deltanet_state_unstable")
    );
    assert_eq!(graphs_record.kv_cache, KvCacheKind::Static);

    // ---- MTP K = 3: the engine, graphs off then on. ----
    let run = |on: bool| {
        let _guard = cuda_graphs_policy_guard(Some(on));
        let runner = GraphRunner::new(&model);
        let stepper: &dyn StepModel<Cache = candle_llm::models::Qwen35Cache> =
            if on { &runner } else { &model };
        let mut proposer = MtpProposer::new(&mtp);
        generate_speculative(
            stepper,
            &mut proposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap()
    };
    let eager_mtp = run(false);
    let graphs_mtp = run(true);
    eprintln!(
        "[cuda_graphs] MTP K=3, graphs on: {} (accepted {} / proposed {}, verify steps {})",
        graphs_mtp.record.cuda_graphs.describe(),
        graphs_mtp.stats.accepted,
        graphs_mtp.stats.proposed,
        graphs_mtp.stats.verify_steps
    );
    assert_eq!(eager_mtp.output.tokens.len(), FIXTURE_TOKENS);
    assert_eq!(
        graphs_mtp.output.tokens,
        eager_mtp.output.tokens,
        "MTP K=3 with graphs on diverged at {:?}",
        first_divergence(&eager_mtp.output.tokens, &graphs_mtp.output.tokens)
    );
    // (MTP vs the token-at-a-time driver is S2's gate, with its enumerated bf16 knife-edge
    // exceptions; this story's claim is graphs on vs off on the same path, asserted above.)
    assert_eq!(graphs_mtp.stats.accepted, eager_mtp.stats.accepted);
    assert_eq!(
        graphs_mtp.record.cuda_graphs.fallback_reason,
        Some("deltanet_state_unstable")
    );
    assert_eq!(graphs_mtp.record.host_syncs_per_verify_step(), Some(1.0));
}

/// The census of a real 27B decode step (1 token) and verify step (4 tokens, all-position
/// logits, hidden states) recorded as CUDA graphs — nothing is launched. Run last: see
/// [`census_step`].
#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn qwen38_27b_step_census() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = graphs_on_device();
    let (model, _mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let mut cache = model.new_cache_for(prompt.len() + 64, 4).unwrap();
    // Prefill and a few eager steps first, so every kernel is compiled before any recording.
    model
        .forward_step(&mut cache, StepRequest::last(&prompt))
        .unwrap();
    for t in [1i32, 2, 3] {
        model
            .forward_step(&mut cache, StepRequest::last(&[t]))
            .unwrap();
    }
    let ids4 = input_ids(&[4, 5, 6, 7], &device).unwrap();
    model
        .forward_step(
            &mut cache,
            StepRequest {
                tokens: StepTokens::Device(&ids4),
                scope: LogitsScope::All,
                want_hidden: true,
            },
        )
        .unwrap();
    device.synchronize().unwrap();
    let base = cache.len();
    // Keep every checkpoint while recording: pruning one inside a capture frees a tensor
    // allocated before it (`cuMemFreeAsync` → INVALID_VALUE), which abandons the recording.
    cache.retain_checkpoints(64).unwrap();

    let decode = census_step(&model, &mut cache, StepRequest::last(&[8])).unwrap();
    assert_eq!(cache.len(), base);
    let verify = census_step(
        &model,
        &mut cache,
        StepRequest {
            tokens: StepTokens::Device(&ids4),
            scope: LogitsScope::All,
            want_hidden: true,
        },
    )
    .unwrap();
    assert_eq!(cache.len(), base);
    let layers = model.config().num_layers;
    for (name, census) in [("decode (1 token)", decode), ("verify (4 tokens)", verify)] {
        match census {
            Some(c) => eprintln!(
                "[cuda_graphs] 27B {name} step, {layers} layers: {} — per layer ~{:.1} kernels, {:.1} host uploads",
                c.describe(),
                c.kernels as f64 / layers as f64,
                c.memcpy_from_host as f64 / layers as f64
            ),
            None => eprintln!("[cuda_graphs] 27B {name} step: the recording was abandoned"),
        }
    }
}
