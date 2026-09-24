//! Real-weight acceptance for the static KV cache (sc-24132) on Qwen3.8-27B (`#[ignore]`d —
//! needs the frozen snapshot and a GPU with ~54 GB free):
//!
//! * **AC1** — a 256-token greedy decode on the preallocated [`StaticKvCache`] (the `StepModel`
//!   driver's default cache) is token-identical to the `AttnKv` reference path — both the
//!   reference `Decode` loop and the same driver with the growing cache selected — and the record
//!   names the cache and the attention formulation that ran. Since S4 the reference path attends
//!   through the same `sdpa_gqa_causal` as the static cache (the coordinator's decision: bit
//!   identity between the un-expanded GQA matmul and the pre-S4 `repeat_kv`-expanded GEMM is a
//!   cuBLAS kernel-selection property and not attainable), so this parity holds by construction;
//!   the pre-S4 `expanded` arithmetic stays selectable only as a labelled bench comparison row.
//! * **AC3 (real weights)** — the static buffers' CUDA device pointers are unchanged across the
//!   whole 256-token run and a rollback.
//! * **sc-24140 (provider)** — the provider's default path for a request whose speculation is off
//!   (the engine with no proposer, on the static cache) is token-identical to the reference loop
//!   selected on the same provider, event for event over the 256-token fixture
//!   (`provider_off_path_is_token_identical_to_the_reference_loop`; evidence row with
//!   `SC24140_EVIDENCE_OUTPUT`).
//! * **Teacher-forced diagnostic** (not an acceptance gate) — the static path against the pre-S4
//!   `expanded` reference arithmetic, both fed the reference's own 256 tokens; reports per-position
//!   argmax agreement, the max |Δlogit| and the reference's top-2 logit gap wherever the argmax
//!   differs (a bf16-ULP tie). It documents the ≤ 1 bf16 ULP knife-edge behaviour the decision
//!   accepted; it is not cited as AC1.
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
use candle_llm::primitives::{kv_materialize_count, AttnFormulation, DecodeCache, KvCacheKind};

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

    // The reference loop: AttnKv slots, attending through `sdpa_gqa_causal` (the S4 reference).
    assert_eq!(model.attn_formulation(), AttnFormulation::Gqa);
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
    assert_eq!(record.attn_formulation, AttnFormulation::Gqa);
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
    assert_eq!(record.attn_formulation, AttnFormulation::Gqa);
    assert_eq!(
        growing.tokens,
        reference.tokens,
        "growing StepModel decode diverged from the reference loop at {:?}",
        first_divergence(&reference.tokens, &growing.tokens)
    );
    eprintln!(
        "[ac1] {} tokens identical across reference / static / growing (gqa); growing record \
         {record:?}",
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

/// **Diagnostic only — not the AC1 gate.** The static path against the pre-S4 `expanded`
/// reference arithmetic (`repeat_kv` + `sdpa`, selected explicitly here), both fed that reference's
/// own greedy tokens position by position: where do the argmaxes differ, by how much do the logits
/// differ, and how close was the reference's top-2 gap there? It characterizes the ≤ 1 bf16 ULP
/// knife-edge difference between the two formulations that the S4 decision accepted (the
/// `attention::tests::gqa_variants_bit_match_survey` companion) and checks every argmax
/// disagreement is a bf16 knife-edge in the expanded reference itself (top-2 gap within 4 ULPs of
/// the top logit). AC1 is `ac1_static_kv_greedy_fixture_is_token_identical_to_attn_kv`.
#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn teacher_forced_static_vs_attn_kv_logit_parity_report() {
    use candle_core::DType;
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (mut model, _mtp) = common::qwen35::load(&snapshot, &device);
    // The pre-S4 arithmetic on the growing slots; the static cache always attends un-expanded.
    model.set_attn_formulation(AttnFormulation::Expanded);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let reference = generate_with(
        &model,
        &prompt,
        &greedy(FIXTURE_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    let host = |t: &candle_core::Tensor| -> Vec<f32> {
        t.flatten_all()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    use candle_llm::primitives::sampler::argmax_device;
    let top2 = |row: &[f32]| -> (usize, f32) {
        let (mut best, mut second) = ((0usize, f32::NEG_INFINITY), f32::NEG_INFINITY);
        for (i, &v) in row.iter().enumerate() {
            if v > best.1 {
                second = best.1;
                best = (i, v);
            } else if v > second {
                second = v;
            }
        }
        (best.0, best.1 - second)
    };
    let capacity = prompt.len() + FIXTURE_TOKENS;
    let mut fixed = model.new_static_cache(capacity, 0).unwrap();
    let mut growing = StepModel::new_cache(&model);
    growing.set_max_checkpoints(0).unwrap();
    let mut a = model
        .forward_step(&mut fixed, StepRequest::last(&prompt))
        .unwrap()
        .logits;
    let mut b = model
        .forward_step(&mut growing, StepRequest::last(&prompt))
        .unwrap()
        .logits;
    // bf16 ULP of a value: 8 significant bits, so 2^(exponent - 7).
    let ulp = |x: f32| -> f32 {
        let x = x.abs().max(f32::MIN_POSITIVE);
        (2f32).powi(x.log2().floor() as i32 - 7)
    };
    let mut max_delta = 0f32;
    let mut max_delta_at = (0usize, 0f32, 0f32); // (position, reference logit, row range)
    let mut sum_delta = 0f32;
    let mut disagreements = Vec::new();
    for (pos, &token) in reference.tokens.iter().enumerate() {
        let (ra, rb) = (host(&a), host(&b));
        let (row_min, row_max) = rb
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let mut delta = 0f32;
        for (&x, &y) in ra.iter().zip(&rb) {
            let d = (x - y).abs();
            if d > delta {
                delta = d;
            }
            if d > max_delta {
                max_delta = d;
                max_delta_at = (pos, y, row_max - row_min);
            }
        }
        sum_delta += delta;
        // The sampler's own greedy pick (device argmax) on each path; the host top-2 gap says how
        // close the runner-up was (a gap of 0 is an exact bf16 tie).
        let arg_a = argmax_device(&a).unwrap();
        let arg_b = argmax_device(&b).unwrap();
        let (_, gap_a) = top2(&ra);
        let (_, gap_b) = top2(&rb);
        let top_b = rb[arg_b as usize];
        assert_eq!(
            arg_b, token,
            "growing path must reproduce the reference token at {pos}"
        );
        if arg_a != arg_b {
            disagreements.push((pos, arg_a, arg_b, gap_a, gap_b, delta, ulp(top_b)));
        }
        if pos + 1 == reference.tokens.len() {
            break;
        }
        a = model
            .forward_step(&mut fixed, StepRequest::last(&[token]))
            .unwrap()
            .logits;
        b = model
            .forward_step(&mut growing, StepRequest::last(&[token]))
            .unwrap()
            .logits;
    }
    let positions = reference.tokens.len();
    eprintln!(
        "[teacher-forced] {positions} positions; argmax agrees at {}; max |delta logit| =          {max_delta} at position {} (reference logit {}, row range {}); mean per-position max          |delta| = {}",
        positions - disagreements.len(),
        max_delta_at.0,
        max_delta_at.1,
        max_delta_at.2,
        sum_delta / positions as f32
    );
    for (pos, arg_a, arg_b, gap_a, gap_b, delta, top_ulp) in &disagreements {
        eprintln!(
            "[teacher-forced] pos {pos}: static argmax {arg_a} (top-2 gap {gap_a}), reference argmax              {arg_b} (top-2 gap {gap_b} = {} bf16 ULPs of the top logit), max |delta logit| {delta}",
            gap_b / top_ulp
        );
    }
    // The claim this gates: the static path picks a different token only where the reference's
    // own top-2 gap is within bf16 noise of a tie (a few ULPs of the top logit) — never where the
    // reference had a clear winner.
    for (pos, _, _, _, gap_b, _, top_ulp) in &disagreements {
        assert!(
            gap_b / top_ulp <= 4.0,
            "position {pos}: argmax differs although the reference top-2 gap is {gap_b}              ({} bf16 ULPs of the top logit)",
            gap_b / top_ulp
        );
    }
}

/// sc-24140 (item 2 of the feature-end review): a Qwen3.8-27B request whose speculation is off
/// — the provider's default path since sc-24140, the engine with no proposer on the static KV
/// cache — is token-identical to the reference `Decode` loop selected on the same provider,
/// greedy, over the 256-token fixture. Both attend through un-expanded GQA and both are M = 1
/// steps, so the parity is exact (no knife-edge exception applies). Every streamed event (token
/// id, index, channel, text) and the final output are compared. `SC24140_EVIDENCE_OUTPUT` names
/// a JSON file to write the evidence row to.
#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn provider_off_path_is_token_identical_to_the_reference_loop() {
    use candle_llm::LlamaProvider;
    use core_llm::{LoadSpec, Message, MtpMode, Sampling, StreamEvent, TextLlm, TextLlmRequest};

    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let started = std::time::Instant::now();
    let mut provider = LlamaProvider::load(&LoadSpec::dense(snapshot.display().to_string()))
        .expect("load the Qwen3.8-27B provider");
    let load_secs = started.elapsed().as_secs_f64();
    assert_eq!(provider.decode_path(), DecodePath::StepModel, "the default");
    let request = TextLlmRequest {
        messages: vec![Message::user(PROMPT)],
        sampling: Sampling::greedy(),
        max_new_tokens: FIXTURE_TOKENS as u32,
        seed: Some(0),
        mtp: MtpMode::Off,
        ..Default::default()
    };
    type Event = (u32, usize, String, String);
    let run = |provider: &LlamaProvider| {
        let mut events: Vec<Event> = Vec::new();
        let started = std::time::Instant::now();
        let out = provider
            .generate(&request, &mut |event| {
                if let StreamEvent::Token {
                    id,
                    index,
                    channel,
                    text,
                } = event
                {
                    events.push((id, index, format!("{channel:?}"), text));
                }
            })
            .expect("generate");
        let secs = started.elapsed().as_secs_f64();
        let record = provider.last_decode_record().expect("record");
        (out, events, record, secs)
    };

    let (engine, engine_events, engine_record, engine_secs) = run(&provider);
    assert_eq!(engine_record.path, DecodePath::StepModel, "the engine ran");
    assert_eq!(engine_record.kv_cache, KvCacheKind::Static);
    assert_eq!(engine_record.attn_formulation, AttnFormulation::Gqa);
    assert_eq!(engine_record.proposer, core_llm::ProposerKind::None);

    provider
        .set_decode_path(DecodePath::Reference)
        .expect("the reference loop is selectable");
    let (reference, reference_events, reference_record, reference_secs) = run(&provider);
    assert_eq!(reference_record.path, DecodePath::Reference);
    assert_eq!(reference_record.kv_cache, KvCacheKind::Growing);
    assert_eq!(reference_record.attn_formulation, AttnFormulation::Gqa);

    let ids = |events: &[Event]| events.iter().map(|e| e.0 as i32).collect::<Vec<i32>>();
    let divergence = first_divergence(&ids(&reference_events), &ids(&engine_events));
    let identical = engine_events == reference_events
        && engine.text == reference.text
        && engine.thinking == reference.thinking
        && engine.usage == reference.usage
        && engine.finish_reason == reference.finish_reason;
    // The code the row was measured at: HEAD, and whether tracked files had uncommitted changes.
    let git = |args: &[&str]| {
        std::process::Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let commit = git(&["rev-parse", "HEAD"]).expect("git rev-parse HEAD");
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(true);
    let evidence = serde_json::json!({
        "story": "sc-24140",
        "commit": commit,
        "worktree_dirty": dirty,
        "check": "feature-end review item 2: Qwen3.8-27B provider default path (MtpMode::Off) vs the reference Decode loop, greedy",
        "snapshot": snapshot.display().to_string(),
        "device": format!("{:?}", candle_llm::device::select_device().unwrap().location()),
        "prompt": PROMPT,
        "max_new_tokens": FIXTURE_TOKENS,
        "load_secs": load_secs,
        "identical": identical,
        "first_event_divergence": divergence,
        "engine": {
            "path": engine_record.path.label(),
            "kv_cache": engine_record.kv_cache.label(),
            "attn_formulation": engine_record.attn_formulation.label(),
            "proposer": engine_record.proposer.label(),
            "generated_tokens": engine.usage.generated_tokens,
            "events": engine_events.len(),
            "target_forwards": engine_record.target_forwards,
            "prefill_forwards": engine_record.prefill_forwards,
            "verify_steps": engine_record.verify_steps,
            "host_syncs": engine_record.host_syncs,
            "sampler": engine_record.sampler.label(),
            "logits_to_host": engine_record.sampler.logits_to_host,
            "cuda_graphs": engine_record.cuda_graphs.describe(),
            "fused_primitives": engine_record.fused_primitives.label(),
            "finish_reason": format!("{:?}", engine.finish_reason),
            "wall_secs": engine_secs,
        },
        "reference": {
            "path": reference_record.path.label(),
            "kv_cache": reference_record.kv_cache.label(),
            "attn_formulation": reference_record.attn_formulation.label(),
            "proposer": reference_record.proposer.label(),
            "generated_tokens": reference.usage.generated_tokens,
            "events": reference_events.len(),
            "target_forwards": reference_record.target_forwards,
            "host_syncs": reference_record.host_syncs,
            "sampler": reference_record.sampler.label(),
            "finish_reason": format!("{:?}", reference.finish_reason),
            "wall_secs": reference_secs,
        },
        "event_token_ids": ids(&engine_events),
    });
    if let Ok(path) = std::env::var("SC24140_EVIDENCE_OUTPUT") {
        if let Some(parent) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, serde_json::to_string_pretty(&evidence).unwrap()).unwrap();
        eprintln!("[sc-24140] wrote {path}");
    }
    eprintln!(
        "[sc-24140] engine {} tokens / {} events, reference {} tokens / {} events, identical={identical}",
        engine.usage.generated_tokens,
        engine_events.len(),
        reference.usage.generated_tokens,
        reference_events.len()
    );
    assert_eq!(
        engine_events, reference_events,
        "the Off path's stream diverged from the reference loop at event {divergence:?}"
    );
    assert_eq!(engine.text, reference.text);
    assert_eq!(engine.thinking, reference.thinking);
    assert_eq!(engine.usage, reference.usage);
    assert_eq!(engine.finish_reason, reference.finish_reason);
}
