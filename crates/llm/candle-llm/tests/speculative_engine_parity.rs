//! Real-weight acceptance and root-cause diagnostics for the unified speculative engine
//! (sc-24130) on Qwen3.8-27B (`#[ignore]`d — needs the frozen snapshot and a GPU with ~54 GB free):
//!
//! * **AC1 (free-running, knife-edge gate)** — the 256-token greedy fixture through the engine
//!   with the MTP proposer at K = 1..5 and with the n-gram proposer, each compared token-for-token
//!   with the speculative-off `StepModel` driver (itself token-identical to the reference loop,
//!   sc-24129 / sc-24132). The literal wording ("token-identical") cannot hold on this hardware:
//!   the op-isolation survey below shows every cuBLAS projection GEMM rounds row 0 differently
//!   for `M >= 2` rows than for the row alone, so a verify forward (`M = K + 1`) lands on the
//!   other side of a bf16 tie wherever the reference's own top-2 logit gap is within one bf16 ULP
//!   of its top logit. The gate therefore is: a row's first divergence, if any, is one of the
//!   reference's **enumerated knife-edge positions** (top-2 gap `<= 1` bf16 ULP), and the
//!   positions are listed in the evidence. The acceptance rate of every MTP row is printed for
//!   comparison with the pre-engine loop's sealed rows.
//! * **AC2** — every engine row's record reports exactly one host sync per verify step.
//! * **Teacher-forced knife-edge gate** — every position of the reference fixture re-decoded
//!   through a *verify-shaped* forward (`M = K + 1` tokens, row 0) against the single-token
//!   forward: positions where the argmax differs are enumerated with the reference's top-2 logit
//!   gap in bf16 ULPs. The gate: argmax-identical at every position whose reference top-2 gap
//!   exceeds one bf16 ULP of its top logit.
//! * **Op isolation survey** — which primitive, at the 27B decode shapes, gives a different
//!   last bit for row 0 of an `M`-row input than for the same row alone: the projection GEMMs
//!   (attention q/k/v/o, DeltaNet in-projections, the MLP, the LM head), grouped-query attention
//!   (`sdpa_gqa_causal`, `q_len = 1` vs `M`), the DeltaNet recurrence and short conv, and RMSNorm.
//!   This is the root-cause evidence for the token-107 knife-edge.
//!
//! ```text
//! BONSAI_QWEN38_SNAPSHOT=E:\...\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
//!   cargo test --release --features cuda -p candle-llm --test speculative_engine_parity -- --ignored --nocapture
//! ```
//!
//! `BONSAI_QWEN38_SNAPSHOT` is the manifest's own environment name for this model
//! (`release/real-weight-models.toml`); it is a passed-in path, never derived.
//!
//! **The llama family (sc-24140).** The knife-edge helpers are generic over [`StepModel`], and
//! `llama_family_qwen3_8b_exact_rows_and_ngram_knife_edge_gate` holds Qwen3-8B (a `CausalLm`,
//! `QWEN3_8B_SNAPSHOT`, ~16 GB bf16) to the same 256-token fixture: the static step seam, the
//! fused-off loops and the CUDA-graph runner's fallback are **token-identical** to the reference
//! loop, and every n-gram row's first divergence, if any, is one of the reference's enumerated
//! knife-edge positions:
//!
//! ```text
//! QWEN3_8B_SNAPSHOT=E:\...\models--Qwen--Qwen3-8B\snapshots\b968826d9c46dd6066d109eabc6255188de91218 \
//!   cargo test --release --features cuda -p candle-llm --test speculative_engine_parity \
//!   llama_family -- --ignored --nocapture
//! ```

mod common;

use candle_core::{DType, Device, Tensor};
use candle_llm::decode::{
    generate_speculative, generate_step, CancelFlag, DecodePath, GenerationConfig, LogitsScope,
    MtpProposer, NgramProposer, Proposer, SpeculativePrompt, StepModel, StepRequest, StepTokens,
};
use candle_llm::device::select_device;
use candle_llm::models::{CausalLm, Qwen35Model};
use candle_llm::primitives::sampler::{argmax_device, argmax_rows_device};
use candle_llm::primitives::{
    causal_depthwise_conv, gated_delta_recurrence, input_ids, rms_norm, sdpa_gqa_causal,
    KvCacheKind, Projection, Weights,
};
use core_llm::ProposerKind;

const SNAPSHOT_VAR: &str = "BONSAI_QWEN38_SNAPSHOT";
/// The manifest's environment name for the llama-family real-weight model (sc-24138 / sc-24140).
const QWEN3_8B_VAR: &str = "QWEN3_8B_SNAPSHOT";
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

fn host(t: &Tensor) -> Vec<f32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// bf16 ULP of a value: 8 significant bits, so `2^(exponent - 7)`.
fn bf16_ulp(x: f32) -> f32 {
    let x = x.abs().max(f32::MIN_POSITIVE);
    (2f32).powi(x.log2().floor() as i32 - 7)
}

/// `(argmax, top-2 gap)` of a logits row.
fn top2(row: &[f32]) -> (usize, f32) {
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
}

/// The single-token path's `(argmax, top-2 gap, bf16 ULP of the top logit)` at every position of
/// the reference fixture, walked on `cache` (fresh; the Qwen3.5 hybrid's growing cache, a
/// `CausalLm`'s static step cache — any [`StepModel`], sc-24140). A position whose gap is within
/// one ULP is a **knife-edge**: a tie the last bit of a differently-rounded GEMM can flip. The
/// device argmax (the sampler's own pick, which the fixture was produced with) is checked against
/// the fixture; the host top-2 scan breaks exact ties by lowest index, which the device kernel
/// need not.
fn single_token_gaps<M: StepModel>(
    model: &M,
    mut cache: M::Cache,
    prompt: &[i32],
    reference: &[i32],
) -> Vec<(i32, f32, f32)> {
    let mut logits = model
        .forward_step(&mut cache, StepRequest::last(prompt))
        .unwrap()
        .logits;
    let mut gaps = Vec::with_capacity(reference.len());
    for (pos, &token) in reference.iter().enumerate() {
        let row = host(&logits);
        let (arg, gap) = top2(&row);
        let device_arg = argmax_device(&logits).unwrap();
        assert_eq!(
            device_arg, token,
            "single-token walk must reproduce the fixture at {pos}"
        );
        gaps.push((device_arg, gap, bf16_ulp(row[arg])));
        if pos + 1 == reference.len() {
            break;
        }
        logits = model
            .forward_step(&mut cache, StepRequest::last(&[token]))
            .unwrap()
            .logits;
    }
    gaps
}

/// The knife-edge positions of [`single_token_gaps`]: `(position, gap, ulp)` with `gap <= ulp`.
fn knife_edges(gaps: &[(i32, f32, f32)]) -> Vec<(usize, f32, f32)> {
    gaps.iter()
        .enumerate()
        .filter(|(_, (_, gap, ulp))| gap <= ulp)
        .map(|(pos, (_, gap, ulp))| (pos, *gap, *ulp))
        .collect()
}

/// The speculative-off fixture through the step driver (the engine's own seam, static KV).
fn fixture_reference(model: &Qwen35Model, prompt: &[i32]) -> Vec<i32> {
    let (out, record) = generate_step(
        model,
        prompt,
        &greedy(FIXTURE_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap();
    assert_eq!(out.tokens.len(), FIXTURE_TOKENS);
    assert_eq!(record.kv_cache, KvCacheKind::Static);
    out.tokens
}

fn engine_row<P: Proposer>(
    model: &Qwen35Model,
    proposer: &mut P,
    prompt: &[i32],
    drafts: usize,
) -> candle_llm::decode::SpeculativeRun {
    generate_speculative(
        model,
        proposer,
        SpeculativePrompt::Tokens(prompt),
        &greedy(FIXTURE_TOKENS),
        drafts,
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap()
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn ac1_ac2_engine_greedy_fixture_rows_against_speculative_off() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (model, mtp) = common::qwen35::load(&snapshot, &device);
    let mtp = mtp.expect("the frozen Qwen3.8 snapshot carries a complete MTP head");
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let reference = fixture_reference(&model, &prompt);
    let gaps = single_token_gaps(
        &model,
        model.new_cache_with_checkpoints(0),
        &prompt,
        &reference,
    );
    let edges = knife_edges(&gaps);
    eprintln!(
        "[ac1] reference knife-edge positions (top-2 gap <= 1 bf16 ULP of the top logit): {:?}",
        edges
    );

    let mut rows: Vec<(String, Option<usize>)> = Vec::new();
    for k in 1..=5usize {
        let mut proposer = MtpProposer::new(&mtp);
        let run = engine_row(&model, &mut proposer, &prompt, k);
        let divergence = first_divergence(&reference, &run.output.tokens);
        let record = run.record;
        eprintln!(
            "[ac1] mtp K={k}: tokens {} divergence {divergence:?} acceptance {:.3} fwd/tok {:.3} \
             syncs/verify {:?} syncs/tok {:.2} kv {} attn {} proposer {} verify_steps {}",
            run.output.tokens.len(),
            record.acceptance_rate().unwrap_or(0.0),
            record.forwards_per_generated_token().unwrap_or(0.0),
            record.host_syncs_per_verify_step(),
            record.host_syncs_per_token().unwrap_or(0.0),
            record.kv_cache.label(),
            record.attn_formulation.label(),
            record.proposer.label(),
            record.verify_steps,
        );
        assert_eq!(record.path, DecodePath::Mtp { drafts: k as u32 });
        assert_eq!(record.proposer, ProposerKind::Mtp);
        assert_eq!(record.kv_cache, KvCacheKind::Static);
        assert_eq!(run.output.tokens.len(), FIXTURE_TOKENS);
        // AC2: one host sync per verify step.
        assert_eq!(
            record.host_syncs_per_verify_step(),
            Some(1.0),
            "AC2 at K={k}"
        );
        assert_eq!(record.host_syncs, 1 + record.verify_steps, "AC2 at K={k}");
        rows.push((format!("mtp K={k}"), divergence));
    }
    let mut proposer = NgramProposer { max_ngram: 3 };
    let run = engine_row(&model, &mut proposer, &prompt, 3);
    let divergence = first_divergence(&reference, &run.output.tokens);
    eprintln!(
        "[ac1] ngram K=3: tokens {} divergence {divergence:?} acceptance {:.3} proposed {} \
         fwd/tok {:.3} syncs/verify {:?} kv {} proposer {}",
        run.output.tokens.len(),
        run.record.acceptance_rate().unwrap_or(0.0),
        run.record.proposed_tokens,
        run.record.forwards_per_generated_token().unwrap_or(0.0),
        run.record.host_syncs_per_verify_step(),
        run.record.kv_cache.label(),
        run.record.proposer.label(),
    );
    assert_eq!(run.record.proposer, ProposerKind::Ngram);
    assert_eq!(run.record.path, DecodePath::PromptLookup);
    assert_eq!(
        run.record.host_syncs_per_verify_step(),
        Some(1.0),
        "AC2 ngram"
    );
    rows.push(("ngram K=3".to_string(), divergence));

    let diverged: Vec<&(String, Option<usize>)> =
        rows.iter().filter(|(_, d)| d.is_some()).collect();
    for (row, d) in &diverged {
        let pos = d.unwrap();
        let (_, gap, ulp) = gaps[pos];
        eprintln!(
            "[ac1] {row}: first divergence at {pos}: reference top-2 gap {gap} = {:.1} bf16 ULP \
             of its top logit",
            gap / ulp
        );
    }
    let off_edge = off_edge_divergences(&rows, &edges);
    assert!(
        off_edge.is_empty(),
        "AC1 gate: rows diverged from speculative-off at a position that is not a reference \
         knife-edge: {off_edge:?} (knife-edges {edges:?})"
    );
    eprintln!(
        "[ac1] {} of {} rows token-identical over {FIXTURE_TOKENS} tokens; every divergence is a \
         reference knife-edge position: {diverged:?}",
        rows.len() - diverged.len(),
        rows.len()
    );
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn teacher_forced_verify_shaped_forward_vs_single_token_knife_edge_gate() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let (model, _mtp) = common::qwen35::load(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let reference = fixture_reference(&model, &prompt);
    let mut sequence = prompt.clone();
    sequence.extend(&reference);
    let gaps = single_token_gaps(
        &model,
        model.new_cache_with_checkpoints(0),
        &prompt,
        &reference,
    );
    let edges = knife_edges(&gaps);
    eprintln!(
        "[teacher-forced] reference knife-edge positions (top-2 gap <= 1 bf16 ULP): {edges:?}"
    );

    // Walk the reference one token at a time on a growing cache (cheap to clone), and at every
    // position run a verify-shaped forward over the next `K + 1` reference tokens from a clone:
    // row `i` of that forward is position `pos + i` computed with `M = K + 1` (every row, not
    // just the first — the engine commits from all of them).
    let mut violations = 0usize;
    let mut worst_flip_ulps = 0f32;
    for k in 1..=5usize {
        let mut single = model.new_cache_with_checkpoints(0);
        model
            .forward_step(&mut single, StepRequest::last(&prompt))
            .unwrap();
        let mut disagreements = Vec::new();
        let mut positions_checked = 0usize;
        for pos in 0..FIXTURE_TOKENS {
            let cur = prompt.len() + pos;
            let end = (cur + k + 1).min(sequence.len());
            if end - cur < 2 {
                break;
            }
            let mut trial = single.try_clone().unwrap();
            let multi = model
                .forward_step(
                    &mut trial,
                    StepRequest {
                        tokens: StepTokens::Host(&sequence[cur..end]),
                        scope: LogitsScope::All,
                        want_hidden: false,
                    },
                )
                .unwrap()
                .logits;
            // Row i is the logits after feeding sequence[cur + i], which pick generated token
            // pos + i + 1: compare with the single-token path's argmax and gap at that index.
            let rows = end - cur;
            let args = argmax_rows_device(&multi).unwrap();
            for (i, &arg_k) in args.iter().enumerate().take(rows) {
                let Some(&(arg_1, gap, ulp)) = gaps.get(pos + i + 1) else {
                    break;
                };
                positions_checked += 1;
                if arg_1 != arg_k {
                    let excused = gap <= ulp;
                    worst_flip_ulps = worst_flip_ulps.max(gap / ulp);
                    disagreements.push((pos + i + 1, i, arg_1, arg_k, gap, ulp, excused));
                    if !excused {
                        violations += 1;
                    }
                }
            }
            model
                .forward_step(&mut single, StepRequest::last(&[sequence[cur]]))
                .unwrap();
        }
        eprintln!(
            "[teacher-forced] M={}: argmax disagreements {} over {positions_checked} (position, \
             row) pairs",
            k + 1,
            disagreements.len(),
        );
        for (pos, row, a1, ak, gap, ulp, excused) in &disagreements {
            eprintln!(
                "[teacher-forced] M={}: position {pos} (verify row {row}): single-token argmax \
                 {a1}, verify-shaped argmax {ak}, reference top-2 gap {gap} = {:.1} bf16 ULP, \
                 knife-edge {excused}",
                k + 1,
                gap / ulp
            );
        }
    }
    eprintln!(
        "[teacher-forced] largest reference top-2 gap at any flipped position: \
         {worst_flip_ulps:.1} bf16 ULP"
    );
    assert_eq!(
        violations, 0,
        "a verify-shaped forward changed the argmax at a position that was not a bf16 knife-edge"
    );
}

/// The positions of `rows` whose first divergence from the reference is **not** an enumerated
/// knife-edge — the AC1 gate's violations (empty = the gate holds).
fn off_edge_divergences<'a>(
    rows: &'a [(String, Option<usize>)],
    edges: &[(usize, f32, f32)],
) -> Vec<&'a (String, Option<usize>)> {
    rows.iter()
        .filter(|(_, d)| d.is_some_and(|pos| !edges.iter().any(|(edge, _, _)| *edge == pos)))
        .collect()
}

#[test]
fn the_knife_edge_gate_excuses_only_enumerated_positions() {
    // Gaps (argmax, gap, ulp): positions 1 and 3 are knife-edges (gap <= 1 ULP).
    let gaps = [
        (5, 0.5, 0.0625),
        (6, 0.0625, 0.0625),
        (7, 0.25, 0.0625),
        (8, 0.0, 0.0625),
    ];
    let edges = knife_edges(&gaps);
    assert_eq!(
        edges.iter().map(|(p, _, _)| *p).collect::<Vec<_>>(),
        vec![1, 3]
    );
    let rows = vec![
        ("identical".to_string(), None),
        ("at an edge".to_string(), Some(3)),
        ("off an edge".to_string(), Some(2)),
    ];
    let off = off_edge_divergences(&rows, &edges);
    assert_eq!(off.len(), 1);
    assert_eq!(off[0].0, "off an edge");
    assert!(off_edge_divergences(&rows[..2], &edges).is_empty());
}

/// The Qwen3-8B `CausalLm` (bf16, the model default: `Gqa` reference arithmetic, static step
/// cache).
fn load_qwen3_8b(snapshot: &std::path::Path, device: &Device) -> CausalLm {
    let cfg = candle_llm::config::ModelConfig::from_dir(snapshot).expect("config.json");
    assert!(
        !cfg.architecture.is_mla() && cfg.moe.is_none(),
        "the llama-family gate runs a dense GQA decoder"
    );
    let weights = Weights::from_dir(snapshot, device).expect("load weights");
    CausalLm::from_weights_format(&weights, "", cfg, None).expect("build model")
}

/// The `CausalLm` reference loop (E2): a `decode_logits` prefill on the model's own growing cache,
/// then the shared token-at-a-time `Decode` loop.
fn causal_reference_tokens(model: &CausalLm, prompt: &[i32], device: &Device) -> Vec<i32> {
    let mut cache = model.new_cache();
    let first = model
        .decode_logits(&input_ids(prompt, device).unwrap(), &mut cache, 0)
        .unwrap();
    candle_llm::decode::generate_from_prefill(
        model,
        &mut cache,
        first,
        prompt.to_vec(),
        &greedy(FIXTURE_TOKENS),
        &CancelFlag::new(),
        &mut |_| {},
        None,
    )
    .unwrap()
    .tokens
}

/// sc-24140 (epic AT3 / E1 on the llama family). Qwen3-8B, the 256-token greedy fixture:
///
/// * **exact** (token-identical to the reference loop): the static step seam; the reference loop
///   and the static step seam with the fused primitives switched off; and the CUDA-graph runner
///   with the switch on — which a `CausalLm` step refuses (`positions_host_scalar`), so every step
///   runs eager through the runner's fallback, recorded with that reason;
/// * **knife-edge gate**: the n-gram rows (K = 2, 3, 4 — the sc-24138 comparison rows that
///   diverged at @65, @51, @132) diverge, if at all, only at an enumerated reference knife-edge
///   (top-2 gap ≤ 1 bf16 ULP of the top logit on the single-token static path), with exactly one
///   host sync per verify step.
#[test]
#[ignore = "needs the Qwen3-8B snapshot via QWEN3_8B_SNAPSHOT and a GPU (~16 GB)"]
fn llama_family_qwen3_8b_exact_rows_and_ngram_knife_edge_gate() {
    use candle_llm::decode::graph::{cuda_graphs_policy_guard, graph_tally, GraphRunner};
    use candle_llm::primitives::fused_policy_guard;

    let snapshot = common::qwen35::snapshot_from_env(QWEN3_8B_VAR)
        .unwrap_or_else(|| panic!("set {QWEN3_8B_VAR}"));
    // The device as a graphs-on load gets it: `select_device` resolves the stream from the switch,
    // so with it on the model runs on its own stream (every row below — the stream changes no
    // arithmetic) and the graph-runner row exercises the `CausalLm` step's own refusal, not the
    // legacy stream's.
    let device = {
        let _graphs = cuda_graphs_policy_guard(Some(true));
        select_device().unwrap()
    };
    let model = load_qwen3_8b(&snapshot, &device);
    let prompt = common::qwen35::render_chat_prompt(&snapshot, PROMPT);
    let reference = causal_reference_tokens(&model, &prompt, &device);
    assert_eq!(reference.len(), FIXTURE_TOKENS);
    let static_cache = || {
        model
            .new_static_cache(prompt.len() + FIXTURE_TOKENS + 8)
            .unwrap()
    };
    let gaps = single_token_gaps(&model, static_cache(), &prompt, &reference);
    let edges = knife_edges(&gaps);
    eprintln!(
        "[llama] Qwen3-8B reference: {} prompt tokens, {FIXTURE_TOKENS} generated; knife-edge \
         positions (top-2 gap <= 1 bf16 ULP of the top logit): {edges:?}",
        prompt.len()
    );

    // The exact rows.
    let step = |m: &dyn StepModel<Cache = candle_llm::primitives::StepKvCache>| {
        generate_step(
            m,
            &prompt,
            &greedy(FIXTURE_TOKENS),
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap()
    };
    let (static_out, static_record) = step(&model);
    assert_eq!(static_record.kv_cache, KvCacheKind::Static);
    let mut exact: Vec<(&str, Option<usize>)> = vec![(
        "static step seam",
        first_divergence(&reference, &static_out.tokens),
    )];
    {
        let _off = fused_policy_guard(Some(false));
        exact.push((
            "reference loop, fused off",
            first_divergence(
                &reference,
                &causal_reference_tokens(&model, &prompt, &device),
            ),
        ));
        exact.push((
            "static step seam, fused off",
            first_divergence(&reference, &step(&model).0.tokens),
        ));
    }
    let graphs = {
        let _on = cuda_graphs_policy_guard(Some(true));
        let runner = GraphRunner::new(&model);
        let before = graph_tally();
        let (out, _) = step(&runner);
        exact.push((
            "graph runner (switch on)",
            first_divergence(&reference, &out.tokens),
        ));
        graph_tally().since(&before)
    };
    eprintln!(
        "[llama] graph runner: {} replayed / {} eager, {} captured, fallback {:?}",
        graphs.replayed, graphs.eager, graphs.captured, graphs.fallback_reason
    );
    for (row, d) in &exact {
        eprintln!("[llama] exact row {row}: first divergence {d:?}");
    }
    assert!(
        exact.iter().all(|(_, d)| d.is_none()),
        "exact rows must be token-identical to the reference loop: {exact:?}"
    );
    assert_eq!(graphs.replayed, 0, "a CausalLm step is not replayable");
    assert_eq!(graphs.fallback_reason, Some("positions_host_scalar"));

    // The n-gram rows against the enumerated knife-edges.
    let mut rows: Vec<(String, Option<usize>)> = Vec::new();
    for k in [2usize, 3, 4] {
        let mut proposer = NgramProposer { max_ngram: 3 };
        let run = generate_speculative(
            &model,
            &mut proposer,
            SpeculativePrompt::Tokens(&prompt),
            &greedy(FIXTURE_TOKENS),
            k,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let divergence = first_divergence(&reference, &run.output.tokens);
        let record = run.record;
        eprintln!(
            "[llama] ngram K={k}: divergence {divergence:?} acceptance {:.3} proposed {} \
             syncs/verify {:?} kv {} proposer {}",
            record.acceptance_rate().unwrap_or(0.0),
            record.proposed_tokens,
            record.host_syncs_per_verify_step(),
            record.kv_cache.label(),
            record.proposer.label(),
        );
        assert_eq!(record.path, DecodePath::PromptLookup);
        assert_eq!(record.proposer, ProposerKind::Ngram);
        assert_eq!(run.output.tokens.len(), FIXTURE_TOKENS);
        assert_eq!(
            record.host_syncs_per_verify_step(),
            Some(1.0),
            "AC2 ngram K={k}"
        );
        if let Some(pos) = divergence {
            let (_, gap, ulp) = gaps[pos];
            eprintln!(
                "[llama] ngram K={k}: first divergence at {pos}: reference top-2 gap {gap} = \
                 {:.2} bf16 ULP of its top logit",
                gap / ulp
            );
        }
        rows.push((format!("ngram K={k}"), divergence));
    }
    let off_edge = off_edge_divergences(&rows, &edges);
    assert!(
        off_edge.is_empty(),
        "knife-edge gate: n-gram rows diverged from the reference at a position that is not an \
         enumerated knife-edge: {off_edge:?} (knife-edges {edges:?})"
    );
    eprintln!(
        "[llama] gate holds: exact rows identical over {FIXTURE_TOKENS} tokens; n-gram divergences \
         {rows:?} are all enumerated knife-edges"
    );
}

/// Compare row 0 of `f` over an `M`-row input with `f` over that row alone.
fn row0_survey(name: &str, f: &dyn Fn(usize) -> Tensor, ms: &[usize]) -> Vec<(usize, bool, f32)> {
    let alone = host(&f(1));
    let mut out = Vec::new();
    for &m in ms {
        let rows = host(&f(m));
        let width = alone.len();
        let row0 = &rows[..width];
        let identical = row0 == alone.as_slice();
        let max_ulps = row0
            .iter()
            .zip(&alone)
            .map(|(a, b)| (a - b).abs() / bf16_ulp(*b))
            .fold(0f32, f32::max);
        out.push((m, identical, max_ulps));
    }
    eprintln!(
        "[op-survey] {name}: {}",
        out.iter()
            .map(|(m, same, ulps)| format!(
                "M={m}: {}",
                if *same {
                    "bit-identical".to_string()
                } else {
                    format!("differs (max {ulps:.1} bf16 ULP)")
                }
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    out
}

#[test]
#[ignore = "needs the Qwen3.8-27B snapshot via BONSAI_QWEN38_SNAPSHOT and a GPU"]
fn op_isolation_survey_which_primitive_depends_on_the_row_count() {
    let snapshot = common::qwen35::snapshot_from_env(SNAPSHOT_VAR)
        .unwrap_or_else(|| panic!("set {SNAPSHOT_VAR}"));
    let device = select_device().unwrap();
    let weights = Weights::from_dir(&snapshot, &device).expect("load weights");
    let dtype = DType::BF16;
    let hidden = 5120usize;
    let ms = [2usize, 3, 4, 5, 6];
    let randn = |shape: &[usize], seed: u64| -> Tensor {
        // Deterministic host normal draws (Box-Muller over SplitMix64), then bf16 on device.
        let n: usize = shape.iter().product();
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut v = Vec::with_capacity(n);
        while v.len() < n {
            let (u1, u2) = (next().max(1e-7), next());
            let r = (-2.0 * u1.ln()).sqrt();
            v.push(r * (2.0 * std::f32::consts::PI * u2).cos());
            if v.len() < n {
                v.push(r * (2.0 * std::f32::consts::PI * u2).sin());
            }
        }
        Tensor::from_vec(v, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
            .to_device(&device)
            .unwrap()
    };
    let x = randn(&[1, 1, hidden], 1);
    let rows_of = |x: &Tensor, m: usize| -> Tensor {
        Tensor::cat(&vec![x; m], 1).unwrap().contiguous().unwrap()
    };
    let mut gemm_dependent = Vec::new();
    // Layer 3 is the first full-attention layer (interval 4); layer 0 a DeltaNet layer.
    for key in [
        "model.language_model.layers.3.self_attn.q_proj.weight",
        "model.language_model.layers.3.self_attn.k_proj.weight",
        "model.language_model.layers.3.self_attn.v_proj.weight",
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.out_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "lm_head.weight",
    ] {
        let Some(w) = weights.get(key) else {
            eprintln!("[op-survey] {key}: absent from the snapshot");
            continue;
        };
        let w = w.to_dtype(dtype).unwrap();
        let in_dim = w.dim(1).unwrap();
        let proj = Projection::load(w, None).unwrap();
        let x = if in_dim == hidden {
            x.clone()
        } else {
            randn(&[1, 1, in_dim], 7)
        };
        let short = key.trim_start_matches("model.language_model.");
        let survey = row0_survey(
            &format!("projection {short} [{in_dim} -> ...]"),
            &|m| proj.forward(&rows_of(&x, m)).unwrap(),
            &ms,
        );
        if survey.iter().any(|(_, same, _)| !same) {
            gemm_dependent.push(short.to_string());
        }
    }

    // Grouped-query attention at the fixture's knife-edge key length (prompt 97 + 107 + 1).
    let (heads, kv_heads, head_dim) = (24usize, 4usize, 128usize);
    let scale = (head_dim as f32).powf(-0.5);
    for key_len in [205usize, 128, 353] {
        let q = randn(&[1, heads, 1, head_dim], 11);
        let k_all = randn(&[1, kv_heads, key_len + 6, head_dim], 12);
        let v_all = randn(&[1, kv_heads, key_len + 6, head_dim], 13);
        row0_survey(
            &format!("sdpa_gqa_causal row 0 over {key_len} keys (q_len = M)"),
            &|m| {
                // Row 0 of an M-row query attends keys[..key_len]: keys/values of length
                // key_len + M - 1 under bottom-right causal alignment.
                let q_m = Tensor::cat(
                    &std::iter::once(q.clone())
                        .chain((1..m).map(|i| randn(&[1, heads, 1, head_dim], 100 + i as u64)))
                        .collect::<Vec<_>>(),
                    2,
                )
                .unwrap();
                let k = k_all.narrow(2, 0, key_len + m - 1).unwrap();
                let v = v_all.narrow(2, 0, key_len + m - 1).unwrap();
                sdpa_gqa_causal(&q_m, &k, &v, scale)
                    .unwrap()
                    .narrow(2, 0, 1)
                    .unwrap()
            },
            &ms,
        );
    }

    // The DeltaNet recurrence (token-sequential by construction) and the short conv.
    let (hv, hk, dk, dv) = (48usize, 16usize, 128usize, 128usize);
    let state = randn(&[1, hv, dv, dk], 21).to_dtype(DType::F32).unwrap();
    let qd = randn(&[1, 6, hk, dk], 22).to_dtype(DType::F32).unwrap();
    let kd = randn(&[1, 6, hk, dk], 23).to_dtype(DType::F32).unwrap();
    let vd = randn(&[1, 6, hv, dv], 24).to_dtype(DType::F32).unwrap();
    let g = randn(&[1, 6, hv], 25)
        .to_dtype(DType::F32)
        .unwrap()
        .abs()
        .unwrap()
        .neg()
        .unwrap();
    let beta = randn(&[1, 6, hv], 26).to_dtype(DType::F32).unwrap();
    row0_survey(
        "gated_delta_recurrence step 0 (T = M)",
        &|m| {
            let (y, _) = gated_delta_recurrence(
                &qd.narrow(1, 0, m).unwrap(),
                &kd.narrow(1, 0, m).unwrap(),
                &vd.narrow(1, 0, m).unwrap(),
                &g.narrow(1, 0, m).unwrap(),
                &beta.narrow(1, 0, m).unwrap(),
                Some(&state),
            )
            .unwrap();
            y.narrow(1, 0, 1).unwrap()
        },
        &ms,
    );
    let conv_dim = 2 * hk * dk + hv * dv;
    let conv_w = randn(&[conv_dim, 4], 31);
    let conv_state = randn(&[1, 3, conv_dim], 32);
    let conv_x = randn(&[1, 6, conv_dim], 33);
    row0_survey(
        "causal_depthwise_conv row 0 (S = M)",
        &|m| {
            causal_depthwise_conv(&conv_x.narrow(1, 0, m).unwrap(), &conv_w, &conv_state)
                .unwrap()
                .0
                .narrow(1, 0, 1)
                .unwrap()
        },
        &ms,
    );
    let norm_w = randn(&[hidden], 41);
    row0_survey(
        "rms_norm row 0",
        &|m| rms_norm(&rows_of(&x, m), &norm_w, 1e-6).unwrap(),
        &ms,
    );
    let _ = input_ids;
    eprintln!(
        "[op-survey] projection GEMMs whose row 0 depends on the row count: {gemm_dependent:?}"
    );
}
