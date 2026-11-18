//! The block-draft proposer's model-inference proofs (SW-7193, SW-7194): a tiny random draft of
//! each format — DSpine and DFlash2 — over a tiny random `qwen3_5` target through the engine's
//! linear verifier. Every test runs learned forwards, so each is ignored in ordinary CI and
//! registered for the explicit model-proof workflow.

use std::collections::HashMap;

use mlx_rs::Dtype;

use super::*;
use crate::decode::engine::{
    generate_speculative, EngineOptions, MlxSampler, Pipelining, RewindableConstraintMask,
    SpeculativePrompt, SpeculativeRun, TokenSampler,
};
use crate::decode::stream::{
    generate_with, ConstraintMask, FinishReason, GenerationConfig, GenerationOutput,
};
use crate::decode::CancelFlag;
use crate::models::dspine::tests::{tiny_random_target, tiny_target, TINY_VOCAB};
use crate::models::{dflash2, dspine};
use crate::primitives::sampler::SamplingParams;
use core_llm::BlockDraftPrecision;

/// Each format's tiny draft at `precision` (`None`: dense BF16), labelled.
fn tiny_drafts(precision: Option<BlockDraftPrecision>) -> [(&'static str, BlockDraft); 2] {
    [
        (
            "dspine",
            BlockDraft::DSpine(dspine::tests::tiny_draft(precision)),
        ),
        ("dflash2", dflash2::tests::tiny_draft(precision)),
    ]
}

/// A prompt longer than the tiny draft's 8-row window.
const PROMPT: [i32; 12] = [3, 9, 4, 11, 3, 9, 4, 11, 3, 9, 4, 11];

/// A constraint forbidding fixed ids, rewindable, recording what it accepted.
struct Forbid {
    allow: Vec<bool>,
    accepted: Vec<i32>,
}

impl Forbid {
    fn new(forbidden: &[usize]) -> Self {
        let mut allow = vec![true; TINY_VOCAB];
        for &f in forbidden {
            allow[f] = false;
        }
        Self {
            allow,
            accepted: Vec::new(),
        }
    }
}

impl ConstraintMask for Forbid {
    fn allowed(&mut self) -> &[bool] {
        &self.allow
    }
    fn accept(&mut self, token: i32) {
        self.accepted.push(token);
    }
}

impl RewindableConstraintMask for Forbid {
    fn checkpoint(&self) -> usize {
        self.accepted.len()
    }
    fn rewind(&mut self, checkpoint: usize) {
        self.accepted.truncate(checkpoint);
    }
}

fn config(max_new_tokens: usize, seed: u64) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(),
        seed: Some(seed),
        stop_tokens: Vec::new(),
    }
}

fn penalized(max_new_tokens: usize) -> GenerationConfig {
    let mut c = config(max_new_tokens, 7);
    c.sampling.repetition_penalty = 1.3;
    c.sampling.repetition_context = 16;
    c
}

/// The plain token-at-a-time loop on the bare target — the reference.
fn plain(
    model: &Qwen35Model,
    prompt: &[i32],
    config: &GenerationConfig,
    constraint: Option<&mut Forbid>,
) -> GenerationOutput {
    generate_with(
        model,
        prompt,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        constraint.map(|c| c as &mut dyn ConstraintMask),
        None,
    )
    .unwrap()
}

/// The engine over the tapped target with `proposer`, from a cache it returns (prefilled here,
/// unpipelined so the cache ends at the committed length).
fn run_block<'d>(
    target: &TappedQwen35<'_>,
    proposer: &mut BlockDraftProposer<'d>,
    prompt: &[i32],
    config: &GenerationConfig,
    depth: usize,
    constraint: Option<&mut Forbid>,
) -> (SpeculativeRun, Qwen35Cache) {
    let mut cache = target.new_cache();
    let out = target
        .forward(&mut cache, &input_ids(prompt), 0, LogitsScope::Last, true)
        .unwrap();
    let run = generate_speculative(
        target,
        proposer,
        SpeculativePrompt::Prefilled {
            cache: &mut cache,
            logits: out.logits,
            hidden: out.hidden,
            history: prompt,
            position_delta: 0,
        },
        config,
        depth,
        &CancelFlag::new(),
        &mut |_| {},
        EngineOptions {
            constraint: constraint.map(|c| c as &mut dyn RewindableConstraintMask),
            pipelining: Pipelining::Off,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    (run, cache)
}

/// AC1 (greedy): for each format, at every draft precision, depth and greedy configuration —
/// plain, penalized, constrained — the block draft commits exactly the plain loop's tokens (the
/// constraint sees the same accepted sequence), proposes, accepts and rejects, and rolls the
/// hybrid back through its checkpoint ring without a replay.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn greedy_block_draft_is_the_plain_loop() {
    let model = tiny_target();
    let mut totals: HashMap<&str, (u64, u64, u64)> = HashMap::new();
    for precision in [
        None,
        Some(BlockDraftPrecision::Q8),
        Some(BlockDraftPrecision::Q4),
    ] {
        for (format, draft) in tiny_drafts(precision) {
            let target = TappedQwen35::for_draft(&model, &draft);
            for depth in [1usize, 3, 5] {
                for (name, cfg, constrained) in [
                    ("greedy", config(24, 7), false),
                    ("penalized", penalized(24), false),
                    ("constrained", config(24, 7), true),
                ] {
                    let label = format!("{format} {precision:?} depth {depth} {name}");
                    let (mut a, mut b) = (Forbid::new(&[1, 9]), Forbid::new(&[1, 9]));
                    let expected = plain(&model, &PROMPT, &cfg, constrained.then_some(&mut a));
                    let mut proposer = BlockDraftProposer::new(&draft);
                    let (run, _) = run_block(
                        &target,
                        &mut proposer,
                        &PROMPT,
                        &cfg,
                        depth,
                        constrained.then_some(&mut b),
                    );
                    let first_difference = run
                        .output
                        .tokens
                        .iter()
                        .zip(&expected.tokens)
                        .position(|(x, y)| x != y);
                    assert_eq!(first_difference, None, "{label}: first differing position");
                    assert_eq!(run.output.tokens, expected.tokens, "{label}");
                    assert_eq!(b.accepted, a.accepted, "{label}: constraint");
                    let r = &run.report;
                    assert_eq!(r.proposer, ProposerKind::BlockDraft, "{label}");
                    assert_eq!(r.path, "block_draft", "{label}");
                    assert_eq!(r.draft_tokens, Some(depth as u32), "{label}");
                    assert_eq!(r.replay_forwards, 0, "{label}: the ring never replays");
                    assert_eq!(
                        run.output.tokens.len() as u64,
                        1 + r.verify_steps + r.accepted_tokens,
                        "{label}: one committed token per pass plus the accepted drafts"
                    );
                    assert!(proposer.draft_forwards > 0, "{label}");
                    assert!(proposer
                        .candidates()
                        .iter()
                        .all(|row| row.len() == 4 && row.windows(2).all(|w| w[0] < w[1])));
                    let t = totals.entry(format).or_default();
                    t.0 += r.proposed_tokens;
                    t.1 += r.accepted_tokens;
                    t.2 += run.stats.direct_rollbacks as u64;
                }
            }
        }
    }
    assert_eq!(totals.len(), 2);
    for (format, (proposed, accepted, rejected)) in totals {
        assert!(
            proposed > 0 && accepted > 0 && rejected > 0,
            "{format}: {proposed} {accepted} {rejected}"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::greedy_block_draft_is_the_plain_loop"
    );
}

/// The rows of `x` as host f32.
fn host(x: &Array) -> Vec<f32> {
    x.as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn assert_close(got: &Array, want: &Array, label: &str) {
    assert_eq!(got.shape(), want.shape(), "{label}");
    let (g, w) = (host(got), host(want));
    let scale = w.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    let worst = g
        .iter()
        .zip(&w)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 0.03 * scale,
        "{label}: max |diff| {worst} at scale {scale}"
    );
}

/// AC1 / E2 (rollback state): after a run with partial and full acceptances, the hybrid cache
/// holds exactly the committed sequence — its next-token logits are a fresh prefill's — and the
/// draft's context is exactly the context a fresh draft builds from the committed rows: never a
/// rejected draft's row, and never more than the window's 7 rows (E5).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn the_rolled_back_state_is_the_committed_sequence() {
    let model = tiny_target();
    let cfg = config(20, 7);
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8)) {
        let target = TappedQwen35::for_draft(&model, &draft);
        for depth in [2usize, 5] {
            let depth_label = format!("{format} depth {depth}");
            let mut proposer = BlockDraftProposer::new(&draft);
            let (run, mut cache) = run_block(&target, &mut proposer, &PROMPT, &cfg, depth, None);
            assert_eq!(run.output.finish_reason, FinishReason::MaxTokens);
            assert!(run.stats.direct_rollbacks > 0 || run.stats.accepted > 0);
            let fed = run.committed_cache_len as usize;
            let mut committed = PROMPT.to_vec();
            committed.extend(&run.output.tokens);
            assert_eq!(cache.offset() as usize, fed, "{depth_label}");
            assert_eq!(fed, committed.len() - 1, "every token but the last was fed");

            // Target: the next step on the rolled-back cache is a fresh prefill's.
            let last = *committed.last().unwrap();
            let next = model
                .decode_logits(&input_ids(&[last]), &mut cache, fed as i32)
                .unwrap();
            let mut fresh = model.new_cache();
            let want = model
                .decode_logits(&input_ids(&committed), &mut fresh, 0)
                .unwrap();
            assert_close(&next, &want, &format!("{depth_label}: target logits"));

            // Draft: the context a fresh draft builds from the committed rows.
            let context = proposer.context().unwrap();
            assert_eq!(context.next_position() as usize, fed, "{depth_label}");
            assert_eq!(context.rows(), 7, "{depth_label}: the window's rows");
            let mut fresh = model.new_cache();
            let (taps, _) = model
                .forward_tapped(
                    &input_ids(&committed[..fed]),
                    &mut fresh,
                    0,
                    &draft.config().target_layer_ids,
                    None,
                    false,
                )
                .unwrap();
            let mut rebuilt = draft.new_context(0);
            draft.append_context(&mut rebuilt, &taps).unwrap();
            assert_eq!(rebuilt.next_position(), context.next_position());
            for (i, (got, want)) in context.arrays().zip(rebuilt.arrays()).enumerate() {
                assert_close(got, want, &format!("{depth_label}: context array {i}"));
            }
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::the_rolled_back_state_is_the_committed_sequence"
    );
}

/// The exact target distribution of a token after `history`, under the request's knobs and mask.
fn target_dist(
    model: &Qwen35Model,
    history: &[i32],
    params: SamplingParams,
    mask: Option<&[bool]>,
) -> Vec<(i32, f64)> {
    let mut cache = model.new_cache();
    let logits = model
        .decode_logits(&input_ids(history), &mut cache, 0)
        .unwrap();
    let mut sampler = MlxSampler::new(params, 0);
    let dist = sampler.distribution(&logits, history, mask).unwrap();
    let total: f64 = dist.iter().map(|&(_, w)| f64::from(w)).sum();
    dist.into_iter()
        .filter(|&(_, w)| w > 0.0)
        .map(|(t, w)| (t, f64::from(w) / total))
        .collect()
}

/// AC1 (sampled): for each format, the block draft's committed tokens follow the target's
/// distribution exactly —
/// the joint law of the first three tokens over thousands of seeded runs, of which the second
/// and third come from one two-draft verify step, matches the target's own chain rule by a
/// chi-square test (cells expected fewer than 5 times pooled), with the constraint mask and
/// without it. The target's head is random (non-periodic) and the draft scores 32 candidates per
/// row, so its top-4 drafts overlap the target's top-4 often enough to be accepted.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn sampled_block_draft_keeps_the_target_distribution() {
    let model = tiny_random_target();
    let q8 = Some(BlockDraftPrecision::Q8);
    let mut dspine_cfg = dspine::tests::tiny_config();
    dspine_cfg.head_topk = 32;
    let mut dflash2_cfg = dflash2::tests::tiny_config();
    dflash2_cfg.head_topk = 32;
    for (format, draft) in [
        (
            "dspine",
            BlockDraft::DSpine(dspine::tests::tiny_draft_with(dspine_cfg, q8)),
        ),
        (
            "dflash2",
            BlockDraft::DFlash2(dflash2::tests::tiny_draft_with(dflash2_cfg, q8)),
        ),
    ] {
        let target = TappedQwen35::for_draft(&model, &draft);
        let prompt = &PROMPT[..6];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        for forbidden in [&[][..], &[1usize, 9][..]] {
            let mask = Forbid::new(forbidden).allow;
            let masked = (!forbidden.is_empty()).then_some(mask.as_slice());
            // The exact joint law of (x0, x1, x2).
            let mut expected: HashMap<[i32; 3], f64> = HashMap::new();
            for (x0, p0) in target_dist(&model, prompt, params, masked) {
                let h0: Vec<i32> = prompt.iter().copied().chain([x0]).collect();
                for (x1, p1) in target_dist(&model, &h0, params, masked) {
                    let h1: Vec<i32> = h0.iter().copied().chain([x1]).collect();
                    for (x2, p2) in target_dist(&model, &h1, params, masked) {
                        expected.insert([x0, x1, x2], p0 * p1 * p2);
                    }
                }
            }
            let n = 2000u64;
            let mut counts: HashMap<[i32; 3], u64> = HashMap::new();
            let (mut proposed, mut accepted) = (0u64, 0u64);
            for seed in 0..n {
                let mut cfg = config(4, seed);
                cfg.sampling = params;
                let mut constraint = Forbid::new(forbidden);
                let mut proposer = BlockDraftProposer::new(&draft);
                let (run, _) = run_block(
                    &target,
                    &mut proposer,
                    prompt,
                    &cfg,
                    2,
                    (!forbidden.is_empty()).then_some(&mut constraint),
                );
                let t = &run.output.tokens;
                assert!(t.iter().all(|&x| !forbidden.contains(&(x as usize))));
                *counts.entry([t[0], t[1], t[2]]).or_default() += 1;
                proposed += run.report.proposed_tokens;
                accepted += run.report.accepted_tokens;
            }
            assert!(
                proposed > 0 && accepted > 0,
                "{format}: {proposed} proposed, {accepted} accepted"
            );
            // Pool the cells expected fewer than 5 times; every observed cell must be possible.
            for key in counts.keys() {
                assert!(
                    expected.contains_key(key),
                    "{key:?} has target probability 0"
                );
            }
            let (mut chi2, mut cells, mut pooled_e, mut pooled_o) =
                (0.0f64, 0usize, 0.0f64, 0.0f64);
            for (key, p) in &expected {
                let e = p * n as f64;
                let o = *counts.get(key).unwrap_or(&0) as f64;
                if e < 5.0 {
                    pooled_e += e;
                    pooled_o += o;
                } else {
                    chi2 += (o - e).powi(2) / e;
                    cells += 1;
                }
            }
            if pooled_e > 0.0 {
                chi2 += (pooled_o - pooled_e).powi(2) / pooled_e.max(1e-9);
                cells += 1;
            }
            // 99.9 % critical values of chi-square with `cells - 1` degrees of freedom, bounded by
            // the Wilson–Hilferty approximation.
            let k = (cells - 1) as f64;
            let z = 3.09;
            let critical = k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3);
            assert!(
            chi2 < critical,
            "{format} forbidden {forbidden:?}: chi-square {chi2:.1} over {cells} cells (critical {critical:.1})"
        );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::sampled_block_draft_keeps_the_target_distribution"
    );
}
