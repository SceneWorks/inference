//! The block-draft proposer's model-inference proofs (SW-7193, SW-7194): a tiny random draft of
//! each format — DSpine and DFlash2 — over a tiny random `qwen3_5` target through the engine's
//! linear verifier. Every test that runs learned forwards is ignored in ordinary CI and
//! registered for the explicit model-proof workflow; the block-length rule is plain arithmetic.

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

/// A non-causal draft runs its trained block whatever the depth, shortened only at the token
/// budget's end; a causal one runs the anchor and the rows it proposes.
#[test]
fn a_non_causal_draft_runs_its_trained_block_and_a_causal_one_its_drafts() {
    // DFlash2's published block of 8 (anchor + 7) at the default depth 4 and a long budget.
    assert_eq!(block_len(false, 8, 4, 100), 8);
    assert_eq!(block_len(false, 8, 1, 100), 8);
    assert_eq!(block_len(false, 8, 7, 7), 8);
    // The budget's end: 2 drafts left, so anchor + 2, as the reference shortens it.
    assert_eq!(block_len(false, 8, 2, 2), 3);
    assert_eq!(block_len(false, 8, 4, 5), 6);
    // Causal (DSpine): anchor + depth, whatever the budget.
    assert_eq!(block_len(true, 8, 4, 100), 5);
    assert_eq!(block_len(true, 8, 1, 1), 2);
}

/// E2 (block conditioning): a non-causal DFlash2 draft's first rows, scored alone, are exactly
/// those rows of its whole-block proposal, and change when the block is shorter (so the proposer
/// runs the trained block); a causal DSpine draft's first rows are the same in a shorter block.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_non_causal_draft_scores_its_rows_over_the_whole_block() {
    let model = tiny_target();
    for (format, draft) in tiny_drafts(None) {
        let cfg = draft.config();
        let mut cache = model.new_cache();
        let (taps, _) = model
            .forward_tapped(
                &input_ids(&PROMPT),
                &mut cache,
                0,
                &cfg.target_layer_ids,
                None,
                false,
            )
            .unwrap();
        let mut context = draft.new_context(0);
        draft.append_context(&mut context, &taps).unwrap();
        let anchor = 5;
        let embed = |len: usize| {
            let mut block = vec![anchor];
            block.resize(len, cfg.mask_token_id as i32);
            model.embed_input_ids(&input_ids(&block)).unwrap()
        };
        let propose = |len: usize, n: usize| {
            let (cand, table) = draft
                .propose(
                    &context,
                    &embed(len),
                    n,
                    anchor,
                    &|rows| model.lm_head(rows),
                    &|ids| model.lm_head_rows(ids),
                )
                .unwrap();
            (host(&cand), host(&table))
        };
        let full = cfg.block_size;
        let worst = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let (whole_cand, whole_table) = propose(full, full - 1);
        let k = whole_cand.len() / (full - 1);
        let (first_cand, first_table) = (&whole_cand[..2 * k], &whole_table[..2 * k * k]);
        let (short_cand, short_table) = propose(3, 2);
        let diff = worst(&short_table, first_table);
        if draft.causal() {
            assert_eq!(
                short_cand, first_cand,
                "{format}: causal rows ignore later rows"
            );
            assert!(
                diff <= 1e-3,
                "{format}: a shorter causal block moved scores by {diff}"
            );
        } else {
            // The whole block's first rows, scored alone, are exactly those rows.
            let (cand, table) = propose(full, 2);
            assert_eq!(cand, first_cand, "{format}: candidates of the first rows");
            let same = worst(&table, first_table);
            assert!(
                same <= 1e-3,
                "{format}: the first rows' scores moved by {same}"
            );
            assert!(
                short_cand != first_cand || diff > 1e-3,
                "{format}: a non-causal draft's rows see the whole block"
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::a_non_causal_draft_scores_its_rows_over_the_whole_block"
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

/// The image placeholder id of the multimodal fixture (any non-mask id of the tiny vocabulary).
const IMAGE: i32 = 7;

/// A fused image + text prompt over the tiny target, prepared as the provider prepares one
/// (SW-7195, E10): `[3, 9, image × 9, 4, 11, 3]` with a `1 × 6 × 6` image grid merged by 2 — nine
/// visual rows on a 3 × 3 M-RoPE grid, so the text after the image sits 6 positions below its
/// sequence index — the visual rows' embeddings replaced by "encoder" features, and one DeepStack
/// feature set fused after layer 0. Longer than the tiny drafts' 8-row window, whose last 7 rows
/// include image rows.
struct ImagePrompt {
    ids: Vec<i32>,
    embeds: Array,
    positions: [Vec<i32>; 3],
    visual: Vec<bool>,
    deepstack: Vec<Array>,
    delta: i32,
}

impl ImagePrompt {
    fn new(model: &Qwen35Model) -> Self {
        let mut ids = vec![3, 9];
        ids.extend([IMAGE; 9]);
        ids.extend([4, 11, 3]);
        let (t, h, w, delta) = model.mrope_positions(&ids, &[[1, 6, 6]], IMAGE, 2).unwrap();
        assert_eq!(delta, -6, "the 3 × 3 grid spans 3 positions, not 9");
        let visual: Vec<bool> = ids.iter().map(|&id| id == IMAGE).collect();
        let embeds = model.embed_input_ids(&input_ids(&ids)).unwrap();
        let shape = embeds.shape().to_vec();
        let hidden = shape[2] as usize;
        let mut values = host(&embeds);
        for (row, _) in visual.iter().enumerate().filter(|(_, &v)| v) {
            for (i, x) in values[row * hidden..(row + 1) * hidden]
                .iter_mut()
                .enumerate()
            {
                *x += 0.25 + ((row * 7 + i) % 13) as f32 * 0.02;
            }
        }
        let embeds = Array::from_slice(&values, &shape)
            .as_dtype(embeds.dtype())
            .unwrap();
        let feature: Vec<f32> = (0..9 * hidden)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        let deepstack = vec![Array::from_slice(&feature, &[9, hidden as i32])];
        Self {
            ids,
            embeds,
            positions: [t, h, w],
            visual,
            deepstack,
            delta,
        }
    }

    fn positions(&self) -> [&[i32]; 3] {
        [&self.positions[0], &self.positions[1], &self.positions[2]]
    }

    /// The tapped multimodal prefill of the whole prompt: the target cache, its tapped rows (the
    /// last `keep`) and the last position's logits.
    fn prefill(
        &self,
        model: &Qwen35Model,
        layers: &[usize],
        keep: Option<usize>,
    ) -> (Qwen35Cache, Array, Array) {
        let mut cache = model.new_cache();
        let (taps, logits) = model
            .prefill_tapped_from_embeds_with_deepstack(
                &self.embeds,
                self.positions(),
                &mut cache,
                &self.visual,
                &self.deepstack,
                layers,
                keep,
            )
            .unwrap();
        (cache, taps, logits)
    }

    /// The prefix-cache shape: the first `m` rows (the image among them) prefilled into the
    /// target cache, their taps dropped as a restored prefix's are, then the text suffix fed
    /// through the tapped target at its M-RoPE text positions. Returns the cache, the suffix's
    /// tapped rows and the last logits.
    fn prefill_split(&self, target: &TappedQwen35<'_>, m: usize) -> (Qwen35Cache, Array, Array) {
        let model = target.model;
        let head = |a: &[i32]| a[..m].to_vec();
        let (t, h, w) = (
            head(&self.positions[0]),
            head(&self.positions[1]),
            head(&self.positions[2]),
        );
        let embeds = rows_prefix(&self.embeds, m);
        let mut cache = model.new_cache();
        model
            .prefill_tapped_from_embeds_with_deepstack(
                &embeds,
                [&t, &h, &w],
                &mut cache,
                &self.visual[..m],
                &self.deepstack,
                &target.layers,
                target.keep,
            )
            .unwrap();
        assert!(self.visual[m..].iter().all(|v| !v), "the suffix is text");
        let suffix = &self.ids[m..];
        assert_eq!(
            self.positions[0][m],
            m as i32 + self.delta,
            "the suffix's text positions"
        );
        let out = target
            .forward(
                &mut cache,
                &input_ids(suffix),
                m as i32 + self.delta,
                LogitsScope::Last,
                true,
            )
            .unwrap();
        (cache, out.hidden.unwrap(), out.logits)
    }
}

/// The first `m` rows of a `[1, S, ..]` array.
fn rows_prefix(a: &Array, m: usize) -> Array {
    a.take_axis(Array::arange::<i32, i32>(0, m as i32, None).unwrap(), 1)
        .unwrap()
}

/// The plain token-at-a-time greedy loop over the fused prompt — the multimodal prefill, then one
/// token per step at its M-RoPE text position — the reference an image request is held to.
fn plain_image(
    model: &Qwen35Model,
    prompt: &ImagePrompt,
    max_new_tokens: usize,
    allow: Option<&[bool]>,
) -> Vec<i32> {
    let mut sampler = MlxSampler::new(SamplingParams::default(), 0);
    let bias: Option<Array> = allow.map(|allow| {
        let b: Vec<f32> = allow
            .iter()
            .map(|&a| if a { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        Array::from_slice(&b, &[1, 1, allow.len() as i32])
    });
    let mut cache = model.new_cache();
    let mut logits = model
        .decode_logits_from_embeds_with_deepstack(
            &prompt.embeds,
            prompt.positions(),
            &mut cache,
            &prompt.visual,
            &prompt.deepstack,
        )
        .unwrap();
    let mut out = Vec::new();
    loop {
        let row = logits
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[1, 1, -1])
            .unwrap();
        let row = match &bias {
            Some(b) => mlx_rs::ops::add(&row, b).unwrap(),
            None => row,
        };
        let token = sampler.argmax_rows(&row).unwrap()[0];
        out.push(token);
        if out.len() == max_new_tokens {
            return out;
        }
        let position = cache.offset() + prompt.delta;
        logits = model
            .decode_logits(&input_ids(&[token]), &mut cache, position)
            .unwrap();
    }
}

/// The engine over the tapped target from an already-prefilled multimodal cache (unpipelined, so
/// the cache ends at the committed length).
#[allow(clippy::too_many_arguments)]
fn run_image_block<'d>(
    target: &TappedQwen35<'_>,
    proposer: &mut BlockDraftProposer<'d>,
    prompt: &ImagePrompt,
    mut cache: Qwen35Cache,
    logits: Array,
    taps: Array,
    config: &GenerationConfig,
    depth: usize,
    constraint: Option<&mut Forbid>,
) -> (SpeculativeRun, Qwen35Cache) {
    let run = generate_speculative(
        target,
        proposer,
        SpeculativePrompt::Prefilled {
            cache: &mut cache,
            logits,
            hidden: Some(taps),
            history: &prompt.ids,
            position_delta: prompt.delta,
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

/// The context a fresh draft builds from the target's own rows for `committed[..fed]` — the fused
/// prompt under its M-RoPE positions, then the continuation at its text positions — fed from
/// sequence position `from` on (`0`: cold; the prefix-cache boundary: a resumed request).
fn rebuilt_image_context(
    target: &TappedQwen35<'_>,
    draft: &BlockDraft,
    prompt: &ImagePrompt,
    committed: &[i32],
    fed: usize,
    from: usize,
) -> BlockDraftContext {
    let model = target.model;
    let p = prompt.ids.len();
    let (mut cache, taps, _) = prompt.prefill(model, &target.layers, None);
    let mut all = vec![taps];
    if fed > p {
        let (rows, _) = model
            .forward_tapped(
                &input_ids(&committed[p..fed]),
                &mut cache,
                p as i32 + prompt.delta,
                &target.layers,
                None,
                false,
            )
            .unwrap();
        all.push(rows);
    }
    let refs: Vec<&Array> = all.iter().collect();
    let taps = mlx_rs::ops::concatenate_axis(&refs, 1).unwrap();
    let taps = crate::models::block_draft::rows_from(&taps, from as i32, 1).unwrap();
    let mut context = draft.new_context(from as i32);
    draft.append_context(&mut context, &taps).unwrap();
    context
}

/// SW-7195 AC1: an image + text request runs each draft format — DSpine and DFlash2, dense and
/// 8-bit, at several depths, greedy and constrained — and commits exactly the plain loop's tokens
/// over the same fused prompt (the constraint sees the same accepted sequence): every committed
/// token is one the target verified, and the hybrid rolls back without a replay.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn greedy_image_block_draft_is_the_plain_loop() {
    let model = tiny_target();
    let prompt = ImagePrompt::new(&model);
    let mut totals: HashMap<&str, (u64, u64)> = HashMap::new();
    for precision in [None, Some(BlockDraftPrecision::Q8)] {
        for (format, draft) in tiny_drafts(precision) {
            let target = TappedQwen35::for_draft(&model, &draft);
            for depth in [1usize, 3, 5] {
                for constrained in [false, true] {
                    let label =
                        format!("{format} {precision:?} depth {depth} constrained {constrained}");
                    let mut constraint = Forbid::new(&[1, 9]);
                    let allow = constraint.allow.clone();
                    let expected =
                        plain_image(&model, &prompt, 20, constrained.then_some(allow.as_slice()));
                    let (cache, taps, logits) = prompt.prefill(&model, &target.layers, target.keep);
                    assert_eq!(
                        taps.shape()[1],
                        7,
                        "{label}: the window's rows, image rows among them"
                    );
                    let mut proposer = BlockDraftProposer::new(&draft);
                    let (run, _) = run_image_block(
                        &target,
                        &mut proposer,
                        &prompt,
                        cache,
                        logits,
                        taps,
                        &config(20, 7),
                        depth,
                        constrained.then_some(&mut constraint),
                    );
                    let first_difference = run
                        .output
                        .tokens
                        .iter()
                        .zip(&expected)
                        .position(|(x, y)| x != y);
                    assert_eq!(first_difference, None, "{label}: first differing position");
                    assert_eq!(run.output.tokens, expected, "{label}");
                    if constrained {
                        assert_eq!(constraint.accepted, expected, "{label}: constraint");
                    }
                    let r = &run.report;
                    assert_eq!(r.proposer, ProposerKind::BlockDraft, "{label}");
                    assert_eq!(r.replay_forwards, 0, "{label}: the ring never replays");
                    assert_eq!(
                        run.output.tokens.len() as u64,
                        1 + r.verify_steps + r.accepted_tokens,
                        "{label}: one committed token per pass plus the accepted drafts"
                    );
                    assert!(proposer.draft_forwards > 0, "{label}");
                    let t = totals.entry(format).or_default();
                    t.0 += r.proposed_tokens;
                    t.1 += r.accepted_tokens;
                }
            }
        }
    }
    assert_eq!(totals.len(), 2);
    for (format, (proposed, accepted)) in totals {
        println!("{format}: {proposed} proposed, {accepted} accepted on the image prompt");
        assert!(proposed > 0, "{format}: the draft proposed");
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::greedy_image_block_draft_is_the_plain_loop"
    );
}

/// SW-7195 AC2: the draft's context follows the target's positions on an image prompt.
///
/// * Cold: the warm-up reads the target's rows at the image positions — the fused rows under
///   M-RoPE, not the placeholder ids re-embedded — and the context's next position is the
///   prompt's sequence length (the draft's own 1-D RoPE runs over sequence indices, as the
///   references' draft cache offset does). After a run, the context is exactly the one a fresh
///   draft builds from the target's rows for the committed sequence, and the target's cache is
///   the committed sequence's.
/// * Prefix-cache shape: with the image inside a restored prefix the target feeds only the text
///   suffix at its M-RoPE text positions; the context starts at the boundary (SW-7193: the
///   uncached suffix), its positions stay sequence indices, the run is still the plain loop's,
///   and its final context is the fresh rebuild from the boundary.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn image_draft_context_follows_the_target_positions() {
    let model = tiny_target();
    let prompt = ImagePrompt::new(&model);
    let p = prompt.ids.len();
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8)) {
        let target = TappedQwen35::for_draft(&model, &draft);
        let expected = plain_image(&model, &prompt, 16, None);

        // The warm-up context: image rows read from the fused prefill.
        let (_, taps, _) = prompt.prefill(&model, &target.layers, target.keep);
        let mut warmed = BlockDraftProposer::new(&draft);
        warmed.warm(&target, &prompt.ids, Some(&taps)).unwrap();
        let context = warmed.context().unwrap();
        assert_eq!(
            context.next_position() as usize,
            p,
            "{format}: next position"
        );
        assert_eq!(
            context.rows(),
            7,
            "{format}: image rows count against the window"
        );
        let mut text_cache = model.new_cache();
        let (text_taps, _) = model
            .forward_tapped(
                &input_ids(&prompt.ids),
                &mut text_cache,
                0,
                &target.layers,
                target.keep,
                false,
            )
            .unwrap();
        let mut placeholder = draft.new_context(0);
        draft.append_context(&mut placeholder, &text_taps).unwrap();
        let moved = context
            .arrays()
            .zip(placeholder.arrays())
            .map(|(a, b)| {
                host(a)
                    .iter()
                    .zip(host(b))
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            })
            .fold(0.0f32, f32::max);
        assert!(
            moved > 1e-2,
            "{format}: the context reads the fused image rows, not re-embedded placeholders ({moved})"
        );

        // Cold run, then the prefix-cache shape with the boundary after the image.
        let boundary = 11;
        for from in [0usize, boundary] {
            let label = format!("{format} from {from}");
            let (cache, taps, logits) = if from == 0 {
                prompt.prefill(&model, &target.layers, target.keep)
            } else {
                prompt.prefill_split(&target, from)
            };
            if from > 0 {
                let mut resumed = BlockDraftProposer::new(&draft);
                resumed.warm(&target, &prompt.ids, Some(&taps)).unwrap();
                let context = resumed.context().unwrap();
                assert_eq!(context.next_position() as usize, p, "{label}");
                assert_eq!(
                    context.rows() as usize,
                    p - from,
                    "{label}: the uncached suffix only"
                );
            }
            let mut proposer = BlockDraftProposer::new(&draft);
            let (run, mut cache) = run_image_block(
                &target,
                &mut proposer,
                &prompt,
                cache,
                logits,
                taps,
                &config(16, 7),
                3,
                None,
            );
            assert_eq!(run.output.tokens, expected, "{label}: the plain loop");
            let fed = run.committed_cache_len as usize;
            let mut committed = prompt.ids.clone();
            committed.extend(&run.output.tokens);
            assert_eq!(cache.offset() as usize, fed, "{label}");
            assert_eq!(
                fed,
                committed.len() - 1,
                "{label}: all but the last token fed"
            );

            // Target: the next step on the rolled-back cache is the plain loop's next step.
            let last = *committed.last().unwrap();
            let next = model
                .decode_logits(&input_ids(&[last]), &mut cache, fed as i32 + prompt.delta)
                .unwrap();
            let (mut fresh, _, _) = prompt.prefill(&model, &target.layers, None);
            if fed > p {
                model
                    .decode_logits(
                        &input_ids(&committed[p..fed]),
                        &mut fresh,
                        p as i32 + prompt.delta,
                    )
                    .unwrap();
            }
            let want = model
                .decode_logits(&input_ids(&[last]), &mut fresh, fed as i32 + prompt.delta)
                .unwrap();
            assert_close(&next, &want, &format!("{label}: target logits"));

            // Draft: positions are sequence indices, and the context is the fresh rebuild.
            let context = proposer.context().unwrap();
            assert_eq!(context.next_position() as usize, fed, "{label}");
            let rebuilt = rebuilt_image_context(&target, &draft, &prompt, &committed, fed, from);
            assert_eq!(rebuilt.next_position(), context.next_position(), "{label}");
            assert_eq!(rebuilt.rows(), context.rows(), "{label}");
            for (i, (got, want)) in context.arrays().zip(rebuilt.arrays()).enumerate() {
                assert_close(got, want, &format!("{label}: context array {i}"));
            }
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::image_draft_context_follows_the_target_positions"
    );
}

// ---- Tree verification (SW-7196). ----

/// [`run_block`] under a tree request of `tree_rows` candidate rows, with an optional cancel flag.
#[allow(clippy::too_many_arguments)]
fn run_block_tree<P: Proposer<TappedQwen35<'static>> + ?Sized>(
    target: &TappedQwen35<'static>,
    proposer: &mut P,
    prompt: &[i32],
    config: &GenerationConfig,
    depth: usize,
    tree_rows: usize,
    constraint: Option<&mut Forbid>,
    cancel: &CancelFlag,
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
        cancel,
        &mut |_| {},
        EngineOptions {
            constraint: constraint.map(|c| c as &mut dyn RewindableConstraintMask),
            pipelining: Pipelining::Off,
            tree_rows,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    (run, cache)
}

/// The tiny target, leaked so a tapped target over it is `'static` (one per test process).
fn leaked_target() -> &'static Qwen35Model {
    Box::leak(Box::new(tiny_target()))
}

/// The tree report of a block-draft tree run: present, verifying branching trees with no linear
/// fallback, and counting every pass's rows and accepted tokens (AC2).
fn tree_report(run: &SpeculativeRun, label: &str) -> core_llm::TreeVerifyReport {
    let r = &run.report;
    assert!(
        r.fallbacks.iter().all(|f| !f.contains("tree")),
        "{label}: {:?}",
        r.fallbacks
    );
    let tree = r.tree_verify.expect("a tree run reports its tree");
    assert_eq!(tree.passes, r.verify_steps, "{label}: every pass counted");
    assert_eq!(tree.accepted_tokens, r.accepted_tokens, "{label}");
    assert!(
        tree.rows_verified >= tree.passes,
        "{label}: anchors counted"
    );
    assert!(
        tree.max_rows_per_pass <= 1 + u64::from(tree.max_rows),
        "{label}: at most the anchor and the bound per pass"
    );
    assert!(tree.rows_per_pass().is_some() && tree.accepted_per_pass().is_some());
    // Tree-pass-only figures: a subset of the every-pass totals, anchors counted per tree pass.
    assert!(tree.tree_passes <= tree.passes, "{label}");
    assert!(tree.tree_rows_verified <= tree.rows_verified, "{label}");
    assert!(tree.tree_accepted_tokens <= tree.accepted_tokens, "{label}");
    if tree.tree_passes > 0 {
        assert!(
            tree.tree_rows_verified > tree.tree_passes,
            "{label}: a tree pass feeds its anchor and at least one candidate"
        );
        assert!(tree.rows_per_tree_pass().is_some() && tree.accepted_per_tree_pass().is_some());
    }
    tree
}

/// SW-7196 AC1 (greedy): with tree verification on, for each format at every draft precision
/// (BF16, Q8, Q4), each greedy configuration — plain, penalized, constrained — and two depths,
/// the block draft's candidate trees commit exactly the plain loop's tokens (the constraint sees
/// the same accepted sequence); the runs verify branching trees in one forward each, accept and
/// reject candidates, and report rows and accepted tokens per pass (AC2).
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn greedy_block_draft_tree_is_the_plain_loop() {
    let model = leaked_target();
    let mut totals: HashMap<&str, (u64, u64, u64)> = HashMap::new();
    for precision in [
        None,
        Some(BlockDraftPrecision::Q8),
        Some(BlockDraftPrecision::Q4),
    ] {
        for (format, draft) in tiny_drafts(precision) {
            let draft: &'static BlockDraft = Box::leak(Box::new(draft));
            let target = TappedQwen35::for_draft(model, draft);
            for depth in [2usize, 5] {
                for (name, cfg, constrained) in [
                    ("greedy", config(24, 7), false),
                    ("penalized", penalized(24), false),
                    ("constrained", config(24, 7), true),
                ] {
                    let label = format!("{format} {precision:?} depth {depth} {name} tree");
                    let (mut a, mut b) = (Forbid::new(&[1, 9]), Forbid::new(&[1, 9]));
                    let expected = plain(model, &PROMPT, &cfg, constrained.then_some(&mut a));
                    let mut proposer = BlockDraftProposer::new(draft);
                    let (run, _) = run_block_tree(
                        &target,
                        &mut proposer,
                        &PROMPT,
                        &cfg,
                        depth,
                        16,
                        constrained.then_some(&mut b),
                        &CancelFlag::new(),
                    );
                    assert_eq!(run.output.tokens, expected.tokens, "{label}");
                    assert_eq!(b.accepted, a.accepted, "{label}: constraint");
                    assert_eq!(run.report.proposer, ProposerKind::BlockDraft, "{label}");
                    assert_eq!(run.report.replay_forwards, 0, "{label}");
                    assert!(
                        proposer.draft_forwards > 0
                            && proposer.draft_forwards <= run.report.verify_steps,
                        "{label}: at most one draft forward per target pass"
                    );
                    let tree = tree_report(&run, &label);
                    println!(
                        "BLOCK_DRAFT_TREE {label}: passes={} tree_passes={} rows_per_pass={:.2} \
                         accepted_per_pass={:.2} rows_per_tree_pass={:.2} \
                         accepted_per_tree_pass={:.2}",
                        tree.passes,
                        tree.tree_passes,
                        tree.rows_per_pass().unwrap(),
                        tree.accepted_per_pass().unwrap(),
                        tree.rows_per_tree_pass().unwrap_or(0.0),
                        tree.accepted_per_tree_pass().unwrap_or(0.0)
                    );
                    let t = totals.entry(format).or_default();
                    t.0 += tree.tree_passes;
                    t.1 += run.report.proposed_tokens;
                    t.2 += run.report.accepted_tokens;
                }
            }
        }
    }
    assert_eq!(totals.len(), 2);
    for (format, (tree_passes, proposed, accepted)) in totals {
        assert!(tree_passes > 0, "{format}: verified branching trees");
        assert!(
            accepted > 0 && proposed > accepted,
            "{format}: {proposed} proposed, {accepted} accepted"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::greedy_block_draft_tree_is_the_plain_loop"
    );
}

/// SW-7196 AC1 (state): after greedy and sampled tree runs with accepted and rejected branches,
/// the hybrid cache holds exactly the committed sequence and the draft's context is exactly the
/// context a fresh draft builds from the committed rows — a draft that only ever saw the accepted
/// path: no rejected sibling's tapped row entered it, for either format.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn block_draft_tree_context_is_the_accepted_path() {
    let model = leaked_target();
    let mut sampled = config(20, 11);
    sampled.sampling = SamplingParams {
        temperature: 1.0,
        top_k: 4,
        top_p: 1.0,
        ..SamplingParams::default()
    };
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8)) {
        let draft: &'static BlockDraft = Box::leak(Box::new(draft));
        let target = TappedQwen35::for_draft(model, draft);
        for (mode, cfg) in [("greedy", config(20, 7)), ("sampled", sampled.clone())] {
            for depth in [2usize, 5] {
                let label = format!("{format} {mode} depth {depth} tree");
                let mut proposer = BlockDraftProposer::new(draft);
                let (run, mut cache) = run_block_tree(
                    &target,
                    &mut proposer,
                    &PROMPT,
                    &cfg,
                    depth,
                    16,
                    None,
                    &CancelFlag::new(),
                );
                let tree = tree_report(&run, &label);
                assert!(tree.tree_passes > 0, "{label}: branching trees verified");
                assert!(
                    run.report.proposed_tokens > run.report.accepted_tokens,
                    "{label}: rejected candidates"
                );
                let fed = run.committed_cache_len as usize;
                let mut committed = PROMPT.to_vec();
                committed.extend(&run.output.tokens);
                assert_eq!(cache.offset() as usize, fed, "{label}");
                assert_eq!(fed, committed.len() - 1, "{label}");

                // Target: the next step on the committed cache is a fresh prefill's.
                let last = *committed.last().unwrap();
                let next = model
                    .decode_logits(&input_ids(&[last]), &mut cache, fed as i32)
                    .unwrap();
                let mut fresh = model.new_cache();
                let want = model
                    .decode_logits(&input_ids(&committed), &mut fresh, 0)
                    .unwrap();
                assert_close(&next, &want, &format!("{label}: target logits"));

                // Draft: the context of a draft that only ever saw the accepted path.
                let context = proposer.context().unwrap();
                assert_eq!(context.next_position() as usize, fed, "{label}");
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
                assert_eq!(rebuilt.next_position(), context.next_position(), "{label}");
                assert_eq!(rebuilt.rows(), context.rows(), "{label}");
                for (i, (got, want)) in context.arrays().zip(rebuilt.arrays()).enumerate() {
                    assert_close(got, want, &format!("{label}: context array {i}"));
                }
            }
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::block_draft_tree_context_is_the_accepted_path"
    );
}

/// SW-7196 AC1 (sampled): through the real tree verify, each format's candidate trees — one
/// sampled child per node drawn from its own `q`, then the most probable other tokens as point
/// masses — commit tokens that follow the target's distribution exactly: the joint law of the
/// first three tokens over thousands of seeded runs (the second and third from one tree pass of
/// depth 2) matches the target's chain rule by a chi-square test, with and without the
/// constraint mask.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn sampled_block_draft_tree_keeps_the_target_distribution() {
    let model: &'static Qwen35Model = Box::leak(Box::new(tiny_random_target()));
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
        let draft: &'static BlockDraft = Box::leak(Box::new(draft));
        let target = TappedQwen35::for_draft(model, draft);
        let prompt = &PROMPT[..6];
        let plain = SamplingParams {
            temperature: 1.0,
            top_k: 4,
            top_p: 1.0,
            ..SamplingParams::default()
        };
        // The penalized case scores every branch row against its own ancestor path (prompt plus
        // the tokens committed above it), so a branch that read the wrong history shows in the
        // chi-square.
        let penalized = SamplingParams {
            repetition_penalty: 1.6,
            presence_penalty: 0.6,
            ..plain
        };
        for (forbidden, params) in [
            (&[][..], plain),
            (&[1usize, 9][..], plain),
            (&[][..], penalized),
        ] {
            let mask = Forbid::new(forbidden).allow;
            let masked = (!forbidden.is_empty()).then_some(mask.as_slice());
            let mut expected: HashMap<[i32; 3], f64> = HashMap::new();
            for (x0, p0) in target_dist(model, prompt, params, masked) {
                let h0: Vec<i32> = prompt.iter().copied().chain([x0]).collect();
                for (x1, p1) in target_dist(model, &h0, params, masked) {
                    let h1: Vec<i32> = h0.iter().copied().chain([x1]).collect();
                    for (x2, p2) in target_dist(model, &h1, params, masked) {
                        expected.insert([x0, x1, x2], p0 * p1 * p2);
                    }
                }
            }
            let n = 2000u64;
            let mut counts: HashMap<[i32; 3], u64> = HashMap::new();
            let (mut tree_passes, mut proposed, mut accepted) = (0u64, 0u64, 0u64);
            for seed in 0..n {
                let mut cfg = config(4, seed);
                cfg.sampling = params;
                let mut constraint = Forbid::new(forbidden);
                let mut proposer = BlockDraftProposer::new(draft);
                let (run, _) = run_block_tree(
                    &target,
                    &mut proposer,
                    prompt,
                    &cfg,
                    2,
                    16,
                    (!forbidden.is_empty()).then_some(&mut constraint),
                    &CancelFlag::new(),
                );
                let t = &run.output.tokens;
                assert!(t.iter().all(|&x| !forbidden.contains(&(x as usize))));
                *counts.entry([t[0], t[1], t[2]]).or_default() += 1;
                tree_passes += tree_report(&run, format).tree_passes;
                proposed += run.report.proposed_tokens;
                accepted += run.report.accepted_tokens;
            }
            assert!(tree_passes > 0, "{format}: branching trees verified");
            assert!(
                accepted > 0 && proposed > accepted,
                "{format}: {proposed} proposed, {accepted} accepted"
            );
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
            // 99.9 % critical value of chi-square with `cells - 1` degrees of freedom
            // (Wilson–Hilferty).
            let k = (cells - 1) as f64;
            let z = 3.09;
            let critical = k * (1.0 - 2.0 / (9.0 * k) + z * (2.0 / (9.0 * k)).sqrt()).powi(3);
            println!(
                "BLOCK_DRAFT_TREE_SAMPLED {format} forbidden {forbidden:?} penalties {}: chi-square {chi2:.1} \
                 over {cells} cells (critical {critical:.1}); {tree_passes} tree passes, \
                 {accepted}/{proposed} accepted",
                params.repetition_penalty != 1.0
            );
            assert!(
                chi2 < critical,
                "{format} forbidden {forbidden:?}: chi-square {chi2:.1} over {cells} cells (critical {critical:.1})"
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::sampled_block_draft_tree_keeps_the_target_distribution"
    );
}

/// A block-draft proposer that cancels the request as it hands over its tree, keeping the tree.
struct CancelOnTree<'d, 'c> {
    inner: BlockDraftProposer<'d>,
    cancel: &'c CancelFlag,
    tree: Option<DraftTree>,
}

impl<'t> Proposer<TappedQwen35<'t>> for CancelOnTree<'_, '_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::BlockDraft
    }
    fn wants_hidden(&self) -> bool {
        true
    }
    fn warm(
        &mut self,
        target: &TappedQwen35<'t>,
        prompt: &[i32],
        hidden: Option<&Array>,
    ) -> Result<Option<Array>> {
        self.inner.warm(target, prompt, hidden)
    }
    fn propose(
        &mut self,
        _: &TappedQwen35<'t>,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        unreachable!("a tree run asks for a tree")
    }
    fn propose_tree(
        &mut self,
        target: &TappedQwen35<'t>,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<DraftTree> {
        let tree = self.inner.propose_tree(target, ctx, sampler)?;
        self.cancel.cancel();
        self.tree = Some(tree.clone());
        Ok(tree)
    }
    fn commit(
        &mut self,
        target: &TappedQwen35<'t>,
        cur: i32,
        accepted: &[i32],
        hidden: Option<&Array>,
        position: i32,
    ) -> Result<()> {
        self.inner.commit(target, cur, accepted, hidden, position)
    }
}

/// The hybrid cache's comparable state: every DeltaNet layer's live state and every attention
/// layer's keys and values.
fn cache_states(cache: &Qwen35Cache) -> Vec<Vec<f32>> {
    cache
        .delta_states()
        .into_iter()
        .chain(cache.attn_states())
        .flat_map(|(a, b)| [a, b])
        .collect()
}

/// SW-7196 (abort / cancel): a request cancelled as its block-draft tree is handed over stops
/// before any tree row reaches the target cache, with the draft's context still the warm-up's;
/// and that tree, verified on the prefilled cache and aborted, returns the target cache to the
/// step start bit for bit while the draft context — which only a commit writes — is untouched.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn an_aborted_or_cancelled_block_draft_tree_leaves_both_caches_clean() {
    let model = leaked_target();
    for (format, draft) in tiny_drafts(Some(BlockDraftPrecision::Q8)) {
        let draft: &'static BlockDraft = Box::leak(Box::new(draft));
        let target = TappedQwen35::for_draft(model, draft);
        let mut prefilled = target.new_cache();
        let out = target
            .forward(
                &mut prefilled,
                &input_ids(&PROMPT),
                0,
                LogitsScope::Last,
                true,
            )
            .unwrap();
        let mut warmed = BlockDraftProposer::new(draft);
        warmed.warm(&target, &PROMPT, out.hidden.as_ref()).unwrap();
        let warm: Vec<Vec<f32>> = warmed.context().unwrap().arrays().map(host).collect();
        let start = cache_states(&prefilled);

        // Cancel while the tree is handed over.
        let cancel = CancelFlag::new();
        let mut proposer = CancelOnTree {
            inner: BlockDraftProposer::new(draft),
            cancel: &cancel,
            tree: None,
        };
        let (run, cache) = run_block_tree(
            &target,
            &mut proposer,
            &PROMPT,
            &config(8, 7),
            4,
            16,
            None,
            &cancel,
        );
        assert_eq!(
            run.output.finish_reason,
            FinishReason::Cancelled,
            "{format}"
        );
        assert_eq!(
            run.output.tokens.len(),
            1,
            "{format}: the prefill's token only"
        );
        assert_eq!(cache_states(&cache), start, "{format}: no tree row written");
        let context: Vec<Vec<f32>> = proposer
            .inner
            .context()
            .unwrap()
            .arrays()
            .map(host)
            .collect();
        assert_eq!(
            context, warm,
            "{format}: the draft context is the warm-up's"
        );
        let tree = proposer.tree.expect("the proposer built a tree");
        assert!(!tree.is_chain(), "{format}: a branching tree");

        // Verify that tree on the prefilled cache, then abort it.
        let mut cache = target.new_cache();
        let out = target
            .forward(&mut cache, &input_ids(&PROMPT), 0, LogitsScope::Last, true)
            .unwrap();
        let anchor = mlx_rs::ops::indexing::argmax_axis(&out.logits, -1, None)
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap()
            .as_slice::<i32>()[0];
        let base = cache.offset();
        let inputs = TreeInputs::new(&tree, anchor, base, base, target.tree_mask_dtype()).unwrap();
        let verified = target.forward_tree(&mut cache, &inputs, true).unwrap();
        assert_eq!(
            verified.hidden.as_ref().unwrap().shape(),
            &[1, tree.rows() as i32, draft.config().tap_width() as i32],
            "{format}: every tree row's taps"
        );
        target.abort_tree(&mut cache, base).unwrap();
        assert_eq!(cache.offset(), base, "{format}");
        assert_eq!(
            cache_states(&cache),
            start,
            "{format}: the aborted tree left nothing"
        );
        let context: Vec<Vec<f32>> = warmed.context().unwrap().arrays().map(host).collect();
        assert_eq!(context, warm, "{format}: an abort never reaches the draft");
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED decode::block_draft::tests::an_aborted_or_cancelled_block_draft_tree_leaves_both_caches_clean"
    );
}
