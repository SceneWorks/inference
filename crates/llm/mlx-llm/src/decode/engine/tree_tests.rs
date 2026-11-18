//! Tree verification on the MLX engine (feature SW-F328, story SW-6914): the tiny random Causal
//! target and the synthetic Qwen35 hybrid verify bounded candidate trees in one forward and keep
//! only the accepted branch. Every test here runs learned forwards, so each is an explicit
//! model-proof workflow case (`scripts/ci/model-inference-proofs.json`), never ordinary CI.

use super::tests::{causal, qwen35};
use super::*;
use crate::decode::proposers::NgramProposer;
use crate::decode::stream::{generate_with, ConstraintMask};
use crate::primitives::sampler::TokenRng;

const PROMPT: [i32; 12] = [3, 9, 4, 11, 3, 9, 4, 11, 3, 9, 4, 11];

/// Logit / state agreement a different kernel shape may round to (bf16 activations): the tree
/// forward runs a masked SDPA over the packed rows and a batched recurrence per depth, the direct
/// decode one row at a time.
const TOL: f32 = 6e-2;

fn host(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shapes agree");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn greedy(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(),
        seed: Some(7),
        stop_tokens: Vec::new(),
    }
}

/// The tree shapes every branch is checked on, up to [`core_llm::MAX_TREE_ROWS`] candidates: a
/// chain, a star, two long branches, a complete binary tree and random trees. Tokens are drawn
/// from `vocab` with a fixed seed.
fn shapes(vocab: i32) -> Vec<(&'static str, DraftTree)> {
    let mut rng = SplitMix64::new(0x6914);
    let mut token = move || (rng.next_u64() % vocab as u64) as i32;
    let build = |parents: &[usize], token: &mut dyn FnMut() -> i32| {
        let mut t = DraftTree::new();
        for &p in parents {
            t.push(p, token(), None).unwrap();
        }
        t
    };
    let chain: Vec<usize> = (0..16).collect();
    let star = vec![0; 16];
    let mut two = vec![0, 0];
    two.extend((0..7).map(|i| if i == 0 { 1 } else { 2 * i + 1 }));
    two.extend((0..7).map(|i| if i == 0 { 2 } else { 2 * i + 2 }));
    let binary: Vec<usize> = (1..=15).map(|r| (r - 1) / 2).collect();
    let mut out = vec![
        ("chain", build(&chain, &mut token)),
        ("star", build(&star, &mut token)),
        ("two branches", build(&two, &mut token)),
        ("binary", build(&binary, &mut token)),
    ];
    let mut shape_rng = SplitMix64::new(0x5eed_6914);
    for name in ["random a", "random b", "random c"] {
        let parents: Vec<usize> = (1..=16)
            .map(|r| (shape_rng.next_u64() % r as u64) as usize)
            .collect();
        out.push((name, build(&parents, &mut token)));
    }
    for (name, tree) in &out {
        assert!(tree.len() <= core_llm::MAX_TREE_ROWS as usize, "{name}");
    }
    out
}

/// A target's comparable state after a step: every layer's cached keys/values (and the hybrid's
/// recurrent + conv states), flattened.
type States<C> = fn(&C) -> Vec<Vec<f32>>;

fn causal_states(cache: &ContiguousKvCache) -> Vec<Vec<f32>> {
    (0..KvCache::num_layers(cache))
        .filter_map(|l| cache.peek(l).unwrap())
        .flat_map(|(k, v)| [host(&k), host(&v)])
        .collect()
}

fn hybrid_states(cache: &Qwen35Cache) -> Vec<Vec<f32>> {
    cache
        .delta_states()
        .into_iter()
        .chain(cache.attn_states())
        .flat_map(|(a, b)| [a, b])
        .collect()
}

/// A fresh cache holding `PROMPT`.
fn prefilled<T: SpeculativeTarget>(target: &T) -> T::Cache {
    let mut cache = target.new_cache();
    target
        .forward(&mut cache, &input_ids(&PROMPT), 0, LogitsScope::Last, false)
        .unwrap();
    cache
}

/// Feed `tokens` one at a time after `PROMPT` (the direct decode of a branch), returning each
/// step's logits row and the cache.
fn direct<T: SpeculativeTarget>(target: &T, tokens: &[i32]) -> (Vec<Vec<f32>>, T::Cache) {
    let mut cache = prefilled(target);
    let rows = tokens
        .iter()
        .map(|&t| {
            let pos = target.cache_len(&cache);
            let out = target
                .forward(&mut cache, &input_ids(&[t]), pos, LogitsScope::Last, false)
                .unwrap();
            host(&out.logits)
        })
        .collect();
    (rows, cache)
}

/// Every row's logits of a tree forward after `PROMPT`, with the cache still holding all rows.
fn tree_forward<T: SpeculativeTarget>(
    target: &T,
    tree: &DraftTree,
    anchor: i32,
) -> (Vec<Vec<f32>>, T::Cache, i32) {
    let mut cache = prefilled(target);
    let base = target.cache_len(&cache);
    let inputs = TreeInputs::new(tree, anchor, base, base, target.tree_mask_dtype()).unwrap();
    let out = target.forward_tree(&mut cache, &inputs, false).unwrap();
    let rows = (0..tree.rows() as i32)
        .map(|r| host(&logits_row(&out.logits, r).unwrap()))
        .collect();
    (rows, cache, base)
}

/// The greedy choice agrees, or the direct decode's top two are a numerical near-tie (E4: a
/// kernel-shape rounding difference, not an algorithmic error).
fn assert_same_choice(tree: &[f32], direct: &[f32], what: &str) {
    let argmax = |x: &[f32]| {
        x.iter().enumerate().fold(
            (0, f32::NEG_INFINITY),
            |b, (i, &v)| if v > b.1 { (i, v) } else { b },
        )
    };
    let (t, _) = argmax(tree);
    let (d, top) = argmax(direct);
    if t != d {
        let gap = top - direct[t];
        assert!(
            gap <= TOL,
            "{what}: tree picks {t}, direct {d}, a {gap} gap is not a near-tie"
        );
    }
}

fn check_every_branch<T: SpeculativeTarget>(
    name: &str,
    target: &T,
    vocab: i32,
    states: States<T::Cache>,
) {
    let anchor = 5;
    for (shape, tree) in shapes(vocab) {
        for end in 0..tree.rows() {
            let path = tree.path(end);
            let what = format!("{name} {shape} branch to row {end}");
            let mut tokens = vec![anchor];
            tokens.extend(path[1..].iter().map(|&r| tree.node(r).token));
            let (direct_rows, mut direct_cache) = direct(target, &tokens);
            let (tree_rows, mut cache, base) = tree_forward(target, &tree, anchor);
            // Every row on the branch scores exactly what its direct decode scores.
            for (i, &row) in path.iter().enumerate() {
                let d = max_diff(&tree_rows[row], &direct_rows[i]);
                assert!(d <= TOL, "{what}: row {row} logits differ by {d}");
                assert_same_choice(&tree_rows[row], &direct_rows[i], &what);
            }
            // Only the branch is kept: the cache is the direct decode's.
            target.commit_tree(&mut cache, base, &path).unwrap();
            assert_eq!(target.cache_len(&cache), base + path.len() as i32, "{what}");
            let (a, b) = (states(&cache), states(&direct_cache));
            assert_eq!(a.len(), b.len(), "{what}: layer count");
            for (layer, (x, y)) in a.iter().zip(&b).enumerate() {
                let d = max_diff(x, y);
                assert!(d <= TOL, "{what}: state {layer} differs by {d}");
            }
            // And the next request step continues identically from it.
            let pos = target.cache_len(&cache);
            let next = |c: &mut T::Cache| {
                host(
                    &target
                        .forward(c, &input_ids(&[7]), pos, LogitsScope::Last, false)
                        .unwrap()
                        .logits,
                )
            };
            let (x, y) = (next(&mut cache), next(&mut direct_cache));
            assert!(max_diff(&x, &y) <= TOL, "{what}: the next step differs");
            assert_same_choice(&x, &y, &what);
        }
    }

    // Negative control: keeping one sibling's branch is far from decoding the other sibling, so
    // the tolerance above separates a near-tie from keeping the wrong branch.
    let mut star = DraftTree::new();
    star.push(0, 1, None).unwrap();
    star.push(0, 2, None).unwrap();
    let (_, mut cache, base) = tree_forward(target, &star, anchor);
    target.commit_tree(&mut cache, base, &[0, 1]).unwrap();
    let (_, other) = direct(target, &[anchor, 2]);
    let worst = states(&cache)
        .iter()
        .zip(&states(&other))
        .map(|(x, y)| max_diff(x, y))
        .fold(0.0, f32::max);
    assert!(
        worst > 4.0 * TOL,
        "{name}: siblings are distinguishable ({worst})"
    );
}

/// AC1: on the ordinary Causal target (contiguous KV cache) and the Qwen35 hybrid (GatedDeltaNet
/// recurrent + conv state beside full attention), a tree forward over every shape up to 16
/// candidate rows scores every row as the direct decode of its branch, and committing any branch
/// leaves exactly the direct decode's cache — attention KV compacted into place, the hybrid's
/// recurrent and conv state of the branch's last row — from which the next step agrees too.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn a_tree_forward_keeps_exactly_the_accepted_branch() {
    check_every_branch("causal", &causal(), 24, causal_states);
    check_every_branch("hybrid", &qwen35(false), 50, hybrid_states);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::a_tree_forward_keeps_exactly_the_accepted_branch");
}

fn check_abort<T: SpeculativeTarget>(name: &str, target: &T, vocab: i32, states: States<T::Cache>) {
    let before = states(&prefilled(target));
    for (shape, tree) in shapes(vocab) {
        let (_, mut cache, base) = tree_forward(target, &tree, 5);
        target.abort_tree(&mut cache, base).unwrap();
        assert_eq!(target.cache_len(&cache), base, "{name} {shape}");
        assert_eq!(
            states(&cache),
            before,
            "{name} {shape}: the cache is untouched"
        );
        // A later request step runs as if the tree never happened.
        let (rows, _) = direct(target, &[6]);
        let out = target
            .forward(&mut cache, &input_ids(&[6]), base, LogitsScope::Last, false)
            .unwrap();
        assert_eq!(host(&out.logits), rows[0], "{name} {shape}: the next step");
    }
}

/// A proposer that cancels the request while it proposes a branching tree: the engine must stop
/// before any tree row reaches the cache.
struct CancelWhileProposing<'c>(&'c CancelFlag);

impl<T: SpeculativeTarget> Proposer<T> for CancelWhileProposing<'_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }
    fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
        Ok(None)
    }
    fn propose(
        &mut self,
        _: &T,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        unreachable!("a tree run asks for a tree")
    }
    fn propose_tree(
        &mut self,
        _: &T,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<DraftTree> {
        self.0.cancel();
        let mut t = DraftTree::new();
        t.push(0, 1, None).unwrap();
        t.push(0, 2, None).unwrap();
        Ok(t)
    }
    fn commit(&mut self, _: &T, _: i32, _: &[i32], _: Option<&Array>, _: i32) -> Result<()> {
        Ok(())
    }
}

fn check_cancel<T: SpeculativeTarget>(name: &str, target: &T, states: States<T::Cache>) {
    let cancel = CancelFlag::new();
    let mut cache = prefilled(target);
    let logits = {
        let mut probe = target.new_cache();
        target
            .forward(&mut probe, &input_ids(&PROMPT), 0, LogitsScope::Last, false)
            .unwrap()
            .logits
    };
    let run = generate_speculative(
        target,
        &mut CancelWhileProposing(&cancel),
        SpeculativePrompt::Prefilled {
            cache: &mut cache,
            logits,
            hidden: None,
            history: &PROMPT,
            position_delta: 0,
        },
        &greedy(8),
        4,
        &cancel,
        &mut |_| {},
        EngineOptions {
            tree_rows: 8,
            pipelining: Pipelining::Off,
            ..EngineOptions::default()
        },
    )
    .unwrap();
    assert_eq!(run.output.finish_reason, FinishReason::Cancelled, "{name}");
    assert_eq!(
        run.output.tokens.len(),
        1,
        "{name}: only the prefill's token"
    );
    assert_eq!(target.cache_len(&cache), run.committed_cache_len, "{name}");
    assert_eq!(
        states(&cache),
        states(&prefilled(target)),
        "{name}: no tree row was written"
    );
}

/// E6: an aborted tree forward (a failed decision or forward) returns either target's cache to
/// the step start bit for bit, and a request cancelled while its tree is being proposed stops
/// before any row is written — no cross-branch or cross-request state survives.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn an_aborted_or_cancelled_tree_step_leaves_a_clean_cache() {
    check_abort("causal", &causal(), 24, causal_states);
    check_abort("hybrid", &qwen35(false), 50, hybrid_states);
    check_cancel("causal", &causal(), causal_states);
    check_cancel("hybrid", &qwen35(false), hybrid_states);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::an_aborted_or_cancelled_tree_step_leaves_a_clean_cache");
}

/// Prompt lookup's tree with random sibling candidates added under random rows, so most verify
/// passes branch and some candidates are rejected.
struct NoisyTree {
    rng: SplitMix64,
    vocab: i32,
}

impl<T: SpeculativeTarget> Proposer<T> for NoisyTree {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }
    fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
        Ok(None)
    }
    fn propose(
        &mut self,
        _: &T,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        unreachable!("a tree run asks for a tree")
    }
    fn propose_tree(
        &mut self,
        _: &T,
        ctx: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<DraftTree> {
        let mut tree =
            core_llm::ngram_propose_tree(ctx.history, 3, ctx.max_drafts, ctx.max_rows / 2);
        let depths = |t: &DraftTree| t.depths();
        while tree.len() < ctx.max_rows {
            let row = (self.rng.next_u64() % tree.rows() as u64) as usize;
            if depths(&tree)[row] as usize >= ctx.max_drafts {
                continue;
            }
            let token = (self.rng.next_u64() % self.vocab as u64) as i32;
            tree.push(row, token, None).unwrap();
        }
        Ok(tree)
    }
    fn commit(&mut self, _: &T, _: i32, _: &[i32], _: Option<&Array>, _: i32) -> Result<()> {
        Ok(())
    }
}

struct Forbid(Vec<bool>, Vec<i32>);

impl ConstraintMask for Forbid {
    fn allowed(&mut self) -> &[bool] {
        &self.0
    }
    fn accept(&mut self, token: i32) {
        self.1.push(token);
    }
}

impl RewindableConstraintMask for Forbid {
    fn checkpoint(&self) -> usize {
        self.1.len()
    }
    fn rewind(&mut self, checkpoint: usize) {
        self.1.truncate(checkpoint);
    }
}

fn forbid(vocab: usize) -> Forbid {
    let mut allow = vec![true; vocab];
    allow[1] = false;
    allow[9] = false;
    Forbid(allow, Vec::new())
}

fn check_greedy<T: SpeculativeTarget + crate::decode::Decode>(name: &str, target: &T, vocab: i32) {
    let mut penalized = greedy(24);
    penalized.sampling.repetition_penalty = 1.3;
    penalized.sampling.repetition_context = 16;
    penalized.sampling.presence_penalty = 0.2;
    // A stop token the plain loop first emits at its fifth token or later, so the run takes
    // several tree passes before it stops (an EOS inside an accepted branch).
    let reference = plain(target, &greedy(24), None).tokens;
    let stop = (4..reference.len()).find(|&i| !reference[..i].contains(&reference[i]));
    let mut cases = vec![
        ("greedy", greedy(24), false),
        ("budget", greedy(7), false),
        ("penalized", penalized, false),
        ("constrained", greedy(24), true),
    ];
    if let Some(i) = stop {
        let mut stopping = greedy(24);
        stopping.stop_tokens = vec![reference[i]];
        cases.push(("stop token", stopping, false));
    }
    for (case, config, constrained) in cases {
        let what = format!("{name} {case}");
        let (mut a, mut b) = (forbid(vocab as usize), forbid(vocab as usize));
        let expected = plain(target, &config, constrained.then_some(&mut a));
        let run = generate_speculative(
            target,
            &mut NoisyTree {
                rng: SplitMix64::new(0x7e3e),
                vocab,
            },
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            4,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                constraint: constrained.then_some(&mut b as &mut dyn RewindableConstraintMask),
                tree_rows: 16,
                ..EngineOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            run.output.tokens, expected.tokens,
            "{what}: tree run != plain loop"
        );
        assert_eq!(run.output.finish_reason, expected.finish_reason, "{what}");
        if constrained {
            assert_eq!(
                b.1, a.1,
                "{what}: the constraint advanced only along accepted tokens"
            );
        }
        let tree = run.report.tree_verify.expect("a tree run reports its tree");
        assert_eq!(tree.max_rows, 16, "{what}");
        assert!(tree.tree_passes > 0, "{what}: verified branching trees");
        assert_eq!(tree.passes, run.report.verify_steps, "{what}");
        assert!(
            tree.max_rows_per_pass <= 17,
            "{what}: anchor + 16 rows at most"
        );
        assert!(tree.peak_workspace_bytes > 0, "{what}");
        assert_eq!(tree.accepted_tokens, run.report.accepted_tokens, "{what}");
        assert!(
            run.report.proposed_tokens > run.report.accepted_tokens,
            "{what}: rejections recovered"
        );
        if case == "greedy" {
            assert!(tree.accepted_tokens > 0, "{what}: branches were accepted");
        }
    }
}

fn plain<D: crate::decode::Decode>(
    model: &D,
    config: &GenerationConfig,
    constraint: Option<&mut Forbid>,
) -> GenerationOutput {
    generate_with(
        model,
        &PROMPT,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        constraint.map(|c| c as &mut dyn ConstraintMask),
        None,
    )
    .unwrap()
}

/// AC2 (greedy) and E4: with branching trees of up to 16 rows — prompt lookup's matches plus
/// random siblings, so candidates are accepted and rejected — a greedy tree run emits exactly the
/// plain token-at-a-time loop's tokens on both targets: plain greedy, under penalties, under a
/// constraint (advanced only by committed tokens), at a token budget and on a stop token inside
/// an accepted branch; and the report counts the tree passes, rows and acceptance.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn greedy_tree_runs_are_the_plain_loop() {
    check_greedy("causal", &causal(), 24);
    check_greedy("hybrid", &qwen35(false), 50);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::greedy_tree_runs_are_the_plain_loop");
}

/// A chi-square goodness-of-fit statistic of `counts` against probabilities `p`, pooling cells
/// expected below five, with the 99.9 % critical value for its degrees of freedom.
fn chi_square(counts: &[u64], p: &[f64]) -> (f64, f64) {
    let n: u64 = counts.iter().sum();
    let (mut chi2, mut cells, mut pooled_c, mut pooled_e) = (0.0, 0usize, 0.0, 0.0);
    for (&c, &pt) in counts.iter().zip(p) {
        let e = n as f64 * pt;
        if e < 5.0 {
            assert!(
                pt > 0.0 || c == 0,
                "a token the target excludes was committed"
            );
            pooled_c += c as f64;
            pooled_e += e;
            continue;
        }
        chi2 += (c as f64 - e).powi(2) / e;
        cells += 1;
    }
    if pooled_e >= 5.0 {
        chi2 += (pooled_c - pooled_e).powi(2) / pooled_e;
        cells += 1;
    }
    let df = cells.saturating_sub(1).max(1) as f64;
    // Wilson–Hilferty: the 99.9 % quantile (z = 3.09) of chi-square with `df` degrees.
    let z = 3.09;
    let crit = df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3);
    (chi2, crit)
}

fn dense(dist: &[(i32, f32)], vocab: usize) -> Vec<f64> {
    let total: f64 = dist.iter().map(|x| f64::from(x.1)).sum();
    let mut p = vec![0.0; vocab];
    for &(t, w) in dist {
        p[t as usize] += f64::from(w) / total;
    }
    p
}

/// AC2 (sampled): through a real tree verify of the Causal target, the committed tokens follow
/// the target distribution. Deterministic siblings (a block draft's top-K candidates) at a fixed
/// tree: the first committed token follows the anchor row's shaped distribution and, after an
/// accepted candidate, the next follows that candidate's row. Siblings drawn from their own `q`,
/// redrawn every trial (one tree forward each, then undone): the first committed token still
/// follows the anchor row's distribution, the multi-draft rejection rule recovering every
/// rejection.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn sampled_tree_acceptance_preserves_the_target_distribution() {
    let target = causal();
    let vocab = 24usize;
    let config = GenerationConfig {
        max_new_tokens: 8,
        sampling: SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            ..SamplingParams::default()
        },
        seed: Some(1),
        stop_tokens: Vec::new(),
    };
    let anchor = 5;
    let mut history = PROMPT.to_vec();
    history.push(anchor);

    // Deterministic siblings, one fixed tree forward, many walks.
    let mut tree = DraftTree::new();
    for t in [3, 9, 4] {
        tree.push(0, t, None).unwrap();
    }
    tree.push(1, 11, None).unwrap();
    tree.push(1, 2, None).unwrap();
    let mut cache = prefilled(&target);
    let base = cache.offset();
    let inputs = TreeInputs::new(&tree, anchor, base, base, target.tree_mask_dtype()).unwrap();
    let out = target.forward_tree(&mut cache, &inputs, false).unwrap();
    eval([&out.logits]).unwrap();
    let mut sampler = MlxSampler::new(config.sampling, 0x6914);
    let p0 = dense(
        &sampler
            .distribution(&logits_row(&out.logits, 0).unwrap(), &history, None)
            .unwrap(),
        vocab,
    );
    let mut after_3 = history.clone();
    after_3.push(3);
    let p1 = dense(
        &sampler
            .distribution(&logits_row(&out.logits, 1).unwrap(), &after_3, None)
            .unwrap(),
        vocab,
    );
    let (mut first, mut second) = (vec![0u64; vocab], vec![0u64; vocab]);
    for _ in 0..20_000 {
        let walk = tree_walk(&out.logits, &tree, &history, &config, &mut sampler, None).unwrap();
        first[walk.committed[0] as usize] += 1;
        if walk.path.get(1) == Some(&1) {
            second[walk.committed[1] as usize] += 1;
        }
    }
    let (chi2, crit) = chi_square(&first, &p0);
    assert!(
        chi2 < crit,
        "point-mass siblings, first token: chi-square {chi2:.1} >= {crit:.1}"
    );
    let (chi2, crit) = chi_square(&second, &p1);
    assert!(
        chi2 < crit,
        "after an accepted candidate: chi-square {chi2:.1} >= {crit:.1}"
    );
    target.abort_tree(&mut cache, base).unwrap();

    // Siblings drawn from their own q, redrawn per trial: one tree forward each, then undone.
    let q1: Vec<(i32, f32)> = (0..vocab as i32)
        .map(|t| (t, 1.0 + (t % 5) as f32))
        .collect();
    let q2: Vec<(i32, f32)> = (0..vocab as i32)
        .map(|t| (t, if t < 6 { 4.0 } else { 0.5 }))
        .collect();
    let mut draws = SplitMix64::new(0xd4a7);
    let mut counts = vec![0u64; vocab];
    for _ in 0..3_000 {
        let mut tree = DraftTree::new();
        for q in [&q1, &q2] {
            let t = core_llm::speculative::sample_weighted(q, draws.next_f32(), 0);
            tree.push(0, t, Some(q.clone())).unwrap();
        }
        let inputs = TreeInputs::new(&tree, anchor, base, base, target.tree_mask_dtype()).unwrap();
        let out = target.forward_tree(&mut cache, &inputs, false).unwrap();
        let walk = tree_walk(&out.logits, &tree, &history, &config, &mut sampler, None).unwrap();
        counts[walk.committed[0] as usize] += 1;
        target.abort_tree(&mut cache, base).unwrap();
    }
    let (chi2, crit) = chi_square(&counts, &p0);
    assert!(
        chi2 < crit,
        "sampled siblings: chi-square {chi2:.1} >= {crit:.1}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::sampled_tree_acceptance_preserves_the_target_distribution");
}

/// A proposer whose tree path must never run.
struct LinearOnly(NgramProposer);

impl<T: SpeculativeTarget> Proposer<T> for LinearOnly {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }
    fn warm(&mut self, t: &T, p: &[i32], h: Option<&Array>) -> Result<Option<Array>> {
        self.0.warm(t, p, h)
    }
    fn propose(
        &mut self,
        t: &T,
        ctx: &ProposeContext<'_>,
        s: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        self.0.propose(t, ctx, s)
    }
    fn commit(&mut self, t: &T, cur: i32, a: &[i32], h: Option<&Array>, p: i32) -> Result<()> {
        Proposer::<T>::commit(&mut self.0, t, cur, a, h, p)
    }
}

fn check_linear<T: SpeculativeTarget>(name: &str, target: &T) {
    let run = |tree_rows: usize, proposer: &mut dyn Proposer<T>| {
        generate_speculative(
            target,
            proposer,
            SpeculativePrompt::Tokens(&PROMPT),
            &greedy(20),
            4,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                tree_rows,
                ..EngineOptions::default()
            },
        )
        .unwrap()
    };
    let linear = run(0, &mut NgramProposer::default());
    assert_eq!(
        linear.report.tree_verify, None,
        "{name}: a linear run reports no tree"
    );
    // A one-branch tree takes the linear path: same tokens, same verify passes, no tree forward.
    let chained = run(8, &mut LinearOnly(NgramProposer::default()));
    assert_eq!(chained.output.tokens, linear.output.tokens, "{name}");
    assert_eq!(
        chained.report.verify_steps, linear.report.verify_steps,
        "{name}"
    );
    assert_eq!(
        chained.report.accepted_tokens, linear.report.accepted_tokens,
        "{name}"
    );
    let tree = chained.report.tree_verify.expect("a tree run reports");
    assert_eq!(
        tree.tree_passes, 0,
        "{name}: chains never run a tree forward"
    );
}

/// AC3: a linear request runs no tree and reports none; a tree request whose proposer drafts one
/// branch verifies through the unchanged linear path, token for token and pass for pass.
#[ignore = "model inference; explicit model-proof workflow"]
#[test]
fn linear_requests_and_one_branch_trees_keep_the_linear_path() {
    check_linear("causal", &causal());
    check_linear("hybrid", &qwen35(false));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::linear_requests_and_one_branch_trees_keep_the_linear_path");
}
