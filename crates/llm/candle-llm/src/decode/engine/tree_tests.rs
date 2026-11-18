//! Tree verification on the Candle engine (feature SW-F328, story SW-6915): the tiny random
//! llama-family `CausalLm` (static step KV cache) and the synthetic `qwen3_5` hybrid (Gated
//! DeltaNet recurrent + conv state beside full attention) verify bounded candidate trees in one
//! forward and keep only the accepted branch. Every test here runs learned forwards, so each is an
//! explicit model-proof workflow case (`scripts/ci/model-inference-proofs.json`, the CPU default
//! and a CUDA recipe), never ordinary CI. A `cuda` build runs them on the CUDA device the
//! production selector opens with the CUDA-graph runner switched on; any other build on the CPU.

use candle_core::{DType, Device, Tensor};
use core_llm::speculative::sample_weighted;

use super::*;
use crate::decode::graph::{graph_tally, GraphRunner, REASON_TREE_VERIFY};
use crate::decode::proposers::NgramProposer;
use crate::decode::record::tree_fallback_reason;
use crate::decode::step::{generate_step, StepOutput, TreeOutput, TREE_MODEL_UNSUPPORTED};
use crate::decode::stream::ConstraintMask;
use crate::models::qwen35::tests::text_model_on;
use crate::models::{CausalLm, Qwen35Cache};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::sampler::{SamplingParams, TokenRng};
use crate::primitives::step_kv_cache::StepKvCache;

const PROMPT: [i32; 12] = [3, 9, 4, 11, 3, 9, 4, 11, 3, 9, 4, 11];

/// The proof device: the CUDA device the production selector opens with the graph runner on (its
/// own capturable stream) in a `cuda` build, else the CPU. The guard holds the graph switch.
fn proof_device() -> (Option<crate::decode::graph::CudaGraphsPolicyGuard>, Device) {
    #[cfg(feature = "cuda")]
    {
        let guard = crate::decode::graph::cuda_graphs_policy_guard(Some(true));
        let device = crate::device::select_device().expect("the CUDA proof device");
        assert!(device.is_cuda(), "a cuda build proves on the CUDA device");
        (Some(guard), device)
    }
    #[cfg(not(feature = "cuda"))]
    {
        (None, Device::Cpu)
    }
}

/// Logit / state agreement a different kernel shape may round to: the tree forward attends the
/// packed rows through an additive mask and recurs per depth level, the direct decode one row at
/// a time (f32 on the CPU; bf16 activations on CUDA, where the decode step also runs the
/// length-aware decode attention).
fn tol(device: &Device) -> f32 {
    if device.is_cpu() {
        1e-3
    } else {
        1.5e-1
    }
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "shapes agree");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn causal(device: &Device) -> CausalLm {
    let (w, cfg) = crate::models::llama::tests::tiny_qwen3(24, false, device);
    CausalLm::from_weights(&w, "", cfg).unwrap()
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
/// chain, a star, two long branches, a complete binary tree and random trees.
fn shapes(vocab: i32) -> Vec<(&'static str, DraftTree)> {
    let mut rng = SplitMix64::new(0x6915);
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
    // Two interleaved eight-deep branches: rows 1, 3, … 15 and rows 2, 4, … 16.
    let two: Vec<usize> = (1..=16usize).map(|r| r.saturating_sub(2)).collect();
    let binary: Vec<usize> = (1..=15).map(|r| (r - 1) / 2).collect();
    let mut out = vec![
        ("chain", build(&chain, &mut token)),
        ("star", build(&star, &mut token)),
        ("two branches", build(&two, &mut token)),
        ("binary", build(&binary, &mut token)),
    ];
    let mut shape_rng = SplitMix64::new(0x5eed_6915);
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

/// A cache's comparable state: every layer's cached keys/values (and the hybrid's recurrent and
/// conv states), on the host.
type States<C> = fn(&C) -> Vec<Vec<f32>>;

fn causal_states(cache: &StepKvCache) -> Vec<Vec<f32>> {
    (0..KvCache::num_layers(cache))
        .filter_map(|l| cache.layer_kv(l).unwrap())
        .flat_map(|(k, v)| [host(&k), host(&v)])
        .collect()
}

fn hybrid_states(cache: &Qwen35Cache) -> Vec<Vec<f32>> {
    cache.prefix_snapshot().unwrap().host_tensors()
}

/// A request-sized cache holding `PROMPT` (the engine's own cache constructor).
fn prefilled<M: StepModel>(model: &M) -> M::Cache {
    let mut cache = model.new_cache_for(PROMPT.len() + 48, 4).unwrap();
    model
        .forward_step(&mut cache, StepRequest::last(&PROMPT).as_prefill())
        .unwrap();
    cache
}

/// Feed `tokens` one decode step at a time after `PROMPT` (the direct decode of a branch),
/// returning each step's logits row and the cache.
fn direct<M: StepModel>(model: &M, tokens: &[i32]) -> (Vec<Vec<f32>>, M::Cache) {
    let mut cache = prefilled(model);
    let rows = tokens
        .iter()
        .map(|&t| {
            let out = model
                .forward_step(&mut cache, StepRequest::last(&[t]))
                .unwrap();
            host(&out.logits)
        })
        .collect();
    (rows, cache)
}

/// Every row's logits of a tree forward after `PROMPT`, with the cache still holding every row.
fn tree_forward<M: StepModel>(
    model: &M,
    tree: &DraftTree,
    anchor: i32,
) -> (Vec<Vec<f32>>, M::Cache, i32, TreeOutput) {
    let mut cache = prefilled(model);
    let base = cache.len();
    let out = model
        .forward_tree(
            &mut cache,
            &TreeVerify {
                tree,
                anchor,
                want_hidden: false,
            },
        )
        .unwrap();
    let rows = (0..tree.rows())
        .map(|r| host(&out.step.logits.narrow(1, r, 1).unwrap()))
        .collect();
    (rows, cache, base, out)
}

/// The greedy choice agrees, or the direct decode's top two are a numerical near-tie (E4: a
/// kernel-shape rounding difference, not an algorithmic error).
fn assert_same_choice(tree: &[f32], direct: &[f32], tol: f32, what: &str) {
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
            gap <= tol,
            "{what}: tree picks {t}, direct {d}, a {gap} gap is not a near-tie"
        );
    }
}

fn check_every_branch<M: StepModel>(name: &str, model: &M, vocab: i32, states: States<M::Cache>) {
    let tol = tol(model.device());
    let anchor = 5;
    for (shape, tree) in shapes(vocab) {
        for end in 0..tree.rows() {
            let path = tree.path(end);
            let what = format!("{name} {shape} branch to row {end}");
            let mut tokens = vec![anchor];
            tokens.extend(path[1..].iter().map(|&r| tree.node(r).token));
            let (direct_rows, mut direct_cache) = direct(model, &tokens);
            let (tree_rows, mut cache, base, out) = tree_forward(model, &tree, anchor);
            assert!(out.mask_bytes > 0, "{what}: the mask is reported");
            // Every row on the branch scores exactly what its direct decode scores.
            for (i, &row) in path.iter().enumerate() {
                let d = max_diff(&tree_rows[row], &direct_rows[i]);
                assert!(d <= tol, "{what}: row {row} logits differ by {d}");
                assert_same_choice(&tree_rows[row], &direct_rows[i], tol, &what);
            }
            // Only the branch is kept: the cache is the direct decode's.
            cache.commit_tree(base, &path).unwrap();
            assert_eq!(cache.len(), base + path.len() as i32, "{what}");
            assert_eq!(
                cache.tree_bytes(),
                0,
                "{what}: no tree state outlives the commit"
            );
            let (a, b) = (states(&cache), states(&direct_cache));
            assert_eq!(a.len(), b.len(), "{what}: layer count");
            for (layer, (x, y)) in a.iter().zip(&b).enumerate() {
                let d = max_diff(x, y);
                assert!(d <= tol, "{what}: state {layer} differs by {d}");
            }
            // And the next request step continues identically from it.
            let next = |c: &mut M::Cache| {
                host(
                    &model
                        .forward_step(c, StepRequest::last(&[7]))
                        .unwrap()
                        .logits,
                )
            };
            let (x, y) = (next(&mut cache), next(&mut direct_cache));
            assert!(max_diff(&x, &y) <= tol, "{what}: the next step differs");
            assert_same_choice(&x, &y, tol, &what);
        }
    }

    // Negative control: keeping one sibling's branch is far from decoding the other sibling, so
    // the tolerance above separates a near-tie from keeping the wrong branch.
    let mut star = DraftTree::new();
    star.push(0, 1, None).unwrap();
    star.push(0, 2, None).unwrap();
    let (_, mut cache, base, _) = tree_forward(model, &star, anchor);
    cache.commit_tree(base, &[0, 1]).unwrap();
    let (_, other) = direct(model, &[anchor, 2]);
    let worst = states(&cache)
        .iter()
        .zip(&states(&other))
        .map(|(x, y)| max_diff(x, y))
        .fold(0.0, f32::max);
    assert!(
        worst > 4.0 * tol,
        "{name}: siblings are distinguishable ({worst})"
    );
}

/// AC1 (E2): on the ordinary attention cache (`CausalLm`, static step KV) and the `qwen3_5`
/// hybrid (Gated DeltaNet recurrent + conv state beside full attention), a tree forward over
/// every shape up to 16 candidate rows scores every row as the direct decode of its branch, and
/// committing any branch leaves exactly the direct decode's cache — attention KV gathered into
/// place, the hybrid's recurrent and conv state of the branch's last row — from which the next
/// step agrees too.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_tree_forward_keeps_exactly_the_accepted_branch() {
    let (_guard, device) = proof_device();
    check_every_branch("causal", &causal(&device), 24, causal_states);
    let (_cfg, hybrid) = text_model_on(&device);
    check_every_branch("hybrid", &hybrid, 50, hybrid_states);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::a_tree_forward_keeps_exactly_the_accepted_branch");
}

fn check_abort<M: StepModel>(name: &str, model: &M, vocab: i32, states: States<M::Cache>) {
    let before = states(&prefilled(model));
    let (rows, _) = direct(model, &[6]);
    for (shape, tree) in shapes(vocab) {
        let (_, mut cache, base, _) = tree_forward(model, &tree, 5);
        cache.abort_tree(base).unwrap();
        assert_eq!(cache.len(), base, "{name} {shape}");
        assert_eq!(cache.tree_bytes(), 0, "{name} {shape}");
        assert_eq!(
            states(&cache),
            before,
            "{name} {shape}: the cache is untouched"
        );
        // A later request step runs as if the tree never happened.
        let out = model
            .forward_step(&mut cache, StepRequest::last(&[6]))
            .unwrap();
        assert_eq!(host(&out.logits), rows[0], "{name} {shape}: the next step");
    }
}

/// A proposer that cancels the request while it proposes a branching tree: the engine must stop
/// before any tree row reaches the cache.
struct CancelWhileProposing<'c>(&'c CancelFlag);

impl Proposer for CancelWhileProposing<'_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }
    fn warm(&mut self, _: &[i32], _: Option<&Tensor>) -> Result<()> {
        Ok(())
    }
    fn propose(
        &mut self,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        unreachable!("a tree run asks for a tree")
    }
    fn propose_tree(
        &mut self,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Option<DraftTree>> {
        self.0.cancel();
        let mut t = DraftTree::new();
        t.push(0, 1, None).unwrap();
        t.push(0, 2, None).unwrap();
        Ok(Some(t))
    }
    fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
        Ok(())
    }
}

/// A two-sibling tree every step.
struct Siblings;

impl Proposer for Siblings {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }
    fn warm(&mut self, _: &[i32], _: Option<&Tensor>) -> Result<()> {
        Ok(())
    }
    fn propose(
        &mut self,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        unreachable!("a tree run asks for a tree")
    }
    fn propose_tree(
        &mut self,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Option<DraftTree>> {
        let mut t = DraftTree::new();
        t.push(0, 1, None).unwrap();
        t.push(0, 2, None).unwrap();
        Ok(Some(t))
    }
    fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
        Ok(())
    }
}

/// A model that cancels the request inside its tree forward, after every row reached the cache.
struct CancelInTree<'a, M> {
    model: &'a M,
    cancel: &'a CancelFlag,
}

impl<M: StepModel> StepModel for CancelInTree<'_, M> {
    type Cache = M::Cache;
    fn new_cache(&self) -> M::Cache {
        self.model.new_cache()
    }
    fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<M::Cache> {
        self.model.new_cache_for(capacity, overshoot)
    }
    fn device(&self) -> &Device {
        self.model.device()
    }
    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }
    fn forward_step(&self, cache: &mut M::Cache, request: StepRequest<'_>) -> Result<StepOutput> {
        self.model.forward_step(cache, request)
    }
    fn tree_unsupported(&self, cache: &M::Cache) -> Option<&'static str> {
        self.model.tree_unsupported(cache)
    }
    fn forward_tree(&self, cache: &mut M::Cache, tree: &TreeVerify<'_>) -> Result<TreeOutput> {
        let out = self.model.forward_tree(cache, tree)?;
        self.cancel.cancel();
        Ok(out)
    }
}

fn prefilled_run<M: StepModel, P: Proposer>(
    model: &M,
    proposer: &mut P,
    cache: &mut M::Cache,
    cancel: &CancelFlag,
) -> SpeculativeRun {
    let logits = {
        let mut probe = model.new_cache_for(PROMPT.len() + 48, 4).unwrap();
        model
            .forward_step(&mut probe, StepRequest::last(&PROMPT).as_prefill())
            .unwrap()
            .logits
    };
    generate_speculative_monitored(
        model,
        proposer,
        SpeculativePrompt::Prefilled {
            cache,
            logits,
            hidden: None,
            history: &PROMPT,
            position_delta: 0,
            warm_proposer: true,
        },
        &greedy(8),
        4,
        cancel,
        &mut |_| {},
        None,
        None,
        None,
        Speculative::Off,
        8,
    )
    .unwrap()
}

fn check_cancel<M: StepModel>(name: &str, model: &M, states: States<M::Cache>) {
    let clean = states(&prefilled(model));
    // Cancelled while the tree is proposed: no row is ever written.
    let cancel = CancelFlag::new();
    let mut cache = prefilled(model);
    let run = prefilled_run(
        model,
        &mut CancelWhileProposing(&cancel),
        &mut cache,
        &cancel,
    );
    assert_eq!(run.output.finish_reason, FinishReason::Cancelled, "{name}");
    assert_eq!(
        run.output.tokens.len(),
        1,
        "{name}: only the prefill's token"
    );
    assert_eq!(cache.len() as usize, PROMPT.len(), "{name}");
    assert_eq!(states(&cache), clean, "{name}: no tree row was written");
    // Cancelled right after the tree forward: every row is dropped again, bit for bit.
    let cancel = CancelFlag::new();
    let wrapped = CancelInTree {
        model,
        cancel: &cancel,
    };
    let mut cache = prefilled(model);
    let run = prefilled_run(&wrapped, &mut Siblings, &mut cache, &cancel);
    assert_eq!(run.output.finish_reason, FinishReason::Cancelled, "{name}");
    assert_eq!(
        run.output.tokens.len(),
        1,
        "{name}: nothing of the tree step"
    );
    assert_eq!(run.stats.verify_steps, 1, "{name}: the tree forward ran");
    assert_eq!(cache.len() as usize, PROMPT.len(), "{name}");
    assert_eq!(cache.tree_bytes(), 0, "{name}");
    assert_eq!(states(&cache), clean, "{name}: the tree rows were dropped");
}

/// E6: an aborted tree forward (a failed decision or forward) returns either cache to the step
/// start bit for bit, and a request cancelled while its tree is proposed — or right after its
/// tree forward — commits nothing of that step and leaves no tree row or per-row state behind:
/// no cross-branch or cross-request state survives.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn an_aborted_or_cancelled_tree_step_leaves_a_clean_cache() {
    let (_guard, device) = proof_device();
    let model = causal(&device);
    let (_cfg, hybrid) = text_model_on(&device);
    check_abort("causal", &model, 24, causal_states);
    check_abort("hybrid", &hybrid, 50, hybrid_states);
    check_cancel("causal", &model, causal_states);
    check_cancel("hybrid", &hybrid, hybrid_states);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::an_aborted_or_cancelled_tree_step_leaves_a_clean_cache");
}

/// Prompt lookup's tree with random sibling candidates added under random rows, so most verify
/// passes branch and some candidates are rejected.
struct NoisyTree {
    rng: SplitMix64,
    vocab: i32,
}

impl Proposer for NoisyTree {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }
    fn warm(&mut self, _: &[i32], _: Option<&Tensor>) -> Result<()> {
        Ok(())
    }
    fn propose(
        &mut self,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        unreachable!("a tree run asks for a tree")
    }
    fn propose_tree(
        &mut self,
        ctx: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Option<DraftTree>> {
        let mut tree =
            core_llm::ngram_propose_tree(ctx.history, 3, ctx.max_drafts, ctx.max_rows / 2);
        while tree.len() < ctx.max_rows {
            let row = (self.rng.next_u64() % tree.rows() as u64) as usize;
            if tree.depths()[row] as usize >= ctx.max_drafts {
                continue;
            }
            let token = (self.rng.next_u64() % self.vocab as u64) as i32;
            tree.push(row, token, None).unwrap();
        }
        Ok(Some(tree))
    }
    fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
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

fn plain<M: StepModel>(
    model: &M,
    config: &GenerationConfig,
    constraint: Option<&mut Forbid>,
) -> GenerationOutput {
    generate_step(
        model,
        &PROMPT,
        config,
        &CancelFlag::new(),
        &mut |_| {},
        constraint.map(|c| c as &mut dyn ConstraintMask),
    )
    .unwrap()
    .0
}

fn tree_run<M: StepModel, P: Proposer>(
    model: &M,
    proposer: &mut P,
    config: &GenerationConfig,
    tree_rows: usize,
    constraint: Option<&mut dyn RewindableConstraintMask>,
) -> SpeculativeRun {
    generate_speculative_monitored(
        model,
        proposer,
        SpeculativePrompt::Tokens(&PROMPT),
        config,
        4,
        &CancelFlag::new(),
        &mut |_| {},
        constraint,
        None,
        None,
        Speculative::Off,
        tree_rows,
    )
    .unwrap()
}

fn check_greedy<M: StepModel>(name: &str, model: &M, vocab: i32) {
    let mut penalized = greedy(24);
    penalized.sampling.repetition_penalty = 1.3;
    penalized.sampling.repetition_context = 16;
    penalized.sampling.presence_penalty = 0.2;
    // A stop token the plain loop first emits at its fifth token or later, so the run takes
    // several tree passes before it stops (an EOS inside an accepted branch).
    let reference = plain(model, &greedy(24), None).tokens;
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
        let expected = plain(model, &config, constrained.then_some(&mut a));
        let run = tree_run(
            model,
            &mut NoisyTree {
                rng: SplitMix64::new(0x7e3e),
                vocab,
            },
            &config,
            16,
            constrained.then_some(&mut b as &mut dyn RewindableConstraintMask),
        );
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
        let record = run.record;
        let tree = record.tree_verify.expect("a tree run reports its tree");
        assert_eq!(record.tree_fallback, None, "{what}");
        assert_eq!(tree.max_rows, 16, "{what}");
        assert!(tree.tree_passes > 0, "{what}: verified branching trees");
        assert_eq!(tree.passes, record.verify_steps, "{what}");
        assert!(
            tree.max_rows_per_pass <= 17,
            "{what}: anchor + 16 rows at most"
        );
        assert!(tree.peak_workspace_bytes > 0, "{what}");
        assert_eq!(tree.accepted_tokens, record.accepted_tokens, "{what}");
        assert!(
            record.proposed_tokens > record.accepted_tokens,
            "{what}: rejections recovered"
        );
        // One target forward per pass: a tree pass never replays.
        assert_eq!(record.replay_forwards, 0, "{what}");
        assert_eq!(
            record.target_forwards_per_verify_step(),
            Some(1.0),
            "{what}"
        );
        let report = record.report(false).tree_verify.expect("reported");
        assert_eq!(report, tree, "{what}: the report carries the record's tree");
        if case == "greedy" {
            assert!(tree.accepted_tokens > 0, "{what}: branches were accepted");
        }
    }
}

/// AC1/E4 (greedy): with branching trees of up to 16 rows — prompt lookup's matches plus random
/// siblings, so candidates are accepted and rejected — a greedy tree run emits exactly the plain
/// token-at-a-time loop's tokens on both cache classes: plain greedy, under penalties (applied
/// along each row's own ancestor path), under a constraint (advanced only by committed tokens),
/// at a token budget and on a stop token inside an accepted branch; and the record (E8) counts the
/// passes, rows, acceptance and peak tree workspace.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn greedy_tree_runs_are_the_plain_loop() {
    let (_guard, device) = proof_device();
    check_greedy("causal", &causal(&device), 24);
    let (_cfg, hybrid) = text_model_on(&device);
    check_greedy("hybrid", &hybrid, 50);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::greedy_tree_runs_are_the_plain_loop");
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

/// AC1/E4 (sampled): through a real tree verify of the Causal target, the committed tokens follow
/// the target distribution. Deterministic siblings (prompt lookup's, a block draft's top-K) at a
/// fixed tree: the first committed token follows the anchor row's shaped distribution and, after
/// an accepted candidate, the next follows that candidate's row. Siblings drawn from their own
/// `q`, redrawn every trial (one tree forward each, then aborted): the first committed token still
/// follows the anchor row's distribution, the multi-draft rejection rule recovering every
/// rejection.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn sampled_tree_acceptance_preserves_the_target_distribution() {
    let (_guard, device) = proof_device();
    let model = causal(&device);
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
    let (_, mut cache, base, out) = tree_forward(&model, &tree, anchor);
    let logits = out.step.logits;
    let rows = crate::primitives::sampler::logits_rows_host(&logits).unwrap();
    let p0 = dense(
        &acceptance_target_host(rows[0].clone(), &history, &config.sampling, None),
        vocab,
    );
    let mut after_3 = history.clone();
    after_3.push(3);
    let p1 = dense(
        &acceptance_target_host(rows[1].clone(), &after_3, &config.sampling, None),
        vocab,
    );
    let mut rng = SplitMix64::new(0x6915);
    let (mut first, mut second) = (vec![0u64; vocab], vec![0u64; vocab]);
    for _ in 0..20_000 {
        let walk = tree_walk(&logits, &tree, &history, &config, &mut rng, false, None).unwrap();
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
    cache.abort_tree(base).unwrap();

    // Siblings drawn from their own q, redrawn per trial: one tree forward each, then aborted.
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
            let t = sample_weighted(q, draws.next_f32(), 0);
            tree.push(0, t, Some(q.clone())).unwrap();
        }
        let out = model
            .forward_tree(
                &mut cache,
                &TreeVerify {
                    tree: &tree,
                    anchor,
                    want_hidden: false,
                },
            )
            .unwrap();
        let walk = tree_walk(
            &out.step.logits,
            &tree,
            &history,
            &config,
            &mut rng,
            false,
            None,
        )
        .unwrap();
        counts[walk.committed[0] as usize] += 1;
        cache.abort_tree(base).unwrap();
    }
    let (chi2, crit) = chi_square(&counts, &p0);
    assert!(
        chi2 < crit,
        "sampled siblings: chi-square {chi2:.1} >= {crit:.1}"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::sampled_tree_acceptance_preserves_the_target_distribution");
}

/// Prompt lookup with the tree hook left at its default: the linear proposal.
struct LinearOnly(NgramProposer);

impl Proposer for LinearOnly {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }
    fn warm(&mut self, p: &[i32], h: Option<&Tensor>) -> Result<()> {
        self.0.warm(p, h)
    }
    fn propose(
        &mut self,
        ctx: &ProposeContext<'_>,
        s: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        self.0.propose(ctx, s)
    }
    fn commit(&mut self, cur: i32, a: &[i32], h: Option<&Tensor>, p: i32) -> Result<()> {
        self.0.commit(cur, a, h, p)
    }
}

/// A one-branch proposer of the draft-model kind (the MTP head behaves alike): every proposal is
/// `max_drafts` copies of `cur`, and the tree hook is left at its default.
struct OneBranch;

impl Proposer for OneBranch {
    fn kind(&self) -> ProposerKind {
        ProposerKind::DraftModel
    }
    fn warm(&mut self, _: &[i32], _: Option<&Tensor>) -> Result<()> {
        Ok(())
    }
    fn propose(
        &mut self,
        ctx: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        Ok(Proposal {
            drafts: Some(Drafts::Host(vec![ctx.cur; ctx.max_drafts])),
            dists: Vec::new(),
        })
    }
    fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
        Ok(())
    }
}

/// A model without a tree forward (the trait's defaults): a tree request on it verifies linearly.
struct NoTree<'a, M>(&'a M);

impl<M: StepModel> StepModel for NoTree<'_, M> {
    type Cache = M::Cache;
    fn new_cache(&self) -> M::Cache {
        self.0.new_cache()
    }
    fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<M::Cache> {
        self.0.new_cache_for(capacity, overshoot)
    }
    fn device(&self) -> &Device {
        self.0.device()
    }
    fn vocab_size(&self) -> usize {
        self.0.vocab_size()
    }
    fn forward_step(&self, cache: &mut M::Cache, request: StepRequest<'_>) -> Result<StepOutput> {
        self.0.forward_step(cache, request)
    }
}

fn check_linear<M: StepModel>(name: &str, model: &M) {
    let config = greedy(20);
    let linear = tree_run(model, &mut NgramProposer::default(), &config, 0, None);
    assert_eq!(
        linear.record.tree_verify, None,
        "{name}: a linear run reports no tree"
    );
    assert!(linear.record.report(false).fallbacks.is_empty(), "{name}");
    // A one-branch proposal takes the linear path: same tokens, same verify passes, no tree
    // forward.
    let chained = tree_run(
        model,
        &mut LinearOnly(NgramProposer::default()),
        &config,
        8,
        None,
    );
    assert_eq!(chained.output.tokens, linear.output.tokens, "{name}");
    assert_eq!(
        chained.stats.verify_steps, linear.stats.verify_steps,
        "{name}"
    );
    assert_eq!(chained.stats.accepted, linear.stats.accepted, "{name}");
    let tree = chained.record.tree_verify.expect("a tree run reports");
    assert_eq!(
        tree.tree_passes, 0,
        "{name}: chains never run a tree forward"
    );
    assert_eq!(tree.passes as usize, chained.stats.verify_steps, "{name}");
    // E1: a one-branch proposer (a draft model, the MTP head) under a two-row bound verifies at
    // most two candidates per pass — MLX prunes the same chain to the bound — however many drafts
    // the engine runs, and still commits the plain loop's tokens.
    let bounded = tree_run(model, &mut OneBranch, &config, 2, None);
    assert_eq!(bounded.output.tokens, linear.output.tokens, "{name}");
    let report = bounded.record.tree_verify.expect("a tree run reports");
    assert_eq!(report.max_rows, 2, "{name}");
    assert_eq!(
        report.tree_passes, 0,
        "{name}: a chain runs no tree forward"
    );
    assert_eq!(
        report.max_rows_per_pass, 3,
        "{name}: the anchor and at most two candidates per pass"
    );
    // E5: a model that cannot keep a tree branch verifies linearly — the same tokens — and the
    // report names why.
    let refused = tree_run(
        &NoTree(model),
        &mut NgramProposer::default(),
        &config,
        8,
        None,
    );
    assert_eq!(refused.output.tokens, linear.output.tokens, "{name}");
    assert_eq!(refused.record.tree_verify, None, "{name}");
    assert_eq!(refused.record.tree_fallback, Some(TREE_MODEL_UNSUPPORTED));
    assert_eq!(
        refused.record.report(false).fallbacks,
        vec![tree_fallback_reason(TREE_MODEL_UNSUPPORTED)],
        "{name}"
    );
    // Off: no proposer, no tree, nothing reported.
    let off = tree_run(model, &mut NoProposer, &config, 8, None);
    assert_eq!(off.record.tree_verify, None, "{name}");
    assert_eq!(off.record.tree_fallback, None, "{name}");
}

/// AC3 (E8): a linear request runs no tree and reports none; a tree request whose proposer drafts
/// one branch verifies through the unchanged linear path, token for token and pass for pass; a
/// tree request on a model that cannot keep a branch verifies linearly with the reason named; and
/// a run without a proposer is untouched by a tree bound.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn linear_requests_and_one_branch_trees_keep_the_linear_path() {
    let (_guard, device) = proof_device();
    check_linear("causal", &causal(&device));
    let (_cfg, hybrid) = text_model_on(&device);
    check_linear("hybrid", &hybrid);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::linear_requests_and_one_branch_trees_keep_the_linear_path");
}

fn check_graphs<M: StepModel>(name: &str, model: &M, vocab: i32) {
    let runner = GraphRunner::new(model);
    // One tree step through the runner: counted eager with the tree reason, never a replay.
    let mut cache = prefilled(&runner);
    let base = cache.len();
    let mut star = DraftTree::new();
    star.push(0, 1, None).unwrap();
    star.push(0, 2, None).unwrap();
    let before = graph_tally();
    runner
        .forward_tree(
            &mut cache,
            &TreeVerify {
                tree: &star,
                anchor: 5,
                want_hidden: false,
            },
        )
        .unwrap();
    let step = graph_tally().since(&before);
    assert_eq!(
        (step.replayed, step.eager),
        (0, 1),
        "{name}: {}",
        step.describe()
    );
    assert_eq!(step.fallback_reason, Some(REASON_TREE_VERIFY), "{name}");
    cache.abort_tree(base).unwrap();

    // A whole greedy tree run through the runner: the plain loop's tokens; every tree pass is
    // one of the eager steps, and on CUDA the linear and plain steps still replay their graphs.
    let config = greedy(24);
    let expected = plain(model, &config, None);
    let run = tree_run(
        &runner,
        &mut NoisyTree {
            rng: SplitMix64::new(0x7e3e),
            vocab,
        },
        &config,
        16,
        None,
    );
    assert_eq!(run.output.tokens, expected.tokens, "{name}");
    let tree = run.record.tree_verify.expect("a tree run reports its tree");
    let graphs = run.record.cuda_graphs;
    assert!(tree.tree_passes > 0, "{name}");
    assert!(
        graphs.eager >= tree.tree_passes,
        "{name}: {} for {} tree passes",
        graphs.describe(),
        tree.tree_passes
    );
    if model.device().is_cuda() {
        assert!(graphs.replayed > 0, "{name}: {}", graphs.describe());
        assert!(graphs.captured > 0, "{name}: {}", graphs.describe());
    } else {
        // Nothing captures off CUDA: no step claims a replay.
        assert_eq!(graphs.replayed, 0, "{name}: {}", graphs.describe());
        assert!(graphs.fallback_reason.is_some(), "{name}");
    }
    let report = run.record.report(true);
    assert_eq!(report.cuda_graphs.replayed, graphs.replayed, "{name}");
    assert_eq!(report.cuda_graphs.eager, graphs.eager, "{name}");
}

/// AC2: the CUDA-graph runner never claims a replay for a tree step — each runs eager, counted
/// with the named reason `tree_verify` — while a tree run's linear and plain steps keep replaying
/// their captured graphs on CUDA (nothing replays on the CPU), and the run's tokens are the plain
/// loop's on both cache classes.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn tree_steps_run_eager_through_the_graph_runner_with_the_reason_named() {
    let (_guard, device) = proof_device();
    check_graphs("causal", &causal(&device), 24);
    let (_cfg, hybrid) = text_model_on(&device);
    check_graphs("hybrid", &hybrid, 50);

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED decode::engine::tree_tests::tree_steps_run_eager_through_the_graph_runner_with_the_reason_named");
}

/// Tensor-free: every fixture shape is built through `core_llm`'s validating
/// [`DraftTree::push`] under its documented contract (row `0` is the anchor; the node pushed
/// `r`-th is row `r`, whose parent row is below it), holds 15–16 candidates, and each shape is
/// what its name says.
#[test]
fn the_fixture_trees_follow_the_draft_tree_contract() {
    let shapes = shapes(24);
    for (name, tree) in &shapes {
        assert!((15..=16).contains(&tree.len()), "{name}: {}", tree.len());
        for row in 1..tree.rows() {
            assert!(tree.parent(row).unwrap() < row, "{name}: row {row}");
        }
    }
    let by = |n: &str| &shapes.iter().find(|(name, _)| *name == n).unwrap().1;
    assert!(by("chain").is_chain());
    assert_eq!(by("chain").max_depth(), 16);
    assert_eq!(by("star").max_depth(), 1);
    assert_eq!(by("two branches").children(0), vec![1, 2]);
    assert_eq!(by("two branches").max_depth(), 8);
    assert_eq!(by("binary").max_depth(), 4);
    // `pruned` keeps a valid tree too (the engine prunes every proposal by depth and rows).
    for (name, tree) in &shapes {
        let pruned = tree.pruned(4, 8, |_| false);
        assert!(pruned.len() <= 8 && pruned.max_depth() <= 4, "{name}");
    }
    assert_eq!(tree_extra_positions(4, 16), 12);
    assert_eq!(tree_extra_positions(4, 40), 12, "capped at MAX_TREE_ROWS");
    assert_eq!(tree_extra_positions(0, 16), 0, "no drafts, no tree");
    assert_eq!(tree_extra_positions(7, 4), 0);
}
