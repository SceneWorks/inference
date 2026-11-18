//! The block-draft proposer (epic SW-F360, stories SW-7193 and SW-7194): a block draft
//! ([`BlockDraft`] — DSpine or DFlash2, as the checkpoint's `architectures` name it) proposing a
//! whole block per step from the Qwen35 target's hidden rows at its `target_layer_ids`, verified
//! by the engine's ordinary linear verifier and rolled back through the hybrid's checkpoint ring
//! ([`CheckpointRingRollback`]).
//!
//! The target side is [`TappedQwen35`]: the Qwen35 hybrid whose hidden rows, when a proposer asks
//! for them, are the **tapped** layer outputs rather than the final-normalized rows — so the
//! engine, the prefix-cache prefill and the rollback are the unchanged ones, and every row the
//! engine hands the proposer (the prompt's at warm-up, `[cur, accepted…]` at each commit) is a
//! row the draft reads.
//!
//! ## Tree verification (SW-7196)
//! Under a tree request the proposer hands the verifier a candidate **tree**
//! ([`Proposer::propose_tree`]) built from the same draft forward, and the target verifies it in
//! one tree forward ([`TappedQwen35`]'s tree methods return the tapped rows of every tree row).
//! The engine keeps the accepted branch only, so a commit appends the accepted branch's tapped
//! rows to the draft's context — the context a draft that only ever saw the accepted path holds —
//! and an aborted tree step never reaches the proposer.
//!
//! Neither format's block reads a drafted token: the block is `[anchor, mask × …]` and the draft
//! rows depend only on the context. A row's dependence on the path is entirely its score table's
//! sender column — row `i`'s candidates scored for each candidate of row `i - 1`. So every branch
//! through the candidate lattice has its exact proposal distribution in the one table, for the
//! causal DSpine as for the non-causal DFlash2: a branch is scored per its own parent, never from
//! the greedy path, and no branch needs a second draft forward.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use core_llm::{DraftTree, ProposerKind};

use crate::decode::engine::{
    CheckpointRingRollback, DraftSampler, LogitsScope, Proposal, ProposeContext, Proposer,
    SpeculativeTarget, TargetOutput, TreeInputs,
};
use crate::error::{Error, Result};
use crate::models::{BlockDraft, BlockDraftContext, Qwen35Cache, Qwen35Model};
use crate::primitives::input_ids;

/// The Qwen35 hybrid as a block draft's target: its hidden rows are the outputs of `layers`
/// (0-based), concatenated, and a forward returns at most the last `keep` of them (the draft's
/// context window, E5 — a long prefill never holds every prompt row's taps).
pub struct TappedQwen35<'m> {
    /// The target.
    pub model: &'m Qwen35Model,
    /// The layers whose outputs the draft reads.
    pub layers: Vec<usize>,
    /// The most rows a forward returns (`None`: all).
    pub keep: Option<usize>,
}

impl<'m> TappedQwen35<'m> {
    /// `model` tapped for `draft`.
    pub fn for_draft(model: &'m Qwen35Model, draft: &BlockDraft) -> Self {
        Self {
            model,
            layers: draft.config().target_layer_ids.clone(),
            keep: draft.config().context_rows(),
        }
    }
}

impl SpeculativeTarget for TappedQwen35<'_> {
    type Cache = Qwen35Cache;
    type Rollback = CheckpointRingRollback;

    fn new_cache(&self) -> Qwen35Cache {
        Qwen35Model::new_cache(self.model)
    }

    fn cache_len(&self, cache: &Qwen35Cache) -> i32 {
        cache.offset()
    }

    fn rollback(&self, width: usize) -> CheckpointRingRollback {
        CheckpointRingRollback::new(width)
    }

    fn forward(
        &self,
        cache: &mut Qwen35Cache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        if !want_hidden {
            return SpeculativeTarget::forward(self.model, cache, ids, rope_offset, scope, false);
        }
        let out = self.model.forward_tapped(
            ids,
            cache,
            rope_offset,
            &self.layers,
            self.keep,
            scope == LogitsScope::All,
        );
        // As the plain hybrid target: a forward that failed part-way drops a pending boundary
        // capture its later layers never reached.
        if out.is_err() {
            cache.clear_boundary_capture();
        }
        let (hidden, logits) = out?;
        Ok(TargetOutput {
            logits,
            hidden: Some(hidden),
        })
    }

    fn attention_label(&self) -> &'static str {
        "gqa"
    }

    fn tree_unsupported(&self, cache: &Qwen35Cache) -> Option<String> {
        SpeculativeTarget::tree_unsupported(self.model, cache)
    }

    fn tree_mask_dtype(&self) -> Dtype {
        SpeculativeTarget::tree_mask_dtype(self.model)
    }

    /// The hybrid's tree forward; with hidden rows asked for, they are every tree row's taps (the
    /// window cap does not apply: a tree carries at most the anchor and sixteen rows).
    fn forward_tree(
        &self,
        cache: &mut Qwen35Cache,
        inputs: &TreeInputs,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        if !want_hidden {
            return SpeculativeTarget::forward_tree(self.model, cache, inputs, false);
        }
        let plan = std::sync::Arc::new(crate::primitives::gated_delta::TreePlan::new(
            &inputs.parents,
        ));
        let (logits, hidden) = self.model.forward_tree_tapped(
            &inputs.ids,
            cache,
            &inputs.positions,
            plan,
            &inputs.mask,
            &self.layers,
        )?;
        eval([&hidden, &logits])?;
        Ok(TargetOutput {
            logits,
            hidden: Some(hidden),
        })
    }

    fn commit_tree(&self, cache: &mut Qwen35Cache, base: i32, path: &[usize]) -> Result<()> {
        SpeculativeTarget::commit_tree(self.model, cache, base, path)
    }

    fn abort_tree(&self, cache: &mut Qwen35Cache, base: i32) -> Result<()> {
        SpeculativeTarget::abort_tree(self.model, cache, base)
    }

    fn tree_workspace_bytes(&self, cache: &Qwen35Cache) -> u64 {
        SpeculativeTarget::tree_workspace_bytes(self.model, cache)
    }
}

/// A block draft as the engine's proposer over a [`TappedQwen35`] target, whichever its format:
/// both yield per-row candidates and a predecessor-conditioned score table
/// ([`BlockDraft::propose`]) the proposer walks into the linear proposal.
///
/// Its only state is the draft's context ([`BlockDraftContext`]): the target-confirmed rows, never a
/// draft row. A proposal reads it; a commit appends the target's rows for `[cur, accepted…]` — the
/// rows the verify forward computed for the committed tokens — so a rejection needs no rollback
/// and a rebuild is never needed: overwritten history never entered the context (E2).
pub struct BlockDraftProposer<'d> {
    draft: &'d BlockDraft,
    context: Option<BlockDraftContext>,
    /// The last proposal's per-row candidates (top-`K` ids of each draft row, ascending; DFlash2's
    /// selector candidates) — what a tree verifier takes in place of the single chain (SW-7196).
    candidates: Vec<Vec<i32>>,
    /// The last proposal's `[rows, K, K]` score table, row-major: row `i`'s candidates scored for
    /// each candidate sender of row `i - 1`.
    table: Vec<f32>,
    /// Draft block forwards issued.
    pub draft_forwards: u64,
}

impl<'d> BlockDraftProposer<'d> {
    /// A proposer over `draft`.
    pub fn new(draft: &'d BlockDraft) -> Self {
        Self {
            draft,
            context: None,
            candidates: Vec::new(),
            table: Vec::new(),
            draft_forwards: 0,
        }
    }

    /// The last proposal's candidates per draft row.
    pub fn candidates(&self) -> &[Vec<i32>] {
        &self.candidates
    }

    /// The draft's context (`None` before the warm-up).
    pub fn context(&self) -> Option<&BlockDraftContext> {
        self.context.as_ref()
    }

    /// Row `row`'s scores for the sender at column `sender` of row `row - 1`.
    fn scores(&self, row: usize, sender: usize) -> &[f32] {
        let k = self.candidates.first().map_or(0, Vec::len);
        &self.table[(row * k + sender) * k..(row * k + sender + 1) * k]
    }

    /// One draft block forward for the step `ctx`: stores each scored row's candidates and the
    /// score table, and returns the vocabulary width the rows are drawn over.
    fn run_draft(&mut self, target: &TappedQwen35<'_>, ctx: &ProposeContext<'_>) -> Result<usize> {
        let draft = self.draft;
        let cfg = draft.config();
        let depth = ctx.max_drafts.min(cfg.block_size - 1);
        let len = block_len(draft.causal(), cfg.block_size, depth, ctx.budget_drafts);
        let mut block = vec![ctx.cur];
        block.resize(len, cfg.mask_token_id as i32);
        let model = target.model;
        let embeds = model.embed_input_ids(&input_ids(&block))?;
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| Error::Msg("BlockDraftProposer: propose before warm".into()))?;
        let (cand, table) = draft.propose(
            context,
            &embeds,
            depth,
            ctx.cur,
            &|rows| model.lm_head(rows),
            &|ids| model.lm_head_rows(ids),
        )?;
        eval([&cand, &table])?;
        self.draft_forwards += 1;
        let k = cand.shape()[1] as usize;
        self.candidates = cand
            .as_slice::<i32>()
            .chunks(k)
            .map(<[i32]>::to_vec)
            .collect();
        self.table = table.as_slice::<f32>().to_vec();
        Ok(model.config().vocab_size as usize)
    }

    fn context_mut(&mut self) -> Result<&mut BlockDraftContext> {
        self.context
            .as_mut()
            .ok_or_else(|| Error::Msg("BlockDraftProposer: used before warm".into()))
    }
}

impl Proposer<TappedQwen35<'_>> for BlockDraftProposer<'_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::BlockDraft
    }

    fn wants_hidden(&self) -> bool {
        true
    }

    fn warm(
        &mut self,
        _: &TappedQwen35<'_>,
        prompt: &[i32],
        prompt_hidden: Option<&Array>,
    ) -> Result<Option<Array>> {
        let hidden = prompt_hidden.ok_or_else(|| {
            Error::Msg("BlockDraftProposer: the target returned no tapped prompt rows".into())
        })?;
        // The rows are the prompt's last ones: the window's tail of a long prompt, or the suffix
        // past a prefix-cache hit.
        let rows = hidden.shape()[1] as usize;
        let start = prompt.len().checked_sub(rows).ok_or_else(|| {
            Error::Msg(format!(
                "BlockDraftProposer: {rows} tapped rows for a {}-token prompt",
                prompt.len()
            ))
        })?;
        let mut context = self.draft.new_context(start as i32);
        self.draft.append_context(&mut context, hidden)?;
        let sync = context.arrays().next().cloned();
        self.context = Some(context);
        self.candidates.clear();
        Ok(sync)
    }

    fn propose(
        &mut self,
        target: &TappedQwen35<'_>,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let width = self.run_draft(target, ctx)?;
        // Walk the chain: row `i`'s scores are its candidates rescored for the sender chosen at
        // row `i - 1` (row 0's sender is the anchor, the same for every column).
        let depth = self.candidates.len();
        let mut drafts = Vec::with_capacity(depth);
        let mut dists = Vec::with_capacity(depth);
        let mut history = ctx.history.to_vec();
        let mut sender = 0usize;
        for (row, candidates) in self.candidates.iter().enumerate() {
            let scores = self.scores(row, sender);
            let Some((token, dist)) = sampler.sample_among(candidates, scores, width, &history)?
            else {
                break;
            };
            let Some(chosen) = candidates.iter().position(|&c| c == token) else {
                break;
            };
            dists.extend(dist);
            drafts.push(token);
            history.push(token);
            sender = chosen;
            if sampler.is_stop(token) {
                break;
            }
        }
        Ok(Proposal { drafts, dists })
    }

    /// The candidate tree of one draft forward (SW-7196): see [`build_tree`].
    fn propose_tree(
        &mut self,
        target: &TappedQwen35<'_>,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<DraftTree> {
        let width = self.run_draft(target, ctx)?;
        let lattice = Lattice {
            candidates: &self.candidates,
            table: &self.table,
            width,
        };
        build_tree(&lattice, ctx, sampler)
    }

    fn commit(
        &mut self,
        _: &TappedQwen35<'_>,
        _: i32,
        _: &[i32],
        kept_hidden: Option<&Array>,
        _: i32,
    ) -> Result<()> {
        let rows = kept_hidden.ok_or_else(|| {
            Error::Msg("BlockDraftProposer: no target rows for the committed tokens".into())
        })?;
        let draft = self.draft;
        draft.append_context(self.context_mut()?, rows)
    }
}

/// One draft forward's candidate lattice: each scored row's candidates (ascending) and the
/// `[rows, K, K]` predecessor-conditioned score table, over a `width`-wide vocabulary.
struct Lattice<'a> {
    candidates: &'a [Vec<i32>],
    table: &'a [f32],
    width: usize,
}

impl Lattice<'_> {
    /// Row `row`'s scores for the sender at column `sender` of row `row - 1`.
    fn scores(&self, row: usize, sender: usize) -> &[f32] {
        let k = self.candidates[row].len();
        &self.table[(row * k + sender) * k..(row * k + sender + 1) * k]
    }
}

/// A tree row under construction: the tokens from the anchor to it, the draft's estimate of
/// reaching it, and — once expanded — its children's proposal.
struct TreeNode {
    /// Tokens from the anchor (exclusive) to this row (inclusive).
    path: Vec<i32>,
    /// This row's column among its draft row's candidates: the sender its children are scored
    /// for (`0` for the anchor, whose sender column is the same for every candidate).
    sender: usize,
    /// The draft's probability of the path: the product of each step's proposal probability.
    mass: f64,
    /// A stochastic run's proposal `q` for this row's children; its sampled child is drawn from it.
    q: Option<Vec<(i32, f32)>>,
    /// The children's tokens, most probable first, with their normalized proposal probability.
    order: Vec<(i32, f64)>,
    /// The first child: the sampled child once drawn (stochastic), the draft's choice (greedy).
    first: Option<i32>,
}

/// What a [`Slot`] adds once popped.
#[derive(Clone, Copy)]
enum Child {
    /// A stochastic run's sampled child: drawn from the parent's `q` when popped.
    Drawn,
    /// A deterministic child with this token.
    Fixed(i32),
    /// A stochastic run's `n`-th most probable token of the parent's `q` other than its sampled
    /// child — deterministic once that child is drawn.
    Ranked(usize),
}

/// A child the best-first build may add under tree row `parent`, ranked by `priority` (the
/// parent's path mass times the child's proposal probability), then by insertion (`seq`).
struct Slot {
    priority: f64,
    seq: u64,
    parent: usize,
    child: Child,
}

impl PartialEq for Slot {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Slot {}

impl PartialOrd for Slot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Slot {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .total_cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// Build the candidate tree of one draft forward (SW-7196), best first: starting at the anchor,
/// repeatedly add the child of highest estimated path probability under the draft (each row's
/// children scored for its own sender column, so a branch's distribution is conditioned on its
/// own parent), until `ctx.max_rows` candidates or nothing left; no candidate deeper than the
/// draft's scored rows or under a stop token. Each node's children see the request's penalties
/// over the node's own path and the constraint advanced along it.
///
/// **Exactness** (`core_llm`'s recursive rejection over siblings): a greedy run's children are
/// all deterministic (`dist: None`) — the draft's choice first, then the other allowed candidates
/// by score — and any tree commits exactly the target's greedy tokens. A stochastic run gives
/// each node **one sampled child** drawn from the node's `q` (the shaped distribution
/// [`DraftSampler::sample_among`] draws a linear draft from), tried first, then deterministic
/// siblings: the most probable tokens of `q` other than the sampled one. The sampled child is a
/// true draw from its own `q`, made when its slot is popped — whether the slot exists and where
/// it sits depend only on draws made before it, never on its own value, and it outranks its
/// siblings so it is tried first — and each later sibling is a point mass fixed once the sampled
/// child is known. After the sampled child's rejection the residual is `norm(max(0, p − q))`
/// whatever was drawn, and a fixed list of distinct point masses is exact against any residual,
/// so the committed token is distributed exactly as the target's. Each node's sampled child is
/// drawn independently of every other node's (fresh uniforms of the run's seeded stream).
fn build_tree(
    lattice: &Lattice<'_>,
    ctx: &ProposeContext<'_>,
    sampler: &mut DraftSampler<'_, '_>,
) -> Result<DraftTree> {
    let max_rows = ctx.max_rows.min(core_llm::MAX_TREE_ROWS as usize);
    let checkpoint = sampler.constraint_checkpoint();
    let mut tree = DraftTree::new();
    let mut nodes = vec![TreeNode {
        path: Vec::new(),
        sender: 0,
        mass: 1.0,
        q: None,
        order: Vec::new(),
        first: None,
    }];
    let mut slots = BinaryHeap::new();
    let mut seq = 0u64;
    expand(
        lattice, ctx, sampler, checkpoint, &mut nodes, 0, &mut slots, &mut seq,
    )?;
    while tree.len() < max_rows {
        let Some(slot) = slots.pop() else {
            break;
        };
        let (token, dist) = match slot.child {
            Child::Drawn => {
                let q = nodes[slot.parent].q.clone().ok_or_else(|| {
                    Error::Msg("block draft tree: a sampled child without its q".into())
                })?;
                let token = sampler.draw(&q);
                nodes[slot.parent].first = Some(token);
                (token, Some(q))
            }
            Child::Fixed(token) => (token, None),
            Child::Ranked(n) => {
                let parent = &nodes[slot.parent];
                let first = parent.first;
                match parent
                    .order
                    .iter()
                    .map(|&(t, _)| t)
                    .filter(|&t| Some(t) != first)
                    .nth(n)
                {
                    Some(token) => (token, None),
                    None => continue,
                }
            }
        };
        let parent = &nodes[slot.parent];
        let depth = parent.path.len();
        let probability = parent
            .order
            .iter()
            .find(|&&(t, _)| t == token)
            .map_or(0.0, |&(_, p)| p);
        let sender = lattice.candidates[depth]
            .iter()
            .position(|&c| c == token)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "block draft tree: {token} is not a candidate of draft row {depth}"
                ))
            })?;
        let mut path = parent.path.clone();
        path.push(token);
        let mass = parent.mass * probability;
        let row = tree
            .push(slot.parent, token, dist)
            .map_err(|e| Error::Msg(e.to_string()))?;
        debug_assert_eq!(row, nodes.len(), "one node per tree row");
        nodes.push(TreeNode {
            path,
            sender,
            mass,
            q: None,
            order: Vec::new(),
            first: None,
        });
        if depth + 1 < lattice.candidates.len() && !sampler.is_stop(token) {
            expand(
                lattice, ctx, sampler, checkpoint, &mut nodes, row, &mut slots, &mut seq,
            )?;
        }
    }
    // Leave the constraint where the proposal found it.
    sampler.enter_branch(checkpoint, &[]);
    Ok(tree)
}

/// Score tree row `row`'s children ([`build_tree`]) — its draft row's candidates for its own
/// sender, under the penalties and constraint of its own path — and queue their slots.
#[allow(clippy::too_many_arguments)]
fn expand(
    lattice: &Lattice<'_>,
    ctx: &ProposeContext<'_>,
    sampler: &mut DraftSampler<'_, '_>,
    checkpoint: Option<usize>,
    nodes: &mut [TreeNode],
    row: usize,
    slots: &mut BinaryHeap<Slot>,
    seq: &mut u64,
) -> Result<()> {
    let node = &nodes[row];
    let depth = node.path.len();
    let candidates = &lattice.candidates[depth];
    let scores = lattice.scores(depth, node.sender);
    let mass = node.mass;
    sampler.enter_branch(checkpoint, &node.path);
    let mut history = ctx.history.to_vec();
    history.extend(&node.path);
    let mut queue = |priority: f64, child: Child| {
        slots.push(Slot {
            priority,
            seq: *seq,
            parent: row,
            child,
        });
        *seq += 1;
    };
    if sampler.greedy() {
        let allowed = sampler.allowed_among(candidates);
        let Some((first, _)) = sampler.sample_among(candidates, scores, lattice.width, &history)?
        else {
            return Ok(());
        };
        // The draft's own probabilities (a softmax of the allowed candidates' scores) rank the
        // deterministic children; the draft's choice comes first.
        let scored: Vec<(i32, f32)> = candidates
            .iter()
            .zip(scores)
            .filter(|(t, _)| allowed.contains(t))
            .map(|(&t, &s)| (t, s))
            .collect();
        let top = scored.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
        let mut order: Vec<(i32, f64)> = scored
            .iter()
            .map(|&(t, s)| (t, f64::from((s - top).exp())))
            .collect();
        normalize(&mut order);
        let p_first = order.iter().find(|x| x.0 == first).map_or(1.0, |x| x.1);
        queue(mass * order[0].1.max(p_first), Child::Fixed(first));
        for &(token, p) in order.iter().filter(|x| x.0 != first) {
            queue(mass * p, Child::Fixed(token));
        }
        let node = &mut nodes[row];
        node.order = order;
        node.first = Some(first);
    } else {
        let Some(q) = sampler.among_distribution(candidates, scores, lattice.width, &history)?
        else {
            return Ok(());
        };
        let mut order: Vec<(i32, f64)> = q
            .iter()
            .filter(|x| x.1 > 0.0)
            .map(|&(t, w)| (t, f64::from(w)))
            .collect();
        if order.is_empty() {
            return Ok(());
        }
        normalize(&mut order);
        // The sampled child's slot outranks (or ties, and precedes) every sibling's, so it is
        // drawn and tried first; sibling `n` is the `n`-th most probable token after it.
        queue(mass * order[0].1, Child::Drawn);
        for (n, &(_, p)) in order.iter().enumerate().skip(1) {
            queue(mass * p, Child::Ranked(n - 1));
        }
        let node = &mut nodes[row];
        node.q = Some(q);
        node.order = order;
    }
    Ok(())
}

/// Normalize `dist`'s weights and order it most probable first (lower id first on ties).
fn normalize(dist: &mut [(i32, f64)]) {
    let total: f64 = dist.iter().map(|x| x.1).sum();
    if total > 0.0 {
        for x in dist.iter_mut() {
            x.1 /= total;
        }
    }
    dist.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
}

/// The rows of the block a proposal of `depth` drafts runs (anchor included) for a draft trained
/// on `block_size`-row blocks, with `budget` drafts left in the token budget. A causal draft's
/// first rows are the same whatever follows them, so it runs only the rows it proposes. A
/// non-causal one (the published DFlash2 drafts) conditions every row on the whole block: it runs
/// the block it was trained on — shortened only where the token budget ends, as the reference
/// does — and proposes its first `depth` rows.
fn block_len(causal: bool, block_size: usize, depth: usize, budget: usize) -> usize {
    if causal {
        depth + 1
    } else {
        block_size.min(budget.max(depth) + 1)
    }
}

#[cfg(test)]
mod tests;
