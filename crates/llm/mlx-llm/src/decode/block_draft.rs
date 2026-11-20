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

use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use core_llm::{build_lattice_tree, DraftLattice, DraftTree, ProposerKind};

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

    /// The candidate tree of one draft forward (SW-7196), grown by the shared builder
    /// [`core_llm::build_lattice_tree`].
    fn propose_tree(
        &mut self,
        target: &TappedQwen35<'_>,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<DraftTree> {
        let width = self.run_draft(target, ctx)?;
        let lattice = DraftLattice {
            candidates: &self.candidates,
            table: &self.table,
            width,
        };
        build_lattice_tree(&lattice, ctx.history, ctx.max_rows, sampler)
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
#[cfg(test)]
mod tree_tests;
