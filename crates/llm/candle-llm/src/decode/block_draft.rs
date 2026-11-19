//! The block-draft proposer on Candle (epic SW-F360, story SW-7197) — the port of mlx-llm's
//! `decode::block_draft`: a [`BlockDraft`] (DSpine or DFlash2, as the checkpoint's `architectures`
//! name it) proposing a whole block per step from the Qwen3.5/3.8 target's hidden rows at its
//! `target_layer_ids`, verified by the engine's ordinary linear verifier and rolled back through
//! the hybrid's checkpoint ring.
//!
//! The target side is [`TappedQwen35`]: the hybrid whose hidden rows, when a proposer asks for
//! them, are the **tapped** layer outputs rather than the final-normalized rows — so the engine,
//! the prefix-cache prefill, the rollback and the CUDA-graph runner are the unchanged ones, and
//! every row the engine hands the proposer (the prompt's at warm-up, `[cur, accepted…]` at each
//! commit) is a row the draft reads. A verify step is an ordinary linear verify step of `K + 1`
//! tokens that asks for hidden rows: the graph runner captures and replays it as any other such
//! shape. The draft's own forward runs outside the runner, eager, between verify steps.
//!
//! Neither format's block reads a drafted token: the block is `[anchor, mask × …]` and the draft
//! rows depend only on the context. A row's dependence on the path is entirely its score table's
//! sender column — row `i`'s candidates scored for each candidate of row `i - 1` — so the proposer
//! keeps every scored row's candidates and the whole table ([`BlockDraftProposer::candidates`]),
//! the lattice a tree proposal branches over.

use candle_core::Tensor;
use core_llm::ProposerKind;

use crate::decode::engine::{DraftSampler, Drafts, Proposal, ProposeContext, Proposer};
use crate::decode::step::{StepModel, StepOutput, StepRequest};
use crate::error::{Error, Result};
use crate::models::{BlockDraft, BlockDraftContext, Qwen35Cache, Qwen35Model};
use crate::primitives::attention::AttnFormulation;
use crate::primitives::input_ids;

/// Why a tree request with a block draft verifies linearly on Candle (feature SW-F328): the
/// block-draft proposer here hands the verifier its linear chain.
pub const BLOCK_DRAFT_TREE_UNSUPPORTED: &str = "the Candle block-draft proposer proposes linearly";

/// The Qwen3.5/3.8 hybrid as a block draft's target: its hidden rows are the outputs of `layers`
/// (0-based), concatenated, and a step returns at most the last `keep` of them (the draft's
/// context window, E5).
pub struct TappedQwen35<'m> {
    /// The target.
    pub model: &'m Qwen35Model,
    /// The layers whose outputs the draft reads.
    pub layers: Vec<usize>,
    /// The most rows a step returns (`None`: all).
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

impl StepModel for TappedQwen35<'_> {
    type Cache = Qwen35Cache;

    fn new_cache(&self) -> Qwen35Cache {
        StepModel::new_cache(self.model)
    }

    fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<Qwen35Cache> {
        self.model.new_cache_for(capacity, overshoot)
    }

    fn attn_formulation(&self, cache: &Qwen35Cache) -> AttnFormulation {
        StepModel::attn_formulation(self.model, cache)
    }

    fn graph_support(&self) -> std::result::Result<(), &'static str> {
        self.model.graph_support()
    }

    fn device(&self) -> &candle_core::Device {
        self.model.device()
    }

    fn vocab_size(&self) -> usize {
        StepModel::vocab_size(self.model)
    }

    fn forward_step(
        &self,
        cache: &mut Qwen35Cache,
        request: StepRequest<'_>,
    ) -> Result<StepOutput> {
        self.model
            .forward_step_tapped(cache, request, Some((&self.layers, self.keep)))
    }

    fn tree_unsupported(&self, _: &Qwen35Cache) -> Option<&'static str> {
        Some(BLOCK_DRAFT_TREE_UNSUPPORTED)
    }
}

/// A block draft as the engine's proposer over a [`TappedQwen35`] target, whichever its format:
/// both yield per-row candidates and a predecessor-conditioned score table
/// ([`BlockDraft::propose`]) the proposer walks into the linear proposal.
///
/// Its only state is the draft's context ([`BlockDraftContext`]): the target-confirmed rows, never
/// a draft row. A proposal reads it; a commit appends the target's rows for `[cur, accepted…]` —
/// the rows the verify forward computed for the committed tokens — so a rejection needs no
/// rollback and the context never holds overwritten history (E2).
pub struct BlockDraftProposer<'d> {
    draft: &'d BlockDraft,
    target: &'d Qwen35Model,
    context: Option<BlockDraftContext>,
    /// The last proposal's per-row candidates (top-`K` ids of each scored draft row, ascending).
    candidates: Vec<Vec<i32>>,
    /// The last proposal's `[rows, K, K]` score table, row-major: row `i`'s candidates scored for
    /// each candidate sender of row `i - 1`.
    table: Vec<f32>,
    /// Draft block forwards issued.
    pub draft_forwards: u64,
}

impl<'d> BlockDraftProposer<'d> {
    /// A proposer over `draft`, embedding its blocks and scoring its rows with `target`.
    pub fn new(draft: &'d BlockDraft, target: &'d Qwen35Model) -> Self {
        Self {
            draft,
            target,
            context: None,
            candidates: Vec::new(),
            table: Vec::new(),
            draft_forwards: 0,
        }
    }

    /// The last proposal's candidates per scored draft row.
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

    fn context_mut(&mut self) -> Result<&mut BlockDraftContext> {
        self.context
            .as_mut()
            .ok_or_else(|| Error::Msg("BlockDraftProposer: used before warm".into()))
    }

    /// Append the target rows `rows` `[1, n, tap_width]` for positions `first..`, skipping any the
    /// context already holds.
    fn append_from(&mut self, rows: &Tensor, first: i32) -> Result<()> {
        let draft = self.draft;
        let context = self.context_mut()?;
        let skip = context.next_position() - first;
        let n = rows.dim(1)? as i32;
        if skip < 0 || skip > n {
            return Err(Error::Msg(format!(
                "BlockDraftProposer: rows from position {first} do not continue a context at {}",
                context.next_position()
            )));
        }
        let rows = rows.narrow(1, skip as usize, (n - skip) as usize)?;
        draft.append_context(context, &rows)
    }

    /// One draft block forward for the step `ctx`: stores each scored row's candidates and the
    /// score table, and returns the vocabulary width the rows are drawn over.
    fn run_draft(&mut self, ctx: &ProposeContext<'_>) -> Result<usize> {
        // After steps the target ran without a proposal (`auto`'s probe), the row before `cur`
        // arrives as the step's previous hidden row.
        let behind = self
            .context
            .as_ref()
            .is_some_and(|c| c.next_position() < ctx.position);
        if behind {
            let row = ctx.previous_hidden.ok_or_else(|| {
                Error::Msg("BlockDraftProposer: no target row before the anchor".into())
            })?;
            self.append_from(row, ctx.position - 1)?;
        }
        let (draft, target) = (self.draft, self.target);
        let cfg = draft.config();
        let depth = ctx.max_drafts.min(cfg.block_size - 1);
        let len = block_len(draft.causal(), cfg.block_size, depth, ctx.budget_drafts);
        let mut block = vec![ctx.cur];
        block.resize(len, cfg.mask_token_id as i32);
        let embeds = target.embed_input_ids(&input_ids(&block, target.device())?)?;
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| Error::Msg("BlockDraftProposer: propose before warm".into()))?;
        if context.next_position() != ctx.position {
            return Err(Error::Msg(format!(
                "BlockDraftProposer: the context ends at {}, the anchor sits at {}",
                context.next_position(),
                ctx.position
            )));
        }
        let (candidates, table) = draft.propose(context, &embeds, depth, ctx.cur, target)?;
        self.draft_forwards += 1;
        self.candidates = candidates;
        self.table = table;
        Ok(StepModel::vocab_size(target))
    }
}

impl Proposer for BlockDraftProposer<'_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::BlockDraft
    }

    fn wants_hidden(&self) -> bool {
        true
    }

    fn warm(&mut self, prompt: &[i32], prompt_hidden: Option<&Tensor>) -> Result<()> {
        let hidden = prompt_hidden.ok_or_else(|| {
            Error::Msg("BlockDraftProposer: the target returned no tapped prompt rows".into())
        })?;
        // The rows are the prompt's last ones: the window's tail of a long prompt, or the suffix
        // past a prefix-cache hit.
        let rows = hidden.dim(1)?;
        let start = prompt.len().checked_sub(rows).ok_or_else(|| {
            Error::Msg(format!(
                "BlockDraftProposer: {rows} tapped rows for a {}-token prompt",
                prompt.len()
            ))
        })?;
        let mut context = self.draft.new_context(start as i32);
        self.draft.append_context(&mut context, hidden)?;
        self.context = Some(context);
        self.candidates.clear();
        self.table.clear();
        Ok(())
    }

    fn propose(
        &mut self,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let width = self.run_draft(ctx)?;
        // Walk the chain: row `i`'s scores are its candidates rescored for the sender chosen at
        // row `i - 1` (row 0's sender is the anchor, the same for every column).
        let mut drafts = Vec::with_capacity(self.candidates.len());
        let mut dists = Vec::with_capacity(self.candidates.len());
        let mut history = ctx.history.to_vec();
        let mut sender = 0usize;
        for row in 0..self.candidates.len() {
            let candidates = &self.candidates[row];
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
        Ok(Proposal {
            drafts: Some(Drafts::Host(drafts)),
            dists,
        })
    }

    fn commit(
        &mut self,
        _: i32,
        _: &[i32],
        kept_hidden: Option<&Tensor>,
        position: i32,
    ) -> Result<()> {
        let rows = kept_hidden.ok_or_else(|| {
            Error::Msg("BlockDraftProposer: no target rows for the committed tokens".into())
        })?;
        // `position` is `accepted[0]`'s; the rows start at `cur`'s.
        self.append_from(rows, position - 1)
    }

    fn catch_up(
        &mut self,
        _: &[i32],
        preceding_hidden: Option<&Tensor>,
        position: i32,
    ) -> Result<()> {
        // Each probed token's predecessor row: positions `position - 1..`.
        match preceding_hidden {
            Some(rows) => self.append_from(rows, position - 1),
            None => Err(Error::Msg(
                "BlockDraftProposer: no target rows for the probed steps".into(),
            )),
        }
    }
}

/// The rows of the block a proposal of `depth` drafts runs (anchor included) for a draft trained
/// on `block_size`-row blocks, with `budget` drafts left in the token budget. A causal draft's
/// first rows are the same whatever follows them, so it runs only the rows it proposes. A
/// non-causal one (the published DFlash2 drafts) conditions every row on the whole block: it runs
/// the block it was trained on — shortened only where the token budget ends, as the reference
/// does — and proposes its first `depth` rows (the SW-7194 rule).
pub(crate) fn block_len(causal: bool, block_size: usize, depth: usize, budget: usize) -> usize {
    if causal {
        depth + 1
    } else {
        block_size.min(budget.max(depth) + 1)
    }
}

#[cfg(test)]
mod tests;
