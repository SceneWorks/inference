//! The block-draft proposer (epic SW-F360, story SW-7193): a DSpine draft
//! ([`DSpineDraft`]) proposing a whole block per step from the Qwen35 target's hidden rows at
//! its `target_layer_ids`, verified by the engine's ordinary linear verifier and rolled back
//! through the hybrid's checkpoint ring ([`CheckpointRingRollback`]).
//!
//! The target side is [`TappedQwen35`]: the Qwen35 hybrid whose hidden rows, when a proposer asks
//! for them, are the **tapped** layer outputs rather than the final-normalized rows — so the
//! engine, the prefix-cache prefill and the rollback are the unchanged ones, and every row the
//! engine hands the proposer (the prompt's at warm-up, `[cur, accepted…]` at each commit) is a
//! row the draft reads.

use mlx_rs::transforms::eval;
use mlx_rs::Array;

use core_llm::ProposerKind;

use crate::decode::engine::{
    CheckpointRingRollback, DraftSampler, LogitsScope, Proposal, ProposeContext, Proposer,
    SpeculativeTarget, TargetOutput,
};
use crate::error::{Error, Result};
use crate::models::{DSpineContext, DSpineDraft, Qwen35Cache, Qwen35Model};
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
    pub fn for_draft(model: &'m Qwen35Model, draft: &DSpineDraft) -> Self {
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
}

/// A DSpine block draft as the engine's proposer over a [`TappedQwen35`] target.
///
/// Its only state is the draft's context ([`DSpineContext`]): the target-confirmed rows, never a
/// draft row. A proposal reads it; a commit appends the target's rows for `[cur, accepted…]` — the
/// rows the verify forward computed for the committed tokens — so a rejection needs no rollback
/// and a rebuild is never needed: overwritten history never entered the context (E2).
pub struct BlockDraftProposer<'d> {
    draft: &'d DSpineDraft,
    context: Option<DSpineContext>,
    /// The last proposal's per-row candidates (top-`K` ids of each draft row, ascending) — what a
    /// tree verifier takes in place of the single chain (SW-7196).
    candidates: Vec<Vec<i32>>,
    /// Draft block forwards issued.
    pub draft_forwards: u64,
}

impl<'d> BlockDraftProposer<'d> {
    /// A proposer over `draft`.
    pub fn new(draft: &'d DSpineDraft) -> Self {
        Self {
            draft,
            context: None,
            candidates: Vec::new(),
            draft_forwards: 0,
        }
    }

    /// The last proposal's candidates per draft row.
    pub fn candidates(&self) -> &[Vec<i32>] {
        &self.candidates
    }

    /// The draft's context (`None` before the warm-up).
    pub fn context(&self) -> Option<&DSpineContext> {
        self.context.as_ref()
    }

    fn context_mut(&mut self) -> Result<&mut DSpineContext> {
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
        let draft = self.draft;
        let cfg = draft.config();
        let depth = ctx.max_drafts.min(cfg.block_size - 1);
        let mut block = vec![ctx.cur];
        block.resize(depth + 1, cfg.mask_token_id as i32);
        let model = target.model;
        let embeds = model.embed_input_ids(&input_ids(&block))?;
        let context = self
            .context
            .as_ref()
            .ok_or_else(|| Error::Msg("BlockDraftProposer: propose before warm".into()))?;
        let out = draft.block(context, &embeds, ctx.cur)?;
        let (cand, table) = draft.table(&out, ctx.cur, &|rows| model.lm_head(rows), &|ids| {
            model.lm_head_rows(ids)
        })?;
        eval([&cand, &table])?;
        self.draft_forwards += 1;
        let k = cand.shape()[1] as usize;
        let cand = cand.as_slice::<i32>().to_vec();
        let table = table.as_slice::<f32>().to_vec();
        let width = model.config().vocab_size as usize;
        self.candidates = cand.chunks(k).map(<[i32]>::to_vec).collect();

        // Walk the chain: row `i`'s scores are its candidates rescored for the sender chosen at
        // row `i - 1` (row 0's sender is the anchor, the same for every column).
        let mut drafts = Vec::with_capacity(depth);
        let mut dists = Vec::with_capacity(depth);
        let mut history = ctx.history.to_vec();
        let mut sender = 0usize;
        for (row, candidates) in self.candidates.iter().enumerate() {
            let scores = &table[(row * k + sender) * k..(row * k + sender + 1) * k];
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

#[cfg(test)]
mod tests;
