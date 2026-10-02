//! The engine's proposal sources (epic sc-24128, story sc-24130): the native Qwen3.8 MTP head
//! ([`MtpProposer`]), prompt lookup ([`NgramProposer`]) and a draft model ([`DraftModelProposer`]).
//! Each implements [`Proposer`]; the loop that verifies, accepts and rolls back is the one
//! [`engine`](super::engine).
//!
//! On a plain greedy run ([`DraftSampler::device_greedy`]) the MTP and draft-model proposers keep
//! their drafts on the device: each draft step's argmax tensor is fed straight into the next draft
//! step as ids and the drafts go to the verify step as a `[1, K]` tensor, so proposing issues no
//! device->host transfer — the verify decision's single transfer brings them over. Otherwise
//! (temperature, penalties or a constraint) drafts are sampled on the host through the engine's
//! [`DraftSampler`], exactly as the pre-engine loops did.

use candle_core::Tensor;
use core_llm::speculative::ngram_propose;
use core_llm::ProposerKind;

use crate::decode::engine::{DraftSampler, Drafts, Proposal, ProposeContext, Proposer};
use crate::decode::step::{StepModel, StepRequest};
use crate::error::{Error, Result};
use crate::models::{Qwen35Mtp, Qwen35MtpCache};
use crate::primitives::decode_cache::DecodeCache;
use crate::primitives::sampler::argmax_rows_tensor;

/// The checkpoint-native Qwen3.8 multi-token predictor as a proposer: the head proposes a run of
/// `K` tokens autoregressively (its single published layer cycled, as upstream vLLM does), paired
/// with the target's hidden states, and after verification its cache is rebuilt from
/// target-confirmed pairs only.
///
/// The head's cache is snapshotted right after the first draft step (which consumes `cur`, a
/// target-selected and therefore always-valid token); the later, recursively drafted state is
/// discarded on commit even when its tokens were accepted, because replay must pair accepted
/// tokens with the **target's** hidden rows rather than the head's own.
///
/// **Prefix-cache resume (story sc-24437).** A prompt whose leading `M` positions came from the
/// cross-turn prefix cache is prefilled from `M` on, so the target returns hidden rows only for
/// positions `M..P`. The warm-up pairs `embed(x[j + 1])` with `H[j]`, so it needs the head's cache
/// at the boundary plus `H[M - 1]` — an [`MtpBoundary`] the prefix cache stored beside the target
/// state ([`resume_from`](Self::resume_from)); it then seeds only positions `M..P`.
/// [`capture_at`](Self::capture_at) records the same state at a boundary inside this prompt for a
/// later request to resume from.
pub struct MtpProposer<'a> {
    mtp: &'a Qwen35Mtp,
    cache: Qwen35MtpCache,
    after_cur: Option<Qwen35MtpCache>,
    resume: Option<MtpBoundary>,
    capture_at: Option<usize>,
    captured: Option<MtpBoundary>,
}

/// The MTP head's state at a prompt boundary `len` (story sc-24437): its cache after warming on
/// the first `len` prompt positions and the target's final-normalized hidden row `H[len - 1]` the
/// next warm-up pair needs. Immutable once captured: the head's growing KV is replaced, never
/// written in place, so a resumed clone cannot reach it.
#[derive(Clone, Debug)]
pub struct MtpBoundary {
    cache: Qwen35MtpCache,
    hidden: Tensor,
    len: usize,
}

impl MtpBoundary {
    /// The prompt positions the state covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the boundary covers no position (never true for a captured boundary).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes the state holds (the head's KV plus the hidden row).
    pub fn bytes(&self) -> usize {
        self.cache
            .bytes()
            .saturating_add(crate::primitives::decode_cache::tensor_bytes(&self.hidden))
    }
}

impl<'a> MtpProposer<'a> {
    /// A proposer over `mtp` with a fresh predictor cache.
    pub fn new(mtp: &'a Qwen35Mtp) -> Self {
        Self {
            mtp,
            cache: mtp.new_cache(),
            after_cur: None,
            resume: None,
            capture_at: None,
            captured: None,
        }
    }

    /// Resume from a prefix-cache boundary: the next [`warm`](Proposer::warm) is handed the full
    /// prompt but hidden rows only for positions `boundary.len()..` and seeds from there.
    pub fn resume_from(mut self, boundary: Option<MtpBoundary>) -> Self {
        self.resume = boundary;
        self
    }

    /// Record the head's state at prompt position `len` during the next warm-up (see
    /// [`take_captured`](Self::take_captured)); ignored unless `len` falls strictly inside the
    /// warmed span.
    pub fn capture_at(mut self, len: Option<usize>) -> Self {
        self.capture_at = len;
        self
    }

    /// The state captured by [`capture_at`](Self::capture_at), once.
    pub fn take_captured(&mut self) -> Option<MtpBoundary> {
        self.captured.take()
    }

    /// Warm the predictor from a **multimodal** prompt: the fused (vision-spliced) prompt
    /// `embeddings` `[1, prompt, hidden]`, the target's hidden rows for every prompt position
    /// `[1, prompt, hidden]`, and the prompt's interleaved M-RoPE position rows. The caller then
    /// starts the engine with `warm_proposer: false`. Pairs `embed(token[j + 1])` with
    /// `hidden[j]`, as the text warm-up does.
    pub fn warm_multimodal(
        &mut self,
        embeddings: &Tensor,
        prompt_hidden: &Tensor,
        positions: [&[i32]; 3],
    ) -> Result<()> {
        let prompt_len = embeddings.dim(1)?;
        if prompt_len > 1 {
            let previous = prompt_hidden.narrow(1, 0, prompt_len - 1)?;
            let shifted = embeddings.narrow(1, 1, prompt_len - 1)?;
            let shifted_positions = [&positions[0][1..], &positions[1][1..], &positions[2][1..]];
            self.mtp.warm_embeddings_mrope(
                &shifted,
                &previous,
                shifted_positions,
                &mut self.cache,
            )?;
        }
        Ok(())
    }

    /// The predictor's attention cache (for tests and diagnostics).
    pub fn cache(&self) -> &Qwen35MtpCache {
        &self.cache
    }
}

impl Proposer for MtpProposer<'_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::Mtp
    }

    fn wants_hidden(&self) -> bool {
        true
    }

    fn warm(&mut self, prompt: &[i32], prompt_hidden: Option<&Tensor>) -> Result<()> {
        let hidden = prompt_hidden.ok_or_else(|| {
            Error::Msg("MtpProposer: the target did not return prompt hidden states".into())
        })?;
        // embed(token[j + 1]) is paired with hidden[j] at position j + 1; the first generated token
        // is paired with the last prompt hidden row at the first draft step. `hidden` holds the
        // rows the target prefilled: every prompt position cold, positions `offset..` when resuming
        // from a prefix-cache boundary at `offset` (whose `H[offset - 1]` the boundary carries).
        let (offset, boundary_row) = match self.resume.take() {
            Some(b) => {
                self.cache = b.cache;
                (b.len, Some(b.hidden))
            }
            None => (0, None),
        };
        let rows = hidden.dim(1)?;
        if offset >= prompt.len() || rows != prompt.len() - offset {
            return Err(Error::Msg(format!(
                "MtpProposer: {rows} hidden rows for a {}-token prompt resumed at {offset}",
                prompt.len()
            )));
        }
        let target_rows = |start: usize, n: usize| -> Result<Tensor> {
            match &boundary_row {
                Some(row) if start + 1 == offset => {
                    if n == 1 {
                        Ok(row.clone())
                    } else {
                        Ok(Tensor::cat(&[row, &hidden.narrow(1, 0, n - 1)?], 1)?)
                    }
                }
                _ => Ok(hidden.narrow(1, start - offset, n)?),
            }
        };
        let first = offset.max(1);
        let split = self
            .capture_at
            .take()
            .filter(|&b| b > offset && b < prompt.len());
        let mut segments = Vec::with_capacity(2);
        let mut from = first;
        if let Some(b) = split {
            segments.push(from..b);
            from = b;
        }
        segments.push(from..prompt.len());
        for segment in segments {
            if !segment.is_empty() {
                let previous = target_rows(segment.start - 1, segment.len())?;
                self.mtp.warm_sequence(
                    &prompt[segment.clone()],
                    &previous,
                    segment.start as i32,
                    &mut self.cache,
                )?;
            }
            if split == Some(segment.end) {
                self.captured = Some(MtpBoundary {
                    cache: self.cache.clone(),
                    hidden: target_rows(segment.end - 1, 1)?.copy()?,
                    len: segment.end,
                });
            }
        }
        Ok(())
    }

    fn propose(
        &mut self,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let previous_hidden = ctx.previous_hidden.ok_or_else(|| {
            Error::Msg("MtpProposer: no previous target hidden row for the draft step".into())
        })?;
        let k = ctx.max_drafts;
        let (mut draft_logits, mut feedback) = self.mtp.step_ids(
            ctx.cur_ids,
            previous_hidden,
            0,
            ctx.position,
            &mut self.cache,
        )?;
        self.after_cur = Some(self.cache.clone());

        if sampler.device_greedy() {
            // Drafts stay on the device: argmax -> [1, 1] ids -> next draft step.
            let mut drafts = Vec::with_capacity(k);
            for step in 0..k {
                let draft = argmax_rows_tensor(&draft_logits)?.reshape((1, 1))?;
                if step + 1 < k {
                    (draft_logits, feedback) = self.mtp.step_ids(
                        &draft,
                        &feedback,
                        step + 1,
                        ctx.position + step as i32 + 1,
                        &mut self.cache,
                    )?;
                }
                drafts.push(draft);
            }
            let refs: Vec<&Tensor> = drafts.iter().collect();
            return Ok(Proposal {
                drafts: Some(Drafts::Device(Tensor::cat(&refs, 1)?)),
                dists: Vec::new(),
            });
        }

        let mut drafts = Vec::with_capacity(k);
        let mut dists = Vec::with_capacity(k);
        let mut draft_history = ctx.history.to_vec();
        for step in 0..k {
            let (draft, dist) = sampler.sample_draft(&draft_logits, &draft_history)?;
            if let Some(dist) = dist {
                dists.push(dist);
            }
            drafts.push(draft);
            draft_history.push(draft);
            if sampler.is_stop(draft) {
                break;
            }
            if step + 1 < k {
                (draft_logits, feedback) = self.mtp.step(
                    draft,
                    &feedback,
                    step + 1,
                    ctx.position + step as i32 + 1,
                    &mut self.cache,
                )?;
            }
        }
        Ok(Proposal {
            drafts: Some(Drafts::Host(drafts)),
            dists,
        })
    }

    fn commit(
        &mut self,
        _cur: i32,
        accepted: &[i32],
        kept_hidden: Option<&Tensor>,
        position: i32,
    ) -> Result<()> {
        // Replace the recursive draft state with target-confirmed state: the snapshot after `cur`,
        // then each accepted draft replayed against the preceding target hidden row.
        if let Some(after_cur) = self.after_cur.take() {
            self.cache = after_cur;
        }
        if !accepted.is_empty() {
            let kept = kept_hidden.ok_or_else(|| {
                Error::Msg("MtpProposer: no target hidden rows for the accepted drafts".into())
            })?;
            let preceding = kept.narrow(1, 0, accepted.len())?;
            self.mtp
                .warm_sequence(accepted, &preceding, position, &mut self.cache)?;
        }
        Ok(())
    }
}

/// Prompt lookup: propose the continuation that followed the most recent earlier occurrence of
/// the trailing n-gram ([`ngram_propose`]). No model, no tensors; drafts are point masses for the
/// stochastic acceptance rule.
#[derive(Clone, Copy, Debug)]
pub struct NgramProposer {
    /// Longest trailing n-gram to try matching (longest first).
    pub max_ngram: usize,
}

impl Default for NgramProposer {
    fn default() -> Self {
        Self { max_ngram: 3 }
    }
}

impl Proposer for NgramProposer {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }

    fn warm(&mut self, _: &[i32], _: Option<&Tensor>) -> Result<()> {
        Ok(())
    }

    fn propose(
        &mut self,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let drafts = ngram_propose(ctx.history, self.max_ngram, ctx.max_drafts);
        let dists = if sampler.greedy() {
            Vec::new()
        } else {
            drafts.iter().map(|&d| vec![(d, 1.0)]).collect()
        };
        Ok(Proposal {
            drafts: Some(Drafts::Host(drafts)),
            dists,
        })
    }

    fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
        Ok(())
    }
}

/// A separate draft model — any [`StepModel`] with a vocabulary compatible with the target's —
/// proposing `K` tokens from its own distribution `q`. Its cache is kept target-synced: the draft
/// steps feed `[cur, d₁ … dₖ]` (so it holds one position past the last draft, like the target's
/// verify), and on commit it rolls back to `start + 1 + accepted` — directly, or via its own step
/// start plus a replay when its cache only checkpoints step starts (it asks the cache to retain
/// `K + 2` checkpoints so that start survives the draft steps; a hybrid draft therefore holds
/// `K + 3` recurrent states, which whoever admits a draft-model request must price).
///
/// A draft padded differently from its target ([`with_vocab`](Self::with_vocab)) proposes only
/// its tokenizer's ids, over logits the target's width.
pub struct DraftModelProposer<'a, D: StepModel> {
    draft: &'a D,
    cache: Option<D::Cache>,
    budget: usize,
    max_drafts: usize,
    base: i32,
    /// Leading draft ids a draft may be drawn from (the tokenizer's tokens the draft scores).
    proposable: usize,
    /// The target's logits width, which the draft's logits are shaped to.
    width: usize,
    /// Draft-model forwards issued (its own cost, separate from the target's).
    pub draft_forwards: u64,
}

impl<'a, D: StepModel> DraftModelProposer<'a, D> {
    /// A proposer over `draft` for a request of at most `prompt_len + max_new_tokens` positions
    /// proposing up to `max_drafts` per step (the draft cache is sized for that bound plus the
    /// `K + 1` overshoot).
    pub fn new(draft: &'a D, capacity: usize, max_drafts: usize) -> Self {
        Self {
            draft,
            cache: None,
            budget: capacity,
            max_drafts,
            base: 0,
            proposable: draft.vocab_size(),
            width: draft.vocab_size(),
            draft_forwards: 0,
        }
    }

    /// Draw drafts only from the first `proposable` draft ids, over logits `width` wide — the
    /// target's — with every other id at `-inf`: a draft whose padding rows differ from its
    /// target's ([`core_llm::draft_compatibility`]) never proposes a padding id, and its `q`
    /// covers the target's id space.
    pub fn with_vocab(mut self, proposable: usize, width: usize) -> Self {
        self.proposable = proposable.min(self.draft.vocab_size()).min(width);
        self.width = width;
        self
    }

    /// The draft's `[1, vocab]` logits over the target's ids.
    pub(crate) fn shaped(&self, logits: Tensor) -> Result<Tensor> {
        if self.proposable == self.width && self.width == self.draft.vocab_size() {
            return Ok(logits);
        }
        let last = logits.rank() - 1;
        let kept = logits.narrow(last, 0, self.proposable)?;
        if self.width == self.proposable {
            return Ok(kept);
        }
        let mut pad = logits.dims().to_vec();
        pad[last] = self.width - self.proposable;
        let pad =
            Tensor::full(f32::NEG_INFINITY, pad, logits.device())?.to_dtype(logits.dtype())?;
        Ok(Tensor::cat(&[&kept, &pad], last)?)
    }

    fn take_cache(&mut self) -> Result<D::Cache> {
        self.cache
            .take()
            .ok_or_else(|| Error::Msg("DraftModelProposer: propose before warm".into()))
    }
}

impl<D: StepModel> Proposer for DraftModelProposer<'_, D> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::DraftModel
    }

    fn vocab_size(&self) -> Option<usize> {
        Some(self.width)
    }

    fn warm(&mut self, prompt: &[i32], _: Option<&Tensor>) -> Result<()> {
        let mut cache = self.draft.new_cache_for(self.budget, self.max_drafts + 1)?;
        // The K + 1 single-token draft steps each start a forward; the step start they must
        // roll back to has to survive them.
        cache.retain_checkpoints(self.max_drafts + 2)?;
        // A prompt prefill: every rollback stops at a proposal's start, past it (sc-24446).
        self.draft
            .forward_step(&mut cache, StepRequest::last(prompt).as_prefill())?;
        self.draft_forwards += 1;
        self.cache = Some(cache);
        Ok(())
    }

    fn propose(
        &mut self,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let k = ctx.max_drafts;
        let device = self.draft.device().clone();
        let mut cache = self.take_cache()?;
        self.base = cache.len();
        let result = self.propose_into(&mut cache, ctx, sampler, k, &device);
        self.cache = Some(cache);
        result
    }

    fn commit(&mut self, cur: i32, accepted: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
        let base = self.base;
        let mut cache = self.take_cache()?;
        // A host-sampled run stops drafting at a stop token without feeding it, so an accepted
        // stop leaves the cache one short of `base + 1 + accepted`: keep what was fed (the run
        // ends at that stop).
        let target = (base + 1 + accepted.len() as i32).min(cache.len());
        let result = match cache.rollback_to(target) {
            Ok(()) => Ok(()),
            Err(Error::RollbackUnavailable { .. }) => cache.rollback_to(base).and_then(|()| {
                let mut replay = Vec::with_capacity(1 + accepted.len());
                replay.push(cur);
                replay.extend_from_slice(accepted);
                self.draft
                    .forward_step(&mut cache, StepRequest::last(&replay))?;
                self.draft_forwards += 1;
                Ok(())
            }),
            Err(e) => Err(e),
        };
        self.cache = Some(cache);
        result
    }
}

impl<D: StepModel> DraftModelProposer<'_, D> {
    fn propose_into(
        &mut self,
        cache: &mut D::Cache,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
        k: usize,
        device: &candle_core::Device,
    ) -> Result<Proposal> {
        if sampler.device_greedy() {
            let mut drafts: Vec<Tensor> = Vec::with_capacity(k);
            let mut feed = ctx.cur_ids.clone();
            for step in 0..=k {
                let logits = self
                    .draft
                    .forward_step(cache, StepRequest::last_ids(&feed))?
                    .logits;
                self.draft_forwards += 1;
                if step < k {
                    feed = argmax_rows_tensor(&self.shaped(logits)?)?.reshape((1, 1))?;
                    drafts.push(feed.clone());
                }
            }
            let refs: Vec<&Tensor> = drafts.iter().collect();
            return Ok(Proposal {
                drafts: Some(Drafts::Device(Tensor::cat(&refs, 1)?)),
                dists: Vec::new(),
            });
        }

        let mut drafts = Vec::with_capacity(k);
        let mut dists = Vec::with_capacity(k);
        let mut draft_history = ctx.history.to_vec();
        let mut feed = ctx.cur;
        for step in 0..=k {
            let ids = crate::primitives::input_ids(&[feed], device)?;
            let logits = self
                .draft
                .forward_step(cache, StepRequest::last_ids(&ids))?
                .logits;
            self.draft_forwards += 1;
            if step < k {
                let (d, dist) = sampler.sample_draft(&self.shaped(logits)?, &draft_history)?;
                if let Some(dist) = dist {
                    dists.push(dist);
                }
                drafts.push(d);
                draft_history.push(d);
                feed = d;
                if sampler.is_stop(d) {
                    break;
                }
            }
        }
        Ok(Proposal {
            drafts: Some(Drafts::Host(drafts)),
            dists,
        })
    }
}
