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
pub struct MtpProposer<'a> {
    mtp: &'a Qwen35Mtp,
    cache: Qwen35MtpCache,
    after_cur: Option<Qwen35MtpCache>,
}

impl<'a> MtpProposer<'a> {
    /// A proposer over `mtp` with a fresh predictor cache.
    pub fn new(mtp: &'a Qwen35Mtp) -> Self {
        Self {
            mtp,
            cache: mtp.new_cache(),
            after_cur: None,
        }
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
        // embed(token[j + 1]) is paired with hidden[j]; the first generated token is paired with
        // the last prompt hidden row at the first draft step.
        if prompt.len() > 1 {
            let previous = hidden.narrow(1, 0, prompt.len() - 1)?;
            self.mtp
                .warm_sequence(&prompt[1..], &previous, 1, &mut self.cache)?;
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
        ProposerKind::Ngram
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
pub struct DraftModelProposer<'a, D: StepModel> {
    draft: &'a D,
    cache: Option<D::Cache>,
    budget: usize,
    max_drafts: usize,
    base: i32,
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
            draft_forwards: 0,
        }
    }

    fn take_cache(&mut self) -> Result<D::Cache> {
        self.cache
            .take()
            .ok_or_else(|| Error::Msg("DraftModelProposer: propose before warm".into()))
    }
}

impl<D: StepModel> Proposer for DraftModelProposer<'_, D> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::Draft
    }

    fn warm(&mut self, prompt: &[i32], _: Option<&Tensor>) -> Result<()> {
        let mut cache = self.draft.new_cache_for(self.budget, self.max_drafts + 1)?;
        // The K + 1 single-token draft steps each start a forward; the step start they must
        // roll back to has to survive them.
        cache.retain_checkpoints(self.max_drafts + 2);
        self.draft
            .forward_step(&mut cache, StepRequest::last(prompt))?;
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
        let target = base + 1 + accepted.len() as i32;
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
                    feed = argmax_rows_tensor(&logits)?.reshape((1, 1))?;
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
                let (d, dist) = sampler.sample_draft(&logits, &draft_history)?;
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
