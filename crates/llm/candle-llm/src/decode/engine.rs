//! The unified speculative engine (epic sc-24128, story sc-24130).
//!
//! **One** speculative loop over the [`StepModel`] / [`DecodeCache`] seams, with the proposal
//! source behind the [`Proposer`] trait — the native MTP head, prompt lookup (n-gram) and a draft
//! model ([`proposers`](super::proposers)). It replaces the Qwen-only `qwen_mtp` loop (whose
//! verify / accept / rollback logic moved here) and is the only speculative path: the pre-epic
//! `CausalLm` prompt-lookup and draft-model loops were retired when the llama family moved onto
//! the seams (S10, sc-24138), their proposers living in [`proposers`](super::proposers).
//!
//! ## One step
//! 1. **Propose** `K` drafts after the current token `cur` (the last committed token, not yet in
//!    the cache). A greedy proposer keeps its drafts **on the device** — each draft step's argmax
//!    tensor feeds the next draft step as ids — so proposing issues no device->host transfer.
//! 2. **Verify** `[cur, d₁ … dₖ]` in one target step ([`LogitsScope::All`]); the cache now holds
//!    `K + 1` provisional positions past the step start.
//! 3. **Decide** on host from **one** transfer: the target's per-position argmax (computed on the
//!    device) concatenated with the device drafts, `2K + 1` integers, for a plain greedy run; or
//!    the `K + 1` logit rows in one copy when penalties, a constraint or stochastic sampling need
//!    them ([`logits_rows_host`]). The policy is `core_llm`'s: [`greedy_commit`] for greedy,
//!    [`accept_token`] (the distribution-preserving `p/q` rule, unchanged) for stochastic. This is
//!    the AC2 figure: exactly one host sync per verify step, where the old loop paid `K + 1`.
//!    A step with **no** drafts (the [`NoProposer`] run, or a step the budget clamped to `K = 0`)
//!    has no decision to make: its one row is an ordinary [`sample`] draw — the device sampler
//!    for a temperature / top-p request — exactly the token-at-a-time loop's (sc-24140).
//! 4. **Recover**: keep the accepted prefix. The engine first asks the cache to roll back to
//!    `start + 1 + accepted` — a **direct rollback**, which every cache with per-token rollback
//!    (a softmax-only cache, the `Qwen35Cache` with its per-token DeltaNet checkpoint ring since
//!    S3, sc-24131) answers at no forward cost. A cache that cannot answers
//!    [`Error::RollbackUnavailable`], and the engine then rolls back to the step start and
//!    **replays** `[cur, accepted drafts…]` in one forward — the **replay fallback**, exactly
//!    what the `qwen_mtp` loop did with a cloned cache, so the acceptance statistics are
//!    unchanged. Both are counted ([`DecodeRecord::direct_rollbacks`] /
//!    [`DecodeRecord::replay_forwards`]), so a run says which recovery it paid for.
//! 5. **Commit** through the same event path as every other loop (`StreamEvent::Token`, stop
//!    tokens, constraint advance, caller stop, cancellation, budget — E7). A cancel observed right
//!    after the verify forward rolls the cache back to the step start and returns `Cancelled`
//!    before any token of that step is committed, as the old loop did. Device-resident drafts
//!    cannot stop at a stop token while drafting, so the decision truncates them at the first stop
//!    token (the host-sampled proposers stop there themselves): nothing past an accepted stop
//!    token is counted as proposed or accepted.
//!
//! The record names the proposer that ran ([`DecodeRecord::proposer`]) and counts verify steps
//! and the host syncs spent inside them ([`DecodeRecord::host_syncs_per_verify_step`]); a run with
//! [`NoProposer`] is the token-at-a-time loop and reports `proposer=none`.
//!
//! [`logits_rows_host`]: crate::primitives::sampler::logits_rows_host
//! [`greedy_commit`]: core_llm::speculative::greedy_commit
//! [`accept_token`]: core_llm::speculative::accept_token

use candle_core::Tensor;
use core_llm::speculative::{accept_token, greedy_commit, Acceptance};
use core_llm::{ProposerKind, SamplerPath};

use crate::decode::cancel::CancelFlag;
use crate::decode::record::{DecodePath, DecodeRecord, RequestSpan};
use crate::decode::speculative::SpeculativeStats;
use crate::decode::step::{LogitsScope, StepModel, StepRequest, StepTokens};
use crate::decode::stream::{
    default_seed, ConstraintMask, FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};
use crate::error::{Error, Result};
use crate::primitives::decode_cache::{CacheMemory, DecodeCache};
use crate::primitives::host_sync::{host_sync_count, note_sampler_path};
use crate::primitives::input_ids;
use crate::primitives::sampler::{
    acceptance_target_host, argmax_rows_tensor, host_row_reason, logits_rows_host, sample,
    sample_row_host, shaped_candidates, SplitMix64, TokenRng,
};

/// Constraint state that can be rewound after speculative exploration. The committed state is
/// advanced only by tokens that are actually emitted.
pub trait RewindableConstraintMask: ConstraintMask {
    /// Opaque checkpoint for the current constraint state.
    fn checkpoint(&self) -> usize;
    /// Restore a checkpoint previously returned by [`Self::checkpoint`].
    fn rewind(&mut self, checkpoint: usize);
}

/// A proposer's drafts: host ids, or a `[1, K]` `u32` tensor that never left the device (the
/// greedy fast path — the ids reach the host inside the verify decision's single transfer).
#[derive(Clone, Debug)]
pub enum Drafts {
    /// Draft ids on the host.
    Host(Vec<i32>),
    /// Draft ids on the device, `[1, K]` `u32`.
    Device(Tensor),
}

impl Drafts {
    /// How many drafts were proposed.
    pub fn len(&self) -> Result<usize> {
        Ok(match self {
            Drafts::Host(d) => d.len(),
            Drafts::Device(t) => t.dims2()?.1,
        })
    }

    /// `len() == 0`.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
}

/// What a proposer returns for one step.
#[derive(Clone, Debug, Default)]
pub struct Proposal {
    /// The proposed continuation after `cur`, in order.
    pub drafts: Option<Drafts>,
    /// For a stochastic run, the proposal distribution `q` each draft was drawn from (the
    /// `(token, weight)` candidate set), one per draft — what [`accept_token`] needs. Empty for a
    /// greedy run.
    pub dists: Vec<Vec<(i32, f32)>>,
}

/// What a proposer sees when asked for drafts.
pub struct ProposeContext<'a> {
    /// The last committed token — the first token the verify step feeds — not yet in the cache.
    pub cur: i32,
    /// `cur` as a `[1, 1]` `u32` tensor on the model's device.
    pub cur_ids: &'a Tensor,
    /// The prompt plus every committed token (ends with `cur`) — the n-gram context and the
    /// repetition-penalty window.
    pub history: &'a [i32],
    /// The target's final-normalized hidden row for the position before `cur` (what a native MTP
    /// head pairs `cur`'s embedding with), when the proposer asked for hidden states.
    pub previous_hidden: Option<&'a Tensor>,
    /// The RoPE position `cur` will occupy.
    pub position: i32,
    /// At most this many drafts (already clamped to the remaining budget).
    pub max_drafts: usize,
}

/// One host-sampled draft: the token and, for a stochastic run, the shaped proposal distribution
/// it was drawn from.
pub type DraftSample = (i32, Option<Vec<(i32, f32)>>);

/// The engine's draft-sampling policy handed to a proposer: the sampling knobs, the seeded RNG
/// and the (rewindable) constraint, plus whether drafts may stay on the device.
pub struct DraftSampler<'a, 'c> {
    config: &'a GenerationConfig,
    rng: &'a mut SplitMix64,
    constraint: Option<&'a mut (dyn RewindableConstraintMask + 'c)>,
    device_greedy: bool,
}

impl DraftSampler<'_, '_> {
    /// True when drafts may be taken as the on-device argmax and stay there: plain greedy (no
    /// temperature, no penalties) with no constraint. Otherwise drafts are sampled on the host
    /// through [`sample_draft`](Self::sample_draft).
    pub fn device_greedy(&self) -> bool {
        self.device_greedy
    }

    /// Whether the run is greedy at all (`temperature <= 0`).
    pub fn greedy(&self) -> bool {
        self.config.sampling.temperature <= 0.0
    }

    /// Whether `token` is a stop token — a proposer stops drafting past one.
    pub fn is_stop(&self, token: i32) -> bool {
        self.config.stop_tokens.contains(&token)
    }

    /// Sample one draft on the host from `[1, vocab]` `logits` given the provisional
    /// `draft_history`, returning the token and — for a stochastic run — the shaped proposal
    /// distribution it was drawn from. Advances the constraint by the draft unless it is a stop
    /// token (the engine rewinds the constraint after the proposal either way).
    pub fn sample_draft(&mut self, logits: &Tensor, draft_history: &[i32]) -> Result<DraftSample> {
        let dist = if self.greedy() {
            None
        } else {
            let mask = self.constraint.as_mut().map(|c| c.allowed());
            Some(shaped_candidates(
                logits,
                draft_history,
                &self.config.sampling,
                mask,
            )?)
        };
        let mask = self.constraint.as_mut().map(|c| c.allowed());
        let draft = sample(logits, draft_history, &self.config.sampling, self.rng, mask)?;
        if !self.is_stop(draft) {
            if let Some(c) = self.constraint.as_mut() {
                c.accept(draft);
            }
        }
        Ok((draft, dist))
    }
}

/// A proposal source for the engine. Implementations: [`MtpProposer`](super::proposers::MtpProposer),
/// [`NgramProposer`](super::proposers::NgramProposer),
/// [`DraftModelProposer`](super::proposers::DraftModelProposer) and [`NoProposer`].
pub trait Proposer {
    /// Which proposer this is (stamped on the record).
    fn kind(&self) -> ProposerKind;

    /// Whether the target's final-normalized hidden states are needed (the MTP head pairs them
    /// with the next token); when `false` the target skips returning them.
    fn wants_hidden(&self) -> bool {
        false
    }

    /// The vocabulary the proposer draws its drafts from, when it is a separate model — the
    /// engine refuses a proposer whose vocabulary is not the target's before any inference
    /// (a draft id past the target's vocabulary, or one naming a different token, would be
    /// verified as garbage). `None` (the default) for a proposer that shares the target's
    /// vocabulary by construction (MTP, n-gram).
    fn vocab_size(&self) -> Option<usize> {
        None
    }

    /// Warm from the prefilled prompt: `prompt` is the effective prompt ids, `prompt_hidden` the
    /// target's hidden rows for every prompt position when [`wants_hidden`](Self::wants_hidden).
    fn warm(&mut self, prompt: &[i32], prompt_hidden: Option<&Tensor>) -> Result<()>;

    /// Propose up to `ctx.max_drafts` drafts after `ctx.cur`. Called only when `max_drafts > 0`.
    fn propose(
        &mut self,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal>;

    /// The verify outcome: `accepted` is the accepted draft prefix (host ids); `kept_hidden` the
    /// target's hidden rows `[1, 1 + accepted.len(), hidden]` for `[cur, accepted…]` when
    /// [`wants_hidden`](Self::wants_hidden); `position` the RoPE position of `accepted[0]`
    /// (`cur`'s + 1). Rejected drafts are simply absent — the proposer drops any state it built
    /// for them.
    fn commit(
        &mut self,
        cur: i32,
        accepted: &[i32],
        kept_hidden: Option<&Tensor>,
        position: i32,
    ) -> Result<()>;
}

/// No proposal source: the engine degenerates to the token-at-a-time loop and the record says
/// `proposer=none`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProposer;

impl Proposer for NoProposer {
    fn kind(&self) -> ProposerKind {
        ProposerKind::None
    }

    fn warm(&mut self, _: &[i32], _: Option<&Tensor>) -> Result<()> {
        Ok(())
    }

    fn propose(
        &mut self,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        Ok(Proposal::default())
    }

    fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
        Ok(())
    }
}

/// How the engine starts: a token prompt it prefills itself, or a cache the caller has already
/// prefilled (the multimodal path, whose prefill is spliced embeddings under M-RoPE).
pub enum SpeculativePrompt<'a, C> {
    /// Prefill these ids through [`StepModel::forward_step`] into a cache the engine builds with
    /// [`StepModel::new_cache_for`] (bound: prompt + budget, overshoot `K`).
    Tokens(&'a [i32]),
    /// A caller-prefilled cache positioned past the prompt.
    Prefilled {
        /// The cache, borrowed: the caller still owns it afterwards.
        cache: &'a mut C,
        /// Last-position logits of the prefill, `[1, vocab]`.
        logits: Tensor,
        /// The target's hidden rows for every prompt position `[1, prompt, hidden]` when the
        /// proposer wants them (the MTP head is warmed from them by the caller or here).
        hidden: Option<Tensor>,
        /// The effective prompt ids (the repetition-penalty window / n-gram context).
        history: &'a [i32],
        /// Shift between cache positions and RoPE positions for the continuation (the M-RoPE
        /// `mrope_delta`; `0` for text). The cache applies it inside `forward_step`; the engine
        /// only needs it to tell the proposer the position of `cur`.
        position_delta: i32,
        /// Whether the proposer still needs [`Proposer::warm`] over the prompt (`false` when the
        /// caller warmed it itself, e.g. from fused visual embeddings).
        warm_proposer: bool,
    },
}

/// A finished engine run.
#[derive(Clone, Debug)]
pub struct SpeculativeRun {
    /// The generated tokens and why generation stopped.
    pub output: GenerationOutput,
    /// The measured record (`path` names the proposer's path, `proposer` the proposer).
    pub record: DecodeRecord,
    /// The raw speculation counters.
    pub stats: SpeculativeStats,
    /// The final cache's own accounting (E6).
    pub memory: CacheMemory,
}

/// Generate through the engine. Returns [`Error::Canceled`] if `cancel` is already set before any
/// inference; a mid-run cancel returns the partial output as [`FinishReason::Cancelled`].
///
/// `drafts` is the proposal width `K`; `0` (or [`NoProposer`]) is the token-at-a-time loop.
#[allow(clippy::too_many_arguments)]
pub fn generate_speculative<M: StepModel + ?Sized, P: Proposer + ?Sized>(
    model: &M,
    proposer: &mut P,
    prompt: SpeculativePrompt<'_, M::Cache>,
    config: &GenerationConfig,
    drafts: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
) -> Result<SpeculativeRun> {
    generate_speculative_with(
        model, proposer, prompt, config, drafts, cancel, on_event, constraint, None, None,
    )
}

/// [`generate_speculative`] with the provider seams: a cooperative caller stop (checked before
/// each step and after each emitted token) and a prefill boundary callback invoked once the prompt
/// is in the cache and the proposer is warmed, before the first token is sampled (it may
/// synchronize the device).
#[allow(clippy::too_many_arguments)]
pub fn generate_speculative_with<M: StepModel + ?Sized, P: Proposer + ?Sized>(
    model: &M,
    proposer: &mut P,
    prompt: SpeculativePrompt<'_, M::Cache>,
    config: &GenerationConfig,
    drafts: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    mut constraint: Option<&mut dyn RewindableConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
    mut on_prefill_complete: Option<&mut dyn FnMut() -> Result<()>>,
) -> Result<SpeculativeRun> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    let span = RequestSpan::begin();
    let mut stats = SpeculativeStats::default();
    let mut verify_host_syncs = 0u64;
    let device = model.device();
    if let Some(draft_vocab) = proposer.vocab_size() {
        if draft_vocab != model.vocab_size() {
            return Err(Error::Msg(format!(
                "draft/target vocab mismatch: draft {draft_vocab} vs target {}",
                model.vocab_size()
            )));
        }
    }
    let wants_hidden = proposer.wants_hidden();
    let kind = proposer.kind();

    // ---- Prefill (or adopt the caller's), warm the proposer. ----
    let mut owned: Option<M::Cache> = None;
    let (cache, logits, mut previous_hidden, mut history, position_delta): (
        &mut M::Cache,
        Tensor,
        Option<Tensor>,
        Vec<i32>,
        i32,
    ) = match prompt {
        SpeculativePrompt::Tokens(prompt_ids) => {
            if prompt_ids.is_empty() {
                return Err(Error::Msg("generate_speculative: empty prompt".into()));
            }
            let capacity = prompt_ids.len().saturating_add(config.max_new_tokens);
            let cache = owned.insert(model.new_cache_for(capacity, drafts)?);
            let out = model.forward_step(
                cache,
                StepRequest::last(prompt_ids).with_hidden(wants_hidden),
            )?;
            stats.forwards += 1;
            stats.prefill_forwards += 1;
            let previous_hidden = last_row(out.hidden.as_ref())?;
            proposer.warm(prompt_ids, out.hidden.as_ref())?;
            (cache, out.logits, previous_hidden, prompt_ids.to_vec(), 0)
        }
        SpeculativePrompt::Prefilled {
            cache,
            logits,
            hidden,
            history: prompt_ids,
            position_delta,
            warm_proposer,
        } => {
            if prompt_ids.is_empty() {
                return Err(Error::Msg("generate_speculative: empty prompt".into()));
            }
            // The caller's prefill: still one of the request's target forwards.
            stats.forwards += 1;
            stats.prefill_forwards += 1;
            let previous_hidden = last_row(hidden.as_ref())?;
            if warm_proposer {
                proposer.warm(prompt_ids, hidden.as_ref())?;
            }
            (
                cache,
                logits,
                previous_hidden,
                prompt_ids.to_vec(),
                position_delta,
            )
        }
    };
    if let Some(boundary) = on_prefill_complete.as_mut() {
        boundary()?;
    }

    let mut rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let greedy = config.sampling.temperature <= 0.0;
    let plain_greedy = config.sampling.is_plain_greedy();
    let mut generated: Vec<i32> = Vec::new();
    let mut finish = FinishReason::MaxTokens;
    let done = |generated: Vec<i32>,
                finish: FinishReason,
                stats: SpeculativeStats,
                verify_host_syncs: u64,
                cache: &M::Cache,
                on_event: &mut dyn FnMut(StreamEvent)| {
        on_event(StreamEvent::Done {
            reason: finish,
            generated: generated.len(),
        });
        let path = match kind {
            ProposerKind::None => DecodePath::StepModel,
            ProposerKind::Mtp => DecodePath::Mtp {
                drafts: u32::try_from(drafts).unwrap_or(u32::MAX),
            },
            ProposerKind::Ngram => DecodePath::PromptLookup,
            ProposerKind::Draft => DecodePath::DraftModel,
        };
        let record = DecodeRecord::speculative(path, stats, generated.len(), span.counters())
            .with_kv_cache(cache.kv_kind())
            .with_attn_formulation(model.attn_formulation(cache))
            .with_proposer(kind)
            .with_verify_syncs(verify_host_syncs)
            .with_span_tallies(&span);
        SpeculativeRun {
            output: GenerationOutput {
                tokens: generated,
                finish_reason: finish,
            },
            record,
            stats,
            memory: cache.memory(),
        }
    };
    if config.max_new_tokens == 0 {
        return Ok(done(
            generated,
            finish,
            stats,
            verify_host_syncs,
            cache,
            on_event,
        ));
    }

    // ---- First token: ordinary sampling from the prefill logits (one sync, as every loop). ----
    let first = {
        let mask = constraint.as_mut().map(|c| c.allowed());
        sample(&logits, &history, &config.sampling, &mut rng, mask)?
    };
    if config.stop_tokens.contains(&first) {
        finish = FinishReason::StopToken;
        return Ok(done(
            generated,
            finish,
            stats,
            verify_host_syncs,
            cache,
            on_event,
        ));
    }
    if let Some(c) = constraint.as_mut() {
        c.accept(first);
    }
    on_event(StreamEvent::Token { id: first, step: 0 });
    generated.push(first);
    history.push(first);
    let mut cur = first;
    if should_stop.is_some_and(|stop| stop()) {
        finish = FinishReason::Stopped;
    }

    // ---- Speculative steps. ----
    'outer: while generated.len() < config.max_new_tokens && finish != FinishReason::Stopped {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }
        if should_stop.is_some_and(|stop| stop()) {
            finish = FinishReason::Stopped;
            break;
        }
        let step_syncs = host_sync_count();
        let remaining = config.max_new_tokens - generated.len();
        let k = drafts.min(remaining.saturating_sub(1));
        let base = cache.len();
        let position = base + position_delta;
        let cur_ids = input_ids(&[cur], device)?;
        let constraint_checkpoint = constraint.as_ref().map(|c| c.checkpoint());

        // 1. Propose.
        let proposal = if k == 0 {
            Proposal::default()
        } else {
            let ctx = ProposeContext {
                cur,
                cur_ids: &cur_ids,
                history: &history,
                previous_hidden: previous_hidden.as_ref(),
                position,
                max_drafts: k,
            };
            let device_greedy = plain_greedy && constraint.is_none();
            let mut sampler = DraftSampler {
                config,
                rng: &mut rng,
                constraint: constraint.as_deref_mut(),
                device_greedy,
            };
            proposer.propose(&ctx, &mut sampler)?
        };
        if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), constraint_checkpoint) {
            c.rewind(checkpoint);
        }
        let drafted = proposal.drafts.unwrap_or(Drafts::Host(Vec::new()));
        let num_drafts = drafted.len()?;
        stats.proposed += num_drafts;
        if cancel.is_cancelled() {
            // Nothing has been written to the target cache yet.
            finish = FinishReason::Cancelled;
            break;
        }

        // 2. Verify `[cur, drafts…]` in one target step.
        let verify_ids = match &drafted {
            Drafts::Host(d) => {
                let mut verify = Vec::with_capacity(1 + d.len());
                verify.push(cur);
                verify.extend_from_slice(d);
                input_ids(&verify, device)?
            }
            Drafts::Device(d) => Tensor::cat(&[&cur_ids, d], 1)?,
        };
        let out = model.forward_step(
            cache,
            StepRequest {
                tokens: StepTokens::Device(&verify_ids),
                scope: LogitsScope::All,
                want_hidden: wants_hidden,
            },
        )?;
        stats.forwards += 1;
        stats.verify_steps += 1;
        if cancel.is_cancelled() {
            // The old loop checked right after the verify forward too (E7): nothing of this step
            // is committed; the cache goes back to the step start.
            cache.rollback_to(base)?;
            finish = FinishReason::Cancelled;
            break;
        }

        // 3. Decide, on host, from one transfer.
        let (draft_ids, committed, accepted) = decide(
            &out.logits,
            &drafted,
            &proposal.dists,
            &history,
            config,
            &mut rng,
            greedy,
            plain_greedy,
            constraint.as_deref_mut(),
        )?;
        if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), constraint_checkpoint) {
            c.rewind(checkpoint);
        }
        // Drafts past a stop token were never a proposal (the decision dropped them); the cache
        // still holds all `num_drafts + 1` verify positions, so `accepted < num_drafts` below
        // rolls it back to the kept prefix.
        stats.proposed -= num_drafts - draft_ids.len();
        stats.accepted += accepted;

        // 4. Recover the cache to `[cur, accepted…]`: a direct rollback when the cache can, else
        //    back to the step start plus a replay of the kept prefix (a cache without per-token
        //    checkpoints).
        let keep_len = 1 + accepted;
        let kept_hidden = if accepted == num_drafts {
            out.hidden
        } else {
            let target = base + keep_len as i32;
            match cache.rollback_to(target) {
                Ok(()) => {
                    stats.direct_rollbacks += 1;
                    match out.hidden {
                        Some(h) => Some(h.narrow(1, 0, keep_len)?),
                        None => None,
                    }
                }
                Err(Error::RollbackUnavailable { .. }) => {
                    cache.rollback_to(base)?;
                    let mut replay = Vec::with_capacity(keep_len);
                    replay.push(cur);
                    replay.extend_from_slice(&draft_ids[..accepted]);
                    let replayed = model.forward_step(
                        cache,
                        StepRequest::last(&replay).with_hidden(wants_hidden),
                    )?;
                    stats.forwards += 1;
                    stats.replays += 1;
                    replayed.hidden
                }
                Err(e) => return Err(e),
            }
        };
        proposer.commit(
            cur,
            &draft_ids[..accepted],
            kept_hidden.as_ref(),
            position + 1,
        )?;
        previous_hidden = last_row(kept_hidden.as_ref())?;
        verify_host_syncs += host_sync_count().wrapping_sub(step_syncs);

        // 5. Commit through the shared event path.
        let mut emitted = 0usize;
        for &token in &committed {
            if config.stop_tokens.contains(&token) {
                finish = FinishReason::StopToken;
                settle(cache, base, emitted, accepted)?;
                break 'outer;
            }
            if let Some(c) = constraint.as_mut() {
                c.accept(token);
            }
            on_event(StreamEvent::Token {
                id: token,
                step: generated.len(),
            });
            generated.push(token);
            history.push(token);
            cur = token;
            emitted += 1;
            if should_stop.is_some_and(|stop| stop()) {
                finish = FinishReason::Stopped;
                settle(cache, base, emitted, accepted)?;
                break 'outer;
            }
            if cancel.is_cancelled() {
                finish = FinishReason::Cancelled;
                settle(cache, base, emitted, accepted)?;
                break 'outer;
            }
            if generated.len() >= config.max_new_tokens {
                finish = FinishReason::MaxTokens;
                // A proposer that honours `max_drafts` cannot leave accepted positions unemitted
                // here (the width is clamped to the budget); one that over-proposes can.
                settle(cache, base, emitted, accepted)?;
                break 'outer;
            }
        }
    }

    Ok(done(
        generated,
        finish,
        stats,
        verify_host_syncs,
        cache,
        on_event,
    ))
}

/// After an early exit inside the commit loop the cache may hold accepted positions whose tokens
/// were never emitted (a stop token, a caller stop or a cancel landed before them). Roll back to
/// the step start — always checkpointed — so the cache never holds a position that is not in the
/// committed history. Past the step start the cache holds `cur` (emitted in the previous step)
/// plus the `accepted` drafts, and this step emitted `emitted` of the committed run, so the cache
/// is ahead of the history exactly when `accepted > emitted`.
fn settle<C: DecodeCache>(cache: &mut C, base: i32, emitted: usize, accepted: usize) -> Result<()> {
    if accepted > emitted {
        cache.rollback_to(base)?;
    }
    Ok(())
}

/// The last sequence row of `[1, n, hidden]`, `[1, 1, hidden]`.
fn last_row(hidden: Option<&Tensor>) -> Result<Option<Tensor>> {
    match hidden {
        Some(h) => {
            let n = h.dim(1)?;
            Ok(Some(h.narrow(1, n - 1, 1)?))
        }
        None => Ok(None),
    }
}

/// The verify decision from one device->host transfer. Returns the drafts as host ids (truncated
/// after the first stop token among them), the committed run (accepted drafts + the bonus /
/// correction token) and the accepted count.
///
/// A step with **no drafts** (the [`NoProposer`] run, or a step whose width the budget clamped to
/// zero) has no decision to make: its one verify row is an ordinary draw through [`sample`] — the
/// device argmax or the device sampler where the request allows it, the host reference where a
/// penalty or a constraint needs the row — exactly the token-at-a-time loop's draw, with the
/// same seeded stream, and recorded by the sampler. With drafts, every host draw is recorded too:
/// each acceptance test and the bonus (sc-24140).
#[allow(clippy::too_many_arguments)]
fn decide(
    logits: &Tensor,
    drafts: &Drafts,
    dists: &[Vec<(i32, f32)>],
    history: &[i32],
    config: &GenerationConfig,
    rng: &mut SplitMix64,
    greedy: bool,
    plain_greedy: bool,
    mut constraint: Option<&mut (dyn RewindableConstraintMask + '_)>,
) -> Result<(Vec<i32>, Vec<i32>, usize)> {
    let n = logits.dim(1)?; // 1 + K verify positions
    if drafts.is_empty()? {
        if n != 1 {
            return Err(Error::Msg(format!(
                "verify returned {n} positions for 0 drafts"
            )));
        }
        let mask = constraint.as_mut().map(|c| c.allowed());
        let token = sample(logits, history, &config.sampling, rng, mask)?;
        return Ok((Vec::new(), vec![token], 0));
    }
    if plain_greedy && constraint.is_none() {
        // The greedy fast path: argmax on device, one transfer of `[argmax (K+1) ‖ drafts (K)]`.
        let argmax = argmax_rows_tensor(logits)?; // [K + 1] u32
        let (target_argmax, draft_ids): (Vec<i32>, Vec<i32>) = match drafts {
            Drafts::Device(d) => {
                let both = Tensor::cat(&[&argmax, &d.flatten_all()?], 0)?;
                crate::primitives::host_sync::note_host_sync();
                let both: Vec<i32> = both
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|t| t as i32)
                    .collect();
                let (a, d) = both.split_at(n);
                (a.to_vec(), d.to_vec())
            }
            Drafts::Host(d) => {
                crate::primitives::host_sync::note_host_sync();
                let argmax: Vec<i32> = argmax
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|t| t as i32)
                    .collect();
                (argmax, d.clone())
            }
        };
        if target_argmax.len() != draft_ids.len() + 1 {
            return Err(Error::Msg(format!(
                "verify returned {} positions for {} drafts",
                target_argmax.len(),
                draft_ids.len()
            )));
        }
        // Device drafting cannot see a stop token; nothing past the first one is a proposal.
        let mut draft_ids = draft_ids;
        if let Some(stop) = draft_ids
            .iter()
            .position(|t| config.stop_tokens.contains(t))
        {
            draft_ids.truncate(stop + 1);
        }
        let (committed, accepted) = greedy_commit(&target_argmax[..=draft_ids.len()], &draft_ids);
        // Every committed token is a device argmax (the sampler telemetry's device path).
        for _ in &committed {
            note_sampler_path(SamplerPath::Device);
        }
        return Ok((draft_ids, committed, accepted));
    }

    // Penalties, a constraint or stochastic sampling: every row in one copy, shaped on host.
    let draft_ids = match drafts {
        Drafts::Host(d) => d.clone(),
        Drafts::Device(d) => {
            crate::primitives::host_sync::note_host_sync();
            d.flatten_all()?
                .to_vec1::<u32>()?
                .into_iter()
                .map(|t| t as i32)
                .collect()
        }
    };
    let rows = logits_rows_host(logits)?;
    if rows.len() != draft_ids.len() + 1 {
        return Err(Error::Msg(format!(
            "verify returned {} positions for {} drafts",
            rows.len(),
            draft_ids.len()
        )));
    }
    let mut rows = rows.into_iter();
    let mut committed = Vec::with_capacity(draft_ids.len() + 1);
    let mut accepted = 0usize;
    let mut running = history.to_vec();
    for (i, &draft) in draft_ids.iter().enumerate() {
        let row = rows.next().expect("one row per draft");
        let mask = constraint.as_mut().map(|c| c.allowed());
        let outcome = if greedy {
            let target = sample_row_host(row, &running, &config.sampling, rng, mask);
            if target == draft {
                Acceptance::Accepted(draft)
            } else {
                Acceptance::Rejected(target)
            }
        } else {
            // The acceptance test is a host draw over the host row: recorded as one.
            note_sampler_path(SamplerPath::Host(host_row_reason(
                &config.sampling,
                mask.is_some(),
            )));
            let target = acceptance_target_host(row, &running, &config.sampling, mask);
            let q = dists.get(i).cloned().unwrap_or_else(|| vec![(draft, 1.0)]);
            accept_token(&target, &q, draft, rng.next_f32(), rng.next_f32())
        };
        let token = outcome.token();
        committed.push(token);
        if outcome.is_accepted() {
            accepted += 1;
        }
        if config.stop_tokens.contains(&token) {
            return Ok((draft_ids, committed, accepted));
        }
        if let Some(c) = constraint.as_mut() {
            c.accept(token);
        }
        running.push(token);
        if !outcome.is_accepted() {
            return Ok((draft_ids, committed, accepted));
        }
    }
    // Every draft accepted: the bonus from the position past the last draft — the host reference
    // draw over the host row (recorded; a row nothing survives the shaping of commits its argmax).
    let row = rows.next().expect("the bonus row");
    let mask = constraint.as_mut().map(|c| c.allowed());
    let bonus = sample_row_host(row, &running, &config.sampling, rng, mask);
    committed.push(bonus);
    Ok((draft_ids, committed, accepted))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use candle_core::{Device, Tensor};
    use core_llm::speculative::sample_weighted;
    use serde_json::json;

    use super::*;
    use crate::decode::proposers::{DraftModelProposer, MtpProposer, NgramProposer};
    use crate::decode::step::{generate_step, StepOutput};
    use crate::decode::stream::generate_with;
    use crate::models::qwen35::tests::{text_model, text_model_with_layers, text_model_with_mtp};
    use crate::models::{Qwen35Config, Qwen35Model, Qwen35Mtp};
    use crate::primitives::decode_cache::CacheMemory;
    use crate::primitives::sampler::SamplingParams;
    use crate::primitives::Weights;

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            sampling: SamplingParams::default(),
            seed: Some(0),
            stop_tokens: Vec::new(),
        }
    }

    fn stochastic(max_new_tokens: usize) -> GenerationConfig {
        let mut config = greedy(max_new_tokens);
        config.sampling.temperature = 0.8;
        config.sampling.top_p = 0.9;
        config.sampling.top_k = 6;
        config
    }

    fn run<M: StepModel, P: Proposer>(
        model: &M,
        proposer: &mut P,
        prompt: &[i32],
        config: &GenerationConfig,
        drafts: usize,
    ) -> SpeculativeRun {
        generate_speculative(
            model,
            proposer,
            SpeculativePrompt::Tokens(prompt),
            config,
            drafts,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap()
    }

    fn step_tokens(model: &Qwen35Model, prompt: &[i32], config: &GenerationConfig) -> Vec<i32> {
        generate_step(model, prompt, config, &CancelFlag::new(), &mut |_| {}, None)
            .unwrap()
            .0
            .tokens
    }

    const PROMPT: [i32; 6] = [3, 7, 11, 2, 7, 11];

    // ---- Parity: every proposer, every K, equals the token-at-a-time driver and the reference ----

    #[test]
    fn mtp_engine_is_token_identical_to_the_step_driver_for_k_1_to_5() {
        let (_cfg, model, mtp) = text_model_with_mtp();
        let config = greedy(24);
        let expected = step_tokens(&model, &PROMPT, &config);
        let reference = generate_with(
            &model,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap()
        .tokens;
        assert_eq!(expected, reference);
        assert_eq!(expected.len(), 24);
        let mut any_rejection = false;
        for k in 1..=5usize {
            let mut proposer = MtpProposer::new(&mtp);
            let run = run(&model, &mut proposer, &PROMPT, &config, k);
            assert_eq!(run.output.tokens, expected, "MTP K={k} diverged");
            assert_eq!(run.output.finish_reason, FinishReason::MaxTokens);
            assert_eq!(run.record.path, DecodePath::Mtp { drafts: k as u32 });
            assert_eq!(run.record.proposer, ProposerKind::Mtp);
            assert!(run.stats.verify_steps > 0);
            assert!(run.stats.proposed > 0);
            any_rejection |= run.stats.accepted < run.stats.proposed;
            // Forwards: the prefill plus exactly one verify per step — every partial rejection is
            // a direct rollback into the per-token checkpoint ring (S3), never a replay.
            assert_eq!(run.stats.forwards, 1 + run.stats.verify_steps, "K={k}");
            assert_eq!(run.stats.replays, 0, "K={k}");
            assert_eq!(
                run.record.target_forwards_per_verify_step(),
                Some(1.0),
                "K={k}"
            );
            // The greedy fast path: exactly one host sync per verify step (AC2), plus the one
            // first-token sample outside the steps.
            assert_eq!(run.record.host_syncs_per_verify_step(), Some(1.0), "K={k}");
            assert_eq!(
                run.record.host_syncs,
                1 + run.stats.verify_steps as u64,
                "K={k}: the first-token sample plus one per verify step"
            );
            assert_eq!(run.record.generated_tokens, 24);
        }
        assert!(any_rejection, "the fixture must reject some drafts");
    }

    // ---- The pre-engine `qwen_mtp` loop's contract, carried over ----
    //
    // Before that loop was deleted, a transient test ran it beside the engine on the pattern
    // fixture for K = 1..5, greedy and stochastic (seed 0): tokens, proposed, accepted and
    // forwards were identical in every case (the stochastic runs also pin the RNG consumption
    // order). Its degenerate fixtures and their pinned counters stay here as the regression.

    fn tensor(map: &mut HashMap<String, Tensor>, key: &str, data: Vec<f32>, dims: &[usize]) {
        map.insert(
            key.to_string(),
            Tensor::from_vec(data, dims.to_vec(), &Device::Cpu).unwrap(),
        );
    }

    fn zero(map: &mut HashMap<String, Tensor>, key: &str, dims: &[usize]) {
        tensor(map, key, vec![0.0; dims.iter().product()], dims);
    }

    /// The pre-engine loop's degenerate fixture: every token embeds to one direction and the
    /// shared head picks token 1; `aligned_predictor` makes the MTP head agree (all accepted) or
    /// zero-projects (all rejected).
    fn degenerate_fixture(aligned_predictor: bool) -> (Qwen35Model, Qwen35Mtp) {
        let cfg = Qwen35Config::from_json(&json!({
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 6,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "max_position_embeddings": 64,
                "tie_word_embeddings": false,
                "full_attention_interval": 1,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 4,
                "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false
            }
        }))
        .unwrap();
        let (h, v, inter, nh, nkv, hd) = (8usize, 6usize, 12usize, 2usize, 1usize, 4usize);
        let mut map = HashMap::new();

        // Every token embeds to the same direction. With zero decoder projections, the target
        // residual path preserves that direction; the shared LM head therefore chooses token 1.
        let mut embeddings = vec![0.0f32; v * h];
        for row in embeddings.chunks_exact_mut(h) {
            row[0] = 1.0;
        }
        tensor(
            &mut map,
            "model.language_model.embed_tokens.weight",
            embeddings,
            &[v, h],
        );
        zero(&mut map, "model.language_model.norm.weight", &[h]);
        let mut lm_head = vec![0.0f32; v * h];
        lm_head[h] = 2.0; // token 1
        tensor(&mut map, "lm_head.weight", lm_head, &[v, h]);

        let lp = |suffix: &str| format!("model.language_model.layers.0.{suffix}");
        zero(&mut map, &lp("input_layernorm.weight"), &[h]);
        zero(&mut map, &lp("post_attention_layernorm.weight"), &[h]);
        zero(&mut map, &lp("self_attn.q_proj.weight"), &[nh * hd * 2, h]);
        zero(&mut map, &lp("self_attn.k_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &lp("self_attn.v_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &lp("self_attn.o_proj.weight"), &[h, nh * hd]);
        zero(&mut map, &lp("self_attn.q_norm.weight"), &[hd]);
        zero(&mut map, &lp("self_attn.k_norm.weight"), &[hd]);
        zero(&mut map, &lp("mlp.gate_proj.weight"), &[inter, h]);
        zero(&mut map, &lp("mlp.up_proj.weight"), &[inter, h]);
        zero(&mut map, &lp("mlp.down_proj.weight"), &[h, inter]);

        zero(&mut map, "mtp.pre_fc_norm_embedding.weight", &[h]);
        zero(&mut map, "mtp.pre_fc_norm_hidden.weight", &[h]);
        zero(&mut map, "mtp.norm.weight", &[h]);
        let mut fc = vec![0.0f32; h * h * 2];
        if aligned_predictor {
            for i in 0..h {
                fc[i * (2 * h) + i] = 1.0;
            }
        }
        tensor(&mut map, "mtp.fc.weight", fc, &[h, 2 * h]);
        let mp = |suffix: &str| format!("mtp.layers.0.{suffix}");
        zero(&mut map, &mp("input_layernorm.weight"), &[h]);
        zero(&mut map, &mp("post_attention_layernorm.weight"), &[h]);
        zero(&mut map, &mp("self_attn.q_proj.weight"), &[nh * hd * 2, h]);
        zero(&mut map, &mp("self_attn.k_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &mp("self_attn.v_proj.weight"), &[nkv * hd, h]);
        zero(&mut map, &mp("self_attn.o_proj.weight"), &[h, nh * hd]);
        zero(&mut map, &mp("self_attn.q_norm.weight"), &[hd]);
        zero(&mut map, &mp("self_attn.k_norm.weight"), &[hd]);
        zero(&mut map, &mp("mlp.gate_proj.weight"), &[inter, h]);
        zero(&mut map, &mp("mlp.up_proj.weight"), &[inter, h]);
        zero(&mut map, &mp("mlp.down_proj.weight"), &[h, inter]);

        let weights = Weights::from_map(map, Device::Cpu);
        assert!(Qwen35Mtp::complete_in(&weights, &cfg));
        let target = Qwen35Model::from_weights(&weights, "model.language_model", cfg).unwrap();
        let mtp = Qwen35Mtp::from_weights_with(&weights, &target, None).unwrap();
        (target, mtp)
    }

    #[test]
    fn degenerate_fixtures_reproduce_the_pre_engine_loop_counters() {
        let config = GenerationConfig {
            max_new_tokens: 7,
            sampling: Default::default(),
            seed: Some(7),
            stop_tokens: Vec::new(),
        };
        // Every draft rejected: the old loop reported proposed 12 / accepted 0 / forwards 12.
        let (target, mtp) = degenerate_fixture(false);
        let baseline = generate_with(
            &target,
            &[2, 3],
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let mut proposer = MtpProposer::new(&mtp);
        let rejected = run(&target, &mut proposer, &[2, 3], &config, 3);
        assert_eq!(rejected.output.tokens, baseline.tokens);
        assert_eq!(rejected.output.tokens, vec![1; 7]);
        assert_eq!(rejected.stats.proposed, 12);
        assert_eq!(rejected.stats.accepted, 0);
        assert_eq!(
            rejected.stats.forwards, 7,
            "prefill + 6 verifies, no replays: every rejection is a direct rollback"
        );
        assert_eq!(rejected.stats.verify_steps, 6);
        assert_eq!(
            (rejected.stats.direct_rollbacks, rejected.stats.replays),
            (5, 0),
            "5 rejected steps recovered directly (the last step has no draft budget)"
        );
        // Every draft accepted: proposed 4 / accepted 4 / forwards 3.
        let (target, mtp) = degenerate_fixture(true);
        let mut proposer = MtpProposer::new(&mtp);
        let accepted = run(&target, &mut proposer, &[2, 3], &config, 3);
        assert_eq!(accepted.output.tokens, vec![1; 7]);
        assert_eq!(accepted.stats.proposed, 4);
        assert_eq!(accepted.stats.accepted, 4);
        assert_eq!(accepted.stats.forwards, 3);
    }

    #[test]
    fn prompt_last_logits_and_predictor_warmup_match_full_projection() {
        let (target, mtp) = degenerate_fixture(true);
        let prompt = [2, 3, 4];
        let ids = input_ids(&prompt, target.device()).unwrap();
        let mut full_cache = target.new_cache();
        let (full_logits, full_hidden) = target
            .forward_with_hidden(&ids, &mut full_cache, 0)
            .unwrap();
        let mut prefill_cache = target.new_cache();
        let (last_logits, prefill_hidden) = target
            .prefill_with_hidden(&ids, &mut prefill_cache, 0)
            .unwrap();
        assert_eq!(
            last_logits.dims(),
            &[1, target.config().vocab_size as usize]
        );
        assert_eq!(prefill_hidden.dims(), full_hidden.dims());
        let row = |t: &Tensor, i: usize| -> Vec<f32> {
            t.narrow(1, i, 1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        assert_eq!(
            last_logits.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            row(&full_logits, prompt.len() - 1)
        );
        assert_eq!(
            prefill_hidden
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            full_hidden.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );

        let previous = full_hidden.narrow(1, 0, prompt.len() - 1).unwrap();
        let mut full_mtp_cache = mtp.new_cache();
        let (full_mtp_logits, _) = mtp
            .forward_sequence(&prompt[1..], &previous, 1, &mut full_mtp_cache)
            .unwrap();
        let mut warm_mtp_cache = mtp.new_cache();
        mtp.warm_sequence(&prompt[1..], &previous, 1, &mut warm_mtp_cache)
            .unwrap();
        assert_eq!(full_mtp_logits.dims(), &[1, prompt.len() - 1, 6]);
        let last_hidden = full_hidden.narrow(1, prompt.len() - 1, 1).unwrap();
        let (full_next, _) = mtp
            .step(1, &last_hidden, 0, prompt.len() as i32, &mut full_mtp_cache)
            .unwrap();
        let (warm_next, _) = mtp
            .step(1, &last_hidden, 0, prompt.len() as i32, &mut warm_mtp_cache)
            .unwrap();
        assert_eq!(
            full_next.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            warm_next.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        // A device-id draft step equals the host-id one.
        let ids = input_ids(&[1], target.device()).unwrap();
        let mut c = mtp.new_cache();
        mtp.warm_sequence(&prompt[1..], &previous, 1, &mut c)
            .unwrap();
        let (via_ids, _) = mtp
            .step_ids(&ids, &last_hidden, 0, prompt.len() as i32, &mut c)
            .unwrap();
        assert_eq!(
            via_ids.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            warm_next.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn equal_embedding_multimodal_prefill_matches_the_text_engine_run() {
        // The provider's multimodal MTP path: a caller prefill from (here: plain token) embeddings
        // under interleaved M-RoPE with all three position rows equal, the predictor warmed from
        // the same fused rows, the engine continuing from the prefilled cache — token- and
        // counter-identical to the text prompt path.
        let (_cfg, model, mtp) = text_model_with_mtp();
        let config = greedy(16);
        let mut proposer = MtpProposer::new(&mtp);
        let text = run(&model, &mut proposer, &PROMPT, &config, 3);

        let mut cache = model
            .new_cache_for(PROMPT.len() + config.max_new_tokens, 3)
            .unwrap();
        let embeds = model
            .embed_input_ids(&input_ids(&PROMPT, model.device()).unwrap())
            .unwrap();
        let positions: Vec<i32> = (0..PROMPT.len() as i32).collect();
        let rows = [
            positions.as_slice(),
            positions.as_slice(),
            positions.as_slice(),
        ];
        let (logits, hidden) = model
            .prefill_from_embeds_deepstack_with_hidden(
                &embeds,
                rows,
                &mut cache,
                &vec![false; PROMPT.len()],
                &[],
            )
            .unwrap();
        cache.set_rope_delta(0);
        let mut proposer = MtpProposer::new(&mtp);
        proposer.warm_multimodal(&embeds, &hidden, rows).unwrap();
        let multimodal = generate_speculative(
            &model,
            &mut proposer,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits,
                hidden: Some(hidden),
                history: &PROMPT,
                position_delta: 0,
                warm_proposer: false,
            },
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(multimodal.output.tokens, text.output.tokens);
        // The caller's prefill is the run's prefill forward: fwd/verify is measured net of it on
        // the Prefilled arm exactly as on the Tokens arm.
        assert_eq!(
            multimodal.record.target_forwards_per_verify_step(),
            Some(1.0)
        );
        assert_eq!(multimodal.stats, text.stats);
    }

    #[test]
    fn ngram_engine_is_token_identical_to_the_step_driver() {
        let (_cfg, model) = text_model();
        let config = greedy(24);
        let expected = step_tokens(&model, &PROMPT, &config);
        for k in [1usize, 3, 5] {
            let mut proposer = NgramProposer { max_ngram: 3 };
            let run = run(&model, &mut proposer, &PROMPT, &config, k);
            assert_eq!(run.output.tokens, expected, "n-gram K={k} diverged");
            assert_eq!(run.record.path, DecodePath::PromptLookup);
            assert_eq!(run.record.proposer, ProposerKind::Ngram);
            assert_eq!(run.record.host_syncs_per_verify_step(), Some(1.0));
            assert!(run.stats.proposed > 0, "the repetitive prompt must draft");
        }
    }

    #[test]
    fn draft_model_engine_is_token_identical_to_the_step_driver() {
        let (_cfg, model) = text_model();
        let config = greedy(20);
        // Six layers of the same pattern weights: a draft that agrees with the target often
        // but not always (measured 13 / 16 at K = 3).
        let (_cfg, draft_target) = text_model_with_layers(6);
        let expected = step_tokens(&model, &PROMPT, &config);
        // The draft is the same decoder (every draft accepted) and a decoder that differs (a
        // mix of accepted and rejected drafts) — both must land on the target's tokens.
        let mut other_accepted = 0usize;
        let mut other_rejected = 0usize;
        for (name, draft) in [("same", &model), ("other", &draft_target)] {
            for k in [1usize, 2, 4] {
                let capacity = PROMPT.len() + config.max_new_tokens;
                let mut proposer = DraftModelProposer::new(draft, capacity, k);
                let run = run(&model, &mut proposer, &PROMPT, &config, k);
                assert_eq!(run.output.tokens, expected, "draft={name} K={k} diverged");
                assert_eq!(run.record.path, DecodePath::DraftModel);
                assert_eq!(run.record.proposer, ProposerKind::Draft);
                assert_eq!(run.record.host_syncs_per_verify_step(), Some(1.0));
                assert!(proposer.draft_forwards > run.stats.verify_steps as u64);
                if name == "same" {
                    assert_eq!(run.stats.accepted, run.stats.proposed);
                } else {
                    other_accepted += run.stats.accepted;
                    other_rejected += run.stats.proposed - run.stats.accepted;
                }
            }
        }
        assert!(
            other_accepted > 0 && other_rejected > 0,
            "{other_accepted}/{other_rejected}"
        );
    }

    #[test]
    fn no_proposer_is_the_step_driver_and_reports_proposer_none() {
        // `generate_step` is this engine with no proposer (sc-24140): one loop, one record
        // convention — the prefill in `prefill_forwards`, every later forward a verify step.
        let (_cfg, model) = text_model();
        let config = greedy(12);
        let (expected, record) = generate_step(
            &model,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let reference = generate_with(
            &model,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(expected.tokens, reference.tokens);
        let run = run(&model, &mut NoProposer, &PROMPT, &config, 3);
        assert_eq!(run.output.tokens, expected.tokens);
        assert_eq!(run.record.proposer, ProposerKind::None);
        assert_eq!(run.record.proposer.label(), "none");
        assert_eq!(run.record.path, DecodePath::StepModel);
        assert_eq!(run.record.proposed_tokens, 0);
        assert_eq!(
            run.record, record,
            "generate_step's record is the engine's, field for field"
        );
        assert_eq!(
            (
                record.target_forwards,
                record.prefill_forwards,
                record.verify_steps
            ),
            (12, 1, 11)
        );
        assert_eq!(record.host_syncs, 12, "one sync per token");
        assert_eq!(run.record.host_syncs_per_verify_step(), Some(1.0));
    }

    // ---- Every host decision is recorded; a degenerate row commits its argmax (sc-24140) ----

    #[test]
    fn stochastic_host_decisions_are_recorded_and_a_degenerate_row_commits_its_argmax() {
        use crate::primitives::host_sync::sampler_counters;

        let vocab = 6usize;
        let config = GenerationConfig {
            max_new_tokens: 4,
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                ..Default::default()
            },
            seed: Some(0),
            stop_tokens: Vec::new(),
        };
        // One point-mass draft `2`, verified over `rows` (the draft's row, then the bonus row).
        let decide_rows = |rows: Vec<Vec<f32>>| {
            let n = rows.len();
            let logits = Tensor::from_vec(rows.concat(), (1, n, vocab), &Device::Cpu).unwrap();
            let before = sampler_counters();
            let (_, committed, accepted) = decide(
                &logits,
                &Drafts::Host(vec![2]),
                &[vec![(2, 1.0)]],
                &[],
                &config,
                &mut SplitMix64::new(3),
                false,
                false,
                None,
            )
            .unwrap();
            let after = sampler_counters();
            (
                committed,
                accepted,
                after.host_draws - before.host_draws,
                after.device_draws - before.device_draws,
            )
        };
        let certain = |t: usize| {
            let mut row = vec![f32::NEG_INFINITY; vocab];
            row[t] = 0.0;
            row
        };
        let inf_at = |t: usize| {
            let mut row = vec![0.0f32; vocab];
            row[t] = f32::INFINITY;
            row
        };
        // The draft accepted, an ordinary bonus row: two host decisions (the acceptance test and
        // the bonus draw), both recorded.
        assert_eq!(
            decide_rows(vec![certain(2), certain(4)]),
            (vec![2, 4], 1, 2, 0)
        );
        // A bonus row nothing survives the shaping of (a `+inf` at id 3): its argmax, not id 0.
        assert_eq!(
            decide_rows(vec![certain(2), inf_at(3)]),
            (vec![2, 3], 1, 2, 0)
        );
        // A degenerate acceptance row: the target is the point mass on its argmax, so the draft is
        // rejected for id 3 — never committed as a fallback the row did not choose.
        assert_eq!(decide_rows(vec![inf_at(3), certain(4)]), (vec![3], 0, 1, 0));
    }

    // ---- One host sync per verify step on the host-shaped paths too ----

    struct Audit(Vec<bool>, Vec<i32>);
    impl ConstraintMask for Audit {
        fn allowed(&mut self) -> &[bool] {
            &self.0
        }
        fn accept(&mut self, t: i32) {
            self.1.push(t);
        }
    }
    impl RewindableConstraintMask for Audit {
        fn checkpoint(&self) -> usize {
            self.1.len()
        }
        fn rewind(&mut self, c: usize) {
            self.1.truncate(c);
        }
    }

    // AC2's `1.00` is the plain-greedy figure. With penalties or a constraint the drafts are
    // sampled on the host — one whole-vocab transfer per draft — and the verify decision pulls
    // the K + 1 rows in one copy: `K + 1` syncs per verify step (the old loop's figure on every
    // path); stochastic runs add a shaped-distribution copy per draft (`2K + 1`, pinned in
    // `stochastic_runs_are_seed_deterministic_and_bounded`).
    #[test]
    fn penalized_and_constrained_verify_steps_cost_one_sync_per_draft_plus_one() {
        let (_cfg, model, mtp) = text_model_with_mtp();
        let mut config = greedy(16);
        config.sampling.repetition_penalty = 1.3;
        config.sampling.repetition_context = 8;
        let expected = step_tokens(&model, &PROMPT, &config);
        let mut proposer = MtpProposer::new(&mtp);
        let run = run(&model, &mut proposer, &PROMPT, &config, 3);
        assert_eq!(run.output.tokens, expected);
        // Penalized greedy: the drafts are sampled on the host (one whole-vocab transfer each)
        // and the verify decision pulls all K + 1 rows in one copy.
        let draft_syncs = run.stats.proposed as u64;
        assert_eq!(
            run.record.verify_host_syncs,
            run.stats.verify_steps as u64 + draft_syncs
        );
        assert!(
            run.record.host_syncs_per_verify_step().unwrap() > 1.0,
            "the host-shaped path is not the 1.00 figure"
        );

        // A constraint: the mask is applied on host rows, still one transfer per verify.
        let config = greedy(16);
        let mut audit = Audit(vec![true; 50], Vec::new());
        let expected = generate_step(
            &model,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            Some(&mut audit),
        )
        .unwrap()
        .0
        .tokens;
        let mut audit = Audit(vec![true; 50], Vec::new());
        let mut proposer = MtpProposer::new(&mtp);
        let run = generate_speculative(
            &model,
            &mut proposer,
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            Some(&mut audit),
        )
        .unwrap();
        assert_eq!(run.output.tokens, expected);
        assert_eq!(
            audit.1, run.output.tokens,
            "only emitted tokens advance the constraint"
        );
        assert_eq!(
            run.record.verify_host_syncs,
            run.stats.verify_steps as u64 + run.stats.proposed as u64
        );
    }

    // ---- Rollback: direct when the cache can, replay otherwise ----

    /// A decoder whose argmax at position `p` is `(p + 1) % vocab` (all positions), with a cache
    /// that can roll back to any position — the shape of a softmax-only decoder with per-token
    /// rollback (or the S3 hybrid cache). `step_start_only` makes it refuse interior positions
    /// like the S1 hybrid cache did, to exercise the engine's replay fallback.
    struct Ramp {
        vocab: usize,
        forwards: Cell<u64>,
        step_start_only: bool,
    }
    struct RampCache {
        len: i32,
        step_start: i32,
        step_start_only: bool,
    }
    impl DecodeCache for RampCache {
        fn len(&self) -> i32 {
            self.len
        }
        fn rollback_to(&mut self, n: i32) -> Result<()> {
            assert!(n >= 0 && n <= self.len);
            if self.step_start_only && n != self.step_start && n != self.len && n != 0 {
                return Err(Error::RollbackUnavailable {
                    n,
                    have: vec![self.step_start],
                });
            }
            self.len = n;
            Ok(())
        }
        fn reset(&mut self) {
            self.len = 0;
            self.step_start = 0;
        }
        fn memory(&self) -> CacheMemory {
            CacheMemory::default()
        }
    }
    impl StepModel for Ramp {
        type Cache = RampCache;
        fn new_cache(&self) -> RampCache {
            RampCache {
                len: 0,
                step_start: 0,
                step_start_only: self.step_start_only,
            }
        }
        fn device(&self) -> &Device {
            static CPU: Device = Device::Cpu;
            &CPU
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn forward_step(
            &self,
            cache: &mut RampCache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            self.forwards.set(self.forwards.get() + 1);
            let n = request.len()?;
            let start = cache.len as usize;
            cache.step_start = cache.len;
            cache.len += n as i32;
            let mut rows = vec![0f32; n * self.vocab];
            for (i, row) in rows.chunks_exact_mut(self.vocab).enumerate() {
                row[(start + i + 1) % self.vocab] = 10.0;
            }
            let all = Tensor::from_vec(rows, (1, n, self.vocab), &Device::Cpu)?;
            let logits = match request.scope {
                LogitsScope::All => all,
                LogitsScope::Last => all.narrow(1, n - 1, 1)?.reshape((1, self.vocab))?,
            };
            Ok(StepOutput {
                logits,
                hidden: None,
            })
        }
    }

    /// A [`Ramp`] whose every forward records one fused primitive leaf and one NVFP4 decode-GEMV
    /// call, as a real decoder's forward records its primitives.
    struct Tallied(Ramp);
    impl StepModel for Tallied {
        type Cache = RampCache;
        fn new_cache(&self) -> RampCache {
            self.0.new_cache()
        }
        fn device(&self) -> &Device {
            StepModel::device(&self.0)
        }
        fn vocab_size(&self) -> usize {
            self.0.vocab
        }
        fn forward_step(
            &self,
            cache: &mut RampCache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            crate::primitives::fused::note_fused();
            crate::primitives::nvfp4_path::note_gemv();
            self.0.forward_step(cache, request)
        }
    }

    #[test]
    fn the_engine_record_carries_every_span_tally() {
        // sc-24140: the engine's record carries every per-thread tally its span measured — the
        // fused primitives, the CUDA-graph runner's steps and the NVFP4 projection paths — so a
        // consumer of `run.record` (the bench, LLaVA, StarVector) sees what ran without patching.
        let _graphs = crate::decode::graph::cuda_graphs_policy_guard(Some(true));
        let model = Tallied(Ramp {
            vocab: 7,
            forwards: Cell::new(0),
            step_start_only: false,
        });
        let runner = crate::decode::GraphRunner::new(&model);
        let run = run(&runner, &mut NoProposer, &[0, 1], &greedy(5), 0);
        let forwards = run.record.target_forwards;
        assert_eq!(forwards, 5);
        assert_eq!(run.record.fused_primitives.fused, forwards);
        assert_eq!(run.record.nvfp4_projections.gemv, forwards);
        assert_eq!(
            run.record.cuda_graphs.eager,
            forwards,
            "every step went through the runner, eager here: {}",
            run.record.cuda_graphs.describe()
        );
    }

    /// Proposes fixed wrong drafts every step so every verify rejects at the first draft.
    struct Wrong(Vec<i32>);
    impl Proposer for Wrong {
        fn kind(&self) -> ProposerKind {
            ProposerKind::Draft
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
                drafts: Some(Drafts::Host(
                    self.0.iter().copied().take(ctx.max_drafts).collect(),
                )),
                dists: Vec::new(),
            })
        }
        fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_cache_with_per_position_rollback_never_replays_and_a_step_start_cache_replays_once() {
        let model = Ramp {
            vocab: 7,
            forwards: Cell::new(0),
            step_start_only: false,
        };
        let config = greedy(10);
        // Prompt of 2 (positions 0, 1) -> the ramp continues 2, 3, 4, 5, 6, 0, 1, ...; the n-gram
        // proposer over the periodic history proposes exactly that, so everything is accepted.
        let mut ngram = NgramProposer { max_ngram: 2 };
        let run_ngram = run(&model, &mut ngram, &[0, 1], &config, 3);
        assert_eq!(run_ngram.output.tokens, vec![2, 3, 4, 5, 6, 0, 1, 2, 3, 4]);
        assert!(run_ngram.stats.accepted > 0);
        assert_eq!(run_ngram.stats.accepted, run_ngram.stats.proposed);
        assert_eq!(run_ngram.stats.forwards, 1 + run_ngram.stats.verify_steps);
        assert_eq!(
            (run_ngram.stats.direct_rollbacks, run_ngram.stats.replays),
            (0, 0),
            "full acceptance needs no recovery at all"
        );
        assert_eq!(run_ngram.record.replay_forwards, 0);
        // Every step rejects at the first draft: the direct rollback to `start + 1` succeeds, so
        // no replay forward is issued and the output is still the ramp.
        let before = model.forwards.get();
        let mut wrong = Wrong(vec![99, 99, 99]);
        let run_wrong = run(&model, &mut wrong, &[0, 1], &config, 3);
        assert_eq!(run_wrong.output.tokens, vec![2, 3, 4, 5, 6, 0, 1, 2, 3, 4]);
        assert_eq!(run_wrong.stats.accepted, 0);
        assert_eq!(run_wrong.stats.forwards, 1 + run_wrong.stats.verify_steps);
        assert_eq!(
            run_wrong.stats.replays, 0,
            "a direct rollback never replays"
        );
        assert_eq!(run_wrong.record.replay_forwards, 0);
        assert_eq!(
            model.forwards.get() - before,
            run_wrong.stats.forwards as u64
        );
        assert_eq!(run_wrong.stats.replays, 0);
        assert_eq!(
            run_wrong.stats.direct_rollbacks,
            run_wrong.stats.verify_steps - 1,
            "every rejected step (the last has no draft budget) was a direct rollback"
        );
        assert_eq!(
            run_wrong.record.direct_rollbacks,
            run_wrong.stats.direct_rollbacks as u64
        );

        // A cache that checkpoints step starts only (the S1 hybrid cache): the same rejection
        // costs one replay forward, and the record says so.
        let step_start = Ramp {
            vocab: 7,
            forwards: Cell::new(0),
            step_start_only: true,
        };
        let mut wrong = Wrong(vec![99, 99, 99]);
        let run_replay = run(&step_start, &mut wrong, &[0, 1], &config, 3);
        assert_eq!(run_replay.output.tokens, vec![2, 3, 4, 5, 6, 0, 1, 2, 3, 4]);
        assert_eq!(run_replay.stats.accepted, 0);
        assert_eq!(
            run_replay.stats.forwards,
            1 + run_replay.stats.verify_steps + run_replay.stats.replays
        );
        assert_eq!(run_replay.stats.replays, run_replay.stats.verify_steps - 1);
        assert_eq!(run_replay.stats.direct_rollbacks, 0);
        assert_eq!(
            run_replay.record.replay_forwards,
            run_replay.stats.replays as u64
        );
        assert_eq!(
            run_replay.record.target_forwards_per_verify_step(),
            Some(
                1.0 + (run_replay.stats.verify_steps - 1) as f64
                    / run_replay.stats.verify_steps as f64
            )
        );

        // The Qwen3.5 cache's per-token checkpoint ring (S3): the same forced rejections are all
        // direct rollbacks — one target forward per verify step, zero replays — for every K.
        let (_cfg, qwen) = text_model();
        let expected = step_tokens(&qwen, &PROMPT, &config);
        for k in 1..=5usize {
            let mut wrong = Wrong(vec![49; k]);
            let run_qwen = run(&qwen, &mut wrong, &PROMPT, &config, k);
            assert_eq!(run_qwen.output.tokens, expected, "K={k}");
            assert_eq!(run_qwen.stats.accepted, 0);
            assert_eq!(run_qwen.stats.verify_steps, 9);
            assert_eq!(
                run_qwen.stats.forwards,
                1 + 9,
                "K={k}: the prefill plus one forward per verify step, no replays"
            );
            // The last step has no draft budget left (k = 0), so it cannot reject: 8 rollbacks.
            assert_eq!(
                (run_qwen.stats.direct_rollbacks, run_qwen.stats.replays),
                (8, 0),
                "K={k}"
            );
            assert_eq!(run_qwen.record.target_forwards_per_verify_step(), Some(1.0));
            assert_eq!(run_qwen.record.replay_forwards, 0);
        }
    }

    /// Proposes the ramp continuation after `cur`, `len` drafts wide **regardless of
    /// `max_drafts`** — a misbehaving proposer whose accepted run can outlive the budget.
    struct Over {
        vocab: i32,
        len: usize,
    }
    impl Proposer for Over {
        fn kind(&self) -> ProposerKind {
            ProposerKind::Draft
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
                drafts: Some(Drafts::Host(
                    (1..=self.len as i32)
                        .map(|i| (ctx.cur + i) % self.vocab)
                        .collect(),
                )),
                dists: Vec::new(),
            })
        }
        fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn max_tokens_exit_settles_the_cache_when_an_accepted_run_outlives_the_budget() {
        // Budget 4, width 3: the first token leaves 3, so the step's width is clamped to 2 — but
        // the proposer answers with 5 drafts, all accepted (6 positions written). The budget is
        // reached after 3 of them: the exit must roll the cache back so it holds no position the
        // committed history does not (the same contract as the stop / cancel exits).
        let model = Ramp {
            vocab: 7,
            forwards: Cell::new(0),
            step_start_only: false,
        };
        let config = greedy(4);
        let prompt = [0, 1];
        let mut cache = model
            .new_cache_for(prompt.len() + config.max_new_tokens, 3)
            .unwrap();
        let out = model
            .forward_step(&mut cache, StepRequest::last(&prompt))
            .unwrap();
        let mut over = Over { vocab: 7, len: 5 };
        let run = generate_speculative(
            &model,
            &mut over,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits: out.logits,
                hidden: None,
                history: &prompt,
                position_delta: 0,
                warm_proposer: false,
            },
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(run.output.tokens, vec![2, 3, 4, 5]);
        assert_eq!(run.output.finish_reason, FinishReason::MaxTokens);
        assert_eq!(run.stats.proposed, 5);
        assert_eq!(run.stats.accepted, 5);
        assert!(
            cache.len() as usize <= prompt.len() + run.output.tokens.len(),
            "cache {} holds positions past the {} committed",
            cache.len(),
            prompt.len() + run.output.tokens.len()
        );
    }

    #[test]
    fn device_greedy_drafts_stop_counting_at_an_accepted_stop_token() {
        // A draft model on the plain-greedy path drafts on the device and cannot see a stop
        // token: it proposes [3, 4, 5] after 2 with 4 the stop token. The decision must truncate
        // the run at the stop token — 5 is neither proposed nor accepted, although the target
        // would have accepted it — and the run ends at the stop with the drafts before it.
        let model = Ramp {
            vocab: 7,
            forwards: Cell::new(0),
            step_start_only: false,
        };
        let mut config = greedy(10);
        config.stop_tokens = vec![4];
        let prompt = [0, 1];
        let mut proposer = DraftModelProposer::new(&model, prompt.len() + config.max_new_tokens, 3);
        let run = run(&model, &mut proposer, &prompt, &config, 3);
        assert_eq!(run.output.tokens, vec![2, 3]);
        assert_eq!(run.output.finish_reason, FinishReason::StopToken);
        assert_eq!(run.record.proposer, ProposerKind::Draft);
        // Drafts up to and including the stop token: 3 and 4.
        assert_eq!(
            run.stats.proposed, 2,
            "the draft past the stop token is not a proposal"
        );
        assert_eq!(
            run.stats.accepted, 2,
            "accepted must not exceed the drafts up to the stop token"
        );
        assert_eq!(run.stats.verify_steps, 1);
    }

    // ---- Stochastic: seed-deterministic, and the decision preserves the target distribution ----

    #[test]
    fn stochastic_runs_are_seed_deterministic_and_bounded() {
        let (_cfg, model, mtp) = text_model_with_mtp();
        let config = stochastic(20);
        let mut a = MtpProposer::new(&mtp);
        let mut b = MtpProposer::new(&mtp);
        let first = run(&model, &mut a, &PROMPT, &config, 3);
        let second = run(&model, &mut b, &PROMPT, &config, 3);
        assert_eq!(first.output.tokens, second.output.tokens);
        assert_eq!(first.stats, second.stats);
        assert!(first.stats.accepted <= first.stats.proposed);
        assert!(first.stats.proposed > 0);
        // Stochastic drafts are host-sampled (a shaped-distribution and a sample transfer each);
        // the verify itself is still one.
        assert_eq!(
            first.record.verify_host_syncs,
            first.stats.verify_steps as u64 + first.stats.proposed as u64 * 2
        );
        let mut n = NgramProposer::default();
        let via_ngram = run(&model, &mut n, &PROMPT, &config, 3);
        assert_eq!(via_ngram.record.proposer, ProposerKind::Ngram);
        assert_eq!(via_ngram.record.host_syncs_per_verify_step(), Some(1.0));
    }

    #[test]
    fn stochastic_decision_preserves_the_target_distribution_chi_square() {
        // The engine's verify decision over point-mass (n-gram) drafts drawn from `q` commits a
        // first token distributed as the target's shaped distribution `p` — the headline
        // guarantee of `core_llm::accept_token`, checked through the engine's own `decide`.
        let vocab = 5usize;
        let p_logits = [1.0f32, 2.2, 0.3, 1.7, -0.5];
        let config = GenerationConfig {
            max_new_tokens: 1,
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                ..Default::default()
            },
            seed: Some(0),
            stop_tokens: Vec::new(),
        };
        // Rows: position 0 (the draft's) and the bonus row (uniform, irrelevant to token 0).
        let mut rows = p_logits.to_vec();
        rows.extend(vec![0.0f32; vocab]);
        let logits = Tensor::from_vec(rows, (1, 2, vocab), &Device::Cpu).unwrap();
        let q: Vec<(i32, f32)> = vec![(0, 0.4), (1, 0.1), (2, 0.2), (3, 0.1), (4, 0.2)];
        let max = p_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = p_logits.iter().map(|&x| ((x - max) as f64).exp()).collect();
        let total: f64 = weights.iter().sum();
        let n = 60_000usize;
        // The draft distribution handed to the decision: the point mass an n-gram proposer
        // reports, and the actual `q` a draft model reports (the `p / q` ratio then decides,
        // not a plain `p` comparison).
        let point_mass = |t: i32| vec![(t, 1.0f32)];
        let draft_q = |_: i32| q.clone();
        type DistOf<'a> = &'a dyn Fn(i32) -> Vec<(i32, f32)>;
        let cases: [(&str, DistOf<'_>); 2] = [("point mass", &point_mass), ("draft q", &draft_q)];
        for (name, dist_of) in cases {
            let mut rng = SplitMix64::new(0x5eed);
            let mut counts = vec![0u64; vocab];
            for _ in 0..n {
                let proposed = sample_weighted(&q, rng.next_f32(), 0);
                let (_, committed, _) = decide(
                    &logits,
                    &Drafts::Host(vec![proposed]),
                    &[dist_of(proposed)],
                    &[],
                    &config,
                    &mut rng,
                    false,
                    false,
                    None,
                )
                .unwrap();
                counts[committed[0] as usize] += 1;
            }
            let mut chi2 = 0.0f64;
            for (t, &c) in counts.iter().enumerate() {
                let expected = n as f64 * weights[t] / total;
                chi2 += (c as f64 - expected).powi(2) / expected;
            }
            // 4 degrees of freedom: chi-square 99.9 % critical value 18.47.
            assert!(
                chi2 < 18.47,
                "{name}: chi-square {chi2:.2} over counts {counts:?}"
            );
        }
    }

    // ---- The stream contract: stop tokens, caller stop, cancellation, cache consistency ----

    #[test]
    fn stop_token_caller_stop_and_cancellation_follow_the_stream_contract() {
        let (_cfg, model, mtp) = text_model_with_mtp();
        let config = greedy(24);
        let full = step_tokens(&model, &PROMPT, &config);
        // A stop token equal to the reference's 5th token ends the run at its first occurrence.
        let mut stop_cfg = config.clone();
        stop_cfg.stop_tokens = vec![full[4]];
        let stop_at = full.iter().position(|&t| t == full[4]).unwrap();
        let mut proposer = MtpProposer::new(&mtp);
        let mut events = Vec::new();
        let stopped = generate_speculative(
            &model,
            &mut proposer,
            SpeculativePrompt::Tokens(&PROMPT),
            &stop_cfg,
            3,
            &CancelFlag::new(),
            &mut |e| events.push(e),
            None,
        )
        .unwrap();
        assert_eq!(stopped.output.tokens, full[..stop_at]);
        assert_eq!(stopped.output.finish_reason, FinishReason::StopToken);
        for (i, e) in events.iter().enumerate() {
            match e {
                StreamEvent::Token { step, .. } => assert_eq!(*step, i),
                StreamEvent::Done { reason, generated } => {
                    assert_eq!(i, events.len() - 1);
                    assert_eq!(*reason, FinishReason::StopToken);
                    assert_eq!(*generated, stop_at);
                }
            }
        }

        let pre = CancelFlag::new();
        pre.cancel();
        let mut proposer = MtpProposer::new(&mtp);
        assert!(matches!(
            generate_speculative(
                &model,
                &mut proposer,
                SpeculativePrompt::Tokens(&PROMPT),
                &config,
                3,
                &pre,
                &mut |_| {},
                None
            ),
            Err(Error::Canceled)
        ));

        // A caller stop after the first token halts before another target step.
        let halted = Cell::new(false);
        let mut proposer = MtpProposer::new(&mtp);
        let halted_run = generate_speculative_with(
            &model,
            &mut proposer,
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            3,
            &CancelFlag::new(),
            &mut |e| {
                if matches!(e, StreamEvent::Token { .. }) {
                    halted.set(true);
                }
            },
            None,
            Some(&|| halted.get()),
            None,
        )
        .unwrap();
        assert_eq!(halted_run.output.tokens.len(), 1);
        assert_eq!(halted_run.output.finish_reason, FinishReason::Stopped);
        assert_eq!(halted_run.stats.verify_steps, 0);

        // Empty prompt and zero budget.
        let mut proposer = MtpProposer::new(&mtp);
        assert!(matches!(
            generate_speculative(
                &model,
                &mut proposer,
                SpeculativePrompt::Tokens(&[]),
                &config,
                3,
                &CancelFlag::new(),
                &mut |_| {},
                None
            ),
            Err(Error::Msg(_))
        ));
        let mut proposer = MtpProposer::new(&mtp);
        let none = run(&model, &mut proposer, &PROMPT, &greedy(0), 3);
        assert!(none.output.tokens.is_empty());
        assert_eq!(none.record.host_syncs, 0);
    }

    #[test]
    fn cancellation_mid_verify_leaves_a_caller_owned_cache_consistent() {
        // Cancel from inside the commit of a verify step: the engine returns Cancelled and the
        // caller's cache holds no position that is not in the committed history, so the caller
        // can continue decoding from it.
        let (_cfg, model, mtp) = text_model_with_mtp();
        let config = greedy(24);
        let full = step_tokens(&model, &PROMPT, &config);
        for cancel_at in [1usize, 2, 3, 5] {
            let mut cache = model
                .new_cache_for(PROMPT.len() + config.max_new_tokens, 3)
                .unwrap();
            let out = model
                .forward_step(&mut cache, StepRequest::last(&PROMPT).with_hidden(true))
                .unwrap();
            let flag = CancelFlag::new();
            let signal = flag.clone();
            let mut proposer = MtpProposer::new(&mtp);
            let run = generate_speculative(
                &model,
                &mut proposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits: out.logits,
                    hidden: out.hidden,
                    history: &PROMPT,
                    position_delta: 0,
                    warm_proposer: true,
                },
                &config,
                3,
                &flag,
                &mut |e| {
                    if let StreamEvent::Token { step, .. } = e {
                        if step + 1 == cancel_at {
                            signal.cancel();
                        }
                    }
                },
                None,
            )
            .unwrap();
            assert_eq!(run.output.finish_reason, FinishReason::Cancelled);
            assert_eq!(run.output.tokens.len(), cancel_at);
            assert_eq!(run.output.tokens, full[..cancel_at]);
            let committed = PROMPT.len() + run.output.tokens.len();
            assert!(
                cache.len() as usize <= committed,
                "cache {} > history {committed}",
                cache.len()
            );
            assert!(cache.len() as usize >= PROMPT.len());
            // Re-decoding from the settled cache continues the same greedy sequence.
            let mut history = PROMPT.to_vec();
            history.extend(&run.output.tokens);
            let fed = &history[cache.len() as usize..committed];
            if !fed.is_empty() {
                let next = model
                    .forward_step(&mut cache, StepRequest::last(fed))
                    .unwrap()
                    .logits;
                let next = crate::primitives::sampler::argmax_device(&next).unwrap();
                assert_eq!(next, full[cancel_at]);
            }
        }
    }

    /// A target whose `forward_step` fires a cancel as its `at`-th forward returns — a cancel
    /// that lands while a step's forward is in flight.
    struct CancelInForward<'a> {
        inner: &'a Qwen35Model,
        flag: CancelFlag,
        at: u64,
        forwards: Cell<u64>,
    }
    impl StepModel for CancelInForward<'_> {
        type Cache = <Qwen35Model as StepModel>::Cache;
        fn new_cache(&self) -> Self::Cache {
            StepModel::new_cache(self.inner)
        }
        fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<Self::Cache> {
            self.inner.new_cache_for(capacity, overshoot)
        }
        fn device(&self) -> &Device {
            StepModel::device(self.inner)
        }
        fn vocab_size(&self) -> usize {
            StepModel::vocab_size(self.inner)
        }
        fn forward_step(
            &self,
            cache: &mut Self::Cache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            let out = self.inner.forward_step(cache, request)?;
            self.forwards.set(self.forwards.get() + 1);
            if self.forwards.get() == self.at {
                self.flag.cancel();
            }
            Ok(out)
        }
    }

    #[test]
    fn cancellation_inside_the_verify_forward_commits_nothing_of_that_step() {
        // The old loop checked the flag right after the verify forward; the engine must too: a
        // cancel that lands during a step's verify forward returns Cancelled with only the tokens
        // committed before that step, and the caller's cache is back at the step start.
        let (_cfg, model, mtp) = text_model_with_mtp();
        let config = greedy(24);
        let full = step_tokens(&model, &PROMPT, &config);
        // `at = 1`: the first forward through the wrapper is the first step's verify. `at = 3`:
        // a forward inside a later step (a verify, or the replay of a rejected one).
        for at in [1u64, 3] {
            let mut cache = model
                .new_cache_for(PROMPT.len() + config.max_new_tokens, 3)
                .unwrap();
            let out = model
                .forward_step(&mut cache, StepRequest::last(&PROMPT).with_hidden(true))
                .unwrap();
            let flag = CancelFlag::new();
            let wrapped = CancelInForward {
                inner: &model,
                flag: flag.clone(),
                at,
                forwards: Cell::new(0),
            };
            let mut proposer = MtpProposer::new(&mtp);
            let run = generate_speculative(
                &wrapped,
                &mut proposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits: out.logits,
                    hidden: out.hidden,
                    history: &PROMPT,
                    position_delta: 0,
                    warm_proposer: true,
                },
                &config,
                3,
                &flag,
                &mut |_| {},
                None,
            )
            .unwrap();
            assert_eq!(run.output.finish_reason, FinishReason::Cancelled, "at={at}");
            let n = run.output.tokens.len();
            assert!(n >= 1, "at={at}: the first token precedes every step");
            assert_eq!(run.output.tokens, full[..n], "at={at}");
            if at == 1 {
                assert_eq!(n, 1, "the verify of the first step commits nothing");
                assert_eq!(run.stats.verify_steps, 1);
                assert_eq!(run.stats.accepted, 0);
                assert_eq!(cache.len() as usize, PROMPT.len(), "back at the step start");
            }
            let committed = PROMPT.len() + n;
            assert!(cache.len() as usize <= committed, "at={at}");
            // Re-decoding from the settled cache continues the same greedy sequence.
            let mut history = PROMPT.to_vec();
            history.extend(&run.output.tokens);
            let fed = &history[cache.len() as usize..committed];
            let next = model
                .forward_step(&mut cache, StepRequest::last(fed))
                .unwrap()
                .logits;
            let next = crate::primitives::sampler::argmax_device(&next).unwrap();
            assert_eq!(next, full[n], "at={at}");
        }
    }

    #[test]
    fn prefilled_prompt_equals_the_token_prompt() {
        let (_cfg, model, mtp) = text_model_with_mtp();
        let config = greedy(16);
        let mut proposer = MtpProposer::new(&mtp);
        let via_tokens = run(&model, &mut proposer, &PROMPT, &config, 3);
        let mut cache = model
            .new_cache_for(PROMPT.len() + config.max_new_tokens, 3)
            .unwrap();
        let out = model
            .forward_step(&mut cache, StepRequest::last(&PROMPT).with_hidden(true))
            .unwrap();
        let mut proposer = MtpProposer::new(&mtp);
        proposer.warm(&PROMPT, out.hidden.as_ref()).unwrap();
        let via_prefilled = generate_speculative(
            &model,
            &mut proposer,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits: out.logits,
                hidden: out.hidden,
                history: &PROMPT,
                position_delta: 0,
                warm_proposer: false,
            },
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(via_prefilled.output.tokens, via_tokens.output.tokens);
        assert_eq!(via_prefilled.stats, via_tokens.stats);
        assert_eq!(via_prefilled.record.kv_cache, via_tokens.record.kv_cache);
    }
}
