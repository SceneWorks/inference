//! The MLX engine's proposal sources (epic sc-24432, stories sc-24434 and sc-24436): prompt
//! lookup ([`NgramProposer`], any target), the native Qwen3.8 multi-token predictor
//! ([`MtpProposer`], the Qwen35 target) and a separate draft model ([`DraftModelProposer`], any
//! target and any draft decoder). Each implements [`Proposer`]; the loop that verifies, accepts
//! and rolls back is the one [`engine`](super::engine).

use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::TryIndexOp;
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use core_llm::speculative::ngram_propose;
use core_llm::ProposerKind;

use crate::decode::cancel::CancelFlag;
use crate::decode::engine::{
    generate_speculative, seq_rows, CacheRollback, DraftSampler, EngineOptions, LogitsScope,
    Proposal, ProposeContext, Proposer, RewindableConstraintMask, Rollback, SpeculativePrompt,
    SpeculativeTarget,
};
use crate::decode::speculative::SpeculativeStats;
use crate::decode::stream::{GenerationConfig, GenerationOutput, StreamEvent};
use crate::error::{Error, Result};
use crate::models::qwen35::MtpCache;
use crate::models::Qwen35Model;
use crate::primitives::input_ids;

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

impl<T: SpeculativeTarget + ?Sized> Proposer<T> for NgramProposer {
    fn kind(&self) -> ProposerKind {
        ProposerKind::PromptLookup
    }

    fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
        Ok(None)
    }

    fn propose(
        &mut self,
        _: &T,
        ctx: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        Ok(Proposal {
            drafts: ngram_propose(ctx.history, self.max_ngram, ctx.max_drafts),
            dists: Vec::new(),
        })
    }

    fn commit(&mut self, _: &T, _: i32, _: &[i32], _: Option<&Array>, _: i32) -> Result<()> {
        Ok(())
    }
}

/// A separate **draft model** as a proposer (sc-24436): any decoder behind the engine's own
/// [`SpeculativeTarget`] seam — its cache and its rollback policy come with it — proposing up to
/// `K` drafts autoregressively from its own distribution `q`, which a stochastic run hands to the
/// exact rejection rule. The draft must score the target's token ids (the provider checks the
/// tokenizer vocabulary and logits width at load, [`core_llm::draft_compatibility`]).
///
/// The draft's cache holds only target-confirmed tokens between steps. A proposal feeds the
/// committed tokens the cache has not seen yet together with `cur` in one forward, marks that
/// point as the step start ([`CacheRollback::begin`]), then feeds each draft but the last. The
/// commit keeps the accepted drafts already fed — by truncation, by the Qwen35 hybrid's DeltaNet
/// checkpoint ring (sc-24435), or by restoring the step start on a cache that can do neither — and
/// queues the rest of the accepted run for the next proposal's first forward, so no step pays an
/// extra draft forward to catch up.
///
/// A draft padded differently from its target ([`with_vocab`](Self::with_vocab)) proposes only
/// its tokenizer's ids, over logits the target's width.
pub struct DraftModelProposer<'d, D: SpeculativeTarget + ?Sized> {
    draft: &'d D,
    cache: Option<D::Cache>,
    rollback: D::Rollback,
    /// `(proposable, width)`: draw only the first `proposable` draft ids, over logits `width`
    /// (the target's) wide. `None`: the draft's logits as they are.
    vocab: Option<(i32, i32)>,
    /// Committed tokens not yet in the draft cache, fed ahead of `cur` by the next proposal.
    pending: Vec<i32>,
    /// Drafts the last proposal wrote into the cache past the step start (`None`: no proposal
    /// this step, so `cur` itself was never fed).
    fed: Option<usize>,
    /// Draft-model forwards issued (its own cost, separate from the target's).
    pub draft_forwards: u64,
}

impl<'d, D: SpeculativeTarget + ?Sized> DraftModelProposer<'d, D> {
    /// A proposer over `draft` for a run of up to `drafts` drafts per step. The draft's rollback
    /// is built for that width: a proposal window — `cur`'s forward marks the step start, then at
    /// most `drafts - 1` single-token forwards — never outgrows it (the Qwen35 hybrid's
    /// checkpoint ring is armed for `drafts + 1` tokens, sc-24435).
    pub fn new(draft: &'d D, drafts: usize) -> Self {
        Self {
            draft,
            cache: None,
            rollback: draft.rollback(drafts),
            vocab: None,
            pending: Vec::new(),
            fed: None,
            draft_forwards: 0,
        }
    }

    /// Draw drafts only from the first `proposable` draft ids, over logits `width` wide — the
    /// target's — with every other id at `-inf`: a draft whose padding rows differ from its
    /// target's ([`core_llm::draft_compatibility`]) never proposes a padding id, and its `q`
    /// covers the target's id space.
    pub fn with_vocab(mut self, proposable: usize, width: usize) -> Self {
        self.vocab = Some((proposable.min(width) as i32, width as i32));
        self
    }

    /// The draft's `[1, vocab]` logits over the target's ids.
    pub(crate) fn shaped(&self, logits: Array) -> Result<Array> {
        let Some((proposable, width)) = self.vocab else {
            return Ok(logits);
        };
        let own = *logits.shape().last().unwrap_or(&0);
        if own == proposable && proposable == width {
            return Ok(logits);
        }
        let rows = logits.reshape(&[-1, own])?;
        let kept = rows.try_index((.., ..proposable.min(own)))?;
        let kept_width = proposable.min(own);
        if kept_width == width {
            return Ok(kept);
        }
        let pad = mlx_rs::ops::full::<f32>(
            &[rows.shape()[0], width - kept_width],
            Array::from_f32(f32::NEG_INFINITY),
        )?
        .as_dtype(logits.dtype())?;
        Ok(concatenate_axis(&[&kept, &pad], 1)?)
    }

    fn cache(&mut self) -> Result<&mut D::Cache> {
        self.cache
            .as_mut()
            .ok_or_else(|| Error::Msg("DraftModelProposer: propose before warm".into()))
    }

    /// Run `ids` through the draft at the end of its cache, returning the last row's logits over
    /// the target's ids ([`shaped`](Self::shaped)).
    fn feed(&mut self, ids: &[i32]) -> Result<Array> {
        let draft = self.draft;
        let cache = self.cache()?;
        let offset = draft.cache_len(cache);
        let out = draft.forward(cache, &input_ids(ids), offset, LogitsScope::Last, false)?;
        self.draft_forwards += 1;
        self.shaped(out.logits)
    }
}

#[cfg(test)]
impl<D: SpeculativeTarget + ?Sized> DraftModelProposer<'_, D> {
    /// The draft cache, the committed tokens it has not been fed yet, and the drafts the last
    /// proposal fed past its step start (`None` once committed).
    pub(crate) fn state(&self) -> (Option<&D::Cache>, &[i32], Option<usize>) {
        (self.cache.as_ref(), &self.pending, self.fed)
    }
}

impl<T, D> Proposer<T> for DraftModelProposer<'_, D>
where
    T: SpeculativeTarget + ?Sized,
    D: SpeculativeTarget + ?Sized,
{
    fn kind(&self) -> ProposerKind {
        ProposerKind::DraftModel
    }

    fn warm(&mut self, _: &T, prompt: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
        self.cache = Some(self.draft.new_cache());
        self.pending.clear();
        self.fed = None;
        // The draft prefill's logits are the warm-up graph the engine synchronizes at the prefill
        // boundary; the first proposal feeds `cur` after it.
        Ok(Some(self.feed(prompt)?))
    }

    fn propose(
        &mut self,
        _: &T,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let mut confirmed = std::mem::take(&mut self.pending);
        confirmed.push(ctx.cur);
        let mut logits = self.feed(&confirmed)?;
        let cache = self
            .cache
            .as_mut()
            .ok_or_else(|| Error::Msg("DraftModelProposer: propose before warm".into()))?;
        // Everything up to `cur` is target-confirmed: the step start a rejection returns to.
        self.rollback.begin(cache);

        let mut fed = 0usize;
        let mut drafts = Vec::with_capacity(ctx.max_drafts);
        let mut dists = Vec::with_capacity(ctx.max_drafts);
        let mut draft_history = ctx.history.to_vec();
        for step in 0..ctx.max_drafts {
            let (token, dist) = sampler.sample_draft(&logits, &draft_history)?;
            dists.extend(dist);
            drafts.push(token);
            draft_history.push(token);
            if sampler.is_stop(token) || step + 1 == ctx.max_drafts {
                break;
            }
            logits = self.feed(&[token])?;
            fed += 1;
        }
        self.fed = Some(fed);
        Ok(Proposal { drafts, dists })
    }

    fn commit(
        &mut self,
        _: &T,
        cur: i32,
        accepted: &[i32],
        _: Option<&Array>,
        _: i32,
    ) -> Result<()> {
        let Some(fed) = self.fed.take() else {
            // A step with no proposal (the budget clamped it to no drafts): `cur` was never fed.
            self.pending.push(cur);
            return Ok(());
        };
        let draft = self.draft;
        let rollback = &mut self.rollback;
        let cache = self
            .cache
            .as_mut()
            .ok_or_else(|| Error::Msg("DraftModelProposer: commit before warm".into()))?;
        let kept = accepted.len().min(fed);
        let step_start = draft.cache_len(cache) - fed as i32;
        self.pending = match rollback.recover(cache, step_start + kept as i32)? {
            // The cache holds `cur` and the first `kept` accepted drafts.
            Rollback::Direct => accepted[kept..].to_vec(),
            // Back at the step start (just past `cur`): the whole accepted run is still to feed.
            Rollback::Restored => accepted.to_vec(),
        };
        Ok(())
    }
}

/// A Qwen3.8 multimodal prompt whose visual rows have already been encoded and fused into the
/// decoder input embeddings. MTP must seed from these embeddings and the explicit three-axis
/// M-RoPE positions; re-embedding the placeholder ids would change the target distribution.
pub struct Qwen35MtpMultimodalPrompt<'a> {
    /// The effective (placeholder-expanded) prompt ids.
    pub input_ids: &'a [i32],
    /// The fused decoder input embeddings `[1, prompt, hidden]`.
    pub embeddings: &'a Array,
    /// The three M-RoPE position rows (temporal, height, width).
    pub positions: [&'a [i32]; 3],
    /// Which prompt rows are visual.
    pub visual_pos_mask: &'a [bool],
    /// DeepStack features fused into the early decoder layers.
    pub deepstack: &'a [Array],
    /// The M-RoPE shift of the text continuation past the prompt.
    pub continuation_delta: i32,
}

/// The checkpoint-native Qwen3.8 multi-token predictor as a proposer: the head proposes a run of
/// `K` tokens autoregressively, each paired with the previous hidden state, and after verification
/// its cache is rebuilt from target-confirmed pairs only.
///
/// The head's cache is snapshotted right after the first draft step (which consumes `cur`, a
/// target-selected and therefore always-valid token); the recursively drafted state after it is
/// discarded on commit even when its tokens were accepted, because replay must pair each accepted
/// token with the **target's** hidden row rather than the head's own.
///
/// **Prefix-cache resume (story sc-24437).** A prompt whose leading `M` positions came from the
/// cross-turn prefix cache is prefilled from `M` on, so the target returns hidden rows only for
/// positions `M..P`. The head's warm-up pairs `embed(x[j + 1])` with `H[j]`, so it needs the head's
/// cache at the boundary plus `H[M - 1]` — an [`MtpBoundary`] the prefix cache stored beside the
/// target state ([`resume_from`](Self::resume_from)); the warm-up then seeds only positions
/// `M..P`. [`capture_at`](Self::capture_at) records the same state at a boundary inside this
/// prompt for a later request to resume from.
#[derive(Default)]
pub struct MtpProposer<'a> {
    multimodal: Option<&'a Qwen35MtpMultimodalPrompt<'a>>,
    cache: Option<MtpCache>,
    after_cur: Option<MtpCache>,
    resume: Option<MtpBoundary>,
    capture_at: Option<usize>,
    captured: Option<MtpBoundary>,
}

/// The MTP head's state at a prompt boundary `len` (story sc-24437): its cache after warming on
/// the first `len` prompt positions, and the target's final-normalized hidden row `H[len - 1]` the
/// next warm-up pair needs. Immutable once captured — a resume clones the cache (MLX arrays are
/// refcounted and the KV write copies a shared block), so the stored boundary is never written.
#[derive(Clone, Debug)]
pub struct MtpBoundary {
    cache: MtpCache,
    hidden: Array,
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

    /// Bytes the state holds (the head's KV buffers plus the hidden row).
    pub fn bytes(&self) -> u64 {
        self.cache.bytes() + self.hidden.nbytes() as u64
    }
}

impl<'a> MtpProposer<'a> {
    /// A proposer over a text prompt.
    pub fn new() -> Self {
        Self::default()
    }

    /// A proposer over a fused multimodal prompt: the warm-up seeds the head from the fused
    /// embeddings at the prompt's explicit M-RoPE positions.
    pub fn multimodal(prompt: &'a Qwen35MtpMultimodalPrompt<'a>) -> Self {
        Self {
            multimodal: Some(prompt),
            ..Self::default()
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

    fn cache(&mut self) -> Result<&mut MtpCache> {
        self.cache
            .as_mut()
            .ok_or_else(|| Error::Msg("MtpProposer: propose before warm".into()))
    }
}

impl MtpProposer<'_> {
    /// The text warm-up: seed the head with the pairs `embed(x[j + 1]) + H[j]` at positions
    /// `j + 1` for every prompt position the target prefilled — all of them cold, or `M..P` when
    /// resuming from a prefix-cache boundary at `M` (whose `H[M - 1]` the boundary carries).
    /// `hidden` holds the target's rows for the prefilled positions only. With a capture point
    /// `B` the seeding is split there and the head's state at `B` recorded.
    fn warm_text(
        &mut self,
        model: &Qwen35Model,
        prompt: &[i32],
        hidden: &Array,
    ) -> Result<Option<Array>> {
        let (mut cache, offset, boundary_row) = match self.resume.take() {
            Some(b) => (b.cache, b.len, Some(b.hidden)),
            None => (
                model.new_mtp_cache().ok_or_else(|| {
                    Error::Msg("MtpProposer: the model has no loaded MTP predictor".into())
                })?,
                0,
                None,
            ),
        };
        let prompt_len = prompt.len();
        let rows = hidden.shape()[1] as usize;
        if offset >= prompt_len || rows != prompt_len - offset {
            return Err(Error::Msg(format!(
                "MtpProposer: {rows} hidden rows for a {prompt_len}-token prompt resumed at {offset}"
            )));
        }
        // The target's hidden rows for positions `start .. start + n` — `H[offset - 1]` comes
        // from the resumed boundary, everything later from this prefill.
        let target_rows = |start: usize, n: usize| -> Result<Array> {
            match &boundary_row {
                Some(row) if start + 1 == offset => {
                    if n == 1 {
                        Ok(row.clone())
                    } else {
                        let rest = seq_rows(hidden, 0, n as i32 - 1)?;
                        Ok(mlx_rs::ops::concatenate_axis(&[row, &rest], 1)?)
                    }
                }
                _ => seq_rows(hidden, (start - offset) as i32, n as i32),
            }
        };
        let first = offset.max(1);
        let split = self
            .capture_at
            .take()
            .filter(|&b| b > offset && b < prompt_len);
        let mut seed = None;
        let mut segments = Vec::with_capacity(2);
        let mut from = first;
        if let Some(b) = split {
            segments.push(from..b);
            from = b;
        }
        segments.push(from..prompt_len);
        for segment in segments {
            if !segment.is_empty() {
                let shifted = model.embed_input_ids(&input_ids(&prompt[segment.clone()]))?;
                let aligned = target_rows(segment.start - 1, segment.len())?;
                let positions: Vec<i32> = segment.clone().map(|p| p as i32).collect();
                seed = Some(model.mtp_warm_from_embeds(
                    &shifted,
                    &aligned,
                    &mut cache,
                    [&positions, &positions, &positions],
                )?);
            }
            if split == Some(segment.end) {
                let row = target_rows(segment.end - 1, 1)?;
                let mut pending: Vec<&Array> = seed.iter().collect();
                pending.push(&row);
                eval(pending)?;
                self.captured = Some(MtpBoundary {
                    cache: cache.clone(),
                    hidden: row,
                    len: segment.end,
                });
            }
        }
        self.cache = Some(cache);
        self.after_cur = None;
        Ok(seed)
    }
}

impl Proposer<Qwen35Model> for MtpProposer<'_> {
    fn kind(&self) -> ProposerKind {
        ProposerKind::Mtp
    }

    fn wants_hidden(&self) -> bool {
        true
    }

    fn warm(
        &mut self,
        model: &Qwen35Model,
        prompt: &[i32],
        prompt_hidden: Option<&Array>,
    ) -> Result<Option<Array>> {
        let hidden = prompt_hidden.ok_or_else(|| {
            Error::Msg("MtpProposer: the target did not return prompt hidden states".into())
        })?;
        if self.multimodal.is_none() {
            return self.warm_text(model, prompt, hidden);
        }
        if self.resume.is_some() || self.capture_at.is_some() {
            return Err(Error::Msg(
                "MtpProposer: a multimodal prompt never resumes from or feeds the prefix cache"
                    .into(),
            ));
        }
        let mut cache = model.new_mtp_cache().ok_or_else(|| {
            Error::Msg("MtpProposer: the model has no loaded MTP predictor".into())
        })?;
        // Seed with shifted prompt pairs: embed(x[j + 1]) + final-normalized target H[j], at
        // absolute positions 1..P-1. The first draft step pairs `cur` with H[P - 1].
        let prompt_len = prompt.len() as i32;
        let seed = if prompt_len > 1 {
            let shifted = match self.multimodal {
                Some(mm) => seq_rows(mm.embeddings, 1, prompt_len - 1)?,
                None => model.embed_input_ids(&input_ids(&prompt[1..]))?,
            };
            let aligned = seq_rows(hidden, 0, prompt_len - 1)?;
            let simple: Vec<i32>;
            let positions = match self.multimodal {
                Some(mm) => [
                    &mm.positions[0][1..],
                    &mm.positions[1][1..],
                    &mm.positions[2][1..],
                ],
                None => {
                    simple = (1..prompt_len).collect();
                    [simple.as_slice(), simple.as_slice(), simple.as_slice()]
                }
            };
            Some(model.mtp_warm_from_embeds(&shifted, &aligned, &mut cache, positions)?)
        } else {
            None
        };
        self.cache = Some(cache);
        self.after_cur = None;
        Ok(seed)
    }

    fn propose(
        &mut self,
        model: &Qwen35Model,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        let previous = ctx.previous_hidden.ok_or_else(|| {
            Error::Msg("MtpProposer: no previous target hidden row for the draft step".into())
        })?;
        let cache = self.cache()?;
        let (mut hidden, mut logits) =
            model.mtp_step(&input_ids(&[ctx.cur]), previous, cache, ctx.position)?;
        eval([&hidden, &logits])?;
        let after_cur = cache.clone();

        let mut drafts = Vec::with_capacity(ctx.max_drafts);
        let mut dists = Vec::with_capacity(ctx.max_drafts);
        let mut draft_history = ctx.history.to_vec();
        for step in 0..ctx.max_drafts {
            let (draft, dist) = sampler.sample_draft(&logits, &draft_history)?;
            dists.extend(dist);
            drafts.push(draft);
            draft_history.push(draft);
            if sampler.is_stop(draft) {
                break;
            }
            if step + 1 < ctx.max_drafts {
                let cache = self.cache()?;
                (hidden, logits) = model.mtp_step(
                    &input_ids(&[draft]),
                    &hidden,
                    cache,
                    ctx.position + 1 + step as i32,
                )?;
                eval([&hidden, &logits])?;
            }
        }
        self.after_cur = Some(after_cur);
        Ok(Proposal { drafts, dists })
    }

    fn commit(
        &mut self,
        model: &Qwen35Model,
        _cur: i32,
        accepted: &[i32],
        kept_hidden: Option<&Array>,
        position: i32,
    ) -> Result<()> {
        // Provisional recursive state never survives reconciliation: restore the cache after
        // `cur`, then replay each accepted draft paired with its authoritative target predecessor.
        if let Some(after_cur) = self.after_cur.take() {
            self.cache = Some(after_cur);
        }
        if accepted.is_empty() {
            return Ok(());
        }
        let kept = kept_hidden.ok_or_else(|| {
            Error::Msg("MtpProposer: no target hidden rows for the accepted drafts".into())
        })?;
        for (i, &draft) in accepted.iter().enumerate() {
            let predecessor = seq_rows(kept, i as i32, 1)?;
            let cache = self.cache()?;
            let (hidden, logits) = model.mtp_step(
                &input_ids(&[draft]),
                &predecessor,
                cache,
                position + i as i32,
            )?;
            eval([&hidden, &logits])?;
        }
        Ok(())
    }
}

/// Generate with Qwen3.8's native MTP predictor and exact target verification — the engine with
/// an [`MtpProposer`] over a text prompt. `num_draft` is the maximum drafts per target pass.
#[allow(clippy::too_many_arguments)]
pub fn generate_qwen35_mtp(
    model: &Qwen35Model,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    num_draft: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn RewindableConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<(GenerationOutput, SpeculativeStats)> {
    if num_draft == 0 {
        return Err(Error::Msg(
            "generate_qwen35_mtp: num_draft must be >= 1".into(),
        ));
    }
    let run = generate_speculative(
        model,
        &mut MtpProposer::new(),
        SpeculativePrompt::Tokens(prompt_ids),
        config,
        num_draft,
        cancel,
        on_event,
        EngineOptions {
            constraint,
            should_stop,
            ..EngineOptions::default()
        },
    )?;
    Ok((run.output, run.stats))
}
