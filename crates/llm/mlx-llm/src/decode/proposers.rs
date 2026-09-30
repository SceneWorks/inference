//! The MLX engine's proposal sources (epic sc-24432, stories sc-24434 and sc-24436): prompt
//! lookup ([`NgramProposer`], any target), the native Qwen3.8 multi-token predictor
//! ([`MtpProposer`], the Qwen35 target) and a separate draft model ([`DraftModelProposer`], any
//! target and any draft decoder). Each implements [`Proposer`]; the loop that verifies, accepts
//! and rolls back is the one [`engine`](super::engine).

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
/// commit keeps the accepted drafts already fed — by truncation, or by restoring the step start
/// on a cache that cannot truncate (the Qwen35 hybrid) — and queues the rest of the accepted run
/// for the next proposal's first forward, so no step pays an extra draft forward to catch up.
pub struct DraftModelProposer<'d, D: SpeculativeTarget + ?Sized> {
    draft: &'d D,
    cache: Option<D::Cache>,
    rollback: D::Rollback,
    /// Committed tokens not yet in the draft cache, fed ahead of `cur` by the next proposal.
    pending: Vec<i32>,
    /// Drafts the last proposal wrote into the cache past the step start (`None`: no proposal
    /// this step, so `cur` itself was never fed).
    fed: Option<usize>,
    /// Draft-model forwards issued (its own cost, separate from the target's).
    pub draft_forwards: u64,
}

impl<'d, D: SpeculativeTarget + ?Sized> DraftModelProposer<'d, D> {
    /// A proposer over `draft`.
    pub fn new(draft: &'d D) -> Self {
        Self {
            draft,
            cache: None,
            rollback: draft.rollback(),
            pending: Vec::new(),
            fed: None,
            draft_forwards: 0,
        }
    }

    fn cache(&mut self) -> Result<&mut D::Cache> {
        self.cache
            .as_mut()
            .ok_or_else(|| Error::Msg("DraftModelProposer: propose before warm".into()))
    }

    /// Run `ids` through the draft at the end of its cache, returning the last row's logits.
    fn feed(&mut self, ids: &[i32]) -> Result<Array> {
        let draft = self.draft;
        let cache = self.cache()?;
        let offset = draft.cache_len(cache);
        let out = draft.forward(cache, &input_ids(ids), offset, LogitsScope::Last, false)?;
        self.draft_forwards += 1;
        Ok(out.logits)
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
#[derive(Default)]
pub struct MtpProposer<'a> {
    multimodal: Option<&'a Qwen35MtpMultimodalPrompt<'a>>,
    cache: Option<MtpCache>,
    after_cur: Option<MtpCache>,
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

    fn cache(&mut self) -> Result<&mut MtpCache> {
        self.cache
            .as_mut()
            .ok_or_else(|| Error::Msg("MtpProposer: propose before warm".into()))
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
        let mut cache = model.new_mtp_cache().ok_or_else(|| {
            Error::Msg("MtpProposer: the model has no loaded MTP predictor".into())
        })?;
        let hidden = prompt_hidden.ok_or_else(|| {
            Error::Msg("MtpProposer: the target did not return prompt hidden states".into())
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
