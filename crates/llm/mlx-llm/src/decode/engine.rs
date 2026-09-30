//! The MLX speculative engine (epic sc-24432, story sc-24434).
//!
//! **One** decode loop for every MLX text decoder, mirroring Candle's `decode::engine`: the target
//! model sits behind [`SpeculativeTarget`], the proposal source behind [`Proposer`], the
//! cache-rollback policy behind [`CacheRollback`] and the token draws behind [`TokenSampler`]. It
//! replaces the Qwen35-only MTP loop (the pre-engine `qwen35_mtp` module, whose verify / accept /
//! rollback logic moved here) and the Causal-only prompt-lookup loop; with [`NoProposer`] it is the
//! token-at-a-time loop the provider runs when speculation is off, token-for-token identical to
//! [`generate`](super::generate) (tested below against that non-engine loop, so engine-vs-engine
//! agreement cannot hide a shared defect).
//!
//! ## One step
//! 1. **Propose** up to `K` drafts after `cur` (the last committed token, not yet in the cache),
//!    clamped so the commit cannot overrun the budget. A proposer draws any sampled drafts through
//!    the engine's [`DraftSampler`] — the same seeded sampler, knobs and (rewindable) constraint.
//! 2. **Verify** `[cur, d₁ … dₖ]` in one target forward. A step with no drafts runs the ordinary
//!    last-position forward ([`LogitsScope::Last`]), exactly the token-at-a-time loop's step;
//!    otherwise every position's logits ([`LogitsScope::All`]).
//! 3. **Decide** with `core_llm`'s policy: [`greedy_commit`] over the per-row argmax (one device ->
//!    host transfer, [`TokenSampler::argmax_rows`]) for a plain greedy run, the per-row host draw
//!    for a penalized or constrained greedy run, and the distribution-preserving
//!    [`accept_token`] rejection rule for a stochastic run (E1).
//! 4. **Recover** the cache to `[cur, accepted…]` through the target's [`CacheRollback`]: a
//!    [`Rollback::Direct`] truncation where the cache supports it (the softmax KV cache), else the
//!    cache is [`Rollback::Restored`] to the step start and the kept prefix is **replayed** in one
//!    forward (the Qwen35 hybrid, whose DeltaNet state cannot be truncated). Both are counted.
//! 5. **Commit** through the shared event path: stop tokens, constraint advance, the caller's
//!    stop predicate and the budget, then the proposer reconciles its own state
//!    ([`Proposer::commit`]).
//!
//! Every run returns a [`DecodeReport`] naming the proposer, the depth, the verify steps and
//! accepted drafts, the recovery replays and the measured sampler path (E3); the provider adds the
//! request's fallback reasons (E2).
//!
//! ## Extension points
//! A new proposal source (a draft model, a companion MTP head) is a [`Proposer`] impl; a cache that
//! can roll back per token (a DeltaNet checkpoint ring) is a [`CacheRollback`] impl returning
//! [`Rollback::Direct`]; a device-side sampler is a [`TokenSampler`] impl handed to
//! [`EngineOptions::sampler`]. None of them touches this loop.
//!
//! The verify-vs-decode kernel caveat of [`speculative`](super::speculative) applies: a multi-token
//! verify rounds a few bf16 ULP differently from the single-token step, so on real weights a greedy
//! speculative run tracks, rather than bit-matches, the token-at-a-time run where a near-tie flips.

use std::time::Instant;

use mlx_rs::transforms::eval;
use mlx_rs::Array;

use core_llm::speculative::{accept_token, greedy_commit, Acceptance};
use core_llm::{
    CudaGraphsReport, DecodeReport, HostSampleReason, PathReport, ProposerKind, SamplerPath,
};

use crate::decode::cancel::CancelFlag;
use crate::decode::speculative::SpeculativeStats;
use crate::decode::stream::{
    default_seed, ConstraintMask, FinishReason, GenerationConfig, GenerationOutput,
    GenerationTimer, StreamEvent,
};
use crate::decode::BufferRelease;
use crate::error::{Error, Result};
use crate::models::{CausalLm, Qwen35Cache, Qwen35Model};
use crate::primitives::input_ids;
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
use crate::primitives::sampler::{
    acceptance_target, argmax_rows_device, sample, SamplingParams, SplitMix64, TokenRng,
};

/// Constraint state that can be rewound after speculative exploration. The committed state is
/// advanced only by tokens that are actually emitted.
pub trait RewindableConstraintMask: ConstraintMask {
    /// Opaque checkpoint for the current committed state.
    fn checkpoint(&self) -> usize;
    /// Restore a checkpoint previously returned by [`Self::checkpoint`].
    fn rewind(&mut self, checkpoint: usize);
}

// ---------------------------------------------------------------------------------------------
// The target seam.
// ---------------------------------------------------------------------------------------------

/// Which positions' logits a target forward returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitsScope {
    /// The last position only, `[1, vocab]` — prefill and the token-at-a-time step.
    Last,
    /// Every position, `[1, n, vocab]` — the verify step.
    All,
}

/// One target forward's outputs.
#[derive(Clone, Debug)]
pub struct TargetOutput {
    /// `[1, vocab]` for [`LogitsScope::Last`], `[1, n, vocab]` for [`LogitsScope::All`].
    pub logits: Array,
    /// The final-normalized hidden rows for every fed position, `[1, n, hidden]`, when asked for.
    pub hidden: Option<Array>,
}

/// How a [`CacheRollback`] recovered the cache after a verify step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rollback {
    /// The cache now holds exactly the kept prefix (a truncation, or nothing to drop).
    Direct,
    /// The cache is back at the step start; the engine must replay the kept prefix.
    Restored,
}

/// The cache-rollback policy the engine recovers a partially accepted verify step through. A
/// strategy lives for one run; [`begin`](Self::begin) precedes every verify forward that carries
/// drafts and [`recover`](Self::recover) ends that step.
pub trait CacheRollback<C> {
    /// Stable lower-case label (`truncate`, `snapshot_replay`).
    fn label(&self) -> &'static str;
    /// Remember whatever a later recovery needs, before the verify forward writes the cache.
    fn begin(&mut self, cache: &C);
    /// Keep the first `keep` positions after the verify forward, ending the step. `keep` is the
    /// step start plus `1 + accepted`, at most the cache's current length.
    fn recover(&mut self, cache: &mut C, keep: i32) -> Result<Rollback>;
}

/// Direct rollback by [`KvCache::truncate`] — every softmax KV cache: dropping rejected positions
/// is bookkeeping, no forward.
#[derive(Clone, Copy, Debug, Default)]
pub struct TruncateRollback;

impl<C: KvCache> CacheRollback<C> for TruncateRollback {
    fn label(&self) -> &'static str {
        "truncate"
    }

    fn begin(&mut self, _: &C) {}

    fn recover(&mut self, cache: &mut C, keep: i32) -> Result<Rollback> {
        if keep < cache.offset() {
            cache.truncate(keep)?;
        }
        Ok(Rollback::Direct)
    }
}

/// Snapshot-and-replay rollback for a cache whose state cannot be truncated (the Qwen35 hybrid's
/// DeltaNet recurrence): the step start is cloned before the verify forward — MLX arrays are
/// refcounted and the KV slots copy on write, so the clone is cheap — and a partial acceptance
/// restores it, after which the engine replays the kept prefix. The snapshot is released at the
/// end of every step so the next step's in-place cache writes never copy a buffer it still holds.
#[derive(Clone, Debug)]
pub struct SnapshotRollback<C> {
    snapshot: Option<C>,
}

impl<C> Default for SnapshotRollback<C> {
    fn default() -> Self {
        Self { snapshot: None }
    }
}

impl<C: Clone + KvCache> CacheRollback<C> for SnapshotRollback<C> {
    fn label(&self) -> &'static str {
        "snapshot_replay"
    }

    fn begin(&mut self, cache: &C) {
        self.snapshot = Some(cache.clone());
    }

    fn recover(&mut self, cache: &mut C, keep: i32) -> Result<Rollback> {
        let snapshot = self.snapshot.take();
        if keep >= cache.offset() {
            return Ok(Rollback::Direct);
        }
        // Moved, not cloned: the replay then writes the restored buffers in place.
        *cache = snapshot.ok_or_else(|| {
            Error::Msg("SnapshotRollback: recover without a step-start snapshot".into())
        })?;
        Ok(Rollback::Restored)
    }
}

/// A target model the engine verifies against.
pub trait SpeculativeTarget {
    /// The target's decode cache.
    type Cache;
    /// How a partially accepted verify step is recovered.
    type Rollback: CacheRollback<Self::Cache>;

    /// A fresh, empty cache.
    fn new_cache(&self) -> Self::Cache;
    /// Positions the cache holds — the next forward's first cache position.
    fn cache_len(&self, cache: &Self::Cache) -> i32;
    /// A fresh rollback strategy for one run.
    fn rollback(&self) -> Self::Rollback;
    /// Run `ids` through the target, appending them to `cache`. `rope_offset` is the RoPE position
    /// of `ids[0]` (the cache length plus any multimodal continuation shift).
    fn forward(
        &self,
        cache: &mut Self::Cache,
        ids: &[i32],
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput>;
    /// The KV cache label for the report (`growing`).
    fn kv_cache_label(&self) -> &'static str {
        "growing"
    }
    /// How attention is computed, for the report (`gqa`, `expanded`).
    fn attention_label(&self) -> &'static str;
}

impl SpeculativeTarget for CausalLm {
    type Cache = ContiguousKvCache;
    type Rollback = TruncateRollback;

    fn new_cache(&self) -> ContiguousKvCache {
        CausalLm::new_cache(self)
    }

    fn cache_len(&self, cache: &ContiguousKvCache) -> i32 {
        cache.offset()
    }

    fn rollback(&self) -> TruncateRollback {
        TruncateRollback
    }

    fn forward(
        &self,
        cache: &mut ContiguousKvCache,
        ids: &[i32],
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        if want_hidden {
            return Err(Error::Msg(
                "CausalLm: the speculative target does not return hidden states".into(),
            ));
        }
        let ids = input_ids(ids);
        let logits = match scope {
            LogitsScope::Last => self.decode_logits(&ids, cache, rope_offset)?,
            LogitsScope::All => self.decode_logits_all(&ids, cache, rope_offset)?,
        };
        Ok(TargetOutput {
            logits,
            hidden: None,
        })
    }

    fn attention_label(&self) -> &'static str {
        // `sdpa_capped` leaves the fused native-GQA kernel for the eager, K/V-expanded path when
        // the scores are soft-capped or the MLA q/k and v head dims differ.
        let cfg = self.config();
        let mla_split = cfg
            .mla
            .is_some_and(|m| m.qk_nope_head_dim + m.qk_rope_head_dim != m.v_head_dim);
        if cfg.attn_logit_softcap.is_some() || mla_split {
            "expanded"
        } else {
            "gqa"
        }
    }
}

impl SpeculativeTarget for Qwen35Model {
    type Cache = Qwen35Cache;
    type Rollback = SnapshotRollback<Qwen35Cache>;

    fn new_cache(&self) -> Qwen35Cache {
        Qwen35Model::new_cache(self)
    }

    fn cache_len(&self, cache: &Qwen35Cache) -> i32 {
        cache.offset()
    }

    fn rollback(&self) -> SnapshotRollback<Qwen35Cache> {
        SnapshotRollback::default()
    }

    fn forward(
        &self,
        cache: &mut Qwen35Cache,
        ids: &[i32],
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        let ids = input_ids(ids);
        Ok(match (scope, want_hidden) {
            (LogitsScope::Last, false) => TargetOutput {
                logits: self.decode_logits(&ids, cache, rope_offset)?,
                hidden: None,
            },
            (LogitsScope::All, false) => TargetOutput {
                logits: self.forward(&ids, cache, rope_offset)?,
                hidden: None,
            },
            (LogitsScope::Last, true) => {
                let (hidden, logits) =
                    self.prefill_hidden_and_last_logits(&ids, cache, rope_offset)?;
                TargetOutput {
                    logits,
                    hidden: Some(hidden),
                }
            }
            (LogitsScope::All, true) => {
                let (hidden, logits) = self.hidden_and_logits(&ids, cache, rope_offset)?;
                TargetOutput {
                    logits,
                    hidden: Some(hidden),
                }
            }
        })
    }

    fn attention_label(&self) -> &'static str {
        "gqa"
    }
}

// ---------------------------------------------------------------------------------------------
// The sampler seam.
// ---------------------------------------------------------------------------------------------

/// The engine's token draws — the seam a device sampler plugs in behind. Every draw is recorded,
/// so [`path`](Self::path) is measured rather than inferred from the request.
pub trait TokenSampler {
    /// The sampling knobs.
    fn params(&self) -> &SamplingParams;
    /// Draw one token from a `[1, vocab]` (or `[vocab]`) logits row given the running `history`
    /// (the penalty window) and an optional constraint mask.
    fn sample(&mut self, logits: &Array, history: &[i32], allowed: Option<&[bool]>) -> Result<i32>;
    /// The argmax of every row of `[1, n, vocab]` — the plain-greedy verify decision.
    fn argmax_rows(&mut self, logits: &Array) -> Result<Vec<i32>>;
    /// The shaped distribution of one row, never empty: the acceptance test's target `p` and a
    /// sampled draft's proposal `q`.
    fn distribution(
        &mut self,
        logits: &Array,
        history: &[i32],
        allowed: Option<&[bool]>,
    ) -> Result<Vec<(i32, f32)>>;
    /// One uniform `[0, 1)` draw from the seeded stream.
    fn uniform(&mut self) -> f32;
    /// Where the run's draws happened: `Host(reason)` if any was on the host (the latest host
    /// reason), `Device` if all were on the device, `None` if nothing was drawn.
    fn path(&self) -> Option<SamplerPath>;
}

/// The MLX sampler: [`sample`] with the seeded [`SplitMix64`], plain-greedy draws taken as the
/// on-device argmax and everything else on the host (MLX has no device sampler for temperature,
/// penalties or a constraint mask).
#[derive(Clone, Debug)]
pub struct MlxSampler {
    params: SamplingParams,
    rng: SplitMix64,
    device_draws: u64,
    host_draws: u64,
    last_host: Option<HostSampleReason>,
}

impl MlxSampler {
    /// A sampler for `params` seeded with `seed`.
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            rng: SplitMix64::new(seed),
            device_draws: 0,
            host_draws: 0,
            last_host: None,
        }
    }

    /// The sampler a [`GenerationConfig`] asks for (its knobs, its seed or a fresh one) — the same
    /// seeded stream [`generate`](super::generate) draws from.
    pub fn from_config(config: &GenerationConfig) -> Self {
        Self::new(config.sampling, config.seed.unwrap_or_else(default_seed))
    }

    fn penalized(&self) -> bool {
        self.params.repetition_penalty != 1.0 || self.params.presence_penalty != 0.0
    }

    fn note_host(&mut self, reason: HostSampleReason) {
        self.host_draws += 1;
        self.last_host = Some(reason);
    }
}

impl TokenSampler for MlxSampler {
    fn params(&self) -> &SamplingParams {
        &self.params
    }

    fn sample(&mut self, logits: &Array, history: &[i32], allowed: Option<&[bool]>) -> Result<i32> {
        if allowed.is_some() {
            self.note_host(HostSampleReason::Constraint);
        } else if self.penalized() {
            self.note_host(HostSampleReason::Penalty);
        } else if self.params.temperature > 0.0 {
            self.note_host(HostSampleReason::DeviceUnavailable);
        } else {
            self.device_draws += 1;
        }
        sample(logits, history, &self.params, &mut self.rng, allowed)
    }

    fn argmax_rows(&mut self, logits: &Array) -> Result<Vec<i32>> {
        let rows = argmax_rows_device(logits)?;
        self.device_draws += rows.len() as u64;
        Ok(rows)
    }

    fn distribution(
        &mut self,
        logits: &Array,
        history: &[i32],
        allowed: Option<&[bool]>,
    ) -> Result<Vec<(i32, f32)>> {
        self.note_host(if allowed.is_some() {
            HostSampleReason::Constraint
        } else if self.penalized() {
            HostSampleReason::Penalty
        } else {
            HostSampleReason::SpeculativeDistribution
        });
        acceptance_target(logits, history, &self.params, allowed)
    }

    fn uniform(&mut self) -> f32 {
        self.rng.next_f32()
    }

    fn path(&self) -> Option<SamplerPath> {
        match (self.host_draws, self.device_draws) {
            (0, 0) => None,
            (0, _) => Some(SamplerPath::Device),
            _ => self.last_host.map(SamplerPath::Host),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The proposer seam.
// ---------------------------------------------------------------------------------------------

/// What a proposer returns for one step.
#[derive(Clone, Debug, Default)]
pub struct Proposal {
    /// The proposed continuation after `cur`, in order.
    pub drafts: Vec<i32>,
    /// For a stochastic run, the proposal distribution `q` each draft was drawn from, one per
    /// draft. Empty for a greedy run, and for a deterministic proposer (a point mass is assumed).
    pub dists: Vec<Vec<(i32, f32)>>,
}

/// What a proposer sees when asked for drafts.
pub struct ProposeContext<'a> {
    /// The last committed token — the first token the verify step feeds — not yet in the cache.
    pub cur: i32,
    /// The prompt plus every committed token (ends with `cur`): the n-gram context and the
    /// penalty window.
    pub history: &'a [i32],
    /// The target's final-normalized hidden row `[1, 1, hidden]` for the position before `cur`,
    /// when the proposer asked for hidden states.
    pub previous_hidden: Option<&'a Array>,
    /// The RoPE position `cur` occupies.
    pub position: i32,
    /// At most this many drafts (already clamped to the remaining budget, `>= 1`).
    pub max_drafts: usize,
}

/// One host-sampled draft: the token and, for a stochastic run, the proposal distribution it was
/// drawn from.
pub type DraftSample = (i32, Option<Vec<(i32, f32)>>);

/// The engine's draft-sampling policy handed to a proposer: the knobs, the run's sampler and the
/// (rewindable) constraint.
pub struct DraftSampler<'a, 'c> {
    config: &'a GenerationConfig,
    sampler: &'a mut dyn TokenSampler,
    constraint: Option<&'a mut (dyn RewindableConstraintMask + 'c)>,
}

impl DraftSampler<'_, '_> {
    /// Whether the run is greedy (`temperature <= 0`).
    pub fn greedy(&self) -> bool {
        self.config.sampling.temperature <= 0.0
    }

    /// Whether `token` is a stop token — a proposer stops drafting past one.
    pub fn is_stop(&self, token: i32) -> bool {
        self.config.stop_tokens.contains(&token)
    }

    /// Sample one draft from `[1, vocab]` `logits` given the provisional `draft_history`: the
    /// token and — for a stochastic run — its proposal distribution. Advances the constraint by
    /// the draft unless it is a stop token (the engine rewinds the constraint after the proposal).
    pub fn sample_draft(&mut self, logits: &Array, draft_history: &[i32]) -> Result<DraftSample> {
        let dist = if self.greedy() {
            None
        } else {
            let mask = self.constraint.as_mut().map(|c| c.allowed());
            Some(self.sampler.distribution(logits, draft_history, mask)?)
        };
        let mask = self.constraint.as_mut().map(|c| c.allowed());
        let draft = self.sampler.sample(logits, draft_history, mask)?;
        if !self.is_stop(draft) {
            if let Some(c) = self.constraint.as_mut() {
                c.accept(draft);
            }
        }
        Ok((draft, dist))
    }
}

/// A proposal source for the engine over target `T`. Implementations:
/// [`NoProposer`], [`NgramProposer`](super::proposers::NgramProposer) (any target) and
/// [`MtpProposer`](super::proposers::MtpProposer) (the Qwen35 MTP head).
pub trait Proposer<T: SpeculativeTarget + ?Sized> {
    /// Which proposer this is (stamped on the report).
    fn kind(&self) -> ProposerKind;

    /// Whether the target's final-normalized hidden states are needed.
    fn wants_hidden(&self) -> bool {
        false
    }

    /// Warm from the prefilled prompt: `prompt` is the effective prompt ids, `prompt_hidden` the
    /// target's hidden rows for every prompt position when [`wants_hidden`](Self::wants_hidden).
    /// Returns an array to synchronize at the prefill boundary (a lazily built warm-up graph).
    fn warm(
        &mut self,
        target: &T,
        prompt: &[i32],
        prompt_hidden: Option<&Array>,
    ) -> Result<Option<Array>>;

    /// Propose up to `ctx.max_drafts` drafts after `ctx.cur`. Called only when `max_drafts > 0`.
    fn propose(
        &mut self,
        target: &T,
        ctx: &ProposeContext<'_>,
        sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal>;

    /// The verify outcome: `accepted` is the accepted draft prefix; `kept_hidden` the target's
    /// hidden rows `[1, 1 + accepted.len(), hidden]` for `[cur, accepted…]` when
    /// [`wants_hidden`](Self::wants_hidden); `position` the RoPE position of `accepted[0]`.
    fn commit(
        &mut self,
        target: &T,
        cur: i32,
        accepted: &[i32],
        kept_hidden: Option<&Array>,
        position: i32,
    ) -> Result<()>;
}

/// No proposal source: the engine is the token-at-a-time loop and reports `proposer=none`.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoProposer;

impl<T: SpeculativeTarget + ?Sized> Proposer<T> for NoProposer {
    fn kind(&self) -> ProposerKind {
        ProposerKind::None
    }

    fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
        Ok(None)
    }

    fn propose(
        &mut self,
        _: &T,
        _: &ProposeContext<'_>,
        _: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        Ok(Proposal::default())
    }

    fn commit(&mut self, _: &T, _: i32, _: &[i32], _: Option<&Array>, _: i32) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// The engine.
// ---------------------------------------------------------------------------------------------

/// How the engine starts: a token prompt it prefills itself, or a cache the caller already
/// prefilled (the multimodal paths, whose prefill is spliced embeddings).
pub enum SpeculativePrompt<'a, C> {
    /// Prefill these ids into a fresh [`SpeculativeTarget::new_cache`] at position 0.
    Tokens(&'a [i32]),
    /// A caller-prefilled cache positioned past the prompt.
    Prefilled {
        /// The cache, borrowed: the caller still owns it afterwards.
        cache: &'a mut C,
        /// Last-position logits of the prefill, `[1, vocab]`.
        logits: Array,
        /// The target's hidden rows for every prompt position, when the proposer wants them.
        hidden: Option<Array>,
        /// The effective prompt ids (the penalty window / n-gram context).
        history: &'a [i32],
        /// Shift between cache positions and RoPE positions for the continuation (the M-RoPE
        /// continuation delta; `0` for text).
        position_delta: i32,
    },
}

/// The engine's optional request seams. `'c` is the lifetime of the constraint's and the
/// sampler's own borrows.
#[derive(Default)]
pub struct EngineOptions<'a, 'c> {
    /// A rewindable per-step constraint (structured output).
    pub constraint: Option<&'a mut (dyn RewindableConstraintMask + 'c)>,
    /// A caller stop predicate checked after each emitted token (request stop strings).
    pub should_stop: Option<&'a dyn Fn() -> bool>,
    /// When set, the run is timed: the prefill phase starts at this instant and closes, after a
    /// device synchronization, once the prompt is prefilled and the proposer warmed.
    pub prefill_clock: Option<Instant>,
    /// The sampler; `None` runs an [`MlxSampler`] seeded from the config.
    pub sampler: Option<&'a mut (dyn TokenSampler + 'c)>,
}

/// A finished engine run.
pub struct SpeculativeRun {
    /// The generated tokens and why generation stopped.
    pub output: GenerationOutput,
    /// The raw speculation counters.
    pub stats: SpeculativeStats,
    /// The measured decode report (fallbacks are the caller's to add).
    pub report: DecodeReport,
    timer: Option<GenerationTimer>,
}

impl SpeculativeRun {
    /// The synchronized phase timer of a timed run ([`EngineOptions::prefill_clock`]), which the
    /// provider finishes after its own stream dispatch.
    pub(crate) fn take_timer(&mut self) -> Option<GenerationTimer> {
        self.timer.take()
    }
}

/// Generate through the engine with `proposer` proposing up to `drafts` tokens per verify step
/// (`0`, or [`NoProposer`], is the token-at-a-time loop). Returns [`Error::Canceled`] if `cancel`
/// is already set before any inference; a mid-run cancel returns the partial output as
/// [`FinishReason::Cancelled`].
#[allow(clippy::too_many_arguments)]
pub fn generate_speculative<T, P>(
    target: &T,
    proposer: &mut P,
    prompt: SpeculativePrompt<'_, T::Cache>,
    config: &GenerationConfig,
    drafts: usize,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    options: EngineOptions<'_, '_>,
) -> Result<SpeculativeRun>
where
    T: SpeculativeTarget + ?Sized,
    P: Proposer<T> + ?Sized,
{
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    let EngineOptions {
        mut constraint,
        should_stop,
        prefill_clock,
        sampler,
    } = options;
    let mut owned_sampler;
    let sampler: &mut dyn TokenSampler = match sampler {
        Some(s) => s,
        None => {
            owned_sampler = MlxSampler::from_config(config);
            &mut owned_sampler
        }
    };
    let kind = proposer.kind();
    let wants_hidden = proposer.wants_hidden();
    let width = if kind == ProposerKind::None {
        0
    } else {
        drafts
    };
    let mut stats = SpeculativeStats::default();
    let mut generated: Vec<i32> = Vec::new();
    let mut finish = FinishReason::MaxTokens;
    let mut timer = prefill_clock.map(GenerationTimer::start_at);
    let finished = |generated: Vec<i32>,
                    finish: FinishReason,
                    stats: SpeculativeStats,
                    sampler: &dyn TokenSampler,
                    timer: Option<GenerationTimer>,
                    on_event: &mut dyn FnMut(StreamEvent)| {
        on_event(StreamEvent::Done {
            reason: finish,
            generated: generated.len(),
        });
        SpeculativeRun {
            output: GenerationOutput {
                tokens: generated,
                finish_reason: finish,
            },
            report: report(target, kind, width, &stats, sampler.path()),
            stats,
            timer,
        }
    };

    if let SpeculativePrompt::Tokens([]) = prompt {
        return Err(Error::Msg("generate_speculative: empty prompt".into()));
    }
    if config.max_new_tokens == 0 {
        if let Some(timer) = timer.as_mut() {
            timer.finish_prefill(std::iter::empty())?;
        }
        return Ok(finished(generated, finish, stats, sampler, timer, on_event));
    }

    // ---- Prefill (or adopt the caller's), warm the proposer, close the prefill phase. ----
    let mut owned_cache = None;
    let (cache, logits, prompt_hidden, mut history, position_delta) = match prompt {
        SpeculativePrompt::Tokens(ids) => {
            let cache = owned_cache.insert(target.new_cache());
            let out = target.forward(cache, ids, 0, LogitsScope::Last, wants_hidden)?;
            (cache, out.logits, out.hidden, ids.to_vec(), 0)
        }
        SpeculativePrompt::Prefilled {
            cache,
            logits,
            hidden,
            history,
            position_delta,
        } => {
            if history.is_empty() {
                return Err(Error::Msg("generate_speculative: empty prompt".into()));
            }
            (cache, logits, hidden, history.to_vec(), position_delta)
        }
    };
    stats.forwards += 1;
    let warm = proposer.warm(target, &history, prompt_hidden.as_ref())?;
    let mut previous_hidden = match prompt_hidden.as_ref() {
        Some(h) => Some(last_row(h)?),
        None => None,
    };
    if let Some(timer) = timer.as_mut() {
        let mut arrays = vec![&logits];
        arrays.extend(prompt_hidden.as_ref());
        arrays.extend(warm.as_ref());
        timer.finish_prefill(arrays)?;
    }
    let mut rollback = target.rollback();

    // ---- First token: an ordinary draw from the prefill logits. ----
    let first = {
        let mask = constraint.as_mut().map(|c| c.allowed());
        sampler.sample(&logits, &history, mask)?
    };
    // The draw evaluated the target prefill; force the proposer's warm-up graph and the last
    // hidden row (a gather that owns its buffer once evaluated), then retire every prompt-length
    // array. The buffer release rides the first `advance`, after step 0 retires its transients.
    let mut pending: Vec<&Array> = warm.iter().collect();
    pending.extend(previous_hidden.as_ref());
    if !pending.is_empty() {
        eval(pending)?;
    }
    drop(warm);
    drop(prompt_hidden);
    drop(logits);
    let mut release = BufferRelease::new();
    if config.stop_tokens.contains(&first) {
        finish = FinishReason::StopToken;
        return Ok(finished(generated, finish, stats, sampler, timer, on_event));
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
        let remaining = config.max_new_tokens - generated.len();
        let k = width.min(remaining.saturating_sub(1));
        let base = target.cache_len(cache);
        let position = base + position_delta;
        let checkpoint = constraint.as_ref().map(|c| c.checkpoint());

        // 1. Propose.
        let proposal = if k == 0 {
            Proposal::default()
        } else {
            let ctx = ProposeContext {
                cur,
                history: &history,
                previous_hidden: previous_hidden.as_ref(),
                position,
                max_drafts: k,
            };
            let mut draft_sampler = DraftSampler {
                config,
                sampler: &mut *sampler,
                constraint: constraint.as_deref_mut(),
            };
            proposer.propose(target, &ctx, &mut draft_sampler)?
        };
        if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), checkpoint) {
            c.rewind(checkpoint);
        }
        let Proposal {
            drafts: mut draft_ids,
            dists,
        } = proposal;
        // Nothing past the first stop token is a proposal: it could never be committed.
        if let Some(stop) = draft_ids
            .iter()
            .position(|t| config.stop_tokens.contains(t))
        {
            draft_ids.truncate(stop + 1);
        }
        draft_ids.truncate(k);
        let num_drafts = draft_ids.len();
        stats.proposed += num_drafts;
        if cancel.is_cancelled() {
            // Nothing has been written to the target cache yet.
            finish = FinishReason::Cancelled;
            break;
        }

        // 2. Verify `[cur, drafts…]` in one target forward.
        let mut verify = Vec::with_capacity(1 + num_drafts);
        verify.push(cur);
        verify.extend_from_slice(&draft_ids);
        let scope = if num_drafts == 0 {
            LogitsScope::Last
        } else {
            rollback.begin(cache);
            LogitsScope::All
        };
        let out = target.forward(cache, &verify, position, scope, wants_hidden)?;
        if let Some(hidden) = out.hidden.as_ref() {
            eval([hidden, &out.logits])?;
        }
        stats.forwards += 1;
        stats.verify_steps += 1;

        // 3. Decide.
        let (committed, accepted) = decide(
            &out.logits,
            &draft_ids,
            &dists,
            &history,
            config,
            sampler,
            constraint.as_deref_mut(),
        )?;
        if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), checkpoint) {
            c.rewind(checkpoint);
        }
        stats.accepted += accepted;

        // 4. Recover the cache to `[cur, accepted…]`.
        let keep_len = 1 + accepted;
        let kept_hidden = if num_drafts == 0 {
            out.hidden
        } else {
            match rollback.recover(cache, base + keep_len as i32)? {
                Rollback::Direct => {
                    if accepted < num_drafts {
                        stats.direct_rollbacks += 1;
                    }
                    match out.hidden {
                        Some(h) if accepted < num_drafts => Some(seq_rows(&h, 0, keep_len as i32)?),
                        other => other,
                    }
                }
                Rollback::Restored => {
                    drop(out);
                    let replayed = target.forward(
                        cache,
                        &verify[..keep_len],
                        position,
                        LogitsScope::Last,
                        wants_hidden,
                    )?;
                    if let Some(hidden) = replayed.hidden.as_ref() {
                        eval([hidden])?;
                    }
                    stats.forwards += 1;
                    stats.replays += 1;
                    replayed.hidden
                }
            }
        };
        release.advance(committed.len());
        proposer.commit(
            target,
            cur,
            &draft_ids[..accepted],
            kept_hidden.as_ref(),
            position + 1,
        )?;
        previous_hidden = match kept_hidden.as_ref() {
            Some(h) => Some(seq_rows(h, accepted as i32, 1)?),
            None => None,
        };

        // 5. Commit through the shared event path.
        for &token in &committed {
            if config.stop_tokens.contains(&token) {
                finish = FinishReason::StopToken;
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
            if should_stop.is_some_and(|stop| stop()) {
                finish = FinishReason::Stopped;
                break 'outer;
            }
            if generated.len() >= config.max_new_tokens {
                finish = FinishReason::MaxTokens;
                break 'outer;
            }
        }
    }

    Ok(finished(generated, finish, stats, sampler, timer, on_event))
}

/// The run's measured report. Backend features MLX does not have (CUDA graphs, NVFP4
/// projections, a fused-versus-reference primitive switch) report `none`.
fn report<T: SpeculativeTarget + ?Sized>(
    target: &T,
    proposer: ProposerKind,
    drafts: usize,
    stats: &SpeculativeStats,
    sampler: Option<SamplerPath>,
) -> DecodeReport {
    let path = match proposer {
        ProposerKind::None => "step_model",
        ProposerKind::Mtp => "mtp",
        ProposerKind::PromptLookup => "prompt_lookup",
        ProposerKind::DraftModel => "draft_model",
    };
    let none = || PathReport {
        path: "none".into(),
        reason: None,
    };
    DecodeReport {
        path: path.into(),
        proposer,
        draft_tokens: (proposer != ProposerKind::None)
            .then(|| u32::try_from(drafts).unwrap_or(u32::MAX)),
        sampler: sampler.map_or_else(|| "none".to_string(), |p| p.to_string()),
        kv_cache: target.kv_cache_label().into(),
        attention: target.attention_label().into(),
        cuda_graphs: CudaGraphsReport {
            enabled: false,
            path: "none".into(),
            ..CudaGraphsReport::default()
        },
        nvfp4_projections: none(),
        fused_primitives: none(),
        target_forwards: stats.forwards as u64,
        proposed_tokens: stats.proposed as u64,
        accepted_tokens: stats.accepted as u64,
        verify_steps: stats.verify_steps as u64,
        replay_forwards: stats.replays as u64,
        prefix_hit_tokens: 0,
        prefix_cache: none(),
        fallbacks: Vec::new(),
    }
}

/// The verify decision. Returns the committed run (accepted drafts + the bonus / correction token)
/// and the accepted draft count.
///
/// A step with **no drafts** has no decision to make: its one row is an ordinary draw, exactly the
/// token-at-a-time loop's. With drafts: a plain greedy, unconstrained run takes every row's argmax
/// in one transfer and applies [`greedy_commit`]; a penalized or constrained greedy run draws each
/// row on the host with the running history; a stochastic run applies the distribution-preserving
/// [`accept_token`] test per draft and draws the bonus from the row past the last accepted draft.
fn decide(
    logits: &Array,
    drafts: &[i32],
    dists: &[Vec<(i32, f32)>],
    history: &[i32],
    config: &GenerationConfig,
    sampler: &mut dyn TokenSampler,
    mut constraint: Option<&mut (dyn RewindableConstraintMask + '_)>,
) -> Result<(Vec<i32>, usize)> {
    if drafts.is_empty() {
        let mask = constraint.as_mut().map(|c| c.allowed());
        let token = sampler.sample(logits, history, mask)?;
        return Ok((vec![token], 0));
    }
    let rows = match logits.shape() {
        [_, n, _] => *n as usize,
        _ => 1,
    };
    if rows != drafts.len() + 1 {
        return Err(Error::Msg(format!(
            "verify returned {rows} positions for {} drafts",
            drafts.len()
        )));
    }
    let greedy = config.sampling.temperature <= 0.0;
    if config.sampling.is_plain_greedy() && constraint.is_none() {
        let target_argmax = sampler.argmax_rows(logits)?;
        return Ok(greedy_commit(&target_argmax, drafts));
    }

    let mut committed = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0usize;
    let mut running = history.to_vec();
    for (i, &draft) in drafts.iter().enumerate() {
        let row = logits_row(logits, i as i32)?;
        let mask = constraint.as_mut().map(|c| c.allowed());
        let outcome = if greedy {
            let target = sampler.sample(&row, &running, mask)?;
            if target == draft {
                Acceptance::Accepted(draft)
            } else {
                Acceptance::Rejected(target)
            }
        } else {
            let target = sampler.distribution(&row, &running, mask)?;
            let point_mass;
            let q = match dists.get(i) {
                Some(q) => q.as_slice(),
                None => {
                    point_mass = [(draft, 1.0)];
                    &point_mass[..]
                }
            };
            let (u_accept, u_resample) = (sampler.uniform(), sampler.uniform());
            accept_token(&target, q, draft, u_accept, u_resample)
        };
        let token = outcome.token();
        committed.push(token);
        if outcome.is_accepted() {
            accepted += 1;
        }
        if config.stop_tokens.contains(&token) {
            return Ok((committed, accepted));
        }
        if let Some(c) = constraint.as_mut() {
            c.accept(token);
        }
        running.push(token);
        if !outcome.is_accepted() {
            return Ok((committed, accepted));
        }
    }
    // Every draft accepted: the bonus from the position past the last draft.
    let row = logits_row(logits, drafts.len() as i32)?;
    let mask = constraint.as_mut().map(|c| c.allowed());
    committed.push(sampler.sample(&row, &running, mask)?);
    Ok((committed, accepted))
}

/// Position `i`'s logits row `[1, vocab]` from an all-positions `[1, n, vocab]` block.
fn logits_row(all: &Array, i: i32) -> Result<Array> {
    let idx = Array::from_slice(&[i], &[1]);
    let sh = all.shape();
    Ok(all.take_axis(&idx, 1)?.reshape(&[sh[0], sh[2]])?)
}

/// Rows `start .. start + len` of the sequence axis of `[1, n, hidden]`.
pub(crate) fn seq_rows(a: &Array, start: i32, len: i32) -> Result<Array> {
    let indices: Vec<i32> = (start..start + len).collect();
    Ok(a.take_axis(Array::from_slice(&indices, &[len]), 1)?)
}

/// The last sequence row of `[1, n, hidden]`, `[1, 1, hidden]`.
fn last_row(hidden: &Array) -> Result<Array> {
    let n = hidden.shape()[1];
    seq_rows(hidden, n - 1, 1)
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::config::{Architecture, ModelConfig};
    use crate::decode::proposers::{MtpProposer, NgramProposer};
    use crate::decode::stream::generate_with;
    use crate::models::qwen35::tests::{cfg_json, cfg_json_mtp, synthetic_weights};
    use crate::models::Qwen35Config;
    use crate::primitives::Weights;

    // ---- Fixtures: a tiny random llama-family decoder and the synthetic Qwen35 hybrid. ----

    /// A tiny random llama-family decoder (vocab 24) whose greedy continuation of [`PROMPT`]
    /// repeats its context, so prompt lookup proposes, accepts and rejects.
    pub(crate) fn causal() -> CausalLm {
        let cfg = ModelConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_layers: 2,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            vocab_size: 24,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            architecture: Architecture::Llama,
            max_position_embeddings: 0,
            quantization: None,
            moe: None,
            attn_logit_softcap: None,
            final_logit_softcap: None,
            query_pre_attn_scalar: None,
            partial_rotary_factor: 1.0,
            mla: None,
            yarn: None,
            mrope_section: None,
            gemma4: None,
        };
        let mut rng = SplitMix64::new(0x5EED_24434);
        let mut randn = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
            Array::from_slice(&data, shape)
        };
        let (h, v, inter) = (cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size);
        let (qd, kvd) = (
            cfg.num_heads * cfg.head_dim,
            cfg.num_kv_heads * cfg.head_dim,
        );
        let ones = |n: i32| Array::ones::<f32>(&[n]).unwrap();
        let mut m: HashMap<String, Array> = HashMap::new();
        m.insert("model.embed_tokens.weight".into(), randn(&[v, h]));
        m.insert("model.norm.weight".into(), ones(h));
        m.insert("lm_head.weight".into(), randn(&[v, h]));
        for i in 0..cfg.num_layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            m.insert(p("input_layernorm.weight"), ones(h));
            m.insert(p("post_attention_layernorm.weight"), ones(h));
            m.insert(p("self_attn.q_proj.weight"), randn(&[qd, h]));
            m.insert(p("self_attn.k_proj.weight"), randn(&[kvd, h]));
            m.insert(p("self_attn.v_proj.weight"), randn(&[kvd, h]));
            m.insert(p("self_attn.o_proj.weight"), randn(&[h, qd]));
            m.insert(p("mlp.gate_proj.weight"), randn(&[inter, h]));
            m.insert(p("mlp.up_proj.weight"), randn(&[inter, h]));
            m.insert(p("mlp.down_proj.weight"), randn(&[h, inter]));
        }
        CausalLm::from_weights(&Weights::from_map(m), "", cfg).unwrap()
    }

    /// The synthetic Qwen35 hybrid (vocab 50; three DeltaNet layers, one full-attention layer),
    /// with or without its MTP head.
    pub(crate) fn qwen35(mtp: bool) -> Qwen35Model {
        let cfg = Qwen35Config::from_json(&if mtp { cfg_json_mtp() } else { cfg_json() }).unwrap();
        Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg).unwrap()
    }

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            sampling: SamplingParams::default(),
            seed: Some(7),
            stop_tokens: Vec::new(),
        }
    }

    fn penalized(max_new_tokens: usize) -> GenerationConfig {
        let mut config = greedy(max_new_tokens);
        config.sampling.repetition_penalty = 1.3;
        config.sampling.repetition_context = 16;
        config.sampling.presence_penalty = 0.2;
        config
    }

    fn stochastic(max_new_tokens: usize) -> GenerationConfig {
        let mut config = greedy(max_new_tokens);
        config.sampling.temperature = 0.8;
        config.sampling.top_p = 0.9;
        config.sampling.top_k = 6;
        config
    }

    /// A deterministic constraint forbidding a fixed set of ids, with the checkpoint / rewind
    /// the engine needs; records every accepted token.
    struct Forbid {
        allow: Vec<bool>,
        accepted: Vec<i32>,
    }

    impl Forbid {
        fn new(vocab: usize, forbidden: &[usize]) -> Self {
            let mut allow = vec![true; vocab];
            for &f in forbidden {
                allow[f] = false;
            }
            Self {
                allow,
                accepted: Vec::new(),
            }
        }
    }

    impl ConstraintMask for Forbid {
        fn allowed(&mut self) -> &[bool] {
            &self.allow
        }
        fn accept(&mut self, token: i32) {
            self.accepted.push(token);
        }
    }

    impl RewindableConstraintMask for Forbid {
        fn checkpoint(&self) -> usize {
            self.accepted.len()
        }
        fn rewind(&mut self, checkpoint: usize) {
            self.accepted.truncate(checkpoint);
        }
    }

    /// The plain, non-engine token-at-a-time loop ([`generate_with`]) — the reference every
    /// engine route is held to.
    fn plain<D: crate::decode::Decode>(
        model: &D,
        prompt: &[i32],
        config: &GenerationConfig,
        constraint: Option<&mut Forbid>,
    ) -> GenerationOutput {
        generate_with(
            model,
            prompt,
            config,
            &CancelFlag::new(),
            &mut |_| {},
            constraint.map(|c| c as &mut dyn ConstraintMask),
            None,
        )
        .unwrap()
    }

    fn engine<T, P>(
        target: &T,
        proposer: &mut P,
        prompt: &[i32],
        config: &GenerationConfig,
        drafts: usize,
        constraint: Option<&mut Forbid>,
    ) -> SpeculativeRun
    where
        T: SpeculativeTarget,
        P: Proposer<T>,
    {
        let mut events = Vec::new();
        let run = generate_speculative(
            target,
            proposer,
            SpeculativePrompt::Tokens(prompt),
            config,
            drafts,
            &CancelFlag::new(),
            &mut |e| events.push(e),
            EngineOptions {
                constraint: constraint.map(|c| c as &mut dyn RewindableConstraintMask),
                ..EngineOptions::default()
            },
        )
        .unwrap();
        // The event stream is the output: one Token per generated id, in order, then Done.
        let ids: Vec<i32> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token { id, .. } => Some(*id),
                StreamEvent::Done { .. } => None,
            })
            .collect();
        assert_eq!(ids, run.output.tokens, "streamed ids == returned tokens");
        assert!(matches!(events.last(), Some(StreamEvent::Done { .. })));
        run
    }

    /// A prompt whose continuation repeats context, so prompt lookup proposes.
    const PROMPT: [i32; 12] = [3, 9, 4, 11, 3, 9, 4, 11, 3, 9, 4, 11];

    /// Every sampler configuration the engine must honour exactly, with the sampler path the
    /// report must name for it.
    fn configs() -> Vec<(&'static str, GenerationConfig, bool, &'static str)> {
        vec![
            ("greedy", greedy(20), false, "device"),
            ("penalized", penalized(20), false, "host:penalty"),
            ("constrained", greedy(20), true, "host:constraint"),
            (
                "stochastic",
                stochastic(20),
                false,
                "host:device_unavailable",
            ),
            (
                "constrained stochastic",
                stochastic(20),
                true,
                "host:constraint",
            ),
        ]
    }

    /// `off` — the engine with [`NoProposer`] — against the plain, non-engine loop on one target:
    /// identical tokens and finish reason for greedy, penalized, constrained and seeded stochastic
    /// sampling, and a report naming the token-at-a-time path.
    fn off_matches_plain<T: SpeculativeTarget + crate::decode::Decode>(target: &T, vocab: usize) {
        for (name, config, constrained, sampler) in configs() {
            let forbid = || Forbid::new(vocab, &[1, 9]);
            let (mut a, mut b) = (forbid(), forbid());
            let expected = plain(target, &PROMPT, &config, constrained.then_some(&mut a));
            let run = engine(
                target,
                &mut NoProposer,
                &PROMPT,
                &config,
                4,
                constrained.then_some(&mut b),
            );
            assert_eq!(
                run.output.tokens, expected.tokens,
                "{name}: off != plain loop"
            );
            assert_eq!(run.output.finish_reason, expected.finish_reason, "{name}");
            assert_eq!(run.output.tokens.len(), 20, "{name}: runs to the budget");
            if constrained {
                assert_eq!(
                    b.accepted, a.accepted,
                    "{name}: constraint advanced identically"
                );
                assert!(!run.output.tokens.iter().any(|t| *t == 1 || *t == 9));
            }
            let report = &run.report;
            assert_eq!(report.path, "step_model", "{name}");
            assert_eq!(report.proposer, ProposerKind::None, "{name}");
            assert_eq!(report.draft_tokens, None, "{name}: no proposer, no depth");
            assert_eq!(
                report.verify_steps, 19,
                "{name}: one verify per token after the first"
            );
            assert_eq!(report.target_forwards, 20, "{name}: prefill + one per step");
            assert_eq!((report.proposed_tokens, report.accepted_tokens), (0, 0));
            assert_eq!(report.mean_accepted_length(), None, "{name}");
            assert_eq!(report.sampler, sampler, "{name}");
        }
    }

    #[test]
    fn off_is_the_plain_loop_on_the_causal_and_hybrid_targets() {
        off_matches_plain(&causal(), 24);
        off_matches_plain(&qwen35(false), 50);
    }

    /// AC1: greedy prompt lookup emits exactly the plain loop's tokens — plain, penalized and
    /// constrained — on the Causal target (direct truncation) and the Qwen35 hybrid (snapshot
    /// restore + replay), while actually proposing, accepting and rejecting.
    #[test]
    fn greedy_prompt_lookup_is_the_plain_loop_and_recovers_both_ways() {
        let causal = causal();
        let hybrid = qwen35(false);
        for (name, config, constrained, _) in configs()
            .into_iter()
            .filter(|c| !c.0.contains("stochastic"))
        {
            let forbid = |vocab| Forbid::new(vocab, &[1, 9]);

            let (mut a, mut b) = (forbid(24), forbid(24));
            let expected = plain(&causal, &PROMPT, &config, constrained.then_some(&mut a));
            let run = engine(
                &causal,
                &mut NgramProposer::default(),
                &PROMPT,
                &config,
                4,
                constrained.then_some(&mut b),
            );
            assert_eq!(run.output.tokens, expected.tokens, "causal {name}");
            assert_eq!(b.accepted, a.accepted, "causal {name}: constraint");
            let s = run.stats;
            assert!(
                s.proposed > 0 && s.accepted > 0,
                "causal {name}: lookup ran: {s:?}"
            );
            assert_eq!(s.replays, 0, "causal {name}: truncation never replays");
            assert_eq!(run.report.proposer, ProposerKind::PromptLookup);

            let (mut a, mut b) = (forbid(50), forbid(50));
            let expected = plain(&hybrid, &PROMPT, &config, constrained.then_some(&mut a));
            let run = engine(
                &hybrid,
                &mut NgramProposer::default(),
                &PROMPT,
                &config,
                4,
                constrained.then_some(&mut b),
            );
            assert_eq!(run.output.tokens, expected.tokens, "hybrid {name}");
            assert_eq!(b.accepted, a.accepted, "hybrid {name}: constraint");
            let s = run.stats;
            // The penalized hybrid continuation never repeats its context, so it proposes nothing
            // (its exactness with drafts is the Causal run's above).
            if name != "penalized" {
                assert!(
                    s.proposed > 0 && s.accepted > 0,
                    "hybrid {name}: lookup ran: {s:?}"
                );
            }
            assert_eq!(
                s.direct_rollbacks, 0,
                "hybrid {name}: DeltaNet cannot truncate"
            );
            assert_eq!(run.report.replay_forwards, s.replays as u64);
        }
        // The plain greedy fixture exercises every recovery kind at least once.
        let run = engine(
            &causal,
            &mut NgramProposer::default(),
            &PROMPT,
            &greedy(24),
            4,
            None,
        );
        assert!(run.stats.direct_rollbacks > 0, "{:?}", run.stats);
        let run = engine(
            &hybrid,
            &mut NgramProposer::default(),
            &PROMPT,
            &greedy(24),
            4,
            None,
        );
        assert!(run.stats.replays > 0, "{:?}", run.stats);
    }

    /// AC1: greedy MTP emits exactly the plain loop's tokens (plain, penalized, constrained) on
    /// the Qwen35 MTP fixture, whose adversarial drafts force the snapshot-replay recovery.
    #[test]
    fn greedy_mtp_is_the_plain_loop() {
        let model = qwen35(true);
        for (name, config, constrained, _) in configs()
            .into_iter()
            .filter(|c| !c.0.contains("stochastic"))
        {
            let (mut a, mut b) = (Forbid::new(50, &[1, 9]), Forbid::new(50, &[1, 9]));
            let expected = plain(&model, &PROMPT, &config, constrained.then_some(&mut a));
            let run = engine(
                &model,
                &mut MtpProposer::new(),
                &PROMPT,
                &config,
                3,
                constrained.then_some(&mut b),
            );
            assert_eq!(run.output.tokens, expected.tokens, "{name}");
            assert_eq!(b.accepted, a.accepted, "{name}: constraint");
            assert!(
                run.stats.proposed > 0 && run.stats.replays > 0,
                "{name}: {:?}",
                run.stats
            );
            assert_eq!(run.report.path, "mtp");
            assert_eq!(run.report.proposer, ProposerKind::Mtp);
            assert_eq!(run.report.draft_tokens, Some(3));
        }
    }

    /// AC2 (engine half): the report names the proposer, its depth, the verify steps and accepted
    /// drafts behind the mean accepted length, the recovery replays and the measured sampler.
    #[test]
    fn the_report_carries_the_measured_speculation() {
        let run = engine(
            &causal(),
            &mut NgramProposer::default(),
            &PROMPT,
            &greedy(24),
            4,
            None,
        );
        let (r, s) = (&run.report, run.stats);
        assert_eq!(r.path, "prompt_lookup");
        assert_eq!(r.proposer, ProposerKind::PromptLookup);
        assert_eq!(r.draft_tokens, Some(4));
        assert_eq!(
            r.sampler, "device",
            "plain greedy draws are device argmaxes"
        );
        assert_eq!(
            (r.kv_cache.as_str(), r.attention.as_str()),
            ("growing", "gqa")
        );
        assert_eq!(r.target_forwards, s.forwards as u64);
        assert_eq!(r.proposed_tokens, s.proposed as u64);
        assert_eq!(r.accepted_tokens, s.accepted as u64);
        assert_eq!(r.verify_steps, s.verify_steps as u64);
        assert!(
            r.verify_steps < 23,
            "speculation took fewer verify steps than tokens"
        );
        assert_eq!(
            r.mean_accepted_length(),
            Some(s.accepted as f64 / s.verify_steps as f64)
        );
        assert!(!r.cuda_graphs.enabled);
        assert_eq!(r.cuda_graphs.path, "none");
        assert!(
            r.fallbacks.is_empty(),
            "fallbacks are the provider's to add"
        );
    }

    /// A stop token inside an accepted draft run ends the run exactly where the plain loop does,
    /// and nothing past it is emitted.
    #[test]
    fn a_stop_token_inside_a_draft_run_ends_like_the_plain_loop() {
        let model = causal();
        let base = plain(&model, &PROMPT, &greedy(24), None).tokens;
        let mut config = greedy(24);
        config.stop_tokens = vec![base[7]];
        let expected = plain(&model, &PROMPT, &config, None);
        let run = engine(
            &model,
            &mut NgramProposer::default(),
            &PROMPT,
            &config,
            4,
            None,
        );
        assert_eq!(run.output.tokens, expected.tokens);
        assert_eq!(run.output.finish_reason, FinishReason::StopToken);
    }

    /// Seeded stochastic speculation is reproducible and stays within its budget; acceptance is
    /// `core_llm`'s distribution-preserving rule (proven there).
    #[test]
    fn stochastic_speculation_is_seeded() {
        let model = causal();
        let a = engine(
            &model,
            &mut NgramProposer::default(),
            &PROMPT,
            &stochastic(20),
            4,
            None,
        );
        let b = engine(
            &model,
            &mut NgramProposer::default(),
            &PROMPT,
            &stochastic(20),
            4,
            None,
        );
        assert_eq!(a.output.tokens, b.output.tokens);
        assert_eq!(a.output.tokens.len(), 20);
        assert!(a.stats.accepted <= a.stats.proposed);
        assert!(
            a.report.sampler.starts_with("host:"),
            "{}",
            a.report.sampler
        );
    }

    /// The sampler is a seam: a caller-supplied [`TokenSampler`] draws every token.
    #[test]
    fn a_caller_sampler_draws_every_token() {
        struct Counting(MlxSampler, usize);
        impl TokenSampler for Counting {
            fn params(&self) -> &SamplingParams {
                self.0.params()
            }
            fn sample(&mut self, l: &Array, h: &[i32], m: Option<&[bool]>) -> Result<i32> {
                self.1 += 1;
                self.0.sample(l, h, m)
            }
            fn argmax_rows(&mut self, l: &Array) -> Result<Vec<i32>> {
                self.1 += 1;
                self.0.argmax_rows(l)
            }
            fn distribution(
                &mut self,
                l: &Array,
                h: &[i32],
                m: Option<&[bool]>,
            ) -> Result<Vec<(i32, f32)>> {
                self.0.distribution(l, h, m)
            }
            fn uniform(&mut self) -> f32 {
                self.0.uniform()
            }
            fn path(&self) -> Option<SamplerPath> {
                self.0.path()
            }
        }
        let model = causal();
        let config = greedy(16);
        let mut sampler = Counting(MlxSampler::from_config(&config), 0);
        let run = generate_speculative(
            &model,
            &mut NgramProposer::default(),
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            4,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                sampler: Some(&mut sampler),
                ..EngineOptions::default()
            },
        )
        .unwrap();
        assert_eq!(
            run.output.tokens,
            plain(&model, &PROMPT, &config, None).tokens
        );
        assert_eq!(
            sampler.1 as u64,
            run.report.verify_steps + 1,
            "one decision per step"
        );
    }
}
