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
//! [`Rollback::Direct`], armed per step through [`CacheRollback::begin`]'s `&mut` cache; a
//! device-side or pipelined sampler is a [`TokenSampler`] impl handed to [`EngineOptions::sampler`]
//! whose draws may stay device-resident ([`SampledToken::Device`]) — the loop feeds the next forward
//! from the array and reads the id back only for the stop check, history, constraint and event. Any
//! step-only [`Decode`] model runs the `off` loop through [`StepTarget`]. None of them touches this
//! loop.
//!
//! The verify-vs-decode kernel caveat of [`speculative`](super::speculative) applies: a multi-token
//! verify rounds a few bf16 ULP differently from the single-token step, so on real weights a greedy
//! speculative run tracks, rather than bit-matches, the token-at-a-time run where a near-tie flips.

use std::time::Instant;

use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

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
use crate::decode::{BufferRelease, Decode};
use crate::error::{Error, Result};
use crate::models::{CausalLm, Qwen35Cache, Qwen35Model};
use crate::primitives::input_ids;
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
use crate::primitives::sampler::{
    acceptance_target, argmax_rows_device, sample_with_path, SamplingParams, SplitMix64, TokenRng,
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
    /// Remember whatever a later recovery needs, before the verify forward writes the cache. The
    /// cache is lent mutably so a strategy can arm per-step state inside it (a DeltaNet checkpoint
    /// ring, story sc-24435).
    fn begin(&mut self, cache: &mut C);
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

    fn begin(&mut self, _: &mut C) {}

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

    fn begin(&mut self, cache: &mut C) {
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
    /// Run the `[1, n]` int32 token `ids` through the target, appending them to `cache`.
    /// `rope_offset` is the RoPE position of the first id (the cache length plus any multimodal
    /// continuation shift). The ids are an array, not a host slice, so a step's input can be a
    /// device-resident [`SampledToken`] that was never read back (the pipelining seam).
    fn forward(
        &self,
        cache: &mut Self::Cache,
        ids: &Array,
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
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        if want_hidden {
            return Err(Error::Msg(
                "CausalLm: the speculative target does not return hidden states".into(),
            ));
        }
        let logits = match scope {
            LogitsScope::Last => self.decode_logits(ids, cache, rope_offset)?,
            LogitsScope::All => self.decode_logits_all(ids, cache, rope_offset)?,
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
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        Ok(match (scope, want_hidden) {
            (LogitsScope::Last, false) => TargetOutput {
                logits: self.decode_logits(ids, cache, rope_offset)?,
                hidden: None,
            },
            (LogitsScope::All, false) => TargetOutput {
                logits: self.forward(ids, cache, rope_offset)?,
                hidden: None,
            },
            (LogitsScope::Last, true) => {
                let (hidden, logits) =
                    self.prefill_hidden_and_last_logits(ids, cache, rope_offset)?;
                TargetOutput {
                    logits,
                    hidden: Some(hidden),
                }
            }
            (LogitsScope::All, true) => {
                let (hidden, logits) = self.hidden_and_logits(ids, cache, rope_offset)?;
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

/// Any token-at-a-time [`Decode`] model as an engine target — the image-conditioned decoders
/// (StarVector's GPTBigCode / StarCoder2) whose prefill is caller-spliced embeddings and which only
/// expose a last-position step. It runs the engine's `off` loop ([`NoProposer`]); a verify forward
/// over drafts is refused, never approximated. Rollback is [`KvCache::truncate`].
pub struct StepTarget<'a>(pub &'a dyn Decode);

impl CacheRollback<Box<dyn KvCache>> for TruncateRollback {
    fn label(&self) -> &'static str {
        "truncate"
    }

    fn begin(&mut self, _: &mut Box<dyn KvCache>) {}

    fn recover(&mut self, cache: &mut Box<dyn KvCache>, keep: i32) -> Result<Rollback> {
        if keep < cache.offset() {
            cache.truncate(keep)?;
        }
        Ok(Rollback::Direct)
    }
}

impl SpeculativeTarget for StepTarget<'_> {
    type Cache = Box<dyn KvCache>;
    type Rollback = TruncateRollback;

    fn new_cache(&self) -> Box<dyn KvCache> {
        self.0.make_cache()
    }

    fn cache_len(&self, cache: &Box<dyn KvCache>) -> i32 {
        cache.offset()
    }

    fn rollback(&self) -> TruncateRollback {
        TruncateRollback
    }

    fn forward(
        &self,
        cache: &mut Box<dyn KvCache>,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        if want_hidden {
            return Err(Error::Msg(
                "step-only target does not return hidden states".into(),
            ));
        }
        match scope {
            LogitsScope::Last => Ok(TargetOutput {
                logits: self.0.step(ids, cache.as_mut(), rope_offset)?,
                hidden: None,
            }),
            LogitsScope::All => Err(Error::Msg("step-only target cannot verify drafts".into())),
        }
    }

    fn attention_label(&self) -> &'static str {
        // The engine sees only `Decode::step`; how the wrapped model attends is not visible here.
        "opaque"
    }
}

// ---------------------------------------------------------------------------------------------
// The sampler seam.
// ---------------------------------------------------------------------------------------------

/// One drawn token: already on the host, or still device-resident — a single-element integer
/// array the next forward can consume before it is ever read back. The engine resolves a device
/// token to the host only where it must (the stop-token check, the history / penalty window, the
/// constraint, the emitted event) and feeds the next step's forward from the array itself, so a
/// pipelined sampler (story sc-24439) can enqueue step `t + 1` before step `t`'s id is read.
#[derive(Clone, Debug)]
pub enum SampledToken {
    /// The id, on the host.
    Host(i32),
    /// A one-element integer array holding the id, on the device.
    Device(Array),
}

impl SampledToken {
    /// The id on the host (a device token is evaluated and read back).
    pub fn resolve(&self) -> Result<i32> {
        match self {
            SampledToken::Host(id) => Ok(*id),
            SampledToken::Device(id) => {
                Ok(id.reshape(&[-1])?.as_dtype(Dtype::Int32)?.item::<i32>())
            }
        }
    }

    /// The `[1, 1]` int32 input ids of the forward that consumes this token — built from the
    /// device array without reading it back.
    pub fn input(&self) -> Result<Array> {
        match self {
            SampledToken::Host(id) => Ok(input_ids(&[*id])),
            SampledToken::Device(id) => Ok(id.as_dtype(Dtype::Int32)?.reshape(&[1, 1])?),
        }
    }
}

/// The engine's token draws — the seam a device sampler plugs in behind. Every draw is recorded,
/// so [`path`](Self::path) is measured rather than inferred from the request.
pub trait TokenSampler {
    /// The sampling knobs.
    fn params(&self) -> &SamplingParams;
    /// Draw one token from a `[1, vocab]` (or `[vocab]`) logits row given the running `history`
    /// (the penalty window) and an optional constraint mask. A sampler may return the token
    /// device-resident ([`SampledToken::Device`]); the engine resolves it only where it must.
    fn sample(
        &mut self,
        logits: &Array,
        history: &[i32],
        allowed: Option<&[bool]>,
    ) -> Result<SampledToken>;
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

/// The MLX sampler: [`sample_with_path`] with the seeded [`SplitMix64`], plain-greedy draws taken
/// as the on-device argmax and everything else on the host (MLX has no device sampler for
/// temperature, penalties or a constraint mask). Every draw is recorded at the branch that ran, and
/// every token is returned resolved ([`SampledToken::Host`]).
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

    fn note(&mut self, path: SamplerPath) {
        match path {
            SamplerPath::Device => self.device_draws += 1,
            SamplerPath::Host(reason) => self.note_host(reason),
        }
    }
}

impl TokenSampler for MlxSampler {
    fn params(&self) -> &SamplingParams {
        &self.params
    }

    fn sample(
        &mut self,
        logits: &Array,
        history: &[i32],
        allowed: Option<&[bool]>,
    ) -> Result<SampledToken> {
        let (token, path) =
            sample_with_path(logits, history, &self.params, &mut self.rng, allowed)?;
        self.note(path);
        Ok(SampledToken::Host(token))
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
    /// Whether a draft advanced the constraint — the engine rewinds it after the proposal only
    /// then.
    advanced: bool,
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
        let draft = self
            .sampler
            .sample(logits, draft_history, mask)?
            .resolve()?;
        if !self.is_stop(draft) {
            if let Some(c) = self.constraint.as_mut() {
                c.accept(draft);
                self.advanced = true;
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
            let out = target.forward(cache, &input_ids(ids), 0, LogitsScope::Last, wants_hidden)?;
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

    // A cancel that tripped during the prefill ends the run before its first draw, exactly as the
    // plain loop's per-step check does.
    if cancel.is_cancelled() {
        finish = FinishReason::Cancelled;
        return Ok(finished(generated, finish, stats, sampler, timer, on_event));
    }

    // ---- First token: an ordinary draw from the prefill logits. ----
    let first = {
        let mask = constraint.as_mut().map(|c| c.allowed());
        sampler.sample(&logits, &history, mask)?
    };
    // The next forward consumes the draw as drawn (device-resident or not); the host id serves
    // the stop check, the history, the constraint and the event.
    let mut cur_input = first.input()?;
    let first = first.resolve()?;
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

        // 1. Propose. The constraint is rewound only when a sampled draft advanced it.
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
                advanced: false,
            };
            let proposal = proposer.propose(target, &ctx, &mut draft_sampler)?;
            if draft_sampler.advanced {
                if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), checkpoint) {
                    c.rewind(checkpoint);
                }
            }
            proposal
        };
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

        // 2. Verify `[cur, drafts…]` in one target forward. A step without drafts feeds `cur` as
        // it was drawn, so a device-resident token is never read back to build the input.
        let mut verify = Vec::with_capacity(1 + num_drafts);
        verify.push(cur);
        verify.extend_from_slice(&draft_ids);
        let (scope, input) = if num_drafts == 0 {
            (LogitsScope::Last, cur_input)
        } else {
            rollback.begin(cache);
            (LogitsScope::All, input_ids(&verify))
        };
        let out = target.forward(cache, &input, position, scope, wants_hidden)?;
        drop(input);
        if let Some(hidden) = out.hidden.as_ref() {
            eval([hidden, &out.logits])?;
        }
        stats.forwards += 1;
        stats.verify_steps += 1;

        // 3. Decide. Only a decision over drafts advances the constraint provisionally.
        let Decision {
            committed,
            accepted,
            last,
        } = decide(
            &out.logits,
            &draft_ids,
            &dists,
            &history,
            config,
            sampler,
            constraint.as_deref_mut(),
        )?;
        if num_drafts > 0 {
            if let (Some(c), Some(checkpoint)) = (constraint.as_mut(), checkpoint) {
                c.rewind(checkpoint);
            }
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
                        &input_ids(&verify[..keep_len]),
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
        // The whole run committed: the next step feeds its last token as drawn.
        cur_input = match last {
            Some(token) => token.input()?,
            None => input_ids(&[cur]),
        };
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
        fallbacks: Vec::new(),
    }
}

/// A verify decision: the committed run, the accepted draft count and, when the run's last token
/// came from [`TokenSampler::sample`], that draw as drawn (possibly device-resident) so the next
/// forward can consume it without a read-back.
struct Decision {
    committed: Vec<i32>,
    accepted: usize,
    last: Option<SampledToken>,
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
) -> Result<Decision> {
    if drafts.is_empty() {
        let mask = constraint.as_mut().map(|c| c.allowed());
        let token = sampler.sample(logits, history, mask)?;
        return Ok(Decision {
            committed: vec![token.resolve()?],
            accepted: 0,
            last: Some(token),
        });
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
        let (committed, accepted) = greedy_commit(&target_argmax, drafts);
        return Ok(Decision {
            committed,
            accepted,
            last: None,
        });
    }

    let mut committed = Vec::with_capacity(drafts.len() + 1);
    let mut accepted = 0usize;
    let mut running = history.to_vec();
    for (i, &draft) in drafts.iter().enumerate() {
        let row = logits_row(logits, i as i32)?;
        let mask = constraint.as_mut().map(|c| c.allowed());
        let outcome = if greedy {
            let target = sampler.sample(&row, &running, mask)?.resolve()?;
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
        let stop = config.stop_tokens.contains(&token);
        if !stop {
            if let Some(c) = constraint.as_mut() {
                c.accept(token);
            }
            running.push(token);
        }
        if stop || !outcome.is_accepted() {
            return Ok(Decision {
                committed,
                accepted,
                last: None,
            });
        }
    }
    // Every draft accepted: the bonus from the position past the last draft.
    let row = logits_row(logits, drafts.len() as i32)?;
    let mask = constraint.as_mut().map(|c| c.allowed());
    let bonus = sampler.sample(&row, &running, mask)?;
    committed.push(bonus.resolve()?);
    Ok(Decision {
        committed,
        accepted,
        last: Some(bonus),
    })
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
    use crate::decode::proposers::{DraftModelProposer, MtpProposer, NgramProposer};
    use crate::decode::stream::generate_with;
    use crate::models::qwen35::tests::{cfg_json, cfg_json_mtp, synthetic_weights};
    use crate::models::Qwen35Config;
    use crate::primitives::Weights;

    // ---- Fixtures: a tiny random llama-family decoder and the synthetic Qwen35 hybrid. ----

    /// A tiny random llama-family decoder (vocab 24) whose greedy continuation of [`PROMPT`]
    /// repeats its context, so prompt lookup proposes, accepts and rejects.
    pub(crate) fn causal() -> CausalLm {
        tiny_llama(24)
    }

    /// The tiny random llama-family decoder over a `vocab`-entry vocabulary (hidden 16, two
    /// layers), drawn from one fixed seed.
    pub(crate) fn tiny_llama(vocab: i32) -> CausalLm {
        causal_model(vocab, 2)
    }

    /// The tiny llama-family decoder over `vocab` ids with its first `layers` layers. The weights
    /// are drawn in a fixed order (embedding, output projection, then layer by layer), so a
    /// shallower model is the deeper one with its later layers skipped — a draft that shares the
    /// target's embedding, output projection and first layer (sc-24436).
    pub(crate) fn causal_model(vocab: i32, layers: usize) -> CausalLm {
        let cfg = ModelConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_layers: layers,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: 4,
            vocab_size: vocab,
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
    /// the engine needs; records every accepted token and counts the engine's rewinds — all of
    /// them, and those that actually moved the state back.
    struct Forbid {
        allow: Vec<bool>,
        accepted: Vec<i32>,
        rewinds: usize,
        state_changing_rewinds: usize,
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
                rewinds: 0,
                state_changing_rewinds: 0,
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
            self.rewinds += 1;
            if checkpoint == self.accepted.len() {
                return;
            }
            self.state_changing_rewinds += 1;
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

    /// The absolute accounting of a budget-bound run with no stop tokens: the first token comes
    /// from the prefill, every verify step commits its accepted drafts plus exactly one bonus or
    /// correction token, and every forward is the prefill, a verify step or a recovery replay.
    fn assert_accounting(label: &str, run: &SpeculativeRun, config: &GenerationConfig) {
        assert!(
            config.stop_tokens.is_empty(),
            "{label}: accounting needs no stop tokens"
        );
        assert_eq!(run.output.finish_reason, FinishReason::MaxTokens, "{label}");
        let r = &run.report;
        assert_eq!(
            run.output.tokens.len() as u64,
            1 + r.verify_steps + r.accepted_tokens,
            "{label}: tokens = first + one per verify step + accepted drafts: {r:?}"
        );
        assert_eq!(
            r.target_forwards,
            1 + r.verify_steps + r.replay_forwards,
            "{label}: forwards = prefill + verify steps + replays: {r:?}"
        );
    }

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
            assert_accounting(&format!("causal {name}"), &run, &config);
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
            assert_accounting(&format!("hybrid {name}"), &run, &config);
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
            assert_accounting(&format!("mtp {name}"), &run, &config);
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
        assert_accounting("causal lookup", &run, &greedy(24));
        assert!(run.report.accepted_tokens > 0, "{:?}", run.report);
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
        assert_accounting("stochastic lookup", &a, &stochastic(20));
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
            fn sample(&mut self, l: &Array, h: &[i32], m: Option<&[bool]>) -> Result<SampledToken> {
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

    // ---- Draft-model speculation on the engine (epic sc-24432, story sc-24436). ----

    /// Every greedy configuration through [`DraftModelProposer`] over `draft` against the plain
    /// loop on `target`: identical tokens and constraint advance, a report naming `draft_model`.
    /// Returns each configuration's counters.
    fn draft_matches_plain<T, D>(
        label: &str,
        target: &T,
        draft: &D,
        vocab: usize,
    ) -> Vec<SpeculativeStats>
    where
        T: SpeculativeTarget + crate::decode::Decode,
        D: SpeculativeTarget,
    {
        let mut all = Vec::new();
        for (name, config, constrained, _) in configs()
            .into_iter()
            .filter(|c| !c.0.contains("stochastic"))
        {
            let (mut a, mut b) = (Forbid::new(vocab, &[1, 9]), Forbid::new(vocab, &[1, 9]));
            let expected = plain(target, &PROMPT, &config, constrained.then_some(&mut a));
            let mut proposer = DraftModelProposer::new(draft);
            let run = engine(
                target,
                &mut proposer,
                &PROMPT,
                &config,
                4,
                constrained.then_some(&mut b),
            );
            assert_eq!(run.output.tokens, expected.tokens, "{label} {name}");
            assert_eq!(b.accepted, a.accepted, "{label} {name}: constraint");
            assert_eq!(run.report.path, "draft_model", "{label} {name}");
            assert_eq!(
                run.report.proposer,
                ProposerKind::DraftModel,
                "{label} {name}"
            );
            assert_eq!(run.report.draft_tokens, Some(4), "{label} {name}");
            assert!(run.stats.proposed > 0, "{label} {name}: the draft proposed");
            assert!(proposer.draft_forwards > 0, "{label} {name}");
            all.push(run.stats);
        }
        all
    }

    /// sc-24436 AC1 (engine half): a draft model proposes through the one engine loop and every
    /// greedy configuration emits exactly the plain loop's tokens, whichever way each cache rolls
    /// back — a truncating target with a truncating layer-skip draft (accepted and rejected
    /// drafts), the hybrid target (snapshot + replay) with a softmax draft, a softmax target with
    /// the hybrid as its draft (the draft restores its own step start), and the hybrid drafting
    /// for an identical hybrid (every draft accepted).
    #[test]
    fn greedy_draft_model_is_the_plain_loop_on_every_cache_pairing() {
        let stats = draft_matches_plain("causal/causal", &causal(), &causal_model(24, 1), 24);
        let (accepted, proposed) = stats
            .iter()
            .fold((0, 0), |(a, p), s| (a + s.accepted, p + s.proposed));
        assert!(
            accepted > 0 && accepted < proposed,
            "the layer-skip draft is accepted and rejected: {accepted} of {proposed}"
        );
        assert!(stats.iter().any(|s| s.direct_rollbacks > 0), "{stats:?}");

        let stats = draft_matches_plain("hybrid/causal", &qwen35(false), &causal_model(50, 1), 50);
        assert!(stats.iter().all(|s| s.direct_rollbacks == 0), "{stats:?}");

        draft_matches_plain("causal/hybrid", &causal_model(50, 2), &qwen35(false), 50);

        let stats = draft_matches_plain("hybrid/hybrid", &qwen35(false), &qwen35(false), 50);
        assert!(
            stats.iter().all(|s| s.accepted == s.proposed),
            "a draft identical to its target is always accepted: {stats:?}"
        );
    }

    /// sc-24436 E1: a stochastic draft hands the engine the proposal distribution `q` each draft
    /// was drawn from — one per draft, each containing its draft — so the exact rejection rule
    /// compares the target's `p` against the draft's own `q`; seeded runs reproduce.
    #[test]
    fn a_stochastic_draft_model_reports_one_q_per_draft_and_is_seeded() {
        let target = causal();
        let draft = causal_model(24, 1);
        let config = stochastic(20);
        let mut proposer = DraftModelProposer::new(&draft);
        Proposer::<CausalLm>::warm(&mut proposer, &target, &PROMPT, None).unwrap();
        let mut history = PROMPT.to_vec();
        history.push(5);
        let mut sampler = MlxSampler::from_config(&config);
        let mut draft_sampler = DraftSampler {
            config: &config,
            sampler: &mut sampler,
            constraint: None,
            advanced: false,
        };
        let ctx = ProposeContext {
            cur: 5,
            history: &history,
            previous_hidden: None,
            position: PROMPT.len() as i32,
            max_drafts: 4,
        };
        let proposal = proposer.propose(&target, &ctx, &mut draft_sampler).unwrap();
        assert_eq!(proposal.drafts.len(), 4);
        assert_eq!(
            proposal.dists.len(),
            proposal.drafts.len(),
            "one q per draft"
        );
        for (i, (token, q)) in proposal.drafts.iter().zip(&proposal.dists).enumerate() {
            assert!(
                q.iter().any(|&(t, w)| t == *token && w > 0.0),
                "{token} in {q:?}"
            );
            // Each `q` is the draft model's own shaped distribution at its position: the
            // sampler's distribution over a fresh draft prefill of everything before the draft.
            let mut before = history.clone();
            before.extend_from_slice(&proposal.drafts[..i]);
            let expected = MlxSampler::from_config(&config)
                .distribution(&last_logits(&draft, &before), &before, None)
                .unwrap();
            assert_distribution_eq(q, &expected, &format!("q of draft {i}"));
        }
        assert!(
            proposal.dists.iter().any(|q| q.len() > 1),
            "a stochastic draft is not a point mass: {:?}",
            proposal.dists
        );
        // The prompt, `cur` and all but the last draft are in the draft cache.
        assert_eq!(proposer.draft_forwards, 1 + 4);

        let run = |seed| {
            let mut config = stochastic(20);
            config.seed = Some(seed);
            engine(
                &target,
                &mut DraftModelProposer::new(&draft),
                &PROMPT,
                &config,
                3,
                None,
            )
        };
        let (a, b) = (run(11), run(11));
        assert_eq!(a.output.tokens, b.output.tokens);
        assert_eq!(a.output.tokens.len(), 20);
        assert_eq!(a.report.proposer, ProposerKind::DraftModel);
        assert!(a.stats.proposed > 0 && a.stats.accepted <= a.stats.proposed);
    }

    /// The last-position logits of a fresh prefill of `ids` through `model`, `[1, vocab]`.
    fn last_logits<M: SpeculativeTarget>(model: &M, ids: &[i32]) -> Array {
        let mut cache = model.new_cache();
        model
            .forward(&mut cache, &input_ids(ids), 0, LogitsScope::Last, false)
            .unwrap()
            .logits
    }

    fn host_row(logits: &Array) -> Vec<f32> {
        logits
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[-1])
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    }

    /// Two `(token, weight)` sets name the same tokens with weights equal to within `1e-4`
    /// (after normalizing each: `accept_token` normalizes, so only the shape matters).
    fn assert_distribution_eq(got: &[(i32, f32)], want: &[(i32, f32)], label: &str) {
        let norm = |d: &[(i32, f32)]| {
            let total: f32 = d.iter().map(|&(_, w)| w).sum();
            let mut d: Vec<(i32, f32)> = d.iter().map(|&(t, w)| (t, w / total)).collect();
            d.sort_by_key(|&(t, _)| t);
            d
        };
        let (got, want) = (norm(got), norm(want));
        assert_eq!(
            got.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            want.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            "{label}: tokens {got:?} vs {want:?}"
        );
        for (&(t, a), &(_, b)) in got.iter().zip(&want) {
            assert!((a - b).abs() < 1e-4, "{label}: token {t}: {a} vs {b}");
        }
    }

    /// A [`DraftModelProposer`] that, after every commit, holds the draft's state against the
    /// committed sequence: the draft cache plus the queued committed tokens are exactly the
    /// committed history, the queue is that history's tail, and the draft's logits for the next
    /// position equal a fresh draft prefill of the history — whatever rollback strategy the
    /// draft's cache recovers through. Counts the commits that kept some but not all of the
    /// drafts the proposal fed (a partial acceptance past the step start) and those that kept
    /// every fed draft.
    struct CheckedDraft<'d, D: SpeculativeTarget> {
        inner: DraftModelProposer<'d, D>,
        draft: &'d D,
        history: Vec<i32>,
        partial: usize,
        full: usize,
    }

    impl<'d, D: SpeculativeTarget> CheckedDraft<'d, D>
    where
        D::Cache: Clone,
    {
        fn new(draft: &'d D) -> Self {
            Self {
                inner: DraftModelProposer::new(draft),
                draft,
                history: Vec::new(),
                partial: 0,
                full: 0,
            }
        }

        fn check(&self) {
            let (cache, pending, _) = self.inner.state();
            let cache = cache.expect("warmed");
            let held = self.draft.cache_len(cache) as usize;
            assert_eq!(
                held + pending.len(),
                self.history.len(),
                "draft cache {held} + queued {pending:?} != committed {}",
                self.history.len()
            );
            assert_eq!(
                pending,
                &self.history[held..],
                "the queue is the history's tail"
            );
            let probe = 5;
            let mut tail = pending.to_vec();
            tail.push(probe);
            let mut resumed = cache.clone();
            let got = self
                .draft
                .forward(
                    &mut resumed,
                    &input_ids(&tail),
                    held as i32,
                    LogitsScope::Last,
                    false,
                )
                .unwrap()
                .logits;
            let mut all = self.history.clone();
            all.push(probe);
            let (got, want) = (host_row(&got), host_row(&last_logits(self.draft, &all)));
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-3,
                "draft logits after commit drift from a fresh prefill by {worst}"
            );
        }
    }

    impl<T, D> Proposer<T> for CheckedDraft<'_, D>
    where
        T: SpeculativeTarget,
        D: SpeculativeTarget,
        D::Cache: Clone,
    {
        fn kind(&self) -> ProposerKind {
            ProposerKind::DraftModel
        }
        fn warm(&mut self, t: &T, prompt: &[i32], h: Option<&Array>) -> Result<Option<Array>> {
            self.history = prompt.to_vec();
            self.inner.warm(t, prompt, h)
        }
        fn propose(
            &mut self,
            t: &T,
            ctx: &ProposeContext<'_>,
            sampler: &mut DraftSampler<'_, '_>,
        ) -> Result<Proposal> {
            self.inner.propose(t, ctx, sampler)
        }
        fn commit(
            &mut self,
            t: &T,
            cur: i32,
            accepted: &[i32],
            h: Option<&Array>,
            position: i32,
        ) -> Result<()> {
            let (_, _, fed) = self.inner.state();
            match fed {
                Some(fed) if fed > 0 && accepted.len() >= fed => self.full += 1,
                Some(fed) if !accepted.is_empty() && accepted.len() < fed => self.partial += 1,
                _ => {}
            }
            self.inner.commit(t, cur, accepted, h, position)?;
            self.history.push(cur);
            self.history.extend_from_slice(accepted);
            self.check();
            Ok(())
        }
    }

    /// sc-24436: after every commit the draft's cache is the committed sequence — its length
    /// plus the queued tokens is the history, and its next logits are a fresh prefill's — for a
    /// draft that restores its step start (the Qwen35 hybrid) and one that truncates, under
    /// greedy, penalized and constrained targets whose verify accepts some fed drafts and
    /// rejects the rest, and ones that accept every fed draft. Generic over the draft's own
    /// rollback strategy, so a strategy that keeps K − 1 single-token draft forwards between one
    /// `begin` and `recover` (the DeltaNet checkpoint ring) is held to the same invariant.
    #[test]
    fn a_draft_cache_is_the_committed_sequence_after_every_commit() {
        fn run_checked<T, D>(label: &str, target: &T, draft: &D, vocab: usize) -> (usize, usize)
        where
            T: SpeculativeTarget + crate::decode::Decode,
            D: SpeculativeTarget,
            D::Cache: Clone,
        {
            let (mut partial, mut full) = (0, 0);
            for (name, config, constrained, _) in configs()
                .into_iter()
                .filter(|c| !c.0.contains("stochastic"))
            {
                let (mut a, mut b) = (Forbid::new(vocab, &[1, 9]), Forbid::new(vocab, &[1, 9]));
                let expected = plain(target, &PROMPT, &config, constrained.then_some(&mut a));
                let mut proposer = CheckedDraft::new(draft);
                let run = engine(
                    target,
                    &mut proposer,
                    &PROMPT,
                    &config,
                    4,
                    constrained.then_some(&mut b),
                );
                assert_eq!(run.output.tokens, expected.tokens, "{label} {name}");
                partial += proposer.partial;
                full += proposer.full;
            }
            (partial, full)
        }
        // The hybrid draft (snapshot restore) for a softmax target that disagrees with it, and
        // for an identical hybrid target that keeps every fed draft.
        let (partial, _) = run_checked("causal/hybrid", &causal_model(50, 2), &qwen35(false), 50);
        let (_, full) = run_checked("hybrid/hybrid", &qwen35(false), &qwen35(false), 50);
        assert!(
            partial > 0 && full > 0,
            "hybrid draft: {partial} partial and {full} full keeps"
        );
        // A truncating draft (layer skip) for its deeper target.
        let (partial, full) = run_checked("causal/causal", &causal(), &causal_model(24, 1), 24);
        assert!(
            partial > 0 && full > 0,
            "causal draft: {partial} partial and {full} full keeps"
        );
    }

    /// sc-24436: a stop token the draft proposes and the target accepts ends the run exactly as
    /// the plain loop does, on the host draft path (a penalty, a near-zero temperature) and the
    /// greedy one, for a truncating and a snapshot-restoring draft — and the draft's cache is
    /// still the committed sequence after that commit (the proposal stopped drafting at the
    /// stop without feeding it).
    #[test]
    fn a_draft_proposed_stop_token_accepted_by_the_target_ends_the_run() {
        fn check<T>(label: &str, target: &T)
        where
            T: SpeculativeTarget + crate::decode::Decode,
            T::Cache: Clone,
        {
            let mut cold = greedy(24);
            cold.sampling.temperature = 0.01;
            let mut light = greedy(24);
            light.sampling.presence_penalty = 0.01;
            for (name, mut config) in [("greedy", greedy(24)), ("penalized", light), ("cold", cold)]
            {
                // The draft is the target itself, so every draft — the stop included — is
                // accepted. The stream the stop is cut from: the plain loop's for a greedy
                // decision, the same seeded draft run's for a sampled one (speculation consumes
                // the seeded stream differently from the plain loop).
                // The draft's state is checked against the committed sequence after every commit,
                // the one that accepts the stop included.
                let draft_run = |config: &GenerationConfig| {
                    engine(
                        target,
                        &mut CheckedDraft::new(target),
                        &PROMPT,
                        config,
                        4,
                        None,
                    )
                };
                let stream = if name == "cold" {
                    draft_run(&config).output.tokens
                } else {
                    plain(target, &PROMPT, &config, None).tokens
                };
                // A stop first appearing a few tokens in, so drafts carry it; it is not emitted.
                let end = (4..stream.len())
                    .find(|&i| !stream[..i].contains(&stream[i]))
                    .unwrap();
                config.stop_tokens = vec![stream[end]];
                let run = draft_run(&config);
                assert_eq!(run.output.tokens, stream[..end], "{label} {name}");
                assert_eq!(
                    run.output.finish_reason,
                    FinishReason::StopToken,
                    "{label} {name}"
                );
                assert_eq!(run.stats.accepted, run.stats.proposed, "{label} {name}");
            }
        }
        check("causal", &causal());
        check("hybrid", &qwen35(false));
    }

    /// sc-24436: a draft whose padding differs from its target's proposes only its tokenizer's
    /// ids, over logits the target's width, and greedy output is still the target's own.
    #[test]
    fn a_draft_padded_differently_proposes_only_tokenizer_ids() {
        let draft = causal_model(24, 1);
        let proposer = DraftModelProposer::new(&draft).with_vocab(20, 28);
        let mut row = vec![0f32; 24];
        row[22] = 10.0;
        row[3] = 1.0;
        let shaped = host_row(&proposer.shaped(Array::from_slice(&row, &[1, 24])).unwrap());
        assert_eq!(shaped.len(), 28);
        assert_eq!(&shaped[..20], &row[..20]);
        assert!(shaped[20..].iter().all(|x| *x == f32::NEG_INFINITY));
        // Same width, proposable = the whole width: untouched.
        let untouched = DraftModelProposer::new(&draft).with_vocab(24, 24);
        assert_eq!(
            host_row(&untouched.shaped(Array::from_slice(&row, &[1, 24])).unwrap()),
            row
        );
        // Through the engine: the target scores 24 ids, the draft proposes from its first 20.
        let target = causal();
        for config in [greedy(20), penalized(20)] {
            let expected = plain(&target, &PROMPT, &config, None);
            let mut proposer = DraftModelProposer::new(&draft).with_vocab(20, 24);
            let run = engine(&target, &mut proposer, &PROMPT, &config, 4, None);
            assert_eq!(run.output.tokens, expected.tokens);
            assert!(run.stats.proposed > 0);
        }
    }

    /// The constraint is rewound only when something advanced it: never on the `off` loop, and on
    /// a speculative run only after a proposal that sampled drafts (MTP) and after a decision over
    /// drafts — so every rewind the engine issues moves the state back. (A `JsonMask` rewind
    /// rebuilds and replays the grammar, so a redundant one per step made every constrained MLX
    /// request quadratic.)
    #[test]
    fn the_constraint_is_rewound_only_when_something_advanced_it() {
        let causal = causal();
        let mut forbid = Forbid::new(24, &[1, 9]);
        let run = engine(
            &causal,
            &mut NoProposer,
            &PROMPT,
            &greedy(20),
            4,
            Some(&mut forbid),
        );
        assert_eq!(run.output.tokens.len(), 20);
        assert_eq!(
            (forbid.rewinds, forbid.state_changing_rewinds),
            (0, 0),
            "off never rewinds"
        );

        // Prompt lookup never samples its drafts: only the decision over drafts rewinds.
        let mut forbid = Forbid::new(24, &[1, 9]);
        let run = engine(
            &causal,
            &mut NgramProposer::default(),
            &PROMPT,
            &greedy(20),
            4,
            Some(&mut forbid),
        );
        assert!(run.stats.proposed > 0, "{:?}", run.stats);
        assert!(forbid.rewinds > 0, "lookup rewinds its decisions");
        assert!(
            (forbid.rewinds as u64) < run.report.verify_steps,
            "no rewind on a step without drafts: {} rewinds over {} steps",
            forbid.rewinds,
            run.report.verify_steps
        );
        assert_eq!(
            forbid.rewinds, forbid.state_changing_rewinds,
            "lookup: every rewind moves the state"
        );

        // MTP samples its drafts: the proposal and the decision each rewind what they advanced.
        let mut forbid = Forbid::new(50, &[1, 9]);
        let run = engine(
            &qwen35(true),
            &mut MtpProposer::new(),
            &PROMPT,
            &greedy(20),
            3,
            Some(&mut forbid),
        );
        assert!(run.stats.proposed > 0, "{:?}", run.stats);
        assert!(forbid.rewinds > 0);
        assert_eq!(
            forbid.rewinds, forbid.state_changing_rewinds,
            "mtp: every rewind moves the state"
        );
    }

    /// A cancel that trips during the prefill (here: while the proposer warms) ends the run before
    /// its first draw — zero tokens, `Cancelled` — as the plain loop's per-step check does.
    #[test]
    fn a_cancel_during_prefill_emits_no_token() {
        struct CancelOnWarm(CancelFlag);
        impl<T: SpeculativeTarget + ?Sized> Proposer<T> for CancelOnWarm {
            fn kind(&self) -> ProposerKind {
                ProposerKind::None
            }
            fn warm(&mut self, _: &T, _: &[i32], _: Option<&Array>) -> Result<Option<Array>> {
                self.0.cancel();
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
            fn commit(
                &mut self,
                _: &T,
                _: i32,
                _: &[i32],
                _: Option<&Array>,
                _: i32,
            ) -> Result<()> {
                Ok(())
            }
        }
        let cancel = CancelFlag::new();
        let mut events = Vec::new();
        let run = generate_speculative(
            &causal(),
            &mut CancelOnWarm(cancel.clone()),
            SpeculativePrompt::Tokens(&PROMPT),
            &greedy(8),
            0,
            &cancel,
            &mut |e| events.push(e),
            EngineOptions::default(),
        )
        .unwrap();
        assert!(run.output.tokens.is_empty(), "{:?}", run.output.tokens);
        assert_eq!(run.output.finish_reason, FinishReason::Cancelled);
        assert_eq!(
            events,
            vec![StreamEvent::Done {
                reason: FinishReason::Cancelled,
                generated: 0
            }]
        );
    }

    /// The sampler seam carries device-resident tokens: a sampler returning every plain-greedy draw
    /// as an unevaluated on-device argmax drives the `off` loop and prompt lookup to exactly the
    /// plain loop's tokens (the next forward consumes the array; the host id serves the rest).
    #[test]
    fn a_device_resident_sampler_drives_the_loop() {
        struct DeviceGreedy(MlxSampler, usize);
        impl TokenSampler for DeviceGreedy {
            fn params(&self) -> &SamplingParams {
                self.0.params()
            }
            fn sample(&mut self, l: &Array, h: &[i32], m: Option<&[bool]>) -> Result<SampledToken> {
                if m.is_some() || !self.0.params().is_plain_greedy() {
                    return self.0.sample(l, h, m);
                }
                self.1 += 1;
                let id = mlx_rs::ops::indexing::argmax(&l.reshape(&[-1])?, None)?;
                Ok(SampledToken::Device(id))
            }
            fn argmax_rows(&mut self, l: &Array) -> Result<Vec<i32>> {
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
                Some(SamplerPath::Device)
            }
        }
        let model = causal();
        let config = greedy(16);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        for (label, drafts) in [("off", 0usize), ("prompt lookup", 4)] {
            let mut sampler = DeviceGreedy(MlxSampler::from_config(&config), 0);
            let mut proposer = NgramProposer::default();
            let run = generate_speculative(
                &model,
                &mut proposer,
                SpeculativePrompt::Tokens(&PROMPT),
                &config,
                drafts,
                &CancelFlag::new(),
                &mut |_| {},
                EngineOptions {
                    sampler: Some(&mut sampler),
                    ..EngineOptions::default()
                },
            )
            .unwrap();
            assert_eq!(run.output.tokens, expected, "{label}");
            assert!(sampler.1 > 0, "{label}: device tokens were drawn");
            assert_eq!(run.report.sampler, "device", "{label}");
        }
    }

    /// Hand-built `[1, n, vocab]` logits, one row per slice.
    fn rows(rows: &[&[f32]]) -> Array {
        let vocab = rows[0].len() as i32;
        let flat: Vec<f32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Array::from_slice(&flat, &[1, rows.len() as i32, vocab])
    }

    /// The decision's bonus row sees the accepted drafts through the constraint: a stateful
    /// constraint forbidding the last accepted token keeps the bonus off draft 0 even though draft
    /// 0 is row 1's argmax.
    #[test]
    fn the_bonus_row_sees_accepted_drafts_through_the_constraint() {
        struct ForbidLast(Vec<bool>, Vec<i32>);
        impl ConstraintMask for ForbidLast {
            fn allowed(&mut self) -> &[bool] {
                self.0.iter_mut().for_each(|a| *a = true);
                if let Some(&last) = self.1.last() {
                    self.0[last as usize] = false;
                }
                &self.0
            }
            fn accept(&mut self, token: i32) {
                self.1.push(token);
            }
        }
        impl RewindableConstraintMask for ForbidLast {
            fn checkpoint(&self) -> usize {
                self.1.len()
            }
            fn rewind(&mut self, checkpoint: usize) {
                self.1.truncate(checkpoint);
            }
        }
        let logits = rows(&[&[0.0, 0.0, 5.0, 0.0, 0.0], &[0.0, 0.0, 5.0, 4.0, 0.0]]);
        let config = greedy(4);
        let mut sampler = MlxSampler::from_config(&config);
        let mut constraint = ForbidLast(vec![true; 5], Vec::new());
        let decision = decide(
            &logits,
            &[2],
            &[],
            &[0, 1],
            &config,
            &mut sampler,
            Some(&mut constraint),
        )
        .unwrap();
        assert_eq!(decision.accepted, 1);
        assert_eq!(decision.committed, vec![2, 3], "the bonus avoids draft 0");
    }

    /// The decision's bonus row sees the accepted drafts through the penalty window: a presence
    /// penalty on draft 0 — absent from the history — flips row 1's argmax only when the running
    /// history carries the accepted draft.
    #[test]
    fn the_bonus_row_sees_accepted_drafts_through_the_penalty() {
        let logits = rows(&[&[0.0, 0.0, 5.0, 0.0, 0.0], &[0.0, 0.0, 5.0, 4.5, 0.0]]);
        let mut config = greedy(4);
        config.sampling.presence_penalty = 1.0;
        let mut sampler = MlxSampler::from_config(&config);
        let decision = decide(&logits, &[2], &[], &[0, 1], &config, &mut sampler, None).unwrap();
        assert_eq!(decision.accepted, 1);
        assert_eq!(
            decision.committed,
            vec![2, 3],
            "the penalty sees the accepted draft"
        );
    }

    /// E1 through the engine's own `decide`: over many seeds, the first committed token of a
    /// stochastic verify step is distributed as the target's shaped distribution `p`, whether the
    /// draft came with a point mass (n-gram) or the non-degenerate `q` it was drawn from (a draft
    /// model / MTP head) — a chi-square test (ported from Candle's engine).
    ///
    /// The rejection rule preserves `p` under *either* proposal distribution (a point mass is
    /// exact conditionally on the draft), so the chi-square alone cannot tell whether `decide`
    /// used the draft's real `q`. The acceptance rate can: with `q` it is `Σ min(p, q)`, with a
    /// point mass `Σ q·p` — both pinned here, so dropping the reported `q` turns this red.
    #[test]
    fn stochastic_decision_preserves_the_target_distribution_chi_square() {
        let vocab = 5usize;
        let p_logits = [1.0f32, 2.2, 0.3, 1.7, -0.5];
        let config = GenerationConfig {
            max_new_tokens: 2,
            sampling: SamplingParams {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                ..SamplingParams::default()
            },
            seed: Some(0),
            stop_tokens: Vec::new(),
        };
        // Rows: position 0 (the draft's) and the bonus row (uniform, irrelevant to token 0).
        let logits = rows(&[&p_logits, &[0.0; 5]]);
        eval([&logits]).unwrap();
        let q: Vec<(i32, f32)> = vec![(0, 0.4), (1, 0.1), (2, 0.2), (3, 0.1), (4, 0.2)];
        let draw = |u: f32| {
            let mut target = u;
            for &(t, w) in &q {
                target -= w;
                if target <= 0.0 {
                    return t;
                }
            }
            q[q.len() - 1].0
        };
        let max = p_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f64> = p_logits.iter().map(|&x| f64::from(x - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let n = 20_000usize;
        let p: Vec<f64> = weights.iter().map(|w| w / total).collect();
        let q_of = |t: usize| f64::from(q[t].1);
        let rate_with_q: f64 = (0..vocab).map(|t| p[t].min(q_of(t))).sum();
        let rate_point_mass: f64 = (0..vocab).map(|t| p[t] * q_of(t)).sum();
        let point_mass = |t: i32| vec![(t, 1.0f32)];
        let draft_q = |_: i32| q.clone();
        type DistOf<'a> = &'a dyn Fn(i32) -> Vec<(i32, f32)>;
        let cases: [(&str, DistOf<'_>, f64); 2] = [
            ("point mass", &point_mass, rate_point_mass),
            ("draft q", &draft_q, rate_with_q),
        ];
        for (name, dist_of, expected_rate) in cases {
            let mut proposals = SplitMix64::new(0x5eed);
            let mut sampler = MlxSampler::new(config.sampling, 0x24434);
            let mut counts = vec![0u64; vocab];
            let mut accepted = 0usize;
            for _ in 0..n {
                let proposed = draw(proposals.next_f32());
                let decision = decide(
                    &logits,
                    &[proposed],
                    &[dist_of(proposed)],
                    &[],
                    &config,
                    &mut sampler,
                    None,
                )
                .unwrap();
                counts[decision.committed[0] as usize] += 1;
                accepted += decision.accepted;
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
            // Binomial acceptance count within 5 standard deviations of its expectation.
            let mean = n as f64 * expected_rate;
            let sd = (n as f64 * expected_rate * (1.0 - expected_rate)).sqrt();
            assert!(
                (accepted as f64 - mean).abs() < 5.0 * sd,
                "{name}: {accepted} accepted of {n}, expected {mean:.0} ± {sd:.0}"
            );
        }
        assert!(
            rate_with_q - rate_point_mass > 0.2,
            "the rates are distinguishable"
        );
    }

    /// `StepTarget` runs any `Decode` model through the engine's `off` loop from a caller-prefilled
    /// cache, token-for-token the plain `generate_from_prefill` loop, and refuses a draft verify.
    #[test]
    fn a_step_target_is_the_plain_prefilled_loop() {
        let model = causal();
        for (name, config, _, _) in configs()
            .into_iter()
            .filter(|c| !c.0.contains("constrained"))
        {
            let prefill = || {
                let mut cache = crate::decode::Decode::make_cache(&model);
                let logits =
                    crate::decode::Decode::step(&model, &input_ids(&PROMPT), cache.as_mut(), 0)
                        .unwrap();
                (cache, logits)
            };
            let (mut cache, logits) = prefill();
            let expected = crate::decode::generate_from_prefill(
                &model,
                cache.as_mut(),
                logits,
                PROMPT.to_vec(),
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
            )
            .unwrap();
            let (mut cache, logits) = prefill();
            let run = generate_speculative(
                &StepTarget(&model),
                &mut NoProposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits,
                    hidden: None,
                    history: &PROMPT,
                    position_delta: 0,
                },
                &config,
                0,
                &CancelFlag::new(),
                &mut |_| {},
                EngineOptions::default(),
            )
            .unwrap();
            assert_eq!(run.output.tokens, expected.tokens, "{name}");
            assert_eq!(run.output.finish_reason, expected.finish_reason, "{name}");
            assert_eq!(run.report.path, "step_model", "{name}");
            assert_accounting(name, &run, &config);
        }
        let mut cache = crate::decode::Decode::make_cache(&model);
        let err = StepTarget(&model)
            .forward(&mut cache, &input_ids(&[1, 2]), 0, LogitsScope::All, false)
            .unwrap_err();
        assert!(err.to_string().contains("cannot verify drafts"), "{err}");
    }
}
