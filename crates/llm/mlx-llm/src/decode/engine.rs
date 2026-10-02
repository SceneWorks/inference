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
//!    [`Rollback::Direct`] truncation where the cache supports it — the softmax KV cache by
//!    offset ([`TruncateRollback`]), the Qwen35 hybrid by selecting its DeltaNet checkpoint
//!    ring's row for the last kept token and truncating its attention KV by offset
//!    ([`CheckpointRingRollback`], sc-24435) — else the cache is [`Rollback::Restored`] to the
//!    step start and the kept prefix is **replayed** in one forward ([`SnapshotRollback`], for a
//!    cache that can do neither). Both are counted.
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
//! sampler is a [`TokenSampler`] impl handed to [`EngineOptions::sampler`] whose draws may stay
//! device-resident ([`SampledToken::Device`]) — the loop feeds the next forward from the array and
//! reads the id back only for the stop check, history, constraint and event; the default
//! [`MlxSampler`] draws greedy and temperature / top-k / top-p on the device, and the `off` loop
//! pipelines such draws ([`generate_speculative`], story sc-24439). Any step-only [`Decode`] model
//! runs the `off` loop through [`StepTarget`]. None of them touches this loop.
//!
//! The verify-vs-decode kernel caveat of [`speculative`](super::speculative) applies: a multi-token
//! verify rounds a few bf16 ULP differently from the single-token step, so on real weights a greedy
//! speculative run tracks, rather than bit-matches, the token-at-a-time run where a near-tie flips.

use std::time::Instant;

use mlx_rs::transforms::{async_eval, eval};
use mlx_rs::Array;

use core_llm::speculative::{accept_token, greedy_commit, Acceptance};
use core_llm::{
    AcceptanceMonitor, CudaGraphsReport, DecodeReport, HostSampleReason, PathReport, PlainDecode,
    ProposerKind, SamplerPath, Speculative,
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
pub use crate::primitives::sampler::SampledToken;
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
    /// Stable lower-case label (`truncate`, `checkpoint_ring`, `snapshot_replay`).
    fn label(&self) -> &'static str;
    /// Remember whatever a later recovery needs, before the verify forward writes the cache. The
    /// cache is lent mutably so a strategy can arm per-step state inside it (a DeltaNet checkpoint
    /// ring, story sc-24435).
    fn begin(&mut self, cache: &mut C);
    /// Keep the first `keep` positions after the verify forward, ending the step. `keep` is the
    /// step start plus `1 + accepted`, at most the cache's current length.
    fn recover(&mut self, cache: &mut C, keep: i32) -> Result<Rollback>;
}

/// The rollback of a target that only serves the token-at-a-time loop ([`NoProposer`]): that loop
/// never arms it (a step without drafts writes nothing to recover), and a recovery is refused,
/// never approximated — e.g. a sliding-window cache that evicts as it grows cannot rebuild a
/// rolled-back position.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoDraftRollback;

impl<C> CacheRollback<C> for NoDraftRollback {
    fn label(&self) -> &'static str {
        "none"
    }

    fn begin(&mut self, _: &mut C) {}

    fn recover(&mut self, _: &mut C, _: i32) -> Result<Rollback> {
        Err(Error::Unsupported(
            "this target decodes token-at-a-time; it never verifies drafts".into(),
        ))
    }
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

/// Snapshot-and-replay rollback for a cache that can neither truncate nor checkpoint: the step
/// start is cloned before the verify forward — MLX arrays are refcounted, so the clone itself is
/// cheap, but while it is held every in-place KV write copies its block — and a partial acceptance
/// restores it, after which the engine replays the kept prefix. The snapshot is released at the
/// end of every step. No shipped target uses it since the Qwen35 hybrid rolls back through its
/// checkpoint ring ([`CheckpointRingRollback`]); it remains the generic fallback the
/// [`Rollback::Restored`] path serves.
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

/// Direct rollback for the Qwen35 hybrid through its DeltaNet checkpoint ring (sc-24435):
/// [`begin`](CacheRollback::begin) opens a checkpoint window at the step start, so every forward
/// until the recovery — the verify forward, or a draft model's several proposal forwards — keeps
/// each token's recurrent + conv state, and [`recover`](CacheRollback::recover) truncates to the
/// kept prefix and closes the window: the DeltaNet layers select the state kept at the last kept
/// position and the attention KV drops the rejected positions by offset. No replay forward, and no
/// snapshot: the cache is never cloned, so the forwards write the attention KV block in place.
///
/// The window is armed for `width + 1` tokens — a full verify step of `width` drafts, or a draft
/// model's proposal of fewer single-token forwards — the most
/// [`Qwen35Model::checkpoint_ring_bytes`] prices for `width`; a forward past it is refused
/// ([`Error::CheckpointWindowFull`]) rather than growing the ring unpriced.
#[derive(Clone, Copy, Debug)]
pub struct CheckpointRingRollback {
    max_tokens: i32,
}

impl CheckpointRingRollback {
    /// The ring rollback for a run of up to `width` drafts per step.
    pub fn new(width: usize) -> Self {
        let max_tokens = i32::try_from(width).map_or(i32::MAX, |w| w.saturating_add(1));
        Self { max_tokens }
    }
}

impl CacheRollback<Qwen35Cache> for CheckpointRingRollback {
    fn label(&self) -> &'static str {
        "checkpoint_ring"
    }

    fn begin(&mut self, cache: &mut Qwen35Cache) {
        cache.arm_checkpoints(self.max_tokens);
    }

    fn recover(&mut self, cache: &mut Qwen35Cache, keep: i32) -> Result<Rollback> {
        // Also closes the window when everything was kept (`keep == offset`).
        cache.truncate(keep)?;
        Ok(Rollback::Direct)
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
    /// A fresh rollback strategy for one run proposing up to `width` drafts per verify step (so
    /// a verify forward carries at most `width + 1` tokens).
    fn rollback(&self, width: usize) -> Self::Rollback;
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

    fn rollback(&self, _: usize) -> TruncateRollback {
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
    type Rollback = CheckpointRingRollback;

    fn new_cache(&self) -> Qwen35Cache {
        Qwen35Model::new_cache(self)
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

    fn rollback(&self, _: usize) -> TruncateRollback {
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
    /// Whether a draw reads the `history` window (a repetition / presence penalty). The engine
    /// pipelines only a sampler that does not: a look-ahead draw is made before the token it
    /// follows is known on the host, so its history is one token short. Conservatively `true`.
    fn reads_history(&self) -> bool {
        true
    }
}

/// The MLX sampler: the shared decode sampler [`sample_with_path`] with the seeded [`SplitMix64`].
/// Plain greedy and temperature / top-k / top-p draws stay on the device and are returned
/// unevaluated ([`SampledToken::Device`]) — only the chosen id is ever read back, and the engine
/// can pipeline them; a penalty, a constraint or a degenerate temperature draws on the host and
/// says why. Every draw is recorded at the branch that ran.
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
        self.params.is_penalized()
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
        Ok(token)
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

    fn reads_history(&self) -> bool {
        self.penalized()
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
    ///
    /// A stochastic draft is drawn from the returned distribution itself (one uniform of the seeded
    /// stream, the host reference's inverse-CDF walk), so the draft is exactly distributed as the
    /// `q` the acceptance test divides by — a device draw from the same knobs matches `q` only in
    /// distribution, and only up to threshold ties.
    pub fn sample_draft(&mut self, logits: &Array, draft_history: &[i32]) -> Result<DraftSample> {
        let greedy = self.greedy();
        let mask = self.constraint.as_mut().map(|c| c.allowed());
        let (draft, dist) = if greedy {
            let draft = self
                .sampler
                .sample(logits, draft_history, mask)?
                .resolve()?;
            (draft, None)
        } else {
            let dist = self.sampler.distribution(logits, draft_history, mask)?;
            (draw_from(&dist, self.sampler.uniform()), Some(dist))
        };
        if !self.is_stop(draft) {
            if let Some(c) = self.constraint.as_mut() {
                c.accept(draft);
                self.advanced = true;
            }
        }
        Ok((draft, dist))
    }
}

/// The inverse-CDF draw over an unnormalised, never-empty distribution given a uniform `u` in
/// `[0, 1)` — [`sample`](crate::primitives::sample)'s host walk; the draft rule of every MLX
/// speculative loop.
fn draw_from(dist: &[(i32, f32)], u: f32) -> i32 {
    let total: f32 = dist.iter().map(|x| x.1).sum();
    let mut target = u * total;
    for &(token, weight) in dist {
        target -= weight;
        if target <= 0.0 {
            return token;
        }
    }
    dist.last().map_or(0, |x| x.0)
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
    /// Whether the token-at-a-time loop may pipeline (see [`generate_speculative`]).
    pub pipelining: Pipelining,
    /// The request's speculative option (sc-24446). [`Speculative::Auto`] runs under `auto`'s
    /// acceptance monitor ([`AcceptanceMonitor::for_request`], against the plain loop this run
    /// would fall back to — pipelined or not), which demotes a proposer that is not paying for
    /// itself to token-at-a-time decoding (see [`generate_speculative`]); any other option — an
    /// explicit proposer, `off`, the default — runs the proposer to the end.
    pub speculative_mode: Speculative,
}

/// Whether [`generate_speculative`] pipelines its token-at-a-time loop (story sc-24439).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pipelining {
    /// Pipeline wherever it applies.
    #[default]
    Auto,
    /// Never: read every token back before the next step is enqueued (the parity reference).
    Off,
}

/// A finished engine run.
pub struct SpeculativeRun {
    /// The generated tokens and why generation stopped.
    pub output: GenerationOutput,
    /// The raw speculation counters.
    pub stats: SpeculativeStats,
    /// The measured decode report (fallbacks are the caller's to add).
    pub report: DecodeReport,
    /// The cache length the committed sequence accounts for: the prompt (or the caller's prefilled
    /// cache) plus every emitted token that has been fed back through the target — all of them
    /// after a stop-token end, all but the last otherwise (the last emitted token is the next
    /// forward's input). This is the length a [`Pipelining::Off`] run leaves, and the one a caller
    /// continuing a [`SpeculativePrompt::Prefilled`] cache resumes from: a pipelined run that ended
    /// with a discarded look-ahead ([`SpeculativeStats::discarded`]) — or a speculative step whose
    /// accepted drafts ran past the end — leaves rows beyond it (every row up to it is exact).
    /// An attention cache truncates back to it; a recurrent-state target (the Qwen35 hybrid's
    /// DeltaNet layers) cannot, so a caller that will continue such a cache runs with
    /// [`Pipelining::Off`].
    pub committed_cache_len: i32,
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
///
/// ## Pipelining (story sc-24439)
/// The token-at-a-time loop ([`NoProposer`]) is **pipelined** when the draws are device-resident
/// and independent of the host: no constraint, and a sampler that does not read the history
/// ([`TokenSampler::reads_history`]). Step `t + 1`'s forward and draw are enqueued on step `t`'s
/// unevaluated token and handed to the device ([`async_eval`]) *before* step `t`'s id is read back,
/// so the device computes the next step while the host commits this one. The look-ahead is never
/// enqueued past the budget; when step `t` ends the run (a stop token, the caller's stop predicate,
/// a cancel) the already-enqueued step `t + 1` is discarded unread — never emitted — and counted
/// ([`SpeculativeStats::discarded`]). A caller-prefilled cache then holds that discarded position
/// past the committed tokens (every committed position is still exact);
/// [`SpeculativeRun::committed_cache_len`] is where the committed sequence ends. The draws, and so the
/// output, are the same with or without pipelining ([`Pipelining::Off`]).
///
/// The first token is handed to the device before step 1 is enqueued behind it, so its read-back
/// waits for the prefill and its own draw only — never for step 1's forward (sc-24446).
///
/// A **speculative** step (any proposer) is not pipelined, even one whose proposer found nothing
/// to draft: the proposer must read the committed token on the host — the n-gram context, the MTP
/// head's hidden row — before it can draft the next step, and the verify decision reads the drafts'
/// rows, so there is no next step to enqueue before this one is read back. A penalized or
/// constrained run is not pipelined either: its draws read the host history / grammar that the
/// unread token would advance.
///
/// ## Demotion (sc-24446)
/// Under `auto` ([`EngineOptions::speculative_mode`]), the run's first
/// [`ACCEPTANCE_PROBE_VERIFIES`](core_llm::ACCEPTANCE_PROBE_VERIFIES) verify steps decide whether
/// the proposer pays for itself; below its break-even the run is **demoted**: no further step
/// proposes, asks for hidden rows or commits to the proposer, and — where the draws allow
/// pipelining — the rest of the run is handed to the pipelined token-at-a-time loop (otherwise it
/// continues as unpipelined single-token verify steps). The output is unchanged: a step without
/// drafts is the plain loop's draw. [`DecodeReport::speculative_demoted_at`] records where.
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
        pipelining,
        speculative_mode,
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
    // The fused-primitive routes this run builds (E3); a caller that prefilled outside the engine
    // reports its own window instead.
    let fused_start = crate::primitives::fused::fused_tally();
    let finished = |generated: Vec<i32>,
                    finish: FinishReason,
                    stats: SpeculativeStats,
                    sampler: &dyn TokenSampler,
                    timer: Option<GenerationTimer>,
                    start_len: i32,
                    on_event: &mut dyn FnMut(StreamEvent)| {
        on_event(StreamEvent::Done {
            reason: finish,
            generated: generated.len(),
        });
        // Every emitted token was fed back unless it is the last one and the run ended on it (a
        // stop token is never emitted, so a stop-token end fed them all).
        let fed = if finish == FinishReason::StopToken {
            generated.len()
        } else {
            generated.len().saturating_sub(1)
        };
        SpeculativeRun {
            committed_cache_len: start_len + fed as i32,
            output: GenerationOutput {
                tokens: generated,
                finish_reason: finish,
            },
            report: report(
                target,
                kind,
                width,
                &stats,
                sampler.path(),
                crate::primitives::fused::fused_tally().since(&fused_start),
            ),
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
        let start_len = match &prompt {
            SpeculativePrompt::Prefilled { cache, .. } => target.cache_len(cache),
            SpeculativePrompt::Tokens(_) => 0,
        };
        return Ok(finished(
            generated, finish, stats, sampler, timer, start_len, on_event,
        ));
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
    let start_len = target.cache_len(cache);
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
    let mut rollback = target.rollback(width);

    // A cancel that tripped during the prefill ends the run before its first draw, exactly as the
    // plain loop's per-step check does.
    if cancel.is_cancelled() {
        finish = FinishReason::Cancelled;
        return Ok(finished(
            generated, finish, stats, sampler, timer, start_len, on_event,
        ));
    }

    // ---- First token: an ordinary draw from the prefill logits. ----
    let first = {
        let mask = constraint.as_mut().map(|c| c.allowed());
        sampler.sample(&logits, &history, mask)?
    };
    // Whether the token-at-a-time steps may pipeline: the draws are device-resident and
    // independent of the host. A run with no proposer pipelines from token 0; a speculative run
    // `auto` demotes hands its remaining plain steps to the same loop.
    let can_pipeline = pipelining == Pipelining::Auto
        && crate::switches::PIPELINING.enabled()
        && constraint.is_none()
        && !sampler.reads_history();
    if can_pipeline && kind == ProposerKind::None && !wants_hidden {
        if let Some(warm) = warm.as_ref() {
            eval([warm])?;
        }
        drop(warm);
        drop(prompt_hidden);
        finish = pipelined_steps(
            target,
            cache,
            position_delta,
            first,
            Some(logits),
            PlainState {
                history: &mut history,
                generated: &mut generated,
                stats: &mut stats,
                release: &mut BufferRelease::new(),
            },
            config,
            &mut *sampler,
            should_stop,
            cancel,
            on_event,
        )?;
        return Ok(finished(
            generated, finish, stats, sampler, timer, start_len, on_event,
        ));
    }
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
        return Ok(finished(
            generated, finish, stats, sampler, timer, start_len, on_event,
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
    // `auto`'s acceptance monitor (sc-24446): once demoted, no step proposes, asks for hidden
    // rows or commits to the proposer — the rest of the run is token-at-a-time, handed to the
    // pipelined loop when the draws allow it.
    let mut monitor = AcceptanceMonitor::for_request(
        speculative_mode,
        kind,
        u32::try_from(width).unwrap_or(u32::MAX),
        if can_pipeline {
            PlainDecode::MlxPipelined
        } else {
            PlainDecode::MlxUnpipelined
        },
    );
    let mut demoted = false;
    'outer: while generated.len() < config.max_new_tokens && finish != FinishReason::Stopped {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }
        let remaining = config.max_new_tokens - generated.len();
        let k = if demoted {
            0
        } else {
            width.min(remaining.saturating_sub(1))
        };
        let wants_hidden = wants_hidden && !demoted;
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
        if !demoted {
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
        }
        let demote_now = monitor.as_mut().is_some_and(|m| m.observe(accepted));

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
        if demote_now {
            demoted = true;
            stats.demoted_at = Some(generated.len());
            previous_hidden = None;
            if can_pipeline {
                // Hand the rest to the pipelined loop: one plain step feeds `cur` (committed,
                // emitted, not yet in the cache) and draws the next token, which the loop then
                // dispatches, enqueues ahead of and reads back like any pipelined token.
                if cancel.is_cancelled() {
                    finish = FinishReason::Cancelled;
                    break;
                }
                let position = target.cache_len(cache) + position_delta;
                let out = target.forward(cache, &cur_input, position, LogitsScope::Last, false)?;
                stats.forwards += 1;
                stats.verify_steps += 1;
                let next = sampler.sample(&out.logits, &history, None)?;
                drop(out);
                finish = pipelined_steps(
                    target,
                    cache,
                    position_delta,
                    next,
                    None,
                    PlainState {
                        history: &mut history,
                        generated: &mut generated,
                        stats: &mut stats,
                        release: &mut release,
                    },
                    config,
                    &mut *sampler,
                    should_stop,
                    cancel,
                    on_event,
                )?;
                break;
            }
        }
    }

    Ok(finished(
        generated, finish, stats, sampler, timer, start_len, on_event,
    ))
}

/// The committed-sequence state the pipelined loop extends: the history (prompt + every committed
/// token), the emitted tokens, the run's counters and its buffer release.
struct PlainState<'s> {
    history: &'s mut Vec<i32>,
    generated: &'s mut Vec<i32>,
    stats: &'s mut SpeculativeStats,
    release: &'s mut BufferRelease,
}

thread_local! {
    /// Drawn tokens [`dispatch_token`] handed to the device on this thread.
    static TOKEN_DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Hand a drawn, device-resident token to the device ([`async_eval`]) so it is computed before
/// any later step enqueued behind it — the read-back then waits for this token only, never for the
/// step enqueued after it. Counted ([`token_dispatches`]) so the order is observable.
fn dispatch_token(id: &Array) -> Result<()> {
    async_eval([id])?;
    TOKEN_DISPATCHES.with(|n| n.set(n.get() + 1));
    Ok(())
}

/// Wait for a dispatched look-ahead the run will never read back (no host read): the device work
/// it enqueued finishes inside the request that enqueued it. Counted ([`drained_tokens`]).
fn drain_token(token: &SampledToken) -> Result<()> {
    if let SampledToken::Device(id) = token {
        eval([id])?;
    }
    #[cfg(test)]
    DRAINED_TOKENS.with(|n| n.set(n.get() + 1));
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Discarded look-aheads [`drain_token`] waited for on this thread.
    static DRAINED_TOKENS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many discarded look-aheads this thread's pipelined loops waited for before returning.
#[cfg(test)]
fn drained_tokens() -> u64 {
    DRAINED_TOKENS.with(std::cell::Cell::get)
}

/// How many drawn tokens this thread's decode loops handed to the device ahead of their
/// read-back ([`dispatch_token`]).
#[cfg(test)]
fn token_dispatches() -> u64 {
    TOKEN_DISPATCHES.with(std::cell::Cell::get)
}

/// The pipelined token-at-a-time loop (story sc-24439) from `pending` — a drawn token not yet
/// emitted whose forward has not run, the next position of `cache` — to the end of the run;
/// returns why it ended. `retire_on_first_read` is an array the first read-back retires (the
/// prefill logits).
///
/// Step `t + 1`'s forward and draw are enqueued on step `t`'s unread token and handed to the
/// device before step `t`'s id is read back, so the device computes the next step while the host
/// commits this one. `pending` itself is handed to the device first (sc-24446): otherwise its
/// read-back, queued behind step `t + 1`, waits for that whole forward — the first token of every
/// run would arrive one decode step late. The look-ahead is never enqueued past the budget, and a
/// look-ahead enqueued behind the token that ends the run is discarded unread and counted
/// ([`SpeculativeStats::discarded`]). The caller has checked the draws are device-resident and
/// independent of the host (no constraint, a sampler that does not read the history).
#[allow(clippy::too_many_arguments)]
fn pipelined_steps<T: SpeculativeTarget + ?Sized>(
    target: &T,
    cache: &mut T::Cache,
    position_delta: i32,
    mut pending: SampledToken,
    mut retire_on_first_read: Option<Array>,
    state: PlainState<'_>,
    config: &GenerationConfig,
    sampler: &mut dyn TokenSampler,
    should_stop: Option<&dyn Fn() -> bool>,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<FinishReason> {
    let PlainState {
        history,
        generated,
        stats,
        release,
    } = state;
    if let SampledToken::Device(id) = &pending {
        dispatch_token(id)?;
    }
    loop {
        // Enqueue step t + 1 on step t's unread token, then read step t back while the device
        // runs it. Never past the budget; a host draw is already read back, so it waits.
        let ahead = if pending.is_device() && generated.len() + 1 < config.max_new_tokens {
            let position = target.cache_len(cache) + position_delta;
            let out =
                target.forward(cache, &pending.input()?, position, LogitsScope::Last, false)?;
            let next = sampler.sample(&out.logits, history, None)?;
            if let SampledToken::Device(id) = &next {
                dispatch_token(id)?;
            }
            stats.forwards += 1;
            stats.pipelined += 1;
            Some(next)
        } else {
            None
        };
        let token = pending.resolve()?;
        // The first read-back evaluated the prefill: its prompt-length logits retire here and
        // the post-prefill buffer release rides the first `advance` below.
        drop(retire_on_first_read.take());
        let end = if config.stop_tokens.contains(&token) {
            Some(FinishReason::StopToken)
        } else {
            on_event(StreamEvent::Token {
                id: token,
                step: generated.len(),
            });
            generated.push(token);
            history.push(token);
            if should_stop.is_some_and(|stop| stop()) {
                Some(FinishReason::Stopped)
            } else if generated.len() >= config.max_new_tokens {
                Some(FinishReason::MaxTokens)
            } else if cancel.is_cancelled() {
                Some(FinishReason::Cancelled)
            } else {
                None
            }
        };
        if let Some(end) = end {
            if let Some(ahead) = ahead {
                stats.discarded += 1; // enqueued, never read back, never emitted
                                      // An end the loop could not foresee (a stop token, a caller stop, a cancel): the
                                      // look-ahead is already on the device. Wait for it here so its device time lands
                                      // in this request rather than in the next request's prefill (sc-24446); it is never
                                      // read back, and the cache row it wrote is past `committed_cache_len`.
                drain_token(&ahead)?;
            }
            return Ok(end);
        }
        release.advance(1);
        stats.verify_steps += 1;
        pending = match ahead {
            Some(next) => next,
            None => {
                let position = target.cache_len(cache) + position_delta;
                let out = target.forward(
                    cache,
                    &input_ids(&[token]),
                    position,
                    LogitsScope::Last,
                    false,
                )?;
                stats.forwards += 1;
                sampler.sample(&out.logits, history, None)?
            }
        };
    }
}

/// The run's measured report. Backend features MLX does not have (CUDA graphs, NVFP4
/// projections) report `none`; the fused primitives report the routes the run's ops took
/// (`fused`, the tally since the run started).
fn report<T: SpeculativeTarget + ?Sized>(
    target: &T,
    proposer: ProposerKind,
    drafts: usize,
    stats: &SpeculativeStats,
    sampler: Option<SamplerPath>,
    fused: core_llm::FusedTally,
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
        graph_path: "none".into(),
        nvfp4_projections: none(),
        fused_primitives: fused.path_report(),
        target_forwards: stats.forwards as u64,
        // The engine prefills the prompt (or counts a caller's prefill) as one forward; a caller
        // whose prefill took more adds them (sc-24437).
        prefill_forwards: 1,
        proposed_tokens: stats.proposed as u64,
        accepted_tokens: stats.accepted as u64,
        verify_steps: stats.verify_steps as u64,
        replay_forwards: stats.replays as u64,
        // A pipelined loop's look-ahead enqueued and never read back (counted in
        // `target_forwards`), so `target_forwards == prefill + verify + replay + discarded`.
        discarded_forwards: stats.discarded as u64,
        speculative_demoted_at: stats.demoted_at.map(|n| n as u64),
        prefix_hit_tokens: 0,
        prefix_cache: none(),
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

    use mlx_rs::Dtype;

    use super::*;
    use crate::config::{Architecture, ModelConfig};
    use crate::decode::proposers::{DraftModelProposer, MtpProposer, NgramProposer};
    use crate::decode::stream::generate_with;
    use crate::models::qwen35::tests::{cfg_json, cfg_json_mtp, synthetic_weights};
    use crate::models::Qwen35Config;
    use crate::primitives::sampler::{host_transfers, HostTransfers};
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
        synthetic_llama(16, layers, vocab)
    }

    /// A random llama-family decoder of width `hidden` (four query heads, two KV heads, a 2x MLP)
    /// and `layers` layers over a `vocab`-entry vocabulary, drawn from one fixed seed in
    /// [`causal_model`]'s order.
    fn synthetic_llama(hidden: i32, layers: usize, vocab: i32) -> CausalLm {
        let cfg = ModelConfig {
            hidden_size: hidden,
            intermediate_size: 2 * hidden,
            num_layers: layers,
            num_heads: 4,
            num_kv_heads: 2,
            head_dim: hidden / 4,
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
            activation_role: Default::default(),
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
        assert_eq!(r.prefill_forwards, 1, "{label}: one prefill forward: {r:?}");
        assert_eq!(
            r.target_forwards,
            r.prefill_forwards + r.verify_steps + r.replay_forwards + r.discarded_forwards,
            "{label}: forwards = prefill + verify steps + replays + discarded: {r:?}"
        );
    }

    /// Every sampler configuration the engine must honour exactly, with the sampler path the
    /// report must name for it.
    fn configs() -> Vec<(&'static str, GenerationConfig, bool, &'static str)> {
        vec![
            ("greedy", greedy(20), false, "device"),
            ("penalized", penalized(20), false, "host:penalty"),
            ("constrained", greedy(20), true, "host:constraint"),
            ("stochastic", stochastic(20), false, "device"),
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
            // Device draws pipeline every step after the first; host draws never do.
            let pipelined = if sampler == "device" { 19 } else { 0 };
            assert_eq!(run.stats.pipelined, pipelined, "{name}");
            assert_eq!(
                run.stats.discarded, 0,
                "{name}: a budget end enqueues nothing past it"
            );
        }
    }

    #[test]
    fn off_is_the_plain_loop_on_the_causal_and_hybrid_targets() {
        off_matches_plain(&causal(), 24);
        off_matches_plain(&qwen35(false), 50);
    }

    /// AC1: greedy prompt lookup emits exactly the plain loop's tokens — plain, penalized and
    /// constrained — on the Causal target (direct truncation) and the Qwen35 hybrid (its DeltaNet
    /// checkpoint ring, sc-24435: direct, never a replay forward), while actually proposing,
    /// accepting and rejecting.
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
            assert_eq!(s.replays, 0, "hybrid {name}: the ring never replays");
            assert_eq!(run.report.replay_forwards, 0, "hybrid {name}");
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
        assert!(
            run.stats.direct_rollbacks > 0 && run.stats.replays == 0,
            "{:?}",
            run.stats
        );
    }

    /// AC1: greedy MTP emits exactly the plain loop's tokens (plain, penalized, constrained) on
    /// the Qwen35 MTP fixture, whose adversarial drafts force a partial rejection — recovered
    /// through the checkpoint ring, with zero replay forwards (sc-24435).
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
                run.stats.proposed > 0 && run.stats.direct_rollbacks > 0,
                "{name}: {:?}",
                run.stats
            );
            assert_eq!(run.stats.replays, 0, "{name}: the ring never replays");
            assert_eq!(run.report.replay_forwards, 0, "{name}");
            assert_eq!(run.report.path, "mtp");
            assert_eq!(run.report.proposer, ProposerKind::Mtp);
            assert_eq!(run.report.draft_tokens, Some(3));
        }
    }

    /// The Qwen35 target under test: every forward is delegated to the model, then the attention
    /// KV's block-buffer addresses are recorded (the in-place probe) — rolled back by `R`, the
    /// checkpoint ring the target ships with or the step-start snapshot it replaced. A
    /// [`LogitsScope::Last`] forward (the prefill, a no-draft step, a replay) runs outside any
    /// checkpoint window: it must leave the cache recording nothing.
    struct Observed<'m, R> {
        model: &'m Qwen35Model,
        addresses: std::cell::RefCell<Vec<Vec<usize>>>,
        rollback: fn(usize) -> R,
    }

    impl<'m, R> Observed<'m, R> {
        fn new(model: &'m Qwen35Model, rollback: fn(usize) -> R) -> Self {
            Self {
                model,
                addresses: Default::default(),
                rollback,
            }
        }
    }

    impl<R: CacheRollback<Qwen35Cache>> SpeculativeTarget for Observed<'_, R> {
        type Cache = Qwen35Cache;
        type Rollback = R;

        fn new_cache(&self) -> Qwen35Cache {
            self.model.new_cache()
        }

        fn cache_len(&self, cache: &Qwen35Cache) -> i32 {
            cache.offset()
        }

        fn rollback(&self, width: usize) -> R {
            (self.rollback)(width)
        }

        fn forward(
            &self,
            cache: &mut Qwen35Cache,
            ids: &Array,
            rope_offset: i32,
            scope: LogitsScope,
            want_hidden: bool,
        ) -> Result<TargetOutput> {
            let out = SpeculativeTarget::forward(
                self.model,
                cache,
                ids,
                rope_offset,
                scope,
                want_hidden,
            )?;
            if scope == LogitsScope::Last {
                assert_eq!(
                    cache.checkpointed_tokens(),
                    0,
                    "a forward outside a verify step recorded checkpoints"
                );
            }
            self.addresses
                .borrow_mut()
                .push(cache.attn_buffer_addresses());
            Ok(out)
        }

        fn attention_label(&self) -> &'static str {
            "gqa"
        }
    }

    /// A proposer wrapper recording each step's `(proposed, accepted)` draft counts.
    struct Tally<P> {
        inner: P,
        proposed: usize,
        steps: Vec<(usize, usize)>,
    }

    impl<P> Tally<P> {
        fn new(inner: P) -> Self {
            Self {
                inner,
                proposed: 0,
                steps: Vec::new(),
            }
        }

        /// Steps whose every proposed draft was accepted.
        fn full_acceptances(&self) -> usize {
            self.steps.iter().filter(|&&(p, a)| p > 0 && a == p).count()
        }

        /// Steps that rejected a proposed draft.
        fn partial_rejections(&self) -> usize {
            self.steps.iter().filter(|&&(p, a)| a < p).count()
        }
    }

    /// A Qwen35 proposer run against the observed target: the model is what it proposes from.
    impl<'m, R, P> Proposer<Observed<'m, R>> for Tally<P>
    where
        R: CacheRollback<Qwen35Cache>,
        P: Proposer<Qwen35Model>,
    {
        fn kind(&self) -> ProposerKind {
            self.inner.kind()
        }

        fn wants_hidden(&self) -> bool {
            self.inner.wants_hidden()
        }

        fn warm(
            &mut self,
            target: &Observed<'m, R>,
            prompt: &[i32],
            hidden: Option<&Array>,
        ) -> Result<Option<Array>> {
            self.inner.warm(target.model, prompt, hidden)
        }

        fn propose(
            &mut self,
            target: &Observed<'m, R>,
            ctx: &ProposeContext<'_>,
            sampler: &mut DraftSampler<'_, '_>,
        ) -> Result<Proposal> {
            let proposal = self.inner.propose(target.model, ctx, sampler)?;
            self.proposed = proposal.drafts.len();
            Ok(proposal)
        }

        fn commit(
            &mut self,
            target: &Observed<'m, R>,
            cur: i32,
            accepted: &[i32],
            kept_hidden: Option<&Array>,
            position: i32,
        ) -> Result<()> {
            self.steps
                .push((std::mem::take(&mut self.proposed), accepted.len()));
            self.inner
                .commit(target.model, cur, accepted, kept_hidden, position)
        }
    }

    /// AC1 + AC2 (sc-24435): on the Qwen35 hybrid, prompt lookup and MTP recover every partial
    /// rejection through the DeltaNet checkpoint ring — zero replay forwards, the greedy tokens
    /// of the plain loop — and no step clones the target cache: across every forward after the
    /// prefill (full-acceptance steps included) the attention KV block buffers keep their
    /// addresses, i.e. every forward wrote them in place. The recurrence routes show the ring's
    /// cost is confined to the verify steps: prefill and no-draft steps run the final-state
    /// kernel, each verify step with drafts the checkpoint kernel.
    ///
    /// The probe is shown to see a copy: under the step-start snapshot the ring replaced, the same
    /// run clones the cache every verify step, the in-place write then lands in a fresh block,
    /// and a partial rejection costs a replay forward — with the same tokens.
    #[test]
    fn the_hybrid_rolls_back_through_its_checkpoint_ring_without_replay_or_clone() {
        use crate::models::qwen35::counting_cache_clones;
        use crate::primitives::gated_delta::{recording_routes, Route};
        mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
            let linear_layers = 3;
            for (label, model, mtp) in [
                ("prompt lookup", qwen35(false), false),
                ("mtp", qwen35(true), true),
            ] {
                let config = greedy(24);
                let expected = plain(&model, &PROMPT, &config, None);
                let ring = Observed::new(&model, CheckpointRingRollback::new);
                let run_with = |proposer: &mut dyn FnMut() -> SpeculativeRun| {
                    let ((run, clones), routes) =
                        recording_routes(|| counting_cache_clones(proposer));
                    (run, clones, routes)
                };
                let mut lookup = Tally::new(NgramProposer::default());
                let mut head = Tally::new(MtpProposer::new());
                let (run, clones, routes) = run_with(&mut || {
                    if mtp {
                        engine(&ring, &mut head, &PROMPT, &config, 3, None)
                    } else {
                        engine(&ring, &mut lookup, &PROMPT, &config, 4, None)
                    }
                });
                let (full, partial, steps) = if mtp {
                    (
                        head.full_acceptances(),
                        head.partial_rejections(),
                        head.steps.clone(),
                    )
                } else {
                    (
                        lookup.full_acceptances(),
                        lookup.partial_rejections(),
                        lookup.steps.clone(),
                    )
                };
                assert_eq!(run.output.tokens, expected.tokens, "{label}: != plain loop");
                assert_accounting(label, &run, &config);
                assert!(partial > 0, "{label}: a partial rejection ran: {steps:?}");
                assert_eq!(run.report.replay_forwards, 0, "{label}: {:?}", run.report);
                assert_eq!(run.stats.direct_rollbacks, partial, "{label}");
                assert_eq!(clones, 0, "{label}: the target cache is never cloned");
                let addresses = ring.addresses.borrow();
                assert_eq!(addresses.len() as u64, run.report.target_forwards);
                assert!(
                    addresses.windows(2).all(|w| w[0] == w[1]),
                    "{label}: the attention KV moved: {addresses:?}"
                );
                if !mtp {
                    // The adversarial MTP head never has a draft accepted; the lookup fixture
                    // covers the full-acceptance steps.
                    assert!(full > 0, "{label}: a full acceptance ran: {steps:?}");
                }
                let drafted = steps.iter().filter(|&&(p, _)| p > 0).count();
                let count = |r: Route| routes.iter().filter(|&&x| x == r).count();
                assert_eq!(
                    count(Route::CheckpointKernel),
                    linear_layers * drafted,
                    "{label}: {routes:?}"
                );
                assert_eq!(
                    count(Route::Kernel),
                    linear_layers * (1 + steps.len() - drafted),
                    "{label}: prefill and no-draft steps keep the final-state kernel"
                );
                assert_eq!(count(Route::Ops) + count(Route::Chunked), 0, "{label}");
            }

            // A full acceptance closes the window: an oracle proposing the plain loop's own next
            // token (always accepted) every other step, so each full acceptance is followed by a
            // no-draft step, whose forward the observed target checks records nothing.
            struct Oracle {
                expected: Vec<i32>,
                steps: usize,
            }
            impl<T: SpeculativeTarget + ?Sized> Proposer<T> for Oracle {
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
                    self.steps += 1;
                    let next = self.expected.get(ctx.history.len() - PROMPT.len());
                    Ok(Proposal {
                        drafts: next
                            .filter(|_| self.steps % 2 == 1)
                            .copied()
                            .into_iter()
                            .collect(),
                        dists: Vec::new(),
                    })
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
            let model = qwen35(false);
            let config = greedy(12);
            let expected = plain(&model, &PROMPT, &config, None).tokens;
            let ring = Observed::new(&model, CheckpointRingRollback::new);
            let mut oracle = Tally::new(Oracle {
                expected: expected.clone(),
                steps: 0,
            });
            let run = engine(&ring, &mut oracle, &PROMPT, &config, 4, None);
            assert_eq!(run.output.tokens, expected, "oracle: != plain loop");
            let followed = oracle
                .steps
                .windows(2)
                .filter(|w| w[0] == (1, 1) && w[1].0 == 0)
                .count();
            assert!(
                followed > 0,
                "a no-draft step followed a full acceptance: {:?}",
                oracle.steps
            );

            // The contrast: the same lookup run on the step-start snapshot.
            let model = qwen35(false);
            let config = greedy(24);
            let snapshot = Observed::new(&model, |_| SnapshotRollback::<Qwen35Cache>::default());
            let (run, clones) = counting_cache_clones(|| {
                engine(
                    &snapshot,
                    &mut Tally::new(NgramProposer::default()),
                    &PROMPT,
                    &config,
                    4,
                    None,
                )
            });
            assert_eq!(
                run.output.tokens,
                plain(&model, &PROMPT, &config, None).tokens,
                "replay recovers the same tokens"
            );
            assert!(run.stats.replays > 0, "{:?}", run.stats);
            assert!(clones > 0, "the snapshot clones every verify step");
            let addresses = snapshot.addresses.borrow();
            assert!(
                addresses.windows(2).any(|w| w[0] != w[1]),
                "a held snapshot makes the in-place write copy its block"
            );
        });
    }

    /// The draft-model window through the [`CacheRollback`] seam (sc-24435; the shape a draft
    /// proposer drives on its own cache): one `begin`, three separate single-token forwards, then
    /// `recover` to each position from the step start (nothing kept) to all three kept. The
    /// window is armed for `width + 1` tokens, so width 2 fits exactly these three forwards and
    /// refuses a fourth (typed, the cache untouched). Every
    /// recovery is direct, its DeltaNet state is bit-exact that of a cache that only ever saw the
    /// kept tokens, the next forward's logits are too (the attention KV was truncated by offset),
    /// and that forward records nothing — full acceptance included.
    #[test]
    fn the_ring_recovers_a_draft_window_of_single_token_forwards() {
        let model = qwen35(false);
        let step = |cache: &mut Qwen35Cache, ids: &[i32]| -> Vec<f32> {
            let offset = cache.offset();
            let out = SpeculativeTarget::forward(
                &model,
                cache,
                &input_ids(ids),
                offset,
                LogitsScope::Last,
                false,
            )
            .unwrap();
            out.logits
                .as_dtype(Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        let prefilled = || {
            let mut cache = model.new_cache();
            step(&mut cache, &PROMPT);
            cache
        };
        let start = PROMPT.len() as i32;
        let drafts = [4, 11, 7];
        for kept in 0..=drafts.len() {
            let mut cache = prefilled();
            let mut rollback = SpeculativeTarget::rollback(&model, 2);
            rollback.begin(&mut cache);
            for &d in &drafts {
                step(&mut cache, &[d]);
            }
            // A fourth forward would outgrow what width 2 prices: refused, cache untouched.
            let before = cache.delta_states();
            let offset = cache.offset();
            let err = SpeculativeTarget::forward(
                &model,
                &mut cache,
                &input_ids(&[20]),
                offset,
                LogitsScope::Last,
                false,
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::CheckpointWindowFull { max_tokens: 3, .. }),
                "{err}"
            );
            assert_eq!((cache.offset(), cache.delta_states()), (offset, before));
            let keep = start + kept as i32;
            assert_eq!(
                rollback.recover(&mut cache, keep).unwrap(),
                Rollback::Direct
            );
            assert_eq!(cache.offset(), keep);
            let mut reference = prefilled();
            for &d in &drafts[..kept] {
                step(&mut reference, &[d]);
            }
            assert_eq!(
                cache.delta_states(),
                reference.delta_states(),
                "kept {kept}: restored state"
            );
            assert_eq!(
                step(&mut cache, &[13]),
                step(&mut reference, &[13]),
                "kept {kept}: next logits"
            );
            assert_eq!(cache.checkpointed_tokens(), 0, "kept {kept}: window closed");
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
        // E3: MLX has no graph runner, so no step went through one — `none`, never `eager`.
        assert_eq!(r.graph_path, "none");
        // A speculative run never looks ahead, and the causal fixture runs no fused primitive.
        assert_eq!(r.discarded_forwards, 0);
        assert_eq!(r.fused_primitives.path, "none");
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
            let mut proposer = DraftModelProposer::new(draft, 4);
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
    /// drafts), the hybrid target (its DeltaNet checkpoint ring, sc-24435: direct, never a
    /// replay) with a softmax draft, a softmax target with the hybrid as its draft (the draft
    /// recovers its own proposal window through its ring), and the hybrid drafting for an
    /// identical hybrid (every draft accepted).
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
        assert!(
            stats
                .iter()
                .all(|s| s.replays == 0 && s.direct_rollbacks > 0),
            "the hybrid target recovers through its ring: {stats:?}"
        );

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
        let mut proposer = DraftModelProposer::new(&draft, 4);
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
                &mut DraftModelProposer::new(&draft, 3),
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
        fn new(draft: &'d D, drafts: usize) -> Self {
            Self {
                inner: DraftModelProposer::new(draft, drafts),
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
                let mut proposer = CheckedDraft::new(draft, 4);
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
        // The hybrid draft (its checkpoint ring) for a softmax target that disagrees with it, and
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
    /// greedy one, for a truncating and a checkpoint-ring draft — and the draft's cache is
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
                        &mut CheckedDraft::new(target, 4),
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
        let proposer = DraftModelProposer::new(&draft, 4).with_vocab(20, 28);
        let mut row = vec![0f32; 24];
        row[22] = 10.0;
        row[3] = 1.0;
        let shaped = host_row(&proposer.shaped(Array::from_slice(&row, &[1, 24])).unwrap());
        assert_eq!(shaped.len(), 28);
        assert_eq!(&shaped[..20], &row[..20]);
        assert!(shaped[20..].iter().all(|x| *x == f32::NEG_INFINITY));
        // Same width, proposable = the whole width: untouched.
        let untouched = DraftModelProposer::new(&draft, 4).with_vocab(24, 24);
        assert_eq!(
            host_row(&untouched.shaped(Array::from_slice(&row, &[1, 24])).unwrap()),
            row
        );
        // Through the engine: the target scores 24 ids, the draft proposes from its first 20.
        let target = causal();
        for config in [greedy(20), penalized(20)] {
            let expected = plain(&target, &PROMPT, &config, None);
            let mut proposer = DraftModelProposer::new(&draft, 4).with_vocab(20, 24);
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

    // ---- Pipelining and the device sampler (story sc-24439). ----

    /// Temperature 0.7 / top-p 0.9 — the device sampler's request.
    fn top_p(max_new_tokens: usize) -> GenerationConfig {
        let mut config = greedy(max_new_tokens);
        config.sampling.temperature = 0.7;
        config.sampling.top_p = 0.9;
        config
    }

    /// Stop the caller's way after this many tokens: `false` through the stop predicate, `true` by
    /// tripping the cancel flag from the event sink.
    type StopAfter = Option<(usize, bool)>;

    /// An `off` run with `pipelining`, stopping the caller's way when asked ([`StopAfter`]).
    fn off_run<T: SpeculativeTarget>(
        target: &T,
        config: &GenerationConfig,
        pipelining: Pipelining,
        stop_after: StopAfter,
    ) -> SpeculativeRun {
        off_run_from(
            target,
            SpeculativePrompt::Tokens(&PROMPT),
            config,
            pipelining,
            stop_after,
        )
    }

    /// [`off_run`] from `prompt` (a caller-prefilled cache, say).
    fn off_run_from<T: SpeculativeTarget>(
        target: &T,
        prompt: SpeculativePrompt<'_, T::Cache>,
        config: &GenerationConfig,
        pipelining: Pipelining,
        stop_after: StopAfter,
    ) -> SpeculativeRun {
        let cancel = CancelFlag::new();
        let emitted = std::cell::Cell::new(0usize);
        let predicate = || matches!(stop_after, Some((n, false)) if emitted.get() >= n);
        let mut events = Vec::new();
        let run = generate_speculative(
            target,
            &mut NoProposer,
            prompt,
            config,
            0,
            &cancel,
            &mut |e| {
                if matches!(e, StreamEvent::Token { .. }) {
                    emitted.set(emitted.get() + 1);
                    if matches!(stop_after, Some((n, true)) if emitted.get() >= n) {
                        cancel.cancel();
                    }
                }
                events.push(e);
            },
            EngineOptions {
                should_stop: Some(&predicate),
                pipelining,
                ..EngineOptions::default()
            },
        )
        .unwrap();
        let ids: Vec<i32> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Token { id, .. } => Some(*id),
                StreamEvent::Done { .. } => None,
            })
            .collect();
        assert_eq!(ids, run.output.tokens, "nothing past the end is emitted");
        run
    }

    /// AC1: pipelining is invisible in the output. On the Causal and Qwen35 targets, greedy and
    /// seeded device-sampled runs emit exactly the unpipelined run's tokens — to the budget, and
    /// ending on a stop token, the caller's stop predicate and a cancel, where the enqueued
    /// look-ahead is discarded unread (one forward, never emitted).
    fn pipelining_is_invisible_on<T: SpeculativeTarget + crate::decode::Decode>(target: &T) {
        for (name, config) in [("greedy", greedy(20)), ("top-p", top_p(20))] {
            let off = off_run(target, &config, Pipelining::Off, None);
            let on = off_run(target, &config, Pipelining::Auto, None);
            assert_eq!(on.output.tokens, off.output.tokens, "{name}");
            assert_eq!(on.output.tokens.len(), 20, "{name}");
            assert_eq!((on.stats.pipelined, off.stats.pipelined), (19, 0), "{name}");
            assert_eq!(on.stats.discarded, 0, "{name}");
            assert_eq!(
                on.report.target_forwards, off.report.target_forwards,
                "{name}"
            );
            if name == "greedy" {
                assert_eq!(
                    on.output.tokens,
                    plain(target, &PROMPT, &config, None).tokens
                );
            }

            let mut stopping = config.clone();
            stopping.stop_tokens = vec![on.output.tokens[7]];
            let first_stop = on
                .output
                .tokens
                .iter()
                .position(|t| *t == on.output.tokens[7]);
            let ends: [(&str, &GenerationConfig, StopAfter); 3] = [
                ("stop token", &stopping, None),
                ("stop predicate", &config, Some((5, false))),
                ("cancel", &config, Some((5, true))),
            ];
            for (end, config, stop_after) in ends {
                let off = off_run(target, config, Pipelining::Off, stop_after);
                let on = off_run(target, config, Pipelining::Auto, stop_after);
                let label = format!("{name} / {end}");
                assert_eq!(on.output.tokens, off.output.tokens, "{label}");
                assert_eq!(on.output.finish_reason, off.output.finish_reason, "{label}");
                let (len, reason) = match end {
                    "stop token" => (first_stop.unwrap(), FinishReason::StopToken),
                    "stop predicate" => (5, FinishReason::Stopped),
                    _ => (5, FinishReason::Cancelled),
                };
                assert_eq!(
                    (on.output.tokens.len(), on.output.finish_reason),
                    (len, reason),
                    "{label}"
                );
                assert_eq!(
                    on.stats.discarded, 1,
                    "{label}: the look-ahead is discarded"
                );
                // E3: the report counts the discarded look-ahead, so the documented forward
                // invariant holds on every end — pipelined or not.
                assert_eq!(on.report.discarded_forwards, 1, "{label}");
                assert_eq!(off.report.discarded_forwards, 0, "{label}");
                for (run, which) in [(&on, "pipelined"), (&off, "unpipelined")] {
                    let r = &run.report;
                    assert_eq!(
                        r.target_forwards,
                        r.prefill_forwards + r.verify_steps + r.replay_forwards + r.discarded_forwards,
                        "{label} ({which}): forwards = prefill + verify + replay + discarded: {r:?}"
                    );
                }
                assert_eq!(
                    on.report.target_forwards,
                    off.report.target_forwards + 1,
                    "{label}: the discarded look-ahead is a forward that ran"
                );
                assert_eq!(on.report.verify_steps, off.report.verify_steps, "{label}");
            }
        }
    }

    #[test]
    fn pipelining_is_invisible_on_the_causal_and_hybrid_targets() {
        pipelining_is_invisible_on(&causal());
        pipelining_is_invisible_on(&qwen35(false));
    }

    /// sc-24446 (E5): the process switches the campaign isolates pipelining and the device sampler
    /// with (`MLX_LLM_PIPELINING`, `MLX_LLM_DEVICE_SAMPLER`; here their thread-scoped layer) turn
    /// each off for a request that would otherwise take it, and the greedy tokens do not move.
    #[test]
    fn the_pipelining_and_device_sampler_switches_turn_each_path_off() {
        use crate::switches::{DEVICE_SAMPLER, PIPELINING};
        fn check<T: SpeculativeTarget>(label: &str, model: &T) {
            let run = || off_run(model, &greedy(12), Pipelining::Auto, None);
            let on = run();
            assert_eq!(on.stats.pipelined, 11, "{label}: pipelined when allowed");
            assert_eq!(on.report.sampler, "device", "{label}");
            let unpipelined = PIPELINING.scoped(false, run);
            assert_eq!(unpipelined.stats.pipelined, 0, "{label}: the switch is off");
            assert_eq!(unpipelined.output.tokens, on.output.tokens, "{label}");
            let host = DEVICE_SAMPLER.scoped(false, run);
            assert_eq!(host.report.sampler, "host:reference", "{label}");
            assert_eq!(host.stats.pipelined, 0, "{label}: a host draw is read back");
            assert_eq!(host.output.tokens, on.output.tokens, "{label}");
        }
        check("causal", &causal());
        check("qwen35", &qwen35(false));
    }

    /// AC1 (speculative half): a run with a proposer is never pipelined — the proposer reads the
    /// committed token on the host before it drafts — so prompt lookup and MTP report no pipelined
    /// steps while the greedy output stays the plain loop's.
    #[test]
    fn speculative_runs_are_not_pipelined() {
        let causal = causal();
        let lookup = engine(
            &causal,
            &mut NgramProposer::default(),
            &PROMPT,
            &greedy(20),
            4,
            None,
        );
        let hybrid = qwen35(true);
        let mtp = engine(
            &hybrid,
            &mut MtpProposer::new(),
            &PROMPT,
            &greedy(20),
            2,
            None,
        );
        for (label, run) in [("prompt lookup", &lookup), ("mtp", &mtp)] {
            assert_eq!(run.stats.pipelined, 0, "{label}");
            assert!(run.stats.verify_steps > 0, "{label}");
        }
        assert_eq!(
            lookup.output.tokens,
            plain(&causal, &PROMPT, &greedy(20), None).tokens
        );
    }

    /// AC2: a temperature 0.7 / top-p 0.9 request draws on the device (`sampler = device`) and
    /// moves exactly one token id per step to the host — counted — pipelined or not; a greedy
    /// request likewise; a penalized request copies the whole row every step.
    #[test]
    fn a_top_p_request_samples_on_the_device_one_id_per_step() {
        let model = causal();
        for pipelining in [Pipelining::Auto, Pipelining::Off] {
            for (name, config) in [("top-p", top_p(16)), ("greedy", greedy(16))] {
                let before = host_transfers();
                let run = off_run(&model, &config, pipelining, None);
                let moved = host_transfers().since(before);
                assert_eq!(run.report.sampler, "device", "{name} {pipelining:?}");
                assert_eq!(run.output.tokens.len(), 16);
                assert_eq!(
                    moved,
                    HostTransfers {
                        reads: 16,
                        elements: 16
                    },
                    "{name} {pipelining:?}: one id per step"
                );
            }
        }
        let before = host_transfers();
        let run = off_run(&model, &penalized(16), Pipelining::Auto, None);
        let moved = host_transfers().since(before);
        assert_eq!(run.report.sampler, "host:penalty");
        assert_eq!(
            moved.elements,
            16 * 24,
            "a penalized draw copies the vocab row"
        );
    }

    /// AC3: a repetition-penalty request samples on the host, says so with its reason, and is not
    /// pipelined (its draw reads the history the unread token would extend); a constrained one
    /// likewise with its own reason.
    #[test]
    fn a_penalized_or_constrained_request_reports_the_host_path_and_is_not_pipelined() {
        let model = causal();
        let mut repetition = top_p(12);
        repetition.sampling.repetition_penalty = 1.2;
        repetition.sampling.repetition_context = 16;
        let run = off_run(&model, &repetition, Pipelining::Auto, None);
        assert_eq!(run.report.sampler, "host:penalty");
        assert_eq!(run.stats.pipelined, 0);
        let mut forbid = Forbid::new(24, &[1, 9]);
        let run = engine(
            &model,
            &mut NoProposer,
            &PROMPT,
            &top_p(12),
            0,
            Some(&mut forbid),
        );
        assert_eq!(run.report.sampler, "host:constraint");
        assert_eq!(run.stats.pipelined, 0);
    }

    /// A target that records, at each forward, how many host reads the sampling seam had made
    /// since the run began — when each step was enqueued relative to the token reads.
    struct ReadsAtForward<'a, T> {
        inner: &'a T,
        start: HostTransfers,
        reads: std::cell::RefCell<Vec<u64>>,
    }

    impl<T: SpeculativeTarget> SpeculativeTarget for ReadsAtForward<'_, T> {
        type Cache = T::Cache;
        type Rollback = T::Rollback;

        fn new_cache(&self) -> T::Cache {
            self.inner.new_cache()
        }

        fn cache_len(&self, cache: &T::Cache) -> i32 {
            self.inner.cache_len(cache)
        }

        fn rollback(&self, width: usize) -> T::Rollback {
            self.inner.rollback(width)
        }

        fn forward(
            &self,
            cache: &mut T::Cache,
            ids: &Array,
            rope_offset: i32,
            scope: LogitsScope,
            want_hidden: bool,
        ) -> Result<TargetOutput> {
            let reads = host_transfers().since(self.start).reads;
            self.reads.borrow_mut().push(reads);
            self.inner
                .forward(cache, ids, rope_offset, scope, want_hidden)
        }

        fn attention_label(&self) -> &'static str {
            self.inner.attention_label()
        }
    }

    /// The pipelining is real: step `t + 1`'s forward is enqueued before token `t` is read back.
    /// With the prefill first, the `k`-th forward (`k >= 1`) runs after `k - 1` token reads under
    /// `Auto` — one behind the unpipelined loop, whose `k`-th forward follows `k` reads.
    #[test]
    fn a_pipelined_step_is_enqueued_before_the_previous_token_is_read() {
        let model = causal();
        for (pipelining, expected) in [
            (Pipelining::Auto, vec![0, 0, 1, 2, 3, 4, 5, 6]),
            (Pipelining::Off, vec![0, 1, 2, 3, 4, 5, 6, 7]),
        ] {
            let target = ReadsAtForward {
                inner: &model,
                start: host_transfers(),
                reads: Default::default(),
            };
            let run = off_run(&target, &top_p(8), pipelining, None);
            assert_eq!(run.output.tokens.len(), 8);
            assert_eq!(target.reads.into_inner(), expected, "{pipelining:?}");
        }
    }

    /// Every device-to-host read a decode loop makes goes through the counted sampling seam
    /// ([`SampledToken::resolve`], the sampler's row copy), so the AC2 counter sees them all: no
    /// loop in `decode/` reads an array back itself. A source scan of the non-test code.
    #[test]
    fn decode_loops_make_no_uncounted_host_read() {
        let sources = [
            ("engine.rs", include_str!("engine.rs")),
            ("stream.rs", include_str!("stream.rs")),
            ("prefix.rs", include_str!("prefix.rs")),
            ("batch.rs", include_str!("batch.rs")),
            ("continuous.rs", include_str!("continuous.rs")),
            ("speculative.rs", include_str!("speculative.rs")),
            ("proposers.rs", include_str!("proposers.rs")),
        ];
        let host_reads = [
            ".item::<",
            ".item(",
            ".try_item",
            "as_slice::<",
            "try_as_slice",
            "as_slice_unchecked",
        ];
        for (name, src) in sources {
            let code = src.split("\n#[cfg(test)]").next().unwrap();
            for read in host_reads {
                assert!(
                    !code.contains(read),
                    "decode/{name} reads an array back outside the counted seam (`{read}`)"
                );
            }
        }
    }

    /// A pipelined run that ends with a discarded look-ahead reports the committed cache length —
    /// the unpipelined run's — and a caller continuing its prefilled cache from there (truncating
    /// the look-ahead row) gets a cold forward's logits.
    #[test]
    fn a_pipelined_end_reports_the_committed_cache_length() {
        let model = causal();
        let prefilled_run = |config: &GenerationConfig, pipelining, stop_after| {
            let mut cache = model.new_cache();
            let logits = SpeculativeTarget::forward(
                &model,
                &mut cache,
                &input_ids(&PROMPT),
                0,
                LogitsScope::Last,
                false,
            )
            .unwrap()
            .logits;
            let run = off_run_from(
                &model,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits,
                    hidden: None,
                    history: &PROMPT,
                    position_delta: 0,
                },
                config,
                pipelining,
                stop_after,
            );
            (run, cache)
        };
        let free = off_run(&model, &top_p(20), Pipelining::Off, None)
            .output
            .tokens;
        let mut stopping = top_p(20);
        stopping.stop_tokens = vec![free[7]];
        let ends: [(&str, &GenerationConfig, StopAfter); 4] = [
            ("stop token", &stopping, None),
            ("stop predicate", &top_p(20), Some((5, false))),
            ("cancel", &top_p(20), Some((5, true))),
            ("budget", &top_p(6), None),
        ];
        for (end, config, stop_after) in ends {
            let (off, off_cache) = prefilled_run(config, Pipelining::Off, stop_after);
            let (on, mut on_cache) = prefilled_run(config, Pipelining::Auto, stop_after);
            assert_eq!(on.output.tokens, off.output.tokens, "{end}");
            let tokens = &on.output.tokens;
            let fed = if on.output.finish_reason == FinishReason::StopToken {
                tokens.len()
            } else {
                tokens.len() - 1
            };
            let committed = PROMPT.len() as i32 + fed as i32;
            assert_eq!(off.committed_cache_len, committed, "{end}");
            assert_eq!(
                off_cache.offset(),
                committed,
                "{end}: off leaves the committed rows"
            );
            assert_eq!(on.committed_cache_len, committed, "{end}");
            let discarded = if end == "budget" { 0 } else { 1 };
            assert_eq!(on.stats.discarded, discarded, "{end}");
            assert_eq!(on_cache.offset(), committed + discarded as i32, "{end}");

            // Continue from the committed length with the next input: the last emitted token, or
            // the stop token that ended the run.
            let next = if fed == tokens.len() {
                stopping.stop_tokens[0]
            } else {
                tokens[fed]
            };
            on_cache.truncate(on.committed_cache_len).unwrap();
            let warm = model
                .decode_logits(&input_ids(&[next]), &mut on_cache, on.committed_cache_len)
                .unwrap();
            let mut sequence = PROMPT.to_vec();
            sequence.extend_from_slice(&tokens[..fed]);
            sequence.push(next);
            let mut cold_cache = model.new_cache();
            let cold = model
                .decode_logits(&input_ids(&sequence), &mut cold_cache, 0)
                .unwrap();
            let host = crate::primitives::kv_cache::testing::host;
            let (warm, cold) = (host(&warm), host(&cold));
            let drift = warm
                .iter()
                .zip(&cold)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(drift < 1e-4, "{end}: continuation drift {drift}");
        }
    }

    /// A seeded device-sampled run is reproducible, and a different seed draws differently: the
    /// seed still drives the device draws (the documented change is only that they are not the
    /// host sampler's per-seed draws).
    #[test]
    fn seeded_device_sampling_is_reproducible() {
        let model = causal();
        let a = off_run(&model, &top_p(24), Pipelining::Auto, None);
        let b = off_run(&model, &top_p(24), Pipelining::Auto, None);
        assert_eq!(a.output.tokens, b.output.tokens);
        let mut other = top_p(24);
        other.seed = Some(8);
        let c = off_run(&model, &other, Pipelining::Auto, None);
        assert_ne!(a.output.tokens, c.output.tokens);
    }

    /// Pipelining overlap microbenchmark (manual, `--ignored`; prints, asserts nothing timed): a
    /// synthetic ~0.25 GB decoder (hidden 1024, 6 layers, vocab 32k), 128 tokens, pipelining off
    /// and on alternated seven times each, greedy and top-p; the median decode rate of each. The
    /// output is identical either way (asserted); the rate shows the host work the look-ahead
    /// hides behind the device. Run it with `--release` for representative host overhead.
    #[test]
    #[ignore = "manual timing microbenchmark"]
    fn pipelining_overlap_microbench() {
        let model = synthetic_llama(1024, 6, 32_000);
        for (name, config) in [("greedy", greedy(128)), ("top-p", top_p(128))] {
            let mut tokens = None;
            let mut rates = [Vec::new(), Vec::new()];
            off_run(&model, &config, Pipelining::Auto, None); // warm the kernels
            for _ in 0..7 {
                for (slot, pipelining) in
                    [Pipelining::Off, Pipelining::Auto].into_iter().enumerate()
                {
                    let started = Instant::now();
                    let run = off_run(&model, &config, pipelining, None);
                    let secs = started.elapsed().as_secs_f64();
                    rates[slot].push(run.output.tokens.len() as f64 / secs);
                    let first = tokens.get_or_insert_with(|| run.output.tokens.clone());
                    assert_eq!(first, &run.output.tokens, "{name}");
                }
            }
            for rates in &mut rates {
                rates.sort_by(f64::total_cmp);
            }
            eprintln!(
                "{name:>6}: off {:.1} tok/s, pipelined {:.1} tok/s (medians of 7)",
                rates[0][3], rates[1][3]
            );
        }
    }

    // ---- `auto`'s acceptance monitor and the first-token dispatch (sc-24446). ----

    /// A target that records, at each forward, how many drawn tokens the decode loops had handed
    /// to the device ([`dispatch_token`]) since the run began, and whether it asked for hidden
    /// rows.
    struct DispatchesAtForward<'a, T> {
        inner: &'a T,
        start: u64,
        dispatched: std::cell::RefCell<Vec<u64>>,
        hidden: std::cell::RefCell<Vec<bool>>,
    }

    impl<'a, T> DispatchesAtForward<'a, T> {
        fn new(inner: &'a T) -> Self {
            Self {
                inner,
                start: token_dispatches(),
                dispatched: Default::default(),
                hidden: Default::default(),
            }
        }
    }

    impl<T: SpeculativeTarget> SpeculativeTarget for DispatchesAtForward<'_, T> {
        type Cache = T::Cache;
        type Rollback = T::Rollback;

        fn new_cache(&self) -> T::Cache {
            self.inner.new_cache()
        }

        fn cache_len(&self, cache: &T::Cache) -> i32 {
            self.inner.cache_len(cache)
        }

        fn rollback(&self, width: usize) -> T::Rollback {
            self.inner.rollback(width)
        }

        fn forward(
            &self,
            cache: &mut T::Cache,
            ids: &Array,
            rope_offset: i32,
            scope: LogitsScope,
            want_hidden: bool,
        ) -> Result<TargetOutput> {
            self.dispatched
                .borrow_mut()
                .push(token_dispatches() - self.start);
            self.hidden.borrow_mut().push(want_hidden);
            self.inner
                .forward(cache, ids, rope_offset, scope, want_hidden)
        }

        fn attention_label(&self) -> &'static str {
            self.inner.attention_label()
        }
    }

    /// Token 0 is handed to the device before step 1 is enqueued behind it, so its read-back
    /// waits for the prefill and its draw only: at the `k`-th forward (`k >= 1`) `k` tokens have
    /// been dispatched — token 0 included — where the pre-fix loop had dispatched `k - 1` (token
    /// 0 then waited for step 1's whole forward). Step 1 is still enqueued before token 0 is read
    /// back ([`a_pipelined_step_is_enqueued_before_the_previous_token_is_read`]), so the overlap
    /// is kept; the unpipelined loop dispatches nothing ahead of its reads.
    #[test]
    fn the_first_token_is_dispatched_before_step_one_is_enqueued() {
        let model = causal();
        for (pipelining, expected) in [
            (Pipelining::Auto, vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (Pipelining::Off, vec![0; 8]),
        ] {
            let target = DispatchesAtForward::new(&model);
            let run = off_run(&target, &top_p(8), pipelining, None);
            assert_eq!(run.output.tokens.len(), 8);
            assert_eq!(target.dispatched.into_inner(), expected, "{pipelining:?}");
            let again = off_run(&model, &top_p(8), pipelining, None);
            assert_eq!(again.output.tokens, run.output.tokens, "{pipelining:?}");
        }
    }

    /// sc-24446: a pipelined run never enqueues a forward past its budget, and a run that ends
    /// where the loop could not foresee it (a stop token, a caller stop, a cancel) waits for its
    /// discarded look-ahead before returning — no forward it enqueued is left in flight to be
    /// charged to the next request's prefill. Every draw past the first is one forward, plus the
    /// discarded look-ahead.
    #[test]
    fn a_pipelined_run_returns_with_no_look_ahead_in_flight() {
        let model = causal();
        let free = off_run(&model, &top_p(20), Pipelining::Off, None)
            .output
            .tokens;
        let mut stopping = top_p(20);
        stopping.stop_tokens = vec![free[7]];
        let ends: [(&str, GenerationConfig, StopAfter); 4] = [
            ("budget", top_p(6), None),
            ("stop token", stopping, None),
            ("stop predicate", top_p(20), Some((5, false))),
            ("cancel", top_p(20), Some((5, true))),
        ];
        for (end, config, stop_after) in ends {
            let target = DispatchesAtForward::new(&model);
            let drained = drained_tokens();
            let run = off_run(&target, &config, Pipelining::Auto, stop_after);
            let forwards = target.dispatched.into_inner().len();
            let draws = run.output.tokens.len()
                + usize::from(run.output.finish_reason == FinishReason::StopToken);
            let discarded = usize::from(end != "budget");
            assert_eq!(run.stats.discarded, discarded, "{end}");
            assert_eq!(
                forwards,
                draws + discarded,
                "{end}: no forward past the end"
            );
            assert_eq!(run.stats.forwards, forwards, "{end}");
            assert_eq!(
                drained_tokens() - drained,
                discarded as u64,
                "{end}: the discarded look-ahead was waited for before returning"
            );
        }
    }

    /// What a [`Scripted`] proposer drafts every step.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Script {
        /// The plain loop's own continuation: always accepted.
        Right,
        /// A token the greedy target never picks: always rejected.
        Wrong,
        /// Nothing: every step is an ordinary single-token draw.
        Empty,
    }

    /// A test proposer of a chosen kind drafting up to `max_drafts` tokens every step by its
    /// [`Script`] against `expected` (the plain loop's tokens after `prompt_len` prompt ids).
    /// Counts its proposals and commits.
    struct Scripted {
        expected: Vec<i32>,
        prompt_len: usize,
        script: Script,
        kind: ProposerKind,
        vocab: i32,
        wants_hidden: bool,
        proposals: usize,
        commits: usize,
    }

    impl Scripted {
        fn new(expected: &[i32], script: Script, kind: ProposerKind, vocab: i32) -> Self {
            Self {
                expected: expected.to_vec(),
                prompt_len: PROMPT.len(),
                script,
                kind,
                vocab,
                wants_hidden: false,
                proposals: 0,
                commits: 0,
            }
        }
    }

    impl<T: SpeculativeTarget + ?Sized> Proposer<T> for Scripted {
        fn kind(&self) -> ProposerKind {
            self.kind
        }
        fn wants_hidden(&self) -> bool {
            self.wants_hidden
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
            self.proposals += 1;
            let at = ctx.history.len() - self.prompt_len;
            let drafts = match self.script {
                Script::Empty => Vec::new(),
                script => self.expected[at.min(self.expected.len())..]
                    .iter()
                    .take(ctx.max_drafts)
                    .map(|&t| match script {
                        Script::Right => t,
                        _ => (t + 1) % self.vocab,
                    })
                    .collect(),
            };
            Ok(Proposal {
                drafts,
                dists: Vec::new(),
            })
        }
        fn commit(&mut self, _: &T, _: i32, _: &[i32], _: Option<&Array>, _: i32) -> Result<()> {
            self.commits += 1;
            Ok(())
        }
    }

    /// One engine run from `prompt` requesting `mode`, with `pipelining`.
    fn monitored_from<T, P>(
        target: &T,
        proposer: &mut P,
        prompt: &[i32],
        config: &GenerationConfig,
        drafts: usize,
        mode: core_llm::Speculative,
        pipelining: Pipelining,
    ) -> SpeculativeRun
    where
        T: SpeculativeTarget,
        P: Proposer<T> + ?Sized,
    {
        let mut ids = Vec::new();
        let run = generate_speculative(
            target,
            proposer,
            SpeculativePrompt::Tokens(prompt),
            config,
            drafts,
            &CancelFlag::new(),
            &mut |e| {
                if let StreamEvent::Token { id, .. } = e {
                    ids.push(id);
                }
            },
            EngineOptions {
                speculative_mode: mode,
                pipelining,
                ..EngineOptions::default()
            },
        )
        .unwrap();
        assert_eq!(ids, run.output.tokens, "streamed ids == returned tokens");
        run
    }

    /// [`monitored_from`] from [`PROMPT`].
    fn monitored<T, P>(
        target: &T,
        proposer: &mut P,
        config: &GenerationConfig,
        drafts: usize,
        mode: core_llm::Speculative,
        pipelining: Pipelining,
    ) -> SpeculativeRun
    where
        T: SpeculativeTarget,
        P: Proposer<T> + ?Sized,
    {
        monitored_from(target, proposer, &PROMPT, config, drafts, mode, pipelining)
    }

    const AUTO: core_llm::Speculative = core_llm::Speculative::Auto;
    const WINDOW: usize = core_llm::ACCEPTANCE_PROBE_VERIFIES as usize;

    /// The token count at which a never-accepting proposer is demoted: the first token plus one
    /// committed token per probe-window verify step.
    const DEMOTED_AT: u64 = 1 + WINDOW as u64;

    /// sc-24446: under `auto`, prompt lookup below its break-even on the pipelinable MLX path is
    /// demoted after the probe window — never proposed to or committed to again — and the rest of
    /// the run is the pipelined plain loop (one handoff step, then every remaining token
    /// pipelined); the output is the plain greedy loop's, the report records the demotion, and
    /// the forward accounting holds — on the causal and the hybrid (checkpoint-ring) targets.
    #[test]
    fn auto_demotes_a_losing_proposer_to_the_pipelined_plain_loop() {
        fn check<T: SpeculativeTarget + crate::decode::Decode>(
            label: &str,
            target: &T,
            vocab: i32,
        ) {
            let config = greedy(40);
            let expected = plain(target, &PROMPT, &config, None).tokens;
            let mut wrong =
                Scripted::new(&expected, Script::Wrong, ProposerKind::PromptLookup, vocab);
            let run = monitored(target, &mut wrong, &config, 4, AUTO, Pipelining::Auto);
            assert_eq!(
                run.output.tokens, expected,
                "{label}: demotion changed the output"
            );
            assert_eq!(
                run.report.speculative_demoted_at,
                Some(DEMOTED_AT),
                "{label}"
            );
            assert_eq!(
                (wrong.proposals, wrong.commits),
                (WINDOW, WINDOW),
                "{label}: the proposer is not driven after the demotion"
            );
            // After the handoff step, every remaining token but the last is a pipelined
            // look-ahead.
            assert_eq!(
                run.stats.pipelined,
                40 - DEMOTED_AT as usize - 1,
                "{label}: the rest is pipelined"
            );
            assert_eq!(run.report.proposer, ProposerKind::PromptLookup, "{label}");
            assert_eq!(run.report.accepted_tokens, 0, "{label}");
            assert_eq!(run.report.proposed_tokens, 4 * WINDOW as u64, "{label}");
            assert_accounting(label, &run, &config);
        }
        check("causal", &causal(), 24);
        check("qwen35", &qwen35(false), 50);
    }

    /// E5 (sc-24446 review): prompt lookup is demoted only where the demoted request would run
    /// MLX's pipelined loop — the regime its break-even was measured in. Where the plain loop is
    /// unpipelined anyway (`Pipelining::Off`, the pipelining switch off, a history-reading
    /// sampler) the same losing lookup runs to the end; an MTP-kind proposer is demoted there
    /// all the same (its cost is its own forwards) and the rest is single-token verify steps.
    #[test]
    fn lookup_is_demoted_only_where_the_plain_loop_is_pipelined() {
        let model = causal();
        let config = greedy(40);
        let mut penalized = penalized(40);
        penalized.sampling.temperature = 0.0;
        type Run<'a> = Box<dyn Fn(&mut Scripted) -> SpeculativeRun + 'a>;
        let unpipelined: [(&str, &GenerationConfig, Run<'_>); 3] = [
            (
                "pipelining off",
                &config,
                Box::new(|p| monitored(&model, p, &config, 4, AUTO, Pipelining::Off)),
            ),
            (
                "switch off",
                &config,
                Box::new(|p| {
                    crate::switches::PIPELINING.scoped(false, || {
                        monitored(&model, p, &config, 4, AUTO, Pipelining::Auto)
                    })
                }),
            ),
            (
                "penalized",
                &penalized,
                Box::new(|p| monitored(&model, p, &penalized, 4, AUTO, Pipelining::Auto)),
            ),
        ];
        for (label, config, run) in unpipelined {
            let expected = plain(&model, &PROMPT, config, None).tokens;
            let mut lookup =
                Scripted::new(&expected, Script::Wrong, ProposerKind::PromptLookup, 24);
            let kept = run(&mut lookup);
            assert_eq!(kept.output.tokens, expected, "{label}");
            assert_eq!(
                kept.report.speculative_demoted_at, None,
                "{label}: lookup kept"
            );
            assert_eq!(
                lookup.proposals as u64,
                kept.report.verify_steps - 1,
                "{label}"
            );
            let mut head = Scripted::new(&expected, Script::Wrong, ProposerKind::Mtp, 24);
            let demoted = run(&mut head);
            assert_eq!(demoted.output.tokens, expected, "{label}");
            assert_eq!(
                demoted.report.speculative_demoted_at,
                Some(DEMOTED_AT),
                "{label}"
            );
            assert_eq!(
                demoted.stats.pipelined, 0,
                "{label}: nothing to pipeline into"
            );
            assert_eq!((head.proposals, head.commits), (WINDOW, WINDOW), "{label}");
            assert_accounting(label, &demoted, config);
        }
    }

    /// A demoted run stops asking the target for hidden rows: the prefill and the probe window's
    /// verify steps do (the proposer wants them), no forward after the demotion does — pipelined
    /// or not.
    #[test]
    fn a_demoted_run_asks_for_no_hidden_rows() {
        let model = qwen35(false);
        let config = greedy(40);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        for pipelining in [Pipelining::Auto, Pipelining::Off] {
            let target = DispatchesAtForward::new(&model);
            let mut wrong = Scripted::new(&expected, Script::Wrong, ProposerKind::Mtp, 50);
            wrong.wants_hidden = true;
            let run = monitored(&target, &mut wrong, &config, 2, AUTO, pipelining);
            assert_eq!(run.output.tokens, expected, "{pipelining:?}");
            assert_eq!(run.report.speculative_demoted_at, Some(DEMOTED_AT));
            let hidden = target.hidden.into_inner();
            let window = 1 + WINDOW;
            assert!(
                hidden[..window].iter().all(|&h| h),
                "{pipelining:?}: {hidden:?}"
            );
            assert!(
                hidden[window..].iter().all(|&h| !h),
                "{pipelining:?}: {hidden:?}"
            );
            assert_eq!(hidden.len() as u64, run.report.target_forwards);
        }
    }

    /// The edges of the handoff: a budget one token past the demotion point (the handoff step
    /// draws the last token, nothing is enqueued after it), and a stop token drawn by the handoff
    /// step itself (the run ends there; on the pipelined path the look-ahead enqueued behind it
    /// is discarded unread) — pipelined (lookup) and unpipelined (MTP kind), each the plain
    /// loop's output.
    #[test]
    fn the_handoff_honours_the_budget_and_a_stop_token_it_draws() {
        let model = causal();
        for (pipelining, kind) in [
            (Pipelining::Auto, ProposerKind::PromptLookup),
            (Pipelining::Off, ProposerKind::Mtp),
        ] {
            let label = format!("{pipelining:?} {kind:?}");
            // Budget = demotion point + 1.
            let config = greedy(DEMOTED_AT as usize + 1);
            let expected = plain(&model, &PROMPT, &config, None).tokens;
            let mut wrong = Scripted::new(&expected, Script::Wrong, kind, 24);
            let run = monitored(&model, &mut wrong, &config, 4, AUTO, pipelining);
            assert_eq!(run.output.tokens, expected, "{label}");
            assert_eq!(run.output.finish_reason, FinishReason::MaxTokens, "{label}");
            assert_eq!(
                run.report.speculative_demoted_at,
                Some(DEMOTED_AT),
                "{label}"
            );
            assert_eq!(run.stats.pipelined, 0, "{label}: nothing past the budget");
            assert_eq!(run.stats.discarded, 0, "{label}");
            assert_accounting(&label, &run, &config);

            // A stop token first drawn by the handoff step (token index `DEMOTED_AT`): a prompt
            // whose plain continuation draws a fresh token there.
            let (prompt, stop) = handoff_stop_prompt(&model);
            let mut stopping = greedy(40);
            stopping.stop_tokens = vec![stop];
            let expected = plain(&model, &prompt, &stopping, None);
            assert_eq!(
                expected.tokens.len(),
                DEMOTED_AT as usize,
                "{label}: fixture premise"
            );
            let mut wrong = Scripted::new(&expected.tokens, Script::Wrong, kind, 24);
            wrong.prompt_len = prompt.len();
            let run = monitored_from(&model, &mut wrong, &prompt, &stopping, 4, AUTO, pipelining);
            assert_eq!(run.output.tokens, expected.tokens, "{label}");
            assert_eq!(run.output.finish_reason, FinishReason::StopToken, "{label}");
            assert_eq!(
                run.report.speculative_demoted_at,
                Some(DEMOTED_AT),
                "{label}"
            );
            let discarded = usize::from(pipelining == Pipelining::Auto);
            assert_eq!(run.stats.discarded, discarded, "{label}: the look-ahead");
            let r = &run.report;
            assert_eq!(
                r.target_forwards,
                r.prefill_forwards + r.verify_steps + r.replay_forwards + r.discarded_forwards,
                "{label}: {r:?}"
            );
        }
    }

    /// A prompt (and the token) whose plain greedy continuation on `model` first draws that token
    /// at index [`DEMOTED_AT`] — the handoff step's draw.
    fn handoff_stop_prompt(model: &CausalLm) -> (Vec<i32>, i32) {
        let config = greedy(DEMOTED_AT as usize + 1);
        for seed in 0..64i32 {
            let prompt: Vec<i32> = (0..12)
                .map(|i| (i * 7 + seed * 5 + i * i * seed) % 24)
                .collect();
            let tokens = plain(model, &prompt, &config, None).tokens;
            let at = DEMOTED_AT as usize;
            if !tokens[..at].contains(&tokens[at]) {
                return (prompt, tokens[at]);
            }
        }
        panic!("no fixture prompt draws a fresh token at the handoff step");
    }

    /// A seeded stochastic run across a demotion draws exactly the plain seeded run's tokens: a
    /// proposer that never drafts makes every verify step the ordinary draw, the demotion hands
    /// the same sampler (and its stream) to the plain loop — pipelined (lookup) or not (MTP kind).
    #[test]
    fn a_seeded_stochastic_run_across_a_demotion_is_the_plain_seeded_run() {
        let model = causal();
        for (pipelining, kind) in [
            (Pipelining::Auto, ProposerKind::PromptLookup),
            (Pipelining::Off, ProposerKind::Mtp),
        ] {
            let config = top_p(40);
            let reference = off_run(&model, &config, pipelining, None).output.tokens;
            let mut empty = Scripted::new(&[], Script::Empty, kind, 24);
            let run = monitored(&model, &mut empty, &config, 4, AUTO, pipelining);
            assert_eq!(run.output.tokens, reference, "{pipelining:?} {kind:?}");
            assert_eq!(run.report.speculative_demoted_at, Some(DEMOTED_AT));
            assert_eq!(run.report.sampler, "device");
        }
    }

    /// A proposer at or above its break-even is never demoted under `auto`: it proposes to the
    /// end, nothing is pipelined, and the output is still the plain loop's.
    #[test]
    fn auto_keeps_a_proposer_that_pays_for_itself() {
        let model = causal();
        let config = greedy(120);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        let mut right = Scripted::new(&expected, Script::Right, ProposerKind::PromptLookup, 24);
        let run = monitored(&model, &mut right, &config, 4, AUTO, Pipelining::Auto);
        assert_eq!(run.output.tokens, expected);
        assert_eq!(run.report.speculative_demoted_at, None);
        assert_eq!(run.stats.pipelined, 0);
        assert!(
            run.report.verify_steps > WINDOW as u64,
            "the run outlasts the probe window: {:?}",
            run.report
        );
        // Every step proposes but one the budget clamped to no drafts.
        assert!(right.proposals as u64 + 1 >= run.report.verify_steps);
        assert_accounting("paying", &run, &config);
    }

    /// An explicit `{proposer, depth}` is the caller's choice: no monitor, so a losing proposer
    /// runs to the end of the request — every step proposes — and nothing is demoted.
    #[test]
    fn an_explicit_proposer_request_is_never_demoted() {
        use core_llm::{Speculative, SpeculativeProposer};
        let model = causal();
        let config = greedy(40);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        for (kind, proposer) in [
            (
                ProposerKind::PromptLookup,
                SpeculativeProposer::PromptLookup,
            ),
            (ProposerKind::Mtp, SpeculativeProposer::Mtp),
        ] {
            let explicit = Speculative::proposer(proposer, 4);
            let mut wrong = Scripted::new(&expected, Script::Wrong, kind, 24);
            let run = monitored(&model, &mut wrong, &config, 4, explicit, Pipelining::Auto);
            assert_eq!(run.output.tokens, expected);
            assert_eq!(run.report.speculative_demoted_at, None, "{kind:?}");
            assert_eq!(run.stats.pipelined, 0);
            // Every step but the budget-clamped last one proposes.
            assert_eq!(wrong.proposals as u64, run.report.verify_steps - 1);
        }
    }

    /// The MTP head (a proposer that wants the target's hidden rows) demotes the same way on the
    /// hybrid: the head is not committed to after the demotion and the rest is the pipelined
    /// plain loop, with the plain greedy output.
    #[test]
    fn a_demoted_mtp_head_hands_the_hybrid_to_the_pipelined_loop() {
        let model = qwen35(true);
        let config = greedy(40);
        let expected = plain(&model, &PROMPT, &config, None).tokens;
        let run = monitored(
            &model,
            &mut MtpProposer::new(),
            &config,
            2,
            AUTO,
            Pipelining::Auto,
        );
        assert_eq!(run.output.tokens, expected);
        assert_eq!(
            run.report.accepted_tokens, 0,
            "fixture premise: the random head never pays"
        );
        assert_eq!(run.report.speculative_demoted_at, Some(DEMOTED_AT));
        assert_eq!(run.stats.pipelined, 40 - DEMOTED_AT as usize - 1);
        assert_eq!(run.report.proposer, ProposerKind::Mtp);
        assert_accounting("mtp", &run, &config);
    }
}
