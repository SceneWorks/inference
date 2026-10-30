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
    ProposerKind, SamplerPath, Speculative, StepObservation,
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
    /// ring, story sc-24435; a quantizing KV cache's exact-rollback window,
    /// [`KvCache::begin_speculation`], sc-20681).
    fn begin(&mut self, cache: &mut C) -> Result<()>;
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

    fn begin(&mut self, _: &mut C) -> Result<()> {
        Ok(())
    }

    fn recover(&mut self, _: &mut C, _: i32) -> Result<Rollback> {
        Err(Error::Unsupported(
            "this target decodes token-at-a-time; it never verifies drafts".into(),
        ))
    }
}

/// Direct rollback by [`KvCache::truncate`] — every softmax KV cache: dropping rejected positions
/// is bookkeeping, no forward. [`begin`](CacheRollback::begin) arms the cache's exact-rollback
/// window ([`KvCache::begin_speculation`]: a no-op on a dense cache, the unquantized step rows on
/// a compressed one, sc-20681) and every recovery truncates — to the current length too, which
/// closes that window — so a quantizing cache ends the step exactly as if only the kept positions
/// had been appended.
#[derive(Clone, Copy, Debug, Default)]
pub struct TruncateRollback;

impl<C: KvCache> CacheRollback<C> for TruncateRollback {
    fn label(&self) -> &'static str {
        "truncate"
    }

    fn begin(&mut self, cache: &mut C) -> Result<()> {
        cache.begin_speculation()
    }

    fn recover(&mut self, cache: &mut C, keep: i32) -> Result<Rollback> {
        cache.truncate(keep.min(cache.offset()))?;
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

    fn begin(&mut self, cache: &mut C) -> Result<()> {
        self.snapshot = Some(cache.clone());
        Ok(())
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

    fn begin(&mut self, cache: &mut Qwen35Cache) -> Result<()> {
        cache.arm_checkpoints(self.max_tokens);
        Ok(())
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
        // A forward that fails part-way leaves a pending boundary capture (sc-24446) in the
        // DeltaNet layers it never reached: drop it, so no later forward on this cache captures
        // a stale boundary.
        let out = (|| -> Result<TargetOutput> {
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
        })();
        if out.is_err() {
            cache.clear_boundary_capture();
        }
        out
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

impl<'c> CacheRollback<Box<dyn KvCache + 'c>> for TruncateRollback {
    fn label(&self) -> &'static str {
        "truncate"
    }

    fn begin(&mut self, cache: &mut Box<dyn KvCache + 'c>) -> Result<()> {
        cache.begin_speculation()
    }

    fn recover(&mut self, cache: &mut Box<dyn KvCache + 'c>, keep: i32) -> Result<Rollback> {
        let keep = keep.min(cache.offset());
        cache.truncate(keep)?;
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

/// A causal decoder over a **caller-chosen KV cache representation** — the compressed
/// group-affine and paged packed caches (epic sc-20669) as well as the dense one — as an engine
/// target, so the compressed cache runs every engine path: the pipelined token-at-a-time loop,
/// prompt-lookup and draft-model verification and the acceptance monitor. A verify step arms the
/// cache's exact-rollback window before the forward ([`KvCache::begin_speculation`], through
/// [`TruncateRollback`]) and truncates to the kept prefix after it, so a quantizing cache rolls
/// back exactly (sc-20681).
pub struct DynCacheTarget<'m>(pub &'m CausalLm);

impl<'m> SpeculativeTarget for DynCacheTarget<'m> {
    type Cache = Box<dyn KvCache + 'm>;
    type Rollback = TruncateRollback;

    fn new_cache(&self) -> Box<dyn KvCache + 'm> {
        Box::new(self.0.new_cache())
    }

    fn cache_len(&self, cache: &Box<dyn KvCache + 'm>) -> i32 {
        cache.offset()
    }

    fn rollback(&self, _: usize) -> TruncateRollback {
        TruncateRollback
    }

    fn forward(
        &self,
        cache: &mut Box<dyn KvCache + 'm>,
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
            LogitsScope::Last => self.0.decode_logits(ids, cache.as_mut(), rope_offset)?,
            LogitsScope::All => self.0.decode_logits_all(ids, cache.as_mut(), rope_offset)?,
        };
        Ok(TargetOutput {
            logits,
            hidden: None,
        })
    }

    fn attention_label(&self) -> &'static str {
        SpeculativeTarget::attention_label(self.0)
    }
}

/// A caller's `&mut dyn KvCache` as an owned [`DynCacheTarget`] cache: every [`KvCache`] method
/// forwards to the borrowed cache, so the engine drives the caller's representation itself (its
/// packed attention, prefix import, speculation window and evidence included).
pub(crate) struct BorrowedCache<'c>(pub(crate) &'c mut dyn KvCache);

impl KvCache for BorrowedCache<'_> {
    fn preflight_packed(&self, query_length: usize, mask: bool) -> crate::primitives::CacheRoute {
        self.0.preflight_packed(query_length, mask)
    }

    fn try_packed_attention(
        &mut self,
        layer: usize,
        query: &Array,
        keys: &Array,
        values: &Array,
        mask: crate::primitives::PackedAttentionMask,
        scale: f32,
        retained_for_sharing: bool,
    ) -> Result<Option<Array>> {
        self.0.try_packed_attention(
            layer,
            query,
            keys,
            values,
            mask,
            scale,
            retained_for_sharing,
        )
    }

    fn import_prefix(&mut self, layers: &[(Array, Array)]) -> Result<bool> {
        self.0.import_prefix(layers)
    }

    fn prepare_dense_fallback(&mut self, operation: &str, reason: &str) -> Result<()> {
        self.0.prepare_dense_fallback(operation, reason)
    }

    fn packed_evidence(&self) -> Option<crate::primitives::PackedCacheEvidence> {
        self.0.packed_evidence()
    }

    fn compressed_storage(&self) -> Result<Option<crate::primitives::CompressedCacheStorage>> {
        self.0.compressed_storage()
    }

    fn compressed_dense_fallback(&self) -> Option<&ContiguousKvCache> {
        self.0.compressed_dense_fallback()
    }

    fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
        self.0.update(layer, keys, values)
    }

    fn offset(&self) -> i32 {
        self.0.offset()
    }

    fn batch_size(&self) -> i32 {
        self.0.batch_size()
    }

    fn num_layers(&self) -> usize {
        self.0.num_layers()
    }

    fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
        self.0.retain_sequences(keep)
    }

    fn truncate(&mut self, len: i32) -> Result<()> {
        self.0.truncate(len)
    }

    fn reset(&mut self) -> Result<()> {
        self.0.reset()
    }

    fn begin_speculation(&mut self) -> Result<()> {
        self.0.begin_speculation()
    }

    fn record_events(&mut self) {
        self.0.record_events()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self.0.as_any_mut()
    }
}

// ---------------------------------------------------------------------------------------------
// The sampler seam.
// ---------------------------------------------------------------------------------------------

/// The engine's token draws — the seam a device sampler plugs in behind. Every draw is recorded,
/// so [`path`](Self::path) is measured rather than inferred from the request.
pub trait TokenSampler {
    /// Resolve the actual prefilled history boundary afresh for every request.
    fn begin_generation(&mut self, prompt_len: usize) -> Result<()> {
        let _ = prompt_len;
        if self.params().presence_penalty_generated_only {
            return Err(Error::Msg(
                "custom sampler does not support generated-only presence penalty".into(),
            ));
        }
        Ok(())
    }

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
    fn begin_generation(&mut self, prompt_len: usize) -> Result<()> {
        self.params.presence_penalty_prompt_len = prompt_len;
        Ok(())
    }

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

    /// Bring the proposer up to date with steps the target ran **without** a proposal — the
    /// plain probe `auto`'s timed acceptance monitor runs before the first proposal (sc-24446,
    /// [`AcceptanceMonitor::probing`]): `tokens` were fed at RoPE positions `position..`, each
    /// paired with its predecessor's target hidden row in `preceding_hidden`
    /// (`[1, tokens.len(), hidden]`, when [`wants_hidden`](Self::wants_hidden)). Called once,
    /// before the next [`propose`](Self::propose). The default does nothing — right for a
    /// proposer whose state is the history alone (n-gram) or that [`commit`](Self::commit)
    /// advances on a step without a proposal (`draft_model`, which `auto` never resolves to).
    fn catch_up(
        &mut self,
        _target: &T,
        _tokens: &[i32],
        _preceding_hidden: Option<&Array>,
        _position: i32,
    ) -> Result<()> {
        Ok(())
    }
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
    /// Whether the caller has delivered output to its consumer yet. A streaming caller can hold
    /// the first emitted tokens back — a thinking marker the segmenter strips, a partial stop
    /// string, a detokenizer's incomplete character — so its time to first token is that of the
    /// first token it *delivers*, not the engine's token 0. Until this reports `true` (for at
    /// most [`FIRST_DELIVERY_HOLD_STEPS`] tokens) the pipelined loop reads every token back
    /// before it enqueues the next step, so the delivered token never waits behind a look-ahead
    /// step's host-side build and dispatch (sc-24446). `None`: every emitted token is delivered.
    pub delivered: Option<&'a dyn Fn() -> bool>,
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

/// The most tokens the pipelined loop reads back unpipelined while the caller has delivered
/// nothing ([`EngineOptions::delivered`]). Holding trades about one step's dispatch overlap per
/// token for the first delivered token's latency, which a look-ahead enqueued ahead of its read
/// raises by up to one step's host-side dispatch; past a few tokens the lost overlap outweighs it,
/// and a run whose output stays held for long (a tool call buffered whole) is pipelined again.
pub const FIRST_DELIVERY_HOLD_STEPS: usize = 8;

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

    /// A run that did not go through the engine — the campaign receipt producer's observed plain
    /// decode (epic sc-20669), which measures the plain loop on purpose: its output with no
    /// speculation counters, no timer, and an empty report the provider does not publish.
    pub(crate) fn unmeasured(output: GenerationOutput, committed_cache_len: i32) -> Self {
        Self {
            output,
            stats: SpeculativeStats::default(),
            report: DecodeReport::default(),
            committed_cache_len,
            timer: None,
        }
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
/// The run's first token is read back and emitted before step 1 is built or enqueued, so its
/// delivery waits for the prefill and its own draw only — never for step 1's graph build and
/// dispatch, nor its forward (sc-24446). Pipelining starts at step 2 (enqueued before step 1 is
/// read back) — or, when the caller holds its first tokens back
/// ([`EngineOptions::delivered`]), after the first token it delivers, so that token's read-back
/// never waits behind a look-ahead's dispatch either.
///
/// A **speculative** step (any proposer) is not pipelined, even one whose proposer found nothing
/// to draft: the proposer must read the committed token on the host — the n-gram context, the MTP
/// head's hidden row — before it can draft the next step, and the verify decision reads the drafts'
/// rows, so there is no next step to enqueue before this one is read back. A penalized or
/// constrained run is not pipelined either: its draws read the host history / grammar that the
/// unread token would advance.
///
/// ## Demotion (sc-24446)
/// Under `auto` ([`EngineOptions::speculative_mode`]), timed by this thread's
/// [`decode_clock`](core_llm::decode_clock) (the wall clock), the run first takes a short plain
/// probe — [`PLAIN_PROBE_MAX_STEPS`](core_llm::PLAIN_PROBE_MAX_STEPS) steps without a proposal,
/// the proposer caught up on them ([`Proposer::catch_up`]) — then every window of
/// [`ACCEPTANCE_PROBE_VERIFIES`](core_llm::ACCEPTANCE_PROBE_VERIFIES) verify steps decides
/// whether the proposer is paying for itself: on the window's measured gain against the
/// request's own plain-step cost, or its static break-even where timing cannot decide (and, on
/// the pipelined path, which the probe cannot time, on either). At the first losing window the
/// run is **demoted**: no further step proposes, asks for hidden rows or commits to the proposer,
/// and — where the draws allow pipelining — the rest of the run is handed to the pipelined
/// token-at-a-time loop (otherwise it continues as unpipelined single-token verify steps). The
/// output is unchanged: a step without drafts is the plain loop's draw.
/// [`DecodeReport::speculative_demoted_at`] and [`DecodeReport::speculative_monitor`] record where
/// and on what.
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
        delivered,
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
            core_llm::report_generation_phase(core_llm::GenerationPhase::Prefilling);
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
    sampler.begin_generation(history.len())?;
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
        core_llm::report_generation_phase(core_llm::GenerationPhase::Generating);
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
            delivered,
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
    core_llm::report_generation_phase(core_llm::GenerationPhase::Generating);
    on_event(StreamEvent::Token { id: first, step: 0 });
    generated.push(first);
    history.push(first);
    let mut cur = first;
    if should_stop.is_some_and(|stop| stop()) {
        finish = FinishReason::Stopped;
    }

    // ---- Speculative steps. ----
    // `auto`'s acceptance monitor (sc-24446): a timed one first runs a plain probe (no proposal;
    // the steps' hidden rows kept so the proposer catches up before its first proposal), then
    // judges each window on the request's own measured costs. Once demoted, no step proposes,
    // asks for hidden rows or commits to the proposer — the rest of the run is token-at-a-time,
    // handed to the pipelined loop when the draws allow it. A step is timed from the top of the
    // loop to the end of the proposer commit (the decision's read-back forces its forward);
    // emitting its tokens is outside.
    let clock = core_llm::decode_clock();
    let mut monitor = AcceptanceMonitor::for_request(
        speculative_mode,
        kind,
        u32::try_from(width).unwrap_or(u32::MAX),
        if can_pipeline {
            PlainDecode::MlxPipelined
        } else {
            PlainDecode::MlxUnpipelined
        },
        clock.is_some(),
    );
    let mut probed: Vec<(i32, Option<Array>)> = Vec::new();
    let mut probe_position = 0i32;
    let mut demoted = false;
    'outer: while generated.len() < config.max_new_tokens && finish != FinishReason::Stopped {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }
        let probing = monitor.as_ref().is_some_and(AcceptanceMonitor::probing);
        if !probing && !probed.is_empty() {
            catch_up(target, &mut *proposer, &mut probed, probe_position)?;
        }
        let started = monitor
            .as_ref()
            .and(clock.as_ref())
            .map(|clock| clock.now());
        let remaining = config.max_new_tokens - generated.len();
        let k = if demoted || probing {
            0
        } else {
            width.min(remaining.saturating_sub(1))
        };
        let wants_hidden = wants_hidden && !demoted;
        let base = target.cache_len(cache);
        let position = base + position_delta;
        if probing {
            if probed.is_empty() {
                probe_position = position;
            }
            probed.push((cur, previous_hidden.clone()));
        }
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
            rollback.begin(cache)?;
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
        let elapsed = started
            .zip(clock.as_ref())
            .map(|(start, clock)| clock.now().saturating_sub(start));
        let demote_now = monitor.as_mut().is_some_and(|m| {
            m.observe_step(StepObservation {
                accepted,
                drafts: num_drafts,
                elapsed,
            })
        });

        // 5. Commit through the shared event path.
        for &token in &committed {
            if config.stop_tokens.contains(&token) {
                finish = FinishReason::StopToken;
                break 'outer;
            }
            if let Some(c) = constraint.as_mut() {
                c.accept(token);
            }
            core_llm::report_generation_phase(core_llm::GenerationPhase::Generating);
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
                    delivered,
                    cancel,
                    on_event,
                )?;
                break;
            }
        }
    }
    stats.monitor = monitor.as_ref().and_then(AcceptanceMonitor::last_decision);
    // A window that demotes on the step that also ends the run (its budget, a stop token, the
    // caller's stop, a cancel) leaves the loop before the demotion is recorded above: the run was
    // still demoted there, so the report's `speculative_demoted_at` and its monitor decision
    // agree (sc-24446).
    if stats.demoted_at.is_none() && stats.monitor.is_some_and(|d| d.demoted) {
        stats.demoted_at = Some(generated.len());
    }

    Ok(finished(
        generated, finish, stats, sampler, timer, start_len, on_event,
    ))
}

/// Hand the plain probe's steps to the proposer ([`Proposer::catch_up`]): each probed token with
/// its predecessor's hidden row (when the proposer asked for rows), fed from `position`.
fn catch_up<T, P>(
    target: &T,
    proposer: &mut P,
    probed: &mut Vec<(i32, Option<Array>)>,
    position: i32,
) -> Result<()>
where
    T: SpeculativeTarget + ?Sized,
    P: Proposer<T> + ?Sized,
{
    let tokens: Vec<i32> = probed.iter().map(|(token, _)| *token).collect();
    let rows: Option<Vec<&Array>> = probed.iter().map(|(_, row)| row.as_ref()).collect();
    let preceding = match rows {
        Some(rows) if !rows.is_empty() => Some(mlx_rs::ops::concatenate_axis(&rows, 1)?),
        _ => None,
    };
    proposer.catch_up(target, &tokens, preceding.as_ref(), position)?;
    probed.clear();
    Ok(())
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
/// commits this one. `pending` itself — and every token drawn without a look-ahead — is handed to
/// the device first (sc-24446): otherwise its read-back, queued behind step `t + 1`, waits for
/// that whole forward. The run's first token (an empty `generated`) — and every token until the
/// caller has delivered one (`delivered`, at most [`FIRST_DELIVERY_HOLD_STEPS`] of them) — is read
/// back before the step after it is built at all, so the first delivered token never waits for a
/// look-ahead's host-side graph build and dispatch. The
/// look-ahead is never enqueued past the budget, and a
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
    delivered: Option<&dyn Fn() -> bool>,
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
    // Until the caller has delivered its first token, every token is read back and emitted
    // before anything is enqueued behind it: the look-ahead's graph build and dispatch are host
    // time spent ahead of the read, and they would add to the time to the first *delivered*
    // token (sc-24446) — the run's token 0, or a later one when the caller holds the first
    // tokens back ([`EngineOptions::delivered`]; a thinking model's opening marker). The next
    // step is then enqueued and dispatched as soon as the token is out, and once the caller has
    // delivered (or after [`FIRST_DELIVERY_HOLD_STEPS`] held tokens) every step overlaps the
    // read of the one before it.
    let holding = |generated: &[i32]| {
        generated.is_empty()
            || (generated.len() < FIRST_DELIVERY_HOLD_STEPS
                && delivered.is_some_and(|delivered| !delivered()))
    };
    loop {
        // Enqueue step t + 1 on step t's unread token, then read step t back while the device
        // runs it. Never past the budget; a host draw is already read back, so it waits.
        let ahead = if !holding(generated)
            && pending.is_device()
            && generated.len() + 1 < config.max_new_tokens
        {
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
            core_llm::report_generation_phase(core_llm::GenerationPhase::Generating);
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
                // An end the loop could not foresee (a stop token, a caller stop, a cancel): the
                // look-ahead is already on the device, enqueued but never read back or emitted.
                // Wait for it here so its device time lands in this request rather than in the
                // next request's prefill (sc-24446); the cache row it wrote is past
                // `committed_cache_len`.
                stats.discarded += 1;
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
                let next = sampler.sample(&out.logits, history, None)?;
                // Handed to the device now, so its read-back never waits for the step enqueued
                // behind it on the next iteration.
                if let SampledToken::Device(id) = &next {
                    dispatch_token(id)?;
                }
                next
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
        speculative_monitor: stats.monitor,
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
pub(crate) mod tests;
