//! The cross-turn prefix cache (epic 7153 story 7168; epic sc-24432 story sc-24437).
//!
//! Requests routinely share a leading run of tokens — a common system prompt, a few-shot preamble,
//! the growing history of a multi-turn chat. A causal decoder's state after position `i` depends
//! only on tokens `0..=i`, so that shared run's state is **identical** across the requests; the
//! [`PrefixCache`] keeps it and a later request restores it instead of re-prefilling it.
//!
//! **One** cache type serves every MLX decoder family (E8). The policy — which stored prefix the
//! prompt extends furthest, a byte budget with least-recently-used eviction — is the backend-neutral
//! [`core_llm::PrefixStore`]; this module owns the MLX state it points at ([`PrefixEntry`]):
//!
//! * a softmax decoder ([`CausalLm`]) stores its whole [`ContiguousKvCache`] after a request, keyed
//!   by the tokens it holds, and any later prompt sharing a leading run reuses that run — the cache
//!   is cloned (MLX arrays are refcounted; the first KV write copies a shared block instead of
//!   donating it, so the stored entry is never written) and truncated to the shared length;
//! * the Qwen3.5/3.6 hybrid ([`Qwen35Model`](crate::models::Qwen35Model)) cannot truncate its
//!   DeltaNet recurrence, so its entry is the **whole** cache snapshotted at a prefill boundary —
//!   the attention KV by offset plus every linear layer's conv tail and recurrent state — reusable
//!   only by a prompt that extends exactly those tokens ([`PrefixReuse::WholeEntry`]). The provider
//!   snapshots at the end of the rendered conversation before its generation prompt, the point a
//!   next chat turn extends. When the request ran the MTP head its state at the boundary rides
//!   along ([`MtpBoundary`]), so an MTP request can resume its warm-up there.
//!
//! Reuse is exact: the restored state is what a prefill of those tokens computed, attended over by
//! the same kernels. [`prefill_with_prefix`] is the seam the provider's text route and the tests
//! drive: restore the longest reusable prefix, prefill only the rest (snapshotting at the
//! boundary), and hand back a cache the engine continues from; [`PrefixCache::store`] keeps the
//! request's state afterwards. [`generate_cached`] is the older single-sequence loop on the same
//! cache.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use mlx_rs::Array;

use core_llm::prefix::{PrefixId, PrefixIndex};
use core_llm::{PrefixReuse, PrefixStore};

use crate::decode::cancel::CancelFlag;
use crate::decode::engine::{LogitsScope, SpeculativeRun, SpeculativeTarget};
use crate::decode::proposers::MtpBoundary;
use crate::decode::stream::{
    decode_loop, default_seed, observe_cache_events, observe_packed_evidence, ConstraintMask,
    GenerationConfig, GenerationOutput, ObservedCache, StreamEvent,
};
use crate::error::{Error, Result};
use crate::models::{CausalLm, Qwen35Cache};
use crate::primitives::input_ids;
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache};
use crate::primitives::sampler::SplitMix64;
use crate::primitives::{
    token_digest, CacheRoute, CompiledKernelHandle, PackedPagePool, PagedCacheIdentity,
    PagedCacheSnapshot, PagedModelKey, PagedPackedKvCache,
};

pub use core_llm::PrefixStats;

/// One stored prefix's MLX state.
#[derive(Clone, Debug)]
pub enum PrefixEntry {
    /// A softmax decoder's KV cache after a request, reused by any leading run.
    Kv(ContiguousKvCache),
    /// The Qwen3.5/3.6 hybrid's whole cache at a prefill boundary, plus the MTP head's state there
    /// when the request that stored it ran the head.
    Hybrid {
        /// The target cache (attention KV and every linear layer's recurrent state).
        cache: Qwen35Cache,
        /// The MTP head at the same boundary.
        mtp: Option<MtpBoundary>,
    },
}

impl PrefixEntry {
    /// Positions the stored state covers.
    fn positions(&self) -> usize {
        match self {
            PrefixEntry::Kv(c) => c.offset().max(0) as usize,
            PrefixEntry::Hybrid { cache, .. } => cache.offset().max(0) as usize,
        }
    }

    /// Bytes the stored arrays hold.
    fn bytes(&self) -> u64 {
        match self {
            PrefixEntry::Kv(c) => c.bytes(),
            PrefixEntry::Hybrid { cache, mtp } => {
                cache.bytes() + mtp.as_ref().map_or(0, MtpBoundary::bytes)
            }
        }
    }

    /// Whether a request can resume from this entry: one running the MTP head needs the head's
    /// state at the boundary too.
    fn serves(&self, mtp: bool) -> bool {
        match self {
            PrefixEntry::Kv(_) => !mtp,
            PrefixEntry::Hybrid { mtp: head, .. } => !mtp || head.is_some(),
        }
    }
}

/// A decode cache the prefix cache can hold and restore.
pub trait PrefixSnapshot: Clone + Sized {
    /// How a stored copy may be reused.
    const REUSE: PrefixReuse;
    /// Restore a cache holding the entry's first `len` positions (`len <=` its positions).
    fn restore(entry: &PrefixEntry, len: usize) -> Result<Self>;
    /// Wrap this cache (and the MTP head's state at the same boundary) as an entry.
    fn into_entry(self, mtp: Option<MtpBoundary>) -> PrefixEntry;
    /// Positions held.
    fn positions(&self) -> usize;
    /// This cache cut back to its first `len` positions — the committed length a finished run
    /// reports ([`SpeculativeRun::committed_cache_len`]). Only an [`PrefixReuse::AnyPrefix`] cache
    /// can be cut; a recurrent state holding any other length is an error.
    fn committed(self, len: usize) -> Result<Self>;
    /// Ask the next forward — a prefill running past `len` — to keep what
    /// [`at_boundary`](Self::at_boundary)`(len)` needs (sc-24446): a recurrent state exists only
    /// where it was taken. A cache that can be cut at any prefix afterwards needs nothing.
    fn capture_boundary(&mut self, len: usize);
    /// A copy of this cache as it was after its first `len` positions, below its length — the
    /// prefix cache's boundary snapshot taken after one prefill forward ran past it. Never shares
    /// a buffer the live cache goes on to write.
    fn at_boundary(&self, len: usize) -> Result<Self>;
}

impl PrefixSnapshot for ContiguousKvCache {
    const REUSE: PrefixReuse = PrefixReuse::AnyPrefix;

    fn restore(entry: &PrefixEntry, len: usize) -> Result<Self> {
        match entry {
            PrefixEntry::Kv(stored) => {
                let mut cache = stored.clone();
                cache.truncate(len as i32)?;
                Ok(cache)
            }
            PrefixEntry::Hybrid { .. } => Err(Error::Msg(
                "prefix cache: a hybrid entry cannot seed a softmax KV cache".into(),
            )),
        }
    }

    fn into_entry(mut self, _mtp: Option<MtpBoundary>) -> PrefixEntry {
        self.clear_events();
        PrefixEntry::Kv(self)
    }

    fn positions(&self) -> usize {
        self.offset().max(0) as usize
    }

    fn committed(mut self, len: usize) -> Result<Self> {
        if self.positions() > len {
            self.truncate(len as i32)?;
        }
        Ok(self)
    }

    fn capture_boundary(&mut self, _len: usize) {}
    fn at_boundary(&self, len: usize) -> Result<Self> {
        self.prefix(len as i32)
    }
}

impl PrefixSnapshot for Qwen35Cache {
    const REUSE: PrefixReuse = PrefixReuse::WholeEntry;

    fn restore(entry: &PrefixEntry, len: usize) -> Result<Self> {
        match entry {
            PrefixEntry::Hybrid { cache, .. } if cache.offset().max(0) as usize == len => {
                Ok(cache.clone())
            }
            PrefixEntry::Hybrid { cache, .. } => Err(Error::Msg(format!(
                "prefix cache: a recurrent state holds exactly {} positions, not {len}",
                cache.offset()
            ))),
            PrefixEntry::Kv(_) => Err(Error::Msg(
                "prefix cache: a softmax KV entry cannot seed the hybrid cache".into(),
            )),
        }
    }

    fn into_entry(mut self, mtp: Option<MtpBoundary>) -> PrefixEntry {
        // An entry is the state at its boundary alone: no checkpoint window rides along into the
        // store (charged, and handed to every restore, as if it were live state).
        self.discard_checkpoints();
        self.clear_kv_events();
        PrefixEntry::Hybrid { cache: self, mtp }
    }

    fn positions(&self) -> usize {
        self.offset().max(0) as usize
    }

    fn committed(self, len: usize) -> Result<Self> {
        if self.positions() == len {
            Ok(self)
        } else {
            Err(Error::Msg(format!(
                "prefix cache: a recurrent state cannot be cut from {} to {len} positions",
                self.offset()
            )))
        }
    }

    fn capture_boundary(&mut self, len: usize) {
        Qwen35Cache::capture_boundary(self, len as i32);
    }
    fn at_boundary(&self, len: usize) -> Result<Self> {
        Qwen35Cache::at_boundary(self, len as i32)
    }
}

/// A restored prefix: a cache holding the prompt's first `reused` positions and, for an MTP
/// request, the head's state at the same boundary.
pub struct Restored<C> {
    /// The cache, positioned at `reused`.
    pub cache: C,
    /// Leading prompt positions it holds.
    pub reused: usize,
    /// The MTP head's state at `reused` (only when the lookup asked for it).
    pub mtp: Option<MtpBoundary>,
}

/// A bounded, least-recently-used cross-turn prefix cache for one loaded model (see the module
/// docs). Resident bytes never exceed the budget; a request short of memory evicts entries first
/// ([`reclaim_for`](Self::reclaim_for)).
pub struct PrefixCache {
    store: PrefixStore<PrefixEntry>,
}

impl PrefixCache {
    /// A cache holding at most `budget_bytes` of stored state (`0` stores nothing).
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            store: PrefixStore::new(budget_bytes),
        }
    }

    /// Cumulative reuse accounting since construction.
    pub fn stats(&self) -> PrefixStats {
        self.store.stats()
    }

    /// Entries held.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// The byte budget.
    pub fn budget_bytes(&self) -> u64 {
        self.store.budget_bytes()
    }

    /// Bytes held, always `<= budget_bytes`.
    pub fn resident_bytes(&self) -> u64 {
        self.store.resident_bytes()
    }

    /// The held entries' token keys, least-recently-used first.
    pub fn keys(&self) -> Vec<Vec<i32>> {
        self.store.keys().into_iter().map(<[i32]>::to_vec).collect()
    }

    /// Evict least-recently-used entries so a request needing `required` bytes fits when
    /// `available` are free; returns the availability to admit against (see
    /// [`PrefixStore::reclaim_for`]).
    pub fn reclaim_for(&mut self, required: u64, available: u64) -> u64 {
        self.store.reclaim_for(required, available)
    }

    /// [`reclaim_for`](Self::reclaim_for) that never evicts the entry the last
    /// [`restore`](Self::restore) hit (the most recently used), whose state the request reuses.
    pub fn reclaim_for_keeping_latest(&mut self, required: u64, available: u64) -> u64 {
        self.store.reclaim_for_keeping_latest(required, available)
    }

    /// Request admission with the cache (E7): evict so the request and the snapshot it would
    /// leave behind (`snapshot_bytes`, an upper bound) fit, or run it without keeping one (see
    /// [`PrefixStore::admit`]).
    pub fn admit(
        &mut self,
        required: u64,
        snapshot_bytes: u64,
        available: u64,
    ) -> core_llm::PrefixAdmission {
        self.store.admit(required, snapshot_bytes, available)
    }

    /// Restore the longest prefix of `prompt` this cache can serve into a fresh `C`, or `None` (a
    /// miss). `mtp`: the request runs the MTP head, so only entries carrying its state qualify.
    /// The reuse is clamped to the positions the stored state really holds — defence in depth
    /// against an entry whose key over-states its state (the budget-finish off-by-one of
    /// sc-12455) — and the stats count what was granted.
    pub fn restore<C: PrefixSnapshot>(
        &mut self,
        prompt: &[i32],
        mtp: bool,
    ) -> Result<Option<Restored<C>>> {
        let Some(hit) = self
            .store
            .lookup(prompt, |e| e.serves(mtp).then(|| e.positions()))
        else {
            return Ok(None);
        };
        let reused = hit.reused;
        let cache = C::restore(hit.state, reused)?;
        let mtp = match hit.state {
            PrefixEntry::Hybrid {
                mtp: Some(head), ..
            } if mtp => Some(head.clone()),
            _ => None,
        };
        Ok(Some(Restored { cache, reused, mtp }))
    }

    /// Keep a request's state for later requests. A [`PrefixReuse::AnyPrefix`] cache is stored
    /// whole after the run, keyed by the tokens it holds — `prompt + generated`, cut to the cache
    /// length (a budget or host-stop finish never feeds the last generated token, sc-12455); a
    /// [`PrefixReuse::WholeEntry`] cache stores only the boundary snapshot the prefill took, with
    /// the MTP head's state there when the request captured one. An entry larger than the budget
    /// is not kept.
    pub fn store<C: PrefixSnapshot>(
        &mut self,
        prompt: &[i32],
        generated: &[i32],
        cache: C,
        boundary: Option<Boundary<C>>,
        mtp: Option<MtpBoundary>,
    ) {
        match C::REUSE {
            PrefixReuse::AnyPrefix => {
                let mut tokens = prompt.to_vec();
                tokens.extend_from_slice(generated);
                tokens.truncate(cache.positions());
                self.insert(tokens, cache.into_entry(None));
            }
            PrefixReuse::WholeEntry => {
                if let Some(Boundary { len, cache }) = boundary {
                    let mtp = mtp.filter(|m| m.len() == len);
                    self.insert(prompt[..len].to_vec(), cache.into_entry(mtp));
                }
            }
        }
    }

    /// [`store`](Self::store) the cache a finished engine `run` left, cut to the run's committed
    /// length first: a pipelined run that ended with a discarded look-ahead (sc-24439) holds a row
    /// past it, which an entry must never carry. A [`PrefixReuse::WholeEntry`] cache keeps only
    /// the prefill's boundary snapshot, so the run's cache is not cut.
    pub fn store_run<C: PrefixSnapshot>(
        &mut self,
        prompt: &[i32],
        run: &SpeculativeRun,
        cache: C,
        boundary: Option<Boundary<C>>,
        mtp: Option<MtpBoundary>,
    ) -> Result<()> {
        let cache = match C::REUSE {
            PrefixReuse::AnyPrefix => cache.committed(run.committed_cache_len.max(0) as usize)?,
            PrefixReuse::WholeEntry => cache,
        };
        self.store(prompt, &run.output.tokens, cache, boundary, mtp);
        Ok(())
    }

    fn insert(&mut self, tokens: Vec<i32>, entry: PrefixEntry) {
        let bytes = entry.bytes();
        let reuse = match entry {
            PrefixEntry::Kv(_) => PrefixReuse::AnyPrefix,
            PrefixEntry::Hybrid { .. } => PrefixReuse::WholeEntry,
        };
        // An entry this insert retires (replaced, superseded or evicted) is released to the
        // system at once: MLX's allocator reuses a freed buffer only for a request of (nearly) the
        // same size and returns its cache to the system only near the device working-set limit,
        // so a retired full-context KV would otherwise stay resident beside the live entry and the
        // next request's cache (sc-20671).
        if self.store.insert(tokens, reuse, bytes, entry).evicted > 0 {
            mlx_rs::memory::clear_cache();
        }
    }
}

/// A snapshot a prefill took at a boundary inside the prompt, for [`PrefixCache::store`].
pub struct Boundary<C> {
    len: usize,
    cache: C,
}

impl<C> Boundary<C> {
    /// The prompt positions the snapshot covers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the snapshot covers no position (never true for a taken snapshot).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A prompt prefilled on top of a restored prefix.
pub struct PrefixPrefill<C> {
    /// The cache, positioned past the prompt.
    pub cache: C,
    /// Last-position logits, `[1, vocab]`.
    pub logits: Array,
    /// The target's hidden rows for the prefilled positions `reused..prompt.len()` (when asked).
    pub hidden: Option<Array>,
    /// Leading prompt positions the lookup restored (the prefill starts past them).
    pub reused: usize,
    /// Prompt tokens the prefill fed through the target — `prompt.len() - reused` when the
    /// restored cache was really prefilled on top of; the provider reports
    /// `prefix_hit_tokens = prompt.len() - fed_tokens`, so the report measures the prefill it ran.
    pub fed_tokens: usize,
    /// The MTP head's state at `reused`, for
    /// [`MtpProposer::resume_from`](super::MtpProposer::resume_from).
    pub mtp: Option<MtpBoundary>,
    /// The snapshot at the requested boundary, for [`PrefixCache::store`].
    pub boundary: Option<Boundary<C>>,
}

/// Prefill `prompt` through `target` on top of the longest prefix `prefix` can restore.
///
/// A hit restores the cache to `reused` positions and runs only `prompt[reused..]`; a miss runs
/// the whole prompt on a fresh cache — in **one** forward either way. `boundary` is a prompt
/// length to snapshot at — honoured for a [`PrefixReuse::WholeEntry`] cache when it falls strictly
/// inside the prefilled span, by capturing the state there inside the forward
/// ([`PrefixSnapshot::capture_boundary`], sc-24446; a softmax cache is stored whole after the run
/// instead). `mtp` asks for the MTP head's state with the restored prefix (and for the target's
/// hidden rows). The cancel flag is checked before and after the forward; a cancelled prefill
/// returns [`Error::Canceled`] and stores nothing.
pub fn prefill_with_prefix<T>(
    target: &T,
    prefix: &mut PrefixCache,
    prompt: &[i32],
    boundary: Option<usize>,
    mtp: bool,
    cancel: &CancelFlag,
) -> Result<PrefixPrefill<T::Cache>>
where
    T: SpeculativeTarget + ?Sized,
    T::Cache: PrefixSnapshot,
{
    if prompt.is_empty() {
        return Err(Error::Msg("prefill_with_prefix: empty prompt".into()));
    }
    let restored = prefix.restore::<T::Cache>(prompt, mtp)?;
    prefill_restored(target, restored, prompt, boundary, mtp, cancel)
}

/// [`prefill_with_prefix`] after the lookup: run `prompt` past what `restored` holds, returning
/// the target's hidden rows for the prefilled positions when `want_hidden`.
pub fn prefill_restored<T>(
    target: &T,
    restored: Option<Restored<T::Cache>>,
    prompt: &[i32],
    boundary: Option<usize>,
    want_hidden: bool,
    cancel: &CancelFlag,
) -> Result<PrefixPrefill<T::Cache>>
where
    T: SpeculativeTarget + ?Sized,
    T::Cache: PrefixSnapshot,
{
    let (mut cache, reused, mtp) = match restored {
        Some(r) => (r.cache, r.reused, r.mtp),
        None => (target.new_cache(), 0, None),
    };
    if reused >= prompt.len() {
        return Err(Error::Msg(format!(
            "prefill_with_prefix: {reused} restored positions leave nothing of a {}-token \
             prompt to prefill",
            prompt.len()
        )));
    }
    let split = boundary
        .filter(|&b| T::Cache::REUSE == PrefixReuse::WholeEntry && b > reused && b < prompt.len());
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // One forward over everything past the restored prefix (sc-24446): a boundary inside it is
    // captured on the way — the recurrent state there kept by the forward, the attention KV
    // copied up to it afterwards — instead of splitting the prefill into two forwards.
    if let Some(b) = split {
        cache.capture_boundary(b);
    }
    let out = target.forward(
        &mut cache,
        &input_ids(&prompt[reused..]),
        reused as i32,
        LogitsScope::Last,
        want_hidden,
    )?;
    if cancel.is_cancelled() {
        // Cancelled while the forward ran: no snapshot of a prefill the request abandons.
        return Err(Error::Canceled);
    }
    // A boundary the forward could not capture (a layer still recording a checkpoint window
    // skips it) costs the cache entry, never the request: the prefill and its output stand, and
    // the boundary is simply not stored.
    let snapshot = split.and_then(|b| {
        cache
            .at_boundary(b)
            .ok()
            .map(|cache| Boundary { len: b, cache })
    });
    Ok(PrefixPrefill {
        cache,
        logits: out.logits,
        hidden: out.hidden,
        reused,
        fed_tokens: prompt.len() - reused,
        mtp,
        boundary: snapshot,
    })
}

/// Like [`generate`](crate::decode::generate), but reusing a shared prefix's KV through
/// `prefix_cache`.
///
/// On each call: restore the longest cached prefix of `prompt_ids` and prefill only the remaining
/// suffix (a miss prefills the whole prompt cold), decode to a stop token / the budget / a
/// mid-stream cancel, then store the request's KV for future reuse. The output is
/// **token-for-token identical** to a cold [`generate`](crate::decode::generate) of the same
/// prompt.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference.
pub fn generate_cached(
    model: &CausalLm,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    prefix_cache: &mut PrefixCache,
) -> Result<GenerationOutput> {
    generate_cached_with(
        model,
        prompt_ids,
        config,
        cancel,
        on_event,
        prefix_cache,
        None,
        None,
    )
}

/// [`generate_cached`] with the same per-step constraint and host-stop seams as
/// [`crate::decode::generate_with`].
///
/// Plain-loop parity reference only: production prefix-cached decoding runs
/// [`prefill_with_prefix`] and the engine, then [`PrefixCache::store_run`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn generate_cached_with(
    model: &CausalLm,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    prefix_cache: &mut PrefixCache,
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<GenerationOutput> {
    generate_cached_with_observer(
        model,
        prompt_ids,
        config,
        cancel,
        on_event,
        prefix_cache,
        constraint,
        should_stop,
        None,
        None,
    )
}

/// Campaign-only observer variant of [`generate_cached_with`].  The observer is attached to the
/// cache-hit prefill and decode that actually execute, rather than to a later single-shot control.
///
/// With a `compressed` arm the request runs on that arm's compressed cache: a prefix hit is
/// imported into it by quantize-on-append (never a reconstruction), and the compressed cache is
/// not stored back into the dense prefix store. A declined import keeps the dense seed and is
/// recorded on the observer as a reasoned `chunked-prefix-import` fallback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_cached_with_observer(
    model: &CausalLm,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    prefix_cache: &mut PrefixCache,
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
    mut observer: Option<&mut dyn crate::campaign::Observer>,
    compressed: Option<&crate::campaign::CompressedKvArm>,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(crate::error::Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(crate::error::Error::Msg(
            "generate_cached: empty prompt".into(),
        ));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));

    // Reuse the longest cached prefix (or start cold), then prefill only the uncached suffix.
    let (mut cache, matched_len) =
        request_cache(model, prompt_ids, prefix_cache, compressed, &mut observer)?;
    // Ownership events are recorded only for an attached campaign observer.
    if observer.is_some() {
        cache.record_events();
    }
    let mut observed_cache = ObservedCache::default();
    let suffix = input_ids(&prompt_ids[matched_len..]);
    let logits = model.decode_logits(&suffix, cache.as_mut(), matched_len as i32)?;
    if let Some(observer) = observer.as_deref_mut() {
        let values = crate::primitives::sampler::counted_host_f32(&logits)?;
        observer.logits("prefill", &values);
        observer.phase("prefill-peak");
    }
    observe_cache_events(cache.as_mut(), &mut observed_cache, &mut observer)?;

    let out = decode_loop(
        model,
        cache.as_mut(),
        logits,
        rng,
        prompt_ids.to_vec(),
        config,
        cancel,
        on_event,
        constraint,
        should_stop,
        &mut observer,
    )?;

    if let Some(observer) = observer.as_deref_mut() {
        observer.phase("decode-steady");
    }
    observe_cache_events(cache.as_mut(), &mut observed_cache, &mut observer)?;
    observe_packed_evidence(cache.as_ref(), &mut observer);
    if let Some(observer) = observer.as_deref_mut() {
        if matches!(out.finish_reason, crate::decode::FinishReason::Cancelled) {
            observer.phase("cancellation-cleanup");
        }
    }

    // Store the sequence whose KV the cache actually holds, so the next shared-prefix request
    // reuses it. On a budget (`MaxTokens`) finish — and on a host-stop (`Stopped`) finish —
    // `decode_loop` breaks *before* feeding the last generated token's KV, so the cache holds one
    // position fewer than `prompt + generated`; truncating to `cache.offset()` keeps the index
    // entry and the stored tensors aligned so a later prompt extending this sequence can never
    // match past the KV (sc-12455).
    let mut full = prompt_ids.to_vec();
    full.extend_from_slice(&out.tokens);
    full.truncate(cache.offset() as usize);
    // Only a dense contiguous cache is stored; a compressed cache never re-enters the dense store.
    // The entry shares the finished cache's buffers (a refcounted clone, no copy), which the
    // reset below then leaves to the entry alone.
    if let Some(dense) = cache.as_any_mut().downcast_ref::<ContiguousKvCache>() {
        let stored = dense.clone().committed(full.len())?;
        prefix_cache.store(&full, &[], stored, None, None);
    }
    cache.reset()?;
    observe_cache_events(cache.as_mut(), &mut observed_cache, &mut observer)?;

    Ok(out)
}

/// The request cache of one prompt-cache lookup and the prompt length it already holds: the
/// longest stored prefix seeded densely (or, with a `compressed` arm, imported into that arm's
/// compressed cache by quantize-on-append), else a cold cache of the request's representation.
fn request_cache(
    model: &CausalLm,
    prompt_ids: &[i32],
    prefix_cache: &mut PrefixCache,
    compressed: Option<&crate::campaign::CompressedKvArm>,
    observer: &mut Option<&mut dyn crate::campaign::Observer>,
) -> Result<(Box<dyn KvCache>, usize)> {
    let seed = prefix_cache
        .restore::<ContiguousKvCache>(prompt_ids, false)?
        .map(|r| (r.cache, r.reused));
    let matched_len = seed.as_ref().map_or(0, |(_, len)| *len);
    let cache: Box<dyn KvCache> = match (seed, compressed) {
        (Some((seed, _)), Some(arm)) => import_compressed_prefix(model, arm, seed, observer)?,
        (None, Some(arm)) => compressed_cache(model, arm, observer).0,
        (Some((seed, _)), None) => Box::new(seed),
        (None, None) => Box::new(model.new_cache()),
    };
    Ok((cache, matched_len))
}

/// The outcome of [`forced_cached_decode`]: the decode, the request cache's packed evidence, and
/// every reasoned fallback its lookup recorded (the reuse itself is in the store's stats).
pub(crate) struct ForcedCachedDecode {
    pub(crate) decode: crate::decode::ForcedDecode,
    pub(crate) packed_evidence: Option<crate::primitives::PackedCacheEvidence>,
    pub(crate) fallbacks: Vec<(String, String)>,
}

/// A fixed-length forced greedy decode served through the prompt cache (SC-20671 multi-turn
/// fixture): `prompt_ids` is looked up and its cache built exactly as
/// [`generate_cached_with_observer`] builds it (dense seed, or a `compressed` import), only the
/// uncached suffix is prefilled, and `tokens` ids are decoded through every stop token
/// ([`crate::decode::forced_greedy_decode`]), teacher-forced on `teacher_forced` when given. The
/// request cache is never stored back.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forced_cached_decode(
    model: &CausalLm,
    prompt_ids: &[i32],
    prefix_cache: &mut PrefixCache,
    compressed: Option<&crate::campaign::CompressedKvArm>,
    tokens: usize,
    stop_tokens: &[i32],
    teacher_forced: Option<&[i32]>,
    score: bool,
) -> Result<ForcedCachedDecode> {
    let mut fallbacks = FallbackCapture::default();
    let (mut cache, reused_prefix_tokens) = {
        let mut observer: Option<&mut dyn crate::campaign::Observer> = Some(&mut fallbacks);
        request_cache(model, prompt_ids, prefix_cache, compressed, &mut observer)?
    };
    let measured = crate::decode::forced_greedy_decode_from(
        model,
        cache.as_mut(),
        prompt_ids,
        reused_prefix_tokens,
        tokens,
        stop_tokens,
        teacher_forced,
        score,
        &mut |_| {},
    );
    let packed_evidence = cache.packed_evidence();
    cache.reset()?;
    Ok(ForcedCachedDecode {
        decode: measured?,
        packed_evidence,
        fallbacks: fallbacks.0,
    })
}

/// Reasoned fallbacks a forced cached decode's lookup recorded.
#[derive(Default)]
struct FallbackCapture(Vec<(String, String)>);

impl crate::campaign::Observer for FallbackCapture {
    fn phase(&mut self, _name: &'static str) {}
    fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
    fn dense_fallback(&mut self, operation: &str, reason: &str) {
        self.0.push((operation.into(), reason.into()));
    }
}

/// Operation name of a reasoned fallback on the compressed prefix-import path.
pub(crate) const COMPRESSED_PREFIX_IMPORT_OPERATION: &str = "chunked-prefix-import";

fn record_prefix_fallback(
    observer: &mut Option<&mut dyn crate::campaign::Observer>,
    operation: &str,
    reason: &str,
) {
    if let Some(observer) = observer.as_deref_mut() {
        observer.dense_fallback(operation, reason);
    }
}

/// The arm's cache and whether its compressed route was selected. A selection the model refuses
/// before any mutation is recorded, exactly as the campaign's compressed decoder records it.
fn compressed_cache(
    model: &CausalLm,
    arm: &crate::campaign::CompressedKvArm,
    observer: &mut Option<&mut dyn crate::campaign::Observer>,
) -> (Box<dyn KvCache>, bool) {
    let selection = arm.select_cache(model);
    let accepted = match selection.route() {
        CacheRoute::DenseFallback { reason } => {
            record_prefix_fallback(observer, "cache-selection", reason);
            false
        }
        CacheRoute::ExperimentalPacked => true,
    };
    (selection.into_cache(), accepted)
}

/// Import a prefix hit into the arm's compressed cache by quantize-on-append. Any refusal keeps the
/// dense seed (the ordinary dense reuse path) and records why.
fn import_compressed_prefix(
    model: &CausalLm,
    arm: &crate::campaign::CompressedKvArm,
    seed: ContiguousKvCache,
    observer: &mut Option<&mut dyn crate::campaign::Observer>,
) -> Result<Box<dyn KvCache>> {
    let (mut cache, accepted) = compressed_cache(model, arm, observer);
    if !accepted {
        return Ok(Box::new(seed));
    }
    let Some(layers) = seed.export()? else {
        record_prefix_fallback(
            observer,
            COMPRESSED_PREFIX_IMPORT_OPERATION,
            "the reused prefix has no exportable dense K/V",
        );
        return Ok(Box::new(seed));
    };
    if !cache.import_prefix(&layers)? {
        record_prefix_fallback(
            observer,
            COMPRESSED_PREFIX_IMPORT_OPERATION,
            "the compressed cache declined the reused prefix (route not live, or geometry/dtype mismatch)",
        );
        return Ok(Box::new(seed));
    }
    Ok(cache)
}

/// Cumulative accounting of a [`PagedPrefixCache`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PagedPrefixStats {
    /// Lookups whose identity matched the store.
    pub lookups: usize,
    /// Lookups that started their sequence on stored pages.
    pub hits: usize,
    /// Positions those hits reused (prefill skipped).
    pub reused_tokens: usize,
    /// Sequences stored.
    pub stored: usize,
    /// Lookups, stores and restores refused for an identity or format mismatch (nothing reused).
    pub refused: usize,
}

/// The outcome of one [`PagedPrefixCache::lookup`].
#[derive(Debug)]
pub enum PagedPrefixLookup {
    /// A new sequence holding the first `tokens` positions of the prompt on shared pages; prefill
    /// the rest from position `tokens`.
    Hit {
        cache: Box<PagedPackedKvCache>,
        tokens: usize,
    },
    /// Nothing stored shares a reusable prefix with the prompt.
    Miss,
    /// The request's cache identity is not the store's: nothing was reused. The reason names the
    /// first differing field.
    Refused(String),
}

struct PagedPrefixEntry {
    tokens: Vec<i32>,
    cache: PagedPackedKvCache,
}

/// A bounded, LRU shared-prefix store over **paged compressed** KV (sc-20681): the paged
/// counterpart of [`PrefixCache`]. An entry holds references to a finished (or still decoding)
/// sequence's pages, not a copy of them; a lookup starts the new sequence on those pages with
/// [`PagedPackedKvCache::fork_prefix`], so prefix sharing costs reference counts, and a sequence
/// that writes into a shared page copies it first (copy-on-write). Evicting or clearing an entry
/// drops its references; a page is freed when its last sequence or entry lets go of it.
///
/// Every entry is keyed by the store's [`PagedCacheIdentity`] — the model, the KV format version,
/// the page layout version and the page geometry — and the token prefix. A lookup, store or
/// restore under any other identity is refused and counted, never served from or into this store.
pub struct PagedPrefixCache {
    model: PagedModelKey,
    identity: PagedCacheIdentity,
    pool: Rc<RefCell<PackedPagePool>>,
    index: PrefixIndex,
    entries: HashMap<PrefixId, PagedPrefixEntry>,
    stats: PagedPrefixStats,
}

impl PagedPrefixCache {
    /// A store of at most `capacity` sequences on `pool`, for caches computed by `model` (its
    /// name and [`crate::models::CausalLm::cache_fingerprint`]).
    pub fn new(pool: Rc<RefCell<PackedPagePool>>, model: PagedModelKey, capacity: usize) -> Self {
        let identity = pool.borrow().identity(&model);
        Self {
            model,
            identity,
            pool,
            index: PrefixIndex::new(capacity),
            entries: HashMap::new(),
            stats: PagedPrefixStats::default(),
        }
    }

    pub fn identity(&self) -> &PagedCacheIdentity {
        &self.identity
    }

    /// The page pool every entry (and every sequence started from one) lives on.
    pub fn pool(&self) -> &Rc<RefCell<PackedPagePool>> {
        &self.pool
    }

    pub fn stats(&self) -> PagedPrefixStats {
        self.stats
    }

    /// Stored sequences.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every stored sequence's cache (for a pool compaction to remap).
    pub fn caches_mut(&mut self) -> impl Iterator<Item = &mut PagedPackedKvCache> {
        self.entries.values_mut().map(|entry| &mut entry.cache)
    }

    /// Distinct pages the entries reference.
    pub fn held_pages(&self) -> usize {
        self.entries
            .values()
            .flat_map(|entry| entry.cache.page_ids().iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    }

    /// Count a request whose run could not use this store (another identity, or no identity).
    pub fn record_refusal(&mut self) {
        self.stats.refused += 1;
    }

    fn refuse(&mut self, identity: &PagedCacheIdentity) -> Option<String> {
        let mismatch = identity.mismatch(&self.identity)?;
        self.stats.refused += 1;
        Some(mismatch)
    }

    /// Start a sequence of `identity` on the longest stored prefix of `prompt`: at most
    /// `prompt.len() - 1` positions (the last prompt token is always prefilled, for its logits),
    /// rounded as [`PagedPackedKvCache::fork_prefix`] rounds. Refused for another identity.
    pub fn lookup(
        &mut self,
        identity: &PagedCacheIdentity,
        prompt: &[i32],
    ) -> Result<PagedPrefixLookup> {
        if let Some(mismatch) = self.refuse(identity) {
            return Ok(PagedPrefixLookup::Refused(mismatch));
        }
        self.stats.lookups += 1;
        let Some(found) = self.index.longest_match(prompt) else {
            return Ok(PagedPrefixLookup::Miss);
        };
        let Some(entry) = self.entries.get(&found.id) else {
            return Ok(PagedPrefixLookup::Miss);
        };
        let wanted = found
            .matched_len
            .min(prompt.len().saturating_sub(1))
            .min(entry.tokens.len());
        let (cache, tokens) = entry.cache.fork_prefix(wanted)?;
        if tokens == 0 {
            return Ok(PagedPrefixLookup::Miss);
        }
        self.stats.hits += 1;
        self.stats.reused_tokens += tokens;
        Ok(PagedPrefixLookup::Hit {
            cache: Box::new(cache),
            tokens,
        })
    }

    /// Store the sequence `cache` holds — whose positions are `tokens` (truncated to what the
    /// cache holds) — by referencing its pages. Returns `false`, storing nothing, when the
    /// identity is not the store's, the cache lives on another pool, or its fused reader faulted
    /// (sc-20688 review): a stored entry outlives the sequence, and every sequence started from
    /// it would inherit the faulted reader state and run on dense gathers.
    pub fn insert(
        &mut self,
        identity: &PagedCacheIdentity,
        tokens: &[i32],
        cache: &PagedPackedKvCache,
    ) -> Result<bool> {
        if self.refuse(identity).is_some() {
            return Ok(false);
        }
        if cache.reader_faulted() {
            return Ok(false);
        }
        if !Rc::ptr_eq(cache.pool(), &self.pool) {
            self.stats.refused += 1;
            return Ok(false);
        }
        let held = (cache.offset().max(0) as usize).min(tokens.len());
        if held == 0 {
            return Ok(false);
        }
        let (entry, held) = cache.fork_prefix(held)?;
        self.store(tokens[..held].to_vec(), entry)?;
        Ok(true)
    }

    fn store(&mut self, tokens: Vec<i32>, cache: PagedPackedKvCache) -> Result<()> {
        let outcome = self.index.insert(tokens.clone());
        let mut released = false;
        for evicted in &outcome.evicted {
            released |= self.entries.remove(evicted).is_some();
        }
        if self.index.contains(outcome.id) {
            released |= self
                .entries
                .insert(outcome.id, PagedPrefixEntry { tokens, cache })
                .is_some();
        }
        self.stats.stored += 1;
        if released {
            // An evicted or replaced entry may have held the pool's top pages.
            self.pool.borrow_mut().trim()?;
        }
        Ok(())
    }

    /// Every stored sequence as `(tokens, snapshot)`, each snapshot bound to its tokens
    /// ([`PagedCacheSnapshot::bound_to`]), for saving across a process restart.
    pub fn snapshots(&self) -> Result<Vec<(Vec<i32>, PagedCacheSnapshot)>> {
        self.entries
            .values()
            .map(|entry| {
                let snapshot = entry.cache.snapshot(&self.model)?.bound_to(&entry.tokens)?;
                Ok((entry.tokens.clone(), snapshot))
            })
            .collect()
    }

    /// Restore a saved sequence into the store, read by `reader`. Refused — and counted — unless
    /// the snapshot carries exactly the store's identity (model and fingerprint, KV format and
    /// page layout versions, geometry) and is bound to exactly these token ids.
    pub fn restore(
        &mut self,
        tokens: Vec<i32>,
        snapshot: &PagedCacheSnapshot,
        reader: CompiledKernelHandle,
    ) -> Result<()> {
        if let Some(mismatch) = self.refuse(snapshot.identity()) {
            return Err(Error::Unsupported(format!(
                "paged prefix restore refused: {mismatch}"
            )));
        }
        if snapshot.tokens() != tokens.len()
            || snapshot.tokens_sha256() != Some(token_digest(&tokens).as_str())
        {
            self.stats.refused += 1;
            return Err(Error::Unsupported(format!(
                "paged prefix restore refused: the snapshot is not bound to these {} token ids \
                 (it holds {} positions)",
                tokens.len(),
                snapshot.tokens()
            )));
        }
        let cache = PagedPackedKvCache::restore(snapshot, self.pool.clone(), reader, &self.model)?;
        self.store(tokens, cache)
    }

    /// Drop every entry (and its page references), returning the pool's freed capacity.
    pub fn clear(&mut self) -> Result<()> {
        self.entries.clear();
        self.index = PrefixIndex::new(self.index.capacity());
        self.pool.borrow_mut().trim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(seq: i32) -> ContiguousKvCache {
        let mut cache = ContiguousKvCache::with_block_tokens(1, 4);
        let t = Array::zeros::<f32>(&[1, 1, seq, 2]).unwrap();
        cache.update(0, &t, &t).unwrap();
        cache
    }

    /// Defence in depth (sc-12455): if an entry's key ever over-states its KV again (the pre-fix
    /// budget-finish state), the restore clamps the reuse to the positions the cache holds instead
    /// of seeding past them.
    #[test]
    fn restore_clamps_the_reuse_to_the_stored_kv() {
        let mut pc = PrefixCache::with_budget(1 << 20);
        // Manufacture the inconsistent state directly: 6 keyed tokens, 5 positions of KV.
        let tokens: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        pc.insert(tokens.clone(), PrefixEntry::Kv(kv(5)));

        // An extending prompt matches all 6 keyed tokens; the restore must clamp to the 5 held.
        let mut prompt = tokens;
        prompt.extend_from_slice(&[7, 8]);
        let r = pc
            .restore::<ContiguousKvCache>(&prompt, false)
            .unwrap()
            .expect("hit");
        assert_eq!(r.reused, 5);
        assert_eq!(r.cache.offset(), 5);
    }

    /// The restored copy is independent of the stored entry: writing past the restored offset
    /// leaves the entry's KV as stored (entries are immutable snapshots).
    #[test]
    fn a_restored_cache_never_writes_the_stored_entry() {
        let mut pc = PrefixCache::with_budget(1 << 20);
        let mut stored = ContiguousKvCache::with_block_tokens(1, 8);
        let ones = Array::ones::<f32>(&[1, 1, 4, 2]).unwrap();
        stored.update(0, &ones, &ones).unwrap();
        pc.insert(vec![1, 2, 3, 4], PrefixEntry::Kv(stored));

        let mut r = pc
            .restore::<ContiguousKvCache>(&[1, 2, 9, 9], false)
            .unwrap()
            .unwrap();
        assert_eq!(r.reused, 2);
        let twos = Array::full::<f32>(&[1, 1, 2, 2], Array::from_f32(2.0)).unwrap();
        r.cache.update(0, &twos, &twos).unwrap();

        let again = pc
            .restore::<ContiguousKvCache>(&[1, 2, 3, 4, 5], false)
            .unwrap()
            .unwrap();
        assert_eq!(again.reused, 4);
        let (k, _) = again.cache.peek(0).unwrap().unwrap();
        let k: Vec<f32> = k.as_slice::<f32>().to_vec();
        assert!(k.iter().all(|&x| x == 1.0), "stored KV was written: {k:?}");
    }

    /// A random-weight Llama of the given geometry (`heads` query and KV heads, vocabulary 64).
    #[cfg(target_os = "macos")]
    fn synthetic_llama(
        hidden_size: i32,
        intermediate_size: i32,
        num_layers: usize,
        heads: i32,
        head_dim: i32,
    ) -> CausalLm {
        use crate::primitives::sampler::{SplitMix64, TokenRng};
        use crate::primitives::Weights;
        let cfg = crate::config::ModelConfig {
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads: heads,
            num_kv_heads: heads,
            head_dim,
            vocab_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling: None,
            tie_word_embeddings: false,
            architecture: crate::config::Architecture::Llama,
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
        let mut rng = SplitMix64::new(0x5c20671);
        let mut randn = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
            Array::from_slice(&data, shape)
        };
        let (h, v, inter) = (cfg.hidden_size, cfg.vocab_size, cfg.intermediate_size);
        let (qd, kvd) = (
            cfg.num_heads * cfg.head_dim,
            cfg.num_kv_heads * cfg.head_dim,
        );
        let ones = || Array::ones::<f32>(&[h]).unwrap();
        let mut m = HashMap::new();
        m.insert("model.embed_tokens.weight".to_string(), randn(&[v, h]));
        m.insert("model.norm.weight".into(), ones());
        m.insert("lm_head.weight".into(), randn(&[v, h]));
        for i in 0..cfg.num_layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            m.insert(p("input_layernorm.weight"), ones());
            m.insert(p("post_attention_layernorm.weight"), ones());
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

    /// sc-20671: prefix reuse at long context holds at most one stored entry plus one request
    /// cache — the "prefix + request" live KV the campaign preflight budgets — and leaves no retired
    /// full-context KV resident. The fit-boundary row (130k tokens, ~14 GiB of KV per copy) was
    /// killed at 68 GiB because the seed-and-hit sequence held ~3 KV copies at once (stored entry,
    /// gathered seed, grown cache, export copy) and left ~4 more in MLX's freed-buffer cache, whose
    /// mismatched sizes are never reused and which is only trimmed near the device working-set
    /// limit.
    ///
    /// The model is KV-dominant (24 layers, 8 × 64 KV heads, 64-wide residual), so one layer's
    /// prefill activations stay a small fraction of the KV, and the 5000-token prompt crosses the
    /// 2048-row prefill blocks and the per-layer checkpoint, as the production path does. Everything
    /// is measured in units of one full-prompt KV: `peak` is MLX's active high-water during one
    /// call, `footprint` the active plus freed-buffer bytes after it (what `phys_footprint` sees).
    #[cfg(target_os = "macos")]
    #[test]
    fn prefix_reuse_holds_one_entry_and_one_request_cache() {
        use mlx_rs::memory;
        let (layers, heads, head_dim) = (24, 8, 64);
        let model = synthetic_llama(64, 64, layers, heads, head_dim);
        let tokens = 5000;
        // K and V, every layer, BF16.
        let kv = (tokens * layers * 2 * (heads * head_dim) as usize * 2) as f64;
        let prompt = (0..tokens)
            .map(|i| (i % 63 + 1) as i32)
            .collect::<Vec<i32>>();
        let config = GenerationConfig {
            max_new_tokens: 1,
            seed: Some(0),
            ..Default::default()
        };
        let mut store = PrefixCache::with_budget(u64::MAX);
        let cancel = CancelFlag::new();
        // Weights are materialized by the first forward; take the baseline after one tiny call on
        // a separate store so only the long prompt's buffers count.
        generate_cached(
            &model,
            &prompt[..4],
            &config,
            &cancel,
            &mut |_| {},
            &mut PrefixCache::with_budget(u64::MAX),
        )
        .unwrap();
        memory::clear_cache();
        let active_base = memory::get_active_memory() as f64;
        for call in ["cold seed", "hit", "second hit", "third hit"] {
            memory::reset_peak_memory();
            generate_cached(&model, &prompt, &config, &cancel, &mut |_| {}, &mut store).unwrap();
            let peak = (memory::get_peak_memory() as f64 - active_base) / kv;
            let footprint = (memory::get_active_memory() as f64
                + memory::get_cache_memory() as f64
                - active_base)
                / kv;
            // A hit holds the stored entry and the request cache grown out of it (2 KV); the cold
            // seed holds one cache plus one layer's prefill activations.
            assert!(
                peak <= 2.25,
                "{call}: active peak {peak:.2} KV; at most the entry plus the request cache fit"
            );
            // Afterwards only the stored entry (one KV plus under a block of padding) and the
            // prefill's reusable activation buffers remain.
            assert!(
                footprint <= 1.5,
                "{call}: {footprint:.2} KV stays resident after the call; only the stored entry \
                 may (a copy or a retired entry is a full KV more)"
            );
        }
        assert_eq!(store.len(), 1);
        assert_eq!(store.stats().hits, 3);
    }

    #[test]
    fn an_entry_past_the_budget_is_not_kept() {
        let entry = kv(4);
        let mut pc = PrefixCache::with_budget(entry.bytes() - 1);
        pc.store(&[1, 2, 3, 4], &[], entry, None, None);
        assert!(pc.is_empty());
        assert_eq!(pc.stats().rejected, 1);
    }
}

/// The prefix cache under the engine on the tiny decoders (story sc-24437): each backend family's
/// second turn restores the shared prefix, prefills only the rest — measured by a token counter at
/// the target seam, not by the cache's own report — and decodes exactly what a cold run decodes.
#[cfg(test)]
mod engine_tests {
    use std::cell::Cell;

    use super::*;
    use crate::decode::engine::tests::{causal, qwen35};
    use crate::decode::engine::{
        generate_speculative, EngineOptions, NoProposer, Proposer, SpeculativePrompt,
        SpeculativeRun, TargetOutput,
    };
    use crate::decode::proposers::MtpProposer;
    use crate::models::Qwen35Model;
    use crate::primitives::sampler::SamplingParams;

    /// A target that counts every token fed through its forward, and can raise a cancel flag from
    /// inside the first forward (a cancel landing mid-prefill).
    struct Counted<'a, T> {
        inner: &'a T,
        fed: Cell<usize>,
        calls: Cell<usize>,
        cancel_in_forward: Option<&'a CancelFlag>,
    }

    impl<'a, T> Counted<'a, T> {
        fn new(inner: &'a T) -> Self {
            Self {
                inner,
                fed: Cell::new(0),
                calls: Cell::new(0),
                cancel_in_forward: None,
            }
        }

        fn take(&self) -> usize {
            self.fed.replace(0)
        }
    }

    impl<T: SpeculativeTarget> SpeculativeTarget for Counted<'_, T> {
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
            self.fed.set(self.fed.get() + ids.shape()[1] as usize);
            self.calls.set(self.calls.get() + 1);
            if let Some(cancel) = self.cancel_in_forward {
                cancel.cancel();
            }
            self.inner
                .forward(cache, ids, rope_offset, scope, want_hidden)
        }
        fn attention_label(&self) -> &'static str {
            self.inner.attention_label()
        }
    }

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            sampling: SamplingParams::default(),
            seed: Some(7),
            stop_tokens: Vec::new(),
        }
    }

    /// A cold engine run of `prompt` (the engine prefills it itself).
    fn cold<T, P>(target: &T, proposer: &mut P, prompt: &[i32], max_new: usize) -> Vec<i32>
    where
        T: SpeculativeTarget,
        P: Proposer<T> + ?Sized,
    {
        generate_speculative(
            target,
            proposer,
            SpeculativePrompt::Tokens(prompt),
            &greedy(max_new),
            2,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions::default(),
        )
        .unwrap()
        .output
        .tokens
    }

    /// One turn through the prefix cache: restore, prefill the rest, decode, store. Returns the
    /// run and how many prompt positions were restored.
    fn turn<T, P>(
        target: &T,
        proposer: &mut P,
        pc: &mut PrefixCache,
        prompt: &[i32],
        boundary: Option<usize>,
        max_new: usize,
    ) -> (SpeculativeRun, usize)
    where
        T: SpeculativeTarget,
        T::Cache: PrefixSnapshot,
        P: Proposer<T> + ?Sized,
    {
        let PrefixPrefill {
            mut cache,
            logits,
            hidden,
            reused,
            boundary,
            ..
        } = prefill_with_prefix(target, pc, prompt, boundary, false, &CancelFlag::new()).unwrap();
        let run = generate_speculative(
            target,
            proposer,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits,
                hidden,
                history: prompt,
                position_delta: 0,
            },
            &greedy(max_new),
            2,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions::default(),
        )
        .unwrap();
        pc.store_run(prompt, &run, cache, boundary, None).unwrap();
        (run, reused)
    }

    /// AC1 on the softmax family: turn 2 extends turn 1's `prompt + reply`; it restores the whole
    /// shared run (turn 1's KV minus the never-fed last reply token), prefills only the rest, and
    /// matches a cold run token for token.
    #[test]
    fn a_causal_second_turn_prefills_only_the_new_tokens() {
        let model = causal();
        let counted = Counted::new(&model);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let p1 = vec![3, 9, 4, 11, 3, 9, 4, 11, 5];
        let (run1, reused1) = turn(&counted, &mut NoProposer, &mut pc, &p1, None, 6);
        assert_eq!(reused1, 0);
        let gen1 = run1.output.tokens;
        counted.take();

        let mut p2 = p1.clone();
        p2.extend_from_slice(&gen1);
        p2.extend_from_slice(&[7, 2, 13]);
        let expected = p1.len() + gen1.len() - 1;
        let prefill =
            prefill_with_prefix(&counted, &mut pc, &p2, None, false, &CancelFlag::new()).unwrap();
        assert_eq!(prefill.reused, expected, "N = turn 1's cached positions");
        assert_eq!(
            counted.take(),
            p2.len() - expected,
            "the prefill fed only the new tokens"
        );
        drop(prefill);

        let (run2, reused2) = turn(&model, &mut NoProposer, &mut pc, &p2, None, 8);
        assert_eq!(reused2, expected);
        assert_eq!(run2.output.tokens, cold(&model, &mut NoProposer, &p2, 8));
    }

    /// AC2: the hybrid restores the recurrent state snapshotted at the boundary, prefills only
    /// past it, and matches a cold run. A third turn restores the same entry after the second
    /// turn's decode continued from it — the entry is an immutable snapshot.
    /// A pipelined run that ends with a discarded look-ahead (sc-24439) stores exactly what the
    /// unpipelined run stores — the committed length, never the look-ahead row — whether it ends
    /// on a stop token (the look-ahead fed the unemitted stop token) or on the caller's stop
    /// predicate (it fed the last emitted token).
    #[test]
    fn a_pipelined_run_stores_only_its_committed_length() {
        use crate::decode::engine::Pipelining;
        let model = causal();
        let prompt = vec![3, 9, 4, 11, 3, 9, 4, 11, 5];
        let free = cold(&model, &mut NoProposer, &prompt, 12);
        let mut stopping = greedy(12);
        // A stop past token 0: the run's first token is read back before any look-ahead is enqueued
        // (sc-24446), so a stop there discards nothing.
        let stop = *free[1..].iter().find(|&&t| t != free[0]).unwrap();
        stopping.stop_tokens = vec![stop];
        let emitted = Cell::new(0usize);
        let after_four = || emitted.get() >= 4;
        type End<'a> = (&'a str, &'a GenerationConfig, Option<&'a dyn Fn() -> bool>);
        let ends: [End<'_>; 2] = [
            ("stop token", &stopping, None),
            ("stop predicate", &greedy(12), Some(&after_four)),
        ];
        for (end, config, should_stop) in ends {
            let mut stored = Vec::new();
            for pipelining in [Pipelining::Off, Pipelining::Auto] {
                emitted.set(0);
                let mut pc = PrefixCache::with_budget(1 << 30);
                let PrefixPrefill {
                    mut cache, logits, ..
                } = prefill_with_prefix(&model, &mut pc, &prompt, None, false, &CancelFlag::new())
                    .unwrap();
                let run = generate_speculative(
                    &model,
                    &mut NoProposer,
                    SpeculativePrompt::Prefilled {
                        cache: &mut cache,
                        logits,
                        hidden: None,
                        history: &prompt,
                        position_delta: 0,
                    },
                    config,
                    0,
                    &CancelFlag::new(),
                    &mut |e| {
                        if matches!(e, StreamEvent::Token { .. }) {
                            emitted.set(emitted.get() + 1);
                        }
                    },
                    EngineOptions {
                        should_stop,
                        pipelining,
                        ..EngineOptions::default()
                    },
                )
                .unwrap();
                let discarded = run.stats.discarded;
                pc.store_run(&prompt, &run, cache, None, None).unwrap();
                // The positions the stored state holds (a key never reveals a row past itself).
                let held = Cell::new(0usize);
                pc.store.lookup(&[], |entry| {
                    held.set(entry.positions());
                    None
                });
                stored.push((pc.keys(), held.get(), discarded));
            }
            let (off, on) = (&stored[0], &stored[1]);
            assert_eq!(
                (off.2, on.2),
                (0, 1),
                "{end}: only the pipelined run discards"
            );
            assert_eq!(on.0, off.0, "{end}: stored key");
            assert_eq!(on.1, off.1, "{end}: stored positions");
        }
    }

    /// sc-24446 (epic E1): on both decoder families, a prefix-cache hit on the boundary entry a
    /// single-forward miss stored decodes exactly the greedy tokens a plain, cache-free run of the
    /// same prompt decodes — over conversations shorter than one Gated DeltaNet chunk and past
    /// it, each continued by two different turns (ids below each fixture's vocabulary `v`). Sensitive to the boundary: a recurrent state
    /// captured one position off changes the hybrid's greedy stream here.
    #[test]
    fn a_hit_after_a_single_forward_miss_decodes_the_plain_run() {
        fn check<T>(model: &T, family: &str, v: i32)
        where
            T: SpeculativeTarget,
            T::Cache: PrefixSnapshot,
        {
            const NEW: usize = 24;
            for conv_len in [5usize, 8, 13, 21, 34, 55, 70, 90] {
                let conversation: Vec<i32> = (0..conv_len as i32)
                    .map(|i| (i * 7 % (v - 4)) + 1)
                    .collect();
                let mut pc = PrefixCache::with_budget(1 << 30);
                let mut p1 = conversation.clone();
                p1.extend_from_slice(&[v - 2, v - 1]);
                let (_, reused1) = turn(model, &mut NoProposer, &mut pc, &p1, Some(conv_len), NEW);
                assert_eq!(reused1, 0, "{family} {conv_len}: a miss");
                for next in [
                    [v - 7, v - 6, v - 5, v - 2, v - 1],
                    [3, 7, v - 3, v - 2, v - 1],
                ] {
                    let mut p = conversation.clone();
                    p.extend_from_slice(&next);
                    let (run, reused) = turn(model, &mut NoProposer, &mut pc, &p, None, NEW);
                    assert!(reused >= conv_len, "{family} {conv_len}: a hit ({reused})");
                    assert_eq!(
                        run.output.tokens,
                        cold(model, &mut NoProposer, &p, NEW),
                        "{family} {conv_len} {next:?}"
                    );
                }
            }
        }
        check(&causal(), "causal", 24);
        check(&qwen35(false), "hybrid", 50);
    }

    #[test]
    fn a_qwen35_second_turn_restores_the_recurrent_state_at_the_boundary() {
        let model = qwen35(false);
        let counted = Counted::new(&model);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let conversation = vec![3, 9, 4, 11, 3, 9, 4, 11];
        let mut p1 = conversation.clone();
        p1.extend_from_slice(&[40, 41]); // a generation prompt past the boundary
        let (_, reused1) = turn(&counted, &mut NoProposer, &mut pc, &p1, Some(8), 5);
        assert_eq!(reused1, 0);
        assert_eq!(pc.keys(), vec![conversation.clone()]);
        counted.take();

        for next in [[20, 21, 22, 40, 41], [30, 31, 32, 40, 41]] {
            let mut p = conversation.clone();
            p.extend_from_slice(&next);
            let prefill =
                prefill_with_prefix(&counted, &mut pc, &p, None, false, &CancelFlag::new())
                    .unwrap();
            assert_eq!(prefill.reused, conversation.len());
            assert_eq!(counted.take(), next.len(), "only past the boundary");
            drop(prefill);
            let (run, reused) = turn(&model, &mut NoProposer, &mut pc, &p, None, 6);
            assert_eq!(reused, conversation.len());
            assert_eq!(run.output.tokens, cold(&model, &mut NoProposer, &p, 6));
        }
        // A prompt that diverges inside the boundary cannot use the recurrent state.
        let mut other = conversation[..5].to_vec();
        other.extend_from_slice(&[1, 2]);
        let restored = pc.restore::<Qwen35Cache>(&other, false).unwrap();
        assert!(restored.is_none());
    }

    /// A hybrid entry is the state at its boundary alone (sc-24435 x sc-24437): a cache stored
    /// while a DeltaNet checkpoint window is open (recording) keeps no window in the store — it
    /// is charged exactly what a cache that never recorded holds — and a restore of it carries
    /// none, so the restored cache can return to no earlier position.
    #[test]
    fn a_stored_hybrid_entry_carries_no_checkpoint_window() {
        let model = qwen35(false);
        let prompt = [3, 9, 4, 11, 3, 9, 4, 11];
        let fwd = |cache: &mut Qwen35Cache, ids: &[i32], at: i32| {
            SpeculativeTarget::forward(
                &model,
                cache,
                &input_ids(ids),
                at,
                LogitsScope::Last,
                false,
            )
            .unwrap();
        };
        let mut plain = model.new_cache();
        fwd(&mut plain, &prompt, 0);
        let mut live = model.new_cache();
        fwd(&mut live, &prompt[..6], 0);
        live.arm_checkpoints(4);
        fwd(&mut live, &prompt[6..], 6);
        assert!(
            live.checkpointed_tokens() > 0,
            "the window is open and recording"
        );
        assert!(
            live.bytes() > plain.bytes(),
            "the window is charged while live"
        );

        let mut pc = PrefixCache::with_budget(1 << 30);
        pc.store(
            &prompt,
            &[],
            live.clone(),
            Some(Boundary {
                len: prompt.len(),
                cache: live,
            }),
            None,
        );
        assert_eq!(
            pc.resident_bytes(),
            plain.bytes(),
            "charged its live state only"
        );
        let mut next = prompt.to_vec();
        next.push(20);
        let restored = pc
            .restore::<Qwen35Cache>(&next, false)
            .unwrap()
            .expect("a hit");
        assert_eq!(restored.reused, prompt.len());
        assert_eq!(restored.cache.checkpointed_tokens(), 0);
        let at = prompt.len() as i32;
        assert_eq!(
            restored.cache.restorable(),
            at..at,
            "no checkpoint window restored"
        );
    }

    /// The boundary snapshot is independent of the live cache it was taken from (MLX arrays are
    /// copy-on-write): that cache goes on to decode — prompt lookup verifying and rolling back on
    /// top of it — and a restore of the stored entry answers exactly as it did before.
    #[test]
    fn a_boundary_snapshot_is_not_written_by_the_decode_that_continues_past_it() {
        let model = qwen35(false);
        let mut prompt = vec![3, 9, 4, 11, 3, 9, 4, 11];
        prompt.extend_from_slice(&[40, 41]);
        let PrefixPrefill {
            mut cache,
            logits,
            boundary,
            ..
        } = prefill_restored(&model, None, &prompt, Some(8), false, &CancelFlag::new()).unwrap();
        let entry = boundary
            .expect("a snapshot at the boundary")
            .cache
            .into_entry(None);
        let probe = |entry: &PrefixEntry| -> Vec<f32> {
            let mut restored = <Qwen35Cache as PrefixSnapshot>::restore(entry, 8).unwrap();
            let out = SpeculativeTarget::forward(
                &model,
                &mut restored,
                &input_ids(&[20, 21]),
                8,
                LogitsScope::Last,
                false,
            )
            .unwrap();
            let logits = out.logits.as_dtype(mlx_rs::Dtype::Float32).unwrap();
            logits.as_slice::<f32>().to_vec()
        };
        let before = probe(&entry);
        generate_speculative(
            &model,
            &mut crate::decode::NgramProposer::default(),
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits,
                hidden: None,
                history: &prompt,
                position_delta: 0,
            },
            &greedy(20),
            3,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions::default(),
        )
        .unwrap();
        assert_eq!(probe(&entry), before);
    }

    /// Speculative decoding on top of a hit: the MTP head resumes from the state stored at the
    /// boundary (its warm-up seeds only past it) and the greedy tokens match a cold MTP run and a
    /// cold plain run.
    #[test]
    fn mtp_resumes_its_warm_up_from_the_stored_boundary() {
        let model = qwen35(true);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let conversation = vec![3, 9, 4, 11, 3, 9, 4, 11];
        let run_mtp = |pc: &mut PrefixCache, prompt: &[i32], boundary: Option<usize>| {
            let PrefixPrefill {
                mut cache,
                logits,
                hidden,
                reused,
                mtp,
                boundary,
                ..
            } = prefill_with_prefix(&model, pc, prompt, boundary, true, &CancelFlag::new())
                .unwrap();
            let mut head = MtpProposer::new()
                .resume_from(mtp)
                .capture_at(boundary.as_ref().map(Boundary::len));
            let run = generate_speculative(
                &model,
                &mut head,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits,
                    hidden,
                    history: prompt,
                    position_delta: 0,
                },
                &greedy(8),
                2,
                &CancelFlag::new(),
                &mut |_| {},
                EngineOptions::default(),
            )
            .unwrap();
            let captured = head.take_captured();
            pc.store(prompt, &[], cache, boundary, captured);
            (run.output.tokens, reused, run.stats)
        };
        let mut p1 = conversation.clone();
        p1.extend_from_slice(&[40, 41]);
        let (tokens1, reused1, _) = run_mtp(&mut pc, &p1, Some(conversation.len()));
        assert_eq!(reused1, 0);
        assert_eq!(tokens1, cold(&model, &mut NoProposer, &p1, 8));

        let mut p2 = conversation.clone();
        p2.extend_from_slice(&[20, 21, 22, 40, 41]);
        let (tokens2, reused2, stats2) = run_mtp(&mut pc, &p2, None);
        assert_eq!(
            reused2,
            conversation.len(),
            "the MTP request hit the boundary"
        );
        let cold_mtp = generate_speculative(
            &model,
            &mut MtpProposer::new(),
            SpeculativePrompt::Tokens(&p2),
            &greedy(8),
            2,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions::default(),
        )
        .unwrap();
        assert_eq!(tokens2, cold_mtp.output.tokens);
        assert_eq!(tokens2, cold(&model, &mut NoProposer, &p2, 8));
        // The resumed head drafts what the cold head drafts: its state at the boundary is the
        // cold warm-up's.
        assert!(stats2.proposed > 0);
        assert_eq!(
            (stats2.proposed, stats2.accepted),
            (cold_mtp.stats.proposed, cold_mtp.stats.accepted)
        );
    }

    /// The head's state resumed at a boundary is the cold warm-up's: seeding only past the
    /// boundary from the captured state yields the same head outputs, row for row, as warming the
    /// whole prompt.
    #[test]
    fn a_resumed_mtp_warm_up_matches_the_cold_warm_up() {
        use crate::decode::engine::seq_rows;
        let model = qwen35(true);
        let prompt = [3, 9, 4, 11, 3, 9, 4, 11, 20, 21];
        let (p, m) = (prompt.len() as i32, 6usize);
        let (hidden, _) = model
            .prefill_hidden_and_last_logits(&input_ids(&prompt), &mut model.new_cache(), 0)
            .unwrap();
        let mut cold = MtpProposer::new().capture_at(Some(m));
        let cold_seed = Proposer::<Qwen35Model>::warm(&mut cold, &model, &prompt, Some(&hidden))
            .unwrap()
            .unwrap();
        let boundary = cold.take_captured().expect("captured at the boundary");
        assert_eq!(boundary.len(), m);
        let mut whole = MtpProposer::new();
        let whole_seed = Proposer::<Qwen35Model>::warm(&mut whole, &model, &prompt, Some(&hidden))
            .unwrap()
            .unwrap();

        let suffix = seq_rows(&hidden, m as i32, p - m as i32).unwrap();
        let mut resumed = MtpProposer::new().resume_from(Some(boundary));
        let resumed_seed =
            Proposer::<Qwen35Model>::warm(&mut resumed, &model, &prompt, Some(&suffix))
                .unwrap()
                .unwrap();
        // Head positions M..P-1 are cold rows M-1..P-2 (the cold seed starts at position 1).
        let rows = p - m as i32;
        let expect = seq_rows(&whole_seed, m as i32 - 1, rows).unwrap();
        let split = seq_rows(&cold_seed, 0, rows).unwrap();
        let diff = |a: &Array, b: &Array| -> f32 {
            let a = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
            let b = b.as_dtype(mlx_rs::Dtype::Float32).unwrap();
            a.as_slice::<f32>()
                .iter()
                .zip(b.as_slice::<f32>())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f32::max)
        };
        assert_eq!(resumed_seed.shape(), expect.shape());
        assert!(diff(&resumed_seed, &expect) < 1e-3, "resumed head diverged");
        assert_eq!(
            split.shape(),
            expect.shape(),
            "the capture split's tail segment"
        );
        assert!(diff(&split, &expect) < 1e-3, "split warm-up diverged");
    }

    /// An entry stored by a request that did not run the MTP head carries no head state, so an
    /// MTP request does not resume from it (it prefills cold) — and still decodes correctly.
    #[test]
    fn an_mtp_request_skips_an_entry_without_head_state() {
        let model = qwen35(true);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let conversation = vec![3, 9, 4, 11, 3, 9];
        let mut p1 = conversation.clone();
        p1.push(40);
        turn(
            &model,
            &mut NoProposer,
            &mut pc,
            &p1,
            Some(conversation.len()),
            4,
        );
        let mut p2 = conversation.clone();
        p2.extend_from_slice(&[20, 40]);
        assert!(pc.restore::<Qwen35Cache>(&p2, true).unwrap().is_none());
        assert!(pc.restore::<Qwen35Cache>(&p2, false).unwrap().is_some());
    }

    /// A cancel landing mid-prefill stops before the next forward, returns the typed cancel and
    /// stores nothing; the next request prefills from the last good entry and decodes correctly.
    #[test]
    fn a_cancel_mid_prefill_stores_nothing() {
        let model: Qwen35Model = qwen35(false);
        let cancel = CancelFlag::new();
        let mut counted = Counted::new(&model);
        counted.cancel_in_forward = Some(&cancel);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let prompt = vec![3, 9, 4, 11, 3, 9, 40];
        let err = prefill_with_prefix(&counted, &mut pc, &prompt, Some(6), false, &cancel)
            .err()
            .expect("cancelled");
        assert!(matches!(err, Error::Canceled), "{err}");
        assert_eq!(
            counted.take(),
            7,
            "one forward, past the boundary (sc-24446)"
        );
        assert!(pc.is_empty());

        let (run, reused) = turn(&model, &mut NoProposer, &mut pc, &prompt, Some(6), 4);
        assert_eq!(reused, 0);
        assert_eq!(run.output.tokens, cold(&model, &mut NoProposer, &prompt, 4));
        assert_eq!(pc.len(), 1);
    }

    /// sc-24446: a forward that fails part-way drops the boundary capture its DeltaNet layers
    /// never consumed, so a later forward on the same cache captures nothing there.
    #[test]
    fn a_failed_forward_leaves_no_stale_boundary_capture() {
        let model = qwen35(false);
        let prompt: Vec<i32> = (0..12).map(|i| (i * 7 % 49) + 1).collect();
        let fwd = |cache: &mut Qwen35Cache| {
            SpeculativeTarget::forward(
                &model,
                cache,
                &input_ids(&prompt),
                0,
                LogitsScope::Last,
                false,
            )
        };
        let mut cache = model.new_cache();
        // A two-token window: the 12-token forward is refused by the first DeltaNet layer.
        cache.arm_checkpoints(2);
        cache.capture_boundary(8);
        assert!(matches!(
            fwd(&mut cache),
            Err(Error::CheckpointWindowFull { .. })
        ));
        assert_eq!(cache.offset(), 0, "the refused forward wrote nothing");
        cache.discard_checkpoints();
        fwd(&mut cache).unwrap();
        assert!(
            cache.at_boundary(8).is_err(),
            "a stale boundary was captured by a later forward"
        );
    }

    /// sc-24446: a boundary the forward cannot capture — a DeltaNet layer still recording a
    /// checkpoint window skips it — costs the prefix-cache entry, never the request: the prefill
    /// returns its output with no boundary snapshot.
    #[test]
    fn an_uncapturable_boundary_skips_the_snapshot_not_the_request() {
        let model = qwen35(false);
        let prompt: Vec<i32> = (0..12).map(|i| (i * 7 % 49) + 1).collect();
        let mut recording = model.new_cache();
        recording.arm_checkpoints(64);
        let pre = prefill_restored(
            &model,
            Some(Restored {
                cache: recording,
                reused: 0,
                mtp: None,
            }),
            &prompt,
            Some(8),
            false,
            &CancelFlag::new(),
        )
        .expect("the request survives an uncapturable boundary");
        assert!(pre.boundary.is_none());
        assert_eq!(pre.fed_tokens, prompt.len());
        assert_eq!(pre.cache.offset(), prompt.len() as i32);
    }

    /// sc-24446 (defect B): a miss with a boundary inside the prompt prefills in **one** forward,
    /// and the snapshot it stores is the one a prefill split at the boundary into two forwards
    /// stores — every DeltaNet state and the attention KV up to the boundary, to the GEMM's
    /// reduction order — and a later hit restoring it decodes the same greedy tokens as a hit on
    /// the split prefill's snapshot. A conversation shorter
    /// than one Gated DeltaNet chunk and one past it.
    #[test]
    fn a_boundary_miss_prefills_in_one_forward_and_stores_the_split_prefills_state() {
        let model = qwen35(false);
        let fwd = |cache: &mut Qwen35Cache, ids: &[i32], at: usize| {
            SpeculativeTarget::forward(
                &model,
                cache,
                &input_ids(ids),
                at as i32,
                LogitsScope::Last,
                false,
            )
            .unwrap()
        };
        let host = |a: &Array| {
            a.as_dtype(mlx_rs::Dtype::Float32)
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        for conv_len in [8usize, 100] {
            let conversation: Vec<i32> = (0..conv_len as i32).map(|i| (i * 7 % 49) + 1).collect();
            let mut p1 = conversation.clone();
            p1.extend_from_slice(&[40, 41, 42, 43, 44]);
            let counted = Counted::new(&model);
            let pre = prefill_restored(
                &counted,
                None,
                &p1,
                Some(conv_len),
                false,
                &CancelFlag::new(),
            )
            .unwrap();
            assert_eq!(counted.calls.get(), 1, "{conv_len}: one forward");
            assert_eq!(counted.take(), p1.len(), "{conv_len}: the whole prompt");

            // The reference: the prefill split at the boundary, snapshotted between the forwards.
            let mut split = model.new_cache();
            fwd(&mut split, &p1[..conv_len], 0);
            let reference = split.clone();
            let split_logits = fwd(&mut split, &p1[conv_len..], conv_len).logits;
            let boundary = pre.boundary.expect("a boundary snapshot");
            assert_eq!(boundary.len(), conv_len);
            assert_eq!(boundary.cache.offset(), conv_len as i32);
            // Equal up to the GEMM's row-count-dependent reduction order (one forward of `T` rows
            // vs `b` then `T - b`; bit-identical on the Metal device this was written on).
            let close = |g: &[f32], w: &[f32], what: &str| {
                assert_eq!(g.len(), w.len(), "{conv_len}: {what}");
                let scale = w.iter().fold(1.0f32, |m, x| m.max(x.abs()));
                let diff = g
                    .iter()
                    .zip(w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(diff <= 1e-4 * scale, "{conv_len}: {what} differs by {diff}");
            };
            let pairs = |got: Vec<(Vec<f32>, Vec<f32>)>, want: Vec<(Vec<f32>, Vec<f32>)>, what| {
                assert_eq!(got.len(), want.len(), "{conv_len}: {what}");
                for (i, ((ga, gb), (wa, wb))) in got.iter().zip(&want).enumerate() {
                    close(ga, wa, &format!("{what} {i}"));
                    close(gb, wb, &format!("{what} {i}"));
                }
            };
            pairs(
                boundary.cache.delta_states(),
                reference.delta_states(),
                "the DeltaNet states at the boundary",
            );
            pairs(
                boundary.cache.attn_states(),
                reference.attn_states(),
                "the attention KV up to the boundary",
            );
            close(&host(&pre.logits), &host(&split_logits), "logits");
            pairs(
                pre.cache.delta_states(),
                split.delta_states(),
                "final states",
            );

            // A later hit on each snapshot decodes the same.
            let mut p2 = conversation.clone();
            p2.extend_from_slice(&[30, 31, 32, 40, 41]);
            let mut runs = Vec::new();
            for snapshot in [boundary.cache, reference] {
                let mut pc = PrefixCache::with_budget(1 << 30);
                pc.store(
                    &p1,
                    &[],
                    split.clone(),
                    Some(Boundary {
                        len: conv_len,
                        cache: snapshot,
                    }),
                    None,
                );
                let (run, reused) = turn(&model, &mut NoProposer, &mut pc, &p2, None, 6);
                assert_eq!(reused, conv_len);
                runs.push(run.output.tokens);
            }
            assert_eq!(runs[0], runs[1], "{conv_len}: decode after the hit");
        }
    }

    /// AC3 on real entries: past the budget the least-recently-used entry is evicted, and the
    /// resident bytes never exceed the budget.
    #[test]
    fn real_entries_are_evicted_least_recently_used_within_the_budget() {
        let model = causal();
        let prompts = [vec![3, 9, 4, 11], vec![5, 6, 7, 8], vec![12, 13, 14, 15]];
        let mut sizing = PrefixCache::with_budget(1 << 30);
        turn(&model, &mut NoProposer, &mut sizing, &prompts[0], None, 3);
        let entry = sizing.resident_bytes();
        // Room for two entries, not three.
        let mut pc = PrefixCache::with_budget(entry * 2 + entry / 2);
        turn(&model, &mut NoProposer, &mut pc, &prompts[0], None, 3);
        turn(&model, &mut NoProposer, &mut pc, &prompts[1], None, 3);
        assert!(pc.resident_bytes() <= pc.budget_bytes());
        // Touch the first entry so the second is least recently used.
        let mut again = prompts[0].clone();
        again.push(1);
        let (_, reused) = turn(&model, &mut NoProposer, &mut pc, &again, None, 3);
        assert_eq!(reused, prompts[0].len());
        turn(&model, &mut NoProposer, &mut pc, &prompts[2], None, 3);
        assert!(pc.resident_bytes() <= pc.budget_bytes());
        assert_eq!(pc.len(), 2);
        let keys = pc.keys();
        assert!(
            keys.iter().all(|k| !k.starts_with(&prompts[1])),
            "the least-recently-used entry went: {keys:?}"
        );
        assert!(pc.stats().evicted >= 1);
    }
}
