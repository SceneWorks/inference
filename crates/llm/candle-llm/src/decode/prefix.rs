//! The cross-turn prefix cache (epic 7253 story 7256; epic sc-24432 story sc-24437).
//!
//! Requests routinely share a leading run of tokens — a common system prompt, a few-shot preamble,
//! the growing history of a multi-turn chat. A causal decoder's state after position `i` depends
//! only on tokens `0..=i`, so that shared run's state is **identical** across the requests; the
//! [`PrefixCache`] keeps it and a later request restores it instead of re-prefilling it.
//!
//! **One** cache type serves every Candle decoder family (E8). The policy — which stored prefix
//! the prompt extends furthest, a byte budget with least-recently-used eviction — is the
//! backend-neutral [`core_llm::PrefixStore`]; this module owns the Candle state it points at
//! ([`PrefixEntry`]), always **copies** (the static KV buffers and the DeltaNet checkpoint ring are
//! written in place, so an entry must never alias a live cache):
//!
//! * a softmax decoder's KV — the step seam's [`StepKvCache`] (static or growing) or the reference
//!   loop's [`ContiguousKvCache`] — is copied after a request, keyed by the tokens it holds, and
//!   any later prompt sharing a leading run reuses that run: the copy's first `len` positions are
//!   written into the new request's cache;
//! * the Qwen3.5/3.6 hybrid cannot truncate its DeltaNet recurrence, so its entry is the **whole**
//!   cache at a prefill boundary ([`Qwen35PrefixState`]: attention KV by offset, every linear
//!   layer's conv tail and recurrent state), reusable only by a prompt extending exactly those
//!   tokens ([`PrefixReuse::WholeEntry`]). The restore hands each linear layer its state through
//!   its checkpoint ring (sc-24131) as the ring's oldest restorable position, so a verify step's
//!   rollback never reaches below the restored prefix. The provider snapshots at the end of the
//!   rendered conversation before its generation prompt; when the request ran the MTP head its
//!   state at the boundary rides along ([`MtpBoundary`]).
//!
//! Reuse is exact: the restored state is what a prefill of those tokens computed, attended over by
//! the same kernels. [`prefill_restored`] is the seam the provider's engine routes and the tests
//! drive; [`generate_cached`] is the older reference-loop entry on the same cache.

use candle_core::Tensor;

use core_llm::{PrefixAdmission, PrefixReuse, PrefixStore};

use crate::decode::cancel::CancelFlag;
use crate::decode::proposers::MtpBoundary;
use crate::decode::step::{StepModel, StepRequest};
use crate::decode::stream::{
    decode_loop, default_seed, GenerationConfig, GenerationOutput, StreamEvent,
};
use crate::error::{Error, Result};
use crate::models::{CausalLm, Qwen35Cache, Qwen35PrefixState};
use crate::primitives::decode_cache::{tensor_bytes, DecodeCache};
use crate::primitives::input_ids;
use crate::primitives::kv_cache::{ContiguousKvCache, KvCache, SEQ_AXIS};
use crate::primitives::sampler::SplitMix64;
use crate::primitives::StepKvCache;

pub use core_llm::PrefixStats;

/// One stored prefix's Candle state (copies, never views of a live cache).
#[derive(Clone, Debug)]
pub enum PrefixEntry {
    /// A softmax decoder's per-layer KV `[1, heads, len, dim]` (`None` for a layer that caches
    /// nothing), reused by any leading run.
    Kv(Vec<Option<(Tensor, Tensor)>>),
    /// The Qwen3.5/3.6 hybrid's whole cache at a prefill boundary, plus the MTP head's state there
    /// when the request that stored it ran the head.
    Hybrid {
        /// Attention KV and every linear layer's recurrent state.
        state: Qwen35PrefixState,
        /// The MTP head at the same boundary.
        mtp: Option<MtpBoundary>,
    },
}

impl PrefixEntry {
    /// Positions the stored state covers.
    fn positions(&self) -> usize {
        match self {
            PrefixEntry::Kv(layers) => layers
                .iter()
                .flatten()
                .map(|(k, _)| k.dims()[SEQ_AXIS])
                .next()
                .unwrap_or(0),
            PrefixEntry::Hybrid { state, .. } => state.len(),
        }
    }

    /// Bytes the stored tensors occupy.
    fn bytes(&self) -> u64 {
        let bytes = match self {
            PrefixEntry::Kv(layers) => layers.iter().flatten().fold(0usize, |acc, (k, v)| {
                acc.saturating_add(tensor_bytes(k))
                    .saturating_add(tensor_bytes(v))
            }),
            PrefixEntry::Hybrid { state, mtp } => state
                .bytes()
                .saturating_add(mtp.as_ref().map_or(0, MtpBoundary::bytes)),
        };
        bytes as u64
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

/// A decode cache the prefix cache can copy and restore.
pub trait PrefixSnapshot {
    /// How a stored copy may be reused.
    const REUSE: PrefixReuse;
    /// Copy the first `len` positions (the hybrid copies its whole state, or the boundary its last
    /// prefill captured: `len` must be one of those).
    fn snapshot(&self, len: usize) -> Result<PrefixEntry>;
    /// Write the entry's first `len` positions into this **empty** cache.
    fn restore(&mut self, entry: &PrefixEntry, len: usize) -> Result<()>;
    /// Positions held.
    fn positions(&self) -> usize;
}

/// Copy `(k, v)` narrowed to `len` positions into compact storage of exactly that size: never a
/// view of the live buffer (a later write would reach it), and never a clone of the whole buffer
/// (`Tensor::copy` keeps the view's layout over a copy of all of its storage, so an entry would
/// hold the cache's full capacity while charged for `len` positions).
fn copy_kv(k: &Tensor, v: &Tensor, len: usize) -> Result<(Tensor, Tensor)> {
    Ok((
        k.narrow(SEQ_AXIS, 0, len)?.force_contiguous()?,
        v.narrow(SEQ_AXIS, 0, len)?.force_contiguous()?,
    ))
}

/// Write a [`PrefixEntry::Kv`]'s first `len` positions into an empty softmax cache.
fn restore_kv(cache: &mut dyn KvCache, entry: &PrefixEntry, len: usize) -> Result<()> {
    let PrefixEntry::Kv(layers) = entry else {
        return Err(Error::Msg(
            "prefix cache: a hybrid entry cannot seed a softmax KV cache".into(),
        ));
    };
    if cache.offset() != 0 {
        return Err(Error::Msg(format!(
            "prefix cache: a prefix restores into an empty cache, not one at {}",
            cache.offset()
        )));
    }
    for (layer, kv) in layers.iter().enumerate() {
        if let Some((k, v)) = kv {
            let (k, v) = if k.dims()[SEQ_AXIS] == len {
                (k.clone(), v.clone())
            } else {
                (
                    k.narrow(SEQ_AXIS, 0, len)?.contiguous()?,
                    v.narrow(SEQ_AXIS, 0, len)?.contiguous()?,
                )
            };
            cache.update(layer, &k, &v)?;
        }
    }
    Ok(())
}

impl PrefixSnapshot for StepKvCache {
    const REUSE: PrefixReuse = PrefixReuse::AnyPrefix;

    fn snapshot(&self, len: usize) -> Result<PrefixEntry> {
        (0..KvCache::num_layers(self))
            .map(|layer| {
                self.layer_kv(layer)?
                    .map(|(k, v)| copy_kv(&k, &v, len))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()
            .map(PrefixEntry::Kv)
    }

    fn restore(&mut self, entry: &PrefixEntry, len: usize) -> Result<()> {
        restore_kv(self, entry, len)
    }

    fn positions(&self) -> usize {
        DecodeCache::len(self).max(0) as usize
    }
}

impl PrefixSnapshot for ContiguousKvCache {
    const REUSE: PrefixReuse = PrefixReuse::AnyPrefix;

    fn snapshot(&self, len: usize) -> Result<PrefixEntry> {
        (0..self.num_layers())
            .map(|layer| {
                self.peek(layer)
                    .map(|(k, v)| copy_kv(k, v, len))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()
            .map(PrefixEntry::Kv)
    }

    fn restore(&mut self, entry: &PrefixEntry, len: usize) -> Result<()> {
        restore_kv(self, entry, len)
    }

    fn positions(&self) -> usize {
        self.offset().max(0) as usize
    }
}

impl PrefixSnapshot for Qwen35Cache {
    const REUSE: PrefixReuse = PrefixReuse::WholeEntry;

    /// The cache at its length, or at a boundary its last prefill captured
    /// ([`StepRequest::snapshot_at`]): a recurrent state exists only where it was taken.
    fn snapshot(&self, len: usize) -> Result<PrefixEntry> {
        Ok(PrefixEntry::Hybrid {
            state: self.prefix_snapshot_at(len)?,
            mtp: None,
        })
    }

    fn restore(&mut self, entry: &PrefixEntry, len: usize) -> Result<()> {
        match entry {
            PrefixEntry::Hybrid { state, .. } if state.len() == len => self.restore_prefix(state),
            PrefixEntry::Hybrid { state, .. } => Err(Error::Msg(format!(
                "prefix cache: a recurrent state holds exactly {} positions, not {len}",
                state.len()
            ))),
            PrefixEntry::Kv(_) => Err(Error::Msg(
                "prefix cache: a softmax KV entry cannot seed the hybrid cache".into(),
            )),
        }
    }

    fn positions(&self) -> usize {
        self.offset().max(0) as usize
    }
}

/// What a restore put into the cache: the prompt positions it holds and, for an MTP request, the
/// head's state at the same boundary.
#[derive(Debug)]
pub struct Restored {
    /// Leading prompt positions restored.
    pub reused: usize,
    /// The MTP head's state at `reused` (only when the lookup asked for it).
    pub mtp: Option<MtpBoundary>,
}

/// A bounded, least-recently-used cross-turn prefix cache for one loaded model (see the module
/// docs). Resident bytes never exceed the budget; a request short of memory evicts entries first
/// ([`admit`](Self::admit)).
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

    /// Request admission with the cache (E7): evict so the request and the snapshot it would
    /// leave behind (`snapshot_bytes`, an upper bound) fit, or run it without keeping one (see
    /// [`PrefixStore::admit`]).
    pub fn admit(&mut self, required: u64, snapshot_bytes: u64, available: u64) -> PrefixAdmission {
        self.store.admit(required, snapshot_bytes, available)
    }

    /// Restore the longest prefix of `prompt` this cache can serve into the **empty** `cache`, or
    /// `None` (a miss; the cache is untouched). `mtp`: the request runs the MTP head, so only
    /// entries carrying its state qualify. The reuse is clamped to the positions the stored state
    /// really holds — defence in depth against an entry whose key over-states it (the
    /// budget-finish off-by-one of sc-12455) — and the stats count what was granted.
    pub fn restore_into<C: PrefixSnapshot>(
        &mut self,
        cache: &mut C,
        prompt: &[i32],
        mtp: bool,
    ) -> Result<Option<Restored>> {
        let Some(hit) = self
            .store
            .lookup(prompt, |e| e.serves(mtp).then(|| e.positions()))
        else {
            return Ok(None);
        };
        let reused = hit.reused;
        cache.restore(hit.state, reused)?;
        let mtp = match hit.state {
            PrefixEntry::Hybrid {
                mtp: Some(head), ..
            } if mtp => Some(head.clone()),
            _ => None,
        };
        Ok(Some(Restored { reused, mtp }))
    }

    /// Keep a request's state for later requests. A [`PrefixReuse::AnyPrefix`] cache is copied
    /// after the run, keyed by the tokens it holds — `prompt + generated`, cut to the cache length
    /// (a budget or host-stop finish never feeds the last generated token, sc-12455); a
    /// [`PrefixReuse::WholeEntry`] cache keeps only the boundary snapshot its prefill took, with
    /// the MTP head's state there when the request captured one. An entry larger than the budget
    /// is not kept. An error is a failed copy; nothing is stored then.
    pub fn store<C: PrefixSnapshot>(
        &mut self,
        prompt: &[i32],
        generated: &[i32],
        cache: &C,
        boundary: Option<Boundary>,
        mtp: Option<MtpBoundary>,
    ) -> Result<()> {
        match C::REUSE {
            PrefixReuse::AnyPrefix => {
                let mut tokens = prompt.to_vec();
                tokens.extend_from_slice(generated);
                tokens.truncate(cache.positions());
                if !tokens.is_empty() {
                    let entry = cache.snapshot(tokens.len())?;
                    self.insert(tokens, entry);
                }
            }
            PrefixReuse::WholeEntry => {
                if let Some(Boundary { len, entry }) = boundary {
                    let entry = match entry {
                        PrefixEntry::Hybrid { state, .. } => PrefixEntry::Hybrid {
                            state,
                            mtp: mtp.filter(|m| m.len() == len),
                        },
                        kv => kv,
                    };
                    self.insert(prompt[..len].to_vec(), entry);
                }
            }
        }
        Ok(())
    }

    fn insert(&mut self, tokens: Vec<i32>, entry: PrefixEntry) {
        let bytes = entry.bytes();
        let reuse = match entry {
            PrefixEntry::Kv(_) => PrefixReuse::AnyPrefix,
            PrefixEntry::Hybrid { .. } => PrefixReuse::WholeEntry,
        };
        self.store.insert(tokens, reuse, bytes, entry);
    }
}

/// A snapshot a prefill took at a boundary inside the prompt, for [`PrefixCache::store`].
#[derive(Debug)]
pub struct Boundary {
    len: usize,
    entry: PrefixEntry,
}

impl Boundary {
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
#[derive(Debug)]
pub struct PrefixPrefill {
    /// Last-position logits, `[1, vocab]`.
    pub logits: Tensor,
    /// The target's hidden rows for the prefilled positions `reused..prompt.len()` (when asked).
    pub hidden: Option<Tensor>,
    /// Leading prompt positions the lookup restored (the prefill starts past them).
    pub reused: usize,
    /// Prompt tokens the prefill fed through the model — `prompt.len() - reused` when the
    /// restored cache was really prefilled on top of; the provider reports
    /// `prefix_hit_tokens = prompt.len() - fed_tokens`, so the report measures the prefill it ran.
    pub fed_tokens: usize,
    /// The MTP head's state at `reused`, for
    /// [`MtpProposer::resume_from`](super::MtpProposer::resume_from).
    pub mtp: Option<MtpBoundary>,
    /// The snapshot at the requested boundary, for [`PrefixCache::store`].
    pub boundary: Option<Boundary>,
}

/// Prefill `prompt` through `model` into `cache`, which holds the `restored` prefix (or is empty
/// on a miss): only `prompt[reused..]` runs, in **one** forward. `boundary` is a prompt length to
/// snapshot at — honoured for a [`PrefixReuse::WholeEntry`] cache when it falls strictly inside
/// the prefilled span, by capturing the state there inside the forward
/// ([`StepRequest::snapshot_at`], sc-24446; a softmax cache is copied after the run instead).
/// `want_hidden` returns the target's hidden rows for the prefilled positions (the MTP warm-up).
/// The cancel flag is checked before and after the forward; a cancelled prefill returns
/// [`Error::Canceled`] and snapshots nothing.
pub fn prefill_restored<M>(
    model: &M,
    cache: &mut M::Cache,
    restored: Option<Restored>,
    prompt: &[i32],
    boundary: Option<usize>,
    want_hidden: bool,
    cancel: &CancelFlag,
) -> Result<PrefixPrefill>
where
    M: StepModel + ?Sized,
    M::Cache: PrefixSnapshot,
{
    let (reused, mtp) = restored.map_or((0, None), |r| (r.reused, r.mtp));
    if reused >= prompt.len() {
        return Err(Error::Msg(format!(
            "prefill_restored: {reused} restored positions leave nothing of a {}-token prompt \
             to prefill",
            prompt.len()
        )));
    }
    let split = boundary.filter(|&b| {
        <M::Cache as PrefixSnapshot>::REUSE == PrefixReuse::WholeEntry
            && b > reused
            && b < prompt.len()
    });
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    // One forward over everything past the restored prefix (sc-24446): a boundary inside it is
    // captured on the way — the recurrent state there kept by the forward, the attention KV
    // narrowed to it afterwards — instead of splitting the prefill into two forwards. A prefill
    // (sc-24441): from a restored, non-empty cache it still attends as a cold prefill does,
    // never on the decode step's device-positions path.
    let mut request = StepRequest::last(&prompt[reused..])
        .with_hidden(want_hidden)
        .as_prefill();
    if let Some(b) = split {
        request = request.with_snapshot_at(b - reused);
    }
    let out = model.forward_step(cache, request)?;
    if cancel.is_cancelled() {
        // Cancelled while the forward ran: no snapshot of a prefill the request abandons.
        return Err(Error::Canceled);
    }
    let snapshot = match split {
        Some(b) => Some(Boundary {
            len: b,
            entry: cache.snapshot(b)?,
        }),
        None => None,
    };
    Ok(PrefixPrefill {
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
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_cached: empty prompt".into()));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let mut cache = model.new_cache();
    let matched_len = prefix_cache
        .restore_into(&mut cache, prompt_ids, false)?
        .map_or(0, |r| r.reused);
    let suffix = input_ids(&prompt_ids[matched_len..], model.device())?;
    let logits = model.decode_logits(&suffix, &mut cache, matched_len as i32)?;

    let out = decode_loop(
        model,
        &mut cache,
        logits,
        rng,
        prompt_ids.to_vec(),
        config,
        cancel,
        on_event,
        None,
    )?;
    prefix_cache.store(prompt_ids, &out.tokens, &cache, None, None)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};

    use super::*;

    fn kv(seq: usize, value: f64) -> ContiguousKvCache {
        let t = (Tensor::ones((1, 1, seq, 2), DType::F32, &Device::Cpu).unwrap() * value).unwrap();
        let mut cache = ContiguousKvCache::new(1);
        cache.update(0, &t, &t).unwrap();
        cache
    }

    /// Defence in depth (sc-12455): if an entry's key ever over-states its KV again (the pre-fix
    /// budget-finish state), the restore clamps the reuse to the positions the tensors hold
    /// instead of failing the request.
    #[test]
    fn restore_clamps_the_reuse_to_the_stored_kv() {
        let mut pc = PrefixCache::with_budget(1 << 20);
        // Manufacture the inconsistent state directly: 6 keyed tokens, 5 positions of KV.
        let tokens: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        pc.insert(tokens.clone(), kv(5, 0.0).snapshot(5).unwrap());

        let mut prompt = tokens;
        prompt.extend_from_slice(&[7, 8]);
        let mut cache = ContiguousKvCache::new(1);
        let r = pc
            .restore_into(&mut cache, &prompt, false)
            .unwrap()
            .expect("hit");
        assert_eq!(r.reused, 5);
        assert_eq!(cache.offset(), 5);
        assert_eq!(pc.stats().reused_prefix_tokens, 5);
    }

    /// An entry is a copy: the cache it was taken from, and a cache restored from it, can be
    /// written without reaching it.
    #[test]
    fn entries_are_copies_no_live_cache_writes() {
        let mut pc = PrefixCache::with_budget(1 << 20);
        let layout = crate::primitives::KvLayout {
            layers: vec![Some(crate::primitives::LayerKvShape {
                kv_heads: 1,
                key_dim: 2,
                value_dim: 2,
                device: Device::Cpu,
            })],
            dtype: DType::F32,
        };
        let mut live = StepKvCache::preallocated(&layout, 8).unwrap();
        let ones = Tensor::ones((1, 1, 4, 2), DType::F32, &Device::Cpu).unwrap();
        live.update(0, &ones, &ones).unwrap();
        pc.store(&[1, 2, 3, 4], &[], &live, None, None).unwrap();
        // The static buffer is rewritten in place after a rollback: the entry must not see it.
        live.truncate(1).unwrap();
        let twos = (ones.narrow(2, 0, 3).unwrap() * 2.0).unwrap();
        live.update(0, &twos, &twos).unwrap();

        let mut restored = StepKvCache::preallocated(&layout, 8).unwrap();
        let r = pc
            .restore_into(&mut restored, &[1, 2, 3, 4, 5], false)
            .unwrap()
            .unwrap();
        assert_eq!(r.reused, 4);
        let (k, _) = restored.layer_kv(0).unwrap().unwrap();
        let k: Vec<f32> = k.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            k.iter().all(|&x| x == 1.0),
            "the entry saw a live write: {k:?}"
        );
    }

    #[test]
    fn an_entry_past_the_budget_is_not_kept() {
        let cache = kv(4, 0.0);
        let bytes = cache.snapshot(4).unwrap().bytes();
        let mut pc = PrefixCache::with_budget(bytes - 1);
        pc.store(&[1, 2, 3, 4], &[], &cache, None, None).unwrap();
        assert!(pc.is_empty());
        assert_eq!(pc.stats().rejected, 1);
    }
}

/// The prefix cache under the engine on the tiny CPU decoders (story sc-24437): each family's
/// second turn restores the shared prefix, prefills only the rest — counted at the step seam, not
/// by the cache's own report — and decodes exactly what a cold run decodes.
#[cfg(test)]
mod engine_tests {
    use std::cell::Cell;
    use std::collections::HashMap;

    use candle_core::Device;

    use super::*;
    use crate::decode::engine::{generate_speculative, NoProposer, Proposer, SpeculativePrompt};
    use crate::decode::proposers::MtpProposer;
    use crate::decode::speculative::SpeculativeStats;
    use crate::decode::step::StepOutput;
    use crate::models::qwen35::tests::{text_model, text_model_with_mtp};
    use crate::primitives::sampler::SamplingParams;
    use crate::primitives::{SplitMix64, TokenRng, Weights};

    /// Counts the tokens every step feeds; can raise a cancel flag from inside a forward.
    struct Counted<'a, M> {
        inner: &'a M,
        fed: Cell<usize>,
        calls: Cell<usize>,
        cancel_in_forward: Option<&'a CancelFlag>,
    }

    impl<'a, M> Counted<'a, M> {
        fn new(inner: &'a M) -> Self {
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

    impl<M: StepModel> StepModel for Counted<'_, M> {
        type Cache = M::Cache;
        fn new_cache(&self) -> M::Cache {
            self.inner.new_cache()
        }
        fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<M::Cache> {
            self.inner.new_cache_for(capacity, overshoot)
        }
        fn device(&self) -> &Device {
            self.inner.device()
        }
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
        fn forward_step(
            &self,
            cache: &mut M::Cache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            self.fed.set(self.fed.get() + request.len()?);
            self.calls.set(self.calls.get() + 1);
            if let Some(cancel) = self.cancel_in_forward {
                cancel.cancel();
            }
            self.inner.forward_step(cache, request)
        }
    }

    fn tiny_llama() -> CausalLm {
        tiny_llama_on(&Device::Cpu, candle_core::DType::F32)
    }

    /// [`tiny_llama`]'s weights on `device`, computing in `dtype`.
    fn tiny_llama_on(device: &Device, dtype: candle_core::DType) -> CausalLm {
        let (vocab, hidden, inter, heads, kv_heads, layers) = (40, 16, 32, 4, 2, 2);
        let head_dim = hidden / heads;
        let mut rng = SplitMix64::new(0x24437);
        let mut rand = |dims: &[usize]| {
            let n: usize = dims.iter().product();
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
            Tensor::from_vec(data, dims.to_vec(), device).unwrap()
        };
        let mut w = HashMap::new();
        w.insert(
            "model.embed_tokens.weight".to_string(),
            rand(&[vocab, hidden]),
        );
        w.insert("model.norm.weight".to_string(), rand(&[hidden]));
        w.insert("lm_head.weight".to_string(), rand(&[vocab, hidden]));
        for i in 0..layers {
            let p = |s: &str| format!("model.layers.{i}.{s}");
            w.insert(p("input_layernorm.weight"), rand(&[hidden]));
            w.insert(p("post_attention_layernorm.weight"), rand(&[hidden]));
            w.insert(
                p("self_attn.q_proj.weight"),
                rand(&[heads * head_dim, hidden]),
            );
            w.insert(
                p("self_attn.k_proj.weight"),
                rand(&[kv_heads * head_dim, hidden]),
            );
            w.insert(
                p("self_attn.v_proj.weight"),
                rand(&[kv_heads * head_dim, hidden]),
            );
            w.insert(
                p("self_attn.o_proj.weight"),
                rand(&[hidden, heads * head_dim]),
            );
            w.insert(p("mlp.gate_proj.weight"), rand(&[inter, hidden]));
            w.insert(p("mlp.up_proj.weight"), rand(&[inter, hidden]));
            w.insert(p("mlp.down_proj.weight"), rand(&[hidden, inter]));
        }
        let cfg = crate::config::ModelConfig::from_json(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "hidden_size": hidden, "intermediate_size": inter, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": kv_heads,
            "vocab_size": vocab, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 256, "tie_word_embeddings": false
        }))
        .unwrap();
        CausalLm::from_weights_dtype(&Weights::from_map(w, device.clone()), "", cfg, None, dtype)
            .unwrap()
    }

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        GenerationConfig {
            max_new_tokens,
            sampling: SamplingParams::default(),
            seed: Some(7),
            stop_tokens: Vec::new(),
        }
    }

    fn cold<M: StepModel + ?Sized, P: Proposer + ?Sized>(
        model: &M,
        proposer: &mut P,
        prompt: &[i32],
        max_new: usize,
    ) -> (Vec<i32>, SpeculativeStats) {
        let run = generate_speculative(
            model,
            proposer,
            SpeculativePrompt::Tokens(prompt),
            &greedy(max_new),
            2,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        (run.output.tokens, run.stats)
    }

    /// One turn through the prefix cache on the engine: restore into a fresh step cache, prefill
    /// the rest (snapshotting at `boundary`), decode with `proposer`, store. Returns the tokens,
    /// the reused positions and the run's counters.
    fn turn<M, P>(
        model: &M,
        proposer: &mut P,
        pc: &mut PrefixCache,
        prompt: &[i32],
        boundary: Option<usize>,
        max_new: usize,
    ) -> (Vec<i32>, usize, SpeculativeStats)
    where
        M: StepModel + ?Sized,
        M::Cache: PrefixSnapshot,
        P: Proposer + ?Sized,
    {
        let mut cache = model.new_cache_for(prompt.len() + max_new, 2).unwrap();
        let restored = pc.restore_into(&mut cache, prompt, false).unwrap();
        let pre = prefill_restored(
            model,
            &mut cache,
            restored,
            prompt,
            boundary,
            false,
            &CancelFlag::new(),
        )
        .unwrap();
        let reused = pre.reused;
        let run = generate_speculative(
            model,
            proposer,
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits: pre.logits,
                hidden: None,
                history: prompt,
                position_delta: 0,
                warm_proposer: true,
            },
            &greedy(max_new),
            2,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        pc.store(prompt, &run.output.tokens, &cache, pre.boundary, None)
            .unwrap();
        (run.output.tokens, reused, run.stats)
    }

    /// sc-24441 × sc-24437: with device positions on, a prefix hit's decode steps attend with the
    /// length-aware decode attention while its restored suffix prefill — marked a prefill — and a
    /// cold prefill both attend `sdpa_gqa`. Greedy tokens of a prefix hit match a cold run (and,
    /// with `tol`, so do the prefill logits) — for a suffix inside and past the device step bound.
    /// The restored request's prefill logits (restore + suffix prefill) against a cold
    /// prefill's of the same prompt: within `tol` (absolute, per logit), when given.
    fn prefill_logits_match<M>(
        model: &M,
        pc: &mut PrefixCache,
        prompt: &[i32],
        tol: Option<f32>,
        what: &str,
    ) where
        M: StepModel + ?Sized,
        M::Cache: PrefixSnapshot,
    {
        let Some(tol) = tol else { return };
        let mut cache = model.new_cache_for(prompt.len() + 8, 2).unwrap();
        let restored = pc.restore_into(&mut cache, prompt, false).unwrap();
        assert!(restored.is_some(), "{what}: a hit");
        let hit = prefill_restored(
            model,
            &mut cache,
            restored,
            prompt,
            None,
            false,
            &CancelFlag::new(),
        )
        .unwrap()
        .logits;
        let mut cold_cache = model.new_cache_for(prompt.len() + 8, 2).unwrap();
        let cold = model
            .forward_step(&mut cold_cache, StepRequest::last(prompt))
            .unwrap()
            .logits;
        let diff = (hit.to_dtype(candle_core::DType::F32).unwrap()
            - cold.to_dtype(candle_core::DType::F32).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
        assert!(
            diff <= tol,
            "{what}: prefill logits differ by {diff} > {tol}"
        );
    }

    fn causal_hit_matches_cold(model: &CausalLm, tol: Option<f32>, what: &str) {
        assert!(model.device_positions_active(), "{what}: device positions");
        let p1 = vec![3, 9, 4, 11, 3, 9, 4, 11, 5];
        for suffix in [3usize, 20] {
            // A fresh cache per case: an earlier case's stored run would serve a longer prefix.
            let mut pc = PrefixCache::with_budget(1 << 30);
            let (gen1, _, _) = turn(model, &mut NoProposer, &mut pc, &p1, None, 6);
            let mut p2 = p1.clone();
            p2.extend_from_slice(&gen1);
            p2.extend((0..suffix).map(|i| (i * 7 % 37) as i32 + 1));
            let expected = p1.len() + gen1.len() - 1;
            prefill_logits_match(
                model,
                &mut pc,
                &p2,
                tol,
                &format!("{what}: suffix {suffix}"),
            );
            let (gen2, reused, _) = turn(model, &mut NoProposer, &mut pc, &p2, None, 8);
            assert_eq!(reused, expected, "{what}: suffix {suffix}");
            assert_eq!(
                gen2,
                cold(model, &mut NoProposer, &p2, 8).0,
                "{what}: suffix {suffix}"
            );
        }
    }

    /// The hybrid's half of [`causal_hit_matches_cold`]: restore the recurrent state at the
    /// boundary, prefill a suffix inside and past the device step bound, match a cold run.
    fn hybrid_hit_matches_cold(model: &crate::models::Qwen35Model, tol: Option<f32>, what: &str) {
        assert!(model.device_positions_active(), "{what}: device positions");
        let conversation = vec![3, 9, 4, 11, 3, 9, 4, 11];
        let mut p1 = conversation.clone();
        p1.extend_from_slice(&[40, 41]);
        for suffix in [5usize, 20] {
            let mut pc = PrefixCache::with_budget(1 << 30);
            turn(model, &mut NoProposer, &mut pc, &p1, Some(8), 5);
            let mut p = conversation.clone();
            p.extend((0..suffix).map(|i| (i * 7 % 37) as i32 + 1));
            prefill_logits_match(model, &mut pc, &p, tol, &format!("{what}: suffix {suffix}"));
            let (tokens, reused, _) = turn(model, &mut NoProposer, &mut pc, &p, None, 6);
            assert_eq!(reused, conversation.len(), "{what}: suffix {suffix}");
            assert_eq!(
                tokens,
                cold(model, &mut NoProposer, &p, 6).0,
                "{what}: suffix {suffix}"
            );
        }
    }

    /// sc-24441 × sc-24437: a prefill segment from a restored (non-empty) cache is marked a
    /// prefill, so with device positions on it runs the host-position path and the reference
    /// attention — bit for bit what the same model with device positions off computes from the
    /// same restored state — never the decode step's length-aware attention. Both families.
    #[test]
    fn a_restored_suffix_prefill_attends_as_a_cold_prefill() {
        fn restored_logits<M>(model: &M, pc: &mut PrefixCache, prompt: &[i32]) -> Vec<f32>
        where
            M: StepModel + ?Sized,
            M::Cache: PrefixSnapshot,
        {
            let mut cache = model.new_cache_for(prompt.len() + 8, 2).unwrap();
            let restored = pc.restore_into(&mut cache, prompt, false).unwrap();
            assert!(restored.is_some(), "a hit");
            prefill_restored(
                model,
                &mut cache,
                restored,
                prompt,
                None,
                false,
                &CancelFlag::new(),
            )
            .unwrap()
            .logits
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
        }
        let off = tiny_llama();
        let mut on = tiny_llama();
        on.set_device_positions(true);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let p1 = vec![3, 9, 4, 11, 3, 9, 4, 11, 5];
        let (gen1, _, _) = turn(&off, &mut NoProposer, &mut pc, &p1, None, 6);
        let mut p2 = p1.clone();
        p2.extend_from_slice(&gen1);
        p2.extend_from_slice(&[7, 2, 13]);
        assert_eq!(
            restored_logits(&on, &mut pc, &p2),
            restored_logits(&off, &mut pc, &p2),
            "causal"
        );

        let (_, off) = text_model();
        let (_, mut on) = text_model();
        on.set_device_positions(true);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let conversation = vec![3, 9, 4, 11, 3, 9, 4, 11];
        let mut p1 = conversation.clone();
        p1.extend_from_slice(&[40, 41]);
        turn(&off, &mut NoProposer, &mut pc, &p1, Some(8), 5);
        let mut p = conversation;
        p.extend_from_slice(&[20, 21, 22, 40, 41]);
        assert_eq!(
            restored_logits(&on, &mut pc, &p),
            restored_logits(&off, &mut pc, &p),
            "hybrid"
        );
    }

    /// sc-24441 × sc-24437 on the CPU: a prefix hit with device positions on (decode steps
    /// through the reference decode attention) matches a cold run, both families.
    #[test]
    fn restored_prefills_on_device_positions_match_a_cold_run() {
        let mut causal = tiny_llama();
        causal.set_device_positions(true);
        causal_hit_matches_cold(&causal, Some(1e-4), "causal f32 cpu");
        let (_, mut hybrid) = text_model();
        hybrid.set_device_positions(true);
        hybrid_hit_matches_cold(&hybrid, Some(1e-4), "hybrid f32 cpu");
    }

    /// sc-24441 × sc-24437 on CUDA (the `windows-cuda` lane): with the CUDA default (device
    /// positions on), a prefix hit's greedy tokens match a cold run's — causal and hybrid, f32
    /// and the production bf16.
    #[cfg(feature = "cuda")]
    #[test]
    fn restored_prefills_match_a_cold_run_on_cuda() {
        let device = crate::device::select_device().unwrap();
        if !device.is_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        // Prefill logits within 1e-3 in f32 (a different but equally rounded order); in bf16 the
        // greedy tokens are the contract.
        for (dtype, tol) in [
            (candle_core::DType::F32, Some(1e-3)),
            (candle_core::DType::BF16, None),
        ] {
            let causal = tiny_llama_on(&device, dtype);
            causal_hit_matches_cold(&causal, tol, &format!("causal {dtype:?} cuda"));
            let (_, hybrid) = crate::models::qwen35::tests::text_model_dtype_on(&device, dtype);
            hybrid_hit_matches_cold(&hybrid, tol, &format!("hybrid {dtype:?} cuda"));
        }
    }

    /// AC1 on the softmax family (the static step cache): turn 2 extends turn 1's
    /// `prompt + reply`; it restores the whole shared run, prefills only the rest, and matches a
    /// cold run token for token.
    #[test]
    fn a_causal_second_turn_prefills_only_the_new_tokens() {
        let model = tiny_llama();
        let counted = Counted::new(&model);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let p1 = vec![3, 9, 4, 11, 3, 9, 4, 11, 5];
        let (gen1, reused1, _) = turn(&counted, &mut NoProposer, &mut pc, &p1, None, 6);
        assert_eq!(reused1, 0);
        counted.take();

        let mut p2 = p1.clone();
        p2.extend_from_slice(&gen1);
        p2.extend_from_slice(&[7, 2, 13]);
        let expected = p1.len() + gen1.len() - 1;
        let mut cache = counted.new_cache_for(p2.len() + 8, 2).unwrap();
        let restored = pc.restore_into(&mut cache, &p2, false).unwrap();
        let pre = prefill_restored(
            &counted,
            &mut cache,
            restored,
            &p2,
            None,
            false,
            &CancelFlag::new(),
        )
        .unwrap();
        assert_eq!(pre.reused, expected, "N = turn 1's cached positions");
        assert_eq!(counted.take(), p2.len() - expected, "only the new tokens");

        let (gen2, reused2, _) = turn(&model, &mut NoProposer, &mut pc, &p2, None, 8);
        assert_eq!(reused2, expected);
        assert_eq!(gen2, cold(&model, &mut NoProposer, &p2, 8).0);
    }

    /// AC2: the hybrid restores the recurrent state snapshotted at the boundary (through the
    /// checkpoint ring), prefills only past it, and matches a cold run; a third turn restores the
    /// same entry after the second decoded past it (the entry is an immutable copy).
    #[test]
    fn a_qwen35_second_turn_restores_the_recurrent_state_at_the_boundary() {
        let (_, model) = text_model();
        let counted = Counted::new(&model);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let conversation = vec![3, 9, 4, 11, 3, 9, 4, 11];
        let mut p1 = conversation.clone();
        p1.extend_from_slice(&[40, 41]);
        let (_, reused1, _) = turn(&counted, &mut NoProposer, &mut pc, &p1, Some(8), 5);
        assert_eq!(reused1, 0);
        assert_eq!(pc.keys(), vec![conversation.clone()]);
        counted.take();

        for next in [[20, 21, 22, 40, 41], [30, 31, 32, 40, 41]] {
            let mut p = conversation.clone();
            p.extend_from_slice(&next);
            let mut cache = counted.new_cache_for(p.len() + 6, 2).unwrap();
            let restored = pc.restore_into(&mut cache, &p, false).unwrap();
            let pre = prefill_restored(
                &counted,
                &mut cache,
                restored,
                &p,
                None,
                false,
                &CancelFlag::new(),
            )
            .unwrap();
            assert_eq!(pre.reused, conversation.len());
            assert_eq!(counted.take(), next.len(), "only past the boundary");
            // The ring's oldest restorable position is the restored boundary.
            let (tokens, reused, _) = turn(&model, &mut NoProposer, &mut pc, &p, None, 6);
            assert_eq!(reused, conversation.len());
            assert_eq!(tokens, cold(&model, &mut NoProposer, &p, 6).0);
            // Prompt lookup verifies (and rolls back through the ring) on top of the restore.
            let mut lookup = crate::decode::NgramProposer::default();
            let (spec, _, _) = turn(&model, &mut lookup, &mut pc, &p, None, 6);
            assert_eq!(spec, tokens);
        }
        let mut other = conversation[..5].to_vec();
        other.extend_from_slice(&[1, 2]);
        let mut cache = model.new_cache_for(16, 2).unwrap();
        assert!(pc
            .restore_into(&mut cache, &other, false)
            .unwrap()
            .is_none());
        assert_eq!(cache.offset(), 0, "a miss leaves the cache untouched");
    }

    /// A softmax entry holds exactly the positions it is charged for: a copy of the static step
    /// cache's first `len` positions, not the whole preallocated buffer behind them.
    #[test]
    fn a_softmax_entry_pins_only_the_positions_it_is_charged_for() {
        use crate::primitives::decode_cache::pinned_f32_elems;
        let model = tiny_llama();
        let prompt = [3, 9, 4, 11, 3];
        let mut cache = model.new_cache_for(prompt.len() + 16, 2).unwrap();
        model
            .forward_step(&mut cache, StepRequest::last(&prompt))
            .unwrap();
        let entry = PrefixSnapshot::snapshot(&cache, prompt.len()).unwrap();
        let PrefixEntry::Kv(layers) = &entry else {
            panic!("a softmax entry");
        };
        let mut charged = 0;
        for (k, v) in layers.iter().flatten() {
            for t in [k, v] {
                assert_eq!(t.dims()[SEQ_AXIS], prompt.len());
                assert_eq!(pinned_f32_elems(t), t.elem_count(), "{:?}", t.layout());
                charged += tensor_bytes(t);
            }
        }
        assert!(charged > 0);
        assert_eq!(entry.bytes(), charged as u64);
    }

    /// The boundary snapshot is a copy, not a view: the live cache it was taken from goes on to
    /// decode (prompt lookup verifying and rolling back through the DeltaNet checkpoint ring,
    /// whose slots are written in place), and a restore of the snapshot answers exactly as it
    /// did before that decode.
    #[test]
    fn a_boundary_snapshot_is_not_written_by_the_decode_that_continues_past_it() {
        use candle_core::DType;
        let (_, model) = text_model();
        let mut prompt = vec![3, 9, 4, 11, 3, 9, 4, 11];
        prompt.extend_from_slice(&[40, 41]);
        let mut cache = model.new_cache_for(prompt.len() + 24, 3).unwrap();
        let pre = prefill_restored(
            &model,
            &mut cache,
            None,
            &prompt,
            Some(8),
            false,
            &CancelFlag::new(),
        )
        .unwrap();
        let boundary = pre.boundary.expect("a snapshot at the boundary");
        let probe = |entry: &PrefixEntry| -> Vec<f32> {
            let mut fresh = model.new_cache_for(16, 3).unwrap();
            PrefixSnapshot::restore(&mut fresh, entry, 8).unwrap();
            model
                .forward_step(&mut fresh, StepRequest::last(&[20, 21]))
                .unwrap()
                .logits
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap()
        };
        let before = probe(&boundary.entry);
        generate_speculative(
            &model,
            &mut crate::decode::NgramProposer::default(),
            SpeculativePrompt::Prefilled {
                cache: &mut cache,
                logits: pre.logits,
                hidden: None,
                history: &prompt,
                position_delta: 0,
                warm_proposer: true,
            },
            &greedy(20),
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(probe(&boundary.entry), before);
    }

    /// Speculative decoding on top of a hit: the MTP head resumes its warm-up from the state
    /// stored at the boundary; its drafts and the greedy tokens equal a cold MTP run's.
    #[test]
    fn mtp_resumes_its_warm_up_from_the_stored_boundary() {
        let (_, model, head) = text_model_with_mtp();
        let mut pc = PrefixCache::with_budget(1 << 30);
        let conversation = vec![3, 9, 4, 11, 3, 9, 4, 11];
        let mut run_mtp = |prompt: &[i32], boundary: Option<usize>| {
            let mut cache = model.new_cache_for(prompt.len() + 8, 2).unwrap();
            let restored = pc.restore_into(&mut cache, prompt, true).unwrap();
            let pre = prefill_restored(
                &model,
                &mut cache,
                restored,
                prompt,
                boundary,
                true,
                &CancelFlag::new(),
            )
            .unwrap();
            let mut proposer = MtpProposer::new(&head)
                .resume_from(pre.mtp)
                .capture_at(pre.boundary.as_ref().map(Boundary::len));
            let run = generate_speculative(
                &model,
                &mut proposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut cache,
                    logits: pre.logits,
                    hidden: pre.hidden,
                    history: prompt,
                    position_delta: 0,
                    warm_proposer: true,
                },
                &greedy(8),
                2,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap();
            let captured = proposer.take_captured();
            pc.store(prompt, &[], &cache, pre.boundary, captured)
                .unwrap();
            (run.output.tokens, pre.reused, run.stats)
        };
        let mut p1 = conversation.clone();
        p1.extend_from_slice(&[40, 41]);
        let (tokens1, reused1, _) = run_mtp(&p1, Some(conversation.len()));
        assert_eq!(reused1, 0);
        assert_eq!(tokens1, cold(&model, &mut NoProposer, &p1, 8).0);

        let mut p2 = conversation.clone();
        p2.extend_from_slice(&[20, 21, 22, 40, 41]);
        let (tokens2, reused2, stats2) = run_mtp(&p2, None);
        assert_eq!(
            reused2,
            conversation.len(),
            "the MTP request hit the boundary"
        );
        let (cold_tokens, cold_stats) = cold(&model, &mut MtpProposer::new(&head), &p2, 8);
        assert_eq!(tokens2, cold_tokens);
        assert_eq!(tokens2, cold(&model, &mut NoProposer, &p2, 8).0);
        assert!(stats2.proposed > 0);
        assert_eq!(
            (stats2.proposed, stats2.accepted),
            (cold_stats.proposed, cold_stats.accepted)
        );
    }

    /// The head's state resumed at a boundary is the cold warm-up's: its cache after a resumed
    /// warm-up equals the one a whole-prompt warm-up builds.
    #[test]
    fn a_resumed_mtp_warm_up_matches_the_cold_warm_up() {
        let (_, model, head) = text_model_with_mtp();
        let prompt = [3, 9, 4, 11, 3, 9, 4, 11, 20, 21];
        let m = 6usize;
        let out = model
            .forward_step(
                &mut model.new_cache_for(prompt.len(), 0).unwrap(),
                StepRequest::last(&prompt).with_hidden(true),
            )
            .unwrap();
        let hidden = out.hidden.unwrap();
        let mut captured = MtpProposer::new(&head).capture_at(Some(m));
        captured.warm(&prompt, Some(&hidden)).unwrap();
        let boundary = captured.take_captured().expect("captured");
        let mut whole = MtpProposer::new(&head);
        whole.warm(&prompt, Some(&hidden)).unwrap();
        let mut resumed = MtpProposer::new(&head).resume_from(Some(boundary));
        let suffix = hidden.narrow(1, m, prompt.len() - m).unwrap();
        resumed.warm(&prompt, Some(&suffix)).unwrap();
        let bytes = |p: &MtpProposer<'_>| p.cache().bytes();
        assert_eq!(bytes(&resumed), bytes(&whole));
        assert_eq!(bytes(&captured), bytes(&whole));
        let flat = |p: &MtpProposer<'_>| -> Vec<f32> { p.cache().flat_keys().unwrap() };
        let (a, b, c) = (flat(&resumed), flat(&whole), flat(&captured));
        let diff = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        };
        assert_eq!(a.len(), b.len());
        assert!(diff(&a, &b) < 1e-4, "resumed head diverged");
        assert!(diff(&c, &b) < 1e-4, "split warm-up diverged");
    }

    /// A cancel landing mid-prefill stops before the next forward, returns the typed cancel and
    /// snapshots nothing; the next request prefills cold and decodes correctly.
    #[test]
    fn a_cancel_mid_prefill_stores_nothing() {
        let (_, model) = text_model();
        let cancel = CancelFlag::new();
        let mut counted = Counted::new(&model);
        counted.cancel_in_forward = Some(&cancel);
        let mut pc = PrefixCache::with_budget(1 << 30);
        let prompt = vec![3, 9, 4, 11, 3, 9, 40];
        let mut cache = counted.new_cache_for(prompt.len() + 4, 2).unwrap();
        let err = prefill_restored(&counted, &mut cache, None, &prompt, Some(6), false, &cancel)
            .expect_err("cancelled");
        assert!(matches!(err, Error::Canceled), "{err}");
        assert_eq!(
            counted.take(),
            7,
            "one forward, past the boundary (sc-24446)"
        );
        assert!(pc.is_empty());

        let (tokens, reused, _) = turn(&model, &mut NoProposer, &mut pc, &prompt, Some(6), 4);
        assert_eq!(reused, 0);
        assert_eq!(tokens, cold(&model, &mut NoProposer, &prompt, 4).0);
        assert_eq!(pc.len(), 1);
    }

    /// sc-24446 (defect B): a miss with a boundary inside the prompt prefills in **one** forward,
    /// and the snapshot it stores is the one a prefill split at the boundary into two forwards
    /// stores; a later hit restoring it decodes as a hit on the split prefill's snapshot does.
    /// A conversation shorter than one Gated DeltaNet chunk and one long enough to run chunkwise.
    #[test]
    fn a_boundary_miss_prefills_in_one_forward_and_stores_the_split_prefills_state() {
        let (_, model) = text_model();
        for conv_len in [8usize, 100] {
            let conversation: Vec<i32> = (0..conv_len as i32).map(|i| (i * 7 % 49) + 1).collect();
            let mut p1 = conversation.clone();
            p1.extend_from_slice(&[40, 41, 42, 43, 44]);
            let counted = Counted::new(&model);
            let mut cache = counted.new_cache_for(p1.len() + 8, 2).unwrap();
            let pre = prefill_restored(
                &counted,
                &mut cache,
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
            let mut split = model.new_cache_for(p1.len() + 8, 2).unwrap();
            model
                .forward_step(&mut split, StepRequest::last(&p1[..conv_len]).as_prefill())
                .unwrap();
            let reference = split.prefix_snapshot().unwrap();
            let split_logits = model
                .forward_step(&mut split, StepRequest::last(&p1[conv_len..]).as_prefill())
                .unwrap()
                .logits;
            let boundary = pre.boundary.expect("a boundary snapshot");
            assert_eq!(boundary.len(), conv_len);
            let PrefixEntry::Hybrid { state, .. } = &boundary.entry else {
                panic!("a hybrid entry");
            };
            assert_eq!(state.len(), conv_len);
            let (got, want) = (state.host_tensors(), reference.host_tensors());
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let diff = g
                    .iter()
                    .zip(w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert_eq!(g, w, "{conv_len}: snapshot tensor {i} differs by {diff}");
            }
            let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(host(&pre.logits), host(&split_logits), "{conv_len}: logits");

            // A later hit on each snapshot decodes the same.
            let mut p2 = conversation.clone();
            p2.extend_from_slice(&[30, 31, 32, 40, 41]);
            let mut runs = Vec::new();
            for entry in [
                boundary.entry.clone(),
                PrefixEntry::Hybrid {
                    state: reference.clone(),
                    mtp: None,
                },
            ] {
                let mut pc = PrefixCache::with_budget(1 << 30);
                pc.insert(conversation.clone(), entry);
                let (tokens, reused, _) = turn(&model, &mut NoProposer, &mut pc, &p2, None, 6);
                assert_eq!(reused, conv_len);
                runs.push(tokens);
            }
            assert_eq!(runs[0], runs[1], "{conv_len}: decode after the hit");
        }
    }

    /// AC3 on real entries: past the budget the least-recently-used entry is evicted, and resident
    /// bytes never exceed the budget.
    #[test]
    fn real_entries_are_evicted_least_recently_used_within_the_budget() {
        let model = tiny_llama();
        let prompts = [vec![3, 9, 4, 11], vec![5, 6, 7, 8], vec![12, 13, 14, 15]];
        let mut sizing = PrefixCache::with_budget(1 << 30);
        turn(&model, &mut NoProposer, &mut sizing, &prompts[0], None, 3);
        let entry = sizing.resident_bytes();
        let mut pc = PrefixCache::with_budget(entry * 2 + entry / 2);
        turn(&model, &mut NoProposer, &mut pc, &prompts[0], None, 3);
        turn(&model, &mut NoProposer, &mut pc, &prompts[1], None, 3);
        assert!(pc.resident_bytes() <= pc.budget_bytes());
        let mut again = prompts[0].clone();
        again.push(1);
        let (_, reused, _) = turn(&model, &mut NoProposer, &mut pc, &again, None, 3);
        assert_eq!(reused, prompts[0].len());
        turn(&model, &mut NoProposer, &mut pc, &prompts[2], None, 3);
        assert!(pc.resident_bytes() <= pc.budget_bytes());
        assert_eq!(pc.len(), 2);
        let keys = pc.keys();
        assert!(
            keys.iter().all(|k| !k.starts_with(&prompts[1])),
            "the least-recently-used entry went: {keys:?}"
        );
    }
}
