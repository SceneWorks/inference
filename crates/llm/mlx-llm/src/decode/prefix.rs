//! Shared-prefix KV reuse (epic 7153, story 7168).
//!
//! Requests routinely share a leading run of tokens — a common system prompt, a few-shot preamble,
//! the growing history of a multi-turn chat. A causal decoder's keys/values at position `i` depend
//! only on tokens `0..=i`, so that shared run has **bit-identical** KV across the requests; the
//! [`PrefixCache`] caches it and reuses it instead of recomputing it.
//!
//! The policy — which stored token sequence shares the longest prefix, and LRU eviction — is the
//! backend-neutral [`core_llm::prefix::PrefixIndex`]; this module owns only the MLX tensors that
//! policy points at. [`generate_cached`] is the single-sequence decode loop with reuse spliced into
//! prefill: longest-match → seed a [`ContiguousKvCache`] to the matched length → prefill only the
//! suffix → decode → store the request's full `prompt + generated` KV for next time.
//!
//! Reuse is exact, not approximate: a cached run is **token-for-token identical** to a cold run for
//! the same prompt (same kernels, just KV that was already computed). Concurrent in-batch prefix
//! sharing with copy-on-write is the paged cache's job (story 7169); this is the simpler
//! contiguous-friendly cousin that lands first.

use std::collections::HashMap;

use mlx_rs::{Array, Dtype};

use core_llm::prefix::{PrefixId, PrefixIndex};

use crate::decode::cancel::CancelFlag;
use crate::decode::stream::{
    decode_loop, default_seed, observe_cache_events, observe_packed_evidence, ConstraintMask,
    GenerationConfig, GenerationOutput, ObservedCache, StreamEvent,
};
use crate::error::{Error, Result};
use crate::models::CausalLm;
use std::cell::RefCell;
use std::rc::Rc;

use crate::primitives::kv_cache::{ContiguousKvCache, KvCache, SEQ_AXIS};
use crate::primitives::sampler::SplitMix64;
use crate::primitives::{
    input_ids, CacheRoute, CompiledKernelHandle, PackedPagePool, PagedCacheIdentity,
    PagedCacheSnapshot, PagedPackedKvCache,
};

/// Cumulative reuse accounting for a [`PrefixCache`] — the measurable payoff of story 7168.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixStats {
    /// Number of [`generate_cached`] lookups performed.
    pub lookups: usize,
    /// Lookups that found and reused a shared prefix.
    pub hits: usize,
    /// Total prompt positions whose KV was reused (prefill skipped) across all hits — the saved span.
    pub reused_prefix_tokens: usize,
    /// Total prompt positions actually run through the model during prefill (cold + post-match
    /// suffixes). With reuse this is smaller than the sum of prompt lengths by exactly
    /// `reused_prefix_tokens`.
    pub computed_prefill_tokens: usize,
}

/// A bounded, LRU shared-prefix KV cache for the single-sequence decode path.
///
/// Pair one with repeated [`generate_cached`] calls that share a prefix (e.g. a fixed system prompt,
/// or successive turns of one conversation) to skip recomputing the shared span. Holds at most
/// `capacity` stored sequences; least-recently-used entries (and their KV tensors) are evicted past
/// that. Single-threaded, like the rest of the engine.
pub struct PrefixCache {
    index: PrefixIndex,
    /// Full-sequence per-layer `(keys, values)` for each live entry, keyed by the index's handle.
    kv: HashMap<PrefixId, Vec<(Array, Array)>>,
    stats: PrefixStats,
    reuse_events: Vec<PrefixReuseEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixReuseEvent {
    pub matched_tokens: usize,
    pub stored_sequences: usize,
}

impl PrefixCache {
    /// A cache retaining at most `capacity` stored sequences (LRU eviction past that).
    pub fn new(capacity: usize) -> Self {
        Self {
            index: PrefixIndex::new(capacity),
            kv: HashMap::new(),
            stats: PrefixStats::default(),
            reuse_events: Vec::new(),
        }
    }

    /// Cumulative reuse accounting since construction.
    pub fn stats(&self) -> PrefixStats {
        self.stats
    }

    /// Producer-only evidence of actual prefix reuse (not a caller-supplied boolean).
    pub fn reuse_events(&self) -> &[PrefixReuseEvent] {
        &self.reuse_events
    }

    /// Number of stored sequences currently held.
    pub fn len(&self) -> usize {
        self.kv.len()
    }

    /// Whether the cache holds no sequences.
    pub fn is_empty(&self) -> bool {
        self.kv.is_empty()
    }

    /// Find the longest cached prefix of `prompt` and, on a hit, build a [`ContiguousKvCache`] seeded
    /// with its KV plus the number of prompt tokens it covers — always `< prompt.len()`, so the
    /// suffix prefill always has at least one token (a whole-prompt match recomputes only the final
    /// token). Returns `None` on a miss (the caller prefills cold). Updates [`PrefixStats`].
    fn seed_for(&mut self, prompt: &[i32]) -> Result<Option<(ContiguousKvCache, usize)>> {
        self.stats.lookups += 1;
        let prompt_len = prompt.len();
        let hit = self.index.longest_match(prompt).and_then(|m| {
            let layers = self.kv.get(&m.id)?;
            // Clamp by the query (a whole-prompt match recomputes only the final token) AND by the
            // sequence length the stored tensors actually hold — defence in depth against an index
            // entry that over-states its KV (the budget-finish off-by-one of sc-12455). The query
            // clamp alone is a no-op for a prompt that *extends* the stored sequence.
            let len = m
                .matched_len
                .min(prompt_len.saturating_sub(1))
                .min(stored_seq_len(layers));
            (len > 0).then_some((m.id, len))
        });

        match hit {
            Some((id, len)) => {
                let layers = slice_layers(&self.kv[&id], len)?;
                self.stats.hits += 1;
                self.stats.reused_prefix_tokens += len;
                self.stats.computed_prefill_tokens += prompt_len - len;
                self.reuse_events.push(PrefixReuseEvent {
                    matched_tokens: len,
                    stored_sequences: self.kv.len(),
                });
                Ok(Some((ContiguousKvCache::seeded(layers), len)))
            }
            None => {
                self.stats.computed_prefill_tokens += prompt_len;
                Ok(None)
            }
        }
    }

    /// Store `tokens`' full per-layer KV (from the just-finished `cache`, which the caller resets
    /// next) for future reuse, freeing any LRU entries the insertion evicts or replaces. A no-op if
    /// the cache has no exportable state.
    ///
    /// The entry takes the finished cache's buffers rather than a copy of them
    /// ([`ContiguousKvCache::share_live`]): a copy held a second full KV next to the finished cache
    /// and, once that cache was reset, left its block buffers in MLX's freed-buffer cache. The
    /// shared views are evaluated here, so the entry is materialized now and keeps its buffers for
    /// as long as it lives — nothing it references is released mid-way through a later request.
    ///
    /// An entry this insert retires is released to the system at once. MLX's allocator reuses a
    /// freed buffer only for a request of (nearly) the same size and returns its cache to the
    /// system only near the device working-set limit, so a retired full-context KV would otherwise
    /// stay resident beside the live entry and the next request's cache (sc-20671: a 130k-token
    /// prefix reuse held ~5× its KV in process footprint).
    fn store(&mut self, tokens: Vec<i32>, cache: &ContiguousKvCache) -> Result<()> {
        let Some(layers) = cache.share_live()? else {
            return Ok(());
        };
        let out = self.index.insert(tokens);
        let mut retired = false;
        for evicted in &out.evicted {
            retired |= self.kv.remove(evicted).is_some();
        }
        // `contains` guards the degenerate `capacity == 0` case, where the insert immediately evicts
        // its own entry (so we must not leave an orphan in `kv`).
        if self.index.contains(out.id) {
            // Re-storing an identical sequence replaces its entry.
            retired |= self.kv.insert(out.id, layers).is_some();
        }
        if retired {
            mlx_rs::memory::clear_cache();
        }
        Ok(())
    }
}

/// Like [`generate`](crate::decode::generate), but reusing shared-prefix KV through `prefix_cache`.
///
/// On each call: look up the longest cached prefix of `prompt_ids`, seed the KV cache with it and
/// prefill only the remaining suffix (a miss prefills the whole prompt cold), decode to a stop token
/// / the budget / a mid-stream cancel, then store the `prompt + generated` KV the cache holds for
/// future reuse (on a budget or host-stop finish the last generated token's KV is never fed, so the
/// stored entry excludes that token). The output is **token-for-token identical** to a cold
/// [`generate`](crate::decode::generate) of the same prompt.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any
/// inference.
pub fn generate_cached(
    model: &CausalLm,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    prefix_cache: &mut PrefixCache,
) -> Result<GenerationOutput> {
    generate_cached_with_observer(
        model,
        prompt_ids,
        config,
        cancel,
        on_event,
        prefix_cache,
        None,
        None,
        None,
        None,
    )
}

/// [`generate_cached`] with the same per-step constraint and host-stop seams as
/// [`crate::decode::generate_with`]. Keeping these on the prefix path lets a caller reuse a static
/// system-prompt prefix without giving up generation semantics such as Hugging Face's
/// `no_repeat_ngram_size`.
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
    let mut observed_cache = ObservedCache::default();
    let suffix = input_ids(&prompt_ids[matched_len..]);
    let logits = model.decode_logits(&suffix, cache.as_mut(), matched_len as i32)?;
    if let Some(observer) = observer.as_deref_mut() {
        let logits_f32 = logits.as_dtype(Dtype::Float32)?;
        let values = logits_f32.as_slice::<f32>().to_vec();
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
    if let Some(dense) = cache.as_any_mut().downcast_ref::<ContiguousKvCache>() {
        prefix_cache.store(full, dense)?;
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
    let seed = prefix_cache.seed_for(prompt_ids)?;
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

/// The sequence length (axis [`SEQ_AXIS`]) the stored per-layer KV actually holds — layer 0 speaks
/// for all (layers advance in lockstep), `0` for an empty layer list.
fn stored_seq_len(stored: &[(Array, Array)]) -> usize {
    stored
        .first()
        .map_or(0, |(k, _)| k.shape()[SEQ_AXIS as usize] as usize)
}

/// Slice each layer's `(keys, values)` to the first `len` sequence positions (axis [`SEQ_AXIS`]).
/// The result is a view of the stored buffers, never a copy: the seeded cache's first update grows
/// out of it into a buffer of its own ([`ContiguousKvCache::seeded`]), so a gathered copy was one
/// more full prefix KV allocated per hit and then left in MLX's freed-buffer cache (sc-20671).
/// When `len` already equals the stored length the tensors are cloned as-is. `len` beyond a stored
/// tensor's sequence length is a typed error, so a misaligned index entry can never seed
/// silently-corrupt KV. [`PrefixCache::seed_for`] clamps before calling, so hitting this error
/// means the index/KV alignment invariant broke.
fn slice_layers(stored: &[(Array, Array)], len: usize) -> Result<Vec<(Array, Array)>> {
    use mlx_rs::ops::indexing::TryIndexOp;
    let mut out = Vec::with_capacity(stored.len());
    let len_i32 = i32::try_from(len)
        .map_err(|_| Error::Msg(format!("prefix cache: {len} positions overflow i32")))?;
    for (k, v) in stored {
        let stored_len = k.shape()[SEQ_AXIS as usize];
        if stored_len < len_i32 {
            return Err(Error::Msg(format!(
                "prefix cache: requested {len} positions but the stored KV holds only {stored_len}"
            )));
        }
        if stored_len == len_i32 {
            out.push((k.clone(), v.clone()));
        } else {
            out.push((
                k.try_index((.., .., ..len_i32, ..))?,
                v.try_index((.., .., ..len_i32, ..))?,
            ));
        }
    }
    Ok(out)
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
    identity: PagedCacheIdentity,
    pool: Rc<RefCell<PackedPagePool>>,
    index: PrefixIndex,
    entries: HashMap<PrefixId, PagedPrefixEntry>,
    stats: PagedPrefixStats,
}

impl PagedPrefixCache {
    /// A store of at most `capacity` sequences on `pool`, for caches computed by `model`.
    pub fn new(pool: Rc<RefCell<PackedPagePool>>, model: &str, capacity: usize) -> Self {
        let identity = pool.borrow().identity(model);
        Self {
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

    /// Distinct pages the entries reference.
    pub fn held_pages(&self) -> usize {
        self.entries
            .values()
            .flat_map(|entry| entry.cache.page_ids().iter().copied())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
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
    /// identity is not the store's or the cache lives on another pool.
    pub fn insert(
        &mut self,
        identity: &PagedCacheIdentity,
        tokens: &[i32],
        cache: &PagedPackedKvCache,
    ) -> Result<bool> {
        if self.refuse(identity).is_some() {
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
        self.store(tokens[..held].to_vec(), entry);
        Ok(true)
    }

    fn store(&mut self, tokens: Vec<i32>, cache: PagedPackedKvCache) {
        let outcome = self.index.insert(tokens.clone());
        for evicted in &outcome.evicted {
            self.entries.remove(evicted);
        }
        if self.index.contains(outcome.id) {
            self.entries
                .insert(outcome.id, PagedPrefixEntry { tokens, cache });
        }
        self.stats.stored += 1;
    }

    /// Every stored sequence as `(tokens, snapshot)`, for saving across a process restart.
    pub fn snapshots(&self) -> Result<Vec<(Vec<i32>, PagedCacheSnapshot)>> {
        self.entries
            .values()
            .map(|entry| {
                Ok((
                    entry.tokens.clone(),
                    entry.cache.snapshot(&self.identity.model)?,
                ))
            })
            .collect()
    }

    /// Restore a saved sequence into the store, read by `reader`. Refused — and counted — unless
    /// the snapshot carries exactly the store's identity (model, KV format and page layout
    /// versions, geometry) and holds `tokens.len()` positions.
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
        if snapshot.tokens() != tokens.len() {
            self.stats.refused += 1;
            return Err(Error::Unsupported(format!(
                "paged prefix restore refused: the snapshot holds {} positions, not the {} tokens \
                 it is stored under",
                snapshot.tokens(),
                tokens.len()
            )));
        }
        let cache =
            PagedPackedKvCache::restore(snapshot, self.pool.clone(), reader, &self.identity.model)?;
        self.store(tokens, cache);
        Ok(())
    }

    /// Drop every entry (and its page references).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.index = PrefixIndex::new(self.index.capacity());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kv(seq: i32) -> Vec<(Array, Array)> {
        let t = || Array::zeros::<f32>(&[1, 1, seq, 2]).unwrap();
        vec![(t(), t())]
    }

    /// Defence in depth (sc-12455): asking for more positions than the stored tensors hold is a
    /// typed error — never a silent out-of-range `take_axis` gather (MLX gathers clamp instead of
    /// erroring, which would seed corrupt KV).
    #[test]
    fn slice_layers_rejects_len_past_stored() {
        let err = slice_layers(&kv(4), 5).unwrap_err();
        assert!(
            err.to_string().contains("stored KV holds only 4"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn slice_layers_views_and_clones() {
        let sliced = slice_layers(&kv(4), 3).unwrap();
        assert_eq!(sliced[0].0.shape()[SEQ_AXIS as usize], 3);
        let cloned = slice_layers(&kv(4), 4).unwrap();
        assert_eq!(cloned[0].0.shape()[SEQ_AXIS as usize], 4);
    }

    /// Whether MLX has materialized `array`'s buffer (its lazy graph, if any, has been evaluated).
    #[cfg(target_os = "macos")]
    fn is_available(array: &Array) -> bool {
        let mut available = false;
        // SAFETY: `array` is a live MLX array handle and `available` outlives the call.
        let status = unsafe { mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr()) };
        assert_eq!(status, 0, "mlx array status query failed");
        available
    }

    /// Tiny synthetic Llama (4 KV heads × head dim 64, 2 layers) for the store-ownership test.
    #[cfg(target_os = "macos")]
    fn tiny_model() -> CausalLm {
        synthetic_llama(256, 128, 2, 4, 64)
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

    /// sc-20671: a stored prefix entry is materialized at store time, over its live positions. A
    /// lazy entry kept an unevaluated reference to the finished cache's padded block buffer, so a
    /// campaign's prefill-window baseline (sampled after seeding the store) counted that buffer
    /// and the hit's first evaluation released it mid-prefill — the phase-local floor then
    /// exceeded the measured active bytes. The entry now holds the finished cache's buffers
    /// themselves (evaluated views), which it keeps until it is itself retired.
    #[cfg(target_os = "macos")]
    #[test]
    fn stored_prefix_entries_are_materialized_at_their_live_length() {
        let model = tiny_model();
        // 40 prompt tokens: well inside one 256-position block, and inside the vocabulary (MLX's
        // embedding gather is not bounds-checked).
        let prompt = (0..40).map(|i| i % 63 + 1).collect::<Vec<i32>>();
        let config = GenerationConfig {
            max_new_tokens: 1,
            seed: Some(0),
            ..Default::default()
        };
        let mut store = PrefixCache::new(2);
        generate_cached(
            &model,
            &prompt,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            &mut store,
        )
        .unwrap();
        let [layers] = store.kv.values().collect::<Vec<_>>()[..] else {
            panic!("one stored sequence");
        };
        assert_eq!(layers.len(), 2);
        for (keys, values) in layers {
            // A budget finish never feeds the last generated token, so the entry is the prompt.
            assert_eq!(keys.shape()[SEQ_AXIS as usize], 40);
            assert_eq!(values.shape()[SEQ_AXIS as usize], 40);
            assert!(
                is_available(keys) && is_available(values),
                "a stored prefix entry must be materialized, not a lazy graph over the finished cache"
            );
        }
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
        let mut store = PrefixCache::new(2);
        let cancel = CancelFlag::new();
        // Weights are materialized by the first forward; take the baseline after one tiny call on
        // a separate store so only the long prompt's buffers count.
        generate_cached(
            &model,
            &prompt[..4],
            &config,
            &cancel,
            &mut |_| {},
            &mut PrefixCache::new(1),
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

    /// Defence in depth (sc-12455): if an index entry ever over-states its KV again (the pre-fix
    /// budget-finish state), `seed_for` clamps the match to the positions the tensors actually
    /// hold instead of gathering out of range.
    #[test]
    fn seed_for_clamps_match_to_stored_kv() {
        let mut pc = PrefixCache::new(4);
        // Manufacture the inconsistent state directly: 6 indexed tokens, 5 positions of KV.
        let tokens: Vec<i32> = vec![1, 2, 3, 4, 5, 6];
        let out = pc.index.insert(tokens.clone());
        pc.kv.insert(out.id, kv(5));

        // An extending prompt matches all 6 indexed tokens; the seed must clamp to the 5 stored.
        let mut prompt = tokens;
        prompt.extend_from_slice(&[7, 8]);
        let (cache, len) = pc.seed_for(&prompt).unwrap().expect("hit");
        assert_eq!(len, 5);
        assert_eq!(cache.offset(), 5);
        assert_eq!(pc.stats().reused_prefix_tokens, 5);
    }
}
