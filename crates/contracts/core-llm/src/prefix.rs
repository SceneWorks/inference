//! Backend-neutral shared-prefix bookkeeping for KV reuse (epic 7153, story 7168).
//!
//! Many requests share a leading run of tokens — a common system prompt, a few-shot preamble, the
//! growing history of a multi-turn chat. The keys/values for that shared run are **identical** across
//! the requests: a causal decoder's K/V at position `i` depends only on tokens `0..=i`, so two prompts
//! agreeing on their first `n` tokens have bit-identical KV for those `n` positions regardless of what
//! follows. Recomputing it each time is pure waste; the prefix cache reuses it.
//!
//! This module owns only the **policy**: which stored token sequence shares the longest prefix with a
//! new prompt, and which entries to evict when the store is full. It is tensor-free, so the same
//! bookkeeping drives every backend ([`mlx-llm`], later `candle-llm`) — a backend pairs each
//! [`PrefixId`] with its own per-layer KV tensors and, on a [`PrefixMatch`], seeds a cache to
//! `matched_len` positions and prefills only the remaining suffix.
//!
//! Matching is **token-granular** (any partial-prefix overlap is reused, not just whole entries) and
//! the store is a small **LRU** bounded by entry count; eviction hands back the dropped [`PrefixId`]s
//! so the backend can free their tensors in lockstep. Block-granular sharing with copy-on-write is
//! the paged cache's job (story 7169); this is the simpler contiguous-friendly cousin that lands
//! first.
//!
//! [`mlx-llm`]: https://github.com/SceneWorks/mlx-llm
//!
//! ```
//! use core_llm::prefix::PrefixIndex;
//!
//! let mut idx = PrefixIndex::new(8);
//! // First request: nothing to reuse; the backend prefills it cold and stores its KV.
//! assert!(idx.longest_match(&[1, 2, 3, 4]).is_none());
//! let sys = idx.insert(vec![1, 2, 3, 4]).id;
//!
//! // A later request sharing the first three tokens reuses them (matched_len = 3).
//! let m = idx.longest_match(&[1, 2, 3, 9, 9]).expect("shares a prefix");
//! assert_eq!(m.id, sys);
//! assert_eq!(m.matched_len, 3);
//! ```

use std::collections::VecDeque;

/// An opaque handle to a stored prefix entry, stable until the entry is evicted.
///
/// A backend uses it as the key into its own table of per-entry KV tensors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrefixId(pub u64);

/// The result of a [`PrefixIndex::longest_match`]: which stored entry shared the longest prefix, and
/// how many leading tokens it shared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixMatch {
    /// The matched entry.
    pub id: PrefixId,
    /// Number of leading tokens shared with the queried prompt (always `>= 1` — a zero-length match
    /// is reported as `None`). May equal the queried prompt's length (a full match); the backend is
    /// expected to recompute at least the final token so a forward step always has a query.
    pub matched_len: usize,
}

/// The outcome of an [`PrefixIndex::insert`]: the handle for the stored sequence and any entries the
/// insertion evicted (so the backend frees their tensors).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InsertOutcome {
    /// Handle for the inserted (or refreshed) sequence.
    pub id: PrefixId,
    /// Entries dropped to stay within capacity, in eviction order (least-recently-used first).
    pub evicted: Vec<PrefixId>,
}

/// One stored token sequence and its handle.
#[derive(Clone, Debug)]
struct Entry {
    id: PrefixId,
    tokens: Vec<i32>,
}

/// An LRU index over stored token sequences with longest-common-prefix lookup (story 7168).
///
/// The backend drives it per request: [`PrefixIndex::longest_match`] before prefill to find reusable
/// KV, then [`PrefixIndex::insert`] after generation to store the request's full token sequence
/// (prompt + generated) for future reuse. Entries are kept most-recently-used at the back; both a
/// successful match and a re-insert refresh recency.
#[derive(Clone, Debug)]
pub struct PrefixIndex {
    /// Max number of stored sequences. Least-recently-used entries are evicted past this.
    capacity: usize,
    /// Monotonic id source; ids are never reused, so a stale [`PrefixId`] never aliases a new entry.
    next_id: u64,
    /// Stored sequences, least-recently-used at the front, most-recently-used at the back.
    entries: VecDeque<Entry>,
}

impl PrefixIndex {
    /// A fresh index holding at most `capacity` sequences (LRU eviction past that). A `capacity` of
    /// `0` stores nothing — every [`PrefixIndex::insert`] immediately evicts what it inserted.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            next_id: 0,
            entries: VecDeque::new(),
        }
    }

    /// Max number of sequences the index retains.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of sequences currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index holds no sequences.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `id` is still stored (not evicted).
    pub fn contains(&self, id: PrefixId) -> bool {
        self.entries.iter().any(|e| e.id == id)
    }

    /// The stored entry sharing the longest leading run of tokens with `tokens`, and that shared
    /// length — or `None` if no stored sequence shares even the first token. A hit refreshes the
    /// matched entry's recency (it becomes most-recently-used).
    ///
    /// Ties (two entries sharing the same length) resolve to the most-recently-used; the choice is
    /// immaterial to correctness because entries sharing a prefix have bit-identical KV over it.
    pub fn longest_match(&mut self, tokens: &[i32]) -> Option<PrefixMatch> {
        if tokens.is_empty() {
            return None;
        }
        // Scan back-to-front so an MRU entry wins a tie naturally.
        let mut best: Option<(usize, usize)> = None; // (deque index, matched_len)
        for (i, e) in self.entries.iter().enumerate().rev() {
            let n = common_prefix_len(&e.tokens, tokens);
            if n > 0 && best.is_none_or(|(_, b)| n > b) {
                best = Some((i, n));
            }
        }
        let (idx, matched_len) = best?;
        let id = self.touch(idx);
        Some(PrefixMatch { id, matched_len })
    }

    /// Store `tokens` for future reuse, returning its handle and any evicted entries.
    ///
    /// An exact re-insert of an already-stored sequence refreshes that entry (same [`PrefixId`], no
    /// eviction) rather than duplicating it — so a backend can re-store a deterministic
    /// prompt+generation without leaking a slot.
    pub fn insert(&mut self, tokens: Vec<i32>) -> InsertOutcome {
        if let Some(idx) = self.entries.iter().position(|e| e.tokens == tokens) {
            let id = self.touch(idx);
            return InsertOutcome {
                id,
                evicted: Vec::new(),
            };
        }
        let id = PrefixId(self.next_id);
        self.next_id += 1;
        self.entries.push_back(Entry { id, tokens });
        let mut evicted = Vec::new();
        while self.entries.len() > self.capacity {
            if let Some(old) = self.entries.pop_front() {
                evicted.push(old.id);
            }
        }
        InsertOutcome { id, evicted }
    }

    /// Move the entry at deque index `idx` to the back (most-recently-used) and return its id. `idx`
    /// must be in range.
    fn touch(&mut self, idx: usize) -> PrefixId {
        let entry = self.entries.remove(idx).expect("index in range");
        let id = entry.id;
        self.entries.push_back(entry);
        id
    }
}

/// Length of the shared leading run of two token slices.
fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

// ---------------------------------------------------------------------------------------------
// The byte-budgeted cross-turn store (epic sc-24432, story sc-24437).
// ---------------------------------------------------------------------------------------------

/// The prefix-cache budget a load reserves when [`LoadSpec::prefix_cache_bytes`] is unset: 1 GiB,
/// clamped to what the load's own admission leaves ([`prefix_cache_budget`]).
///
/// **Default ON** (epic sc-24432 E5: a speed-up ships on unless a measured regression justifies
/// off). The epic's terminal campaign (story sc-24446) measures TTFT and parity with the cache on
/// against `prefix_cache_bytes: Some(0)` and confirms this value or flips it to `0`; this constant
/// is the one place that decision lands. The budget is admission-honest either way: a load
/// reports what it settled ([`LoadReport::prefix_cache_bytes`]) and a backend's load estimate
/// names what it asks for, beside — never inside — what the load requires.
///
/// [`LoadSpec::prefix_cache_bytes`]: crate::LoadSpec::prefix_cache_bytes
/// [`LoadReport::prefix_cache_bytes`]: crate::LoadReport::prefix_cache_bytes
pub const DEFAULT_PREFIX_CACHE_BYTES: u64 = 1 << 30;

/// The prefix-cache budget a load asks for before admission clamps it: the requested budget, or
/// [`DEFAULT_PREFIX_CACHE_BYTES`] when unset. A load estimate reports it beside the bytes the load
/// requires; [`prefix_cache_budget`] settles it against the load's headroom.
pub fn requested_prefix_cache_bytes(requested: Option<u64>) -> u64 {
    requested.unwrap_or(DEFAULT_PREFIX_CACHE_BYTES)
}

/// The prefix-cache byte budget a load settles (E7): the requested budget (`None`:
/// [`DEFAULT_PREFIX_CACHE_BYTES`]) clamped to the headroom the load's admission leaves —
/// `available - load_required`, the same two figures the load was admitted on. So the cache can
/// never push the load past its admission budget, and a load that fits without the cache never
/// fails because of it; a tight host simply gets a smaller (or zero) cache.
pub fn prefix_cache_budget(requested: Option<u64>, load_required: u64, available: u64) -> u64 {
    requested_prefix_cache_bytes(requested).min(available.saturating_sub(load_required))
}

/// Why a multimodal request never reads or feeds the cross-turn prefix cache (story sc-24437):
/// the cache keys on token ids, which cannot tell two images or clips behind the same placeholder
/// ids apart. One string on both backends (E8).
pub const PREFIX_MULTIMODAL_BYPASS: &str =
    "a multimodal prompt is never cached — its image / video / audio rows are not in the token key";

/// Why a request's own prefix-cache snapshot was not taken: request admission could not hold it
/// beside the request (E7).
pub const PREFIX_NOT_ADMITTED: &str =
    "not kept: admission could not hold this request's snapshot beside it";

/// Why a request's own prefix-cache snapshot was dropped: its copy failed.
pub const PREFIX_COPY_FAILED: &str = "not kept: the snapshot copy failed";

/// Why a request on a paged KV cache keeps no snapshot: its blocks are shared through the pool's
/// own copy-on-write, and the prefix cache does not copy them out.
pub const PREFIX_PAGED_NOT_SNAPSHOTTED: &str = "not kept: paged KV backing is not snapshotted";

/// The cross-turn prefix cache's part in a request before any lookup (story sc-24437) — the
/// model-agnostic rule both backends start from (E8): `off` when the load settled a zero budget,
/// `bypassed` with [`PREFIX_MULTIMODAL_BYPASS`] for a multimodal prompt, else `miss` (the lookup
/// may still turn it into a `hit`). Only a `miss` request reads or feeds the cache.
pub fn prefix_path_before_lookup(
    prefix_on: bool,
    multimodal: bool,
) -> (&'static str, Option<&'static str>) {
    if !prefix_on {
        ("off", None)
    } else if multimodal {
        ("bypassed", Some(PREFIX_MULTIMODAL_BYPASS))
    } else {
        ("miss", None)
    }
}

/// How a stored entry may be reused by a later prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefixReuse {
    /// Any leading run of the stored tokens: a softmax KV cache is sliced by offset, so a prompt
    /// sharing only part of the entry still reuses that part.
    AnyPrefix,
    /// Only the whole stored sequence: a recurrent (DeltaNet) state exists only at the boundary it
    /// was snapshotted at, so the prompt must extend the entry's tokens exactly.
    WholeEntry,
}

/// Cumulative accounting of a [`PrefixStore`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixStats {
    /// [`PrefixStore::lookup`] calls.
    pub lookups: usize,
    /// Lookups that reused a stored prefix.
    pub hits: usize,
    /// Prompt positions whose state came from the store across every hit (prefill skipped).
    pub reused_prefix_tokens: usize,
    /// Prompt positions left to prefill across every lookup (the whole prompt on a miss, the
    /// suffix past the reused span on a hit).
    pub computed_prefill_tokens: usize,
    /// Entries stored.
    pub inserted: usize,
    /// Entries dropped least-recently-used first — to fit an insert under the budget, or to free
    /// room for a request ([`PrefixStore::reclaim_for`]).
    pub evicted: usize,
    /// Entries refused because they alone exceed the budget.
    pub rejected: usize,
}

/// One store entry.
#[derive(Debug)]
struct StoredPrefix<S> {
    tokens: Vec<i32>,
    reuse: PrefixReuse,
    bytes: u64,
    state: S,
}

/// A hit: the stored state and how many leading prompt positions it covers.
#[derive(Debug)]
pub struct PrefixHit<'a, S> {
    /// The stored state (an immutable snapshot the backend restores from, never writes).
    pub state: &'a S,
    /// Leading prompt tokens the state covers — always `< prompt.len()`, so the suffix prefill has
    /// at least one token.
    pub reused: usize,
    /// How the entry is reused ([`PrefixReuse::AnyPrefix`] entries are sliced to `reused`).
    pub reuse: PrefixReuse,
}

/// What [`PrefixStore::admit`] settled for a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixAdmission {
    /// The availability to admit the request against (what was free plus what was evicted).
    pub available: u64,
    /// Whether the request may take (and the cache keep) its snapshot; when `true` the request is
    /// admitted with the snapshot's bytes on top of its own.
    pub snapshot: bool,
}

/// What [`PrefixStore::insert`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixInsert {
    /// Whether the entry is now held.
    pub stored: bool,
    /// Entries dropped to make room (or replaced / superseded by this one).
    pub evicted: usize,
}

/// The cross-turn prefix cache's policy (story sc-24437): per-entry backend state `S` (KV tensors,
/// or a hybrid decoder's KV plus recurrent state) under a **byte** budget with least-recently-used
/// eviction, and longest-usable-prefix lookup that respects each entry's [`PrefixReuse`].
///
/// The store is tensor-free: each backend owns one per loaded provider, keyed by the tokens the
/// entry's state was computed from, and reports each entry's resident bytes. The invariant the
/// backend's admission leans on (E7): [`resident_bytes`](Self::resident_bytes) never exceeds
/// [`budget_bytes`](Self::budget_bytes), and a request that needs room evicts entries first
/// ([`reclaim_for`](Self::reclaim_for)).
#[derive(Debug)]
pub struct PrefixStore<S> {
    budget_bytes: u64,
    resident_bytes: u64,
    /// Least-recently-used at the front.
    entries: VecDeque<StoredPrefix<S>>,
    stats: PrefixStats,
}

impl<S> PrefixStore<S> {
    /// An empty store holding at most `budget_bytes` of entry state (`0` stores nothing).
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            resident_bytes: 0,
            entries: VecDeque::new(),
            stats: PrefixStats::default(),
        }
    }

    /// The byte budget.
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// Bytes the held entries report, always `<= budget_bytes`.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Entries held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cumulative accounting since construction.
    pub fn stats(&self) -> PrefixStats {
        self.stats
    }

    /// The held entries' token sequences, least-recently-used first (diagnostics and tests).
    pub fn keys(&self) -> Vec<&[i32]> {
        self.entries.iter().map(|e| e.tokens.as_slice()).collect()
    }

    /// The entry reusing the most leading tokens of `prompt`, or `None` (a miss). `held` says,
    /// per entry, how many positions its state really holds — `None` when this request cannot use
    /// it at all — and the reuse never exceeds it (defence in depth against a key that over-states
    /// its state, sc-12455). An [`AnyPrefix`](PrefixReuse::AnyPrefix) entry reuses its shared
    /// leading run, a [`WholeEntry`](PrefixReuse::WholeEntry) one only its whole sequence (and
    /// only when its state holds exactly that); either way the reuse stops short of the whole
    /// prompt so a forward always has a query. Ties go to the most-recently-used entry, and a hit
    /// becomes most-recently-used. Updates [`PrefixStats`] with the reuse actually granted.
    pub fn lookup(
        &mut self,
        prompt: &[i32],
        held: impl Fn(&S) -> Option<usize>,
    ) -> Option<PrefixHit<'_, S>> {
        self.stats.lookups += 1;
        let limit = prompt.len().saturating_sub(1);
        let mut best: Option<(usize, usize)> = None; // (deque index, reused)
        for (i, e) in self.entries.iter().enumerate().rev() {
            let Some(held) = held(&e.state) else {
                continue;
            };
            let shared = common_prefix_len(&e.tokens, prompt);
            let reused = match e.reuse {
                PrefixReuse::AnyPrefix => shared.min(limit).min(held),
                PrefixReuse::WholeEntry
                    if shared == e.tokens.len() && shared <= limit && held == shared =>
                {
                    shared
                }
                PrefixReuse::WholeEntry => 0,
            };
            if reused > 0 && best.is_none_or(|(_, b)| reused > b) {
                best = Some((i, reused));
            }
        }
        let Some((idx, reused)) = best else {
            self.stats.computed_prefill_tokens += prompt.len();
            return None;
        };
        let entry = self.entries.remove(idx).expect("index in range");
        self.entries.push_back(entry);
        self.stats.hits += 1;
        self.stats.reused_prefix_tokens += reused;
        self.stats.computed_prefill_tokens += prompt.len() - reused;
        let entry = self.entries.back().expect("just pushed");
        Some(PrefixHit {
            state: &entry.state,
            reused,
            reuse: entry.reuse,
        })
    }

    /// Hold `state` for the prefix `tokens` (its `bytes` as the backend measured them). An entry
    /// larger than the whole budget is refused without evicting anything. An entry with the same
    /// tokens is replaced, and an [`AnyPrefix`](PrefixReuse::AnyPrefix) entry supersedes every
    /// `AnyPrefix` entry whose tokens it extends (it reuses everything they could). Then
    /// least-recently-used entries are evicted until the new one fits.
    pub fn insert(
        &mut self,
        tokens: Vec<i32>,
        reuse: PrefixReuse,
        bytes: u64,
        state: S,
    ) -> PrefixInsert {
        if tokens.is_empty() || bytes > self.budget_bytes {
            self.stats.rejected += 1;
            return PrefixInsert::default();
        }
        let mut evicted = 0;
        let mut i = 0;
        while i < self.entries.len() {
            let e = &self.entries[i];
            let superseded = e.tokens == tokens
                || (reuse == PrefixReuse::AnyPrefix
                    && e.reuse == PrefixReuse::AnyPrefix
                    && tokens.starts_with(&e.tokens));
            if superseded {
                let gone = self.entries.remove(i).expect("index in range");
                self.resident_bytes -= gone.bytes;
                evicted += 1;
            } else {
                i += 1;
            }
        }
        while self.resident_bytes + bytes > self.budget_bytes {
            let gone = self
                .entries
                .pop_front()
                .expect("resident bytes > 0 means an entry is held");
            self.resident_bytes -= gone.bytes;
            evicted += 1;
        }
        self.resident_bytes += bytes;
        self.entries.push_back(StoredPrefix {
            tokens,
            reuse,
            bytes,
            state,
        });
        self.stats.inserted += 1;
        self.stats.evicted += evicted;
        PrefixInsert {
            stored: true,
            evicted,
        }
    }

    /// Make room for a request that needs `required` bytes when `available` are free (E7): when
    /// the shortfall can be covered by evicting held entries, evict least-recently-used entries
    /// until it is, and return the availability the request is then admitted against (`available`
    /// plus what was freed). Otherwise — no shortfall, or one the whole store could not cover —
    /// evict nothing and return `available`, so the caller's admission refuses exactly as it
    /// would without the cache.
    pub fn reclaim_for(&mut self, required: u64, available: u64) -> u64 {
        let shortfall = required.saturating_sub(available);
        if shortfall == 0 || shortfall > self.resident_bytes {
            return available;
        }
        available.saturating_add(self.evict_at_least(shortfall))
    }

    /// Request admission with the cache in the picture (E7). The request needs `required` bytes,
    /// `available` are free, and running it would take a snapshot of up to `snapshot_bytes` for
    /// the cache to keep (a copy made during the request, which the request's own estimate does
    /// not cover). Least-recently-used entries are evicted until the request **and** the snapshot
    /// fit in memory and the snapshot fits in the budget beside what stays resident; then the
    /// request is admitted against the returned availability with `required + snapshot_bytes`.
    /// When that cannot be reached the request runs without keeping a snapshot
    /// ([`PrefixAdmission::snapshot`] is `false`) and is admitted as [`reclaim_for`](Self::reclaim_for)
    /// admits it — so caching never makes a request fail, and never takes memory admission did
    /// not grant.
    pub fn admit(&mut self, required: u64, snapshot_bytes: u64, available: u64) -> PrefixAdmission {
        if snapshot_bytes <= self.budget_bytes {
            let over_budget =
                (self.resident_bytes + snapshot_bytes).saturating_sub(self.budget_bytes);
            let short = required
                .saturating_add(snapshot_bytes)
                .saturating_sub(available);
            let need = over_budget.max(short);
            if need <= self.resident_bytes {
                let freed = self.evict_at_least(need);
                return PrefixAdmission {
                    available: available.saturating_add(freed),
                    snapshot: true,
                };
            }
        }
        PrefixAdmission {
            available: self.reclaim_for(required, available),
            snapshot: false,
        }
    }

    /// Evict least-recently-used entries until at least `bytes` are freed (the caller has checked
    /// that the resident bytes cover it); returns what was freed.
    fn evict_at_least(&mut self, bytes: u64) -> u64 {
        let mut freed = 0u64;
        while freed < bytes {
            let gone = self
                .entries
                .pop_front()
                .expect("the caller checked the resident bytes cover it");
            self.resident_bytes -= gone.bytes;
            freed += gone.bytes;
            self.stats.evicted += 1;
        }
        freed
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.stats.evicted += self.entries.len();
        self.entries.clear();
        self.resident_bytes = 0;
    }
}

#[cfg(test)]
mod store_tests {
    use super::*;

    #[test]
    fn a_whole_entry_is_reused_only_when_the_prompt_extends_it() {
        let mut store = PrefixStore::new(1_000);
        store.insert(vec![1, 2, 3], PrefixReuse::WholeEntry, 10, "b3");
        assert!(
            store.lookup(&[1, 2, 9, 9], |_| Some(3)).is_none(),
            "partial"
        );
        assert!(store.lookup(&[1, 2, 3], |_| Some(3)).is_none(), "no suffix");
        assert!(
            store.lookup(&[1, 2, 3, 4], |_| Some(2)).is_none(),
            "a state that does not hold exactly the key"
        );
        let hit = store.lookup(&[1, 2, 3, 4], |_| Some(3)).unwrap();
        assert_eq!((hit.reused, *hit.state), (3, "b3"));
    }

    #[test]
    fn an_any_prefix_entry_is_sliced_and_leaves_one_token_to_prefill() {
        let mut store = PrefixStore::new(1_000);
        store.insert(vec![1, 2, 3, 4, 5], PrefixReuse::AnyPrefix, 10, ());
        assert_eq!(
            store
                .lookup(&[1, 2, 7], |_| Some(usize::MAX))
                .unwrap()
                .reused,
            2
        );
        assert_eq!(
            store
                .lookup(&[1, 2, 3], |_| Some(usize::MAX))
                .unwrap()
                .reused,
            2
        );
        assert_eq!(
            store
                .lookup(&[1, 2, 3, 4, 5, 6], |_| Some(usize::MAX))
                .unwrap()
                .reused,
            5
        );
        assert!(store.lookup(&[9], |_| Some(usize::MAX)).is_none());
        // A state holding fewer positions than its key clamps the reuse, and the stats say so.
        assert_eq!(
            store
                .lookup(&[1, 2, 3, 4, 5, 6], |_| Some(4))
                .unwrap()
                .reused,
            4
        );
        let s = store.stats();
        assert_eq!((s.lookups, s.hits, s.reused_prefix_tokens), (5, 4, 13));
        assert_eq!(s.computed_prefill_tokens, 1 + 1 + 1 + 1 + 2);
    }

    #[test]
    fn ineligible_entries_are_skipped() {
        let mut store = PrefixStore::new(1_000);
        store.insert(vec![1, 2, 3], PrefixReuse::WholeEntry, 10, (false, 3));
        store.insert(vec![1, 2], PrefixReuse::WholeEntry, 10, (true, 2));
        let hit = store
            .lookup(&[1, 2, 3, 4], |&(mtp, len)| mtp.then_some(len))
            .unwrap();
        assert_eq!(
            hit.reused, 2,
            "the longer entry lacks what the request needs"
        );
    }

    /// AC3: past the budget the least-recently-used entry goes, and resident bytes never exceed
    /// the budget.
    #[test]
    fn the_least_recently_used_entry_is_evicted_and_bytes_stay_in_budget() {
        let mut store = PrefixStore::new(100);
        store.insert(vec![1], PrefixReuse::WholeEntry, 40, 'a');
        store.insert(vec![2], PrefixReuse::WholeEntry, 40, 'b');
        // Touch `a`, so `b` is now least recently used.
        assert!(store.lookup(&[1, 0], |_| Some(1)).is_some());
        let out = store.insert(vec![3], PrefixReuse::WholeEntry, 40, 'c');
        assert_eq!(
            out,
            PrefixInsert {
                stored: true,
                evicted: 1
            }
        );
        assert_eq!(store.keys(), vec![&[1][..], &[3][..]]);
        assert_eq!(store.resident_bytes(), 80);
        assert!(store.resident_bytes() <= store.budget_bytes());
        // An entry needing everything evicts everything else; one past the budget is refused.
        store.insert(vec![4], PrefixReuse::WholeEntry, 100, 'd');
        assert_eq!((store.len(), store.resident_bytes()), (1, 100));
        let refused = store.insert(vec![5], PrefixReuse::WholeEntry, 101, 'e');
        assert!(!refused.stored);
        assert_eq!((store.len(), store.resident_bytes()), (1, 100));
        assert_eq!(store.stats().rejected, 1);
    }

    #[test]
    fn a_longer_any_prefix_entry_supersedes_the_one_it_extends() {
        let mut store = PrefixStore::new(1_000);
        store.insert(vec![1, 2], PrefixReuse::AnyPrefix, 10, ());
        store.insert(vec![7, 8], PrefixReuse::AnyPrefix, 10, ());
        let out = store.insert(vec![1, 2, 3], PrefixReuse::AnyPrefix, 15, ());
        assert_eq!(out.evicted, 1);
        assert_eq!(store.keys(), vec![&[7, 8][..], &[1, 2, 3][..]]);
        assert_eq!(store.resident_bytes(), 25);
        // A re-insert of the same tokens replaces (no duplicate).
        store.insert(vec![7, 8], PrefixReuse::AnyPrefix, 12, ());
        assert_eq!(store.len(), 2);
        assert_eq!(store.resident_bytes(), 27);
    }

    /// E7: a request short of memory evicts exactly enough, one that the whole store could not
    /// cover evicts nothing, and the boundary (`required == available + resident`) is admitted.
    #[test]
    fn a_request_reclaims_room_up_to_the_boundary() {
        let fill = || {
            let mut store = PrefixStore::new(1_000);
            store.insert(vec![1], PrefixReuse::WholeEntry, 300, ());
            store.insert(vec![2], PrefixReuse::WholeEntry, 300, ());
            store
        };
        let mut store = fill();
        assert_eq!(store.reclaim_for(500, 500), 500, "no shortfall");
        assert_eq!(store.len(), 2);
        assert_eq!(store.reclaim_for(700, 500), 800, "one entry covers 200");
        assert_eq!(
            (store.keys(), store.resident_bytes()),
            (vec![&[2][..]], 300)
        );

        let mut store = fill();
        assert_eq!(store.reclaim_for(1_100, 500), 1_100, "exactly the boundary");
        assert!(store.is_empty());
        assert!(crate::admit_request_memory(1_100, 1_100).is_ok());

        let mut store = fill();
        assert_eq!(store.reclaim_for(1_101, 500), 500, "past the boundary");
        assert_eq!(store.len(), 2, "a refused request keeps the cache");
        assert!(crate::admit_request_memory(1_101, 500).is_err());
    }

    /// E7: a request keeps its snapshot only when both fit — evicting exactly enough for memory
    /// and budget — and otherwise runs without it, admitted as without the cache.
    #[test]
    fn a_request_snapshot_is_admitted_only_when_memory_and_budget_hold_it() {
        let fill = || {
            let mut store = PrefixStore::new(1_000);
            store.insert(vec![1], PrefixReuse::WholeEntry, 400, ());
            store.insert(vec![2], PrefixReuse::WholeEntry, 400, ());
            store
        };
        // Budget room: 800 held + 300 would exceed 1 000, so one entry goes.
        let mut store = fill();
        let a = store.admit(100, 300, 10_000);
        assert_eq!(
            a,
            PrefixAdmission {
                available: 10_400,
                snapshot: true
            }
        );
        assert_eq!(store.keys(), vec![&[2][..]]);
        // Memory room at the exact boundary: 500 + 300 with nothing free needs all 800 held.
        let mut store = fill();
        let a = store.admit(500, 300, 0);
        assert_eq!(
            a,
            PrefixAdmission {
                available: 800,
                snapshot: true
            }
        );
        assert!(crate::admit_request_memory(500 + 300, a.available).is_ok());
        assert!(store.is_empty());
        // One byte past what eviction can free: no snapshot, the request alone is reclaimed for.
        let mut store = fill();
        let a = store.admit(501, 300, 0);
        assert_eq!(
            a,
            PrefixAdmission {
                available: 800,
                snapshot: false
            }
        );
        assert!(crate::admit_request_memory(501, a.available).is_ok());
        // A snapshot past the whole budget is never taken.
        let mut store = fill();
        let a = store.admit(10, 1_001, 10_000);
        assert_eq!(
            a,
            PrefixAdmission {
                available: 10_000,
                snapshot: false
            }
        );
        assert_eq!(store.len(), 2);
    }

    /// E7: the budget a load reserves is what its admission leaves, never more.
    #[test]
    fn the_load_budget_is_clamped_to_the_admission_headroom() {
        assert_eq!(
            prefix_cache_budget(None, 10, 1 << 40),
            DEFAULT_PREFIX_CACHE_BYTES
        );
        assert_eq!(prefix_cache_budget(Some(500), 100, 1_000), 500);
        assert_eq!(
            prefix_cache_budget(Some(500), 600, 1_000),
            400,
            "the boundary"
        );
        assert_eq!(prefix_cache_budget(Some(500), 1_000, 1_000), 0);
        assert_eq!(prefix_cache_budget(Some(0), 0, 1_000), 0, "disabled");
        assert_eq!(
            requested_prefix_cache_bytes(None),
            DEFAULT_PREFIX_CACHE_BYTES
        );
        assert_eq!(requested_prefix_cache_bytes(Some(7)), 7);
        let (required, available) = (600u64, 1_000u64);
        let budget = prefix_cache_budget(Some(u64::MAX), required, available);
        assert!(crate::admit_request_memory(required + budget, available).is_ok());
        assert!(crate::admit_request_memory(required + budget + 1, available).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E8: the shared pre-lookup rule — off, bypassed (named) for a multimodal prompt, else miss.
    #[test]
    fn the_pre_lookup_path_is_off_bypassed_or_miss() {
        assert_eq!(prefix_path_before_lookup(false, true), ("off", None));
        assert_eq!(
            prefix_path_before_lookup(true, true),
            ("bypassed", Some(PREFIX_MULTIMODAL_BYPASS))
        );
        assert_eq!(prefix_path_before_lookup(true, false), ("miss", None));
    }

    #[test]
    fn empty_index_never_matches() {
        let mut idx = PrefixIndex::new(4);
        assert!(idx.is_empty());
        assert_eq!(idx.longest_match(&[1, 2, 3]), None);
        assert_eq!(idx.longest_match(&[]), None);
    }

    #[test]
    fn insert_then_exact_match() {
        let mut idx = PrefixIndex::new(4);
        let out = idx.insert(vec![1, 2, 3]);
        assert!(out.evicted.is_empty());
        let m = idx.longest_match(&[1, 2, 3]).unwrap();
        assert_eq!(m.id, out.id);
        assert_eq!(m.matched_len, 3); // full match; backend clamps to recompute the last token
    }

    #[test]
    fn partial_prefix_match_reports_shared_length() {
        let mut idx = PrefixIndex::new(4);
        let sys = idx.insert(vec![1, 2, 3, 4, 5]).id;
        // Shares the first 3 tokens, then diverges.
        let m = idx.longest_match(&[1, 2, 3, 9, 9, 9]).unwrap();
        assert_eq!(m.id, sys);
        assert_eq!(m.matched_len, 3);
    }

    #[test]
    fn no_shared_first_token_is_a_miss() {
        let mut idx = PrefixIndex::new(4);
        idx.insert(vec![1, 2, 3]);
        assert_eq!(idx.longest_match(&[7, 2, 3]), None);
    }

    #[test]
    fn longest_among_several_wins() {
        let mut idx = PrefixIndex::new(8);
        let _a = idx.insert(vec![1, 2]).id;
        let b = idx.insert(vec![1, 2, 3, 4]).id;
        let _c = idx.insert(vec![1, 9]).id;
        // [1,2,3,4,5] shares 4 with b, 2 with a, 1 with c -> b wins.
        let m = idx.longest_match(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(m.id, b);
        assert_eq!(m.matched_len, 4);
    }

    #[test]
    fn matched_len_capped_at_shorter_length() {
        let mut idx = PrefixIndex::new(4);
        let e = idx.insert(vec![1, 2, 3, 4, 5, 6]).id;
        // Query is shorter than the stored entry: matched_len is the query length.
        let m = idx.longest_match(&[1, 2, 3]).unwrap();
        assert_eq!(m.id, e);
        assert_eq!(m.matched_len, 3);
    }

    #[test]
    fn exact_reinsert_refreshes_without_duplicating() {
        let mut idx = PrefixIndex::new(4);
        let first = idx.insert(vec![1, 2, 3]);
        let again = idx.insert(vec![1, 2, 3]);
        assert_eq!(first.id, again.id, "same sequence keeps its id");
        assert!(again.evicted.is_empty());
        assert_eq!(idx.len(), 1, "no duplicate entry");
    }

    #[test]
    fn lru_eviction_returns_dropped_ids() {
        let mut idx = PrefixIndex::new(2);
        let a = idx.insert(vec![1]).id;
        let b = idx.insert(vec![2]).id;
        // Inserting a third evicts the least-recently-used (a).
        let out = idx.insert(vec![3]);
        assert_eq!(out.evicted, vec![a]);
        assert!(!idx.contains(a));
        assert!(idx.contains(b));
        assert!(idx.contains(out.id));
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn a_match_refreshes_recency_and_survives_eviction() {
        let mut idx = PrefixIndex::new(2);
        let a = idx.insert(vec![1, 1]).id;
        let b = idx.insert(vec![2, 2]).id;
        // Touch `a` so it is most-recently-used; the next insert must then evict `b`, not `a`.
        assert_eq!(idx.longest_match(&[1, 1]).unwrap().id, a);
        let out = idx.insert(vec![3, 3]);
        assert_eq!(
            out.evicted,
            vec![b],
            "the now-LRU entry b is evicted, not the refreshed a"
        );
        assert!(idx.contains(a));
        assert!(!idx.contains(b));
    }

    #[test]
    fn a_match_keeps_its_id_when_reinserted() {
        let mut idx = PrefixIndex::new(2);
        let stored = idx.insert(vec![1, 2, 3]).id;

        let matched = idx.longest_match(&[1, 2, 3, 4]).unwrap();
        let reinserted = idx.insert(vec![1, 2, 3]);

        assert_eq!(matched.id, stored);
        assert_eq!(reinserted.id, matched.id);
    }

    #[test]
    fn zero_capacity_stores_nothing() {
        let mut idx = PrefixIndex::new(0);
        let out = idx.insert(vec![1, 2, 3]);
        assert_eq!(out.evicted, vec![out.id]); // inserted then immediately evicted
        assert!(idx.is_empty());
        assert_eq!(idx.longest_match(&[1, 2, 3]), None);
    }

    #[test]
    fn ids_are_never_reused_after_eviction() {
        let mut idx = PrefixIndex::new(1);
        let a = idx.insert(vec![1]).id;
        let out = idx.insert(vec![2]);
        assert_eq!(out.evicted, vec![a]);
        assert_ne!(out.id, a, "a fresh entry never aliases an evicted id");
    }
}
