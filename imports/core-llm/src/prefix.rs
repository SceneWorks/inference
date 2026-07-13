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
        let entry = self.touch(idx);
        Some(PrefixMatch {
            id: entry.id,
            matched_len,
        })
    }

    /// Store `tokens` for future reuse, returning its handle and any evicted entries.
    ///
    /// An exact re-insert of an already-stored sequence refreshes that entry (same [`PrefixId`], no
    /// eviction) rather than duplicating it — so a backend can re-store a deterministic
    /// prompt+generation without leaking a slot.
    pub fn insert(&mut self, tokens: Vec<i32>) -> InsertOutcome {
        if let Some(idx) = self.entries.iter().position(|e| e.tokens == tokens) {
            let entry = self.touch(idx);
            return InsertOutcome {
                id: entry.id,
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

    /// Move the entry at deque index `idx` to the back (most-recently-used) and return a copy of its
    /// header (id is `Copy`). `idx` must be in range.
    fn touch(&mut self, idx: usize) -> Entry {
        let entry = self.entries.remove(idx).expect("index in range");
        self.entries.push_back(entry.clone());
        entry
    }
}

/// Length of the shared leading run of two token slices.
fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(out.evicted, vec![b], "the now-LRU entry b is evicted, not the refreshed a");
        assert!(idx.contains(a));
        assert!(!idx.contains(b));
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
