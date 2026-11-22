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
use std::time::Duration;

use crate::defaults::DecodeBackend;

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

/// The prefix-cache budget a load on `backend` asks for before admission clamps it: the requested
/// budget, or the backend's default ([`DecodeDefaults::prefix_cache_bytes`]) when unset. A load
/// estimate reports it beside the bytes the load requires; [`prefix_cache_budget`] settles it
/// against the load's headroom.
///
/// The default lives in the per-backend defaults table (epic sc-24432 E5, [`crate::defaults`]):
/// on, PROVISIONAL until the terminal campaign (sc-24446) measures TTFT and parity with the cache
/// on against `prefix_cache_bytes: Some(0)`. The budget is admission-honest either way: a load
/// reports what it settled ([`LoadReport::prefix_cache_bytes`]) and a backend's load estimate
/// names what it asks for, beside — never inside — what the load requires.
///
/// [`DecodeDefaults::prefix_cache_bytes`]: crate::DecodeDefaults::prefix_cache_bytes
/// [`LoadReport::prefix_cache_bytes`]: crate::LoadReport::prefix_cache_bytes
pub fn requested_prefix_cache_bytes(backend: DecodeBackend, requested: Option<u64>) -> u64 {
    requested.unwrap_or(backend.defaults().prefix_cache_bytes)
}

/// The prefix-cache byte budget a load on `backend` settles (E7): the requested budget (`None`:
/// the backend's default, [`requested_prefix_cache_bytes`]) clamped to the headroom the load's
/// admission leaves — `available - load_required`, the same two figures the load was admitted on.
/// So the cache can never push the load past its admission budget, and a load that fits without
/// the cache never fails because of it; a tight host simply gets a smaller (or zero) cache.
pub fn prefix_cache_budget(
    backend: DecodeBackend,
    requested: Option<u64>,
    load_required: u64,
    available: u64,
) -> u64 {
    requested_prefix_cache_bytes(backend, requested).min(available.saturating_sub(load_required))
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

/// Why a request on a compressed KV cache (epic sc-20669) keeps no snapshot: the store holds dense
/// K/V and a compressed cache never re-enters it; the request still reads it — a dense hit is
/// imported into its compressed cache.
pub const PREFIX_COMPRESSED_NOT_STORED: &str =
    "not kept: a compressed KV cache is not stored (a dense hit is imported into it)";

/// Why a compressed request's prefix-cache hit was not reused: its compressed KV cache declined to
/// import the stored dense prefix (a refused selection runs dense, or the geometry does not
/// match), so the whole prompt was prefilled on it.
pub const PREFIX_COMPRESSED_IMPORT_DECLINED: &str =
    "not reused: the compressed KV cache declined the stored dense prefix";

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

/// Maximum boundary checkpoints planned/captured for a single request (feature SW-F356).
pub const MAX_CHECKPOINTS_PER_REQUEST: usize = 4;

/// Fallback reason when an explicit [`PrefixDocumentSpan`](crate::PrefixDocumentSpan) could not
/// be aligned to a reusable token boundary `< prompt.len()`.
pub const PREFIX_DOCUMENT_UNALIGNED: &str =
    "prefix_checkpoints: designated document span did not align to a reusable token boundary";

/// Fallback reason when a protected checkpoint exceeded the protected byte cap and was stored
/// as unprotected instead.
pub const PREFIX_PROTECTED_CAP_DEMOTED: &str =
    "prefix_checkpoints: checkpoint exceeded protected byte cap and was stored unprotected";

/// Fallback reason when a longer stored prefix existed in the cache but lacked compatible
/// proposer state (for example MTP state when MTP was requested).
pub const PREFIX_PROPOSER_INCOMPATIBLE: &str =
    "prefix_checkpoints: longer cached prefix lacked compatible proposer state and was skipped";

/// Fallback reason when a cancelled request skipped storing partial/unconfirmed prefix state.
pub const PREFIX_CANCELLED_NOT_STORED: &str =
    "prefix_checkpoints: request cancelled before completion; state not stored";

/// Fallback reason when an oversized unprotected checkpoint was skipped so protected document
/// or shared-root checkpoints stay resident.
pub const PREFIX_PROTECTED_PRESERVED: &str =
    "prefix_checkpoints: oversized unprotected checkpoint skipped to preserve protected entries";

/// Fallback reason when tight memory or cache budget trimmed lower-priority planned boundaries.
pub const PREFIX_PARTIAL_ADMISSION_TRIMMED: &str =
    "prefix_checkpoints: lower-priority planned checkpoints trimmed to fit memory admission";

/// Semantic tier of a stored prefix checkpoint (feature SW-F356).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum PrefixKind {
    /// End-of-conversation / prompt boundary (default pre-SW-F356 boundary).
    #[default]
    Conversation,
    /// Boundary at an earlier conversation message (enables edited-turn prefix reuse).
    MessageBoundary,
    /// Periodic intra-prompt checkpoint near the tail of a long uncheckpointed prompt span.
    NearTail,
    /// Leading shared root (system prompt / initial turn shared across conversation branches).
    SharedRoot,
    /// Explicitly designated reusable document/context span protected against intervening turns.
    Document,
}

impl PrefixKind {
    /// Stable lower-case label for telemetry.
    pub const fn label(self) -> &'static str {
        match self {
            PrefixKind::Conversation => "conversation",
            PrefixKind::MessageBoundary => "message_boundary",
            PrefixKind::NearTail => "near_tail",
            PrefixKind::SharedRoot => "shared_root",
            PrefixKind::Document => "document",
        }
    }

    /// Retention priority (`Document` > `SharedRoot` > `MessageBoundary` > `Conversation` > `NearTail`).
    pub const fn priority(self) -> u8 {
        match self {
            PrefixKind::NearTail => 1,
            PrefixKind::Conversation => 2,
            PrefixKind::MessageBoundary => 3,
            PrefixKind::SharedRoot => 4,
            PrefixKind::Document => 5,
        }
    }

    const fn retention_weight(self) -> u64 {
        match self {
            PrefixKind::NearTail => 1,
            PrefixKind::Conversation => 2,
            PrefixKind::MessageBoundary => 3,
            PrefixKind::SharedRoot => 4,
            PrefixKind::Document => 6,
        }
    }
}

impl Ord for PrefixKind {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&other.priority())
    }
}

impl PartialOrd for PrefixKind {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// One planned prefix checkpoint boundary produced by [`plan_prefix_boundaries`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedBoundary {
    /// Absolute prompt token count covered by this boundary (`0 < position < prompt_len`).
    pub position: usize,
    /// Semantic kind of the boundary.
    pub kind: PrefixKind,
    /// Whether this boundary should be protected from unprotected eviction.
    pub protected: bool,
}

/// Tuning parameters for intra-prompt near-tail checkpoint placement on [`PrefixReuse::WholeEntry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NearTailConfig {
    /// Minimum uncheckpointed suffix tokens needed before inserting a near-tail checkpoint.
    pub min_span: usize,
    /// Margin kept between the near-tail checkpoint and `prompt_len`.
    pub margin: usize,
    /// Token alignment quantum for stable near-tail placement.
    pub align: usize,
    /// Trailing window `[prompt_len - window, prompt_len)` considered already well-covered.
    pub window: usize,
}

impl Default for NearTailConfig {
    fn default() -> Self {
        Self {
            min_span: 256,
            margin: 64,
            align: 64,
            window: 128,
        }
    }
}

/// Summary of storing a request's checkpoints into a backend prefix cache (SW-F356).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrefixStoreOutcome {
    /// Number of checkpoint entries stored.
    pub checkpoints_captured: u32,
    /// Sum of stored entry byte sizes.
    pub captured_bytes: u64,
    /// Number of entries evicted during storage.
    pub evicted_entries: u32,
    /// Wall duration spent copying/slicing and storing checkpoint entries.
    pub capture_time: Duration,
    /// Fallback diagnostics emitted during storage.
    pub fallbacks: Vec<String>,
}

/// Request-scoped accumulator for [`PrefixCheckpointReport`](crate::report::PrefixCheckpointReport)
/// shared by MLX and Candle providers (SW-F356).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointTelemetry {
    /// Wall duration spent looking up and restoring cached prefix state.
    pub restore_time: Duration,
    /// Cumulative wall duration spent capturing and storing checkpoints.
    pub capture_time: Duration,
    /// Semantic kind label of the matched checkpoint, when a prefix hit occurred.
    pub matched_kind: Option<String>,
    /// Number of checkpoints captured/stored for this request.
    pub checkpoints_captured: u32,
    /// Total bytes captured/stored across checkpoints for this request.
    pub captured_bytes: u64,
    /// Number of resident entries evicted while storing checkpoints for this request.
    pub evicted_entries: u32,
    /// Deduplicated fallback/advisory diagnostics emitted during restore/capture/storage.
    pub fallbacks: Vec<String>,
}

impl CheckpointTelemetry {
    /// Append a fallback diagnostic if not already recorded.
    pub fn push_fallback(&mut self, fb: &str) {
        if !self.fallbacks.iter().any(|existing| existing == fb) {
            self.fallbacks.push(fb.to_string());
        }
    }

    /// Fold a backend checkpoint-store outcome into the request accumulator.
    pub fn record_store(&mut self, outcome: PrefixStoreOutcome) {
        self.checkpoints_captured = self
            .checkpoints_captured
            .saturating_add(outcome.checkpoints_captured);
        self.captured_bytes = self.captured_bytes.saturating_add(outcome.captured_bytes);
        self.evicted_entries = self.evicted_entries.saturating_add(outcome.evicted_entries);
        self.capture_time = self.capture_time.saturating_add(outcome.capture_time);
        for fb in outcome.fallbacks {
            self.push_fallback(&fb);
        }
    }

    /// Convert the accumulated telemetry into the public [`PrefixCheckpointReport`](crate::report::PrefixCheckpointReport).
    pub fn into_report(
        self,
        policy: &str,
        reused_tokens: u64,
        reread_tokens: u64,
        resident_bytes: u64,
        protected_bytes: u64,
    ) -> crate::report::PrefixCheckpointReport {
        crate::report::PrefixCheckpointReport {
            policy: policy.to_string(),
            reused_tokens,
            reread_tokens,
            matched_kind: self.matched_kind,
            checkpoints_captured: self.checkpoints_captured,
            captured_bytes: self.captured_bytes,
            restore_time: self.restore_time,
            capture_time: self.capture_time,
            resident_bytes,
            protected_bytes,
            evicted_entries: self.evicted_entries,
            fallbacks: self.fallbacks,
        }
    }
}

/// Candidate token boundaries extracted by a provider for the current request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoundaryCandidates {
    /// End of explicitly designated document text (`PrefixCheckpointPolicy::document`).
    pub document: Option<usize>,
    /// End of leading shared root (typically the first message when `messages.len() >= 2`).
    pub shared_root: Option<usize>,
    /// End of `messages[..messages.len() - 1]` when `messages.len() >= 2`.
    pub prev_message: Option<usize>,
    /// End of conversation / prompt header before the final generation lead-in.
    pub conversation_end: Option<usize>,
}

/// Render `prefix_messages` without a generation prompt and return its token length iff it is
/// a strict token prefix of `prompt_ids`.
pub fn render_prefix_boundary(
    template: &dyn crate::template::ChatTemplate,
    tokenizer: &crate::tokenizer::Tokenizer,
    prefix_messages: &[crate::message::Message],
    req: &crate::request::TextLlmRequest,
    prompt_ids: &[i32],
    date_string: Option<&str>,
) -> Option<usize> {
    let rendered = template
        .render_with(
            prefix_messages,
            &crate::template::RenderOptions {
                add_generation_prompt: false,
                enable_thinking: req.enable_thinking_kwarg(),
                reasoning_effort: req.reasoning_effort,
                preserve_thinking: req.preserve_thinking,
                tools: &req.tools,
                date_string,
            },
        )
        .ok()?;
    let ids = tokenizer.encode(&rendered, false).ok()?;
    let len = ids.len();
    (len > 0 && len < prompt_ids.len() && ids.iter().zip(prompt_ids).all(|(&a, &b)| a as i32 == b))
        .then_some(len)
}

/// Resolve an explicit [`PrefixDocumentSpan`](crate::request::PrefixDocumentSpan) into a strict
/// token prefix length `< prompt_ids.len()`.
///
/// Uses dual PUA sentinels (`\u{E000}` / `\u{E001}`) for
/// [`PrefixDocumentSpan::MessageChars`](crate::request::PrefixDocumentSpan::MessageChars) so BPE
/// merges across the character boundary back off cleanly to the last identical token.
pub fn resolve_document_boundary(
    template: &dyn crate::template::ChatTemplate,
    tokenizer: &crate::tokenizer::Tokenizer,
    span: crate::request::PrefixDocumentSpan,
    messages: &[crate::message::Message],
    req: &crate::request::TextLlmRequest,
    prompt_ids: &[i32],
    date_string: Option<&str>,
) -> Option<usize> {
    match span {
        crate::request::PrefixDocumentSpan::Messages { count } => (count > 0
            && count <= messages.len())
        .then(|| {
            render_prefix_boundary(
                template,
                tokenizer,
                &messages[..count],
                req,
                prompt_ids,
                date_string,
            )
        })
        .flatten(),
        crate::request::PrefixDocumentSpan::Tokens { count } => {
            (count > 0 && count < prompt_ids.len()).then_some(count)
        }
        crate::request::PrefixDocumentSpan::MessageChars { message, chars } => {
            if chars == 0 {
                return None;
            }
            let target = messages.get(message)?;
            let text = target.text_content();
            let total_chars = text.chars().count();
            if chars > total_chars {
                return None;
            }
            let byte_split = text
                .char_indices()
                .nth(chars)
                .map(|(idx, _)| idx)
                .unwrap_or(text.len());
            let (prefix_str, suffix_str) = (&text[..byte_split], &text[byte_split..]);
            let tokenize_with_sentinel = |sentinel: char| -> Option<Vec<u32>> {
                let mut modified = messages.to_vec();
                modified[message].content = vec![crate::message::Content::Text(format!(
                    "{prefix_str}{sentinel}{suffix_str}"
                ))];
                let rendered = template
                    .render_with(
                        &modified,
                        &crate::template::RenderOptions {
                            add_generation_prompt: true,
                            enable_thinking: req.enable_thinking_kwarg(),
                            reasoning_effort: req.reasoning_effort,
                            preserve_thinking: req.preserve_thinking,
                            tools: &req.tools,
                            date_string,
                        },
                    )
                    .ok()?;
                tokenizer.encode(&rendered, false).ok()
            };
            let ids_a = tokenize_with_sentinel('\u{E000}')?;
            let ids_b = tokenize_with_sentinel('\u{E001}')?;
            let same_as_prompt = |ids: &[u32]| {
                ids.len() == prompt_ids.len()
                    && ids.iter().zip(prompt_ids).all(|(&a, &b)| a as i32 == b)
            };
            if same_as_prompt(&ids_a) && same_as_prompt(&ids_b) {
                return None;
            }
            let aligned = ids_a
                .iter()
                .zip(&ids_b)
                .zip(prompt_ids)
                .take_while(|&((&a, &b), &p)| a as i32 == p && b as i32 == p)
                .count();
            (aligned > 0 && aligned < prompt_ids.len()).then_some(aligned)
        }
    }
}

/// Resolve all candidate prefix boundaries (`document`, `shared_root`, `prev_message`,
/// `conversation_end`), plus whether an explicit document span failed to align.
pub fn resolve_boundary_candidates(
    template: &dyn crate::template::ChatTemplate,
    tokenizer: &crate::tokenizer::Tokenizer,
    messages: &[crate::message::Message],
    req: &crate::request::TextLlmRequest,
    prompt_ids: &[i32],
    policy: crate::request::PrefixCheckpointPolicy,
    date_string: Option<&str>,
) -> (BoundaryCandidates, bool) {
    let document = policy.document.and_then(|span| {
        resolve_document_boundary(
            template,
            tokenizer,
            span,
            messages,
            req,
            prompt_ids,
            date_string,
        )
    });
    let doc_unaligned = policy.document.is_some() && document.is_none();
    let conversation_end =
        render_prefix_boundary(template, tokenizer, messages, req, prompt_ids, date_string);
    let shared_root = if policy.is_active() && messages.len() >= 2 {
        render_prefix_boundary(
            template,
            tokenizer,
            &messages[..1],
            req,
            prompt_ids,
            date_string,
        )
    } else {
        None
    };
    let prev_message = if policy.is_active() && messages.len() >= 2 {
        render_prefix_boundary(
            template,
            tokenizer,
            &messages[..messages.len() - 1],
            req,
            prompt_ids,
            date_string,
        )
    } else {
        None
    };
    (
        BoundaryCandidates {
            document,
            shared_root,
            prev_message,
            conversation_end,
        },
        doc_unaligned,
    )
}

/// Boundary capture and promotion plan returned by [`plan_prefix_boundaries`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryPlan {
    /// Ordered (`position` ascending), deduplicated boundaries to capture during suffix prefill
    /// (for `WholeEntry`) or slice as compact protected entries (for `AnyPrefix`).
    pub captures: Vec<PlannedBoundary>,
    /// Protected `Document` or `SharedRoot` boundary already covered by the restored `reused`
    /// prefix that can be promoted/retained without re-capturing during prefill.
    pub promote_reused: Option<PlannedBoundary>,
}

/// Plan which exact prefix checkpoints to capture and/or promote for a request.
pub fn plan_prefix_boundaries(
    prompt_len: usize,
    reused: usize,
    reuse: PrefixReuse,
    policy: crate::request::PrefixCheckpointPolicy,
    candidates: BoundaryCandidates,
    near_tail: NearTailConfig,
) -> BoundaryPlan {
    plan_prefix_boundaries_with_exact(
        prompt_len,
        reused,
        reuse,
        policy,
        candidates,
        near_tail,
        |_| true,
    )
}

/// Plan which exact prefix checkpoints to capture and/or promote for a request, consulting
/// `has_exact_key(pos)` so that on [`PrefixReuse::AnyPrefix`] only boundaries already backed by
/// an exact stored key (`pos <= reused && has_exact_key(pos)`) go into `promote_reused` — any
/// non-exact boundary stays in `captures` to be priced, admitted, and bounded by E7.
pub fn plan_prefix_boundaries_with_exact(
    prompt_len: usize,
    reused: usize,
    reuse: PrefixReuse,
    policy: crate::request::PrefixCheckpointPolicy,
    candidates: BoundaryCandidates,
    near_tail: NearTailConfig,
    mut has_exact_key: impl FnMut(usize) -> bool,
) -> BoundaryPlan {
    if prompt_len <= 1 {
        return BoundaryPlan::default();
    }

    if !policy.is_active() {
        let mut captures = Vec::new();
        if reuse == PrefixReuse::WholeEntry {
            if let Some(pos) = candidates.conversation_end {
                if pos > reused && pos < prompt_len {
                    captures.push(PlannedBoundary {
                        position: pos,
                        kind: PrefixKind::Conversation,
                        protected: false,
                    });
                }
            }
        }
        return BoundaryPlan {
            captures,
            promote_reused: None,
        };
    }

    let valid_reused_promote = |pos: usize, has_exact: bool| {
        pos > 0
            && pos < prompt_len
            && has_exact
            && (pos == reused || (reuse == PrefixReuse::AnyPrefix && pos <= reused))
    };
    let doc_exact = candidates.document.is_some_and(&mut has_exact_key);
    let root_exact = candidates.shared_root.is_some_and(&mut has_exact_key);
    let promote_reused = candidates
        .document
        .filter(|&p| valid_reused_promote(p, doc_exact))
        .map(|position| PlannedBoundary {
            position,
            kind: PrefixKind::Document,
            protected: true,
        })
        .or_else(|| {
            candidates
                .shared_root
                .filter(|&p| {
                    valid_reused_promote(p, root_exact) && Some(p) != candidates.conversation_end
                })
                .map(|position| PlannedBoundary {
                    position,
                    kind: PrefixKind::SharedRoot,
                    protected: true,
                })
        });

    let mut captures: Vec<PlannedBoundary> = Vec::new();
    let push_boundary = |captures: &mut Vec<PlannedBoundary>,
                         position: usize,
                         kind: PrefixKind,
                         protected: bool| {
        let already_exact = match reuse {
            PrefixReuse::WholeEntry => position <= reused,
            PrefixReuse::AnyPrefix => {
                position <= reused
                    && match kind {
                        PrefixKind::Document => doc_exact,
                        PrefixKind::SharedRoot => root_exact,
                        _ => true,
                    }
            }
        };
        if position == 0
            || position >= prompt_len
            || promote_reused.is_some_and(|pb| pb.position == position)
            || already_exact
        {
            return;
        }
        if let Some(existing) = captures.iter_mut().find(|b| b.position == position) {
            existing.protected |= protected;
            if kind.priority() > existing.kind.priority() {
                existing.kind = kind;
            }
        } else {
            captures.push(PlannedBoundary {
                position,
                kind,
                protected,
            });
        }
    };

    if let Some(doc) = candidates.document {
        push_boundary(&mut captures, doc, PrefixKind::Document, true);
    }
    if let Some(root) = candidates.shared_root {
        if Some(root) != candidates.conversation_end {
            push_boundary(&mut captures, root, PrefixKind::SharedRoot, true);
        }
    }

    if reuse == PrefixReuse::WholeEntry {
        if let Some(prev) = candidates.prev_message {
            push_boundary(&mut captures, prev, PrefixKind::MessageBoundary, false);
        }
        if let Some(conv) = candidates.conversation_end {
            push_boundary(&mut captures, conv, PrefixKind::Conversation, false);
        }
        let window_start = prompt_len.saturating_sub(near_tail.window);
        let has_in_window = captures.iter().any(|b| b.position >= window_start);
        let last_anchor = captures
            .iter()
            .map(|b| b.position)
            .max()
            .unwrap_or(0)
            .max(reused);
        if !has_in_window
            && near_tail.align > 0
            && prompt_len.saturating_sub(last_anchor) >= near_tail.min_span
            && prompt_len > near_tail.margin
        {
            let pos = ((prompt_len - near_tail.margin) / near_tail.align) * near_tail.align;
            if pos > last_anchor && pos < prompt_len {
                push_boundary(&mut captures, pos, PrefixKind::NearTail, false);
            }
        }
    }

    if captures.len() > MAX_CHECKPOINTS_PER_REQUEST {
        captures.sort_by(|a, b| {
            (b.protected, b.kind.priority(), b.position).cmp(&(
                a.protected,
                a.kind.priority(),
                a.position,
            ))
        });
        captures.truncate(MAX_CHECKPOINTS_PER_REQUEST);
    }
    captures.sort_by_key(|b| b.position);

    BoundaryPlan {
        captures,
        promote_reused,
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
    kind: PrefixKind,
    protected: bool,
    hit_count: u64,
    reused_tokens_total: u64,
    last_used_tick: u64,
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
    /// Semantic kind of the matched checkpoint.
    pub kind: PrefixKind,
    /// Whether this entry is currently protected against unprotected eviction.
    pub protected: bool,
}

/// Lookup result with diagnostic signaling when a longer cached prefix lacked compatible
/// proposer state (feature SW-F356).
#[derive(Debug)]
pub struct PrefixLookup<'a, S> {
    /// Best reusable hit, or `None` on a miss.
    pub hit: Option<PrefixHit<'a, S>>,
    /// `true` when an entry in the store could have matched more tokens than `hit` (or matched on
    /// a miss) except that `held` returned `None` (for example incompatible proposer state).
    pub proposer_skipped_longer: bool,
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

/// Detailed outcome from [`PrefixStore::insert_with_meta`] / [`PrefixStore::insert_checkpoint`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixInsertOutcome {
    /// Whether the entry is now held.
    pub stored: bool,
    /// Entries dropped to make room (or replaced / superseded by this one).
    pub evicted: usize,
    /// `true` when a `protected` insert exceeded the protected byte cap and was stored unprotected.
    pub demoted_from_protected: bool,
    /// `true` when an unprotected insert was skipped so protected entries would not be evicted.
    pub skipped_for_protected: bool,
}

/// Optional context passed to [`PrefixStore::admit_planned_with`] so admission can preserve an
/// already-matched prefix entry, account for in-place protected promotion, and recognize when an
/// [`PrefixReuse::AnyPrefix`] `Conversation` entry supersedes unprotected matching prefixes.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixPlannedContext<'a> {
    /// Prompt token ids of the request being admitted.
    pub prompt: Option<&'a [i32]>,
    /// Index in the store returned by [`PrefixStore::peek_match`], kept across optional snapshot
    /// admission so an entry about to be restored is evicted only as a last resort for bare
    /// `required` memory.
    pub keep_match: Option<usize>,
    /// Boundary slated for in-place protected promotion on hit, pinned against optional eviction.
    pub promote_reused: Option<PlannedBoundary>,
    /// Reuse mode of the request's final `Conversation` snapshot, if one is stored (`AnyPrefix`
    /// supersedes matching unprotected `AnyPrefix` entries at store time).
    pub conversation_reuse: Option<PrefixReuse>,
}

/// Result of [`PrefixStore::admit_planned`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedAdmission {
    /// Memory admission outcome (whether snapshots are admitted and updated availability).
    pub admission: PrefixAdmission,
    /// Surviving boundaries that fit under budget and memory admission.
    pub kept: Vec<PlannedBoundary>,
    /// Number of store entries evicted during admission.
    pub evicted_entries: u32,
    /// Optional diagnostic (`PREFIX_PARTIAL_ADMISSION_TRIMMED`, `PREFIX_PROTECTED_PRESERVED`,
    /// or `PREFIX_NOT_ADMITTED`).
    pub fallback: Option<&'static str>,
}

/// The cross-turn prefix cache's policy (story sc-24437): per-entry backend state `S` (KV tensors,
/// or a hybrid decoder's KV plus recurrent state) under a **byte** budget with value-aware /
/// protected eviction, and longest-usable-prefix lookup that respects each entry's [`PrefixReuse`].
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
    protected_bytes: u64,
    tick: u64,
    /// Held entries; hits and inserts move/push to back so back-to-front scan naturally breaks
    /// ties toward most-recently-used.
    entries: VecDeque<StoredPrefix<S>>,
    stats: PrefixStats,
}

impl<S> PrefixStore<S> {
    /// An empty store holding at most `budget_bytes` of entry state (`0` stores nothing).
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            resident_bytes: 0,
            protected_bytes: 0,
            tick: 0,
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

    /// Bytes currently held in protected entries (`Document` / `SharedRoot`).
    pub fn protected_bytes(&self) -> u64 {
        self.protected_bytes
    }

    /// Protected byte ceiling: `budget_bytes / 2` normally, or up to `budget_bytes * 3 / 4` when
    /// accommodating an explicitly designated [`PrefixKind::Document`] (never 100% of a positive
    /// budget so recent turns always keep room).
    pub fn protected_budget_cap(&self) -> u64 {
        if self
            .entries
            .iter()
            .any(|e| e.protected && e.kind == PrefixKind::Document)
        {
            self.document_protected_cap()
        } else {
            self.budget_bytes / 2
        }
    }

    fn document_protected_cap(&self) -> u64 {
        self.budget_bytes.saturating_mul(3) / 4
    }

    fn protected_cap_for(&self, kind: PrefixKind) -> u64 {
        match kind {
            PrefixKind::Document => self.document_protected_cap(),
            _ => self.budget_bytes / 2,
        }
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

    /// The held entries' token sequences, oldest-position first (diagnostics and tests).
    pub fn keys(&self) -> Vec<&[i32]> {
        self.entries.iter().map(|e| e.tokens.as_slice()).collect()
    }

    /// Whether an entry with exact token key `tokens` is currently held in the store.
    pub fn contains_exact(&self, tokens: &[i32]) -> bool {
        !tokens.is_empty() && self.entries.iter().any(|e| e.tokens == tokens)
    }

    /// Compute how many leading prompt tokens an entry could reuse without mutating state.
    fn entry_match(
        entry: &StoredPrefix<S>,
        prompt: &[i32],
        limit: usize,
        held: &impl Fn(&S) -> Option<usize>,
    ) -> (usize, bool) {
        let shared = common_prefix_len(&entry.tokens, prompt);
        let Some(held_len) = held(&entry.state) else {
            let potential = match entry.reuse {
                PrefixReuse::AnyPrefix => shared.min(limit),
                PrefixReuse::WholeEntry if shared == entry.tokens.len() && shared <= limit => {
                    shared
                }
                PrefixReuse::WholeEntry => 0,
            };
            return (potential, false);
        };
        let reused = match entry.reuse {
            PrefixReuse::AnyPrefix => shared.min(limit).min(held_len),
            PrefixReuse::WholeEntry
                if shared == entry.tokens.len() && shared <= limit && held_len == shared =>
            {
                shared
            }
            PrefixReuse::WholeEntry => 0,
        };
        (reused, true)
    }

    /// Shared best-match selector: prefers longest `reused` length, then `protected`, then
    /// highest semantic `kind.priority()`, then most-recently-used (back of deque).
    fn best_match_with_diag(
        &self,
        prompt: &[i32],
        held: &impl Fn(&S) -> Option<usize>,
    ) -> (Option<(usize, usize)>, usize) {
        let limit = prompt.len().saturating_sub(1);
        let mut best: Option<(usize, (usize, bool, u8))> = None;
        let mut skipped_longest = 0usize;
        for (i, e) in self.entries.iter().enumerate().rev() {
            let (len, compatible) = Self::entry_match(e, prompt, limit, held);
            if !compatible {
                skipped_longest = skipped_longest.max(len);
                continue;
            }
            if len > 0 {
                let key = (len, e.protected, e.kind.priority());
                if best.is_none_or(|(_, best_key)| key > best_key) {
                    best = Some((i, key));
                }
            }
        }
        (best.map(|(i, (len, _, _))| (i, len)), skipped_longest)
    }

    /// Read-only inspection returning the index and reused token count of the entry
    /// [`lookup_with_proposer_diagnostic`](Self::lookup_with_proposer_diagnostic) would select,
    /// without mutating stats or recency.
    pub fn peek_match(
        &self,
        prompt: &[i32],
        held: impl Fn(&S) -> Option<usize>,
    ) -> Option<(usize, usize)> {
        self.best_match_with_diag(prompt, &held).0
    }

    /// Read-only inspection returning how many leading tokens of `prompt` a cached entry would
    /// currently reuse without updating stats or recency (useful for upfront boundary planning
    /// and snapshot admission before lookup).
    pub fn peek_reused(&self, prompt: &[i32], held: impl Fn(&S) -> Option<usize>) -> usize {
        self.peek_match(prompt, held).map_or(0, |(_, len)| len)
    }

    /// Same as [`lookup`](Self::lookup), also reporting whether a longer stored entry matched more
    /// tokens than the chosen hit (or matched on a miss) except that `held` rejected it
    /// (for example incompatible proposer state).
    pub fn lookup_with_proposer_diagnostic(
        &mut self,
        prompt: &[i32],
        held: impl Fn(&S) -> Option<usize>,
    ) -> PrefixLookup<'_, S> {
        self.stats.lookups += 1;
        let (best, skipped_longest) = self.best_match_with_diag(prompt, &held);
        let best_reused = best.map_or(0, |(_, r)| r);
        let proposer_skipped_longer = skipped_longest > best_reused;
        let Some((idx, reused)) = best else {
            self.stats.computed_prefill_tokens += prompt.len();
            return PrefixLookup {
                hit: None,
                proposer_skipped_longer,
            };
        };
        self.tick = self.tick.saturating_add(1);
        let mut entry = self.entries.remove(idx).expect("index in range");
        entry.hit_count = entry.hit_count.saturating_add(1);
        entry.reused_tokens_total = entry.reused_tokens_total.saturating_add(reused as u64);
        entry.last_used_tick = self.tick;
        self.entries.push_back(entry);
        self.stats.hits += 1;
        self.stats.reused_prefix_tokens += reused;
        self.stats.computed_prefill_tokens += prompt.len() - reused;
        let entry = self.entries.back().expect("just pushed");
        PrefixLookup {
            hit: Some(PrefixHit {
                state: &entry.state,
                reused,
                reuse: entry.reuse,
                kind: entry.kind,
                protected: entry.protected,
            }),
            proposer_skipped_longer,
        }
    }

    /// The entry reusing the most leading tokens of `prompt`, or `None` (a miss). `held` says,
    /// per entry, how many positions its state really holds — `None` when this request cannot use
    /// it at all — and the reuse never exceeds it (defence in depth against a key that over-states
    /// its state, sc-12455). An [`AnyPrefix`](PrefixReuse::AnyPrefix) entry reuses its shared
    /// leading run, a [`WholeEntry`](PrefixReuse::WholeEntry) one only its whole sequence (and
    /// only when its state holds exactly that); either way the reuse stops short of the whole
    /// prompt so a forward always has a query. Ties go to the most-recently-used entry, and a hit
    /// updates recency/value metadata. Updates [`PrefixStats`] with the reuse actually granted.
    pub fn lookup(
        &mut self,
        prompt: &[i32],
        held: impl Fn(&S) -> Option<usize>,
    ) -> Option<PrefixHit<'_, S>> {
        self.lookup_with_proposer_diagnostic(prompt, held).hit
    }

    /// Compare two victim candidates; returns `Less` when `a` should be evicted BEFORE `b`.
    fn compare_unprotected_victim(&self, a_idx: usize, b_idx: usize) -> std::cmp::Ordering {
        let a = &self.entries[a_idx];
        let b = &self.entries[b_idx];
        let a_near_tail = a.kind == PrefixKind::NearTail;
        let b_near_tail = b.kind == PrefixKind::NearTail;
        if a_near_tail != b_near_tail {
            // NearTail entries are always evicted before non-NearTail unprotected entries.
            return b_near_tail.cmp(&a_near_tail);
        }
        self.compare_score(a, b)
            .then_with(|| a.last_used_tick.cmp(&b.last_used_tick))
            .then_with(|| a_idx.cmp(&b_idx))
    }

    /// Compare two protected victim candidates (`SharedRoot` before `Document`, then lowest value
    /// score, then oldest tick).
    fn compare_protected_victim(&self, a_idx: usize, b_idx: usize) -> std::cmp::Ordering {
        let a = &self.entries[a_idx];
        let b = &self.entries[b_idx];
        a.kind
            .priority()
            .cmp(&b.kind.priority())
            .then_with(|| self.compare_score(a, b))
            .then_with(|| a.last_used_tick.cmp(&b.last_used_tick))
            .then_with(|| a_idx.cmp(&b_idx))
    }

    /// Compare `score(a)` vs `score(b)` where smaller score means better victim (`Less`).
    /// `score = kind_weight(kind) * (tokens.len() + reused_tokens_total) * (1 + hit_count) / (bytes.max(1) * (1 + age_ticks))`.
    fn compare_score(&self, a: &StoredPrefix<S>, b: &StoredPrefix<S>) -> std::cmp::Ordering {
        let num = |e: &StoredPrefix<S>| -> Option<u128> {
            let w = e.kind.retention_weight() as u128;
            let val = (e.tokens.len() as u128).checked_add(e.reused_tokens_total as u128)?;
            let hits = 1u128.checked_add(e.hit_count as u128)?;
            w.checked_mul(val)?.checked_mul(hits)
        };
        let den = |e: &StoredPrefix<S>| -> Option<u128> {
            let bytes = e.bytes.max(1) as u128;
            let age = 1u128.checked_add(self.tick.saturating_sub(e.last_used_tick) as u128)?;
            bytes.checked_mul(age)
        };
        if let (Some(na), Some(da), Some(nb), Some(db)) = (num(a), den(a), num(b), den(b)) {
            if let (Some(lhs), Some(rhs)) = (na.checked_mul(db), nb.checked_mul(da)) {
                return lhs.cmp(&rhs);
            }
        }
        let score_f64 = |e: &StoredPrefix<S>| -> f64 {
            let w = e.kind.retention_weight() as f64;
            let val = e.tokens.len() as f64 + e.reused_tokens_total as f64;
            let hits = 1.0 + e.hit_count as f64;
            let bytes = e.bytes.max(1) as f64;
            let age = 1.0 + self.tick.saturating_sub(e.last_used_tick) as f64;
            (w * val * hits) / (bytes * age)
        };
        score_f64(a).total_cmp(&score_f64(b))
    }

    fn pick_unprotected_victim(&self, skip_back: bool) -> Option<usize> {
        let end = if skip_back && !self.entries.is_empty() {
            self.entries.len() - 1
        } else {
            self.entries.len()
        };
        let mut best: Option<usize> = None;
        for i in 0..end {
            if self.entries[i].protected {
                continue;
            }
            if best
                .is_none_or(|b| self.compare_unprotected_victim(i, b) == std::cmp::Ordering::Less)
            {
                best = Some(i);
            }
        }
        best
    }

    fn pick_protected_victim(&self, max_priority: Option<u8>, skip_back: bool) -> Option<usize> {
        let end = if skip_back && !self.entries.is_empty() {
            self.entries.len() - 1
        } else {
            self.entries.len()
        };
        let mut best: Option<usize> = None;
        for i in 0..end {
            let e = &self.entries[i];
            if !e.protected || max_priority.is_some_and(|p| e.kind.priority() > p) {
                continue;
            }
            if best.is_none_or(|b| self.compare_protected_victim(i, b) == std::cmp::Ordering::Less)
            {
                best = Some(i);
            }
        }
        best
    }

    fn remove_entry(&mut self, idx: usize) -> u64 {
        let gone = self.entries.remove(idx).expect("index in range");
        self.resident_bytes -= gone.bytes;
        if gone.protected {
            self.protected_bytes -= gone.bytes;
        }
        self.stats.evicted += 1;
        gone.bytes
    }

    /// Hold `state` for the prefix `tokens` with explicit semantic `kind` and `protected` flag.
    pub fn insert_with_meta(
        &mut self,
        tokens: Vec<i32>,
        reuse: PrefixReuse,
        bytes: u64,
        state: S,
        mut kind: PrefixKind,
        mut protected: bool,
    ) -> PrefixInsertOutcome {
        if tokens.is_empty() || bytes > self.budget_bytes {
            self.stats.rejected += 1;
            return PrefixInsertOutcome::default();
        }
        self.tick = self.tick.saturating_add(1);

        // Preserve existing protection and highest semantic kind if an insert matches exact tokens.
        if let Some(existing) = self.entries.iter().find(|e| e.tokens == tokens) {
            if existing.protected {
                protected = true;
                if existing.kind.priority() > kind.priority() {
                    kind = existing.kind;
                }
            }
        }

        // Compute protected cap feasibility BEFORE disturbing existing entries when unprotected.
        let mut demoted_from_protected = false;
        if protected {
            let cap = self.protected_cap_for(kind);
            let higher_protected: u64 = self
                .entries
                .iter()
                .filter(|e| {
                    e.tokens != tokens && e.protected && e.kind.priority() > kind.priority()
                })
                .map(|e| e.bytes)
                .sum();
            if bytes > cap || higher_protected.saturating_add(bytes) > cap {
                protected = false;
                demoted_from_protected = true;
            }
        }

        if !protected {
            let remaining_protected: u64 = self
                .entries
                .iter()
                .filter(|e| e.tokens != tokens && e.protected)
                .map(|e| e.bytes)
                .sum();
            if remaining_protected.saturating_add(bytes) > self.budget_bytes {
                return PrefixInsertOutcome {
                    stored: false,
                    evicted: 0,
                    demoted_from_protected,
                    skipped_for_protected: true,
                };
            }
        }

        let mut evicted = 0usize;
        let mut i = 0;
        while i < self.entries.len() {
            let e = &self.entries[i];
            let superseded = e.tokens == tokens
                || (!e.protected
                    && reuse == PrefixReuse::AnyPrefix
                    && e.reuse == PrefixReuse::AnyPrefix
                    && tokens.starts_with(&e.tokens));
            if superseded {
                self.remove_entry(i);
                evicted += 1;
            } else {
                i += 1;
            }
        }

        if protected {
            let cap = self.protected_cap_for(kind);
            while self.protected_bytes.saturating_add(bytes) > cap {
                let idx = self
                    .pick_protected_victim(Some(kind.priority()), false)
                    .expect("protected cap check guaranteed an eligible protected victim");
                self.remove_entry(idx);
                evicted += 1;
            }
        }

        while self.resident_bytes.saturating_add(bytes) > self.budget_bytes {
            let idx = match self.pick_unprotected_victim(false) {
                Some(idx) => idx,
                None if protected => {
                    unreachable!("protected cap check leaves room for protected entries")
                }
                None => {
                    unreachable!("remaining_protected + bytes <= budget_bytes checked above")
                }
            };
            self.remove_entry(idx);
            evicted += 1;
        }

        self.resident_bytes += bytes;
        if protected {
            self.protected_bytes += bytes;
        }
        self.entries.push_back(StoredPrefix {
            tokens,
            reuse,
            bytes,
            state,
            kind,
            protected,
            hit_count: 0,
            reused_tokens_total: 0,
            last_used_tick: self.tick,
        });
        self.stats.inserted += 1;
        PrefixInsertOutcome {
            stored: true,
            evicted,
            demoted_from_protected,
            skipped_for_protected: false,
        }
    }

    /// Alias for [`insert_with_meta`](Self::insert_with_meta).
    pub fn insert_checkpoint(
        &mut self,
        tokens: Vec<i32>,
        reuse: PrefixReuse,
        bytes: u64,
        state: S,
        kind: PrefixKind,
        protected: bool,
    ) -> PrefixInsertOutcome {
        self.insert_with_meta(tokens, reuse, bytes, state, kind, protected)
    }

    /// Promote an existing exact entry `tokens` to protected `kind` (if protected cap allows).
    /// Returns `true` if the entry exists and is now protected.
    pub fn promote_protected(&mut self, tokens: &[i32], kind: PrefixKind) -> bool {
        let Some(pos) = self.entries.iter().position(|e| e.tokens == tokens) else {
            return false;
        };
        let existing_kind = self.entries[pos].kind;
        let target_kind = if kind.priority() > existing_kind.priority() {
            kind
        } else {
            existing_kind
        };
        let bytes = self.entries[pos].bytes;
        let already_protected = self.entries[pos].protected;
        let cap = self.protected_cap_for(target_kind);

        if already_protected {
            if existing_kind.priority() >= kind.priority() || self.protected_bytes <= cap {
                self.entries[pos].kind = target_kind;
                return true;
            }
            return false;
        }

        let higher_protected: u64 = self
            .entries
            .iter()
            .filter(|e| e.protected && e.kind.priority() > target_kind.priority())
            .map(|e| e.bytes)
            .sum();
        if bytes > cap || higher_protected.saturating_add(bytes) > cap {
            return false;
        }

        let target_tokens = tokens.to_vec();
        while self.protected_bytes.saturating_add(bytes) > cap {
            let idx = self
                .pick_protected_victim(Some(target_kind.priority()), false)
                .expect("higher_protected + bytes <= cap guarantees an eligible victim");
            self.remove_entry(idx);
        }
        let Some(new_pos) = self.entries.iter().position(|e| e.tokens == target_tokens) else {
            return false;
        };
        self.entries[new_pos].kind = target_kind;
        self.entries[new_pos].protected = true;
        self.protected_bytes += bytes;
        true
    }

    /// Hold `state` for the prefix `tokens` (its `bytes` as the backend measured them). An entry
    /// larger than the whole budget is refused without evicting anything. An entry with the same
    /// tokens is replaced, and an [`AnyPrefix`](PrefixReuse::AnyPrefix) entry supersedes every
    /// unprotected `AnyPrefix` entry whose tokens it extends.
    pub fn insert(
        &mut self,
        tokens: Vec<i32>,
        reuse: PrefixReuse,
        bytes: u64,
        state: S,
    ) -> PrefixInsert {
        let out =
            self.insert_with_meta(tokens, reuse, bytes, state, PrefixKind::Conversation, false);
        PrefixInsert {
            stored: out.stored,
            evicted: out.evicted,
        }
    }

    /// Make room for a request that needs `required` bytes when `available` are free (E7): when
    /// the shortfall can be covered by evicting held entries (evicting unprotected entries first,
    /// then protected only when unprotected bytes do not cover `shortfall`), evict until it is and
    /// return `available` plus what was freed. Otherwise evict nothing and return `available`.
    pub fn reclaim_for(&mut self, required: u64, available: u64) -> u64 {
        let shortfall = required.saturating_sub(available);
        if shortfall == 0 || shortfall > self.resident_bytes {
            return available;
        }
        available.saturating_add(self.evict_at_least(shortfall, false))
    }

    /// [`reclaim_for`](Self::reclaim_for) that never evicts the most-recently-used entry — the one
    /// a [`lookup`](Self::lookup) just hit, whose state the request is about to reuse.
    pub fn reclaim_for_keeping_latest(&mut self, required: u64, available: u64) -> u64 {
        let kept = self.entries.back().map_or(0, |e| e.bytes);
        let shortfall = required.saturating_sub(available);
        if shortfall == 0 || shortfall > self.resident_bytes - kept {
            return available;
        }
        available.saturating_add(self.evict_at_least(shortfall, true))
    }

    /// Request admission with the cache in the picture (E7).
    pub fn admit(&mut self, required: u64, snapshot_bytes: u64, available: u64) -> PrefixAdmission {
        if snapshot_bytes <= self.budget_bytes {
            let over_budget =
                (self.resident_bytes + snapshot_bytes).saturating_sub(self.budget_bytes);
            let short = required
                .saturating_add(snapshot_bytes)
                .saturating_sub(available);
            let need = over_budget.max(short);
            if need <= self.resident_bytes {
                let freed = self.evict_at_least(need, false);
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

    /// Admit a set of planned boundary snapshots `(boundary, snapshot_bytes)` alongside a request
    /// needing `required` bytes when `available` are free. Gracefully drops lowest-priority
    /// boundaries first when tight under cache budget or device headroom, never forces request
    /// refusal, and avoids evicting protected entries solely for unprotected snapshots.
    pub fn admit_planned(
        &mut self,
        required: u64,
        planned: &[(PlannedBoundary, u64)],
        available: u64,
    ) -> PlannedAdmission {
        self.admit_planned_with(
            required,
            planned,
            available,
            PrefixPlannedContext::default(),
        )
    }

    /// Context-aware [`admit_planned`](Self::admit_planned) honoring match pinning (`keep_match`),
    /// in-place promotion (`promote_reused`), and `AnyPrefix` `Conversation` supersession.
    pub fn admit_planned_with(
        &mut self,
        required: u64,
        planned: &[(PlannedBoundary, u64)],
        available: u64,
        ctx: PrefixPlannedContext<'_>,
    ) -> PlannedAdmission {
        let keep_idx = ctx.keep_match.filter(|&i| i < self.entries.len());
        if planned.is_empty() {
            let evicted_before = self.stats.evicted;
            let avail = self.reclaim_for_keeping_idx(required, available, keep_idx);
            let evicted = (self.stats.evicted - evicted_before) as u32;
            return PlannedAdmission {
                admission: PrefixAdmission {
                    available: avail,
                    snapshot: false,
                },
                kept: Vec::new(),
                evicted_entries: evicted,
                fallback: None,
            };
        }

        // Identify an unprotected AnyPrefix Conversation boundary if present and itself feasible,
        // so individually feasible protected boundaries that only fit when it supersedes an
        // AnyPrefix hit are not mistakenly dropped during single-candidate filtering.
        let conv_offset_candidate =
            if ctx.conversation_reuse == Some(PrefixReuse::AnyPrefix) && ctx.prompt.is_some() {
                planned
                    .iter()
                    .copied()
                    .find(|(b, _)| !b.protected && b.kind == PrefixKind::Conversation)
                    .filter(|&c| {
                        self.simulate_evictions(&[c], required, available, ctx)
                            .is_ok()
                    })
            } else {
                None
            };

        // Step 1: filter out individually infeasible candidates first (before combined trimming can
        // drop a feasible unprotected boundary ahead of an impossible protected candidate).
        let mut working = Vec::with_capacity(planned.len());
        let mut trimmed = false;
        for &item in planned {
            let feasible_alone = self
                .simulate_evictions(&[item], required, available, ctx)
                .is_ok();
            let feasible_with_conv = conv_offset_candidate.is_some_and(|cc| {
                cc.0 != item.0
                    && self
                        .simulate_evictions(&[item, cc], required, available, ctx)
                        .is_ok()
            });
            if feasible_alone || feasible_with_conv {
                working.push(item);
            } else {
                trimmed = true;
            }
        }

        let mut blocked_only_by_protected = false;
        while !working.is_empty() {
            match self.simulate_evictions(&working, required, available, ctx) {
                Ok(mut victim_indices) => {
                    victim_indices.sort_unstable_by(|a, b| b.cmp(a));
                    let evicted_before = self.stats.evicted;
                    let mut freed = 0u64;
                    for idx in victim_indices {
                        freed = freed.saturating_add(self.remove_entry(idx));
                    }
                    let evicted = (self.stats.evicted - evicted_before) as u32;
                    let kept: Vec<PlannedBoundary> = working.into_iter().map(|(b, _)| b).collect();
                    return PlannedAdmission {
                        admission: PrefixAdmission {
                            available: available.saturating_add(freed),
                            snapshot: true,
                        },
                        kept,
                        evicted_entries: evicted,
                        fallback: trimmed.then_some(PREFIX_PARTIAL_ADMISSION_TRIMMED),
                    };
                }
                Err(blocked) => {
                    if blocked {
                        blocked_only_by_protected = true;
                    }
                    // If the remaining protected candidates exceed their combined protected cap,
                    // drop the lowest-priority protected boundary before touching unprotected ones.
                    let drop_prot = !self.combined_protected_cap_ok(&working);
                    let drop_idx = working
                        .iter()
                        .enumerate()
                        .filter(|(_, (b, _))| !drop_prot || b.protected)
                        .min_by(|(ia, (a, _)), (ib, (b, _))| {
                            (a.protected, a.kind.priority(), a.position)
                                .cmp(&(b.protected, b.kind.priority(), b.position))
                                .then_with(|| ia.cmp(ib))
                        })
                        .map(|(i, _)| i)
                        .expect("non-empty working set");
                    working.remove(drop_idx);
                    trimmed = true;
                }
            }
        }

        let evicted_before = self.stats.evicted;
        let avail = self.reclaim_for_keeping_idx(required, available, keep_idx);
        let evicted = (self.stats.evicted - evicted_before) as u32;
        PlannedAdmission {
            admission: PrefixAdmission {
                available: avail,
                snapshot: false,
            },
            kept: Vec::new(),
            evicted_entries: evicted,
            fallback: Some(if blocked_only_by_protected {
                PREFIX_PROTECTED_PRESERVED
            } else {
                PREFIX_NOT_ADMITTED
            }),
        }
    }

    fn combined_protected_cap_ok(&self, working: &[(PlannedBoundary, u64)]) -> bool {
        let Some(max_kind) = working
            .iter()
            .filter(|(b, _)| b.protected)
            .map(|(b, _)| b.kind)
            .max()
        else {
            return true;
        };
        let w_prot_bytes = working
            .iter()
            .filter(|(b, _)| b.protected)
            .fold(0u64, |acc, &(_, b)| acc.saturating_add(b));
        if w_prot_bytes > self.protected_cap_for(max_kind) {
            return false;
        }
        for (b, bytes) in working.iter().filter(|(b, _)| b.protected) {
            if *bytes > self.protected_cap_for(b.kind) {
                return false;
            }
        }
        true
    }

    fn pick_unprotected_victim_excluding(
        &self,
        excluded: &[bool],
        keep_pin: Option<usize>,
        promo_pin: Option<usize>,
    ) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if excluded[i] || Some(i) == keep_pin || Some(i) == promo_pin || e.protected {
                continue;
            }
            if best
                .is_none_or(|b| self.compare_unprotected_victim(i, b) == std::cmp::Ordering::Less)
            {
                best = Some(i);
            }
        }
        best
    }

    fn pick_protected_victim_excluding(
        &self,
        max_priority: Option<u8>,
        excluded: &[bool],
        keep_pin: Option<usize>,
        promo_pin: Option<usize>,
    ) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if excluded[i]
                || Some(i) == keep_pin
                || Some(i) == promo_pin
                || !e.protected
                || max_priority.is_some_and(|p| e.kind.priority() > p)
            {
                continue;
            }
            if best.is_none_or(|b| self.compare_protected_victim(i, b) == std::cmp::Ordering::Less)
            {
                best = Some(i);
            }
        }
        best
    }

    fn simulate_evictions(
        &self,
        working: &[(PlannedBoundary, u64)],
        required: u64,
        available: u64,
        ctx: PrefixPlannedContext<'_>,
    ) -> Result<Vec<usize>, bool> {
        let w_prot_bytes = working
            .iter()
            .filter(|(b, _)| b.protected)
            .fold(0u64, |acc, &(_, b)| acc.saturating_add(b));
        let w_unprot_bytes = working
            .iter()
            .filter(|(b, _)| !b.protected)
            .fold(0u64, |acc, &(_, b)| acc.saturating_add(b));
        let total_snap = w_prot_bytes.saturating_add(w_unprot_bytes);
        if total_snap > self.budget_bytes {
            return Err(false);
        }

        let keep_pin = ctx.keep_match.filter(|&i| i < self.entries.len());
        let promo_pin = match (ctx.promote_reused, ctx.prompt) {
            (Some(pb), Some(prompt)) if pb.position > 0 && pb.position <= prompt.len() => {
                let slice = &prompt[..pb.position];
                self.entries.iter().position(|e| e.tokens == slice)
            }
            _ => None,
        };
        let has_any_prefix_conv = ctx.conversation_reuse == Some(PrefixReuse::AnyPrefix)
            && ctx.prompt.is_some()
            && working
                .iter()
                .any(|(b, _)| !b.protected && b.kind == PrefixKind::Conversation);

        let mut excluded = vec![false; self.entries.len()];
        let mut order = Vec::new();
        let mut freed_avail = 0u64;
        let mut sim_resident = self.resident_bytes;
        let mut sim_protected = self.protected_bytes;

        // Phase A: satisfy bare `required` memory shortfall before considering optional snapshots.
        let short_req = required.saturating_sub(available);
        while freed_avail < short_req {
            let idx = self
                .pick_unprotected_victim_excluding(&excluded, keep_pin, promo_pin)
                .or_else(|| {
                    self.pick_protected_victim_excluding(None, &excluded, keep_pin, promo_pin)
                })
                .ok_or(false)?;
            excluded[idx] = true;
            order.push(idx);
            let e = &self.entries[idx];
            freed_avail = freed_avail.saturating_add(e.bytes);
            sim_resident -= e.bytes;
            if e.protected {
                sim_protected -= e.bytes;
            }
        }

        let compute_superseded = |excluded: &[bool]| -> u64 {
            if !has_any_prefix_conv {
                return 0;
            }
            let Some(prompt) = ctx.prompt else {
                return 0;
            };
            self.entries
                .iter()
                .enumerate()
                .filter(|(i, e)| {
                    !excluded[*i]
                        && Some(*i) != promo_pin
                        && !e.protected
                        && e.reuse == PrefixReuse::AnyPrefix
                        && prompt.starts_with(&e.tokens)
                })
                .map(|(_, e)| e.bytes)
                .sum()
        };

        let (promo_extra_prot, promo_target_kind) = match (promo_pin, ctx.promote_reused) {
            (Some(pi), Some(pb)) if !excluded[pi] && !self.entries[pi].protected => {
                let target_kind = if pb.kind.priority() > self.entries[pi].kind.priority() {
                    pb.kind
                } else {
                    self.entries[pi].kind
                };
                let cap = self.protected_cap_for(target_kind);
                let bytes = self.entries[pi].bytes;
                let higher_existing: u64 = self
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(i, e)| {
                        !excluded[*i] && e.protected && e.kind.priority() > target_kind.priority()
                    })
                    .map(|(_, e)| e.bytes)
                    .sum();
                if bytes <= cap && higher_existing.saturating_add(bytes) <= cap {
                    (bytes, Some(target_kind))
                } else {
                    (0, None)
                }
            }
            _ => (0, None),
        };
        sim_protected = sim_protected.saturating_add(promo_extra_prot);

        // Phase B: satisfy protected caps, budget, and headroom for protected candidates ONLY.
        if let Some(max_prot_kind) = working
            .iter()
            .filter(|(b, _)| b.protected)
            .map(|(b, _)| b.kind)
            .max()
        {
            let max_cap = self.protected_cap_for(max_prot_kind);
            if w_prot_bytes > max_cap {
                return Err(false);
            }
            let higher_surviving: u64 = self
                .entries
                .iter()
                .enumerate()
                .filter(|(i, e)| {
                    !excluded[*i] && e.protected && e.kind.priority() > max_prot_kind.priority()
                })
                .map(|(_, e)| e.bytes)
                .sum();
            let promo_higher = promo_target_kind
                .filter(|k| k.priority() > max_prot_kind.priority())
                .map_or(0, |_| promo_extra_prot);
            if higher_surviving
                .saturating_add(promo_higher)
                .saturating_add(w_prot_bytes)
                > max_cap
            {
                return Err(false);
            }
            while sim_protected.saturating_add(w_prot_bytes) > max_cap {
                let idx = self
                    .pick_protected_victim_excluding(
                        Some(max_prot_kind.priority()),
                        &excluded,
                        keep_pin,
                        promo_pin,
                    )
                    .ok_or(false)?;
                excluded[idx] = true;
                order.push(idx);
                let e = &self.entries[idx];
                freed_avail = freed_avail.saturating_add(e.bytes);
                sim_resident -= e.bytes;
                sim_protected -= e.bytes;
            }

            // Ensure every lower-priority protected kind (e.g. `SharedRoot` inserted before
            // `Document`) fits its own cap against surviving higher-priority protected entries.
            for (b, bytes) in working.iter().filter(|(b, _)| b.protected) {
                let cap_b = self.protected_cap_for(b.kind);
                let higher_b: u64 = self
                    .entries
                    .iter()
                    .enumerate()
                    .filter(|(i, e)| {
                        !excluded[*i] && e.protected && e.kind.priority() > b.kind.priority()
                    })
                    .map(|(_, e)| e.bytes)
                    .sum();
                let promo_higher_b = promo_target_kind
                    .filter(|k| k.priority() > b.kind.priority())
                    .map_or(0, |_| promo_extra_prot);
                let same_or_lower_working: u64 = working
                    .iter()
                    .filter(|(wb, _)| wb.protected && wb.kind.priority() <= b.kind.priority())
                    .fold(0u64, |acc, &(_, wbytes)| acc.saturating_add(wbytes));
                if *bytes > cap_b
                    || higher_b
                        .saturating_add(promo_higher_b)
                        .saturating_add(same_or_lower_working)
                        > cap_b
                {
                    return Err(false);
                }
            }

            loop {
                let superseded = compute_superseded(&excluded);
                let eff_resident = sim_resident.saturating_sub(superseded);
                let prot_over_budget = eff_resident
                    .saturating_add(w_prot_bytes)
                    .saturating_sub(self.budget_bytes);
                let prot_short = required
                    .saturating_add(w_prot_bytes)
                    .saturating_sub(available.saturating_add(freed_avail));
                if prot_over_budget == 0 && prot_short == 0 {
                    break;
                }
                let idx = self
                    .pick_unprotected_victim_excluding(&excluded, keep_pin, promo_pin)
                    .or_else(|| {
                        self.pick_protected_victim_excluding(
                            Some(max_prot_kind.priority()),
                            &excluded,
                            keep_pin,
                            promo_pin,
                        )
                    })
                    .ok_or(false)?;
                excluded[idx] = true;
                order.push(idx);
                let e = &self.entries[idx];
                freed_avail = freed_avail.saturating_add(e.bytes);
                sim_resident -= e.bytes;
            }
        }

        // Phase C: satisfy combined budget & headroom for unprotected candidates by evicting ONLY
        // non-pinned unprotected victims. Never evict resident protected entries for unprotected
        // snapshots.
        if w_unprot_bytes > 0 || has_any_prefix_conv {
            loop {
                let superseded = compute_superseded(&excluded);
                let eff_resident = sim_resident.saturating_sub(superseded);
                let over_budget = eff_resident
                    .saturating_add(total_snap)
                    .saturating_sub(self.budget_bytes);
                let short = required
                    .saturating_add(total_snap)
                    .saturating_sub(available.saturating_add(freed_avail));
                let need = over_budget.max(short);
                if need == 0 {
                    break;
                }
                if let Some(idx) =
                    self.pick_unprotected_victim_excluding(&excluded, keep_pin, promo_pin)
                {
                    excluded[idx] = true;
                    order.push(idx);
                    let e = &self.entries[idx];
                    freed_avail = freed_avail.saturating_add(e.bytes);
                    sim_resident -= e.bytes;
                } else {
                    let evictable_prot: u64 = self
                        .entries
                        .iter()
                        .enumerate()
                        .filter(|(i, e)| {
                            !excluded[*i]
                                && Some(*i) != keep_pin
                                && Some(*i) != promo_pin
                                && e.protected
                        })
                        .map(|(_, e)| e.bytes)
                        .sum();
                    let blocked_by_prot = w_prot_bytes == 0
                        && total_snap <= self.budget_bytes
                        && need <= evictable_prot;
                    return Err(blocked_by_prot);
                }
            }
        }

        Ok(order)
    }

    fn reclaim_for_keeping_idx(
        &mut self,
        required: u64,
        available: u64,
        keep: Option<usize>,
    ) -> u64 {
        let shortfall = required.saturating_sub(available);
        if shortfall == 0 || shortfall > self.resident_bytes {
            return available;
        }
        let Some(mut keep_idx) = keep.filter(|&i| i < self.entries.len()) else {
            return self.reclaim_for(required, available);
        };
        let mut freed = 0u64;
        while freed < shortfall {
            let mut best_unprot: Option<usize> = None;
            for i in 0..self.entries.len() {
                if i == keep_idx || self.entries[i].protected {
                    continue;
                }
                if best_unprot.is_none_or(|b| {
                    self.compare_unprotected_victim(i, b) == std::cmp::Ordering::Less
                }) {
                    best_unprot = Some(i);
                }
            }
            let victim = best_unprot.or_else(|| {
                let mut best_prot: Option<usize> = None;
                for i in 0..self.entries.len() {
                    if i == keep_idx || !self.entries[i].protected {
                        continue;
                    }
                    if best_prot.is_none_or(|b| {
                        self.compare_protected_victim(i, b) == std::cmp::Ordering::Less
                    }) {
                        best_prot = Some(i);
                    }
                }
                best_prot
            });
            let idx = match victim {
                Some(i) => i,
                None => {
                    freed = freed.saturating_add(self.remove_entry(keep_idx));
                    return available.saturating_add(freed);
                }
            };
            if idx < keep_idx {
                keep_idx -= 1;
            }
            freed = freed.saturating_add(self.remove_entry(idx));
        }
        available.saturating_add(freed)
    }

    /// Evict entries until at least `bytes` are freed: unprotected entries first (NearTail first,
    /// then lowest score, then oldest tick), then protected entries if unprotected entries alone
    /// cannot cover `bytes` (`SharedRoot` before `Document`).
    fn evict_at_least(&mut self, bytes: u64, skip_back: bool) -> u64 {
        let mut freed = 0u64;
        while freed < bytes {
            let idx = self
                .pick_unprotected_victim(skip_back)
                .or_else(|| self.pick_protected_victim(None, skip_back))
                .expect("the caller checked the eligible resident bytes cover it");
            freed += self.remove_entry(idx);
        }
        freed
    }

    /// Drop every entry.
    pub fn clear(&mut self) {
        self.stats.evicted += self.entries.len();
        self.entries.clear();
        self.resident_bytes = 0;
        self.protected_bytes = 0;
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
        for backend in DecodeBackend::ALL {
            let default = backend.defaults().prefix_cache_bytes;
            assert_eq!(prefix_cache_budget(backend, None, 10, 1 << 40), default);
            assert_eq!(requested_prefix_cache_bytes(backend, None), default);
        }
        let b = DecodeBackend::Mlx;
        assert_eq!(prefix_cache_budget(b, Some(500), 100, 1_000), 500);
        assert_eq!(
            prefix_cache_budget(b, Some(500), 600, 1_000),
            400,
            "the boundary"
        );
        assert_eq!(prefix_cache_budget(b, Some(500), 1_000, 1_000), 0);
        assert_eq!(prefix_cache_budget(b, Some(0), 0, 1_000), 0, "disabled");
        assert_eq!(requested_prefix_cache_bytes(b, Some(7)), 7);
        let (required, available) = (600u64, 1_000u64);
        let budget = prefix_cache_budget(b, Some(u64::MAX), required, available);
        assert!(crate::admit_request_memory(required + budget, available).is_ok());
        assert!(crate::admit_request_memory(required + budget + 1, available).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reclaim beside a hit never evicts the hit entry (the most recently used): what the other
    /// entries free is all it can grant, and a shortfall they cannot cover evicts nothing.
    #[test]
    fn reclaim_keeping_latest_never_evicts_the_hit_entry() {
        let mut store = PrefixStore::new(1_000);
        store.insert(vec![1, 2], PrefixReuse::AnyPrefix, 100, "old");
        store.insert(vec![7, 8], PrefixReuse::AnyPrefix, 300, "hit");
        assert!(store.lookup(&[7, 8, 9], |_| Some(2)).is_some());
        assert_eq!(store.reclaim_for_keeping_latest(500, 450), 550);
        assert_eq!(store.keys(), vec![&[7, 8][..]]);
        assert_eq!(
            store.reclaim_for_keeping_latest(500, 450),
            450,
            "nothing left to free"
        );
        assert_eq!(store.keys(), vec![&[7, 8][..]]);
    }

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

    #[test]
    fn boundary_planner_covers_off_auto_document_near_tail_and_cap() {
        use crate::request::{PrefixCheckpointPolicy, PrefixDocumentSpan};

        let candidates = BoundaryCandidates {
            document: Some(80),
            shared_root: Some(40),
            prev_message: Some(80), // collides with document -> must keep Document + protected: true
            conversation_end: Some(150),
        };
        // Inactive policy -> only conversation_end on WholeEntry; empty on AnyPrefix.
        let off_whole = plan_prefix_boundaries(
            180,
            0,
            PrefixReuse::WholeEntry,
            PrefixCheckpointPolicy::off(),
            candidates,
            NearTailConfig::default(),
        );
        assert_eq!(
            off_whole.captures,
            vec![PlannedBoundary {
                position: 150,
                kind: PrefixKind::Conversation,
                protected: false,
            }]
        );
        assert_eq!(off_whole.promote_reused, None);
        let off_any = plan_prefix_boundaries(
            180,
            0,
            PrefixReuse::AnyPrefix,
            PrefixCheckpointPolicy::off(),
            candidates,
            NearTailConfig::default(),
        );
        assert!(off_any.captures.is_empty());
        assert_eq!(off_any.promote_reused, None);

        // Active AnyPrefix -> only protected boundaries (Document & SharedRoot).
        let auto_any = plan_prefix_boundaries(
            180,
            0,
            PrefixReuse::AnyPrefix,
            PrefixCheckpointPolicy::with_document(PrefixDocumentSpan::Tokens { count: 80 }),
            candidates,
            NearTailConfig::default(),
        );
        assert_eq!(
            auto_any.captures,
            vec![
                PlannedBoundary {
                    position: 40,
                    kind: PrefixKind::SharedRoot,
                    protected: true,
                },
                PlannedBoundary {
                    position: 80,
                    kind: PrefixKind::Document,
                    protected: true,
                },
            ]
        );

        // Active WholeEntry with long uncheckpointed suffix adds aligned NearTail.
        let long_plan = plan_prefix_boundaries(
            600,
            0,
            PrefixReuse::WholeEntry,
            PrefixCheckpointPolicy::with_document(PrefixDocumentSpan::Tokens { count: 80 }),
            candidates,
            NearTailConfig::default(),
        );
        // last_anchor = 150, 600 - 150 = 450 >= 256, no boundary in 472..600 -> ((600-64)/64)*64 = 512.
        assert_eq!(
            long_plan.captures,
            vec![
                PlannedBoundary {
                    position: 40,
                    kind: PrefixKind::SharedRoot,
                    protected: true,
                },
                PlannedBoundary {
                    position: 80,
                    kind: PrefixKind::Document,
                    protected: true,
                },
                PlannedBoundary {
                    position: 150,
                    kind: PrefixKind::Conversation,
                    protected: false,
                },
                PlannedBoundary {
                    position: 512,
                    kind: PrefixKind::NearTail,
                    protected: false,
                },
            ]
        );

        // promote_reused when boundary <= reused on AnyPrefix or == reused on WholeEntry.
        let promoted_whole = plan_prefix_boundaries(
            200,
            80,
            PrefixReuse::WholeEntry,
            PrefixCheckpointPolicy::with_document(PrefixDocumentSpan::Tokens { count: 80 }),
            candidates,
            NearTailConfig::default(),
        );
        assert_eq!(
            promoted_whole.promote_reused,
            Some(PlannedBoundary {
                position: 80,
                kind: PrefixKind::Document,
                protected: true,
            })
        );
        assert_eq!(
            promoted_whole.captures,
            vec![PlannedBoundary {
                position: 150,
                kind: PrefixKind::Conversation,
                protected: false,
            }]
        );
        let promoted_any = plan_prefix_boundaries(
            200,
            150,
            PrefixReuse::AnyPrefix,
            PrefixCheckpointPolicy::with_document(PrefixDocumentSpan::Tokens { count: 80 }),
            candidates,
            NearTailConfig::default(),
        );
        assert_eq!(
            promoted_any.promote_reused,
            Some(PlannedBoundary {
                position: 80,
                kind: PrefixKind::Document,
                protected: true,
            })
        );
        assert!(promoted_any.captures.is_empty());

        // Capping at MAX_CHECKPOINTS_PER_REQUEST preserves protected boundaries over NearTail.
        let five_candidates = BoundaryCandidates {
            document: Some(20),
            shared_root: Some(40),
            prev_message: Some(60),
            conversation_end: Some(80),
        };
        let capped = plan_prefix_boundaries(
            600,
            0,
            PrefixReuse::WholeEntry,
            PrefixCheckpointPolicy::auto(),
            five_candidates,
            NearTailConfig::default(),
        );
        assert_eq!(capped.captures.len(), MAX_CHECKPOINTS_PER_REQUEST);
        assert!(capped
            .captures
            .iter()
            .any(|b| b.position == 20 && b.kind == PrefixKind::Document));
        assert!(capped
            .captures
            .iter()
            .any(|b| b.position == 40 && b.kind == PrefixKind::SharedRoot));
        assert!(!capped
            .captures
            .iter()
            .any(|b| b.kind == PrefixKind::NearTail));
    }

    #[test]
    fn prefix_store_protects_documents_and_scores_unprotected_victims() {
        let mut store = PrefixStore::new(100);
        assert_eq!(store.protected_budget_cap(), 50);

        // Single Document fits up to 3/4 (75 bytes), never 100%, leaving 25 bytes for recent turns.
        let doc = store.insert_with_meta(
            vec![1, 2],
            PrefixReuse::AnyPrefix,
            70,
            "doc",
            PrefixKind::Document,
            true,
        );
        assert!(doc.stored && !doc.demoted_from_protected);
        assert_eq!(store.protected_bytes(), 70);
        assert_eq!(store.protected_budget_cap(), 75);

        // Unprotected AnyPrefix NEVER supersedes a protected AnyPrefix entry.
        let recent = store.insert_with_meta(
            vec![1, 2, 3],
            PrefixReuse::AnyPrefix,
            20,
            "turn",
            PrefixKind::Conversation,
            false,
        );
        assert!(recent.stored && !recent.skipped_for_protected);
        assert_eq!(store.len(), 2);
        assert_eq!(store.protected_bytes(), 70);

        // Oversized unprotected entry that cannot fit beside protected entries is skipped instead
        // of evicting protected Document.
        let oversized_turn = store.insert_with_meta(
            vec![9, 9],
            PrefixReuse::AnyPrefix,
            40,
            "huge_turn",
            PrefixKind::Conversation,
            false,
        );
        assert!(!oversized_turn.stored && oversized_turn.skipped_for_protected);
        assert_eq!(store.keys(), vec![&[1, 2][..], &[1, 2, 3][..]]);
        assert_eq!(store.protected_bytes(), 70);

        // Protected insert past cap demotes instead of monopolizing cache.
        let oversized_root = store.insert_with_meta(
            vec![5, 5],
            PrefixReuse::WholeEntry,
            60,
            "oversized_root",
            PrefixKind::SharedRoot,
            true,
        );
        // Existing Document is 70, remainingprotected + 60 = 130 > 100 -> demotes AND skipped for protected.
        assert!(oversized_root.demoted_from_protected);
        assert!(oversized_root.skipped_for_protected);

        // Value-aware victim selection: NearTail is evicted before Conversation / MessageBoundary.
        let mut val_store = PrefixStore::new(100);
        val_store.insert_with_meta(
            vec![10, 11],
            PrefixReuse::WholeEntry,
            40,
            "msg",
            PrefixKind::MessageBoundary,
            false,
        );
        val_store.insert_with_meta(
            vec![20, 21],
            PrefixReuse::WholeEntry,
            40,
            "tail",
            PrefixKind::NearTail,
            false,
        );
        // Inserting another 40-byte entry must evict `tail` even though `msg` is older.
        val_store.insert_with_meta(
            vec![30, 31],
            PrefixReuse::WholeEntry,
            40,
            "conv",
            PrefixKind::Conversation,
            false,
        );
        assert_eq!(val_store.keys(), vec![&[10, 11][..], &[30, 31][..]]);

        // Reclaim under tight memory pressure evicts unprotected first, then protected only when needed.
        assert_eq!(store.reclaim_for(120, 50), 140); // shortfall=70 frees recent(20) then doc(70) -> 50 + 90 = 140
        assert!(store.is_empty());
        assert_eq!(store.protected_bytes(), 0);
    }

    #[test]
    fn lookup_diagnostic_promote_protected_and_partial_admission_work() {
        let mut store = PrefixStore::new(100);
        // Stored without MTP: `(false, len)`
        store.insert_with_meta(
            vec![1, 2, 3, 4],
            PrefixReuse::WholeEntry,
            20,
            (false, 4),
            PrefixKind::Conversation,
            false,
        );
        store.insert_with_meta(
            vec![1, 2],
            PrefixReuse::WholeEntry,
            10,
            (true, 2),
            PrefixKind::SharedRoot,
            false,
        );

        let diag = store.lookup_with_proposer_diagnostic(&[1, 2, 3, 4, 5], |&(has_mtp, len)| {
            has_mtp.then_some(len)
        });
        assert!(diag.proposer_skipped_longer);
        let hit = diag.hit.unwrap();
        assert_eq!(hit.reused, 2);
        assert_eq!(hit.kind, PrefixKind::SharedRoot);

        // Promote existing unprotected entry to protected Document.
        assert!(store.promote_protected(&[1, 2], PrefixKind::Document));
        assert_eq!(store.protected_bytes(), 10);

        // Partial admission trimming drops NearTail before Conversation to stay under budget.
        let plan = vec![
            (
                PlannedBoundary {
                    position: 10,
                    kind: PrefixKind::MessageBoundary,
                    protected: false,
                },
                30,
            ),
            (
                PlannedBoundary {
                    position: 50,
                    kind: PrefixKind::Conversation,
                    protected: false,
                },
                30,
            ),
            (
                PlannedBoundary {
                    position: 90,
                    kind: PrefixKind::NearTail,
                    protected: false,
                },
                50,
            ),
        ];
        // Protected = 10 -> unprotected budget = 90. 30+30+50 = 110 > 90 -> trims NearTail(50) -> 60 <= 90!
        let adm = store.admit_planned(10, &plan, 1_000);
        assert!(adm.admission.snapshot);
        assert_eq!(adm.kept.len(), 2);
        assert!(!adm.kept.iter().any(|b| b.kind == PrefixKind::NearTail));
        assert_eq!(adm.fallback, Some(PREFIX_PARTIAL_ADMISSION_TRIMMED));
    }

    #[test]
    fn any_prefix_non_exact_document_inside_reused_turn_stays_priced_in_captures() {
        let mut store = PrefixStore::new(500);
        let turn1_prompt: Vec<i32> = (1..=100).collect();
        store.insert_with_meta(
            turn1_prompt.clone(),
            PrefixReuse::AnyPrefix,
            200,
            "turn1",
            PrefixKind::Conversation,
            false,
        );

        let mut turn2_prompt = turn1_prompt;
        turn2_prompt.extend(101..=150);
        let (match_idx, reused) = store
            .peek_match(&turn2_prompt, |_| Some(100))
            .expect("turn1 reused");
        assert_eq!(reused, 100);

        let cands = BoundaryCandidates {
            document: Some(60), // <= reused (100), but NOT an exact key in store!
            shared_root: None,
            prev_message: Some(100),
            conversation_end: None,
        };
        let plan = plan_prefix_boundaries_with_exact(
            turn2_prompt.len(),
            reused,
            PrefixReuse::AnyPrefix,
            crate::request::PrefixCheckpointPolicy::with_document(
                crate::request::PrefixDocumentSpan::Tokens { count: 60 },
            ),
            cands,
            NearTailConfig::default(),
            |pos| store.contains_exact(&turn2_prompt[..pos]),
        );
        assert_eq!(plan.promote_reused, None);
        assert_eq!(
            plan.captures,
            vec![PlannedBoundary {
                position: 60,
                kind: PrefixKind::Document,
                protected: true,
            }]
        );

        let planned_with_costs = vec![
            (plan.captures[0], 120),
            (
                PlannedBoundary {
                    position: turn2_prompt.len(),
                    kind: PrefixKind::Conversation,
                    protected: false,
                },
                260,
            ),
        ];
        let ctx = PrefixPlannedContext {
            prompt: Some(&turn2_prompt),
            keep_match: Some(match_idx),
            promote_reused: plan.promote_reused,
            conversation_reuse: Some(PrefixReuse::AnyPrefix),
        };
        // Full headroom keeps both Document@60 and superseded Conversation@150 without evicting
        // Turn-1 before restore.
        let adm = store.admit_planned_with(100, &planned_with_costs, 1_000, ctx);
        assert!(adm.admission.snapshot);
        assert_eq!(adm.evicted_entries, 0);
        assert_eq!(adm.kept.len(), 2);
        assert!(store.contains_exact(&turn2_prompt[..100]));

        // Tight headroom (`available = 200`, `required = 100`, `Document` = 120 fails `100 + 120 > 200`,
        // while `Conversation` if <= 100 or partial trims cleanly) drops Document rather than
        // bypassing E7 headroom.
        let tight_costs = vec![(plan.captures[0], 120), (planned_with_costs[1].0, 80)];
        let tight = store.admit_planned_with(100, &tight_costs, 200, ctx);
        assert!(tight.admission.snapshot);
        assert!(!tight.kept.iter().any(|b| b.kind == PrefixKind::Document));
        assert_eq!(tight.fallback, Some(PREFIX_PARTIAL_ADMISSION_TRIMMED));
    }

    #[test]
    fn promote_protected_preserves_existing_protection_above_half_budget() {
        let mut store = PrefixStore::new(100);
        store.insert_with_meta(
            vec![3, 4],
            PrefixReuse::WholeEntry,
            10,
            "root",
            PrefixKind::SharedRoot,
            true,
        );
        store.insert_with_meta(
            vec![1, 2],
            PrefixReuse::AnyPrefix,
            60,
            "doc",
            PrefixKind::Document,
            true,
        );
        assert_eq!(store.protected_bytes(), 70);

        // Re-promoting already-protected SharedRoot must still succeed even though D1 holds
        // protected_bytes (70) > budget_bytes / 2 (50).
        assert!(store.promote_protected(&[3, 4], PrefixKind::SharedRoot));
        // Promoting existing Document with lower kind must succeed and never downgrade Document.
        assert!(store.promote_protected(&[1, 2], PrefixKind::SharedRoot));
        let doc_hit = store.lookup(&[1, 2, 9], |_| Some(2)).unwrap();
        assert_eq!(doc_hit.kind, PrefixKind::Document);
        assert!(doc_hit.protected);
        // Upgrading SharedRoot to Document succeeds within 3/4 cap (70 <= 75).
        assert!(store.promote_protected(&[3, 4], PrefixKind::Document));
        let upgraded = store.lookup(&[3, 4, 9], |_| Some(2)).unwrap();
        assert_eq!(upgraded.kind, PrefixKind::Document);
    }

    #[test]
    fn admit_planned_with_pins_hit_and_handles_over_half_budget_self_supersede() {
        let mut store = PrefixStore::new(1_000);
        store.insert_with_meta(
            vec![1, 2, 3, 4],
            PrefixReuse::AnyPrefix,
            600,
            "c1",
            PrefixKind::Conversation,
            false,
        );
        let prompt = [1, 2, 3, 4, 5, 6];
        let (idx, reused) = store.peek_match(&prompt, |_| Some(4)).unwrap();
        assert_eq!((idx, reused), (0, 4));
        let conv_boundary = PlannedBoundary {
            position: prompt.len(),
            kind: PrefixKind::Conversation,
            protected: false,
        };
        let adm = store.admit_planned_with(
            300,
            &[(conv_boundary, 650)],
            2_000,
            PrefixPlannedContext {
                prompt: Some(&prompt),
                keep_match: Some(idx),
                promote_reused: None,
                conversation_reuse: Some(PrefixReuse::AnyPrefix),
            },
        );
        assert!(adm.admission.snapshot);
        assert_eq!(
            adm.evicted_entries, 0,
            "hit entry must not be evicted before restore"
        );
        assert!(store.contains_exact(&[1, 2, 3, 4]));

        // When optional snapshot cannot fit available memory alongside pinned C1, snapshot is
        // refused while C1 remains intact; only bare `required` shortfall evicts C1.
        let no_snap = store.admit_planned_with(
            300,
            &[(conv_boundary, 650)],
            500,
            PrefixPlannedContext {
                prompt: Some(&prompt),
                keep_match: Some(idx),
                promote_reused: None,
                conversation_reuse: Some(PrefixReuse::AnyPrefix),
            },
        );
        assert!(!no_snap.admission.snapshot);
        assert_eq!(no_snap.evicted_entries, 0);
        assert!(store.contains_exact(&[1, 2, 3, 4]));

        let bare_req = store.admit_planned_with(
            800,
            &[(conv_boundary, 650)],
            500,
            PrefixPlannedContext {
                prompt: Some(&prompt),
                keep_match: Some(idx),
                promote_reused: None,
                conversation_reuse: Some(PrefixReuse::AnyPrefix),
            },
        );
        assert!(!bare_req.admission.snapshot);
        assert_eq!(bare_req.evicted_entries, 1);
        assert_eq!(bare_req.admission.available, 1_100);
        assert!(!store.contains_exact(&[1, 2, 3, 4]));
    }

    #[test]
    fn admit_planned_drops_infeasible_protected_before_feasible_conversation() {
        // Case (a1): oversized Document (> 3/4 cap) dropped first; Conversation kept.
        let mut store: PrefixStore<&str> = PrefixStore::new(100);
        let doc_huge = PlannedBoundary {
            position: 20,
            kind: PrefixKind::Document,
            protected: true,
        };
        let conv = PlannedBoundary {
            position: 50,
            kind: PrefixKind::Conversation,
            protected: false,
        };
        let adm = store.admit_planned(10, &[(doc_huge, 80), (conv, 40)], 1_000);
        assert!(adm.admission.snapshot);
        assert_eq!(adm.kept, vec![conv]);
        assert_eq!(adm.fallback, Some(PREFIX_PARTIAL_ADMISSION_TRIMMED));

        // Case (a2): Document violates tight headroom (`required 70 + 50 > 100`), Conversation(20)
        // still fits headroom -> drops Document, keeps Conversation.
        let doc_ok_cap = PlannedBoundary {
            position: 20,
            kind: PrefixKind::Document,
            protected: true,
        };
        let adm_tight = store.admit_planned(70, &[(doc_ok_cap, 50), (conv, 20)], 100);
        assert!(adm_tight.admission.snapshot);
        assert_eq!(adm_tight.kept, vec![conv]);
        assert_eq!(adm_tight.fallback, Some(PREFIX_PARTIAL_ADMISSION_TRIMMED));

        // Case (a3): SharedRoot(40) + Document(50) exceed Document cap (75) together -> drops
        // excess protected SharedRoot before touching Conversation(20).
        let root = PlannedBoundary {
            position: 10,
            kind: PrefixKind::SharedRoot,
            protected: true,
        };
        let adm_multi_prot =
            store.admit_planned(0, &[(root, 40), (doc_ok_cap, 50), (conv, 20)], 1_000);
        assert!(adm_multi_prot.admission.snapshot);
        assert_eq!(adm_multi_prot.kept, vec![doc_ok_cap, conv]);
        assert_eq!(
            adm_multi_prot.fallback,
            Some(PREFIX_PARTIAL_ADMISSION_TRIMMED)
        );
    }

    #[test]
    fn admit_planned_never_evicts_resident_protected_for_unprotected_snapshot() {
        let mut store = PrefixStore::new(100);
        store.insert_with_meta(
            vec![1, 2],
            PrefixReuse::WholeEntry,
            40,
            "d_old",
            PrefixKind::Document,
            true,
        );
        store.insert_with_meta(
            vec![3, 4],
            PrefixReuse::WholeEntry,
            20,
            "u_old",
            PrefixKind::Conversation,
            false,
        );

        let doc_new = PlannedBoundary {
            position: 20,
            kind: PrefixKind::Document,
            protected: true,
        };
        let conv_new = PlannedBoundary {
            position: 80,
            kind: PrefixKind::Conversation,
            protected: false,
        };
        // D_new(20) fits cleanly alongside D_old(40) + U_old(20) (80 <= 100, 60 <= 75), whereas
        // adding Conv_new(50) would push total to 130 > 100 and U_old(20) cannot cover over=30.
        // Conv_new must be dropped without evicting resident protected D_old.
        let adm = store.admit_planned(0, &[(doc_new, 20), (conv_new, 50)], 1_000);
        assert!(adm.admission.snapshot);
        assert_eq!(adm.kept, vec![doc_new]);
        assert_eq!(adm.fallback, Some(PREFIX_PARTIAL_ADMISSION_TRIMMED));
        assert!(store.contains_exact(&[1, 2]), "D_old must remain resident");
        assert_eq!(store.protected_bytes(), 40);
    }

    #[test]
    fn lookup_tie_prefers_protected_and_insert_never_downgrades_protected_kind() {
        let mut store = PrefixStore::new(200);
        store.insert_with_meta(
            vec![1, 2, 3, 4],
            PrefixReuse::WholeEntry,
            20,
            "doc",
            PrefixKind::Document,
            true,
        );
        // Newer MRU unprotected Conversation with same 4-token overlap on prompt `[1,2,3,4,5]`.
        store.insert_with_meta(
            vec![1, 2, 3, 4, 9],
            PrefixReuse::AnyPrefix,
            30,
            "conv_mru",
            PrefixKind::Conversation,
            false,
        );

        let hit = store
            .lookup_with_proposer_diagnostic(&[1, 2, 3, 4, 5], |_| Some(4))
            .hit
            .expect("tie hit");
        assert_eq!(hit.reused, 4);
        assert_eq!(hit.kind, PrefixKind::Document);
        assert!(hit.protected);
        assert_eq!(store.stats().hits, 1);

        // Inserting protected SharedRoot over an existing exact protected Document keeps Document.
        let out = store.insert_with_meta(
            vec![1, 2, 3, 4],
            PrefixReuse::WholeEntry,
            20,
            "doc2",
            PrefixKind::SharedRoot,
            true,
        );
        assert!(out.stored);
        let hit2 = store.lookup(&[1, 2, 3, 4, 5], |_| Some(4)).unwrap();
        assert_eq!(hit2.kind, PrefixKind::Document);
        assert!(hit2.protected);
    }
}
