//! Backend-neutral continuous-batching scheduler policy (epic 7153, story 7167).
//!
//! On-device serving runs all concurrency in the **batch dimension** under a single eval thread
//! (MLX's default Metal device is not thread-safe; the same is true of a single Candle CUDA stream).
//! Throughput therefore comes from packing several requests into one forward step, not from threads.
//! The host-side bookkeeping that decides *which* sequences are in the batch, *appends* each sampled
//! token, and *retires* a sequence when it hits a stop token or its length budget is identical across
//! backends — so, following the epic's concrete-first rule, it is lifted here once the working MLX
//! batched decode validated the shape. A backend ([`mlx-llm`], later `candle-llm`) owns only the
//! tensors; it drives a [`Scheduler`] for the policy.
//!
//! The [`Scheduler`] is **tensor-free**: the backend feeds it sampled token ids and reads back which
//! sequences are still active and which have finished. The per-sequence offset ([`Scheduler::offset`])
//! is first-class so the same policy drives a ragged paged batch (story 7169) without change — only
//! the cache behind it swaps.
//!
//! Retirement matches the single-sequence decode loop exactly so a batched row is identical to the
//! same request run alone: a sampled **stop token ends the sequence and is excluded** from the
//! output; otherwise the token is appended and, if that fills the `max_new_tokens` budget, the
//! sequence finishes with [`FinishReason::Length`] (the budget-filling token *is* included).
//!
//! [`mlx-llm`]: https://github.com/SceneWorks/mlx-llm
//!
//! ```
//! use core_llm::schedule::{Scheduler, SeqSpec};
//! use core_llm::FinishReason;
//!
//! let mut sched = Scheduler::new();
//! let a = sched.admit(SeqSpec::new(vec![1, 2, 3], 4, vec![99]));
//! let b = sched.admit(SeqSpec::new(vec![7], 4, vec![99]));
//!
//! // The backend runs a batched step over `active()`, samples a token per row, then records it.
//! assert_eq!(sched.active(), vec![a, b]);
//! sched.record(a, 5);   // appended
//! sched.record(b, 99);  // stop token -> b retires, excluded from b's output
//!
//! assert_eq!(sched.active(), vec![a]);          // b retired; a still running
//! assert_eq!(sched.finish_reason(b), Some(FinishReason::Stop));
//! assert_eq!(sched.generated(b), &[] as &[i32]); // stop token not in the output
//! ```

use crate::output::FinishReason;

/// A handle to a sequence admitted to a [`Scheduler`], stable for the sequence's lifetime.
///
/// It is the index into the scheduler's admission order, so a backend can use it directly to label
/// per-sequence host state (RNG, detokenizer, left-pad width) that lives alongside the batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SeqId(pub usize);

/// How to admit one sequence: its prompt plus the per-sequence decode policy.
#[derive(Clone, Debug)]
pub struct SeqSpec {
    /// The prompt token ids (already rendered + tokenized by the caller).
    pub prompt_tokens: Vec<i32>,
    /// Maximum new tokens to generate for this sequence. `0` ⇒ the sequence is admitted already
    /// finished ([`FinishReason::Length`], empty output), mirroring a zero-budget single request.
    pub max_new_tokens: usize,
    /// Token ids that stop this sequence when sampled (EOS / EOT / …). A stop token ends the
    /// sequence and is excluded from its output.
    pub stop_tokens: Vec<i32>,
}

impl SeqSpec {
    /// Construct a spec from its parts.
    pub fn new(prompt_tokens: Vec<i32>, max_new_tokens: usize, stop_tokens: Vec<i32>) -> Self {
        Self {
            prompt_tokens,
            max_new_tokens,
            stop_tokens,
        }
    }
}

/// Per-sequence decode state the scheduler tracks.
#[derive(Clone, Debug)]
struct SeqState {
    prompt_len: usize,
    generated: Vec<i32>,
    max_new_tokens: usize,
    stop_tokens: Vec<i32>,
    /// `Some` once the sequence has retired; `None` while it is still active.
    finish: Option<FinishReason>,
}

impl SeqState {
    fn from_spec(spec: SeqSpec) -> Self {
        let prompt_len = spec.prompt_tokens.len();
        // A zero-budget request produces no tokens and retires immediately, like `0..0` in the
        // single-sequence loop.
        let finish = (spec.max_new_tokens == 0).then_some(FinishReason::Length);
        Self {
            prompt_len,
            generated: Vec::new(),
            max_new_tokens: spec.max_new_tokens,
            stop_tokens: spec.stop_tokens,
            finish,
        }
    }
}

/// The continuous-batching scheduler: admission, per-step active set, and per-sequence retirement.
///
/// A backend drives it each step: run a batched forward over [`Scheduler::active`], sample one token
/// per active sequence, [`Scheduler::record`] each, and repeat until [`Scheduler::all_finished`].
/// The scheduler owns no tensors and makes no scheduling-vs-tensor assumptions beyond "one sampled
/// token per active sequence per step", so it is reused verbatim by every backend.
#[derive(Clone, Debug, Default)]
pub struct Scheduler {
    /// Indexed by `SeqId.0`, in admission order. Sequences are never removed (so ids stay stable);
    /// retirement is recorded in `finish`.
    seqs: Vec<SeqState>,
}

impl Scheduler {
    /// A fresh scheduler with no sequences.
    pub fn new() -> Self {
        Self::default()
    }

    /// Admit a sequence, returning its stable [`SeqId`]. A zero-budget sequence is admitted already
    /// finished (it never appears in [`Scheduler::active`]).
    pub fn admit(&mut self, spec: SeqSpec) -> SeqId {
        let id = SeqId(self.seqs.len());
        self.seqs.push(SeqState::from_spec(spec));
        id
    }

    /// Total sequences ever admitted (active + finished).
    pub fn len(&self) -> usize {
        self.seqs.len()
    }

    /// Whether no sequence has been admitted.
    pub fn is_empty(&self) -> bool {
        self.seqs.is_empty()
    }

    /// The sequences still generating, in admission order — the batch the backend runs this step.
    pub fn active(&self) -> Vec<SeqId> {
        self.seqs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.finish.is_none())
            .map(|(i, _)| SeqId(i))
            .collect()
    }

    /// Whether `id` is still generating (not yet retired).
    pub fn is_active(&self, id: SeqId) -> bool {
        self.state(id).is_some_and(|s| s.finish.is_none())
    }

    /// Whether every admitted sequence has retired (the driver's loop condition).
    pub fn all_finished(&self) -> bool {
        self.seqs.iter().all(|s| s.finish.is_some())
    }

    /// Every retired sequence, in admission order.
    pub fn finished(&self) -> Vec<SeqId> {
        self.seqs
            .iter()
            .enumerate()
            .filter(|(_, s)| s.finish.is_some())
            .map(|(i, _)| SeqId(i))
            .collect()
    }

    /// Record the token sampled for active sequence `id` and apply the retirement policy.
    ///
    /// Returns the [`FinishReason`] if this token retired the sequence, else `None`. A stop token
    /// retires the sequence ([`FinishReason::Stop`]) and is **not** appended to its output; any other
    /// token is appended and, if it fills the `max_new_tokens` budget, retires the sequence
    /// ([`FinishReason::Length`]). Recording onto an already-finished sequence is a no-op (returns
    /// `None`); the backend only feeds [`Scheduler::active`] sequences, so this guards against a
    /// double-record rather than expecting one.
    pub fn record(&mut self, id: SeqId, token: i32) -> Option<FinishReason> {
        let s = self.seqs.get_mut(id.0)?;
        if s.finish.is_some() {
            return None;
        }
        if s.stop_tokens.contains(&token) {
            s.finish = Some(FinishReason::Stop); // stop token excluded from the output
            return Some(FinishReason::Stop);
        }
        s.generated.push(token);
        if s.generated.len() >= s.max_new_tokens {
            s.finish = Some(FinishReason::Length); // budget-filling token is included
            return Some(FinishReason::Length);
        }
        None
    }

    /// The tokens generated so far for `id` (excludes the prompt and any stop token).
    pub fn generated(&self, id: SeqId) -> &[i32] {
        self.state(id).map(|s| s.generated.as_slice()).unwrap_or(&[])
    }

    /// Why `id` retired, or `None` if it is still active (or unknown).
    pub fn finish_reason(&self, id: SeqId) -> Option<FinishReason> {
        self.state(id).and_then(|s| s.finish)
    }

    /// The prompt length `id` was admitted with.
    pub fn prompt_len(&self, id: SeqId) -> usize {
        self.state(id).map(|s| s.prompt_len).unwrap_or(0)
    }

    /// The next absolute position for `id` — `prompt_len + generated.len()`, i.e. the RoPE offset
    /// for its next token. Per-sequence by construction, so the same value drives a ragged paged
    /// batch unchanged (story 7169).
    pub fn offset(&self, id: SeqId) -> usize {
        self.state(id)
            .map(|s| s.prompt_len + s.generated.len())
            .unwrap_or(0)
    }

    fn state(&self, id: SeqId) -> Option<&SeqState> {
        self.seqs.get(id.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admit_assigns_sequential_ids() {
        let mut s = Scheduler::new();
        assert!(s.is_empty());
        let a = s.admit(SeqSpec::new(vec![1, 2], 8, vec![]));
        let b = s.admit(SeqSpec::new(vec![3], 8, vec![]));
        assert_eq!(a, SeqId(0));
        assert_eq!(b, SeqId(1));
        assert_eq!(s.len(), 2);
        assert_eq!(s.active(), vec![a, b]);
        assert!(!s.all_finished());
    }

    #[test]
    fn offset_tracks_prompt_plus_generated() {
        let mut s = Scheduler::new();
        let a = s.admit(SeqSpec::new(vec![1, 2, 3], 8, vec![]));
        assert_eq!(s.prompt_len(a), 3);
        assert_eq!(s.offset(a), 3); // next position is right after the prompt
        s.record(a, 10);
        assert_eq!(s.offset(a), 4);
        s.record(a, 11);
        assert_eq!(s.offset(a), 5);
        assert_eq!(s.generated(a), &[10, 11]);
    }

    #[test]
    fn stop_token_retires_and_is_excluded() {
        let mut s = Scheduler::new();
        let a = s.admit(SeqSpec::new(vec![1], 8, vec![99]));
        assert_eq!(s.record(a, 5), None);
        assert_eq!(s.record(a, 99), Some(FinishReason::Stop));
        assert!(!s.is_active(a));
        assert_eq!(s.finish_reason(a), Some(FinishReason::Stop));
        assert_eq!(s.generated(a), &[5]); // 99 excluded
        assert!(s.all_finished());
        assert_eq!(s.active(), vec![]);
        assert_eq!(s.finished(), vec![a]);
    }

    #[test]
    fn length_budget_retires_and_includes_last_token() {
        let mut s = Scheduler::new();
        let a = s.admit(SeqSpec::new(vec![1], 3, vec![99]));
        assert_eq!(s.record(a, 10), None);
        assert_eq!(s.record(a, 11), None);
        assert_eq!(s.record(a, 12), Some(FinishReason::Length)); // 3rd token fills the budget
        assert_eq!(s.finish_reason(a), Some(FinishReason::Length));
        assert_eq!(s.generated(a), &[10, 11, 12]); // budget-filling token included
    }

    #[test]
    fn zero_budget_is_admitted_already_finished() {
        let mut s = Scheduler::new();
        let a = s.admit(SeqSpec::new(vec![1, 2], 0, vec![]));
        assert!(!s.is_active(a));
        assert!(s.all_finished());
        assert_eq!(s.active(), vec![]);
        assert_eq!(s.finish_reason(a), Some(FinishReason::Length));
        assert_eq!(s.generated(a), &[] as &[i32]);
    }

    #[test]
    fn recording_onto_finished_sequence_is_a_noop() {
        let mut s = Scheduler::new();
        let a = s.admit(SeqSpec::new(vec![1], 1, vec![]));
        assert_eq!(s.record(a, 10), Some(FinishReason::Length));
        // A stray record after retirement must not append or change the reason.
        assert_eq!(s.record(a, 11), None);
        assert_eq!(s.generated(a), &[10]);
        assert_eq!(s.finish_reason(a), Some(FinishReason::Length));
    }

    #[test]
    fn active_set_shrinks_as_sequences_retire_independently() {
        // Three sequences of differing budgets retire at different steps; the active set tracks it.
        let mut s = Scheduler::new();
        let a = s.admit(SeqSpec::new(vec![1], 1, vec![]));
        let b = s.admit(SeqSpec::new(vec![2], 2, vec![]));
        let c = s.admit(SeqSpec::new(vec![3], 3, vec![]));
        assert_eq!(s.active(), vec![a, b, c]);

        // Step 1: all three take a token; a (budget 1) retires.
        s.record(a, 10);
        s.record(b, 10);
        s.record(c, 10);
        assert_eq!(s.active(), vec![b, c]);

        // Step 2: b and c take a token; b (budget 2) retires.
        s.record(b, 11);
        s.record(c, 11);
        assert_eq!(s.active(), vec![c]);

        // Step 3: c retires.
        s.record(c, 12);
        assert!(s.all_finished());
        assert_eq!(s.generated(a), &[10]);
        assert_eq!(s.generated(b), &[10, 11]);
        assert_eq!(s.generated(c), &[10, 11, 12]);
    }

    #[test]
    fn unknown_id_queries_are_safe() {
        let s = Scheduler::new();
        let ghost = SeqId(42);
        assert!(!s.is_active(ghost));
        assert_eq!(s.finish_reason(ghost), None);
        assert_eq!(s.generated(ghost), &[] as &[i32]);
        assert_eq!(s.offset(ghost), 0);
    }
}
