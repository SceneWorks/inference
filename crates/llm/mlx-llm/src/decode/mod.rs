//! Streaming, cancellable decoding (story 7156).
//!
//! [`generate`] is the model-agnostic decode loop; [`Decode`] is the seam any model implements to
//! be driven by it. [`StreamEvent`]s are emitted per token through a callback. This is the internal
//! streaming API the backend-neutral `core-llm` contract (story 7154) is later extracted from.

use core_llm::schedule::{Scheduler, SeqId};
use core_llm::FinishReason as CoreFinish;

use self::stream::{FinishReason as StreamFinishReason, StreamEvent as DecodeEvent};
use crate::primitives::kv_cache::KV_BLOCK_TOKENS;

pub mod batch;
pub mod cancel;
pub mod continuous;
pub mod prefix;
pub mod qwen35_mtp;
pub mod speculative;
pub mod stream;

pub use batch::{generate_batch, BatchRequest};
pub use cancel::CancelFlag;
pub use continuous::{generate_continuous, BatchExactness, ContinuousConfig};
pub use prefix::{generate_cached, generate_cached_with, PrefixCache, PrefixStats};
pub use qwen35_mtp::{generate_qwen35_mtp, Qwen35MtpMultimodalPrompt, RewindableConstraintMask};
pub(crate) use qwen35_mtp::{
    generate_qwen35_mtp_multimodal_with_timings, generate_qwen35_mtp_with_timings,
};
pub use speculative::{
    generate_draft_speculative, generate_prompt_lookup, SpeculativeConfig, SpeculativeStats,
};
pub use stream::{
    generate, generate_from_prefill, generate_with, generate_with_cache, ConstraintMask, Decode,
    FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};
pub(crate) use stream::{generate_from_prefill_with_timings, generate_with_timings};

/// Generated tokens between releases of MLX's freed-buffer cache during decode. The KV block size
/// ([`KV_BLOCK_TOKENS`]) so each release lands right after a block growth has retired the
/// previous, smaller buffers — the allocator only reuses a freed buffer for a same-sized request,
/// so without a periodic release those retired buffers pile up for the whole generation.
pub(crate) const BUFFER_RELEASE_TOKENS: usize = KV_BLOCK_TOKENS as usize;

/// Releases MLX's freed-buffer cache once the first decode step has retired the prefill's
/// transients, and once every [`BUFFER_RELEASE_TOKENS`] generated tokens after that. One instance
/// per generation, shared by every decode loop; it only ever returns *unused* buffers to the OS
/// and never touches live arrays, so it is safe at any point between forwards.
///
/// The post-prefill release rides [`Self::advance`] instead of sitting in the constructor. Every
/// loop builds its counter where borrows allow, which is immediately after the first sample — and
/// at that instant the prefill `logits` (plus, on the MTP path, the prompt hidden states) are
/// *still live*. A release taken there clears a nearly empty cache: the prefill residue is freed a
/// moment later, when step 0 reassigns or drops those bindings, and then sits in MLX's
/// freed-buffer cache until the first periodic release, [`BUFFER_RELEASE_TOKENS`] tokens later.
/// Measured on a 5.5k-token prefill, that stranded ~5.4 GB for the first 256 generated tokens.
///
/// Deferring to the first `advance` fixes all six loops under one rule — *release only at a
/// completed-token boundary, never between the prefill and step 0* — rather than six separate
/// per-loop orderings, and it costs one line per call site. Construction is now side-effect free,
/// so the loops that must build the counter early can keep doing so. A call site still has to drop
/// any prefill array it owns before that first `advance`; the bindings that outlive the loop body
/// (`batch`, `speculative` ×2, `qwen35_mtp`) do so explicitly.
pub(crate) struct BufferRelease {
    tokens: usize,
    prefill_released: bool,
}

impl BufferRelease {
    /// A counter for a fresh generation. Takes no release of its own — the prefill's transients
    /// are still live at every loop's construction point.
    pub(crate) fn new() -> Self {
        Self {
            tokens: 0,
            prefill_released: false,
        }
    }

    /// Account for `n` more generated tokens, releasing MLX's freed-buffer cache on the first call
    /// (the post-prefill release, now that step 0 has retired the prompt-length transients) and
    /// then whenever the count crosses a [`BUFFER_RELEASE_TOKENS`] boundary.
    pub(crate) fn advance(&mut self, n: usize) {
        if self.count(n) {
            mlx_rs::memory::clear_cache();
        }
    }

    /// The release decision for `n` more tokens. Plain counting with no MLX calls, so the cadence
    /// is unit-testable without a device.
    fn count(&mut self, n: usize) -> bool {
        let before = self.tokens / BUFFER_RELEASE_TOKENS;
        self.tokens += n;
        if !self.prefill_released {
            self.prefill_released = true;
            return true;
        }
        self.tokens / BUFFER_RELEASE_TOKENS > before
    }
}

pub(super) enum LaneStep {
    Continue,
    Done,
}

pub(super) fn record_lane_token(
    sched: &mut Scheduler,
    seq: SeqId,
    request_index: usize,
    tok: i32,
    history: &mut Vec<i32>,
    next_token: &mut i32,
    on_event: &mut dyn FnMut(usize, DecodeEvent),
) -> LaneStep {
    match sched.record(seq, tok) {
        Some(CoreFinish::Stop) => {
            on_event(
                request_index,
                DecodeEvent::Done {
                    reason: StreamFinishReason::StopToken,
                    generated: sched.generated(seq).len(),
                },
            );
            LaneStep::Done
        }
        other => {
            let step = sched.generated(seq).len() - 1;
            on_event(request_index, DecodeEvent::Token { id: tok, step });
            history.push(tok);
            match other {
                Some(CoreFinish::Length) => {
                    on_event(
                        request_index,
                        DecodeEvent::Done {
                            reason: StreamFinishReason::MaxTokens,
                            generated: sched.generated(seq).len(),
                        },
                    );
                    LaneStep::Done
                }
                _ => {
                    *next_token = tok;
                    LaneStep::Continue
                }
            }
        }
    }
}

#[cfg(test)]
mod buffer_release_tests {
    use super::{BufferRelease, BUFFER_RELEASE_TOKENS};

    // `count` is the whole of `BufferRelease`'s logic; `advance` only forwards its verdict to
    // `clear_cache`. Driving `count` keeps these CPU-only — no device, no MLX allocator.

    #[test]
    fn the_post_prefill_release_lands_on_the_first_advance_not_on_construction() {
        let mut release = BufferRelease::new();
        // Construction takes no release: the prefill logits are still live at that point in every
        // loop. The first advance is the earliest moment step 0 has retired them.
        assert!(
            release.count(1),
            "first advance must take the post-prefill release"
        );
        // ...and it is taken exactly once, not again on the next token.
        assert!(
            !release.count(1),
            "post-prefill release must not repeat on the second token"
        );
    }

    #[test]
    fn periodic_releases_continue_on_every_block_boundary() {
        let mut release = BufferRelease::new();
        assert!(release.count(1), "token 1: post-prefill release");
        // Tokens 2..BUFFER_RELEASE_TOKENS sit inside the first block: no release.
        for token in 2..BUFFER_RELEASE_TOKENS {
            assert!(!release.count(1), "token {token} must not release");
        }
        // The first block boundary, and every one after it, still releases.
        assert!(
            release.count(1),
            "token {BUFFER_RELEASE_TOKENS}: first periodic release"
        );
        for token in BUFFER_RELEASE_TOKENS + 1..BUFFER_RELEASE_TOKENS * 2 {
            assert!(!release.count(1), "token {token} must not release");
        }
        assert!(
            release.count(1),
            "token {}: second periodic release",
            BUFFER_RELEASE_TOKENS * 2
        );
    }

    #[test]
    fn a_multi_token_advance_releases_once_for_the_boundary_it_crosses() {
        // The speculative and MTP loops commit several tokens per advance.
        let mut release = BufferRelease::new();
        assert!(
            release.count(4),
            "post-prefill release on the first advance"
        );
        // 4 + (BUFFER_RELEASE_TOKENS - 8) tokens still sits below the first boundary.
        assert!(
            !release.count(BUFFER_RELEASE_TOKENS - 8),
            "a commit that stays inside the block must not release"
        );
        // This commit steps over the boundary: exactly one release, not one per token.
        assert!(
            release.count(8),
            "a commit that crosses the block boundary releases"
        );
        assert!(
            !release.count(1),
            "the token after the crossing must not release again"
        );
    }
}
