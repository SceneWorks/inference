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

/// Releases MLX's freed-buffer cache once after prefill and once every
/// [`BUFFER_RELEASE_TOKENS`] generated tokens. One instance per generation, shared by every decode
/// loop; it only ever returns *unused* buffers to the OS and never touches live arrays, so it is
/// safe at any point between forwards.
pub(crate) struct BufferRelease {
    tokens: usize,
}

impl BufferRelease {
    /// Release the buffers the prefill left behind (the prompt-length activations are the largest
    /// transient of the generation) and start counting decode tokens.
    ///
    /// Call this only once the prefill graph has actually been **evaluated** — after the first
    /// token has been sampled from the prefill logits, in practice. Every loop's prefill hands
    /// back *lazy* logits; MLX allocates and frees the prompt-length activations while evaluating
    /// them, so a release before that point clears an empty cache and the prefill transients are
    /// then held until the next periodic release, [`BUFFER_RELEASE_TOKENS`] tokens later.
    pub(crate) fn after_prefill() -> Self {
        mlx_rs::memory::clear_cache();
        Self { tokens: 0 }
    }

    /// Account for `n` more generated tokens, releasing the cache when the count crosses a
    /// [`BUFFER_RELEASE_TOKENS`] boundary.
    pub(crate) fn advance(&mut self, n: usize) {
        let before = self.tokens / BUFFER_RELEASE_TOKENS;
        self.tokens += n;
        if self.tokens / BUFFER_RELEASE_TOKENS > before {
            mlx_rs::memory::clear_cache();
        }
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
