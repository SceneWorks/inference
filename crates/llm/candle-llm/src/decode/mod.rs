//! Streaming, cancellable decoding.
//!
//! [`generate`] is the model-agnostic decode loop; [`Decode`] is the seam any model implements to be
//! driven by it. [`StreamEvent`]s are emitted per token through a callback. The Candle port of
//! `mlx-llm`'s `decode` module.

use core_llm::schedule::{Scheduler, SeqId};
use core_llm::FinishReason as CoreFinish;

use self::stream::{FinishReason as StreamFinishReason, StreamEvent as DecodeEvent};

pub mod batch;
pub mod cancel;
pub mod continuous;
pub mod prefix;
pub mod speculative;
pub mod stream;

pub use batch::{generate_batch, BatchRequest};
pub use cancel::CancelFlag;
pub use continuous::{generate_continuous, BatchExactness, ContinuousConfig};
pub use prefix::{generate_cached, PrefixCache, PrefixStats};
pub use speculative::{
    generate_draft_speculative, generate_prompt_lookup, SpeculativeConfig, SpeculativeStats,
};
pub use stream::{
    generate, generate_from_prefill, generate_with, generate_with_cache, ConstraintMask, Decode,
    FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};

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
