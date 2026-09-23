//! Streaming, cancellable decoding.
//!
//! [`generate`] is the model-agnostic decode loop; [`Decode`] is the seam any model implements to be
//! driven by it. [`StreamEvent`]s are emitted per token through a callback. The Candle port of
//! `mlx-llm`'s `decode` module.
//!
//! The Blackwell fast-decode epic (sc-24128) adds a second, narrower seam beside it:
//! [`StepModel`] (one N-token step against a [`DecodeCache`](crate::primitives::DecodeCache)),
//! driven by [`generate_step`], and the measured per-request [`DecodeRecord`] every path reports.
//! The `Decode` loop stays as the parity oracle; the record's [`DecodePath`] says which one ran.
//! Speculation over that seam is the one [`engine`] loop ([`generate_speculative`]) with a
//! [`Proposer`] — MTP, n-gram or a draft model ([`proposers`]) — behind it (story sc-24130).

use core_llm::schedule::{Scheduler, SeqId};
use core_llm::FinishReason as CoreFinish;

use self::stream::{FinishReason as StreamFinishReason, StreamEvent as DecodeEvent};

pub mod batch;
pub mod cancel;
pub mod continuous;
pub mod engine;
pub mod prefix;
pub mod proposers;
pub mod record;
pub mod speculative;
pub mod step;
pub mod stream;

pub use batch::{generate_batch, BatchRequest};
pub use cancel::CancelFlag;
pub use continuous::{generate_continuous, BatchExactness, ContinuousConfig};
pub use engine::{
    generate_speculative, generate_speculative_with, DraftSample, DraftSampler, Drafts, NoProposer,
    Proposal, ProposeContext, Proposer, RewindableConstraintMask, SpeculativePrompt,
    SpeculativeRun,
};
pub use prefix::{generate_cached, PrefixCache, PrefixStats};
pub use proposers::{DraftModelProposer, MtpProposer, NgramProposer};
pub use record::{
    CountingDecode, DecodePath, DecodeRecord, RequestSpan, SamplerTelemetry, SpanCounters,
};
pub use speculative::{
    generate_draft_speculative, generate_prompt_lookup, SpeculativeConfig, SpeculativeStats,
};
pub use step::{
    generate_step, generate_step_timed, LogitsScope, StepModel, StepOutput, StepRequest, StepTokens,
};
pub use stream::{
    generate, generate_from_prefill, generate_from_prefill_with_stop, generate_with,
    generate_with_cache, ConstraintMask, Decode, FinishReason, GenerationConfig, GenerationOutput,
    StreamEvent,
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
