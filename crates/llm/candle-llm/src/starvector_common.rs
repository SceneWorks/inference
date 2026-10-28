//! The Candle side of the StarVector SVG text scaffolding (sc-27301). The scaffolding itself — the
//! `TextLlm::generate` answer built from a bounded SVG decode, the terminal `Done` event, the
//! conditioning-image and context-cap checks — is tensor-free and lives once in
//! [`core_llm::starvector`], shared with MLX; this module only turns the Candle engine's measured
//! [`DecodeRecord`] into the contract report it consumes.

use std::time::Duration;

use core_llm::{
    Result as CoreResult, Speculative, StarVectorOutput, StarVectorRequest, StarVectorStreamEvent,
    StreamEvent, TextLlmOutput, TextLlmRequest,
};

use crate::decode::DecodeRecord;

/// [`core_llm::svg_text_output`] over a Candle decode: `decode`'s measured record (sc-24139)
/// becomes the report with the CUDA-graph switch off — no graph runner wraps a StarVector decoder.
pub(crate) fn svg_text_output(
    request: &TextLlmRequest,
    prompt_tokens: u32,
    max_svg_bytes: usize,
    max_wall_time: Duration,
    speculative_default: Speculative,
    on_event: &mut dyn FnMut(StreamEvent),
    decode: impl FnOnce(
        &StarVectorRequest,
        &mut dyn FnMut(StarVectorStreamEvent),
    ) -> CoreResult<(StarVectorOutput, Option<DecodeRecord>)>,
) -> CoreResult<TextLlmOutput> {
    core_llm::svg_text_output(
        request,
        prompt_tokens,
        max_svg_bytes,
        max_wall_time,
        speculative_default,
        on_event,
        |svg_request, events| {
            decode(svg_request, events)
                .map(|(output, record)| (output, record.map(|record| record.report(false))))
        },
    )
}

/// An SVG continuation's measured report as [`svg_text_output`] publishes it: the engine record
/// with the CUDA-graph switch off and the captioner reasons for the request's option resolved
/// against the provider's `default` (E2, E5, E8).
#[cfg(test)]
pub(crate) fn svg_report(
    record: &DecodeRecord,
    request: &TextLlmRequest,
    default: Speculative,
) -> core_llm::DecodeReport {
    record
        .report(false)
        .with_captioner_reasons(request.speculative_or(default))
}
