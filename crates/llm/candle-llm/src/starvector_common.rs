//! The SVG text-contract scaffolding the StarVector-1B ([`crate::starvector`]) and StarVector-8B
//! ([`crate::starvector_8b`]) providers share (sc-27301): the `TextLlm::generate` answer built
//! from a bounded SVG decode, its measured report, the terminal `Done` stream event and the
//! loaded context-cap check. Model-specific loading, preprocessing and decoding stay per file.

use std::time::Duration;

use core_llm::{
    Channel, DecodeReport, Error as CoreError, FinishReason, KvCacheReport, Result as CoreResult,
    Speculative, StarVectorDescriptor, StarVectorFinishReason, StarVectorOutput, StarVectorRequest,
    StarVectorStreamEvent, StreamEvent, TextLlmDescriptor, TextLlmOutput, TextLlmRequest, Usage,
};

use crate::decode::DecodeRecord;

/// The text contract's finish reason for a bounded SVG stream's terminal reason. A stream cut by
/// any budget (tokens, bytes or wall time) is a truncated answer, so it is `Length`.
pub(crate) fn svg_finish_reason(reason: StarVectorFinishReason) -> FinishReason {
    match reason {
        StarVectorFinishReason::CompleteRoot | StarVectorFinishReason::Eos => FinishReason::Stop,
        StarVectorFinishReason::TokenLimit
        | StarVectorFinishReason::ByteLimit
        | StarVectorFinishReason::WallTimeLimit => FinishReason::Length,
        StarVectorFinishReason::Cancelled => FinishReason::Cancelled,
    }
}

/// Emit the bounded stream's terminal `Done` event for `output` and hand the output back.
pub(crate) fn emit_done(
    output: StarVectorOutput,
    events: &mut dyn FnMut(StarVectorStreamEvent),
) -> StarVectorOutput {
    events(StarVectorStreamEvent::Done {
        finish_reason: output.finish_reason,
        generated_tokens: output.generated_tokens,
        generated_bytes: output.generated_bytes,
    });
    output
}

/// Refuse a loaded snapshot whose advertised generation cap does not fit its context once the
/// image prefix and the tokenized SVG prompt are prefilled. `model` names the tier in the error.
pub(crate) fn validate_loaded_context_cap(
    model: &str,
    descriptor: &TextLlmDescriptor,
    starvector: &StarVectorDescriptor,
    prompt_tokens: usize,
) -> CoreResult<()> {
    let prefill_tokens = usize::try_from(starvector.projection.image_token_count)
        .map_err(|_| CoreError::InvalidRequest(format!("{model} image prefix does not fit usize")))?
        .checked_add(prompt_tokens)
        .ok_or_else(|| {
            CoreError::InvalidRequest(format!("{model} prefill token count overflow"))
        })?;
    core_llm::validate_advertised_generated_token_cap(
        descriptor.capabilities.max_new_tokens,
        descriptor.capabilities.max_context_tokens,
        prefill_tokens,
    )
}

/// The text contract's answer for one SVG generation: `decode` runs the bounded SVG request built
/// from `request` (capped at `max_svg_bytes` and `max_wall_time`), its stream events re-emitted
/// on `on_event` as content tokens and `Done`, its output the text, usage and finish reason — and
/// its measured record (sc-24139) the report, carrying the captioner reasons with the request's
/// option resolved against the provider's `speculative_default` (E2, E5, E8). `decode` must end
/// every successful stream with [`emit_done`], which is the only source of the text `Done`.
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
    let svg_request = StarVectorRequest::new(request.clone(), max_svg_bytes, max_wall_time);
    let (output, record) = decode(&svg_request, &mut |event| match event {
        StarVectorStreamEvent::Source { text, index } => on_event(StreamEvent::Token {
            id: index,
            text,
            index: index as usize,
            channel: Channel::Content,
        }),
        StarVectorStreamEvent::Progress { .. } => {}
        StarVectorStreamEvent::Done {
            finish_reason,
            generated_tokens,
            ..
        } => on_event(StreamEvent::Done {
            finish_reason: svg_finish_reason(finish_reason),
            usage: Usage {
                prompt_tokens,
                generated_tokens,
            },
        }),
    })?;
    Ok(TextLlmOutput {
        timings: None,
        text: output.svg.unwrap_or_default(),
        thinking: None,
        tool_calls: Vec::new(),
        usage: Usage {
            prompt_tokens,
            generated_tokens: output.generated_tokens,
        },
        mtp: None,
        // The continuation decodes through the shared engine (sc-24138), so it reports its
        // path. No graph runner wraps this decoder, so the CUDA-graph switch was off here.
        // `None` only when the bounded stream stopped on the static prefix, before a decode.
        // StarVector advertises no proposer and has no prefix cache: both are named, in the
        // words MLX's StarVector uses (E2, E8).
        decode: record.map(|record| svg_report(&record, request, speculative_default)),
        finish_reason: Some(svg_finish_reason(output.finish_reason)),
        // The StarVector wrapper has no compressed-KV table family: the dense KV-cache report for
        // the request's policy (sc-20683).
        kv_cache: Some(KvCacheReport::without_table_family(request.kv_compression)),
    })
}

/// An SVG continuation's measured report (sc-24139): the engine record with the CUDA-graph
/// switch off (no graph runner wraps this decoder), and — StarVector advertising no proposer and
/// having no prefix cache — the request's speculative fallback and the prefix-cache reason named
/// in the words MLX's StarVector uses (E2, E8). The request's option is resolved against the
/// provider's per-backend `default` (E5), so a table default of `auto` is named too.
pub(crate) fn svg_report(
    record: &DecodeRecord,
    request: &TextLlmRequest,
    default: Speculative,
) -> DecodeReport {
    record
        .report(false)
        .with_captioner_reasons(request.speculative_or(default))
}
