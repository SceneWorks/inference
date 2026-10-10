//! The OpenAI chat-completions wire format: request parsing → the backend-neutral
//! [`mlx_llm::core_llm::TextLlmRequest`], and response/SSE-chunk construction from the contract's
//! events.
//!
//! These are pure data transforms (no model, no I/O), so they're unit-tested directly. The server
//! ([`crate::main`]) wires them to a TCP socket + a loaded `core_llm::TextLlm` provider.

use mlx_llm::core_llm::{
    Constraint, Content, DecodeReport, KvCacheReport, KvCompressionPolicy, Message, Role, Sampling,
    Speculative, TextLlmRequest,
};
use serde::Deserialize;
use serde_json::{json, Value};

/// Default `max_tokens` when a request omits it (OpenAI has no implicit cap; we pick a sane one).
const DEFAULT_MAX_TOKENS: u32 = 512;

/// An OpenAI `POST /v1/chat/completions` request body (the subset this example serves, plus the
/// common `top_k`/`repetition_penalty` extensions other on-device servers accept).
#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    /// Requested model id (informational — this server hosts a single loaded model; echoed back).
    #[serde(default)]
    pub model: Option<String>,
    /// The conversation.
    pub messages: Vec<ChatMessage>,
    /// Stream the response as Server-Sent Events.
    #[serde(default)]
    pub stream: bool,
    /// Max new tokens to generate.
    pub max_tokens: Option<u32>,
    /// Sampling temperature (`0` ⇒ greedy).
    pub temperature: Option<f32>,
    /// Nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Top-k cutoff (non-OpenAI extension).
    pub top_k: Option<usize>,
    /// Repetition penalty (non-OpenAI extension).
    pub repetition_penalty: Option<f32>,
    /// RNG seed for reproducible sampling.
    pub seed: Option<u64>,
    /// Stop strings (string or array).
    pub stop: Option<StringOrVec>,
    /// `{"type":"json_object"}` ⇒ constrain output to valid JSON.
    pub response_format: Option<ResponseFormat>,
    /// Speculative decoding (non-OpenAI extension, epic sc-24432 E4): `"off"`, `"auto"` or
    /// `{"proposer": "mtp" | "prompt_lookup" | "draft_model", "depth": N}`, mapped onto the
    /// contract's `speculative` option. The legacy `mtp` field (`{"mode": "off" | "auto"}`,
    /// `{"mode": "enabled", "draft_tokens": N}`) is read into the same option; sending both is a
    /// duplicate field, and an unknown value in either is refused by the contract's parser — a
    /// 400, never silently decoded as `off`. Omitted ⇒ `off`.
    #[serde(default, alias = "mtp")]
    pub speculative: Option<Speculative>,
    /// Compressed-KV opt-in (non-OpenAI extension, sc-20681): `"off"` (the default) or
    /// `"qualified"` — compressed where the engine's qualification table admits the request, dense
    /// with a reason everywhere else. The response's `kv_cache` object says which ran.
    #[serde(default)]
    pub kv_compression: Option<String>,
}

/// One chat turn. `content` is a string or an array of typed parts (the vision wire form); this
/// text-only server uses the text parts.
#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContent,
}

/// OpenAI message content: a plain string, or an array of `{type, text}` parts.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One content part (only the `text` kind is consumed by this text-only server).
#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: Option<String>,
}

impl MessageContent {
    /// Flatten to plain text (concatenating text parts; non-text parts are dropped — the provider's
    /// capabilities reject vision input up front anyway).
    fn into_text(self) -> String {
        match self {
            MessageContent::Text(s) => s,
            MessageContent::Parts(parts) => parts
                .into_iter()
                .filter(|p| p.kind == "text")
                .filter_map(|p| p.text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// A JSON value that may be a single string or a list of strings (e.g. OpenAI `stop`).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    One(String),
    Many(Vec<String>),
}

impl StringOrVec {
    fn into_vec(self) -> Vec<String> {
        match self {
            StringOrVec::One(s) => vec![s],
            StringOrVec::Many(v) => v,
        }
    }
}

/// `response_format` — only `{"type":"json_object"}` is acted on (JSON-constrained decode).
#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
}

/// Map an OpenAI role string to the contract [`Role`] (unknown ⇒ treated as a user turn).
fn role_of(s: &str) -> Role {
    match s {
        "system" | "developer" => Role::System,
        "assistant" => Role::Assistant,
        "tool" => Role::Tool,
        _ => Role::User,
    }
}

impl ChatRequest {
    /// Build the backend-neutral [`TextLlmRequest`]. Returns an error string for an empty
    /// conversation. Sampling starts from the engine's chat defaults; provided fields override.
    pub fn into_text_llm_request(self) -> Result<TextLlmRequest, String> {
        if self.messages.is_empty() {
            return Err("`messages` must not be empty".into());
        }
        let messages = self
            .messages
            .into_iter()
            .map(|m| Message {
                role: role_of(&m.role),
                content: vec![Content::Text(m.content.into_text())],
                // This OpenAI shim does not yet accept prior-turn reasoning or assistant tool calls
                // on input; default both to the contract's "absent" values (no behavior change).
                thinking: None,
                tool_calls: Vec::new(),
            })
            .collect();

        let mut sampling = Sampling::default();
        if let Some(t) = self.temperature {
            sampling.temperature = t;
        }
        if let Some(p) = self.top_p {
            sampling.top_p = p;
        }
        if let Some(k) = self.top_k {
            sampling.top_k = k;
        }
        if let Some(rp) = self.repetition_penalty {
            sampling.repetition_penalty = rp;
        }

        let constraint = self
            .response_format
            .as_ref()
            .filter(|rf| rf.kind == "json_object")
            .map(|_| Constraint::Json);
        let kv_compression = match self.kv_compression.as_deref() {
            None | Some("off") => KvCompressionPolicy::Off,
            Some("qualified") => KvCompressionPolicy::Qualified,
            Some(other) => {
                return Err(format!(
                    "`kv_compression` must be \"off\" or \"qualified\", not {other:?}"
                ))
            }
        };

        Ok(TextLlmRequest {
            messages,
            sampling,
            max_new_tokens: self.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            seed: self.seed,
            constraint,
            // This shim does not yet surface request-level thinking/tools controls; leave thinking
            // at the template default (Auto) and offer no tools (no behavior change).
            thinking: Default::default(),
            reasoning_effort: None,
            preserve_thinking: None,
            // The request's speculative option (or its legacy `mtp` spelling) as sent; the
            // provider validates it against what the loaded model advertises (sc-24438).
            speculative: self.speculative,
            mtp: None,
            tools: Vec::new(),
            stop: self.stop.map(StringOrVec::into_vec).unwrap_or_default(),
            cancel: Default::default(),
            kv_compression,
        })
    }
}

/// Map a contract finish reason to the OpenAI `finish_reason` string.
pub fn finish_reason_str(f: mlx_llm::core_llm::FinishReason) -> &'static str {
    use mlx_llm::core_llm::FinishReason::*;
    match f {
        Stop | Cancelled => "stop",
        Length => "length",
        ContentFilter => "content_filter",
    }
}

/// The first SSE chunk: an empty assistant-role delta (matches OpenAI clients' expectations).
pub fn role_chunk(id: &str, model: &str, created: u64) -> String {
    chunk(
        id,
        model,
        created,
        json!({ "role": "assistant" }),
        Value::Null,
    )
}

/// A content SSE chunk carrying the next text delta.
pub fn content_chunk(id: &str, model: &str, created: u64, delta: &str) -> String {
    chunk(id, model, created, json!({ "content": delta }), Value::Null)
}

/// The terminal SSE chunk: an empty delta plus the finish reason, the request's decode report as
/// [`x_decode`] (sc-24438), and the generation's `kv_cache` report when the engine produced one
/// (sc-20681).
pub fn final_chunk(
    id: &str,
    model: &str,
    created: u64,
    finish: &str,
    decode: Option<&DecodeReport>,
    kv_cache: Option<&KvCacheReport>,
) -> String {
    let mut v = chunk_value(id, model, created, json!({}), json!(finish));
    if let Some(report) = decode {
        v["x_decode"] = x_decode(report);
    }
    if let Some(report) = kv_cache {
        v["kv_cache"] = kv_cache_json(report);
    }
    v.to_string()
}

/// The decode report an OpenAI client can see (sc-24438, epic sc-24432 E2/E3) — the whole
/// [`DecodeReport`], field for field: the path and the proposer that ran, its depth and the
/// realized mean accepted length, the sampler path, the KV cache and attention, the graph path
/// and CUDA-graph runner, the NVFP4 and fused-primitive paths, every forward count, the prefix
/// cache's part, and every fallback — a clamped depth, an unavailable proposer — by name, so no
/// downgrade is silent over HTTP. Carried as the `x_decode` extension member of the non-streaming
/// body and of the final SSE chunk before `[DONE]`.
pub fn x_decode(report: &DecodeReport) -> Value {
    let path = |p: &mlx_llm::core_llm::PathReport| json!({ "path": p.path, "reason": p.reason });
    json!({
        "path": report.path,
        "proposer": report.proposer.label(),
        "draft_tokens": report.draft_tokens,
        "mean_accepted_length": report.mean_accepted_length(),
        "sampler": report.sampler,
        "kv_cache": report.kv_cache,
        "attention": report.attention,
        "graph_path": report.graph_path,
        "cuda_graphs": {
            "enabled": report.cuda_graphs.enabled,
            "path": report.cuda_graphs.path,
            "replayed": report.cuda_graphs.replayed,
            "eager": report.cuda_graphs.eager,
            "captured": report.cuda_graphs.captured,
            "fallback_reason": report.cuda_graphs.fallback_reason,
        },
        "nvfp4_projections": path(&report.nvfp4_projections),
        "fused_primitives": path(&report.fused_primitives),
        "target_forwards": report.target_forwards,
        "prefill_forwards": report.prefill_forwards,
        "proposed_tokens": report.proposed_tokens,
        "accepted_tokens": report.accepted_tokens,
        "verify_steps": report.verify_steps,
        "replay_forwards": report.replay_forwards,
        "discarded_forwards": report.discarded_forwards,
        "speculative_demoted_at": report.speculative_demoted_at,
        "speculative_monitor": report.speculative_monitor.map(monitor),
        "prefix_cache": path(&report.prefix_cache),
        "prefix_hit_tokens": report.prefix_hit_tokens,
        "fallbacks": report.fallbacks,
    })
}

/// `auto`'s last judged window (sc-24446,
/// [`MonitorDecision`](mlx_llm::core_llm::MonitorDecision)): its inputs and the measured verify
/// cost and gain derived from them (`null` where not measured).
fn monitor(d: mlx_llm::core_llm::MonitorDecision) -> Value {
    json!({
        "window": d.window,
        "basis": d.basis.label(),
        "demoted": d.demoted,
        "verifies": d.verifies,
        "accepted": d.accepted,
        "timed_steps": d.timed_steps,
        "timed_tokens": d.timed_tokens,
        "timed_ns": d.timed_ns,
        "plain_step_ns": d.plain_step_ns,
        "verify_cost_ratio": d.verify_cost_ratio(),
        "gain": d.gain(),
    })
}

/// The `kv_cache` extension object (sc-20681): the KV cache one generation ran on — its compressed
/// format, or the reason it ran dense — and the engine's measured counters.
pub fn kv_cache_json(report: &KvCacheReport) -> Value {
    let counters = &report.counters;
    json!({
        "format_version": report.format_version,
        "format": report.format.map(|format| format.id()),
        "fallback": report.fallback.map(|reason| reason.id()),
        "detail": report.detail,
        "counters": {
            "fused_attention_calls": counters.fused_attention_calls,
            "dense_fallback_events": counters.dense_fallback_events,
            "full_cache_dequantizations": counters.full_cache_dequantizations,
            "dense_gather_fallbacks": counters.dense_gather_fallbacks,
            "compressed_cache_bytes": counters.compressed_cache_bytes,
            "pool_held_bytes": counters.pool_held_bytes,
        },
    })
}

fn chunk(id: &str, model: &str, created: u64, delta: Value, finish_reason: Value) -> String {
    chunk_value(id, model, created, delta, finish_reason).to_string()
}

fn chunk_value(id: &str, model: &str, created: u64, delta: Value, finish_reason: Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    })
}

/// The token counts a completion reports.
pub struct CompletionUsage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Generated tokens.
    pub completion_tokens: u32,
}

/// A non-streaming `chat.completion` response body, with the request's decode report as
/// [`x_decode`] (sc-24438) and the generation's `kv_cache` report when the engine produced one
/// (sc-20681).
#[allow(clippy::too_many_arguments)]
pub fn completion(
    id: &str,
    model: &str,
    created: u64,
    text: &str,
    finish: &str,
    usage: CompletionUsage,
    decode: Option<&DecodeReport>,
    kv_cache: Option<&KvCacheReport>,
) -> String {
    let CompletionUsage {
        prompt_tokens,
        completion_tokens,
    } = usage;
    let mut v = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    });
    if let Some(report) = decode {
        v["x_decode"] = x_decode(report);
    }
    if let Some(report) = kv_cache {
        v["kv_cache"] = kv_cache_json(report);
    }
    v.to_string()
}

/// The `GET /v1/models` body listing the single hosted model.
pub fn models_list(model: &str, created: u64) -> String {
    json!({
        "object": "list",
        "data": [{ "id": model, "object": "model", "created": created, "owned_by": "mlx-llm" }],
    })
    .to_string()
}

/// An OpenAI-style error body.
pub fn error_body(message: &str, kind: &str) -> String {
    json!({ "error": { "message": message, "type": kind } }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> ChatRequest {
        serde_json::from_str(body).unwrap()
    }

    #[test]
    fn maps_messages_sampling_and_max_tokens() {
        let req = parse(
            r#"{"model":"m","messages":[
                {"role":"system","content":"be brief"},
                {"role":"user","content":"hi"}
            ],"temperature":0.0,"max_tokens":32,"seed":7,"stream":true}"#,
        );
        assert!(req.stream);
        assert_eq!(req.model.as_deref(), Some("m"));
        let r = req.into_text_llm_request().unwrap();
        assert_eq!(r.messages.len(), 2);
        assert_eq!(r.messages[0].role, Role::System);
        assert_eq!(r.messages[1].role, Role::User);
        assert_eq!(r.messages[1].content, vec![Content::Text("hi".into())]);
        assert_eq!(r.sampling.temperature, 0.0); // explicit override (greedy)
        assert_eq!(r.sampling.top_p, Sampling::default().top_p); // untouched -> engine default
        assert_eq!(r.max_new_tokens, 32);
        assert_eq!(r.seed, Some(7));
        assert!(r.constraint.is_none());
    }

    #[test]
    fn defaults_when_fields_omitted() {
        let r = parse(r#"{"messages":[{"role":"user","content":"x"}]}"#)
            .into_text_llm_request()
            .unwrap();
        assert_eq!(r.max_new_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(r.sampling, Sampling::default());
        assert!(r.seed.is_none());
    }

    #[test]
    fn json_object_response_format_sets_constraint() {
        let r = parse(
            r#"{"messages":[{"role":"user","content":"x"}],"response_format":{"type":"json_object"}}"#,
        )
        .into_text_llm_request()
        .unwrap();
        assert_eq!(r.constraint, Some(Constraint::Json));
    }

    #[test]
    fn content_parts_and_stop_string_or_array() {
        let r = parse(
            r#"{"messages":[{"role":"user","content":[
                {"type":"text","text":"a"},{"type":"text","text":"b"}
            ]}],"stop":"END"}"#,
        )
        .into_text_llm_request()
        .unwrap();
        assert_eq!(r.messages[0].content, vec![Content::Text("ab".into())]);
        assert_eq!(r.stop, vec!["END".to_string()]);

        let r2 = parse(r#"{"messages":[{"role":"user","content":"x"}],"stop":["a","b"]}"#)
            .into_text_llm_request()
            .unwrap();
        assert_eq!(r2.stop, vec!["a".to_string(), "b".to_string()]);
    }

    /// sc-24438 AC3: the new `speculative` option and the legacy `mtp` shape both reach the
    /// contract as sent (never forced to off), and an omitted option stays unset — the provider
    /// applies its per-backend default (`TextLlmRequest::speculative_or`, E5).
    #[test]
    fn speculative_and_the_legacy_mtp_field_map_onto_the_contract() {
        use mlx_llm::core_llm::SpeculativeProposer;
        let spec = |extra: &str| {
            parse(&format!(
                r#"{{"messages":[{{"role":"user","content":"x"}}]{extra}}}"#
            ))
            .into_text_llm_request()
            .unwrap()
        };
        let r = spec("");
        assert_eq!(r.speculative, None);
        assert_eq!(r.requested_speculative(), None);
        assert_eq!(r.speculative_or(Speculative::Auto), Speculative::Auto);
        for (extra, want) in [
            (r#","speculative":"auto""#, Speculative::Auto),
            (r#","speculative":"off""#, Speculative::Off),
            (
                r#","speculative":{"proposer":"prompt_lookup","depth":4}"#,
                Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
            ),
            (
                r#","speculative":{"proposer":"mtp","depth":12}"#,
                Speculative::proposer(SpeculativeProposer::Mtp, 12),
            ),
            (r#","mtp":{"mode":"auto"}"#, Speculative::Auto),
            (r#","mtp":{"mode":"off"}"#, Speculative::Off),
            (
                r#","mtp":{"mode":"enabled","draft_tokens":3}"#,
                Speculative::proposer(SpeculativeProposer::Mtp, 3),
            ),
        ] {
            let r = spec(extra);
            assert_eq!(r.speculative, Some(want), "{extra}");
            assert_eq!(r.speculative_mode(), want, "{extra}");
            assert_eq!(r.mtp, None, "{extra}");
        }
    }

    /// sc-24438 AC3: an unknown speculative value — in either field — or both fields at once is
    /// a parse error (the server's 400), never ignored.
    #[test]
    fn unknown_speculative_values_are_refused_not_ignored() {
        for extra in [
            r#","speculative":"fast""#,
            r#","speculative":true"#,
            r#","speculative":{"proposer":"ngram","depth":2}"#,
            r#","speculative":{"proposer":"mtp","depth":-1}"#,
            r#","speculative":{"proposer":"mtp"}"#,
            r#","speculative":{"proposer":"mtp","depth":2,"width":3}"#,
            r#","mtp":{"mode":"always"}"#,
            r#","mtp":{"mode":"enabled"}"#,
            r#","mtp":"sometimes""#,
            r#","speculative":"auto","mtp":{"mode":"auto"}"#,
        ] {
            let body = format!(r#"{{"messages":[{{"role":"user","content":"x"}}]{extra}}}"#);
            let err = serde_json::from_str::<ChatRequest>(&body)
                .expect_err(&format!("{extra} must be refused"));
            assert!(!err.to_string().is_empty(), "{extra}");
        }
    }

    #[test]
    fn empty_messages_rejected() {
        assert!(parse(r#"{"messages":[]}"#).into_text_llm_request().is_err());
    }

    #[test]
    fn sse_chunks_have_openai_shape() {
        let role = serde_json::from_str::<Value>(&role_chunk("id1", "m", 100)).unwrap();
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(role["choices"][0]["finish_reason"], Value::Null);

        let content =
            serde_json::from_str::<Value>(&content_chunk("id1", "m", 100, "hello")).unwrap();
        assert_eq!(content["choices"][0]["delta"]["content"], "hello");

        let fin =
            serde_json::from_str::<Value>(&final_chunk("id1", "m", 100, "length", None, None))
                .unwrap();
        assert_eq!(fin["choices"][0]["finish_reason"], "length");
        assert_eq!(fin["choices"][0]["delta"], json!({}));
        assert!(fin.get("x_decode").is_none(), "no report, no member");
        assert!(fin.get("kv_cache").is_none());
    }

    #[test]
    fn kv_compression_opt_in_reaches_the_request_and_the_report_the_response() {
        use mlx_llm::core_llm::{KvCacheFallbackReason, KvCompressionFormat};
        let off = parse(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(
            off.into_text_llm_request().unwrap().kv_compression,
            KvCompressionPolicy::Off
        );
        let on =
            parse(r#"{"messages":[{"role":"user","content":"hi"}],"kv_compression":"qualified"}"#);
        assert_eq!(
            on.into_text_llm_request().unwrap().kv_compression,
            KvCompressionPolicy::Qualified
        );
        let bad =
            parse(r#"{"messages":[{"role":"user","content":"hi"}],"kv_compression":"always"}"#);
        assert!(bad.into_text_llm_request().is_err());

        let dense = KvCacheReport::dense(KvCacheFallbackReason::BelowMinimumContext, None);
        let v = serde_json::from_str::<Value>(&completion(
            "id",
            "m",
            1,
            "x",
            "stop",
            CompletionUsage {
                prompt_tokens: 1,
                completion_tokens: 1,
            },
            None,
            Some(&dense),
        ))
        .unwrap();
        assert_eq!(v["kv_cache"]["fallback"], "below_minimum_context");
        assert_eq!(v["kv_cache"]["format"], Value::Null);
        let compressed = KvCacheReport {
            format: Some(KvCompressionFormat::GroupAffineK8V8),
            fallback: None,
            ..dense
        };
        let fin = serde_json::from_str::<Value>(&final_chunk(
            "id",
            "m",
            1,
            "stop",
            None,
            Some(&compressed),
        ))
        .unwrap();
        assert_eq!(fin["kv_cache"]["format"], "group-affine-k8v8");
        assert_eq!(fin["kv_cache"]["fallback"], Value::Null);
    }

    #[test]
    fn completion_body_carries_usage_and_message() {
        let usage = CompletionUsage {
            prompt_tokens: 3,
            completion_tokens: 2,
        };
        let v = serde_json::from_str::<Value>(&completion(
            "id", "m", 1, "hi there", "stop", usage, None, None,
        ))
        .unwrap();
        assert!(v.get("kv_cache").is_none());
        assert!(v.get("x_decode").is_none());
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi there");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 3);
        assert_eq!(v["usage"]["completion_tokens"], 2);
        assert_eq!(v["usage"]["total_tokens"], 5);
    }
}
