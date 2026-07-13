//! Conformance suite for the [`core_llm::TextLlm`] contract.
//!
//! A generic harness that drives any provider through the contract's guarantees: capability-honest
//! validation, token streaming (deltas reconstruct the output), mid-stream + pre-inference
//! cancellation, sampling/seed determinism, and multimodal (text + vision) handling. It is the LLM
//! analog of gen-core-testkit and, like it, is **tensor-free** — it depends only on `core-llm` and
//! drives providers purely through the public trait, so it runs on any platform.
//!
//! # Usage
//! A backend's tests dev-depend on this crate and hand the entrypoint a closure that loads a
//! registered provider:
//!
//! ```ignore
//! core_llm_testkit::textllm_conformance(
//!     || mlx_llm::LlamaProvider::load(&spec).map(|p| Box::new(p) as _).unwrap(),
//!     &core_llm_testkit::TextLlmProfile::cheap(),
//! );
//! ```
//!
//! Each `check_*` function is public and returns `Result<(), String>` so a provider can target one
//! guarantee at a time; [`textllm_conformance`] runs them all, aggregates failures, and panics once
//! with a combined message.

use core_llm::{
    load_for_model, prepare_snapshot, Channel, Content, Error, FinishReason, ImageRef, LoadSpec,
    Message, PrepareSpec, Quantize, Role, Sampling, StreamEvent, TextLlm, TextLlmRequest,
    ThinkingMode, ToolSpec, VideoRef, VERSION,
};
use std::path::PathBuf;

/// Configures the conformance run: the prompt, token budget, the sampling used for the determinism
/// check, and an optional image for the multimodal check.
#[derive(Clone, Debug)]
pub struct TextLlmProfile {
    /// The user prompt fed to the provider.
    pub prompt: String,
    /// Token budget for each generation (kept small so the run is cheap).
    pub max_new_tokens: u32,
    /// Seed for the determinism check.
    pub determinism_seed: u64,
    /// Sampling (must be non-greedy, i.e. `temperature > 0`) for the determinism check.
    pub determinism_sampling: Sampling,
    /// A synthetic image for the multimodal check (`Some` exercises vision support or its honest
    /// rejection; `None` skips the multimodal check).
    pub image: Option<ImageRef>,
    /// A synthetic video for the video check (`Some` exercises video support or its honest rejection;
    /// `None` skips the video check).
    pub video: Option<VideoRef>,
    /// Tools offered in the tool-calling check (exercises tool support or its honest rejection).
    /// Non-empty so a tools-capable provider renders + runs them and a non-tools provider rejects
    /// them as `Unsupported`.
    pub tools: Vec<ToolSpec>,
}

impl TextLlmProfile {
    /// A cheap default profile: short prompt, 16-token budget, a small synthetic image.
    pub fn cheap() -> Self {
        Self {
            prompt: "Hello".to_string(),
            max_new_tokens: 16,
            determinism_seed: 7,
            determinism_sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                repetition_context: 0,
            },
            image: Some(ImageRef::new(8, 8, vec![0u8; 8 * 8 * 3]).expect("8x8 RGB")),
            video: Some(
                VideoRef::new(
                    vec![
                        ImageRef::new(8, 8, vec![0u8; 8 * 8 * 3]).expect("8x8 RGB frame"),
                        ImageRef::new(8, 8, vec![255u8; 8 * 8 * 3]).expect("8x8 RGB frame"),
                    ],
                    vec![0.0, 0.5],
                )
                .expect("2-frame synthetic video"),
            ),
            tools: vec![ToolSpec::new(
                "get_weather",
                "Get the current weather for a city",
                serde_json::json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"]
                }),
            )],
        }
    }

    fn request(&self, sampling: Sampling, seed: Option<u64>) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![Message::user(self.prompt.clone())],
            sampling,
            max_new_tokens: self.max_new_tokens,
            seed,
            ..Default::default()
        }
    }

    fn greedy_request(&self) -> TextLlmRequest {
        self.request(Sampling::greedy(), Some(0))
    }

    fn image_request(&self, img: &ImageRef) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text(self.prompt.clone()), Content::Image(img.clone())],
                thinking: None,
                tool_calls: Vec::new(),
            }],
            sampling: Sampling::greedy(),
            max_new_tokens: self.max_new_tokens,
            seed: Some(0),
            ..Default::default()
        }
    }

    fn video_request(&self, video: &VideoRef) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![Message {
                role: Role::User,
                content: vec![Content::Text(self.prompt.clone()), Content::Video(video.clone())],
                thinking: None,
                tool_calls: Vec::new(),
            }],
            sampling: Sampling::greedy(),
            max_new_tokens: self.max_new_tokens,
            seed: Some(0),
            ..Default::default()
        }
    }

    fn tools_request(&self) -> TextLlmRequest {
        TextLlmRequest {
            messages: vec![Message::user(self.prompt.clone())],
            sampling: Sampling::greedy(),
            max_new_tokens: self.max_new_tokens,
            seed: Some(0),
            tools: self.tools.clone(),
            ..Default::default()
        }
    }
}

/// A conformance check that consults the profile.
type ProfileCheck = fn(&dyn TextLlm, &TextLlmProfile) -> Result<(), String>;

/// Run the full conformance suite, aggregating every failure into a single panic.
pub fn textllm_conformance(make: impl Fn() -> Box<dyn TextLlm>, profile: &TextLlmProfile) {
    let provider = make();
    let p: &dyn TextLlm = provider.as_ref();

    let mut failures: Vec<String> = Vec::new();
    let with_profile: [ProfileCheck; 8] = [
        check_validate,
        check_streaming,
        check_mid_stream_cancel,
        check_seed_determinism,
        check_multimodal,
        check_video,
        check_thinking,
        check_tools,
    ];
    for f in with_profile {
        if let Err(e) = f(p, profile) {
            failures.push(e);
        }
    }
    for r in [check_descriptor(p), check_cancellation(p, profile), check_registry(p)] {
        if let Err(e) = r {
            failures.push(e);
        }
    }

    if !failures.is_empty() {
        panic!(
            "core-llm textllm conformance FAILED for `{}` (core-llm {VERSION}):\n  - {}",
            p.descriptor().id,
            failures.join("\n  - ")
        );
    }
}

/// The descriptor is well-formed (non-empty id/backend).
pub fn check_descriptor(p: &dyn TextLlm) -> Result<(), String> {
    let d = p.descriptor();
    if d.id.trim().is_empty() {
        return Err("check_descriptor: descriptor id is empty".to_string());
    }
    if d.backend.trim().is_empty() {
        return Err(format!("check_descriptor[{}]: backend is empty", d.id));
    }
    Ok(())
}

/// The provider's id is discoverable through the registry (i.e. it was registered + linked).
pub fn check_registry(p: &dyn TextLlm) -> Result<(), String> {
    let id = &p.descriptor().id;
    if core_llm::textllms().any(|r| &(r.descriptor)().id == id) {
        Ok(())
    } else {
        Err(format!(
            "check_registry[{id}]: id not discoverable via core_llm::textllms() — was the provider \
             registered with inventory::submit! and linked into the test binary? (core-llm {VERSION})"
        ))
    }
}

/// `validate` is honest: it accepts the base request and rejects things outside declared caps.
pub fn check_validate(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    let base = profile.greedy_request();
    p.validate(&base)
        .map_err(|e| format!("check_validate[{id}]: base request rejected by validate(): {e}"))?;

    let caps = &p.descriptor().capabilities;
    if caps.max_new_tokens > 0 {
        if let Some(over) = caps.max_new_tokens.checked_add(1) {
            let mut r = base.clone();
            r.max_new_tokens = over;
            if p.validate(&r).is_ok() {
                return Err(format!(
                    "check_validate[{id}]: accepted max_new_tokens {over} above declared cap {}",
                    caps.max_new_tokens
                ));
            }
        }
    }
    if !caps.supports_system_prompt {
        let mut r = base.clone();
        r.messages.insert(0, Message::system("be brief"));
        if p.validate(&r).is_ok() {
            return Err(format!(
                "check_validate[{id}]: accepted a system prompt despite supports_system_prompt=false"
            ));
        }
    }
    Ok(())
}

/// `generate` streams tokens whose deltas reconstruct the output, with a terminal Done event and a
/// consistent usage/finish-reason.
pub fn check_streaming(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    let req = profile.greedy_request();

    let mut indices = Vec::new();
    let mut streamed = String::new();
    let mut streamed_thinking = String::new();
    let mut done: Option<(FinishReason, u32)> = None;
    let out = p
        .generate(&req, &mut |ev| match ev {
            // Content deltas reconstruct output.text; thinking deltas reconstruct output.thinking.
            StreamEvent::Token { text, index, channel, .. } => {
                indices.push(index);
                match channel {
                    Channel::Thinking => streamed_thinking.push_str(&text),
                    Channel::Content => streamed.push_str(&text),
                }
            }
            StreamEvent::Done { finish_reason, usage } => {
                done = Some((finish_reason, usage.generated_tokens));
            }
        })
        .map_err(|e| format!("check_streaming[{id}]: generate() failed: {e}"))?;

    if indices.is_empty() {
        return Err(format!("check_streaming[{id}]: no Token events emitted"));
    }
    for (i, idx) in indices.iter().enumerate() {
        if *idx != i {
            return Err(format!(
                "check_streaming[{id}]: token index {idx} out of order (expected {i})"
            ));
        }
    }
    let (fr, gen) = done.ok_or_else(|| format!("check_streaming[{id}]: no Done event emitted"))?;
    if out.finish_reason != Some(fr) {
        return Err(format!(
            "check_streaming[{id}]: Done finish_reason {fr:?} != output finish_reason {:?}",
            out.finish_reason
        ));
    }
    if out.usage.generated_tokens != gen {
        return Err(format!(
            "check_streaming[{id}]: output.usage.generated_tokens ({}) disagrees with Done usage ({gen})",
            out.usage.generated_tokens
        ));
    }
    // A provider may legitimately emit *fewer* Token events than it generated: a token that only
    // completes a multibyte char yields no delta, a stripped reasoning marker (`<think>`) yields no
    // text, and a request stop-string trims trailing tokens. So the invariant is "no more events
    // than generated tokens" — the deltas reconstructing (text, thinking) below is the real guarantee.
    if indices.len() > gen as usize {
        return Err(format!(
            "check_streaming[{id}]: emitted {} Token events but only {gen} tokens were generated",
            indices.len()
        ));
    }
    if out.usage.prompt_tokens == 0 {
        return Err(format!("check_streaming[{id}]: usage.prompt_tokens is 0"));
    }
    if streamed != out.text {
        return Err(format!(
            "check_streaming[{id}]: streamed Content-channel deltas do not reconstruct output.text"
        ));
    }
    if streamed_thinking != out.thinking.clone().unwrap_or_default() {
        return Err(format!(
            "check_streaming[{id}]: streamed Thinking-channel deltas do not reconstruct output.thinking"
        ));
    }
    Ok(())
}

/// An already-cancelled request returns the typed `Error::Canceled` before any inference.
pub fn check_cancellation(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    let req = profile.greedy_request();
    req.cancel.cancel();
    match p.generate(&req, &mut |_| {}) {
        Ok(_) => Err(format!(
            "check_cancellation[{id}]: returned Ok despite an already-cancelled request; must return \
             Err(Error::Canceled) before running inference"
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "check_cancellation[{id}]: must return the typed Err(Error::Canceled), got {other:?}"
        )),
    }
}

/// A cancel that trips mid-stream stops promptly and yields a partial result marked `Cancelled`
/// (or the typed `Canceled` error).
pub fn check_mid_stream_cancel(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    let mut req = profile.greedy_request();
    req.max_new_tokens = profile.max_new_tokens.max(16); // headroom so a cancel is visible
    let cancel = req.cancel.clone();

    let mut tripped = false;
    let mut after_trip = 0usize;
    let result = p.generate(&req, &mut |ev| {
        if let StreamEvent::Token { .. } = ev {
            if tripped {
                after_trip += 1;
            } else {
                cancel.cancel();
                tripped = true;
            }
        }
    });

    match result {
        Ok(out) => {
            if !tripped {
                return Err(format!(
                    "check_mid_stream_cancel[{id}]: no token emitted; cancel could not be exercised"
                ));
            }
            if out.finish_reason != Some(FinishReason::Cancelled) {
                return Err(format!(
                    "check_mid_stream_cancel[{id}]: mid-stream cancel must yield \
                     FinishReason::Cancelled, got {:?}",
                    out.finish_reason
                ));
            }
            if after_trip > 2 {
                return Err(format!(
                    "check_mid_stream_cancel[{id}]: emitted {after_trip} tokens after the cancel \
                     trip (contract allows at most a couple)"
                ));
            }
            Ok(())
        }
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "check_mid_stream_cancel[{id}]: mid-stream cancel errored non-Canceled: {other:?}"
        )),
    }
}

/// Sampling honors the seed: identical seed ⇒ identical output; a different seed ⇒ different output
/// (the anti-cheat — a provider that ignores the seed fails the second leg).
pub fn check_seed_determinism(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    // Compare the answer *and* the reasoning: a thinking model on a short budget may produce only a
    // `<think>` block (empty answer), so comparing `text` alone would see two empty strings and
    // falsely flag the seed as ignored.
    let key = |o: &core_llm::TextLlmOutput| format!("{}\u{1}{}", o.text, o.thinking.clone().unwrap_or_default());
    let req = profile.request(profile.determinism_sampling, Some(profile.determinism_seed));
    let a = p
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("check_seed_determinism[{id}]: generate() failed: {e}"))?;
    let b = p
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("check_seed_determinism[{id}]: generate() failed: {e}"))?;
    if key(&a) != key(&b) {
        return Err(format!(
            "check_seed_determinism[{id}]: the same seed produced different output across two runs"
        ));
    }
    let mut req2 = req.clone();
    req2.seed = Some(profile.determinism_seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
    let c = p
        .generate(&req2, &mut |_| {})
        .map_err(|e| format!("check_seed_determinism[{id}]: generate() failed: {e}"))?;
    if key(&a) == key(&c) {
        return Err(format!(
            "check_seed_determinism[{id}]: a different seed produced identical output — the provider \
             appears to ignore the seed"
        ));
    }
    Ok(())
}

/// Multimodal handling: a vision-capable provider generates from an image; a text-only provider
/// rejects image input as `Unsupported` (never silently ignores it).
pub fn check_multimodal(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    let Some(img) = &profile.image else {
        return Ok(());
    };
    let req = profile.image_request(img);

    if p.descriptor().capabilities.supports_vision {
        let out = p
            .generate(&req, &mut |_| {})
            .map_err(|e| format!("check_multimodal[{id}]: vision generate() failed: {e}"))?;
        if out.usage.generated_tokens == 0 {
            return Err(format!(
                "check_multimodal[{id}]: vision generation produced no tokens"
            ));
        }
        Ok(())
    } else {
        match p.validate(&req) {
            Err(Error::Unsupported(_)) => Ok(()),
            Err(other) => Err(format!(
                "check_multimodal[{id}]: image input rejected, but not as Error::Unsupported: {other:?}"
            )),
            Ok(()) => Err(format!(
                "check_multimodal[{id}]: accepted image input despite supports_vision=false"
            )),
        }
    }
}

/// Video handling: a video-capable provider generates from a sampled video (frames + per-frame
/// timestamps); a provider without video support rejects video input as `Unsupported` (never
/// silently ignores it).
pub fn check_video(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    let Some(video) = &profile.video else {
        return Ok(());
    };
    let req = profile.video_request(video);

    if p.descriptor().capabilities.supports_video {
        let out = p
            .generate(&req, &mut |_| {})
            .map_err(|e| format!("check_video[{id}]: video generate() failed: {e}"))?;
        if out.usage.generated_tokens == 0 {
            return Err(format!("check_video[{id}]: video generation produced no tokens"));
        }
        Ok(())
    } else {
        match p.validate(&req) {
            Err(Error::Unsupported(_)) => Ok(()),
            Err(other) => Err(format!(
                "check_video[{id}]: video input rejected, but not as Error::Unsupported: {other:?}"
            )),
            Ok(()) => Err(format!(
                "check_video[{id}]: accepted video input despite supports_video=false"
            )),
        }
    }
}

/// Thinking ("reasoning") handling is capability-honest. A non-thinking provider rejects an
/// explicit enable as `Unsupported` (and accepts the no-op Auto / no-think modes). A thinking
/// provider validates every mode, keeps the streamed reasoning channel in sync with
/// `output.thinking`, and produces **no** reasoning for a no-think (Disabled) request.
pub fn check_thinking(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    let supports = p.descriptor().capabilities.supports_thinking;
    let with_mode = |mode: ThinkingMode| {
        let mut r = profile.greedy_request();
        r.thinking = mode;
        r
    };

    if !supports {
        // An explicit enable is unsatisfiable → reject as Unsupported (never silently ignore).
        match p.validate(&with_mode(ThinkingMode::Enabled)) {
            Err(Error::Unsupported(_)) => {}
            Err(other) => {
                return Err(format!(
                    "check_thinking[{id}]: enable-thinking rejected, but not as Error::Unsupported: {other:?}"
                ));
            }
            Ok(()) => {
                return Err(format!(
                    "check_thinking[{id}]: accepted an explicit enable-thinking request despite supports_thinking=false"
                ));
            }
        }
        // Auto and no-think are no-ops for a model that never reasons → must be accepted.
        for mode in [ThinkingMode::Auto, ThinkingMode::Disabled] {
            if let Err(e) = p.validate(&with_mode(mode)) {
                return Err(format!(
                    "check_thinking[{id}]: rejected {mode:?} despite it being a no-op for a non-thinking model: {e}"
                ));
            }
        }
        return Ok(());
    }

    // Thinking-capable: every mode validates.
    for mode in [ThinkingMode::Auto, ThinkingMode::Enabled, ThinkingMode::Disabled] {
        p.validate(&with_mode(mode))
            .map_err(|e| format!("check_thinking[{id}]: thinking provider rejected {mode:?}: {e}"))?;
    }

    // Stream/result sync: the streamed channels must reconstruct (text, thinking). Run with Auto —
    // the model decides whether to reason; this asserts consistency, not that it reasons.
    let mut content = String::new();
    let mut thinking = String::new();
    let out = p
        .generate(&with_mode(ThinkingMode::Auto), &mut |ev| {
            if let StreamEvent::Token { text, channel, .. } = ev {
                match channel {
                    Channel::Thinking => thinking.push_str(&text),
                    Channel::Content => content.push_str(&text),
                }
            }
        })
        .map_err(|e| format!("check_thinking[{id}]: generate() failed: {e}"))?;
    if content != out.text {
        return Err(format!(
            "check_thinking[{id}]: streamed Content-channel deltas do not reconstruct output.text"
        ));
    }
    if thinking != out.thinking.clone().unwrap_or_default() {
        return Err(format!(
            "check_thinking[{id}]: streamed Thinking-channel deltas do not reconstruct output.thinking"
        ));
    }

    // No-think: a Disabled request must produce no reasoning at all.
    let mut saw_thinking = false;
    let nout = p
        .generate(&with_mode(ThinkingMode::Disabled), &mut |ev| {
            if let StreamEvent::Token { channel: Channel::Thinking, .. } = ev {
                saw_thinking = true;
            }
        })
        .map_err(|e| format!("check_thinking[{id}]: no-think generate() failed: {e}"))?;
    if saw_thinking {
        return Err(format!(
            "check_thinking[{id}]: no-think (Disabled) request emitted a Thinking-channel token"
        ));
    }
    if nout.thinking.as_deref().is_some_and(|t| !t.is_empty()) {
        return Err(format!(
            "check_thinking[{id}]: no-think (Disabled) request produced reasoning in output.thinking"
        ));
    }
    Ok(())
}

/// Tool ("function") calling is capability-honest. A non-tools provider rejects a request carrying
/// tools as `Unsupported` (never silently drops them). A tools provider validates a tools request,
/// generates without error, keeps its streamed Content channel in sync with `output.text`, excludes
/// the raw `<tool_call>` markup from `output.text`, and returns only well-formed (named) tool calls.
/// It deliberately does **not** assert the model *chooses* to call a tool (model- and
/// prompt-dependent) — the live parse of an actually-emitted call is proven by a backend's gated
/// real-weight test.
pub fn check_tools(p: &dyn TextLlm, profile: &TextLlmProfile) -> Result<(), String> {
    let id = p.descriptor().id.clone();
    if profile.tools.is_empty() {
        return Ok(()); // nothing to exercise
    }
    let supports = p.descriptor().capabilities.supports_tools;
    let req = profile.tools_request();

    if !supports {
        // Offered tools the provider can't honor must be rejected, not ignored.
        return match p.validate(&req) {
            Err(Error::Unsupported(_)) => Ok(()),
            Err(other) => Err(format!(
                "check_tools[{id}]: tools request rejected, but not as Error::Unsupported: {other:?}"
            )),
            Ok(()) => Err(format!(
                "check_tools[{id}]: accepted a tools request despite supports_tools=false"
            )),
        };
    }

    // Tools-capable: a tools request validates and generates.
    p.validate(&req)
        .map_err(|e| format!("check_tools[{id}]: tools provider rejected a tools request: {e}"))?;

    let mut content = String::new();
    let out = p
        .generate(&req, &mut |ev| {
            if let StreamEvent::Token { text, channel: Channel::Content, .. } = ev {
                content.push_str(&text);
            }
        })
        .map_err(|e| format!("check_tools[{id}]: tools generate() failed: {e}"))?;

    if content != out.text {
        return Err(format!(
            "check_tools[{id}]: streamed Content-channel deltas do not reconstruct output.text with tools active"
        ));
    }
    if out.text.contains("<tool_call>") {
        return Err(format!(
            "check_tools[{id}]: raw <tool_call> markup leaked into output.text (must be parsed out)"
        ));
    }
    for (i, call) in out.tool_calls.iter().enumerate() {
        if call.name.trim().is_empty() {
            return Err(format!(
                "check_tools[{id}]: parsed tool_call #{i} has an empty function name"
            ));
        }
    }
    Ok(())
}

/// Profile for the snapshot-preparer conformance check: a real model `source` the linked backend can
/// prepare and then load, a writable `out_dir` the test owns, and the `quantize` scheme to exercise
/// (`None` covers the dense / passthrough path; `Some(_)` exercises persisted re-quantization).
#[derive(Clone, Debug)]
pub struct SnapshotPreparerProfile {
    /// A downloaded model the linked backend can prepare (an HF-safetensors dir or a `*.gguf`).
    pub source: PathBuf,
    /// A writable directory the prepared snapshot is written into.
    pub out_dir: PathBuf,
    /// The quantization to bake in (`None` ⇒ dense).
    pub quantize: Option<Quantize>,
}

/// Drive the registered snapshot preparer through the contract's guarantees end-to-end on a real
/// fixture: [`prepare_snapshot`](core_llm::prepare_snapshot) materializes a persisted snapshot from
/// `profile.source` whose [`PrepareReport`](core_llm::PrepareReport) is self-consistent and whose
/// `out_dir` [`load_for_model`](core_llm::load_for_model) can load, and an unrecognized source is a
/// typed [`Error::Unsupported`] rather than a panic.
///
/// This is **not** part of the always-on [`textllm_conformance`] run — it needs a model on disk and
/// both a preparer and a provider linked, so a backend's tests call it directly behind their
/// gated-real-model fixture.
pub fn check_snapshot_preparer(profile: &SnapshotPreparerProfile) -> Result<(), String> {
    let spec = PrepareSpec {
        source: profile.source.clone(),
        out_dir: profile.out_dir.clone(),
        quantize: profile.quantize,
    };

    let report = prepare_snapshot(&spec).map_err(|e| {
        format!(
            "check_snapshot_preparer: prepare_snapshot('{}') failed: {e}",
            profile.source.display()
        )
    })?;

    if report.quantized != profile.quantize {
        return Err(format!(
            "check_snapshot_preparer: report.quantized {:?} != requested {:?}",
            report.quantized, profile.quantize
        ));
    }
    if report.num_tensors == 0 {
        return Err(
            "check_snapshot_preparer: report.num_tensors is 0 (nothing was written?)".to_string(),
        );
    }
    if !report.out_dir.exists() {
        return Err(format!(
            "check_snapshot_preparer: report.out_dir '{}' does not exist after prepare",
            report.out_dir.display()
        ));
    }

    // The headline guarantee: the prepared snapshot is loadable through the contract.
    load_for_model(&LoadSpec::dense(report.out_dir.to_string_lossy().to_string())).map_err(|e| {
        format!(
            "check_snapshot_preparer: prepared snapshot '{}' not loadable via load_for_model: {e}",
            report.out_dir.display()
        )
    })?;

    // An unrecognized source is a typed Unsupported, not a panic or a generic error.
    let bogus = profile.out_dir.join("core-llm-testkit-not-a-model");
    let bogus_out = profile.out_dir.join("core-llm-testkit-bogus-out");
    match prepare_snapshot(&PrepareSpec::dense(bogus, bogus_out)) {
        Err(Error::Unsupported(_)) => {}
        Err(other) => {
            return Err(format!(
                "check_snapshot_preparer: unknown source gave `{other}` (want Unsupported)"
            ));
        }
        Ok(_) => {
            return Err(
                "check_snapshot_preparer: prepare_snapshot succeeded on a non-model source"
                    .to_string(),
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests;
