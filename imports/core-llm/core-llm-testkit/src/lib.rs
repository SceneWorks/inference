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
    Content, Error, FinishReason, ImageRef, Message, Role, Sampling, StreamEvent, TextLlm,
    TextLlmRequest, VERSION,
};

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
            }],
            sampling: Sampling::greedy(),
            max_new_tokens: self.max_new_tokens,
            seed: Some(0),
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
    let with_profile: [ProfileCheck; 5] = [
        check_validate,
        check_streaming,
        check_mid_stream_cancel,
        check_seed_determinism,
        check_multimodal,
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
    let mut done: Option<(FinishReason, u32)> = None;
    let out = p
        .generate(&req, &mut |ev| match ev {
            StreamEvent::Token { text, index, .. } => {
                indices.push(index);
                streamed.push_str(&text);
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
    if out.usage.generated_tokens != gen || gen as usize != indices.len() {
        return Err(format!(
            "check_streaming[{id}]: usage.generated_tokens ({}) disagrees with {} streamed tokens / Done {gen}",
            out.usage.generated_tokens,
            indices.len()
        ));
    }
    if out.usage.prompt_tokens == 0 {
        return Err(format!("check_streaming[{id}]: usage.prompt_tokens is 0"));
    }
    if streamed != out.text {
        return Err(format!(
            "check_streaming[{id}]: streamed token deltas do not reconstruct output.text"
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
    let req = profile.request(profile.determinism_sampling, Some(profile.determinism_seed));
    let a = p
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("check_seed_determinism[{id}]: generate() failed: {e}"))?;
    let b = p
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("check_seed_determinism[{id}]: generate() failed: {e}"))?;
    if a.text != b.text {
        return Err(format!(
            "check_seed_determinism[{id}]: the same seed produced different output across two runs"
        ));
    }
    let mut req2 = req.clone();
    req2.seed = Some(profile.determinism_seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
    let c = p
        .generate(&req2, &mut |_| {})
        .map_err(|e| format!("check_seed_determinism[{id}]: generate() failed: {e}"))?;
    if a.text == c.text {
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

#[cfg(test)]
mod tests;
