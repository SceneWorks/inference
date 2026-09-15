//! Tensor-free contract and host policy for native StarVector SVG providers.
//!
//! A StarVector provider extends [`crate::TextLlm`]: it is loaded through the existing text-provider
//! registry and keeps the same descriptor, multimodal request, cancellation, and error lifecycle.
//! This module adds only SVG-specific metadata and the byte/token/time boundary that MLX and Candle
//! apply before any source is published.

use crate::{Error, Result, TextLlm, TextLlmRequest};
use std::time::Duration;

/// The checkpoint quality tier a StarVector descriptor represents.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarVectorTier {
    /// The StarVector-1B family.
    OneB,
    /// The StarVector-8B family.
    EightB,
}

/// Vision encoder architecture declared by a StarVector checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionEncoderArchitecture {
    /// CLIP image tower used by the 1B checkpoint.
    Clip,
    /// SigLIP image tower used by the 8B checkpoint.
    Siglip,
}

/// Autoregressive decoder architecture declared by a StarVector checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderArchitecture {
    /// GPTBigCode / StarCoderBase-compatible decoder used by the 1B checkpoint.
    GptBigCode,
    /// StarCoder2-compatible decoder used by the 8B checkpoint.
    StarCoder2,
}

/// Image preprocessing facts that a backend must reproduce before vision projection.
///
/// The RGB8 [`crate::ImageRef`] stays tensor-free. A backend turns these declared facts into its native
/// image/tensor operations at its own boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImagePreprocessing {
    /// Square image size expected by the vision encoder.
    pub image_size: u32,
    /// Channel count expected after decode; StarVector currently consumes RGB.
    pub channels: u8,
    /// Whether resize preserves aspect ratio before the final center crop.
    pub preserve_aspect_ratio: bool,
}

impl ImagePreprocessing {
    /// Validate metadata before a provider publishes it in a descriptor.
    pub fn validate(&self) -> Result<()> {
        if self.image_size == 0 {
            return Err(Error::InvalidRequest(
                "StarVector preprocessing image_size must be non-zero".into(),
            ));
        }
        if self.channels != 3 {
            return Err(Error::InvalidRequest(format!(
                "StarVector preprocessing requires RGB (3 channels), got {}",
                self.channels
            )));
        }
        Ok(())
    }
}

/// The tensor-neutral shape and architecture facts for the vision-to-decoder projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionMetadata {
    /// The checkpoint's image tower.
    pub vision_encoder: VisionEncoderArchitecture,
    /// The checkpoint's text decoder.
    pub decoder: DecoderArchitecture,
    /// Vision tower hidden width before projection.
    pub vision_hidden_size: u32,
    /// Decoder hidden width after projection.
    pub decoder_hidden_size: u32,
    /// Number of image embeddings inserted into the decoder prompt.
    pub image_token_count: u32,
}

impl ProjectionMetadata {
    /// Validate shape facts without naming a backend tensor type.
    pub fn validate(&self) -> Result<()> {
        if self.vision_hidden_size == 0
            || self.decoder_hidden_size == 0
            || self.image_token_count == 0
        {
            return Err(Error::InvalidRequest(
                "StarVector projection dimensions and image_token_count must be non-zero".into(),
            ));
        }
        Ok(())
    }
}

/// SVG-specific model facts. The provider identity and capabilities stay exclusively on its
/// [`TextLlm::descriptor`] result, preventing a StarVector extension from drifting into a second
/// provider identity.
#[derive(Clone, Debug)]
pub struct StarVectorDescriptor {
    /// Model tier.
    pub tier: StarVectorTier,
    /// Host-visible image preprocessing contract.
    pub preprocessing: ImagePreprocessing,
    /// Host-visible image projection contract.
    pub projection: ProjectionMetadata,
    /// Hard maximum emitted SVG UTF-8 bytes (`0` means no descriptor-specific tighter cap).
    pub max_svg_bytes: usize,
    /// Hard maximum wall time (`None` means no descriptor-specific tighter cap).
    pub max_wall_time: Option<Duration>,
}

impl StarVectorDescriptor {
    /// Validate descriptor facts before using this provider.
    pub fn validate(&self) -> Result<()> {
        self.preprocessing.validate()?;
        self.projection.validate()
    }
}

/// A request to generate one SVG document. Image and text are independently optional, but a request
/// needs at least one non-empty conditioning input. This allows image-to-SVG, disclosed
/// image-plus-text guidance, and future text-to-SVG providers without overloading raster contracts.
#[derive(Clone, Debug)]
pub struct StarVectorRequest {
    /// The existing cancellable multimodal text request. Its messages own optional image/text
    /// conditioning, and its sampling, token budget, seed, and cancellation handle are reused.
    pub text_request: TextLlmRequest,
    /// Maximum UTF-8 source bytes allowed to reach the SVG boundary.
    pub max_svg_bytes: usize,
    /// Maximum elapsed generation time. Providers pass their measured elapsed duration into the
    /// bounded stream guard; the contract never sleeps or relies on scheduler timing in tests.
    pub max_wall_time: Duration,
}

impl StarVectorRequest {
    /// Build a request with explicit limits and optional conditioning.
    pub fn new(
        text_request: TextLlmRequest,
        max_svg_bytes: usize,
        max_wall_time: Duration,
    ) -> Self {
        Self {
            text_request,
            max_svg_bytes,
            max_wall_time,
        }
    }

    /// Whether this request has non-empty textual conditioning.
    pub fn has_text(&self) -> bool {
        self.text_request
            .messages
            .iter()
            .any(|message| !message.text_content().trim().is_empty())
    }

    /// Validate StarVector-only resource bounds. A provider's [`TextLlm::validate`] remains the
    /// sole authority for message/image/text capability validation.
    pub fn validate_bounds(&self, descriptor: &StarVectorDescriptor) -> Result<()> {
        descriptor.validate()?;
        if self.max_svg_bytes == 0 || self.max_wall_time.is_zero() {
            return Err(Error::InvalidRequest(
                "StarVector token, byte, and wall-time limits must be non-zero".into(),
            ));
        }
        if descriptor.max_svg_bytes > 0 && self.max_svg_bytes > descriptor.max_svg_bytes {
            return Err(Error::InvalidRequest(format!(
                "StarVector max_svg_bytes {} exceeds provider cap {}",
                self.max_svg_bytes, descriptor.max_svg_bytes
            )));
        }
        if let Some(cap) = descriptor.max_wall_time {
            if self.max_wall_time > cap {
                return Err(Error::InvalidRequest(format!(
                    "StarVector max_wall_time {:?} exceeds provider cap {:?}",
                    self.max_wall_time, cap
                )));
            }
        }
        Ok(())
    }
}

/// Typed terminal result for a bounded StarVector stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarVectorFinishReason {
    /// Exactly one `<svg>…</svg>` root was completed with no active prefix or suffix content.
    CompleteRoot,
    /// The backend saw EOS after an already complete root.
    Eos,
    /// The token budget ended before a document could complete.
    TokenLimit,
    /// The UTF-8 byte budget ended before a document could complete.
    ByteLimit,
    /// The supplied elapsed duration reached the request wall-time budget.
    WallTimeLimit,
    /// Cooperative cancellation was observed.
    Cancelled,
}

/// A streamed UTF-8 source fragment, token progress, or a terminal event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StarVectorStreamEvent {
    /// A complete UTF-8 source fragment for one generated token.
    Source {
        /// Source delta. It is a Rust [`String`], and is therefore always valid UTF-8.
        text: String,
        /// Source index. Hidden tokens may leave gaps; a static prefix can occupy index zero.
        index: u32,
    },
    /// Accepted decoder-token progress, including tokens without a visible UTF-8 delta.
    Progress {
        /// Count excludes the static SVG prefix and EOS, matching the final output counter.
        generated_tokens: u32,
    },
    /// The deterministic terminal result and counters.
    Done {
        /// Terminal classification.
        finish_reason: StarVectorFinishReason,
        /// Number of accepted decoder tokens.
        generated_tokens: u32,
        /// Number of accepted UTF-8 source bytes.
        generated_bytes: usize,
    },
}

/// Status returned by [`StarVectorBoundedStream::push`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarVectorStreamStatus {
    /// The decoder may continue.
    Continue,
    /// The decoder must stop with this typed reason.
    Stop(StarVectorFinishReason),
}

/// Final provider result. `svg` is present only for a complete-root/EOS success; budget and
/// cancellation outcomes intentionally expose no partial SVG for publication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarVectorOutput {
    /// One complete SVG root, if a publishable source document completed.
    pub svg: Option<String>,
    /// Number of accepted decoder tokens.
    pub generated_tokens: u32,
    /// Number of accepted UTF-8 source bytes.
    pub generated_bytes: usize,
    /// Typed terminal classification.
    pub finish_reason: StarVectorFinishReason,
}

/// Reusable host-side stop guard for provider decoding loops.
///
/// Backends call [`push`](Self::push) once per decoded token with a monotonic elapsed duration. The
/// guard checks cancellation, time, token count, byte count, UTF-8 source (by construction), and
/// exactly-one-root policy in a stable order. It makes timeout tests deterministic because callers
/// supply elapsed time rather than relying on a busy host clock.
pub struct StarVectorBoundedStream<'a> {
    request: &'a StarVectorRequest,
    source: String,
    generated_tokens: u32,
    finish_reason: Option<StarVectorFinishReason>,
}

impl<'a> StarVectorBoundedStream<'a> {
    /// Start an empty bounded stream for an already-validated request.
    pub fn new(request: &'a StarVectorRequest) -> Self {
        Self {
            request,
            source: String::new(),
            generated_tokens: 0,
            finish_reason: None,
        }
    }

    /// Seed a fixed decoder prompt into the source without charging a generated token.
    ///
    /// Some checkpoints prefill a literal root opener (StarVector-1B uses `<svg`) before the
    /// first sampled continuation. The prompt is still part of the published UTF-8 byte budget,
    /// but is not a sampled token and therefore must not consume `max_new_tokens`. Only an
    /// incomplete, root-valid prefix is accepted; a provider cannot use this to publish a complete
    /// document or to smuggle active markup before the root.
    pub fn push_static_prefix(&mut self, fragment: &str) -> Result<StarVectorStreamStatus> {
        if self.finish_reason.is_some() || self.generated_tokens != 0 || !self.source.is_empty() {
            return Err(Error::InvalidRequest(
                "StarVector static source prefix must be the first stream input".into(),
            ));
        }
        if self.request.text_request.cancel.is_cancelled() {
            return Ok(self.stop(StarVectorFinishReason::Cancelled));
        }
        let next_bytes = fragment.len();
        if next_bytes > self.request.max_svg_bytes {
            return Ok(self.stop(StarVectorFinishReason::ByteLimit));
        }
        self.source.push_str(fragment);
        match scan_svg_root(&self.source)? {
            SvgRootState::Incomplete => Ok(StarVectorStreamStatus::Continue),
            SvgRootState::Complete => Err(Error::InvalidRequest(
                "StarVector static source prefix may not complete an SVG root".into(),
            )),
        }
    }

    /// Accept one decoded token's valid UTF-8 source fragment.
    pub fn push(&mut self, fragment: &str, elapsed: Duration) -> Result<StarVectorStreamStatus> {
        if self.finish_reason.is_some() {
            return Err(Error::InvalidRequest(
                "StarVector stream received source after it had already stopped".into(),
            ));
        }
        if self.request.text_request.cancel.is_cancelled() {
            return Ok(self.stop(StarVectorFinishReason::Cancelled));
        }
        if elapsed >= self.request.max_wall_time {
            return Ok(self.stop(StarVectorFinishReason::WallTimeLimit));
        }
        if self.generated_tokens >= self.request.text_request.max_new_tokens {
            return Ok(self.stop(StarVectorFinishReason::TokenLimit));
        }
        let next_bytes = self
            .source
            .len()
            .checked_add(fragment.len())
            .ok_or_else(|| Error::InvalidRequest("StarVector source byte count overflow".into()))?;
        if next_bytes > self.request.max_svg_bytes {
            return Ok(self.stop(StarVectorFinishReason::ByteLimit));
        }

        self.source.push_str(fragment);
        self.generated_tokens += 1;
        match scan_svg_root(&self.source)? {
            SvgRootState::Incomplete => Ok(StarVectorStreamStatus::Continue),
            SvgRootState::Complete => Ok(self.stop(StarVectorFinishReason::CompleteRoot)),
        }
    }

    /// Accepted decoder tokens so far, including hidden tokens passed as empty fragments.
    pub fn generated_tokens(&self) -> u32 {
        self.generated_tokens
    }

    /// Record EOS. EOS is publishable only after the source already forms exactly one complete root.
    pub fn finish_eos(&mut self) -> Result<StarVectorStreamStatus> {
        if self.finish_reason == Some(StarVectorFinishReason::CompleteRoot) {
            self.finish_reason = Some(StarVectorFinishReason::Eos);
            return Ok(StarVectorStreamStatus::Stop(StarVectorFinishReason::Eos));
        }
        if self.finish_reason.is_some() {
            return Err(Error::InvalidRequest(
                "StarVector stream received EOS after it had already stopped".into(),
            ));
        }
        if self.request.text_request.cancel.is_cancelled() {
            return Ok(self.stop(StarVectorFinishReason::Cancelled));
        }
        match scan_svg_root(&self.source)? {
            SvgRootState::Complete => Ok(self.stop(StarVectorFinishReason::Eos)),
            SvgRootState::Incomplete => Err(Error::InvalidRequest(
                "StarVector reached EOS before one complete SVG root".into(),
            )),
        }
    }

    /// Return the final result after a terminal status was observed.
    pub fn output(&self) -> Result<StarVectorOutput> {
        let finish_reason = self.finish_reason.ok_or_else(|| {
            Error::InvalidRequest("StarVector stream has not reached a terminal state".into())
        })?;
        let svg = match finish_reason {
            StarVectorFinishReason::CompleteRoot | StarVectorFinishReason::Eos => {
                Some(self.source.clone())
            }
            StarVectorFinishReason::TokenLimit
            | StarVectorFinishReason::ByteLimit
            | StarVectorFinishReason::WallTimeLimit
            | StarVectorFinishReason::Cancelled => None,
        };
        Ok(StarVectorOutput {
            svg,
            generated_tokens: self.generated_tokens,
            generated_bytes: self.source.len(),
            finish_reason,
        })
    }

    fn stop(&mut self, reason: StarVectorFinishReason) -> StarVectorStreamStatus {
        self.finish_reason = Some(reason);
        StarVectorStreamStatus::Stop(reason)
    }
}

/// A loaded native StarVector provider.
pub trait StarVectorProvider: TextLlm {
    /// SVG-specific architecture, preprocessing, and declared limits. Identity and capabilities
    /// remain available only through [`TextLlm::descriptor`].
    fn starvector_descriptor(&self) -> &StarVectorDescriptor;

    /// Validate a request before materializing tensors or starting decode. The default deliberately
    /// delegates to this exact provider's [`TextLlm::validate`] rather than a copied descriptor.
    fn validate_svg(&self, request: &StarVectorRequest) -> Result<()> {
        self.validate(&request.text_request)?;
        request.validate_bounds(self.starvector_descriptor())
    }

    /// Generate one bounded SVG source document while reporting source stream events.
    fn generate_svg(
        &self,
        request: &StarVectorRequest,
        on_event: &mut dyn FnMut(StarVectorStreamEvent),
    ) -> Result<StarVectorOutput>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SvgRootState {
    Incomplete,
    Complete,
}

/// Recognize exactly one SVG root without treating it as safe-to-render XML. Sanitization remains a
/// later boundary; this policy merely prevents a decoder from publishing partial source or hiding
/// active prefix/suffix markup around a root.
fn scan_svg_root(source: &str) -> Result<SvgRootState> {
    let bytes = source.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    if index == bytes.len() {
        return Ok(SvgRootState::Incomplete);
    }
    let prefix = &source[index..];
    if !prefix.starts_with("<svg") {
        if "<svg".starts_with(prefix) {
            return Ok(SvgRootState::Incomplete);
        }
        return Err(Error::InvalidRequest(
            "StarVector SVG source must have no content before its <svg> root".into(),
        ));
    }

    let mut stack = Vec::<String>::new();
    let mut complete_at = None;
    while index < bytes.len() {
        if bytes[index] != b'<' {
            index += 1;
            continue;
        }
        if source[index..].starts_with("<!--") {
            let Some(end) = source[index + 4..].find("-->") else {
                return Ok(SvgRootState::Incomplete);
            };
            index += 4 + end + 3;
            continue;
        }
        if source[index..].starts_with("<?") {
            let Some(end) = source[index + 2..].find("?>") else {
                return Ok(SvgRootState::Incomplete);
            };
            index += 2 + end + 2;
            continue;
        }
        if source[index..].starts_with("<![CDATA[") {
            let Some(end) = source[index + 9..].find("]]>") else {
                return Ok(SvgRootState::Incomplete);
            };
            index += 9 + end + 3;
            continue;
        }
        if source[index..].starts_with("<!") {
            return Err(Error::InvalidRequest(
                "StarVector SVG source may not contain declarations before sanitization".into(),
            ));
        }

        let Some(end) = tag_end(source, index + 1) else {
            return Ok(SvgRootState::Incomplete);
        };
        let tag = &source[index + 1..end];
        let closing = tag.starts_with('/');
        let tag_body = if closing { &tag[1..] } else { tag };
        let name = tag_name(tag_body).ok_or_else(|| {
            Error::InvalidRequest("StarVector SVG source has an invalid XML tag".into())
        })?;
        let self_closing = !closing && tag.trim_end().ends_with('/');

        if stack.is_empty() {
            if closing || name != "svg" {
                return Err(Error::InvalidRequest(
                    "StarVector source must begin with an <svg> root".into(),
                ));
            }
            if self_closing {
                complete_at = Some(end + 1);
                break;
            }
            stack.push(name.to_string());
        } else if closing {
            let opened = stack.pop().expect("non-empty stack checked above");
            if opened != name {
                return Err(Error::InvalidRequest(format!(
                    "StarVector SVG source closes </{name}> while <{opened}> is open"
                )));
            }
            if stack.is_empty() {
                complete_at = Some(end + 1);
                break;
            }
        } else if !self_closing {
            stack.push(name.to_string());
        }
        index = end + 1;
    }

    let Some(root_end) = complete_at else {
        return Ok(SvgRootState::Incomplete);
    };
    if source[root_end..].trim().is_empty() {
        Ok(SvgRootState::Complete)
    } else {
        Err(Error::InvalidRequest(
            "StarVector SVG source has content after its complete root".into(),
        ))
    }
}

fn tag_end(source: &str, mut index: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut quote = None;
    while index < bytes.len() {
        match (quote, bytes[index]) {
            (Some(q), byte) if byte == q => quote = None,
            (None, b'\'' | b'\"') => quote = Some(bytes[index]),
            (None, b'>') => return Some(index),
            _ => {}
        }
        index += 1;
    }
    None
}

fn tag_name(tag: &str) -> Option<&str> {
    let trimmed = tag.trim_start();
    let end = trimmed
        .bytes()
        .take_while(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b':' | b'_' | b'-'))
        .count();
    (end > 0).then_some(&trimmed[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    const STEP: Duration = Duration::ZERO;
    const TEST_LIMIT: Duration = Duration::from_secs(1);
    const TEST_TIMEOUT: Duration = Duration::from_millis(5);

    fn request(max_tokens: u32, max_bytes: usize, max_time: Duration) -> StarVectorRequest {
        StarVectorRequest::new(
            TextLlmRequest::new(vec![Message::user("vectorize a logo")], max_tokens),
            max_bytes,
            max_time,
        )
    }

    fn push(
        stream: &mut StarVectorBoundedStream<'_>,
        fragment: &str,
        tick: Duration,
    ) -> Result<StarVectorStreamStatus> {
        stream.push(fragment, tick)
    }

    fn assert_status(actual: StarVectorStreamStatus, expected: StarVectorStreamStatus) {
        assert_eq!(actual, expected);
    }

    #[test]
    fn bounded_stream_stops_at_exact_complete_root_and_preserves_utf8() {
        let req = request(4, 128, TEST_LIMIT);
        let mut stream = StarVectorBoundedStream::new(&req);
        let first = push(&mut stream, "<svg data-name=\"caf", STEP).unwrap();
        assert_eq!(first, StarVectorStreamStatus::Continue);
        let second = push(&mut stream, "é\"><path d=\"M0 0\"/></svg>", STEP).unwrap();
        assert_eq!(
            second,
            StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot)
        );
        let out = stream.output().unwrap();
        assert_eq!(out.finish_reason, StarVectorFinishReason::CompleteRoot);
        assert_eq!(
            out.svg.as_deref(),
            Some("<svg data-name=\"café\"><path d=\"M0 0\"/></svg>")
        );
        assert_eq!(out.generated_bytes, out.svg.as_ref().unwrap().len());
    }

    #[test]
    fn static_root_prefix_counts_bytes_but_not_generated_tokens() {
        let req = request(1, 32, TEST_LIMIT);
        let mut stream = StarVectorBoundedStream::new(&req);
        assert_eq!(
            stream.push_static_prefix("<svg").unwrap(),
            StarVectorStreamStatus::Continue
        );
        assert_eq!(
            push(&mut stream, "></svg>", STEP).unwrap(),
            StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot)
        );
        let output = stream.output().unwrap();
        assert_eq!(output.svg.as_deref(), Some("<svg></svg>"));
        assert_eq!(output.generated_tokens, 1);
        assert_eq!(output.generated_bytes, "<svg></svg>".len());

        let too_small = request(1, 3, TEST_LIMIT);
        let mut limited = StarVectorBoundedStream::new(&too_small);
        assert_eq!(
            limited.push_static_prefix("<svg").unwrap(),
            StarVectorStreamStatus::Stop(StarVectorFinishReason::ByteLimit)
        );
        assert_eq!(limited.output().unwrap().svg, None);
    }

    #[test]
    fn source_rejects_any_non_whitespace_prefix_and_suffix() {
        let req = request(4, 128, TEST_LIMIT);
        for prefix in [
            "<script/> <svg></svg>",
            "<!-- prefix --><svg></svg>",
            "<?xml-stylesheet href=\"https://bad.example/a.css\"?><svg></svg>",
            "<![CDATA[prefix]]><svg></svg>",
        ] {
            let mut stream = StarVectorBoundedStream::new(&req);
            let result = push(&mut stream, prefix, STEP);
            assert!(result.is_err(), "{prefix}");
        }

        let mut suffix = StarVectorBoundedStream::new(&req);
        let result = push(&mut suffix, "<svg></svg><script/>", STEP);
        assert!(result.is_err());
    }

    #[test]
    fn limits_and_cancellation_are_typed_without_partial_publication() {
        let token_request = request(1, 128, TEST_LIMIT);
        let mut token = StarVectorBoundedStream::new(&token_request);
        let first = push(&mut token, "<svg>", STEP).unwrap();
        assert_eq!(first, StarVectorStreamStatus::Continue);
        let second = push(&mut token, "</svg>", STEP).unwrap();
        assert_eq!(
            second,
            StarVectorStreamStatus::Stop(StarVectorFinishReason::TokenLimit)
        );
        assert_eq!(token.output().unwrap().svg, None);

        let byte_request = request(4, 5, TEST_LIMIT);
        let mut bytes = StarVectorBoundedStream::new(&byte_request);
        let first = push(&mut bytes, "<svg>", STEP).unwrap();
        assert_eq!(first, StarVectorStreamStatus::Continue);
        let second = push(&mut bytes, "</svg>", STEP).unwrap();
        assert_eq!(
            second,
            StarVectorStreamStatus::Stop(StarVectorFinishReason::ByteLimit)
        );
        assert_eq!(bytes.output().unwrap().svg, None);

        let time_request = request(4, 128, TEST_TIMEOUT);
        let mut time = StarVectorBoundedStream::new(&time_request);
        let result = push(&mut time, "<svg>", TEST_TIMEOUT).unwrap();
        assert_status(
            result,
            StarVectorStreamStatus::Stop(StarVectorFinishReason::WallTimeLimit),
        );
        assert_eq!(time.output().unwrap().svg, None);

        let cancelled = request(4, 128, TEST_LIMIT);
        cancelled.text_request.cancel.cancel();
        let mut stream = StarVectorBoundedStream::new(&cancelled);
        let result = push(&mut stream, "<svg></svg>", STEP).unwrap();
        assert_eq!(
            result,
            StarVectorStreamStatus::Stop(StarVectorFinishReason::Cancelled)
        );
        assert_eq!(stream.output().unwrap().svg, None);
    }

    #[test]
    fn eos_requires_a_complete_root() {
        let incomplete_request = request(4, 128, TEST_LIMIT);
        let mut incomplete = StarVectorBoundedStream::new(&incomplete_request);
        push(&mut incomplete, "<svg>", STEP).unwrap();
        assert!(incomplete.finish_eos().is_err());

        let complete_request = request(4, 128, TEST_LIMIT);
        let mut complete = StarVectorBoundedStream::new(&complete_request);
        push(&mut complete, "<svg></svg>", STEP).unwrap();
        assert_eq!(
            complete.finish_eos().unwrap(),
            StarVectorStreamStatus::Stop(StarVectorFinishReason::Eos)
        );
        assert_eq!(
            complete.output().unwrap().finish_reason,
            StarVectorFinishReason::Eos
        );
    }
}
