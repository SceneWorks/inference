//! Shared conformance checks for native StarVector SVG providers.
//!
//! This module links only `core-llm`. MLX and Candle tests can therefore use the same fixture and
//! contract assertions without requiring weights, tensor types, or a wall-clock-dependent timeout.

use core_llm::{
    Content, Error, ImageRef, Message, Sampling, StarVectorBoundedStream, StarVectorFinishReason,
    StarVectorOutput, StarVectorProvider, StarVectorRequest, StarVectorStreamEvent,
    StarVectorStreamStatus, TextLlmRequest,
};
use std::time::Duration;

/// Input and bound configuration for [`starvector_conformance`]. Image and text are independently
/// optional; at least one is needed when constructing the concrete request.
#[derive(Clone, Debug)]
pub struct StarVectorProfile {
    /// Optional decoded RGB input image.
    pub image: Option<ImageRef>,
    /// Optional text guidance.
    pub text: Option<String>,
    /// Existing `TextLlmRequest` token budget.
    pub max_new_tokens: u32,
    /// SVG UTF-8 byte limit.
    pub max_svg_bytes: usize,
    /// Bounded-stream elapsed-time limit.
    pub max_wall_time: Duration,
    /// Deterministic decoding seed passed through the existing text request.
    pub seed: u64,
}

impl StarVectorProfile {
    /// A cheap image-plus-text profile that exercises both optional conditioning fields.
    pub fn cheap() -> Self {
        Self {
            image: Some(ImageRef::new(2, 2, vec![0x40; 12]).expect("2x2 RGB")),
            text: Some("convert this compact mark to SVG".into()),
            max_new_tokens: 16,
            max_svg_bytes: 1024,
            max_wall_time: Duration::from_secs(1),
            seed: 7,
        }
    }

    /// Construct the public request with image/text conditioning in the existing multimodal
    /// message representation.
    pub fn request(&self) -> StarVectorRequest {
        let mut content = Vec::new();
        if let Some(text) = &self.text {
            content.push(Content::Text(text.clone()));
        }
        if let Some(image) = &self.image {
            content.push(Content::Image(image.clone()));
        }
        let mut text_request = TextLlmRequest::new(
            vec![Message {
                role: core_llm::Role::User,
                content,
                thinking: None,
                tool_calls: Vec::new(),
            }],
            self.max_new_tokens,
        );
        text_request.sampling = Sampling::greedy();
        text_request.seed = Some(self.seed);
        StarVectorRequest::new(text_request, self.max_svg_bytes, self.max_wall_time)
    }
}

/// Deterministic fixture streamed in two source fragments. The accented attribute value asserts
/// byte accounting is over UTF-8 source bytes rather than characters, while avoiding SVG text.
#[derive(Clone, Debug)]
pub struct StarVectorFixture {
    /// Stable fixture identity for backend test diagnostics.
    pub name: &'static str,
    /// Decoded, valid-UTF-8 source fragments in token order.
    pub fragments: &'static [&'static str],
    /// One complete expected root.
    pub expected_svg: &'static str,
}

/// The common greedy fixture both backend implementations must be able to drive.
pub fn deterministic_svg_fixture() -> StarVectorFixture {
    StarVectorFixture {
        name: "starvector-compact-mark-v1",
        fragments: &["<svg data-name=\"caf", "é\"><path d=\"M0 0\"/></svg>"],
        expected_svg: "<svg data-name=\"café\"><path d=\"M0 0\"/></svg>",
    }
}

/// Run the full shared suite. A provider-specific test supplies its loaded MLX or Candle instance;
/// this harness does not register providers or know anything about tensor backends.
pub fn starvector_conformance(
    make: impl Fn() -> Box<dyn StarVectorProvider>,
    profile: &StarVectorProfile,
) {
    let provider = make();
    let p = provider.as_ref();
    let checks = [
        check_starvector_descriptor(p),
        check_starvector_validate(p, profile),
        check_starvector_streaming(p, profile),
        check_starvector_cancellation(p, profile),
        check_starvector_bounded_fixture(profile, &deterministic_svg_fixture()),
    ];
    let failures: Vec<String> = checks.into_iter().filter_map(Result::err).collect();
    if !failures.is_empty() {
        panic!(
            "core-llm StarVector conformance FAILED for `{}`:\n  - {}",
            p.descriptor().id,
            failures.join("\n  - ")
        );
    }
}

/// The SVG metadata is internally valid; identity and capabilities remain on the actual `TextLlm`.
pub fn check_starvector_descriptor(p: &dyn StarVectorProvider) -> Result<(), String> {
    let descriptor = p.starvector_descriptor();
    descriptor
        .validate()
        .map_err(|error| format!("check_starvector_descriptor: invalid descriptor: {error}"))?;
    if p.descriptor().id.trim().is_empty() || p.descriptor().backend.trim().is_empty() {
        return Err("check_starvector_descriptor: TextLlm identity is empty".into());
    }
    Ok(())
}

/// The provider accepts the supplied optional image/text conditioning through the existing request
/// shape, and rejects a request with neither form of conditioning.
pub fn check_starvector_validate(
    p: &dyn StarVectorProvider,
    profile: &StarVectorProfile,
) -> Result<(), String> {
    let request = profile.request();
    p.validate(&request.text_request).map_err(|error| {
        format!("check_starvector_validate: actual TextLlm rejected valid request: {error}")
    })?;
    p.validate_svg(&request)
        .map_err(|error| format!("check_starvector_validate: valid request rejected: {error}"))?;

    let empty_profile = StarVectorProfile {
        image: None,
        text: None,
        ..profile.clone()
    };
    let empty_request = empty_profile.request();
    if p.validate(&empty_request.text_request).is_ok() || p.validate_svg(&empty_request).is_ok() {
        return Err("check_starvector_validate: accepted missing image/text conditioning".into());
    }
    Ok(())
}

/// Streamed source reconstructs the final, publishable root exactly, stays within token and UTF-8
/// byte bounds, and carries the same typed reason in `Done` and the returned output.
pub fn check_starvector_streaming(
    p: &dyn StarVectorProvider,
    profile: &StarVectorProfile,
) -> Result<(), String> {
    let request = profile.request();
    let mut source = String::new();
    let mut indices = Vec::new();
    let mut progress = Vec::new();
    let mut done = None;
    let output = p
        .generate_svg(&request, &mut |event| match event {
            StarVectorStreamEvent::Source { text, index } => {
                source.push_str(&text);
                indices.push(index);
            }
            StarVectorStreamEvent::Progress { generated_tokens } => progress.push(generated_tokens),
            StarVectorStreamEvent::Done {
                finish_reason,
                generated_tokens,
                generated_bytes,
            } => done = Some((finish_reason, generated_tokens, generated_bytes)),
        })
        .map_err(|error| format!("check_starvector_streaming: generate_svg failed: {error}"))?;
    let (reason, tokens, bytes) =
        done.ok_or_else(|| "check_starvector_streaming: no terminal Done event".to_string())?;
    if indices.is_empty()
        || indices.windows(2).any(|pair| pair[0] >= pair[1])
        || indices.last().is_some_and(|index| *index > tokens)
    {
        return Err(
            "check_starvector_streaming: source indices are missing or non-monotonic".into(),
        );
    }
    // Source indices describe visible fragments, so hidden tokenizer output can leave gaps.
    // Progress is the accepted decoder-token count, including those hidden tokens.
    if progress.len() != tokens as usize
        || progress
            .iter()
            .enumerate()
            .any(|(index, count)| *count != index as u32 + 1)
    {
        return Err(
            "check_starvector_streaming: Progress omits tokens or disagrees with Done".into(),
        );
    }
    if output.finish_reason != reason
        || output.generated_tokens != tokens
        || output.generated_bytes != bytes
    {
        return Err("check_starvector_streaming: Done counters/reason disagree with output".into());
    }
    if tokens > request.text_request.max_new_tokens || bytes > request.max_svg_bytes {
        return Err("check_starvector_streaming: output exceeded its declared bounds".into());
    }
    if output.svg.as_deref() != Some(source.as_str()) {
        return Err(
            "check_starvector_streaming: streamed source does not reconstruct output.svg".into(),
        );
    }
    match reason {
        StarVectorFinishReason::CompleteRoot | StarVectorFinishReason::Eos => Ok(()),
        other => Err(format!(
            "check_starvector_streaming: ordinary conformance generation ended {other:?}, not a publishable root"
        )),
    }
}

/// An already-cancelled request reports the existing typed `Error::Canceled` before decode.
pub fn check_starvector_cancellation(
    p: &dyn StarVectorProvider,
    profile: &StarVectorProfile,
) -> Result<(), String> {
    let request = profile.request();
    request.text_request.cancel.cancel();
    match p.generate_svg(&request, &mut |_| {}) {
        Err(Error::Canceled) => Ok(()),
        Ok(_) => Err("check_starvector_cancellation: cancelled request returned Ok".into()),
        Err(other) => Err(format!(
            "check_starvector_cancellation: expected Error::Canceled, got {other:?}"
        )),
    }
}

/// Exercise the shared pure boundary with a deterministic fixture. Providers can use this directly
/// in backend tests to prove their decoder-loop integration without using wall-clock sleeps.
pub fn check_starvector_bounded_fixture(
    profile: &StarVectorProfile,
    fixture: &StarVectorFixture,
) -> Result<(), String> {
    let request = profile.request();
    let mut stream = StarVectorBoundedStream::new(&request);
    for fragment in fixture.fragments {
        let status = stream.push(fragment, Duration::ZERO).map_err(|error| {
            format!(
                "check_starvector_bounded_fixture[{}]: {error}",
                fixture.name
            )
        })?;
        if status == StarVectorStreamStatus::Stop(StarVectorFinishReason::CompleteRoot) {
            break;
        }
    }
    let output = stream.output().map_err(|error| {
        format!(
            "check_starvector_bounded_fixture[{}]: {error}",
            fixture.name
        )
    })?;
    if output != expected_fixture_output(fixture) {
        return Err(format!(
            "check_starvector_bounded_fixture[{}]: deterministic fixture output mismatched",
            fixture.name
        ));
    }
    Ok(())
}

fn expected_fixture_output(fixture: &StarVectorFixture) -> StarVectorOutput {
    StarVectorOutput {
        svg: Some(fixture.expected_svg.to_string()),
        generated_tokens: fixture.fragments.len() as u32,
        generated_bytes: fixture.expected_svg.len(),
        finish_reason: StarVectorFinishReason::CompleteRoot,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core_llm::{
        Channel, Result as CoreResult, StarVectorDescriptor, StarVectorTier, StreamEvent,
        TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, Usage,
    };

    struct Stub {
        text: TextLlmDescriptor,
        star: StarVectorDescriptor,
        hidden_token: bool,
        bad_indices: bool,
        progress_offset: u32,
        omit_progress: bool,
    }

    fn stub() -> Stub {
        let text = TextLlmDescriptor {
            id: "stub-starvector".into(),
            family: "starvector".into(),
            backend: "test".into(),
            capabilities: TextLlmCapabilities {
                max_context_tokens: 0,
                max_new_tokens: 32,
                supports_system_prompt: false,
                supports_vision: true,
                supports_video: false,
                supports_audio: false,
                supports_thinking: false,
                supports_tools: false,
                supported_constraints: Vec::new(),
            },
        };
        let star = StarVectorDescriptor {
            tier: StarVectorTier::OneB,
            preprocessing: core_llm::ImagePreprocessing {
                image_size: 224,
                channels: 3,
                preserve_aspect_ratio: true,
            },
            projection: core_llm::ProjectionMetadata {
                vision_encoder: core_llm::VisionEncoderArchitecture::Clip,
                decoder: core_llm::DecoderArchitecture::GptBigCode,
                vision_hidden_size: 768,
                decoder_hidden_size: 2048,
                image_token_count: 256,
            },
            max_svg_bytes: 1024,
            max_wall_time: Some(Duration::from_secs(1)),
        };
        Stub {
            text,
            star,
            hidden_token: false,
            bad_indices: false,
            progress_offset: 0,
            omit_progress: false,
        }
    }

    impl core_llm::TextLlm for Stub {
        fn descriptor(&self) -> &TextLlmDescriptor {
            &self.text
        }

        fn validate(&self, request: &TextLlmRequest) -> CoreResult<()> {
            self.text
                .capabilities
                .validate_request(&self.text.id, request)
        }

        fn generate(
            &self,
            request: &TextLlmRequest,
            on_event: &mut dyn FnMut(StreamEvent),
        ) -> CoreResult<TextLlmOutput> {
            if request.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let usage = Usage {
                prompt_tokens: 1,
                generated_tokens: 1,
            };
            on_event(StreamEvent::Token {
                id: 0,
                text: "<svg></svg>".into(),
                index: 0,
                channel: Channel::Content,
            });
            on_event(StreamEvent::Done {
                finish_reason: core_llm::FinishReason::Stop,
                usage,
            });
            Ok(TextLlmOutput {
                text: "<svg></svg>".into(),
                thinking: None,
                tool_calls: Vec::new(),
                usage,
                finish_reason: Some(core_llm::FinishReason::Stop),
            })
        }
    }

    impl StarVectorProvider for Stub {
        fn starvector_descriptor(&self) -> &StarVectorDescriptor {
            &self.star
        }

        fn generate_svg(
            &self,
            request: &StarVectorRequest,
            on_event: &mut dyn FnMut(StarVectorStreamEvent),
        ) -> CoreResult<StarVectorOutput> {
            self.validate_svg(request)?;
            if request.text_request.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let fixture = deterministic_svg_fixture();
            let mut stream = StarVectorBoundedStream::new(request);
            for (index, fragment) in fixture.fragments.iter().enumerate() {
                if index == 1 && self.hidden_token {
                    stream.push("", Duration::ZERO)?;
                    if !self.omit_progress {
                        on_event(StarVectorStreamEvent::Progress {
                            generated_tokens: stream.generated_tokens() + self.progress_offset,
                        });
                    }
                }
                let status = stream.push(fragment, Duration::ZERO)?;
                on_event(StarVectorStreamEvent::Source {
                    text: (*fragment).to_string(),
                    index: if self.bad_indices {
                        0
                    } else {
                        index as u32 + u32::from(index > 0 && self.hidden_token)
                    },
                });
                if !self.omit_progress {
                    on_event(StarVectorStreamEvent::Progress {
                        generated_tokens: stream.generated_tokens() + self.progress_offset,
                    });
                }
                if matches!(status, StarVectorStreamStatus::Stop(_)) {
                    break;
                }
            }
            let output = stream.output()?;
            on_event(StarVectorStreamEvent::Done {
                finish_reason: output.finish_reason,
                generated_tokens: output.generated_tokens,
                generated_bytes: output.generated_bytes,
            });
            Ok(output)
        }
    }

    #[test]
    fn good_stub_satisfies_shared_starvector_conformance() {
        starvector_conformance(|| Box::new(stub()), &StarVectorProfile::cheap());
    }

    #[test]
    fn streaming_accepts_hidden_token_gaps_and_rejects_false_progress() {
        let profile = StarVectorProfile::cheap();
        let mut provider = stub();
        provider.hidden_token = true;
        check_starvector_streaming(&provider, &profile).unwrap();
        provider.progress_offset = 1;
        assert!(check_starvector_streaming(&provider, &profile)
            .unwrap_err()
            .contains("Progress"));
        provider.progress_offset = 0;
        provider.omit_progress = true;
        assert!(check_starvector_streaming(&provider, &profile)
            .unwrap_err()
            .contains("Progress"));
        provider.omit_progress = false;
        provider.bad_indices = true;
        assert!(check_starvector_streaming(&provider, &profile)
            .unwrap_err()
            .contains("non-monotonic"));
    }

    #[test]
    fn optional_image_and_text_profiles_construct_independent_requests() {
        let image_only = StarVectorProfile {
            text: None,
            ..StarVectorProfile::cheap()
        };
        assert!(image_only.request().text_request.has_image());
        assert!(!image_only.request().has_text());

        let text_only = StarVectorProfile {
            image: None,
            ..StarVectorProfile::cheap()
        };
        assert!(!text_only.request().text_request.has_image());
        assert!(text_only.request().has_text());
    }

    #[test]
    fn malformed_fixture_cannot_be_reported_as_complete() {
        let bad = StarVectorFixture {
            name: "suffix",
            fragments: &["<svg></svg><script/>"],
            expected_svg: "<svg></svg>",
        };
        assert!(check_starvector_bounded_fixture(&StarVectorProfile::cheap(), &bad).is_err());
    }
}
