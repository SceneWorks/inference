//! Self-tests: a configurable stub provider proves each conformance check fires (good stub passes
//! everything; flipping one behavior fails exactly the matching check). Pure host — runs anywhere.

use super::*;

use core_llm::{
    Channel, Error, FinishReason, LoadSpec, Result as CoreResult, StreamEvent, TextLlm,
    TextLlmCapabilities, TextLlmDescriptor, TextLlmOutput, TextLlmRegistration, TextLlmRequest,
    Usage,
};

#[derive(Clone)]
struct Behavior {
    honest_validate: bool,
    emit_stream: bool,
    honor_cancel: bool,
    typed_cancel: bool,
    honor_seed: bool,
    reconstruct: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            honest_validate: true,
            emit_stream: true,
            honor_cancel: true,
            typed_cancel: true,
            honor_seed: true,
            reconstruct: true,
        }
    }
}

struct StubTextLlm {
    descriptor: TextLlmDescriptor,
    behavior: Behavior,
}

fn stub_caps() -> TextLlmCapabilities {
    TextLlmCapabilities {
        max_context_tokens: 0,
        max_new_tokens: 0,
        supports_system_prompt: true,
        supports_vision: false,
        supports_video: false,
        supports_thinking: false,
        supports_tools: false,
        supported_constraints: Vec::new(),
    }
}

fn descriptor_with(id: &str) -> TextLlmDescriptor {
    TextLlmDescriptor {
        id: id.to_string(),
        family: "stub".to_string(),
        backend: "test".to_string(),
        capabilities: stub_caps(),
    }
}

fn stub(id: &str, behavior: Behavior) -> StubTextLlm {
    StubTextLlm {
        descriptor: descriptor_with(id),
        behavior,
    }
}

fn good() -> StubTextLlm {
    stub("stub-llm", Behavior::good())
}

impl TextLlm for StubTextLlm {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
        if self.behavior.honest_validate {
            self.descriptor
                .capabilities
                .validate_request(&self.descriptor.id, req)
        } else {
            Ok(())
        }
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> CoreResult<TextLlmOutput> {
        if req.cancel.is_cancelled() {
            return if self.behavior.typed_cancel {
                Err(Error::Canceled)
            } else {
                Err(Error::Msg("cancelled".to_string()))
            };
        }

        let seed = if self.behavior.honor_seed {
            req.seed.unwrap_or(0)
        } else {
            0
        };
        let mut state = seed ^ 0x1234_5678;
        let mut text = String::new();
        let mut emitted = 0u32;

        for i in 0..req.max_new_tokens as usize {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                let usage = Usage {
                    prompt_tokens: 3,
                    generated_tokens: emitted,
                };
                on_event(StreamEvent::Done {
                    finish_reason: FinishReason::Cancelled,
                    usage,
                });
                return Ok(TextLlmOutput {
                    text,
                    thinking: None,
                    tool_calls: Vec::new(),
                    usage,
                    finish_reason: Some(FinishReason::Cancelled),
                });
            }
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let val = (state >> 33) % 100;
            let piece = format!("{val} ");
            text.push_str(&piece);
            emitted += 1;
            if self.behavior.emit_stream {
                on_event(StreamEvent::Token {
                    id: val as u32,
                    text: piece,
                    index: i,
                    channel: Channel::Content,
                });
            }
        }

        let usage = Usage {
            prompt_tokens: 3,
            generated_tokens: emitted,
        };
        let final_text = if self.behavior.reconstruct {
            text.clone()
        } else {
            format!("{text}X") // deliberately not the streamed concatenation
        };
        on_event(StreamEvent::Done {
            finish_reason: FinishReason::Length,
            usage,
        });
        Ok(TextLlmOutput {
            text: final_text,
            thinking: None,
            tool_calls: Vec::new(),
            usage,
            finish_reason: Some(FinishReason::Length),
        })
    }
}

// Register the good stub so the registry check has something to resolve.
fn stub_descriptor() -> TextLlmDescriptor {
    descriptor_with("stub-llm")
}
fn load_stub(_spec: &LoadSpec) -> CoreResult<Box<dyn TextLlm>> {
    Ok(Box::new(good()))
}
fn can_load_stub(_spec: &LoadSpec) -> bool {
    true
}
inventory::submit! {
    TextLlmRegistration {
        descriptor: stub_descriptor,
        load: load_stub,
        can_load: can_load_stub,
        weightless_vision: None,
    }
}

#[test]
fn good_stub_passes_full_conformance() {
    textllm_conformance(|| Box::new(good()), &TextLlmProfile::cheap());
}

#[test]
fn each_check_passes_for_good_stub() {
    let g = good();
    let p: &dyn TextLlm = &g;
    let pr = TextLlmProfile::cheap();
    check_descriptor(p).unwrap();
    check_registry(p).unwrap();
    check_validate(p, &pr).unwrap();
    check_streaming(p, &pr).unwrap();
    check_mid_stream_cancel(p, &pr).unwrap();
    check_cancellation(p, &pr).unwrap();
    check_seed_determinism(p, &pr).unwrap();
    check_multimodal(p, &pr).unwrap();
    check_thinking(p, &pr).unwrap();
}

#[test]
fn thinking_capable_stub_passes_thinking() {
    // An honest provider that advertises supports_thinking: every mode validates, the no-think path
    // produces no reasoning, and the (here empty) reasoning channel stays consistent with output.
    let mut s = good();
    s.descriptor.capabilities.supports_thinking = true;
    check_thinking(&s, &TextLlmProfile::cheap()).unwrap();
}

#[test]
fn dishonest_enable_thinking_is_caught() {
    // supports_thinking=false but validate accepts everything → check_thinking catches the lie.
    let s = stub(
        "dishonest-think",
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    assert!(check_thinking(&s, &TextLlmProfile::cheap()).is_err());
}

#[test]
fn dishonest_validate_is_caught() {
    let mut s = stub(
        "dishonest",
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    s.descriptor.capabilities.max_new_tokens = 10; // a finite cap so the over-cap negative applies
    assert!(check_validate(&s, &TextLlmProfile::cheap()).is_err());
}

#[test]
fn missing_stream_is_caught() {
    let s = stub(
        "nostream",
        Behavior {
            emit_stream: false,
            ..Behavior::good()
        },
    );
    assert!(check_streaming(&s, &TextLlmProfile::cheap()).is_err());
}

#[test]
fn non_reconstructing_stream_is_caught() {
    let s = stub(
        "norecon",
        Behavior {
            reconstruct: false,
            ..Behavior::good()
        },
    );
    assert!(check_streaming(&s, &TextLlmProfile::cheap()).is_err());
}

#[test]
fn untyped_cancel_is_caught() {
    let s = stub(
        "untyped",
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    assert!(check_cancellation(&s, &TextLlmProfile::cheap()).is_err());
}

#[test]
fn ignored_mid_stream_cancel_is_caught() {
    let s = stub(
        "nocancel",
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    assert!(check_mid_stream_cancel(&s, &TextLlmProfile::cheap()).is_err());
}

#[test]
fn ignored_seed_is_caught() {
    let s = stub(
        "noseed",
        Behavior {
            honor_seed: false,
            ..Behavior::good()
        },
    );
    assert!(check_seed_determinism(&s, &TextLlmProfile::cheap()).is_err());
}

#[test]
fn unregistered_id_fails_registry() {
    let s = stub("not-registered", Behavior::good());
    assert!(check_registry(&s).is_err());
}

#[test]
#[should_panic(expected = "conformance FAILED")]
fn aggregator_panics_on_any_failure() {
    // Unregistered + non-streaming -> at least two checks fail -> aggregated panic.
    textllm_conformance(
        || {
            Box::new(stub(
                "not-registered",
                Behavior {
                    emit_stream: false,
                    ..Behavior::good()
                },
            ))
        },
        &TextLlmProfile::cheap(),
    );
}
