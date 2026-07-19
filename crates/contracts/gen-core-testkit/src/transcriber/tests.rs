//! The transcriber testkit verifying itself: a configurable in-crate stub drives each check, and one
//! deliberately-broken variant per check proves the check fires (sc-12853). Pure-host — runs on the
//! Linux gen-core lane.

use super::*;
use gen_core::registry::TranscriberRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    Progress, TranscribeCapabilities, TranscribeRequest, Transcriber, TranscriberDescriptor,
    TranscriptOutput, TranscriptSegment,
};

const STUB_ID: &str = "testkit_transcriber_stub";
const UNREG_ID: &str = "testkit_transcriber_unregistered_stub";

/// Which contract guarantees the stub upholds.
#[derive(Clone, Copy)]
struct Behavior {
    /// `validate()` enforces the advertised capability surface (vs. rubber-stamping).
    honest_validate: bool,
    /// Checks `CancelFlag` before inference and bails.
    honor_cancel: bool,
    /// On cancel, returns the typed `Error::Canceled` (vs. a stringified `Error::Msg`).
    typed_cancel: bool,
    /// Emits `Progress::Step` events during decoding.
    emit_progress: bool,
    /// Emits well-ordered segment timestamps (vs. end-before-start garbage).
    well_ordered_segments: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            honest_validate: true,
            honor_cancel: true,
            typed_cancel: true,
            emit_progress: true,
            well_ordered_segments: true,
        }
    }
}

struct StubTranscriber {
    desc: TranscriberDescriptor,
    behavior: Behavior,
}

fn stub_caps() -> TranscribeCapabilities {
    TranscribeCapabilities {
        languages: vec!["en", "fr"],
        // Translate unsupported so the capability-gap negative fires.
        supports_translate: false,
        supports_segment_timestamps: true,
        supports_word_timestamps: false,
        max_audio_seconds: 30.0,
        max_new_tokens: 448,
        mac_only: false,
    }
}

fn stub_desc(id: &'static str) -> TranscriberDescriptor {
    TranscriberDescriptor {
        id,
        family: "asr",
        backend: "stub",
        capabilities: stub_caps(),
    }
}

impl StubTranscriber {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
        }
    }
    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn Transcriber> {
        Box::new(Self::new(id, behavior))
    }
}

impl Transcriber for StubTranscriber {
    fn descriptor(&self) -> &TranscriberDescriptor {
        &self.desc
    }
    fn validate(&self, req: &TranscribeRequest) -> gen_core::Result<()> {
        if self.behavior.honest_validate {
            self.desc.capabilities.validate_request(self.desc.id, req)
        } else {
            Ok(())
        }
    }
    fn transcribe(
        &self,
        req: &TranscribeRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<TranscriptOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        if self.behavior.honor_cancel && req.cancel.is_cancelled() {
            return Err(if self.behavior.typed_cancel {
                gen_core::Error::Canceled
            } else {
                gen_core::Error::Msg("stub transcriber: cancelled".into())
            });
        }
        if self.behavior.emit_progress {
            on_progress(Progress::Step {
                current: 1,
                total: 2,
            });
            on_progress(Progress::Step {
                current: 2,
                total: 2,
            });
        }
        let (start, end) = if self.behavior.well_ordered_segments {
            (0.0, 1.0)
        } else {
            (1.0, 0.0) // end precedes start — the garbled-timestamp class
        };
        Ok(TranscriptOutput {
            text: "a stub transcript".to_owned(),
            segments: vec![TranscriptSegment {
                text: "a stub transcript".to_owned(),
                start,
                end,
                words: Vec::new(),
            }],
            language: Some("en".to_owned()),
            generated_tokens: Some(4),
            finish_reason: Some(gen_core::TranscribeFinishReason::StopToken),
        })
    }
}

fn stub_descriptor() -> TranscriberDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Transcriber>> {
    Ok(StubTranscriber::boxed(STUB_ID, Behavior::good()))
}
const STUB_REGISTRATION: TranscriberRegistration = TranscriberRegistration {
    descriptor: stub_descriptor,
    load: stub_load,
};

fn registry() -> gen_core::ProviderRegistry {
    gen_core::ProviderRegistryBuilder::new()
        .register_transcriber(STUB_REGISTRATION)
        .build()
        .expect("stub transcriber registry should build")
}

fn cheap() -> TranscriberProfile {
    TranscriberProfile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    transcriber_conformance(
        || StubTranscriber::boxed(STUB_ID, Behavior::good()),
        &cheap(),
    );
}

#[test]
fn good_stub_passes_every_check_individually() {
    let t = StubTranscriber::new(STUB_ID, Behavior::good());
    check_transcriber_validate(&t, &cheap()).unwrap();
    check_transcriber_progress(&t, &cheap()).unwrap();
    check_transcriber_output(&t, &cheap()).unwrap();
    check_transcriber_cancellation(&t, &cheap()).unwrap();
    check_transcriber_registry(&registry(), &t).unwrap();
}

#[test]
fn missing_progress_fails_progress_check() {
    // Long-running ASR must report progress; a transcriber that emits none fails the check.
    let t = StubTranscriber::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    let err = check_transcriber_progress(&t, &cheap()).unwrap_err();
    assert!(err.contains("no Progress::Step events"), "got: {err}");
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let t = StubTranscriber::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    let err = check_transcriber_validate(&t, &cheap()).unwrap_err();
    assert!(err.contains("was accepted by validate()"), "got: {err}");
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    // The DoD's transcriber broken-stub: never returns Canceled.
    let t = StubTranscriber::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_transcriber_cancellation(&t, &cheap()).unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    let t = StubTranscriber::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_transcriber_cancellation(&t, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn garbled_timestamps_fail_output_check() {
    let t = StubTranscriber::new(
        STUB_ID,
        Behavior {
            well_ordered_segments: false,
            ..Behavior::good()
        },
    );
    let err = check_transcriber_output(&t, &cheap()).unwrap_err();
    assert!(err.contains("precedes start"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let t = StubTranscriber::new(UNREG_ID, Behavior::good());
    assert!(check_transcriber_registry(&registry(), &t).is_err());
}

#[test]
#[should_panic(expected = "transcriber conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    transcriber_conformance(
        || {
            StubTranscriber::boxed(
                STUB_ID,
                Behavior {
                    honor_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &cheap(),
    );
}
