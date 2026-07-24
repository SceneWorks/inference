//! The audio-generator testkit verifying itself: a configurable in-crate stub audio generator drives
//! each conformance check, and one deliberately-broken variant per check proves the check fires
//! (sc-12853). The stub is pure-host (no tensor library), so these run on the Linux gen-core lane.

use super::*;
use gen_core::registry::ModelRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    AudioChunk, AudioTrack, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, Modality, ModelDescriptor, Progress,
};
use std::cell::Cell;

const STUB_ID: &str = "testkit_audio_stub";
const UNREG_ID: &str = "testkit_audio_unregistered_stub";

/// Which contract guarantees the stub upholds. `good()` upholds all; each broken-stub test flips
/// exactly one to false and asserts the matching check fails.
#[derive(Clone, Copy)]
struct Behavior {
    honest_validate: bool,
    emit_progress: bool,
    decoding_events: u32,
    honor_cancel: bool,
    typed_cancel: bool,
    deterministic: bool,
    /// Emits `GenerationOutput::Audio` (vs. wrongly emitting an image).
    audio_output: bool,
    /// Emits a well-formed track (vs. a zero-sample-rate/empty one).
    well_formed_track: bool,
    /// Advertises `supports_multi_speaker` (+ a `max_speakers` cap) and accepts a valid script
    /// (sc-12848). `false` ⇒ a single-voice stub whose descriptor leaves `supports_multi_speaker`
    /// unset, so the shared floor rejects a script as the typed Unsupported.
    multi_speaker: bool,
    /// A dishonest multi-speaker stub: advertises `supports_multi_speaker` yet its `validate`
    /// rejects a valid script (drives the multi-speaker broken-stub self-test). Only meaningful
    /// with `multi_speaker: true`.
    reject_valid_script: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            honest_validate: true,
            emit_progress: true,
            decoding_events: 1,
            honor_cancel: true,
            typed_cancel: true,
            deterministic: true,
            audio_output: true,
            well_formed_track: true,
            multi_speaker: false,
            reject_valid_script: false,
        }
    }
}

struct StubAudioGen {
    desc: ModelDescriptor,
    behavior: Behavior,
    runs: Cell<u32>,
}

fn stub_caps() -> Capabilities {
    Capabilities {
        max_count: 4,
        // Audio has no width/height. The weights-free descriptor sweep
        // (`descriptor_conformance_errors`) now exempts `Modality::Audio` from the
        // `1 <= min_size <= max_size` floor (sc-13314), so a valid audio descriptor leaves the bounds
        // at the unused 0 — `registry_sweep_passes_for_the_registered_stub` exercises exactly that,
        // and `validate_request_audio` skips the size range regardless.
        min_size: 0,
        max_size: 0,
        audio_sample_rates: vec![24_000],
        audio_voices: vec!["narrator"],
        audio_languages: vec!["en"],
        max_audio_duration_secs: Some(30.0),
        ..Default::default()
    }
}

fn stub_desc(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        required_components: &[],
        id,
        family: "testkit",
        backend: "stub",
        modality: Modality::Audio,
        capabilities: stub_caps(),
    }
}

impl StubAudioGen {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        let mut desc = stub_desc(id);
        if behavior.multi_speaker {
            desc.capabilities.supports_multi_speaker = true;
            desc.capabilities.max_speakers = Some(4);
        }
        Self {
            desc,
            behavior,
            runs: Cell::new(0),
        }
    }
    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn Generator> {
        Box::new(Self::new(id, behavior))
    }
}

impl Generator for StubAudioGen {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        if !self.behavior.honest_validate {
            return Ok(());
        }
        // The dishonest multi-speaker stub: claims support yet refuses any script.
        if self.behavior.reject_valid_script {
            if let Some(a) = &req.audio {
                if a.script.is_some() {
                    return Err(Error::Unsupported(
                        "stub refuses multi-speaker scripts despite advertising support".into(),
                    ));
                }
            }
        }
        self.desc
            .capabilities
            .validate_request_audio(self.desc.id, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        let total = req.steps.unwrap_or(2);
        let run = self.runs.get();
        self.runs.set(run + 1);
        for i in 1..=total {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                return Err(if self.behavior.typed_cancel {
                    Error::Canceled
                } else {
                    Error::Msg("audio generation cancelled".into())
                });
            }
            if self.behavior.emit_progress {
                on_progress(Progress::Step { current: i, total });
            }
        }
        for _ in 0..self.behavior.decoding_events {
            on_progress(Progress::Decoding);
        }
        // Output pixels/samples depend only on the seed (good) or drift per call (broken).
        let fill = if self.behavior.deterministic {
            req.seed.unwrap_or(0) as f32
        } else {
            run as f32
        };
        if !self.behavior.audio_output {
            // A misconfigured audio model that wrongly returns an image.
            return Ok(GenerationOutput::Images(vec![Image {
                width: 4,
                height: 4,
                pixels: vec![0u8; 4 * 4 * 3],
            }]));
        }
        let track = if self.behavior.well_formed_track {
            AudioTrack {
                samples: vec![fill; 480],
                sample_rate: 24_000,
                channels: 1,
                ..Default::default()
            }
        } else {
            // Zero sample rate + empty samples — the malformed-output class.
            AudioTrack {
                samples: Vec::new(),
                sample_rate: 0,
                channels: 1,
                ..Default::default()
            }
        };
        Ok(GenerationOutput::Audio(track))
    }
}

const STREAM_ID: &str = "testkit_audio_streaming_stub";
const STREAM_TOTAL_SAMPLES: usize = 480;

/// A **streaming** audio stub (sc-12846): advertises `supports_streaming` and overrides
/// [`Generator::generate_streaming`] to emit the deterministic one-shot track as `chunks` contiguous
/// [`AudioChunk`]s. The knobs drive the streaming broken-stub self-tests:
/// `chunks` controls incrementality (1 = "buffers everything, emits one terminal chunk") and
/// `reassemble` controls whether the emitted chunks concatenate back to the track.
struct StreamingStubAudioGen {
    desc: ModelDescriptor,
    chunks: u32,
    reassemble: bool,
    /// Games the count-only gate: emit a zero-length chunk followed by one full-track chunk (2
    /// chunks that reassemble and frame-align, but the whole track arrived in one block).
    empty_then_full: bool,
    honor_cancel: bool,
    runs: Cell<u32>,
}

fn streaming_stub_desc() -> ModelDescriptor {
    let mut desc = stub_desc(STREAM_ID);
    desc.capabilities.supports_streaming = true;
    desc
}

impl StreamingStubAudioGen {
    fn new(chunks: u32, reassemble: bool) -> Self {
        Self {
            desc: streaming_stub_desc(),
            chunks,
            reassemble,
            empty_then_full: false,
            honor_cancel: true,
            runs: Cell::new(0),
        }
    }
    fn boxed(chunks: u32, reassemble: bool) -> Box<dyn Generator> {
        Box::new(Self::new(chunks, reassemble))
    }
    /// A streaming stub that games the `>= 2` count with `[empty chunk, full-track chunk]`.
    fn empty_then_full() -> Self {
        Self {
            empty_then_full: true,
            ..Self::new(2, true)
        }
    }
    /// The deterministic one-shot track: `STREAM_TOTAL_SAMPLES` samples filled from the seed.
    fn track(&self, req: &GenerationRequest) -> AudioTrack {
        let fill = req.seed.unwrap_or(0) as f32;
        AudioTrack {
            samples: vec![fill; STREAM_TOTAL_SAMPLES],
            sample_rate: 24_000,
            channels: 1,
            ..Default::default()
        }
    }
}

impl Generator for StreamingStubAudioGen {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.desc
            .capabilities
            .validate_request_audio(self.desc.id, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        self.runs.set(self.runs.get() + 1);
        let total = req.steps.unwrap_or(2);
        for i in 1..=total {
            if self.honor_cancel && req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            on_progress(Progress::Step { current: i, total });
        }
        on_progress(Progress::Decoding);
        Ok(GenerationOutput::Audio(self.track(req)))
    }

    fn generate_streaming(
        &self,
        req: &GenerationRequest,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        // The streaming path IS the primary path; generate() is its aggregate. Compute the full
        // deterministic track, then partition it into `chunks` contiguous slices emitted as the
        // audio becomes "available".
        let out = self.generate(req, on_progress)?;
        let GenerationOutput::Audio(track) = &out else {
            return Ok(out);
        };
        // The gaming variant: a zero-length chunk then one full-track chunk — 2 chunks that
        // reassemble and frame-align, so it slips past the count-only gate while the whole track
        // actually arrived in a single block.
        if self.empty_then_full {
            on_chunk(AudioChunk {
                samples: Vec::new(),
                sample_rate: track.sample_rate,
                channels: track.channels,
                index: 0,
            });
            on_chunk(AudioChunk {
                samples: track.samples.clone(),
                sample_rate: track.sample_rate,
                channels: track.channels,
                index: 1,
            });
            return Ok(out);
        }
        let n = self.chunks.max(1) as usize;
        let len = track.samples.len();
        let base = len / n;
        let mut start = 0usize;
        for idx in 0..n {
            // Last slice absorbs the remainder so the partition is exact.
            let end = if idx == n - 1 { len } else { start + base };
            let mut samples = track.samples[start..end].to_vec();
            // The broken variant tampers with the first chunk so the concatenation no longer equals
            // the track — exercising the reassembly-law assertion.
            if !self.reassemble && idx == 0 && !samples.is_empty() {
                samples[0] += 1.0;
            }
            on_chunk(AudioChunk {
                samples,
                sample_rate: track.sample_rate,
                channels: track.channels,
                index: idx,
            });
            start = end;
        }
        Ok(out)
    }
}

fn stub_descriptor() -> ModelDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    Ok(StubAudioGen::boxed(STUB_ID, Behavior::good()))
}
const STUB_REGISTRATION: ModelRegistration = ModelRegistration {
    descriptor: stub_descriptor,
    load: stub_load,
    footprint: None,
};

fn registry() -> gen_core::ProviderRegistry {
    gen_core::ProviderRegistryBuilder::new()
        .register_generator(STUB_REGISTRATION)
        .build()
        .expect("stub audio registry should build")
}

fn cheap() -> AudioProfile {
    AudioProfile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    audio_conformance(|| StubAudioGen::boxed(STUB_ID, Behavior::good()), &cheap());
}

#[test]
fn good_stub_passes_every_check_individually() {
    let g = StubAudioGen::new(STUB_ID, Behavior::good());
    check_audio_validate_honesty(&g, &cheap()).unwrap();
    check_audio_output(&g, &cheap()).unwrap();
    check_audio_progress(&g, &cheap()).unwrap();
    check_audio_progress_contract(&g, &cheap()).unwrap();
    check_audio_cancellation(&g, &cheap()).unwrap();
    check_audio_precancellation(&g, &cheap()).unwrap();
    check_audio_seed_determinism(&g, &cheap()).unwrap();
    // The non-streaming stub exercises the additive default `generate_streaming` (one terminal chunk).
    check_audio_streaming(&g, &cheap()).unwrap();
    // The single-turn stub advertises neither multi-turn path, so it must reject both a
    // ConversationHistory request and open_conversation (the additive surface does not perturb it).
    check_multi_turn(&g, &cheap()).unwrap();
    crate::check_registry_roundtrip(&registry(), &g).unwrap();
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    // A rubber-stamp validate accepts an unadvertised voice instead of the typed Unsupported.
    let err = check_audio_validate_honesty(&g, &cheap()).unwrap_err();
    assert!(err.contains("was accepted by validate()"), "got: {err}");
}

#[test]
fn missing_progress_fails_progress_check() {
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    assert!(check_audio_progress(&g, &cheap()).is_err());
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    // The DoD's headline broken-stub: an audio generator that never returns Canceled.
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_cancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("ran to completion"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            typed_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_cancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn ignoring_cancel_fails_precancellation_check() {
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            honor_cancel: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_precancellation(&g, &cheap()).unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn nondeterministic_fails_seed_check() {
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            deterministic: false,
            ..Behavior::good()
        },
    );
    assert!(check_audio_seed_determinism(&g, &cheap()).is_err());
}

#[test]
fn wrong_output_kind_fails_output_check() {
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            audio_output: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_output(&g, &cheap()).unwrap_err();
    assert!(
        err.contains("must emit GenerationOutput::Audio"),
        "got: {err}"
    );
}

#[test]
fn malformed_track_fails_output_check() {
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            well_formed_track: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_output(&g, &cheap()).unwrap_err();
    assert!(err.contains("sample_rate is 0"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let g = StubAudioGen::new(UNREG_ID, Behavior::good());
    assert!(crate::check_registry_roundtrip(&registry(), &g).is_err());
}

#[test]
fn registry_sweep_passes_for_the_registered_stub() {
    crate::registry_conformance(&registry());
}

#[test]
#[should_panic(expected = "audio conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    audio_conformance(
        || {
            StubAudioGen::boxed(
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

// --- Streaming (sc-12846) --------------------------------------------------------------------

#[test]
fn non_streaming_stub_passes_streaming_check_via_default_impl() {
    // A provider that does NOT advertise supports_streaming rides the additive default
    // `generate_streaming`, which emits exactly one terminal chunk equal to the whole track.
    let g = StubAudioGen::new(STUB_ID, Behavior::good());
    assert!(!g.descriptor().capabilities.supports_streaming);
    check_audio_streaming(&g, &cheap()).unwrap();
}

#[test]
fn streaming_stub_passes_streaming_check() {
    // A genuinely-incremental streaming provider: 4 chunks that reassemble to the one-shot track.
    let g = StreamingStubAudioGen::new(4, true);
    assert!(g.descriptor().capabilities.supports_streaming);
    check_audio_streaming(&g, &cheap()).unwrap();
}

#[test]
fn streaming_stub_passes_full_conformance() {
    // A streaming generator must also be a well-behaved one-shot generator (progress, cancel,
    // determinism, output well-formedness) — the whole suite, including the streaming check.
    audio_conformance(|| StreamingStubAudioGen::boxed(4, true), &cheap());
}

#[test]
fn streaming_stub_emitting_everything_at_end_fails_incrementality() {
    // The headline broken-stub: advertises streaming but buffers everything into ONE terminal chunk.
    let g = StreamingStubAudioGen::new(1, true);
    let err = check_audio_streaming(&g, &cheap()).unwrap_err();
    assert!(err.contains("must emit >= 2 chunks"), "got: {err}");
}

#[test]
fn streaming_stub_with_nonreassembling_chunks_fails_reassembly() {
    // Chunks that do not concatenate back to the returned track violate the reassembly law.
    let g = StreamingStubAudioGen::new(4, false);
    let err = check_audio_streaming(&g, &cheap()).unwrap_err();
    assert!(err.contains("reassembly law is violated"), "got: {err}");
}

#[test]
fn streaming_stub_empty_then_full_chunk_fails_incrementality() {
    // Games the >= 2 count with [empty chunk, full-track chunk]: 2 chunks that reassemble and
    // frame-align, but a single chunk carries the entire track — the hardened per-chunk length gate
    // must reject it as non-incremental.
    let g = StreamingStubAudioGen::empty_then_full();
    let err = check_audio_streaming(&g, &cheap()).unwrap_err();
    assert!(err.contains("carries the entire track"), "got: {err}");
}

#[test]
#[should_panic(expected = "audio conformance FAILED")]
fn conformance_panics_on_a_broken_streaming_stub() {
    audio_conformance(|| StreamingStubAudioGen::boxed(1, true), &cheap());
}

// --- Multi-speaker script contract (sc-12848) ------------------------------------------------

#[test]
fn non_multi_speaker_stub_rejects_a_script_as_unsupported() {
    // A single-voice provider (the default stub, supports_multi_speaker == false) must reject a
    // multi-speaker script as the typed Unsupported — it can never silently read only the first
    // segment.
    let g = StubAudioGen::new(STUB_ID, Behavior::good());
    assert!(!g.descriptor().capabilities.supports_multi_speaker);
    check_audio_multi_speaker(&g, &cheap()).unwrap();
}

#[test]
fn multi_speaker_stub_passes_multi_speaker_check() {
    // A provider that advertises supports_multi_speaker accepts + renders a valid 2-speaker script
    // and rejects an over-`max_speakers` script.
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            multi_speaker: true,
            ..Behavior::good()
        },
    );
    assert!(g.descriptor().capabilities.supports_multi_speaker);
    assert_eq!(g.descriptor().capabilities.max_speakers, Some(4));
    check_audio_multi_speaker(&g, &cheap()).unwrap();
}

#[test]
fn multi_speaker_stub_passes_full_conformance() {
    // A multi-speaker generator must also be a well-behaved one-shot generator (the whole suite,
    // now including the multi-speaker check).
    audio_conformance(
        || {
            StubAudioGen::boxed(
                STUB_ID,
                Behavior {
                    multi_speaker: true,
                    ..Behavior::good()
                },
            )
        },
        &cheap(),
    );
}

#[test]
fn multi_speaker_stub_rejecting_a_valid_script_fails_the_check() {
    // The headline multi-speaker broken-stub: advertises supports_multi_speaker but its validate
    // refuses a valid script — the check must catch the dishonest advertisement.
    let g = StubAudioGen::new(
        STUB_ID,
        Behavior {
            multi_speaker: true,
            reject_valid_script: true,
            ..Behavior::good()
        },
    );
    let err = check_audio_multi_speaker(&g, &cheap()).unwrap_err();
    assert!(
        err.contains("advertises supports_multi_speaker but validate() rejected"),
        "got: {err}"
    );
}

#[test]
#[should_panic(expected = "audio conformance FAILED")]
fn conformance_panics_on_a_dishonest_multi_speaker_stub() {
    audio_conformance(
        || {
            StubAudioGen::boxed(
                STUB_ID,
                Behavior {
                    multi_speaker: true,
                    reject_valid_script: true,
                    ..Behavior::good()
                },
            )
        },
        &cheap(),
    );
}

// --- Video→audio (Foley) sync contract (sc-13436) --------------------------------------------

const VIDEO_SYNC_ID: &str = "testkit_audio_video_sync_stub";

/// A **video→audio (Foley) stub** (sc-13436): a `Modality::Audio` generator that advertises the
/// `VideoSync` conditioning kind and renders a non-silent track whose fill derives from the clip's
/// frame pixels + the seed, with a length matching the clip (`frames / fps`). The knobs drive the
/// broken-stub self-tests: `silent` emits an all-zero track (ignores the frames, emits silence) and
/// `ignore_frames` renders from the seed alone (advertises VideoSync but never reads the pixels).
struct VideoSyncStubAudioGen {
    desc: ModelDescriptor,
    silent: bool,
    ignore_frames: bool,
    runs: Cell<u32>,
}

fn video_sync_stub_desc() -> ModelDescriptor {
    let mut desc = stub_desc(VIDEO_SYNC_ID);
    // A Foley model opts in by advertising VideoSync (in addition to any audio surface it carries).
    desc.capabilities.conditioning = vec![ConditioningKind::VideoSync];
    desc
}

impl VideoSyncStubAudioGen {
    fn new(silent: bool, ignore_frames: bool) -> Self {
        Self {
            desc: video_sync_stub_desc(),
            silent,
            ignore_frames,
            runs: Cell::new(0),
        }
    }
    fn boxed(silent: bool, ignore_frames: bool) -> Box<dyn Generator> {
        Box::new(Self::new(silent, ignore_frames))
    }

    /// The deterministic soundtrack: `frames / fps` seconds at 24 kHz mono, DC-filled from a value
    /// derived from the seed and (unless `ignore_frames`) the clip's pixels.
    fn track(&self, req: &GenerationRequest) -> AudioTrack {
        let n_frames = req
            .conditioning
            .iter()
            .find_map(|c| match c {
                Conditioning::VideoSync { frames } => Some(frames.len()),
                _ => None,
            })
            .unwrap_or(0);
        let fps = req.fps.unwrap_or(8).max(1);
        let sample_rate = 24_000u32;
        let n_samples =
            (((n_frames as f32 / fps as f32) * sample_rate as f32).round() as usize).max(1);

        let fill = if self.silent {
            0.0
        } else {
            let mut acc = req.seed.unwrap_or(0);
            if !self.ignore_frames {
                for c in &req.conditioning {
                    if let Conditioning::VideoSync { frames } = c {
                        for f in frames {
                            for &p in &f.pixels {
                                acc = acc.wrapping_mul(31).wrapping_add(p as u64);
                            }
                        }
                    }
                }
            }
            // A distinct, non-zero, finite DC level.
            (acc % 1_000_003) as f32 * 1e-3 + 1.0
        };

        AudioTrack {
            samples: vec![fill; n_samples],
            sample_rate,
            channels: 1,
            ..Default::default()
        }
    }
}

impl Generator for VideoSyncStubAudioGen {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.desc
            .capabilities
            .validate_request_audio(self.desc.id, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        self.runs.set(self.runs.get() + 1);
        let total = req.steps.unwrap_or(2);
        for i in 1..=total {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            on_progress(Progress::Step { current: i, total });
        }
        on_progress(Progress::Decoding);
        Ok(GenerationOutput::Audio(self.track(req)))
    }
}

#[test]
fn non_video_sync_stub_rejects_a_clip_as_unsupported() {
    // A model that does not advertise VideoSync (the default audio stub) must reject a Foley clip as
    // the typed Unsupported — it can never silently ignore the frames and emit unconditioned audio.
    let g = StubAudioGen::new(STUB_ID, Behavior::good());
    assert!(!g
        .descriptor()
        .capabilities
        .accepts(ConditioningKind::VideoSync));
    check_video_to_audio(&g, &cheap()).unwrap();
}

#[test]
fn video_sync_stub_passes_video_to_audio_check() {
    // The honest Foley stub: advertises VideoSync and renders a non-silent, clip-length,
    // reproducible, frame-dependent soundtrack.
    let g = VideoSyncStubAudioGen::new(false, false);
    assert!(g
        .descriptor()
        .capabilities
        .accepts(ConditioningKind::VideoSync));
    check_video_to_audio(&g, &cheap()).unwrap();
}

#[test]
fn video_sync_stub_passes_full_conformance() {
    // A Foley generator must also be a well-behaved one-shot audio generator (progress, cancel,
    // determinism, output well-formedness) — the whole suite, including the video→audio check.
    audio_conformance(|| VideoSyncStubAudioGen::boxed(false, false), &cheap());
}

#[test]
fn silent_video_sync_stub_fails_the_check() {
    // The headline dishonest stub: advertises VideoSync but renders silence (ignores the frames).
    let g = VideoSyncStubAudioGen::new(true, false);
    let err = check_video_to_audio(&g, &cheap()).unwrap_err();
    assert!(err.contains("is silent"), "got: {err}");
}

#[test]
fn frame_ignoring_video_sync_stub_fails_the_check() {
    // A subtler dishonest stub: non-silent and reproducible, but its audio derives from the seed
    // alone — two different clips render byte-identical audio. The frame-dependence assertion catches
    // it.
    let g = VideoSyncStubAudioGen::new(false, true);
    let err = check_video_to_audio(&g, &cheap()).unwrap_err();
    assert!(
        err.contains("appears to ignore the VideoSync frames"),
        "got: {err}"
    );
}

#[test]
#[should_panic(expected = "audio conformance FAILED")]
fn conformance_panics_on_a_silent_video_sync_stub() {
    audio_conformance(|| VideoSyncStubAudioGen::boxed(true, false), &cheap());
}

// --- Multi-turn conversational contract (sc-14150) --------------------------------------------

use gen_core::{ConversationRole, ConversationSession, ConversationTurn};

const MT_ID: &str = "testkit_audio_multi_turn_stub";
const MT_SAMPLE_RATE: u32 = 24_000;

/// Which multi-turn paths the stub advertises + which contract it (honestly or dishonestly) upholds.
#[derive(Clone, Copy)]
struct MtBehavior {
    /// Advertise `supports_conversation_history` (path A) + the `ConversationHistory` kind.
    history: bool,
    /// Advertise `supports_conversation_session` (path B).
    session: bool,
    /// Broken: synthesize each turn context-free (ignore the prior turns) — fails conditioned-on-prior.
    ignore_history: bool,
    /// Broken: the session (path B) perturbs its output so it diverges from the batch render (path A)
    /// — fails the A≡B equivalence law.
    session_diverges: bool,
}

impl MtBehavior {
    fn both() -> Self {
        Self {
            history: true,
            session: true,
            ignore_history: false,
            session_diverges: false,
        }
    }
}

/// FNV-1a-style fold, so the running conversation digest depends on every prior turn's role, text,
/// and audio (context or generated) — the "turn N is conditioned on turns 1..N-1" content.
fn mt_fold(acc: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *acc ^= b as u64;
        *acc = acc.wrapping_mul(0x0000_0100_0000_01B3);
    }
}

/// The synthesized reply for a turn: a DC-filled 480-sample mono track whose level derives from the
/// running context digest `ctx` (the prior turns), this turn's `text`, and the `seed` — so a later
/// turn genuinely depends on the earlier ones.
fn mt_synth(ctx: u64, text: &str, seed: u64) -> AudioTrack {
    let mut d = ctx;
    mt_fold(&mut d, text.as_bytes());
    mt_fold(&mut d, &seed.to_le_bytes());
    let fill = (d % 1_000_003) as f32 * 1e-3 + 1.0;
    AudioTrack {
        samples: vec![fill; 480],
        sample_rate: MT_SAMPLE_RATE,
        channels: 1,
        ..Default::default()
    }
}

/// The single per-turn core shared by path A (`generate`) and path B (session `step`): fold this
/// turn's role+text into `ctx`, then either fold a context turn's audio (returns `None`) or
/// synthesize an assistant reply conditioned on `ctx`, fold the reply into `ctx`, and return it. A
/// and B calling this identically is what makes the A≡B render byte-identical.
fn mt_process(
    ctx: &mut u64,
    turn: &ConversationTurn,
    seed: u64,
    ignore_history: bool,
) -> Option<AudioTrack> {
    mt_fold(
        ctx,
        &[match turn.role {
            ConversationRole::User => 1,
            ConversationRole::Assistant => 2,
        }],
    );
    mt_fold(ctx, turn.text.as_bytes());
    match &turn.audio {
        Some(track) => {
            for &s in &track.samples {
                mt_fold(ctx, &s.to_le_bytes());
            }
            None
        }
        None => {
            // The honest stub conditions on the running digest; the broken `ignore_history` stub
            // synthesizes from a fixed base (each turn context-free).
            let cond = if ignore_history { 0 } else { *ctx };
            let reply = mt_synth(cond, &turn.text, seed);
            for &s in &reply.samples {
                mt_fold(ctx, &s.to_le_bytes());
            }
            Some(reply)
        }
    }
}

struct MultiTurnStubAudioGen {
    desc: ModelDescriptor,
    behavior: MtBehavior,
}

fn mt_desc(behavior: MtBehavior) -> ModelDescriptor {
    let mut desc = stub_desc(MT_ID);
    if behavior.history {
        desc.capabilities.conditioning = vec![ConditioningKind::ConversationHistory];
        desc.capabilities.supports_conversation_history = true;
    }
    desc.capabilities.supports_conversation_session = behavior.session;
    desc
}

impl MultiTurnStubAudioGen {
    fn new(behavior: MtBehavior) -> Self {
        Self {
            desc: mt_desc(behavior),
            behavior,
        }
    }
    fn boxed(behavior: MtBehavior) -> Box<dyn Generator> {
        Box::new(Self::new(behavior))
    }
}

impl Generator for MultiTurnStubAudioGen {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.desc
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.desc
            .capabilities
            .validate_request_audio(self.desc.id, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let seed = req.seed.unwrap_or(0);
        // Emit step progress and honor cancellation, so a conversational generator is also a
        // well-behaved one-shot audio generator (the shared progress/cancel conformance).
        let total = req.steps.unwrap_or(2);
        for i in 1..=total {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            on_progress(Progress::Step { current: i, total });
        }
        on_progress(Progress::Decoding);
        // Path A: render the whole conversation, keeping the digest warm across turns (never
        // round-tripping generated audio) and concatenating the assistant replies.
        if let Some(Conditioning::ConversationHistory { turns }) = req
            .conditioning
            .iter()
            .find(|c| matches!(c, Conditioning::ConversationHistory { .. }))
        {
            let mut ctx = seed;
            let mut samples = Vec::new();
            for turn in turns {
                if let Some(reply) = mt_process(&mut ctx, turn, seed, self.behavior.ignore_history)
                {
                    samples.extend(reply.samples);
                }
            }
            return Ok(GenerationOutput::Audio(AudioTrack {
                samples,
                sample_rate: MT_SAMPLE_RATE,
                channels: 1,
                ..Default::default()
            }));
        }
        // Single-shot: a deterministic seed-filled track, so this stub is also a well-behaved
        // one-shot audio generator (it passes the rest of the suite).
        Ok(GenerationOutput::Audio(AudioTrack {
            samples: vec![seed as f32; 480],
            sample_rate: MT_SAMPLE_RATE,
            channels: 1,
            ..Default::default()
        }))
    }

    fn open_conversation(
        &self,
        req: &GenerationRequest,
    ) -> gen_core::Result<Box<dyn ConversationSession + '_>> {
        if !self.behavior.session {
            return Err(Error::Unsupported(format!(
                "{}: stateful conversational sessions are not supported",
                self.desc.id
            )));
        }
        Ok(Box::new(MtSession {
            ctx: req.seed.unwrap_or(0),
            seed: req.seed.unwrap_or(0),
            behavior: self.behavior,
        }))
    }
}

/// Path B: the stateful session. It keeps the running digest warm across `step`s — the exact same
/// per-turn `mt_process` path A runs, so the two are byte-identical (unless `session_diverges`).
struct MtSession {
    ctx: u64,
    seed: u64,
    behavior: MtBehavior,
}

impl ConversationSession for MtSession {
    fn step(
        &mut self,
        turn: &ConversationTurn,
        on_chunk: &mut dyn FnMut(AudioChunk),
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<AudioTrack> {
        on_progress(Progress::Decoding);
        match mt_process(&mut self.ctx, turn, self.seed, self.behavior.ignore_history) {
            Some(mut reply) => {
                // Broken variant: perturb the session output so path B diverges from path A.
                if self.behavior.session_diverges {
                    reply.samples[0] += 1.0;
                }
                // Stream the reply as two contiguous chunks (exercises the per-turn reassembly law).
                let mid = reply.samples.len() / 2;
                on_chunk(AudioChunk {
                    samples: reply.samples[..mid].to_vec(),
                    sample_rate: reply.sample_rate,
                    channels: reply.channels,
                    index: 0,
                });
                on_chunk(AudioChunk {
                    samples: reply.samples[mid..].to_vec(),
                    sample_rate: reply.sample_rate,
                    channels: reply.channels,
                    index: 1,
                });
                Ok(reply)
            }
            // A context turn is folded into the session and echoes its provided audio.
            None => Ok(turn.audio.clone().unwrap_or_default()),
        }
    }
}

#[test]
fn good_multi_turn_stub_passes_the_check() {
    // Honest stub advertising BOTH paths: A≡B byte-identical, conditioned-on-prior, deterministic.
    let g = MultiTurnStubAudioGen::new(MtBehavior::both());
    check_multi_turn(&g, &cheap()).unwrap();
}

#[test]
fn good_multi_turn_stub_passes_full_conformance() {
    // A conversational generator must also be a well-behaved one-shot audio generator.
    audio_conformance(
        || MultiTurnStubAudioGen::boxed(MtBehavior::both()),
        &cheap(),
    );
}

#[test]
fn history_only_multi_turn_stub_passes_the_check() {
    // Advertises only path A: path A works, and open_conversation is rejected as typed Unsupported.
    let g = MultiTurnStubAudioGen::new(MtBehavior {
        session: false,
        ..MtBehavior::both()
    });
    check_multi_turn(&g, &cheap()).unwrap();
}

#[test]
fn session_only_multi_turn_stub_passes_the_check() {
    // Advertises only path B: path B works, and a ConversationHistory request is rejected as
    // Unsupported (path A gated by the conditioning allowlist).
    let g = MultiTurnStubAudioGen::new(MtBehavior {
        history: false,
        ..MtBehavior::both()
    });
    check_multi_turn(&g, &cheap()).unwrap();
}

#[test]
fn history_ignoring_multi_turn_stub_fails_conditioned_on_prior() {
    // The headline dishonest stub: renders each turn context-free, so a turn WITH prior context is
    // byte-identical to it rendered ALONE.
    let g = MultiTurnStubAudioGen::new(MtBehavior {
        ignore_history: true,
        ..MtBehavior::both()
    });
    let err = check_multi_turn(&g, &cheap()).unwrap_err();
    assert!(
        err.contains("ignore the conversation history"),
        "got: {err}"
    );
}

#[test]
fn session_diverging_multi_turn_stub_fails_a_equiv_b() {
    // A subtler dishonest stub: honest per-path, but the session render diverges from the batch
    // render — the A≡B equivalence law catches it.
    let g = MultiTurnStubAudioGen::new(MtBehavior {
        session_diverges: true,
        ..MtBehavior::both()
    });
    let err = check_multi_turn(&g, &cheap()).unwrap_err();
    assert!(err.contains("A≡B equivalence law"), "got: {err}");
}

#[test]
fn single_turn_stub_rejects_conversation_and_session() {
    // The default (neither-path) stub must reject a ConversationHistory request AND open_conversation
    // as the typed Unsupported — the additive contract does not perturb single-turn providers.
    let g = StubAudioGen::new(STUB_ID, Behavior::good());
    check_multi_turn(&g, &cheap()).unwrap();
    let req = GenerationRequest {
        conditioning: vec![Conditioning::ConversationHistory {
            turns: vec![ConversationTurn {
                role: ConversationRole::Assistant,
                text: "hi".into(),
                audio: None,
            }],
        }],
        ..Default::default()
    };
    assert!(matches!(g.validate(&req), Err(Error::Unsupported(_))));
    assert!(matches!(
        g.open_conversation(&req),
        Err(Error::Unsupported(_))
    ));
}

#[test]
#[should_panic(expected = "audio conformance FAILED")]
fn conformance_panics_on_a_history_ignoring_multi_turn_stub() {
    audio_conformance(
        || {
            MultiTurnStubAudioGen::boxed(MtBehavior {
                ignore_history: true,
                ..MtBehavior::both()
            })
        },
        &cheap(),
    );
}
