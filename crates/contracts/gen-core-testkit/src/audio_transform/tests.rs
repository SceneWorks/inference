//! The audio-transform testkit verifying itself: a configurable in-crate stub drives each check
//! across the three kinds, and deliberately-broken variants prove the checks fire (sc-12853).
//! Pure-host — runs on the Linux gen-core lane.

use super::*;
use gen_core::registry::AudioTransformRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    AudioTarget, AudioTrack, AudioTransform, AudioTransformCapabilities, AudioTransformDescriptor,
    AudioTransformKind, AudioTransformRequest, Progress,
};

const STUB_ID: &str = "testkit_audio_transform_stub";
const UNREG_ID: &str = "testkit_audio_transform_unregistered_stub";

struct StubAudioTransform {
    desc: AudioTransformDescriptor,
    /// How many tracks `apply` actually returns (independent of the advertised kind, so a broken
    /// stub can under/over-produce).
    output_tracks: usize,
}

fn desc(id: &'static str, kind: AudioTransformKind, stem_count: u16) -> AudioTransformDescriptor {
    AudioTransformDescriptor {
        id,
        family: "audio",
        backend: "stub",
        capabilities: AudioTransformCapabilities {
            kind,
            stem_count,
            supports_resample: kind == AudioTransformKind::SuperResolution,
            ..Default::default()
        },
    }
}

impl AudioTransform for StubAudioTransform {
    fn descriptor(&self) -> &AudioTransformDescriptor {
        &self.desc
    }
    fn validate(&self, _req: &AudioTransformRequest) -> gen_core::Result<()> {
        Ok(())
    }
    fn apply(
        &self,
        req: &AudioTransformRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<Vec<AudioTrack>> {
        on_progress(Progress::Step {
            current: 1,
            total: 1,
        });
        let rate = match req.target {
            AudioTarget::Preserve => req.audio.sample_rate,
            AudioTarget::SampleRate(r) => r,
        };
        Ok(vec![
            AudioTrack {
                sample_rate: rate,
                ..req.audio.clone()
            };
            self.output_tracks
        ])
    }
}

fn boxed(
    id: &'static str,
    kind: AudioTransformKind,
    stem_count: u16,
    output_tracks: usize,
) -> Box<dyn AudioTransform> {
    Box::new(StubAudioTransform {
        desc: desc(id, kind, stem_count),
        output_tracks,
    })
}

/// A well-formed voice converter (1 in, 1 out).
fn good_vc() -> Box<dyn AudioTransform> {
    boxed(STUB_ID, AudioTransformKind::VoiceConversion, 0, 1)
}
/// A well-formed 4-stem separator.
fn good_stems() -> Box<dyn AudioTransform> {
    boxed(STUB_ID, AudioTransformKind::StemSeparation, 4, 4)
}
/// A well-formed super-resolver (1 in, 1 out, resample).
fn good_sr() -> Box<dyn AudioTransform> {
    boxed(STUB_ID, AudioTransformKind::SuperResolution, 0, 1)
}

fn registry(reg_desc: fn() -> AudioTransformDescriptor) -> gen_core::ProviderRegistry {
    fn load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn AudioTransform>> {
        Ok(good_vc())
    }
    gen_core::ProviderRegistryBuilder::new()
        .register_audio_transform(AudioTransformRegistration {
            descriptor: reg_desc,
            load,
        })
        .build()
        .expect("stub audio-transform registry should build")
}

fn stub_vc_descriptor() -> AudioTransformDescriptor {
    desc(STUB_ID, AudioTransformKind::VoiceConversion, 0)
}

fn cheap() -> AudioTransformProfile {
    AudioTransformProfile::cheap()
}

#[test]
fn good_stubs_pass_full_conformance() {
    audio_transform_conformance(good_vc, &cheap());
    audio_transform_conformance(good_stems, &cheap());
    audio_transform_conformance(good_sr, &cheap());
}

#[test]
fn good_vc_passes_every_check_individually() {
    let t = good_vc();
    let t: &dyn AudioTransform = t.as_ref();
    check_audio_transform_coherence(t).unwrap();
    check_audio_transform_validate(t, &cheap()).unwrap();
    check_audio_transform_cardinality(t, &cheap()).unwrap();
    check_audio_transform_registry(&registry(stub_vc_descriptor), t).unwrap();
}

#[test]
fn stem_separator_cardinality_is_stem_count() {
    // The separator returns exactly stem_count (4) tracks.
    let t = good_stems();
    check_audio_transform_cardinality(t.as_ref(), &cheap()).unwrap();
}

#[test]
fn wrong_cardinality_fails_cardinality_check() {
    // A separator that returns only 1 track instead of its advertised 4 stems.
    let t = boxed(STUB_ID, AudioTransformKind::StemSeparation, 4, 1);
    let err = check_audio_transform_cardinality(t.as_ref(), &cheap()).unwrap_err();
    assert!(
        err.contains("returned 1 track(s), expected 4"),
        "got: {err}"
    );
}

#[test]
fn voice_converter_overproducing_fails_cardinality_check() {
    // A single-output converter that wrongly returns 2 tracks.
    let t = boxed(STUB_ID, AudioTransformKind::VoiceConversion, 0, 2);
    let err = check_audio_transform_cardinality(t.as_ref(), &cheap()).unwrap_err();
    assert!(
        err.contains("returned 2 track(s), expected 1"),
        "got: {err}"
    );
}

#[test]
fn incoherent_stem_count_fails_coherence_check() {
    // StemSeparation advertising stem_count 1 (< 2).
    let t = boxed(STUB_ID, AudioTransformKind::StemSeparation, 1, 1);
    let err = check_audio_transform_coherence(t.as_ref()).unwrap_err();
    assert!(err.contains("must advertise stem_count >= 2"), "got: {err}");
}

#[test]
fn single_output_with_stems_fails_coherence_check() {
    // VoiceConversion advertising a non-zero stem_count.
    let t = boxed(STUB_ID, AudioTransformKind::VoiceConversion, 3, 1);
    let err = check_audio_transform_coherence(t.as_ref()).unwrap_err();
    assert!(err.contains("must advertise stem_count 0"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let t = boxed(UNREG_ID, AudioTransformKind::VoiceConversion, 0, 1);
    assert!(check_audio_transform_registry(&registry(stub_vc_descriptor), t.as_ref()).is_err());
}

#[test]
#[should_panic(expected = "audio-transform conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    audio_transform_conformance(
        || boxed(STUB_ID, AudioTransformKind::StemSeparation, 4, 1),
        &cheap(),
    );
}
