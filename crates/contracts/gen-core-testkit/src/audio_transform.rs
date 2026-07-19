//! Contract conformance for [`gen_core::AudioTransform`] providers (sc-12839) — non-prompt
//! audio→audio / audio→stems: voice conversion (`candle-audio-openvoice`), stem separation,
//! super-resolution. The audio sibling of the image [`Transform`](gen_core::Transform). The defining
//! guarantee this suite enforces is **output cardinality by kind**: `apply` returns exactly one track
//! for the single-output kinds ([`VoiceConversion`](gen_core::AudioTransformKind::VoiceConversion) /
//! [`SuperResolution`](gen_core::AudioTransformKind::SuperResolution)) and exactly
//! [`stem_count`](gen_core::AudioTransformCapabilities::stem_count) tracks for
//! [`StemSeparation`](gen_core::AudioTransformKind::StemSeparation) — plus the kind/stem_count
//! coherence the descriptor must satisfy, mirrored here per-provider.

use gen_core::{
    AudioTarget, AudioTrack, AudioTransform, AudioTransformKind, AudioTransformRequest,
};

/// Parameters for an audio-transform conformance run — one in-range source clip `apply` transforms.
/// The request shape (target rate, whether a tone-color reference is attached) is derived from the
/// descriptor's kind inside the checks, so a single profile drives all three kinds.
#[derive(Clone, Debug)]
pub struct AudioTransformProfile {
    /// A valid, short mono source clip.
    pub audio: AudioTrack,
    /// The output rate a [`SuperResolution`](gen_core::AudioTransformKind::SuperResolution) provider
    /// is asked to restore to.
    pub super_resolution_target: u32,
}

impl Default for AudioTransformProfile {
    fn default() -> Self {
        Self {
            audio: AudioTrack {
                samples: vec![0.01; 8_000],
                sample_rate: 16_000,
                channels: 1,
                ..Default::default()
            },
            super_resolution_target: 48_000,
        }
    }
}

impl AudioTransformProfile {
    pub fn cheap() -> Self {
        Self::default()
    }
}

/// The in-capability request for this transform's kind: a super-resolver gets an
/// [`AudioTarget::SampleRate`] target, a reference-based voice converter gets a tone-color reference
/// clip, everything else the bare source clip.
fn base_request(t: &dyn AudioTransform, profile: &AudioTransformProfile) -> AudioTransformRequest {
    let caps = &t.descriptor().capabilities;
    let mut req = AudioTransformRequest {
        audio: profile.audio.clone(),
        ..Default::default()
    };
    match caps.kind {
        AudioTransformKind::SuperResolution if caps.supports_resample => {
            req.target = AudioTarget::SampleRate(profile.super_resolution_target);
        }
        _ => {}
    }
    req
}

/// **Output cardinality by kind + well-formedness.** `apply` returns the number of tracks the kind
/// requires (1 for voice conversion / super-resolution, `stem_count` for stem separation), and every
/// returned track is well-formed (positive sample rate/channels, non-empty, finite PCM). This is the
/// contract the single-signature `apply(...) -> Vec<AudioTrack>` cannot express in its type.
pub fn check_audio_transform_cardinality(
    t: &dyn AudioTransform,
    profile: &AudioTransformProfile,
) -> Result<(), String> {
    let desc = t.descriptor();
    let id = desc.id;
    let caps = &desc.capabilities;
    let expected = match caps.kind {
        AudioTransformKind::VoiceConversion | AudioTransformKind::SuperResolution => 1,
        AudioTransformKind::StemSeparation => caps.stem_count as usize,
    };
    let req = base_request(t, profile);
    let out = t
        .apply(&req, &mut |_| {})
        .map_err(|e| format!("cardinality[{id}]: apply() failed on the cheap request: {e}"))?;
    if out.len() != expected {
        return Err(format!(
            "cardinality[{id}]: {:?} apply() returned {} track(s), expected {expected} \
             (VoiceConversion/SuperResolution = 1, StemSeparation = stem_count {})",
            caps.kind,
            out.len(),
            caps.stem_count
        ));
    }
    for (i, track) in out.iter().enumerate() {
        crate::audio_generator::validate_track(id, "cardinality", track)
            .map_err(|e| format!("{e} (output track {i})"))?;
    }
    Ok(())
}

/// **Kind / stem_count coherence.** Mirrors the descriptor conformance sweep for a single provider:
/// [`StemSeparation`](gen_core::AudioTransformKind::StemSeparation) must advertise `stem_count >= 2`,
/// and the single-output kinds must advertise `stem_count == 0` — a descriptor whose shape and count
/// disagree is a mis-registration a consumer would trip on.
pub fn check_audio_transform_coherence(t: &dyn AudioTransform) -> Result<(), String> {
    let desc = t.descriptor();
    let id = desc.id;
    let caps = &desc.capabilities;
    match caps.kind {
        AudioTransformKind::StemSeparation if caps.stem_count < 2 => Err(format!(
            "coherence[{id}]: StemSeparation must advertise stem_count >= 2, got {}",
            caps.stem_count
        )),
        AudioTransformKind::VoiceConversion | AudioTransformKind::SuperResolution
            if caps.stem_count != 0 =>
        {
            Err(format!(
                "coherence[{id}]: {:?} is single-output and must advertise stem_count 0, got {}",
                caps.kind, caps.stem_count
            ))
        }
        _ => Ok(()),
    }
}

/// **Validate honesty.** The in-capability request for this transform's kind is accepted by
/// `validate` before any expensive work.
pub fn check_audio_transform_validate(
    t: &dyn AudioTransform,
    profile: &AudioTransformProfile,
) -> Result<(), String> {
    let id = t.descriptor().id;
    t.validate(&base_request(t, profile)).map_err(|e| {
        format!("validate-honesty[{id}]: the in-capability cheap request was rejected by validate(): {e}")
    })
}

/// **Registry round-trip.** The transform's descriptor `id` is present in the explicit registry
/// supplied by the caller.
pub fn check_audio_transform_registry(
    registry: &gen_core::ProviderRegistry,
    t: &dyn AudioTransform,
) -> Result<(), String> {
    let id = t.descriptor().id;
    if registry
        .audio_transforms()
        .any(|registration| (registration.descriptor)().id == id)
    {
        Ok(())
    } else {
        Err(format!(
            "registry[{id}]: descriptor id not found in the explicit provider registry (gen-core {})",
            gen_core::VERSION
        ))
    }
}

/// Run the full audio-transform conformance suite against a freshly-`make`d transform. Panics with
/// every failure aggregated.
pub fn audio_transform_conformance(
    make: impl Fn() -> Box<dyn AudioTransform>,
    profile: &AudioTransformProfile,
) {
    let t = make();
    let t: &dyn AudioTransform = t.as_ref();

    let failures: Vec<String> = [
        check_audio_transform_coherence(t),
        check_audio_transform_validate(t, profile),
        check_audio_transform_cardinality(t, profile),
    ]
    .into_iter()
    .filter_map(|r| r.err())
    .collect();
    if !failures.is_empty() {
        panic!(
            "gen-core audio-transform conformance FAILED for `{}` (gen-core {}):\n  - {}",
            t.descriptor().id,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
