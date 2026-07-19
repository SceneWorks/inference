//! Contract conformance for pure-audio [`gen_core::Generator`] providers — a text→audio synthesis
//! model ([`Modality::Audio`], sc-12834): TTS (`candle-audio-kokoro`), SFX (`candle-audio-moss-sfx`),
//! music (`candle-audio-acestep`). The audio analog of the image/video generator suite in
//! [`crate`](crate::conformance): it exercises the same behavioral guarantees — capability honesty,
//! progress monotonicity, typed cancellation, seed determinism — over an audio request whose output
//! is a [`GenerationOutput::Audio`] track rather than an image.
//!
//! ## Why a parallel entry point rather than the image suite
//!
//! An audio generator validates through the **size-skipping floor**
//! ([`Capabilities::validate_request_audio`](gen_core::Capabilities::validate_request_audio)): the
//! `width`/`height` range check is skipped, because those fields are unused by a pure audio model.
//! The image [`conformance`](crate::conformance) suite builds a text+size request and asserts an
//! *oversize* rejection at `max_size + 64` — exactly the check an audio model must NOT enforce
//! through the size axis. This module therefore builds an audio request (prompt + an in-range size
//! read from the descriptor's advertised bounds + [`AudioParams`]) and asserts the *audio-surface*
//! capability gaps — an unadvertised voice / language / sample-rate, or visual-only conditioning —
//! are the ones rejected (as typed [`Error::Unsupported`]). It deliberately makes **no** claim about
//! how the model treats an out-of-range visual size: a conformant provider may still honor its
//! advertised `min_size..=max_size` bounds even under `Modality::Audio` (candle-audio-kokoro does),
//! so the harness keeps its size inside those bounds rather than probing that provider-specific
//! choice. The modality-agnostic progress/cancellation helpers
//! ([`check_progress_with`](crate::check_progress_with),
//! [`check_cancellation_with`](crate::check_cancellation_with),
//! [`check_precancellation_with`](crate::check_precancellation_with)) from the crate root cover the
//! guarantees that are identical across modalities.

use gen_core::{
    AudioParams, AudioTrack, Conditioning, Error, GenerationOutput, GenerationRequest, Generator,
    Image, Modality,
};

/// Cheap-request parameters for an audio conformance run — the audio analog of
/// [`Profile`](crate::Profile). Keep the step count tiny (the suite calls `generate` several times)
/// and the audio sub-block empty by default so the positive request stays inside any model's
/// advertised surface (an unset `voice`/`language`/`sample_rate` always passes the floor — the
/// model's native choice).
#[derive(Clone, Debug)]
pub struct AudioProfile {
    pub prompt: String,
    /// Denoise/decode steps the request asks for and the value [`check_audio_progress`] expects the
    /// model to resolve to (`Progress::Step.total == steps`).
    pub steps: u32,
    pub seed: u64,
    /// Steps requested for [`check_audio_cancellation`] only — needs headroom (≥ 3) so a provider
    /// honoring cancellation visibly stops before completion.
    pub cancel_steps: u32,
    /// The valid audio sub-block the positive checks synthesize from. Default is empty (every field
    /// `None`) so the request is accepted by any audio descriptor; a caller whose model *requires* a
    /// voice/language sets the in-surface values here.
    pub audio: AudioParams,
}

impl Default for AudioProfile {
    fn default() -> Self {
        Self {
            prompt: "a short spoken phrase".to_owned(),
            steps: 2,
            seed: 42,
            cancel_steps: 6,
            audio: AudioParams::default(),
        }
    }
}

impl AudioProfile {
    /// The cheapest generally-valid audio profile: a short prompt, 2 steps, fixed seed, and an empty
    /// (all-native) audio sub-block.
    pub fn cheap() -> Self {
        Self::default()
    }
}

/// The in-capability audio request the positive checks synthesize from. `width`/`height` are set to
/// the descriptor's advertised `min_size` (always `>= 1`, and `<= max_size`, per the descriptor
/// sweep), so the request is in-range for a provider that still honors its size bounds under
/// `Modality::Audio` (candle-audio-kokoro) while being trivially ignored by one that fully skips the
/// size axis. The provider-specific choice of whether to range-check size at all is deliberately not
/// probed.
pub(crate) fn audio_base_request(g: &dyn Generator, profile: &AudioProfile) -> GenerationRequest {
    let size = g.descriptor().capabilities.min_size.max(1);
    GenerationRequest {
        prompt: profile.prompt.clone(),
        width: size,
        height: size,
        steps: Some(profile.steps),
        seed: Some(profile.seed),
        audio: Some(profile.audio.clone()),
        ..Default::default()
    }
}

/// A tiny all-zero RGB image, for building visual conditioning an audio model must reject.
fn blank_image() -> Image {
    Image {
        width: 8,
        height: 8,
        pixels: vec![0u8; 8 * 8 * 3],
    }
}

/// **Validate honesty (audio floor).** The in-capability audio request (prompt + an in-range size +
/// the profile's audio sub-block) is accepted, and requests that exceed the advertised *audio*
/// surface are rejected — capability gaps (an unadvertised voice / language / sample-rate, or
/// visual-only conditioning) as the **typed** [`Error::Unsupported`], not a stringified `Msg`; the
/// count / duration range violations as `Error::Msg`. The size axis is deliberately not probed
/// (see the module docs): a conformant provider may or may not honor its bounds under
/// `Modality::Audio`, so the request stays in-range.
pub fn check_audio_validate_honesty(
    g: &dyn Generator,
    profile: &AudioProfile,
) -> Result<(), String> {
    let desc = g.descriptor();
    let caps = &desc.capabilities;
    let id = desc.id;

    if desc.modality != Modality::Audio {
        return Err(format!(
            "validate-honesty[{id}]: audio conformance requires Modality::Audio, got {:?}",
            desc.modality
        ));
    }

    // Positive: the declared cheap audio request must be accepted.
    g.validate(&audio_base_request(g, profile)).map_err(|e| {
        format!("validate-honesty[{id}]: the in-capability cheap audio request was rejected by validate(): {e}")
    })?;

    // Negative (capability gap): an unadvertised voice is a typed Unsupported. A synthetic id that is
    // (essentially certainly) not advertised is rejected whether the model has a closed voice list or
    // no selectable voice surface at all (an empty list rejects any explicit voice).
    let mut r = audio_base_request(g, profile);
    r.audio = Some(AudioParams {
        voice: Some("__testkit_unsupported_voice__".to_owned()),
        ..profile.audio.clone()
    });
    expect_unsupported(g, &r, id, "an unadvertised audio.voice")?;

    // Negative (capability gap): an unadvertised language is a typed Unsupported.
    let mut r = audio_base_request(g, profile);
    r.audio = Some(AudioParams {
        language: Some("__testkit_unsupported_lang__".to_owned()),
        ..profile.audio.clone()
    });
    expect_unsupported(g, &r, id, "an unadvertised audio.language")?;

    // Negative (capability gap): an unadvertised sample rate is a typed Unsupported.
    if let Some(sr) = unadvertised_sample_rate(&caps.audio_sample_rates) {
        let mut r = audio_base_request(g, profile);
        r.audio = Some(AudioParams {
            sample_rate: Some(sr),
            ..profile.audio.clone()
        });
        expect_unsupported(g, &r, id, "an unadvertised audio.sample_rate")?;
    }

    // Negative (capability gap): visual-only conditioning an audio model does not advertise is a
    // typed Unsupported.
    let cond = Conditioning::Reference {
        image: blank_image(),
        strength: None,
    };
    if !caps.accepts(cond.kind()) {
        let mut r = audio_base_request(g, profile);
        r.conditioning = vec![cond];
        expect_unsupported(g, &r, id, "visual-only Reference conditioning")?;
    }

    // Negative (range): count above max_count is rejected (Error::Msg — a malformed value, not a
    // capability gap).
    if let Some(many) = caps.max_count.checked_add(1) {
        let mut r = audio_base_request(g, profile);
        r.count = many;
        if g.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: count {many} (above max_count {}) was accepted by validate()",
                caps.max_count
            ));
        }
    }

    // Negative (range): a target_duration above the advertised cap is rejected.
    if let Some(cap) = caps.max_audio_duration_secs {
        let mut r = audio_base_request(g, profile);
        r.audio = Some(AudioParams {
            target_duration: Some(cap + 1.0),
            ..profile.audio.clone()
        });
        if g.validate(&r).is_ok() {
            return Err(format!(
                "validate-honesty[{id}]: audio.target_duration {}s (above the advertised max {cap}s) \
                 was accepted by validate()",
                cap + 1.0
            ));
        }
    }

    Ok(())
}

/// Assert `validate(req)` returns the typed [`Error::Unsupported`] for a capability-gap request.
fn expect_unsupported(
    g: &dyn Generator,
    req: &GenerationRequest,
    id: &str,
    what: &str,
) -> Result<(), String> {
    match g.validate(req) {
        Ok(()) => Err(format!(
            "validate-honesty[{id}]: {what} was accepted by validate() — a capability gap must be \
             rejected"
        )),
        Err(Error::Unsupported(_)) => Ok(()),
        Err(other) => Err(format!(
            "validate-honesty[{id}]: {what} must be rejected as the typed Err(Error::Unsupported), \
             got {other:?} — a stringified Error::Msg breaks the typed capability-gap contract"
        )),
    }
}

/// The first sample rate not in the advertised list — or a fixed sentinel when the list is empty (an
/// empty list rejects any explicit rate). `None` only if every candidate is somehow advertised.
fn unadvertised_sample_rate(advertised: &[u32]) -> Option<u32> {
    [12_345u32, 7, 1, 99_991]
        .into_iter()
        .find(|c| !advertised.contains(c))
}

/// **Output well-formedness.** `generate` on the cheap audio request returns a
/// [`GenerationOutput::Audio`] track (not an image/video), with a positive `sample_rate`, positive
/// `channels`, a non-empty, whole number of frames, and all-finite PCM — the audio analog of the
/// image suite's "non-empty pixels" floor.
pub fn check_audio_output(g: &dyn Generator, profile: &AudioProfile) -> Result<(), String> {
    let id = g.descriptor().id;
    let out = g
        .generate(&audio_base_request(g, profile), &mut |_| {})
        .map_err(|e| format!("output[{id}]: generate() failed on the cheap request: {e}"))?;
    let track = match &out {
        GenerationOutput::Audio(track) => track,
        other => {
            return Err(format!(
                "output[{id}]: a Modality::Audio generator must emit GenerationOutput::Audio, got \
                 {}",
                variant_name(other)
            ));
        }
    };
    validate_track(id, "output", track)
}

/// Shared well-formedness floor for a synthesized [`AudioTrack`].
pub(crate) fn validate_track(id: &str, op: &str, track: &AudioTrack) -> Result<(), String> {
    if track.sample_rate == 0 {
        return Err(format!("{op}[{id}]: AudioTrack.sample_rate is 0"));
    }
    if track.channels == 0 {
        return Err(format!("{op}[{id}]: AudioTrack.channels is 0"));
    }
    if track.samples.is_empty() {
        return Err(format!("{op}[{id}]: AudioTrack.samples is empty"));
    }
    if !track.samples.len().is_multiple_of(track.channels as usize) {
        return Err(format!(
            "{op}[{id}]: AudioTrack has {} interleaved samples, not a whole number of {}-channel \
             frames",
            track.samples.len(),
            track.channels
        ));
    }
    if let Some(i) = track.samples.iter().position(|s| !s.is_finite()) {
        return Err(format!(
            "{op}[{id}]: AudioTrack.samples[{i}] is non-finite ({}) — synthesized PCM must be finite",
            track.samples[i]
        ));
    }
    Ok(())
}

fn variant_name(out: &GenerationOutput) -> &'static str {
    match out {
        GenerationOutput::Images(_) => "GenerationOutput::Images",
        GenerationOutput::Video { .. } => "GenerationOutput::Video",
        GenerationOutput::Audio(_) => "GenerationOutput::Audio",
    }
}

/// **Progress.** `Progress::Step{current,total}` runs exactly `1..=steps` — the modality-agnostic
/// [`check_progress_with`](crate::check_progress_with) driven from the audio request.
pub fn check_audio_progress(g: &dyn Generator, profile: &AudioProfile) -> Result<(), String> {
    crate::check_progress_with(g, &audio_base_request(g, profile), Some(profile.steps))
}

/// **Progress contract.** Monotone in-bounds `Step`, the bar reaches `total`, and `Decoding` fires
/// exactly once — the whole-class property via
/// [`check_progress_contract_with`](crate::check_progress_contract_with).
pub fn check_audio_progress_contract(
    g: &dyn Generator,
    profile: &AudioProfile,
) -> Result<(), String> {
    crate::check_progress_contract_with(g, &audio_base_request(g, profile))
}

/// **Cancellation (mid-generate).** Tripping `CancelFlag` at the first step boundary makes `generate`
/// return the typed `Err(Error::Canceled)` within ≤ 2 further steps — the modality-agnostic
/// [`check_cancellation_with`](crate::check_cancellation_with) with a headroom step count.
pub fn check_audio_cancellation(g: &dyn Generator, profile: &AudioProfile) -> Result<(), String> {
    let mut req = audio_base_request(g, profile);
    req.steps = Some(profile.cancel_steps);
    crate::check_cancellation_with(g, &req)
}

/// **Pre-generate cancellation.** A request whose `CancelFlag` is already tripped at call time must
/// return the typed `Err(Error::Canceled)` before any expensive work — via
/// [`check_precancellation_with`](crate::check_precancellation_with).
pub fn check_audio_precancellation(
    g: &dyn Generator,
    profile: &AudioProfile,
) -> Result<(), String> {
    crate::check_precancellation_with(g, &audio_base_request(g, profile))
}

/// **Seed determinism (same backend).** Two runs of the identical request+seed produce byte-identical
/// PCM, and a *different* seed changes it — the audio twin of
/// [`check_seed_determinism`](crate::check_seed_determinism), comparing the little-endian sample bytes
/// of the resolved `GenerationOutput::Audio` track.
pub fn check_audio_seed_determinism(
    g: &dyn Generator,
    profile: &AudioProfile,
) -> Result<(), String> {
    let id = g.descriptor().id;
    let req = audio_base_request(g, profile);
    let a = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: first generate() failed: {e}"))?;
    let b = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: second generate() failed: {e}"))?;
    let (ba, bb) = (crate::output_bytes(&a), crate::output_bytes(&b));
    if ba != bb {
        return Err(format!(
            "seed[{id}]: same request+seed produced different audio ({} vs {} bytes)",
            ba.len(),
            bb.len()
        ));
    }
    let mut req_alt = audio_base_request(g, profile);
    req_alt.seed = Some(profile.seed.wrapping_add(0x9E37_79B9));
    let c = g
        .generate(&req_alt, &mut |_| {})
        .map_err(|e| format!("seed[{id}]: alternate-seed generate() failed: {e}"))?;
    let bc = crate::output_bytes(&c);
    if bc == ba {
        return Err(format!(
            "seed[{id}]: a different seed produced byte-identical audio ({} bytes) — the provider \
             appears to ignore the seed",
            ba.len()
        ));
    }
    Ok(())
}

/// Run the full audio-generator conformance suite against a freshly-`make`d generator. Panics with
/// every failure aggregated — the audio twin of [`conformance`](crate::conformance).
pub fn audio_conformance(make: impl Fn() -> Box<dyn Generator>, profile: &AudioProfile) {
    let g = make();
    let g: &dyn Generator = g.as_ref();

    type Check = fn(&dyn Generator, &AudioProfile) -> Result<(), String>;
    let checks: [Check; 7] = [
        check_audio_validate_honesty,
        check_audio_output,
        check_audio_progress,
        check_audio_progress_contract,
        check_audio_cancellation,
        check_audio_precancellation,
        check_audio_seed_determinism,
    ];
    let failures: Vec<String> = checks
        .into_iter()
        .filter_map(|f| f(g, profile).err())
        .collect();
    if !failures.is_empty() {
        panic!(
            "gen-core audio conformance FAILED for `{}` ({} backend, gen-core {}):\n  - {}",
            g.descriptor().id,
            g.descriptor().backend,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
