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
//! through the size axis. This module therefore builds an audio request (prompt + a nominal size +
//! [`AudioParams`]) and asserts the *audio-surface* capability gaps — an unadvertised voice /
//! language / sample-rate, or visual-only conditioning — are the ones rejected (as typed
//! [`Error::Unsupported`]). It deliberately makes **no** claim about how the model treats a visual
//! size: an audio descriptor advertises no size bounds (`min_size`/`max_size` are the unused 0 —
//! sc-13314) and a conformant provider ignores width/height entirely, so the harness sets a nominal
//! in-range size rather than probing the size axis. The modality-agnostic progress/cancellation
//! helpers
//! ([`check_progress_with`](crate::check_progress_with),
//! [`check_cancellation_with`](crate::check_cancellation_with),
//! [`check_precancellation_with`](crate::check_precancellation_with)) from the crate root cover the
//! guarantees that are identical across modalities.

use gen_core::{
    AudioChunk, AudioParams, AudioTrack, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, Modality,
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

/// The in-capability audio request the positive checks synthesize from. An audio descriptor
/// advertises no size bounds (`min_size`/`max_size` are the unused 0 — sc-13314) and a pure-audio
/// model ignores width/height, so the harness sets a fixed nominal size rather than reading the
/// (absent) bounds. The provider-specific choice of whether to range-check size at all is
/// deliberately not probed.
pub(crate) fn audio_base_request(g: &dyn Generator, profile: &AudioProfile) -> GenerationRequest {
    // Nominal, ignored by audio models. `min_size.max(1)` keeps a sane value whether the descriptor
    // leaves the bound at 0 (the audio convention) or an older provider still advertises a floor.
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

/// **Incremental-streaming contract (sc-12846).** Drives the provider through
/// [`Generator::generate_streaming`] and enforces the [`AudioChunk`] reassembly law plus the
/// [`Capabilities::supports_streaming`](gen_core::Capabilities::supports_streaming) opt-in — the
/// audio analog of the token-streaming guarantee `core_llm`'s stream events carry:
///
/// - the streamed call returns a well-formed [`GenerationOutput::Audio`] track (the shared
///   `validate_track` floor);
/// - **reassembly**: concatenating every emitted chunk's `samples` in order equals the returned
///   track's `samples` byte-for-byte — a provider that streams chunks which do *not* reassemble to
///   the audio fails here;
/// - **chunk well-formedness**: each chunk's `sample_rate` / `channels` match the track, `index`
///   runs `0..N` with no gaps, and each chunk is a whole number of frames;
/// - **the streamed output equals the one-shot [`generate`](Generator::generate)** for the same
///   request+seed (deterministic providers) — so the streaming path is a strict refinement of the
///   one-shot contract, never a different rendering;
/// - **opt-in shape**: a provider advertising
///   [`supports_streaming`](gen_core::Capabilities::supports_streaming) must emit **≥ 2** chunks
///   before completion **and no single chunk may carry the entire track** (each chunk is strictly
///   shorter than the full track) — the two together are genuine incrementality: the count alone is
///   gameable by a zero-length chunk plus one full-track chunk, and the per-chunk length bound closes
///   that (the audio must not arrive in one block); a provider that does **not** advertise streaming
///   must be byte-for-byte unaffected by the additive method — the default passthrough emits exactly
///   one chunk equal to the whole track.
///
/// Runs for every audio provider (streaming or not), so the additive `generate_streaming` surface is
/// proven not to perturb the eight one-shot audio families while gating the streaming families on
/// real incrementality.
pub fn check_audio_streaming(g: &dyn Generator, profile: &AudioProfile) -> Result<(), String> {
    let desc = g.descriptor();
    let id = desc.id;
    let advertises_streaming = desc.capabilities.supports_streaming;
    let req = audio_base_request(g, profile);

    // Baseline: the one-shot output the stream must reproduce.
    let one_shot = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("streaming[{id}]: baseline one-shot generate() failed: {e}"))?;
    let one_shot_track = match &one_shot {
        GenerationOutput::Audio(t) => t,
        other => {
            return Err(format!(
            "streaming[{id}]: a Modality::Audio generator must emit GenerationOutput::Audio from \
                 generate(), got {}",
            variant_name(other)
        ))
        }
    };

    // Drive the streaming entry point, collecting the chunks.
    let mut chunks: Vec<AudioChunk> = Vec::new();
    let streamed = g
        .generate_streaming(&req, &mut |c| chunks.push(c), &mut |_| {})
        .map_err(|e| format!("streaming[{id}]: generate_streaming() failed: {e}"))?;
    let streamed_track = match &streamed {
        GenerationOutput::Audio(t) => t,
        other => {
            return Err(format!(
                "streaming[{id}]: generate_streaming() must return GenerationOutput::Audio, got {}",
                variant_name(other)
            ))
        }
    };
    validate_track(id, "streaming", streamed_track)?;

    if chunks.is_empty() {
        return Err(format!(
            "streaming[{id}]: generate_streaming() emitted no AudioChunk — even the non-streaming \
             default passthrough must emit one terminal chunk for an audio output"
        ));
    }

    // Chunk indices run 0..N with no gaps, and every chunk agrees with the track's format.
    for (i, c) in chunks.iter().enumerate() {
        if c.index != i {
            return Err(format!(
                "streaming[{id}]: AudioChunk index {} out of order at position {i} (chunks must be \
                 0..N with no gaps)",
                c.index
            ));
        }
        if c.sample_rate != streamed_track.sample_rate {
            return Err(format!(
                "streaming[{id}]: chunk {i} sample_rate {} != track sample_rate {}",
                c.sample_rate, streamed_track.sample_rate
            ));
        }
        if c.channels != streamed_track.channels {
            return Err(format!(
                "streaming[{id}]: chunk {i} channels {} != track channels {}",
                c.channels, streamed_track.channels
            ));
        }
        if streamed_track.channels != 0
            && !c
                .samples
                .len()
                .is_multiple_of(streamed_track.channels as usize)
        {
            return Err(format!(
                "streaming[{id}]: chunk {i} has {} samples, not a whole number of {}-channel frames",
                c.samples.len(),
                streamed_track.channels
            ));
        }
    }

    // The reassembly law: concatenated chunk PCM equals the returned track PCM, byte-for-byte.
    let reassembled: Vec<f32> = chunks
        .iter()
        .flat_map(|c| c.samples.iter().copied())
        .collect();
    if reassembled != streamed_track.samples {
        return Err(format!(
            "streaming[{id}]: concatenated chunks ({} samples) do not reassemble to the returned \
             track ({} samples) — the AudioChunk reassembly law is violated",
            reassembled.len(),
            streamed_track.samples.len()
        ));
    }

    // The streamed output must equal the one-shot output (deterministic providers) — the stream is a
    // refinement of generate(), not a distinct rendering.
    if streamed_track.samples != one_shot_track.samples {
        return Err(format!(
            "streaming[{id}]: streamed audio ({} samples) differs from the one-shot generate() output \
             ({} samples) for the same request+seed — the streaming path must reproduce generate()",
            streamed_track.samples.len(),
            one_shot_track.samples.len()
        ));
    }

    if advertises_streaming {
        // A provider that advertises streaming must be genuinely incremental: ≥ 2 chunks before
        // completion. This is the assertion that fails if a "streaming" provider buffers the whole
        // output and emits it as one terminal chunk.
        if chunks.len() < 2 {
            return Err(format!(
                "streaming[{id}]: advertises supports_streaming but emitted {} chunk(s) — a streaming \
                 provider must emit >= 2 chunks before completion (it appears to buffer everything and \
                 emit one terminal chunk)",
                chunks.len()
            ));
        }
        // The chunk-count gate alone is gameable: a provider could emit a zero-length chunk plus one
        // full-track chunk (2 chunks, reassembles, frame-aligned) while not being incremental at all.
        // Harden it — no single chunk may hold the entire track. Since the chunks reassemble to the
        // track, any chunk whose length equals the total forces every other chunk to be empty, i.e.
        // the whole output arrived in one block. Every chunk must therefore be strictly shorter than
        // the full track.
        let total = streamed_track.samples.len();
        if let Some(i) = chunks.iter().position(|c| c.samples.len() >= total) {
            return Err(format!(
                "streaming[{id}]: advertises supports_streaming but chunk {i} carries the entire track \
                 ({} of {total} samples) — the audio arrived in a single block (an empty chunk plus one \
                 full-track chunk games the >= 2 count), so the provider is not genuinely incremental",
                chunks[i].samples.len()
            ));
        }
    } else {
        // The additive surface must not perturb a non-streaming provider: the default passthrough
        // emits exactly one chunk equal to the whole track.
        if chunks.len() != 1 {
            return Err(format!(
                "streaming[{id}]: does not advertise supports_streaming yet emitted {} chunks — the \
                 additive default must pass generate() through as a single terminal chunk",
                chunks.len()
            ));
        }
    }

    Ok(())
}

/// Build a two-segment dialogue [`script`](gen_core::SpeechSegment) whose speaker labels are
/// in-surface for `caps`: two advertised voices when the model has a closed voice surface (both
/// segments fall back to the single advertised voice when only one exists), or the opaque dialogue
/// labels `"S1"` / `"S2"` when the model advertises no voice surface (a dialogue model that maps
/// labels itself). Kept trivially short — the check probes *rendering + gating*, not audio quality
/// (the real per-segment voice-distinctness measurement lives in the provider's own real-weights
/// conformance, e.g. the MOSS multi-speaker test).
fn two_speaker_script(caps: &gen_core::Capabilities) -> Vec<gen_core::SpeechSegment> {
    let (a, b) = match caps.audio_voices.as_slice() {
        [] => ("S1".to_owned(), "S2".to_owned()),
        [only] => ((*only).to_owned(), (*only).to_owned()),
        [first, second, ..] => ((*first).to_owned(), (*second).to_owned()),
    };
    vec![
        gen_core::SpeechSegment {
            text: "Hello, how are you today?".to_owned(),
            speaker: Some(a),
            ..Default::default()
        },
        gen_core::SpeechSegment {
            text: "I'm doing great, thanks for asking!".to_owned(),
            speaker: Some(b),
            ..Default::default()
        },
    ]
}

/// **Multi-speaker script contract (sc-12848).** A [`Generator`] that advertises
/// [`Capabilities::supports_multi_speaker`](gen_core::Capabilities::supports_multi_speaker) must
/// *accept and render* a valid multi-speaker [`script`](gen_core::AudioParams::script) into one
/// well-formed [`GenerationOutput::Audio`] track, and reject a script naming more distinct speakers
/// than any advertised [`max_speakers`](gen_core::Capabilities::max_speakers) cap; a provider that
/// does **not** advertise multi-speaker support must reject a script as the typed
/// [`Error::Unsupported`] (never silently read only the first segment). This is the contract-level
/// gate every audio provider inherits — the deeper "the two speaker segments are genuinely rendered
/// in *different* voices" assertion is a provider-specific real-weights check (it needs a real model
/// and a voice-identity embedder), out of scope for the pure-host testkit.
pub fn check_audio_multi_speaker(g: &dyn Generator, profile: &AudioProfile) -> Result<(), String> {
    let desc = g.descriptor();
    let id = desc.id;
    let caps = &desc.capabilities;

    let mut r = audio_base_request(g, profile);
    r.audio = Some(AudioParams {
        script: Some(two_speaker_script(caps)),
        ..profile.audio.clone()
    });

    if caps.supports_multi_speaker {
        // Positive: a valid 2-speaker script is accepted and renders one well-formed audio track.
        g.validate(&r).map_err(|e| {
            format!(
                "multi-speaker[{id}]: advertises supports_multi_speaker but validate() rejected a \
                 valid 2-speaker script: {e}"
            )
        })?;
        let out = g.generate(&r, &mut |_| {}).map_err(|e| {
            format!("multi-speaker[{id}]: generate() failed on a 2-speaker script: {e}")
        })?;
        match &out {
            GenerationOutput::Audio(track) => validate_track(id, "multi-speaker", track)?,
            other => {
                return Err(format!(
                    "multi-speaker[{id}]: a script must render GenerationOutput::Audio, got {}",
                    variant_name(other)
                ));
            }
        }
        // Range: a script naming more than `max_speakers` distinct speakers is rejected.
        if let Some(max) = caps.max_speakers {
            let too_many: Vec<gen_core::SpeechSegment> = (0..=max)
                .map(|i| gen_core::SpeechSegment {
                    text: format!("Line from speaker {i}."),
                    speaker: Some(format!("__testkit_spk_{i}__")),
                    ..Default::default()
                })
                .collect();
            let mut rr = audio_base_request(g, profile);
            rr.audio = Some(AudioParams {
                script: Some(too_many),
                ..profile.audio.clone()
            });
            if g.validate(&rr).is_ok() {
                return Err(format!(
                    "multi-speaker[{id}]: a script naming {} distinct speakers (above \
                     max_speakers {max}) was accepted by validate()",
                    max + 1
                ));
            }
        }
    } else {
        // Negative: a non-multi-speaker provider must reject a script as the typed Unsupported.
        expect_unsupported(g, &r, id, "a multi-speaker audio.script")?;
    }

    Ok(())
}

/// The video→audio (Foley) clip the [`check_video_to_audio`] gate conditions on: a short run of small
/// RGB frames with distinct per-frame, non-uniform content (a gradient seeded by `variant`), so a
/// conformant model's soundtrack genuinely depends on the pixels. Kept tiny — the check probes the
/// *contract* (a synchronized track is produced, reproducibly, and actually reads the frames), not
/// synchronization fidelity (a real MMAudio-style provider proves that in its own real-weights
/// conformance).
const VIDEO_SYNC_FRAMES: usize = 8;
const VIDEO_SYNC_FPS: u32 = 8;

fn foley_clip(variant: u8) -> Vec<Image> {
    const W: u32 = 16;
    const H: u32 = 16;
    (0..VIDEO_SYNC_FRAMES)
        .map(|f| {
            let mut pixels = vec![0u8; (W * H * 3) as usize];
            for (i, p) in pixels.iter_mut().enumerate() {
                // Vary by pixel position, frame index, and the clip variant so frames are distinct
                // and non-uniform, and two variants differ in every pixel neighborhood.
                *p = ((i as u32 + f as u32 * 37 + variant as u32 * 101) % 251) as u8;
            }
            Image {
                width: W,
                height: H,
                pixels,
            }
        })
        .collect()
}

/// The in-capability video→audio request: the profile's prompt + seed, a nominal (ignored) size, the
/// frame rate on [`GenerationRequest::fps`] (never on the variant), and the clip as a single
/// [`Conditioning::VideoSync`].
fn foley_request(
    g: &dyn Generator,
    profile: &AudioProfile,
    frames: Vec<Image>,
) -> GenerationRequest {
    let mut r = audio_base_request(g, profile);
    r.fps = Some(VIDEO_SYNC_FPS);
    r.conditioning = vec![Conditioning::VideoSync { frames }];
    r
}

/// **Video→audio (Foley) contract (sc-13436).** A [`Generator`] that advertises
/// [`ConditioningKind::VideoSync`] in its
/// [`Capabilities::conditioning`](gen_core::Capabilities::conditioning) must *accept and render* a
/// silent clip's frames ([`Conditioning::VideoSync`]) into **one**
/// well-formed [`GenerationOutput::Audio`] track that is:
///
/// - **non-empty and non-silent** — at least one audible sample; a provider that ignores the frames
///   and emits silence (or an empty track) fails here (the headline dishonest-provider gate);
/// - **plausibly the clip's length** — the track duration sits within a generous band of
///   `frames / fps` (the frame rate rides [`GenerationRequest::fps`], the story's fps-from-req.fps
///   decision), tolerating codec-frame padding while catching a wildly-wrong or empty length;
/// - **byte-identical on re-synth with the same clip + seed** — the reproducibility law, the
///   video→audio twin of [`check_audio_seed_determinism`]; and
/// - **frame-dependent** — a *visually different* clip (same seed) must change the audio, so a
///   provider that advertises `VideoSync` yet ignores the pixels and renders seed-only audio is
///   caught (the visual-condition analog of "a different seed changes the output").
///
/// A provider that does **not** advertise `VideoSync` must reject a Foley clip as the typed
/// [`Error::Unsupported`] (F-008) — it can never silently drop the frames and emit unconditioned
/// audio. Runs for every audio provider (the non-advertising branch proves the additive variant does
/// not perturb the existing single-modality families), so it is a member of [`audio_conformance`].
pub fn check_video_to_audio(g: &dyn Generator, profile: &AudioProfile) -> Result<(), String> {
    let desc = g.descriptor();
    let id = desc.id;
    let advertises = desc.capabilities.accepts(ConditioningKind::VideoSync);

    let frames = foley_clip(0);
    let expected_secs = frames.len() as f32 / VIDEO_SYNC_FPS as f32;
    let req = foley_request(g, profile, frames);

    if !advertises {
        // A model that does not advertise VideoSync must reject a Foley clip as the typed
        // Unsupported — never silently emit unconditioned audio.
        return expect_unsupported(g, &req, id, "a VideoSync (video→audio) clip");
    }

    // Positive: validate accepts, and generate renders one well-formed audio track.
    g.validate(&req).map_err(|e| {
        format!(
            "video-to-audio[{id}]: advertises VideoSync but validate() rejected a valid Foley clip: \
             {e}"
        )
    })?;
    let out = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("video-to-audio[{id}]: generate() failed on a VideoSync clip: {e}"))?;
    let track = match &out {
        GenerationOutput::Audio(t) => t,
        other => {
            return Err(format!(
                "video-to-audio[{id}]: a VideoSync clip must render GenerationOutput::Audio, got {}",
                variant_name(other)
            ));
        }
    };
    validate_track(id, "video-to-audio", track)?;

    // Non-silent: at least one sample above a tiny epsilon. An all-zero (or empty) track means the
    // model produced no soundtrack for the clip — the primary dishonest-provider catch.
    if !track.samples.iter().any(|s| s.abs() > 1e-6) {
        return Err(format!(
            "video-to-audio[{id}]: the rendered track is silent (all samples ~0) — a video→audio \
             model must synthesize an audible soundtrack for the clip, not silence"
        ));
    }

    // Plausible duration: within a generous band of the clip length (frames / fps). Catches an
    // empty or grossly-wrong-length track while tolerating codec-frame padding on real models.
    let secs =
        track.samples.len() as f32 / track.channels.max(1) as f32 / track.sample_rate.max(1) as f32;
    if !(secs.is_finite() && secs >= expected_secs * 0.25 && secs <= expected_secs * 4.0) {
        return Err(format!(
            "video-to-audio[{id}]: rendered track is {secs:.3}s but the {VIDEO_SYNC_FRAMES}-frame \
             clip at {VIDEO_SYNC_FPS} fps is ~{expected_secs:.3}s — the soundtrack length is \
             implausible for the clip"
        ));
    }

    // Reproducibility law: the same clip + seed re-synthesizes byte-identical PCM.
    let out2 = g
        .generate(&req, &mut |_| {})
        .map_err(|e| format!("video-to-audio[{id}]: second generate() failed: {e}"))?;
    if crate::output_bytes(&out) != crate::output_bytes(&out2) {
        return Err(format!(
            "video-to-audio[{id}]: the same clip + seed produced different audio on re-synth — the \
             reproducibility law is violated"
        ));
    }

    // Frame-dependence: a visually different clip (same seed) must change the output — the assertion
    // that catches a provider advertising VideoSync while ignoring the frames (seed-only audio).
    let other_req = foley_request(g, profile, foley_clip(128));
    let other = g.generate(&other_req, &mut |_| {}).map_err(|e| {
        format!("video-to-audio[{id}]: generate() on a second (different) clip failed: {e}")
    })?;
    if crate::output_bytes(&other) == crate::output_bytes(&out) {
        return Err(format!(
            "video-to-audio[{id}]: two visually-different clips (same seed) produced byte-identical \
             audio — the provider appears to ignore the VideoSync frames"
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
    let checks: [Check; 10] = [
        check_audio_validate_honesty,
        check_audio_output,
        check_audio_progress,
        check_audio_progress_contract,
        check_audio_cancellation,
        check_audio_precancellation,
        check_audio_seed_determinism,
        check_audio_streaming,
        check_audio_multi_speaker,
        check_video_to_audio,
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
