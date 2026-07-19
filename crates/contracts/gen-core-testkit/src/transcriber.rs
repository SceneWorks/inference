//! Contract conformance for [`gen_core::Transcriber`] providers (sc-12850) — speech/audio→text ASR
//! (`candle-audio-whisper`). The audio sibling of the [`Captioner`](gen_core::Captioner) suite: same
//! shape of guarantees (capability honesty, typed cancellation, registry discoverability), one
//! modality over — an [`AudioTrack`](gen_core::AudioTrack) in, a [`TranscriptOutput`] out.
//!
//! ## Cancellation semantics
//!
//! Like a captioner, a transcriber that has already emitted tokens when cancel trips may return a
//! **partial** `Ok`. The typed [`Error::Canceled`] contract this suite
//! enforces covers cancellation *before inference starts*: a transcriber handed an already-cancelled
//! request must check the flag up front and return `Canceled` rather than running the audio encoder +
//! text decoder to produce a transcript nobody asked for — exactly the captioner's pre-inference
//! cancellation check, one modality over.

use gen_core::{
    Error, TranscribeOptions, TranscribeRequest, TranscribeSampling, TranscribeTask, Transcriber,
    TranscriptOutput,
};

/// Parameters for a transcriber conformance run — one in-capability audio clip the positive checks
/// transcribe, with a minimal (timestamp-free, auto-language) options surface so the positive request
/// is accepted by any transcriber. The negatives are derived from the descriptor's advertised
/// surface.
#[derive(Clone, Debug)]
pub struct TranscriberProfile {
    /// A valid ~3s mono clip.
    pub audio: gen_core::AudioTrack,
    /// The options the positive checks use — minimal (no timestamps, no language pin, transcribe
    /// task) so any transcriber accepts it.
    pub options: TranscribeOptions,
    pub sampling: TranscribeSampling,
}

impl Default for TranscriberProfile {
    fn default() -> Self {
        Self {
            audio: gen_core::AudioTrack {
                samples: vec![0.01; 48_000],
                sample_rate: 16_000,
                channels: 1,
                ..Default::default()
            },
            options: TranscribeOptions {
                language: None,
                task: TranscribeTask::Transcribe,
                timestamps: gen_core::TimestampGranularity::None,
            },
            sampling: TranscribeSampling {
                temperature: 0.0,
                max_new_tokens: 16,
                ..Default::default()
            },
        }
    }
}

impl TranscriberProfile {
    pub fn cheap() -> Self {
        Self::default()
    }
}

/// Build the in-capability request the positive checks transcribe, with a fresh cancel flag.
fn base_request(profile: &TranscriberProfile) -> TranscribeRequest {
    TranscribeRequest {
        audio: profile.audio.clone(),
        options: profile.options.clone(),
        sampling: profile.sampling,
        cancel: Default::default(),
    }
}

/// **Validate honesty.** The in-capability request is accepted, and capability-gap requests
/// (a translate task a transcribe-only model cannot serve, an out-of-set language) are rejected as
/// the **typed** [`Error::Unsupported`] — matching the ASR capability floor the worker's gating
/// depends on.
pub fn check_transcriber_validate(
    t: &dyn Transcriber,
    profile: &TranscriberProfile,
) -> Result<(), String> {
    let desc = t.descriptor();
    let caps = &desc.capabilities;
    let id = desc.id;

    // Positive: the declared cheap request must be accepted.
    t.validate(&base_request(profile)).map_err(|e| {
        format!("validate-honesty[{id}]: the in-capability cheap request was rejected by validate(): {e}")
    })?;

    // Negative (capability gap): a translate task on a transcribe-only model is a typed Unsupported.
    if !caps.supports_translate {
        let mut r = base_request(profile);
        r.options.task = TranscribeTask::Translate;
        expect_unsupported(t, &r, id, "a Translate task on a transcribe-only model")?;
    }

    // Negative (capability gap): a language outside the advertised closed set is a typed Unsupported.
    // (An empty `languages` list means "any", so there is no gap to exercise then.)
    if !caps.languages.is_empty() {
        let bogus = "__testkit_unsupported_lang__";
        if !caps.languages.contains(&bogus) {
            let mut r = base_request(profile);
            r.options.language = Some(bogus.to_owned());
            expect_unsupported(t, &r, id, "a language outside the advertised set")?;
        }
    }

    Ok(())
}

fn expect_unsupported(
    t: &dyn Transcriber,
    req: &TranscribeRequest,
    id: &str,
    what: &str,
) -> Result<(), String> {
    match t.validate(req) {
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

/// **Output well-formedness.** `transcribe` on the cheap request returns an `Ok` [`TranscriptOutput`]
/// whose segments (when present) are well-ordered: `start >= 0`, `end >= start`, both finite. A
/// transcriber that emits garbled timestamps trips this.
pub fn check_transcriber_output(
    t: &dyn Transcriber,
    profile: &TranscriberProfile,
) -> Result<(), String> {
    let id = t.descriptor().id;
    let out: TranscriptOutput = t
        .transcribe(&base_request(profile), &mut |_| {})
        .map_err(|e| format!("output[{id}]: transcribe() failed on the cheap request: {e}"))?;
    for (i, seg) in out.segments.iter().enumerate() {
        if !(seg.start.is_finite() && seg.end.is_finite()) {
            return Err(format!(
                "output[{id}]: segment[{i}] has a non-finite timestamp ({}..{})",
                seg.start, seg.end
            ));
        }
        if seg.start < 0.0 {
            return Err(format!(
                "output[{id}]: segment[{i}] start {} is negative",
                seg.start
            ));
        }
        if seg.end < seg.start {
            return Err(format!(
                "output[{id}]: segment[{i}] end {} precedes start {}",
                seg.end, seg.start
            ));
        }
    }
    Ok(())
}

/// **Pre-inference cancellation.** A transcriber handed an already-cancelled request must return the
/// **typed** `Err(Error::Canceled)` (not a stringified `Msg`, and not an `Ok` transcript) — it must
/// check the flag before running the encoder/decoder. Mirrors
/// [`check_captioner_cancellation`](crate::check_captioner_cancellation).
pub fn check_transcriber_cancellation(
    t: &dyn Transcriber,
    profile: &TranscriberProfile,
) -> Result<(), String> {
    let id = t.descriptor().id;
    let req = base_request(profile);
    req.cancel.cancel();
    match t.transcribe(&req, &mut |_| {}) {
        Ok(out) => Err(format!(
            "cancellation[{id}]: transcribe() returned Ok ({:?}) despite an already-cancelled \
             request; it must return Err(Error::Canceled) before running inference",
            out.text
        )),
        Err(Error::Canceled) => Ok(()),
        Err(other) => Err(format!(
            "cancellation[{id}]: must return the typed Err(Error::Canceled) on cancel, got {other:?} \
             — a stringified Error::Msg breaks the typed-cancellation contract"
        )),
    }
}

/// **Registry round-trip.** The transcriber's descriptor `id` is present in the explicit registry
/// supplied by the caller.
pub fn check_transcriber_registry(
    registry: &gen_core::ProviderRegistry,
    t: &dyn Transcriber,
) -> Result<(), String> {
    let id = t.descriptor().id;
    if registry
        .transcribers()
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

/// Run the full transcriber conformance suite against a freshly-`make`d transcriber. Panics with
/// every failure aggregated.
pub fn transcriber_conformance(
    make: impl Fn() -> Box<dyn Transcriber>,
    profile: &TranscriberProfile,
) {
    let t = make();
    let t: &dyn Transcriber = t.as_ref();

    type Check = fn(&dyn Transcriber, &TranscriberProfile) -> Result<(), String>;
    let checks: [Check; 3] = [
        check_transcriber_validate,
        check_transcriber_output,
        check_transcriber_cancellation,
    ];
    let failures: Vec<String> = checks
        .into_iter()
        .filter_map(|f| f(t, profile).err())
        .collect();
    if !failures.is_empty() {
        panic!(
            "gen-core transcriber conformance FAILED for `{}` (gen-core {}):\n  - {}",
            t.descriptor().id,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
