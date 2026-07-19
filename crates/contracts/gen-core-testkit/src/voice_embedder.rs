//! Contract conformance for [`gen_core::VoiceEmbedder`] providers (sc-12838) — a speaker-identity
//! encoder over reference audio (`candle-audio-chatterbox-ve`). The audio-identity analog of the
//! image [`FaceEmbedder`](gen_core::FaceEmbedder): it exercises the guarantees the contract promises
//! but cannot type — a finite, fixed-dimension embedding whose length matches the advertised
//! `embedding_dim`, and a typed error on degenerate (too-short / empty) input rather than a garbage
//! vector.

use gen_core::{AudioTrack, VoiceEmbedder};

/// Parameters for a voice-embedder conformance run — one in-range reference clip the positive check
/// embeds, and a degenerate (empty) clip the rejection check feeds.
#[derive(Clone, Debug)]
pub struct VoiceEmbedderProfile {
    /// A valid, ~1s mono reference clip.
    pub clip: AudioTrack,
    /// A degenerate clip (empty PCM) a conformant embedder must reject rather than embed.
    pub too_short: AudioTrack,
}

impl Default for VoiceEmbedderProfile {
    fn default() -> Self {
        Self {
            clip: AudioTrack {
                samples: vec![0.01; 24_000],
                sample_rate: 24_000,
                channels: 1,
                ..Default::default()
            },
            too_short: AudioTrack {
                samples: Vec::new(),
                sample_rate: 24_000,
                channels: 1,
                ..Default::default()
            },
        }
    }
}

impl VoiceEmbedderProfile {
    pub fn cheap() -> Self {
        Self::default()
    }
}

/// **Embedding shape + finiteness.** `embed` returns a vector of exactly
/// [`VoiceEmbedderDescriptor::embedding_dim`](gen_core::VoiceEmbedderDescriptor::embedding_dim)
/// elements — non-empty, and every element finite — so the identity pipeline that feeds it raw
/// through [`Conditioning::VoiceEmbedding`](gen_core::Conditioning::VoiceEmbedding) gets a well-shaped
/// vector, not a truncated or NaN-poisoned one.
pub fn check_voice_embed(
    e: &dyn VoiceEmbedder,
    profile: &VoiceEmbedderProfile,
) -> Result<(), String> {
    let desc = e.descriptor();
    let id = desc.id;
    let dim = desc.embedding_dim;
    if dim == 0 {
        return Err(format!(
            "embed[{id}]: descriptor advertises embedding_dim 0 — a voice embedder must have a \
             fixed, non-zero dimension"
        ));
    }
    let v = e
        .embed(&profile.clip)
        .map_err(|err| format!("embed[{id}]: embed() failed on the valid reference clip: {err}"))?;
    if v.len() != dim {
        return Err(format!(
            "embed[{id}]: embed() returned {} elements, but the descriptor advertises embedding_dim \
             {dim}",
            v.len()
        ));
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(format!(
            "embed[{id}]: embedding[{i}] is non-finite ({}) — a speaker vector must be finite",
            v[i]
        ));
    }
    Ok(())
}

/// **Degenerate-input rejection.** An empty / too-short reference clip is rejected with an `Err`
/// rather than embedded into a meaningless vector — a speaker encoder needs enough audio to form an
/// identity, so a conformant provider guards its input at the edge.
pub fn check_voice_embed_rejects_short(
    e: &dyn VoiceEmbedder,
    profile: &VoiceEmbedderProfile,
) -> Result<(), String> {
    let id = e.descriptor().id;
    match e.embed(&profile.too_short) {
        Err(_) => Ok(()),
        Ok(v) => Err(format!(
            "embed[{id}]: embed() returned an Ok vector ({} elements) for a degenerate (empty) clip; \
             it must reject too-short input with an Err",
            v.len()
        )),
    }
}

/// **Registry round-trip.** The embedder's descriptor `id` is present in the explicit registry
/// supplied by the caller (a missing catalog entry is the runtime "embedder not found" trap).
pub fn check_voice_embedder_registry(
    registry: &gen_core::ProviderRegistry,
    e: &dyn VoiceEmbedder,
) -> Result<(), String> {
    let id = e.descriptor().id;
    if registry
        .voice_embedders()
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

/// Run the full voice-embedder conformance suite against a freshly-`make`d embedder. Panics with
/// every failure aggregated.
pub fn voice_embedder_conformance(
    make: impl Fn() -> Box<dyn VoiceEmbedder>,
    profile: &VoiceEmbedderProfile,
) {
    let e = make();
    let e: &dyn VoiceEmbedder = e.as_ref();

    type Check = fn(&dyn VoiceEmbedder, &VoiceEmbedderProfile) -> Result<(), String>;
    let checks: [Check; 2] = [check_voice_embed, check_voice_embed_rejects_short];
    let failures: Vec<String> = checks
        .into_iter()
        .filter_map(|f| f(e, profile).err())
        .collect();
    if !failures.is_empty() {
        panic!(
            "gen-core voice-embedder conformance FAILED for `{}` (gen-core {}):\n  - {}",
            e.descriptor().id,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
