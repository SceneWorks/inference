//! Contract conformance for [`gen_core::AudioEmbedder`] providers (sc-12851) — a CLAP-style dual
//! encoder that maps audio and text into one joint space (`candle-audio-clap`). The audio parallel of
//! the image [`ImageEmbedder`](gen_core::ImageEmbedder). The defining guarantees this suite enforces
//! are the ones the joint-retrieval math depends on: [`embed`](gen_core::AudioEmbedder::embed) and
//! [`embed_text`](gen_core::AudioEmbedder::embed_text) return vectors of the **same** length (the
//! advertised `embedding_dim`), both **finite** and **L2-normalized** (CLAP's native retrieval
//! feature, so cosine similarity is a plain dot product) — a provider that returns un-normalized,
//! mismatched-dim, or NaN vectors would silently corrupt audio↔text ranking.

use gen_core::AudioEmbedder;

/// Tolerance on the L2 norm: a conformant provider returns unit vectors, but floating-point
/// accumulation leaves a small residue, so accept `|‖v‖ - 1| <= EPS`.
const NORM_EPS: f32 = 1e-3;

/// Parameters for an audio-embedder conformance run — one in-range clip and one text query, both
/// embedded into the joint space and checked for shape / finiteness / normalization.
#[derive(Clone, Debug)]
pub struct AudioEmbedderProfile {
    pub clip: gen_core::AudioTrack,
    pub query: String,
}

impl Default for AudioEmbedderProfile {
    fn default() -> Self {
        Self {
            clip: gen_core::AudioTrack {
                samples: vec![0.01; 48_000],
                sample_rate: 48_000,
                channels: 1,
                ..Default::default()
            },
            query: "a dog barking".to_owned(),
        }
    }
}

impl AudioEmbedderProfile {
    pub fn cheap() -> Self {
        Self::default()
    }
}

/// **Joint-space shape + normalization.** `embed(clip)` and `embed_text(query)` each return a vector
/// of exactly [`AudioEmbedderDescriptor::embedding_dim`](gen_core::AudioEmbedderDescriptor::embedding_dim)
/// finite elements, both L2-normalized to unit length, and of the **same** length — so
/// `cosine(embed_text(query), embed(clip))` is a well-defined dot product over one shared space.
pub fn check_audio_embed_joint(
    e: &dyn AudioEmbedder,
    profile: &AudioEmbedderProfile,
) -> Result<(), String> {
    let desc = e.descriptor();
    let id = desc.id;
    let dim = desc.embedding_dim;
    if dim == 0 {
        return Err(format!(
            "embed[{id}]: descriptor advertises embedding_dim 0 — an audio embedder must have a \
             fixed, non-zero dimension"
        ));
    }

    let audio = e
        .embed(&profile.clip)
        .map_err(|err| format!("embed[{id}]: embed() failed on the valid clip: {err}"))?;
    check_vector(id, "embed()", &audio, dim)?;

    let text = e
        .embed_text(&profile.query)
        .map_err(|err| format!("embed[{id}]: embed_text() failed on the query: {err}"))?;
    check_vector(id, "embed_text()", &text, dim)?;

    if audio.len() != text.len() {
        return Err(format!(
            "embed[{id}]: embed() returned {} elements but embed_text() returned {} — the audio and \
             text vectors must share the joint dimension",
            audio.len(),
            text.len()
        ));
    }
    Ok(())
}

/// One vector's shape / finiteness / unit-norm floor.
fn check_vector(id: &str, op: &str, v: &[f32], dim: usize) -> Result<(), String> {
    if v.len() != dim {
        return Err(format!(
            "embed[{id}]: {op} returned {} elements, but the descriptor advertises embedding_dim \
             {dim}",
            v.len()
        ));
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(format!(
            "embed[{id}]: {op}[{i}] is non-finite ({}) — a joint-space vector must be finite",
            v[i]
        ));
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if (norm - 1.0).abs() > NORM_EPS {
        return Err(format!(
            "embed[{id}]: {op} returned a vector with L2 norm {norm:.4}, expected ~1.0 — CLAP-style \
             retrieval requires L2-normalized vectors so cosine similarity is a plain dot product"
        ));
    }
    Ok(())
}

/// **Registry round-trip.** The embedder's descriptor `id` is present in the explicit registry
/// supplied by the caller.
pub fn check_audio_embedder_registry(
    registry: &gen_core::ProviderRegistry,
    e: &dyn AudioEmbedder,
) -> Result<(), String> {
    let id = e.descriptor().id;
    if registry
        .audio_embedders()
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

/// Run the full audio-embedder conformance suite against a freshly-`make`d embedder. Panics with
/// every failure aggregated.
pub fn audio_embedder_conformance(
    make: impl Fn() -> Box<dyn AudioEmbedder>,
    profile: &AudioEmbedderProfile,
) {
    let e = make();
    let e: &dyn AudioEmbedder = e.as_ref();

    let failures: Vec<String> = [check_audio_embed_joint(e, profile)]
        .into_iter()
        .filter_map(|r| r.err())
        .collect();
    if !failures.is_empty() {
        panic!(
            "gen-core audio-embedder conformance FAILED for `{}` (gen-core {}):\n  - {}",
            e.descriptor().id,
            gen_core::VERSION,
            failures.join("\n  - ")
        );
    }
}

#[cfg(test)]
mod tests;
