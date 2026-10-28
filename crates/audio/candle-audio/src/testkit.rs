//! Test-support seam for the candle audio provider crates (`testkit` feature, sc-12835).
//!
//! This module is the audio family's home for **cache-free** shared test helpers, enabled by
//! provider crates under `[dev-dependencies]` (mirroring `candle_gen::testkit`). It is compiled
//! explicitly in CI via `--features candle-audio/testkit` (ci.yml) so the seam is exercised even
//! when no in-tree target dev-depends on it (the sc-11990 cfg-hole guard).
//!
//! The HF-cache snapshot resolvers this module once carried were removed under epic 13657:
//! inference never self-fetches or derives a local model-cache location. Real-weight tests now
//! take a passed-in snapshot directory from a `<CRATE>_SNAPSHOT` env var instead of scanning a
//! derived cache path. Add new shared, cache-free audio test helpers here as they are needed.

/// A minimal valid safetensors byte blob whose `names` are zero-size F32 tensors — the snapshot
/// preparers' header-only fixture (sc-27265). `"__metadata__"` among `names` is written as an
/// ordinary entry, which the header readers skip.
pub fn tiny_safetensors(names: &[&str]) -> Vec<u8> {
    let entries: Vec<String> = names
        .iter()
        .map(|n| format!(r#""{n}": {{"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}}"#))
        .collect();
    let header = format!("{{{}}}", entries.join(", "));
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out
}
