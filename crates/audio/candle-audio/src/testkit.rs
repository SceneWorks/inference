//! Test-support seam for the candle audio provider crates (`testkit` feature, sc-12835).
//!
//! This module is the audio family's home for **cache-free** shared test helpers, enabled by
//! provider crates under `[dev-dependencies]` (mirroring `candle_gen::testkit`). It is compiled
//! explicitly in CI via `--features candle-audio/testkit` (ci.yml) so the seam is exercised even
//! though no in-tree target currently dev-depends on it (the sc-11990 cfg-hole guard).
//!
//! The HF-cache snapshot resolvers this module once carried were removed under epic 13657:
//! inference never self-fetches or derives a local model-cache location. Real-weight tests now
//! take a passed-in snapshot directory from a `<CRATE>_SNAPSHOT` env var instead of scanning a
//! derived cache path. Add new shared, cache-free audio test helpers here as they are needed.
