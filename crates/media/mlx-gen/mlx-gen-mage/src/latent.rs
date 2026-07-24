//! Gaussian-Shading watermarked initial noise — **owned by sc-14104**.
//!
//! Port of `_vendor/mage_flow/models/modules/mage_latent.py`. **Not optional for parity:** the
//! reference computes a plain `randn` via `get_noise` and then *discards* it, replacing the initial
//! latent with `encode_noise(shape, key=…, seed=…)` on both the generation (`pipeline.py:307`) and
//! edit (`:506`) paths. There is no toggle — every real Mage-Flow output carries the watermark, and
//! a port that starts from plain `randn` cannot match any golden (measured max_abs 5.99 apart).
//!
//! Mechanism: payload [`GS_PAYLOAD`](crate::config::GS_PAYLOAD) → SHA-256-expanded
//! [`GS_MESSAGE_BITS`](crate::config::GS_MESSAGE_BITS)-bit message; a key-seeded per-entry XOR pad
//! and message-index map; per-entry `u ~ U(0,1)` from a seeded generator; then
//! `z = Φ⁻¹(clamp((half + u)/2, 1e-6, 1 - 1e-6))` — still ~N(0,1), so nothing downstream changes
//! shape or scale. Detection is `invert_to_noise` (reverse-Euler flow ODE from a clean latent,
//! empty prompt at cfg 1 — `pipeline.py:577`) followed by `decode_bits` (sign vote + binomial
//! p-value).
//!
//! ## Two divergences this story must decide, not inherit
//!
//! 1. **Key provisioning.** The reference resolves its key from a `MAGEFLOW_GS_KEY` env var or a
//!    `~/.mageflow/gs_key` keyfile (`mage_latent.py:12-14`). Neither is portable here: this
//!    workspace derives no paths and reads no production env side channels (the epic-13657
//!    guardrail in `scripts/check-workspace.py`). Surface the key through `LoadSpec` / the request
//!    instead, defaulting to [`GS_DEFAULT_KEY`](crate::config::GS_DEFAULT_KEY).
//! 2. **RNG equivalence.** `u` comes from a seeded `torch.Generator`; MLX's PRNG is a different
//!    stream, so bit-exact noise parity requires reproducing torch's uniform draw explicitly rather
//!    than calling MLX's `uniform`. Decide and pin this in the parity test — it is the difference
//!    between "matches the golden" and "is a valid watermark".
//!
//! Epic posture (recorded on sc-14105): **keep provenance marking, drop blocking.** This module
//! ships; the content classifier does not.
