//! Timestep conditioning — **owned by sc-14040**.
//!
//! `MageFlowTimestepProjEmbeddings` (`_vendor/mage_flow/models/modules/mage_layers.py:24-104`):
//! `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000, max_period=10000)`
//! → `TimestepEmbedding(256 → hidden_size)`. Constants live in [`crate::config`]
//! ([`FREQUENCY_EMBEDDING_SIZE`](crate::config::FREQUENCY_EMBEDDING_SIZE),
//! [`TIMESTEP_SCALE`](crate::config::TIMESTEP_SCALE),
//! [`TIMESTEP_MAX_PERIOD`](crate::config::TIMESTEP_MAX_PERIOD)).
//!
//! Two traps:
//!
//! 1. The input is the scheduler **sigma ∈ [0, 1]**, fed straight in (`pipeline.py:189`) — not a
//!    0..1000 timestep index. The `scale=1000` inside the embedder is what maps it.
//! 2. The sinusoidal frequency table is **deliberately rounded to bf16** (`mage_layers.py:45`).
//!    The model was trained with that rounding, so computing it in f32 is a divergence, not an
//!    improvement ([`TIMESTEP_FREQS_BF16`](crate::config::TIMESTEP_FREQS_BF16)).
//!
//! The reference also computes a pooled text vector `vec`, then throws it away: the DiT overwrites
//! it with zeros before `temb = temb + txt_vec` (`models/mage_flow.py:116-118`). A port needs no
//! pooled text vector.
