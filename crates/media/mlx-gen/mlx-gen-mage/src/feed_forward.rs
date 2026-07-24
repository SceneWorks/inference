//! Per-stream feed-forward network — **owned by sc-14040**.
//!
//! `FeedForward(dim, dim * mlp_ratio, activation_fn="gelu-approximate")`
//! (`_vendor/mage_flow/models/modules/mage_layers.py:547`, `:557`) on **both** streams.
//!
//! **This is not SwiGLU.** The epic's original reuse note assumed the `mlx-gen-z-image` sibling's
//! SwiGLU FFN because Mage's own config declares `schedule_mode: "z-image"`; the vendored code says
//! otherwise. [`crate::config::FFN_ACTIVATION`] pins it so the sibling's activation cannot be
//! inherited by accident, and [`crate::config::MLP_RATIO`] pins the 4.0 expansion (hardcoded in
//! the reference's code, *not* read from `transformer/config.json`).
