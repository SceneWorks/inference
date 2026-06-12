//! The Lens text encoder — gpt-oss-20b run encoder-only (forward to layer 23, capture hidden states
//! at layers `[5, 11, 17, 23]`). See epic 3164.
//!
//! This slice (sc-3165) ports the **attention core** of a gpt-oss decoder layer; the MoE
//! feed-forward + full layer/stack assembly (sc-3166) and the multi-layer capture (sc-3171) follow.

pub mod gpt_oss;
