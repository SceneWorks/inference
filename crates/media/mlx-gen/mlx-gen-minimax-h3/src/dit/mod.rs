//! The **MiniMax-H3 DiT block stack** and its 3-axis MM-RoPE (sc-17144).
//!
//! `MiniMaxH3Transformer3DModel` runs **one stack of 50 blocks over a single packed 1-D sequence**
//! holding the text condition, the conditioning image/video rows, the audio rows and the target
//! video rows. Attention is full self-attention over that sequence: there is no cross-attention,
//! no per-modality block weights, and no dual stream. Modality enters only through
//!
//! * the two input patch projections and `context_embedder`,
//! * the per-row AdaLN modality tag,
//! * the two output heads.
//!
//! # Geometry (`transformer/config.json`)
//!
//! 50 layers, `hidden_size` 5376, 56 heads × 128 — so the attention inner width is **7168, wider
//! than the residual stream**, which a `hidden_size / heads` derivation silently gets wrong.
//! `ffn_dim` 14336, `text_dim` 5120 with no projection on the text-encoder side,
//! `rope_freq_dim` 16 (⇒ 96 of 128 head channels rotate), `token_refiner` 2 layers.
//!
//! # What this module carries
//!
//! | | |
//! |---|---|
//! | [`config`] | the published geometry |
//! | [`rope`] | MM-RoPE — the `(t, h, w)` axis split and the partial rotary |
//! | [`positions`] | the per-modality `(t, h, w)` coordinate conventions, **including audio's** |
//! | [`block`] | `MiniMaxH3TransformerBlock` + its AdaLN projection |
//! | [`adaln`] | the schedule-keyed modulation precompute and the **26.02 GB evict** (sc-17145) |
//! | [`refiner`] | the 2-layer text token refiner |
//! | [`qkv`] | the published fused-QKV transform, as an executable contract |
//!
//! Deliberately **not** here: the joint denoise loop and the packed-sequence assembly, which are
//! [`crate::denoise`] (sc-17146); and, each owned by a later story, the pipeline and the
//! input/output projections that wrap this stack — including the timestep MLP
//! [`adaln::AdaLnCache::precompute`] takes as a closure — (sc-17147), and Ref2VA's
//! `transformer_ref` (sc-17149).
//!
//! # Read [`crate::layout`] first
//!
//! The DiT carries **both** conversion transforms the video VAE taught this crate about: the
//! gated-FFN half-swap (same `[value | gate]` published layout, via
//! [`crate::layout::split_gate_value`]) and a fused-QKV reorder whose DiT arm is *not* the video
//! VAE's — see [`qkv`].

pub mod adaln;
pub mod block;
pub mod config;
pub mod heads;
pub mod layers;
pub mod model;
pub mod positions;
pub mod qkv;
pub mod refiner;
pub mod rope;

pub use adaln::{AdaLnCache, AdaLnResidency, ScheduleKey, TimestepSchedule};
pub use block::{AdaLnModulation, AdaLnProjection, DitBlock};
pub use config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
pub use heads::{
    timestep_sincos, AdaLayerNormOut, DitProjections, LinearBias, NormOutModulation,
    TimestepEmbedder,
};
pub use layers::{DitAttention, DitFeedForward, LinearNoBias, RmsNorm};
pub use model::{JointDit, MiniMaxH3Dit, PUBLISHED_DIT_TENSORS};
pub use positions::{
    audio_position_ids, frame_grid, keyframe_anchor_time, keyframe_position_ids, spatial_axis_grid,
    temporal_grid, text_position_ids, video_position_ids, KeyframeAnchor, AUDIO_CHANNELS,
    ROPE_FRAMES_PER_LATENT, ROPE_FRAME_RESCALE, ROPE_SPATIAL_SCALE,
};
pub use refiner::{TokenRefiner, TokenRefinerBlock};
pub use rope::{MmRope, MmRopeTables};
