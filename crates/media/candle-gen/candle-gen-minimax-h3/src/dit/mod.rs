//! The **MiniMax-H3 DiT block stack** and its 3-axis MM-RoPE — the candle sibling of
//! `mlx_gen_minimax_h3::dit`.
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
//! | [`layers`] | the bias-free attention and SwiGLU feed-forward every block is built from |
//! | [`block`] | `MiniMaxH3TransformerBlock` + its AdaLN projection |
//! | [`adaln`] | the schedule-keyed modulation precompute and the **26.02 GB evict** |
//! | [`refiner`] | the 2-layer text token refiner |
//! | [`heads`] | the 17 input/output tensors, and the timestep MLP |
//! | [`model`] | the whole model, and the `JointVelocity` the denoise loop drives |
//! | [`qkv`] | the published fused-QKV transform, as an executable contract |
//!
//! Deliberately **not** here: the joint denoise loop and the packed-sequence assembly, which are
//! [`crate::denoise`]; and, each owned by a later story, the pipeline and measured `vramGbByTier`
//! (sc-17156) and Ref2VA's `transformer_ref` (sc-17157).
//!
//! # Read [`crate::layout`] first
//!
//! The DiT carries **both** conversion transforms the video VAE taught this crate about: the
//! gated-FFN half-swap (same `[value | gate]` published layout, via
//! [`crate::layout::split_gate_value`]) and a fused-QKV reorder whose DiT arm is *not* the video
//! VAE's — see [`qkv`].
//!
//! # Where this lane deliberately differs from the MLX one
//!
//! The two ports are held to the same committed goldens, not typed twice from one design. Three
//! choices differ, each for a stated reason, and each is what makes cross-backend agreement
//! evidence rather than tautology:
//!
//! * **the rotary tables are built on the host in f64** ([`rope`]), where MLX multiplies on device
//!   at f32;
//! * **attention materializes its scores in bounded blocks** ([`layers`]), where MLX's fused kernel
//!   streams them — a genuine memory-shape difference, not a detail;
//! * **the eviction synchronizes rather than draining an allocator cache** ([`adaln`]), because
//!   candle is eager and its CUDA pool releases on synchronize.

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

pub use adaln::{release_device_memory, AdaLnCache, AdaLnResidency, ScheduleKey, TimestepSchedule};
pub use block::{AdaLnModulation, AdaLnProjection, DitBlock};
pub use config::{MiniMaxH3DitConfig, MODALITY_NUM, MODULATION_PARAMS};
pub use heads::{
    timestep_sincos, AdaLayerNormOut, DitProjections, LinearBias, NormOutModulation,
    TimestepEmbedder,
};
pub use layers::{DitAttention, DitFeedForward, LinearNoBias, RmsNorm};
pub use model::{BlockModulation, JointDit, MiniMaxH3Dit, PackedForward, PUBLISHED_DIT_TENSORS};
pub use positions::{
    audio_position_ids, frame_grid, keyframe_anchor_time, keyframe_position_ids,
    reference_block_position_ids, spatial_axis_grid, temporal_grid, text_position_ids,
    video_position_ids, KeyframeAnchor, ReferenceBlockRows, ReferenceLatentGeometry,
    AUDIO_CHANNELS, ROPE_FRAMES_PER_LATENT, ROPE_FRAME_RESCALE, ROPE_SPATIAL_SCALE,
};
pub use refiner::{TokenRefiner, TokenRefinerBlock};
pub use rope::{MmRope, MmRopeTables};
