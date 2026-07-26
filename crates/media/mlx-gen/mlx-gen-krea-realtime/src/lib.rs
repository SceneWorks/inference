//! Krea **Realtime Video 14B** — native MLX provider crate (epic 8431, sc-8435 S2).
//!
//! Krea Realtime 14B is an autoregressive, self-forcing, real-time text-to-video denoiser. The S1
//! audit established the load-critical fact this crate is built on: **the shipped checkpoint is Wan
//! 2.1 T2V 14B, weight-for-weight.** It is transformer-only (1095 tensors, all F16,
//! 14,288,491,584 params), its DiT dimensions are exactly [`WanModelConfig::wan21_t2v_14b`], and its
//! tensor **set** is identical to stock Wan 2.1 T2V 14B. So the DiT, z16 VAE, UMT5-XXL text encoder,
//! 3-axis RoPE, patchify, and schedulers are all **reused from [`mlx_gen_wan`]** rather than
//! reimplemented here.
//!
//! What makes Krea Realtime distinct from stock Wan is its **inference regime**, not its weights: a
//! short per-frame-block `denoising_step_list`, a rolling KV cache, and (optional) local/block-sparse
//! causal attention. Those knobs live on [`KreaArConfig`] so the preset is complete, but they are
//! **not consumed until S3–S5** (causal attention / KV cache / the AR self-forcing loop). Keep this
//! crate mentally distinct from the unrelated **image** crate `mlx-gen-krea` (Krea 2 Turbo, engine
//! `krea_2_turbo`).
//!
//! ## Scope of this crate today (S2 load path + the S3 causal AR forward)
//!
//! * [`config`] — the [`KreaRealtimeConfig`] preset (`wan21_t2v_14b` DiT dims + the AR knobs).
//! * [`convert`] — [`sanitize_krea_realtime_transformer`]: collapse either on-disk layout (single-file
//!   `model.`-prefixed or sharded `transformer/` bare) to the plain Wan native names, map them onto
//!   the internal [`mlx_gen_wan::WanTransformer`] layout via the shared Wan sanitizer, and cast
//!   F16 → bf16.
//! * [`load`] — [`load_krea_realtime_transformer`]: sanitize a native tensor map, **verify every
//!   expected tensor is present at its config-derived shape** ([`verify_transformer_tensors`]), and
//!   load the result into the reused Wan DiT.
//! * [`causal`] — **S3**: the causal autoregressive transformer forward + persistent KV cache
//!   ([`CausalKreaTransformer`], [`CausalKvCache`], [`build_block_causal_mask`]). The one net-new
//!   compute delta over stock Wan is confined to self-attention — a block-causal mask, a per-layer
//!   post-RoPE KV cache, and a causal RoPE temporal offset — reusing every other Wan DiT piece
//!   verbatim (the reused Wan attention gained one additive `forward_causal_chunk` variant). The
//!   Self-Forcing few-step scheduler / AR chunk loop (S4), KV-cache recompute (S5), VAE decode (S6),
//!   and `Generator` registration (S7) are still deferred.
//!
//! The converter and load path are validated against the S1 tensor inventory with synthesized
//! fixtures (`tests/`) — never the real 28.58 GB checkpoint. Real-weight byte-parity validation and
//! the MLX rehost to the SceneWorks HF org are the **gated** S2 remainder, tracked on sc-8435; a
//! registered `Generator` is deferred to S6.

pub mod causal;
pub mod config;
pub mod convert;
pub mod load;

pub use causal::{
    block_causal_mask, build_block_causal_mask, CausalKreaTransformer, CausalKvCache,
};
pub use config::{KreaArConfig, KreaRealtimeConfig, MODEL_ID};
pub use convert::{
    convert_krea_realtime_transformer, normalize_krea_keys, sanitize_krea_realtime_transformer,
    strip_model_prefix, KREA_MODEL_PREFIX, TRANSFORMER_DTYPE,
};
pub use load::{
    expected_transformer_tensors, load_krea_realtime_transformer, verify_transformer_tensors,
    TensorSpec,
};

// Re-export the reused Wan config type so callers can name the DiT dimensions without a direct
// `mlx-gen-wan` dependency.
pub use mlx_gen_wan::config::WanModelConfig;
