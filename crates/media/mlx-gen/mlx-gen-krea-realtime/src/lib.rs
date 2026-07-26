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
//! ## Scope of this crate today (S2 load path + the S3 causal AR forward + the S4 AR loop)
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
//!   verbatim (the reused Wan attention gained one additive `forward_causal_chunk` variant), plus the
//!   read-only [`CausalKreaTransformer::forward_chunk_readonly`] variant S4 drives.
//! * [`scheduler`] — **S4**: the [`FewStepSchedule`] Self-Forcing few-step flow-matching scheduler
//!   (the shifted-sigma table + argmin timestep→sigma lookup, the Euler [`euler_x0`] step, and the
//!   [`renoise_step`]).
//! * [`generate`] — **S4/S5**: the autoregressive chunk driver [`generate_latents`] — per-chunk
//!   few-step denoise (CFG off, one forward per step) threading the persistent KV cache across chunks,
//!   emitting the latent sequence for a t2v clip. **S5** adds the togglable clean-context **KV-cache
//!   recompute** ([`KreaArConfig::do_kv_recomp`], default on): after a chunk's (now read-only) denoise
//!   steps, one forward reruns on the clean `x0` at [`KreaArConfig::context_noise`] and commits *that*
//!   clean-context K/V (exactly one commit per chunk either way).
//! * [`causal`] — **S5** also bounds the KV cache: [`CausalKvCache`] evicts older tail K/V beyond the
//!   sink + read window (unit-corrected to the reference's `local_attn_size × frame_seq_length` frames),
//!   so long clips stay memory-feasible on Mac. Pure cache slicing.
//!
//! **Attention-bias reconciliation (S5).** The block-causal mask ([`build_block_causal_mask`]) + the KV
//! read window + the causal RoPE offset are the *complete* causal mechanism: the released reference
//! sampling path (`wan/modules/causal_model.py` cached branch → plain `attention(q, k, v)`;
//! `wan/modules/attention.py` `attn_mask=None`; `flex_attention` gets only a structural `block_mask`,
//! never a `score_mod`) applies **no additive attention bias**. The "KV-cache attention bias" the Krea
//! blog describes (a negative bias on past-frame scores) is *not* present in the open-source code and no
//! magnitude is specified — so none is invented here.
//!
//! **VAE decode / clip assembly / `Generator` registration** (S6, including the first-frame VAE
//! re-anchor that re-encodes the first output frame — needs the VAE) and **i2v/v2v conditioning** (S7)
//! remain deferred; long-clip coherence measurement with real weights is S13.
//!
//! The converter and load path are validated against the S1 tensor inventory with synthesized
//! fixtures (`tests/`) — never the real 28.58 GB checkpoint. Real-weight byte-parity validation and
//! the MLX rehost to the SceneWorks HF org are the **gated** S2 remainder, tracked on sc-8435; a
//! registered `Generator` is deferred to S6.

pub mod causal;
pub mod config;
pub mod convert;
pub mod generate;
pub mod load;
pub mod scheduler;

pub use causal::{
    block_causal_mask, build_block_causal_mask, CausalKreaTransformer, CausalKvCache,
};
pub use config::{KreaArConfig, KreaRealtimeConfig, MODEL_ID};
pub use convert::{
    convert_krea_realtime_transformer, normalize_krea_keys, sanitize_krea_realtime_transformer,
    strip_model_prefix, KREA_MODEL_PREFIX, TRANSFORMER_DTYPE,
};
pub use generate::{generate_latents, generate_latents_into, ArGenParams};
pub use load::{
    expected_transformer_tensors, load_krea_realtime_transformer, verify_transformer_tensors,
    TensorSpec,
};
pub use scheduler::{euler_x0, renoise_step, FewStepSchedule, NUM_TRAIN_TIMESTEPS};

// Re-export the reused Wan config type so callers can name the DiT dimensions without a direct
// `mlx-gen-wan` dependency.
pub use mlx_gen_wan::config::WanModelConfig;
