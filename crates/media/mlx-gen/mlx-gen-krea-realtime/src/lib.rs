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
//! causal attention. That autoregressive regime is **adapted from** the reference implementation
//! (`krea-ai/realtime-video`) — an architecture reimplementation in native MLX, not a copy of its
//! source. Those knobs live on [`KreaArConfig`] so the preset is complete, but they are
//! **not consumed until S3–S5** (causal attention / KV cache / the AR self-forcing loop). Keep this
//! crate mentally distinct from the unrelated **image** crate `mlx-gen-krea` (Krea 2 Turbo, engine
//! `krea_2_turbo`).
//!
//! ## Scope of this crate today (S2 load path → S3 causal AR forward → S4/S5 AR loop → S6 t2v pipeline → S7 i2v/v2v)
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
//!   clean-context K/V (exactly one commit per chunk either way). **S7** adds the i2v/v2v conditioning
//!   surface ([`generate_i2v_latents`] / [`generate_v2v_latents`] over [`RefConditioning`]): a
//!   VAE-encoded reference still **warms** the KV cache from clean context (the S5 commit applied at
//!   timestep 0, `causal_inference.py:136-170`) and is prepended to the output (i2v), or a VAE-encoded
//!   source clip drives a **strength-controlled** init ([`FewStepSchedule::for_strength`],
//!   `v2v.py::get_denoising_schedule`) so a lower strength preserves more of the source (v2v).
//! * [`causal`] — **S5** also bounds the KV cache: [`CausalKvCache`] evicts older tail K/V beyond the
//!   sink + read window (unit-corrected to the reference's `local_attn_size × frame_seq_length` frames),
//!   so long clips stay memory-feasible on Mac. Pure cache slicing.
//! * [`causal`] — **sc-17807** makes the cache's *per-token cost* a knob on top of that *token count*.
//!   The cache holds activations, so a Q4 DiT does not shrink it: a DiT token costs **800 KiB** of
//!   bf16 KV ([`KreaRealtimeConfig::kv_bytes_per_token`]), which for an AR video model outweighs the
//!   ~9 GiB of weights. [`KreaArConfig::kv_cache_quant`] stores K/V group-wise-quantized and
//!   dequantizes the read window per layer (there is no fused quantized SDPA to attend over packed
//!   K/V with — see [`KvCacheQuant`]); Q8 measures **0.53×** the bf16 cache. It defaults to `None`,
//!   because a cheaper cache perturbs the same long-clip coherence sc-15127/sc-15571 measure and
//!   turning it on is therefore a measured decision rather than a free one.
//!
//! **Attention-bias reconciliation (S5).** The block-causal mask ([`build_block_causal_mask`]) + the KV
//! read window + the causal RoPE offset are the *complete* causal mechanism: the released reference
//! sampling path (`wan/modules/causal_model.py` cached branch → plain `attention(q, k, v)`;
//! `wan/modules/attention.py` `attn_mask=None`; `flex_attention` gets only a structural `block_mask`,
//! never a `score_mod`) applies **no additive attention bias**. The "KV-cache attention bias" the Krea
//! blog describes (a negative bias on past-frame scores) is *not* present in the open-source code and no
//! magnitude is specified — so none is invented here.
//!
//! * [`t2v`] — **S6**: the text-to-video pipeline orchestration — `prompt → UMT5 encode → context →
//!   [`generate_latents`] → z16 Wan VAE decode → assembled clip` ([`t2v::generate_t2v`]), with the
//!   weight-free component seam [`t2v::generate_t2v_from_components`] and the Mac streaming-window bound
//!   ([`t2v::mac_ar_config`], the sc-8438 S5 follow-up). The VAE decode + `frames_to_images` clip
//!   assembly reuse `mlx_gen_wan` (`decode_to_frames`). BATCH form = whole-latent single-pass decode;
//!   the streaming per-chunk feat-cached decode is the realtime path (streaming epic).
//! * [`t2v`] — **S7** also adds the i2v/v2v pipeline seams ([`generate_i2v_from_components`] /
//!   [`generate_v2v_from_components`] weight-free component seams, [`generate_i2v`] / [`generate_v2v`]
//!   real-weight paths): VAE-**encode** the reference still (`.mode()`) / source clip (`.sample()`,
//!   reusing `mlx_gen_wan`'s `WanVae::encode`/`encode_sample` + `preprocess_i2v_image`) → the S7 AR
//!   conditioning → VAE decode.
//! * [`pipeline`] — **S6**: the registered [`mlx_gen::Generator`] ([`pipeline::KreaRealtime`]) + its
//!   [`pipeline::descriptor`] (**CFG-off** video, the Self-Forcing few-step sampler) +
//!   [`register_providers`], so `provider_registry().load("krea_realtime_14b", spec)` yields a working
//!   provider. **S7** advertises i2v (`Reference`) + v2v (`VideoClip`) conditioning on the descriptor
//!   and routes a request's conditioning to [`generate_i2v`] / [`generate_v2v`]. The crate is composed
//!   into the platform via `mlx-gen-catalog` (explicit composition, not linker discovery).
//!
//! The reference's **first-frame VAE re-anchor** (`release_server.py::get_clean_context_frames`,
//! re-encoding the first decoded output frame as a persistent clean-context anchor) re-encodes decoded
//! pixels *mid-generation*, so that *mechanism* is streaming-coupled and correctly out of this batch path
//! (which decodes once at the end). The bounded Mac window runs for every clip and the shipped 14B
//! config sets `sink_size = 0`, so a long batch clip slides its window with no persistent anchor.
//! **sc-15127 (S18) measured that on real weights (q4, three seeds, 13 window rolls) and found a long
//! clip *does* drift, well past the measurement's budget** — headline mode a colour-cast/tone drift
//! (the blue–yellow opponent axis wins every row-A cell at 832×480), alongside a separately observable
//! saturation rise. **Whether the bounded window causes it is not resolved**: an enlarged
//! within-regime dose ladder at 13/10/5 eviction rolls fits +0.571 ±1.897/255 per roll at 640×384 and
//! −0.278 ±1.678/255 per roll at 832×480 (mean ± the predeclared 2·SEM heuristic). Neither direction
//! clears the heuristic. Across the full eight-roll span, effects below practical floors of 19.75/255
//! and 15.65/255 respectively remain unresolved. So no sink anchor is wired — the sink comparison is
//! likewise unresolved at three seeds — the `sink_size` knob stays plumbed for a checkpoint that ships
//! one, and the drift itself is tracked as **sc-15571**.
//! See [`t2v::generate_t2v_from_components`] for the table and the limits of the claim.
//! **i2v/v2v conditioning** (S7) is now wired (see [`generate`] / [`t2v`] above); its real-weight
//! watchable-clip coherence overlaps the S13 real-weight validation.
//!
//! The converter, load, AR, and pipeline paths are validated against the S1 tensor inventory with
//! synthesized / tiny random-weight fixtures (`tests/`) — never the real 28.58 GB checkpoint. The
//! **real-weight watchable-clip e2e** (the real 28 GB DiT + stock Wan VAE/UMT5 → a watchable t2v clip)
//! is the **gated** S6 remainder, overlapping the S13 real-weight validation; the MLX rehost to the
//! SceneWorks HF org is the gated S2 remainder (sc-8435).

pub mod causal;
pub mod config;
pub mod convert;
pub mod generate;
pub mod load;
pub mod pipeline;
pub mod scheduler;
pub mod t2v;

pub use causal::{
    block_causal_mask, build_block_causal_mask, CausalKreaTransformer, CausalKvCache,
};
pub use config::{KreaArConfig, KreaRealtimeConfig, KvCacheQuant, MODEL_ID};
pub use convert::{
    convert_krea_realtime_tier, convert_krea_realtime_tier_sharded,
    convert_krea_realtime_tier_with_config, convert_krea_realtime_transformer, normalize_krea_keys,
    quantize_krea_realtime_transformer, sanitize_krea_realtime_transformer, strip_model_prefix,
    DEFAULT_SHARD_BYTES, DIT_FILE, KREA_MODEL_PREFIX, TRANSFORMER_DIR, TRANSFORMER_DTYPE,
};
pub use generate::{
    generate_i2v_latents, generate_latents, generate_latents_conditioned_into,
    generate_latents_into, generate_v2v_latents, ArGenParams, RefConditioning,
};
pub use load::{
    expected_transformer_tensors, load_krea_realtime_transformer,
    load_krea_realtime_transformer_with_quant, probe_packed_quant, resolve_load_time_quant,
    resolve_snapshot_quant, verify_transformer_tensors, TensorSpec, PACKED_LINEARS_PER_BLOCK,
};
pub use pipeline::{descriptor, load as load_generator, KreaRealtime, SELF_FORCING_SAMPLER};
pub use scheduler::{euler_x0, renoise_step, FewStepSchedule, NUM_TRAIN_TIMESTEPS};
pub use t2v::{
    decode_latents_to_video, decode_tiling, generate_i2v, generate_i2v_from_components,
    generate_t2v, generate_t2v_from_components, generate_v2v, generate_v2v_from_components,
    mac_ar_config, KreaRealtimeJob,
};

// Re-export the reused Wan config types so callers can name the DiT dimensions — and a snapshot's
// pre-quantized tier (sc-15203) — without a direct `mlx-gen-wan` dependency.
pub use mlx_gen_wan::config::{WanModelConfig, WanQuant};

/// Add the MLX Krea Realtime provider to an explicit media registry builder. Composed by the platform
/// catalog (explicit, never linker-discovered) — see `mlx-gen-catalog`.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(pipeline::REGISTRATION)
}

/// Build the complete explicit MLX Krea Realtime provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    /// The crate's explicit catalog exposes exactly the registered `krea_realtime_14b` generator, and
    /// `provider_registry().load("krea_realtime_14b", spec)` resolves it (a real load would then stage
    /// weights — here we only assert the id is discoverable + routable). Weights-free.
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(explicit, ["krea_realtime_14b"]);
        // The descriptor sweep passes for the whole (one-generator) catalog.
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );

        // `load(id, spec)` routes to this provider by id (it fails on the nonexistent snapshot dir, not
        // on an unknown id) — proving `mlx_gen::load("krea_realtime_14b", spec)` reaches the generator.
        use mlx_gen::{LoadSpec, WeightsSource};
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-krea-realtime".into()));
        let err = registry
            .load("krea_realtime_14b", &spec)
            .err()
            .expect("nonexistent snapshot must fail to load")
            .to_string();
        assert!(
            err.contains("does not exist") || err.contains("snapshot"),
            "expected a snapshot-load error (id resolved), got: {err}"
        );
        // An unknown id is a distinct error.
        let unknown = registry
            .load("not_a_model", &spec)
            .err()
            .expect("unknown id must fail")
            .to_string();
        assert!(
            unknown.contains("no generator registered"),
            "got: {unknown}"
        );
    }
}
