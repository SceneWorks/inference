//! # candle-gen-minimax-h3
//!
//! Native-Rust / candle inference for **MiniMax-H3 (Hailuo 3.0)** (`MiniMaxAI/MiniMax-H3`) — a
//! joint audio+video generation family. This is the Windows/Linux/CUDA sibling of
//! `mlx-gen-minimax-h3`, and it currently carries the two **VAE decode** paths (sc-17154).
//!
//! ## Video VAE
//!
//! - [`config`] carries the VAE geometry, the temporal knobs and the per-channel latent
//!   statistics, reconciling the two published configs (neither is sufficient alone);
//! - [`chunking`] models the `clip_length` / `token_drop` temporal chunk plan as pure integer
//!   arithmetic;
//! - [`rope`] implements the 3-D **partial** rotary embedding and its normalized token ids;
//! - [`blocks`] and [`decoder`] implement the 36-layer transformer decoder;
//! - [`vae_encoder`] is the 3-D causal CNN **encode** half — 118 published tensors, `fl2va`'s
//!   keyframe path (sc-19008), including the reference's spatial tiling, which is **on by
//!   default**;
//! - [`vae`] assembles the de-normalize → chunk → decode → cross-fade decode path and the
//!   `encode` → `quant_conv` → posterior path.
//!
//! The video VAE is unusual: the encoder is a 3-D causal CNN but the **decoder is a transformer**
//! (36 layers, 2048 dim, ~5.2 B params) that performs all 16× spatial and 4× temporal upsampling
//! in its output projection. Chunk/overlap/drop bookkeeping — not the blocks — is where that port
//! is most easily wrong, so it is isolated and asserted independently of any tensor math.
//!
//! ## Audio VAE
//!
//! - [`audio_config`] reconciles the FL2VA source triple (`config.json` + `config.yaml` +
//!   `metadata.json`) with the diffusers-repackaged root config, and reconstructs the five BigVGAN
//!   knobs that exist only in the reference's `sample_rate` branch;
//! - [`alias_free`] implements the Kaiser-sinc resamplers and the `SnakeBeta` periodic activation;
//! - [`audio_vae`] assembles `dec_in_proj` → BigVGAN → clamp, plus the stereo/interleave path that
//!   produces a `gen-core` `AudioTrack` at 32 kHz.
//!
//! The audio VAE is a **DAC-lineage encoder + BigVGAN decoder** — not an LTX-style audio VAE — and
//! its decoder is **mono**: stereo is two independent 32-channel latents decoded through the same
//! weights. Its numerical risk concentrates in 127 anti-aliased activations, so `SnakeBeta` and the
//! resamplers each carry their own parity fixture rather than only end-to-end coverage.
//!
//! ## Read [`layout`] before porting anything else in this family
//!
//! MiniMax publishes the checkpoint in the **converted diffusers layout**, and that conversion
//! applies tensor transforms which are *shape-identical* to their inputs — a swapped gated-FFN
//! projection and a fused-QKV reorder. No shape check, exhaustive-key-mapping proof or checksum can
//! see them; only an explicit assertion can. [`layout`] states the contract, backs it with
//! [`layout::split_gate_value`], and records how sc-18740 shipped a functionally wrong 36-layer
//! decoder past a fully green parity suite. **The DiT (sc-17155) inherits the same FFN swap plus a
//! grouped-QKV reorder.**
//!
//! ## Relationship to the MLX lane
//!
//! Both backends assert against the **same committed goldens**, produced from the official
//! `diffusers` `AutoencoderKLMiniMaxH3` and from the Apache-2.0 audio reference shipped inside the
//! snapshot. `tests/cross_backend.rs` additionally holds this port against the MLX lane's own
//! recorded decode of those goldens, and documents the noise floor that comparison sits on — and
//! what it therefore cannot detect.
//!
//! Two implementation choices differ deliberately, so the two lanes are not one implementation
//! typed twice:
//!
//! - the audio decoder runs in **NCL** here and NLC on MLX (each backend's native conv layout);
//! - candle's norms and `sdpa` are composed from primitives, where MLX calls fused kernels.
//!
//! One choice is deliberately kept **identical**: every transposed convolution runs as zero-insert
//! plus a forward convolution (`alias_free::transposed_conv1d`). On MLX that is forced by Metal's
//! reduced-precision `conv_transpose1d`; here it is a choice, made so the cross-backend residual is
//! attributable to matmul precision rather than to two different resampling algorithms.
//!
//! ## The DiT, the AdaLN evict and the joint denoise (sc-17155)
//!
//! - [`dit`] is the 50-block stack: [`dit::config`] the published geometry, [`dit::rope`] the
//!   3-axis **MM-RoPE**, [`dit::positions`] the per-modality coordinate conventions (audio's is the
//!   highest-risk detail in the port), [`dit::layers`] / [`dit::block`] / [`dit::refiner`] the
//!   modules, [`dit::heads`] the 17 mixed-precision input/output tensors, and [`dit::model`] the
//!   whole `MiniMaxH3Transformer3DModel`;
//! - [`dit::adaln`] is the **precompute-and-evict** lever — 26_020_915_200 B of `adaln_proj`
//!   released before denoise — and it is the module whose answer differs most from the MLX lane's,
//!   because candle is eager and its CUDA pool releases on *synchronize* rather than on a cache
//!   drain;
//! - [`denoise`] is the joint loop: [`denoise::geometry`] the AV time alignment,
//!   [`denoise::schedule`] the two sigma shifts and the reversed velocity sign,
//!   [`denoise::packing`] the packed sequence, and the loop itself, which runs **one** forward per
//!   step because the checkpoint is guidance-distilled.
//!
//! ## Not in this crate
//!
//! The Qwen3-VL-32B text encoder, the pipeline and measured `vramGbByTier` (sc-17156), and Ref2VA
//! (sc-17157). Nothing is registered with `candle-gen-catalog` — there is no generator to ship
//! until the pipeline lands. **The MLX sibling is not in this state**: it ships
//! `model::REGISTRATION` and `mlx-gen-catalog` calls its `register_providers`, which is what lets
//! its memory contract reach the registry while this crate's cannot (see [`register_providers`]).

pub mod alias_free;
pub mod audio_config;
pub mod audio_vae;
pub mod blocks;
pub mod chunking;
pub mod config;
pub mod decoder;
pub mod denoise;
pub mod dit;
pub mod layout;
pub mod memory_strategy;
pub mod nn;
pub mod rope;
pub mod spatial_tiling;
pub mod tensor;
pub mod vae;
pub mod vae_encoder;

pub use alias_free::{kaiser_sinc_filter1d, Activation1d, LowPassFilter1d, SnakeBeta, UpSample1d};
pub use audio_config::{
    BigVganConfig, MiniMaxH3AudioVaeConfig, ACTIVATION_KERNEL_SIZE, ACTIVATION_RESAMPLE_RATIO,
    AUDIO_LATENTS_MEAN, AUDIO_LATENTS_STD, AUDIO_LATENT_CHANNELS, AUDIO_OUTPUT_CHANNELS,
    AUDIO_SAMPLE_RATE, AUDIO_TOKEN_RATE_HZ,
};
pub use audio_vae::{AmpBlock1, BigVgan, MiniMaxH3AudioVae};
pub use blocks::{blend, TransformerBlock};
pub use chunking::{ChunkSpan, TemporalGeometry, TemporalPlan};
pub use config::{
    MiniMaxH3VaeConfig, CLIP_LENGTH, DECODER_HEAD_DIM, DECODER_NUM_HEADS, DECODER_NUM_LAYERS,
    DECODER_NUM_REGISTER_TOKENS, DECODER_ROPE_DIM_RATIO, DECODER_ROPE_THETA,
    ENCODER_BLOCK_OUT_CHANNELS, ENCODER_IN_CHANNELS, ENCODER_LAYERS_PER_BLOCK, ENCODER_NORM_EPS,
    ENCODER_NORM_NUM_GROUPS, ENCODER_SPATIAL_DOWNSAMPLE_FACTORS,
    ENCODER_TEMPORAL_DOWNSAMPLE_FACTORS, LATENTS_MEAN, LATENTS_STD, LATENT_CHANNELS, TOKEN_DROP,
    VAE_RATIO, VAE_RATIO_T,
};
pub use decoder::ViT3dDecoder;
pub use denoise::{
    adaln_schedule, denoise_av, DenoiseModality, JointGeometry, JointSchedule, JointStep,
    JointVelocity, PackedLayout, RowClass, SigmaSchedule, AUDIO_SIGMA_SHIFT, LEGAL_FRAME_COUNTS,
    NUM_ROW_CLASSES, VIDEO_SIGMA_SHIFT,
};
pub use dit::{
    release_device_memory, AdaLnCache, AdaLnResidency, DitBlock, JointDit, KeyframeAnchor,
    MiniMaxH3Dit, MiniMaxH3DitConfig, MmRope, TimestepSchedule, TokenRefiner, MODALITY_NUM,
    PUBLISHED_DIT_TENSORS,
};
pub use layout::{
    split_gate_value, swap_gated_halves, GatedFfnLayout, AUDIO_VAE_IS_UNCONVERTED,
    PUBLISHED_GATED_FFN_LAYOUT,
};
pub use rope::{create_token_ids, Rope3d, RopeTables};
pub use vae::{split_fused_qkv, MiniMaxH3VideoVae};
pub use vae_encoder::{
    conv3d_ncthw, reflect_pad_axis, stitch_tiles, zero_pad_front_time, CausalConv3d,
    DiagonalGaussian, DownBlock3d, FrameIsolatedGroupNorm, ResnetBlock3d, TilePlan, VideoEncoder3d,
    ENCODER_TEMPORAL_PADDING, ENCODER_TILING_IS_ON_BY_DEFAULT, LOGVAR_CLAMP,
    TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};

/// The published model id this crate targets. Matches the MLX sibling's, so a manifest entry names
/// one family across both backends.
pub const MODEL_ID: &str = "minimax_h3";

/// Pixel alignment a generated canvas must satisfy — the VAE's [`VAE_RATIO`] spatial compression
/// **times the DiT's width patch of 2**, i.e. 32. Same number as `mlx_gen_minimax_h3::SIZE_MULTIPLE`.
///
/// This is the reference's own product, not a guess: diffusers 0.40.0.dev0 @ 7564fb01 computes
/// `MiniMaxH3ModularPipeline.canvas_multiple` as `vae_spatial_compression_ratio * patch_size[2]`
/// (`modular_pipelines/minimax_h3/modular_pipeline.py:236`), and the released checkpoint supplies
/// both factors — `vae/config.json` has `spatial_downsample_factors = [2, 2, 2, 2, 1, 1]`, whose
/// product is the 16 of [`VAE_RATIO`], and `transformer/config.json` (and `transformer_ref/`) has
/// `patch_size = [1, 2, 2]`. The reference *rejects* an unaligned canvas rather than rounding it
/// (`before_denoise.py:389`, `before_encoder.py:393`).
///
/// Consumers read this for `requiresDimensionsMultipleOf`, so it must be the number the engine
/// actually enforces, not the looser VAE-only one. A 16-aligned canvas survives the VAE and then
/// has an odd number of latent columns, which has no patched representation at all. Until sc-19419
/// this crate advertised that looser 16 — correct while it was VAE-only, stale from the moment the
/// DiT landed in sc-17155 — and its own test pinned the stale value as if it were right.
pub const SIZE_MULTIPLE: u32 = VAE_RATIO as u32 * 2;

/// Add every provider this crate ships to an explicit registry builder — the sibling of
/// `mlx_gen_minimax_h3::register_providers`, and **the exact function `candle-gen-catalog` will
/// call** once there is something to call it for.
///
/// It registers the memory contract and its weights-free fixture and **no generator**, because this
/// crate has none: sc-17156 owns the pipeline. A builder in that state cannot `build()` —
/// `ProviderRegistryBuilder::build` rejects a memory-strategy registration whose `provider_id` has
/// no matching generator — which is precisely why the catalog line is still absent, and why adding
/// it today would break `candle_gen_catalog::provider_registry()` rather than wire anything up.
///
/// This function exists so that state is a **fact the tests can read off the crate's real
/// registration inventory** rather than a claim in a comment:
/// `memory_strategy::tests::a_generator_landing_here_forces_the_catalog_line` builds what this
/// returns, and the moment a `.register_generator(…)` line joins it the build succeeds and that
/// test demands the catalog line. Registering a generator straight from the catalog instead would
/// leave [`memory_strategy::MEMORY_REGISTRATION`] unregistered, so the same test also fails if the
/// catalog reaches this crate by any path while this function still ships no generator.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_CONTRACT_FIXTURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The crate ships no generator yet, so it is deliberately absent from `candle-gen-catalog`.
    /// This pins the compression contract the later pipeline slices will build on.
    ///
    /// **What this test does and does not guarantee.** It pins *this* crate's numbers against the
    /// diffusers reference (see [`SIZE_MULTIPLE`] for the provenance of the 32). It is blind, by
    /// construction, to whether `mlx-gen-minimax-h3` declares the same numbers — nothing in a
    /// per-crate assertion can see the sibling backend, which is exactly how the two crates came to
    /// disagree about `SIZE_MULTIPLE` (16 here vs 32 there) while this test stayed green on the
    /// wrong value and its comment claimed otherwise (sc-19419).
    ///
    /// The cross-backend claim is made where it can actually be checked: `check_cross_backend_geometry`
    /// in `scripts/check-workspace.py`, which reads both crates' sources and holds them to the
    /// released checkpoint. It lives there because the `workspace` lane is the one CI never skips —
    /// a Rust test here would not run on an mlx-only change, and one in the mlx crate would not run
    /// on Linux at all.
    ///
    /// Note the assertions below are *literals* on purpose. Restating `SIZE_MULTIPLE` as
    /// `VAE_RATIO * 2` would only repeat its own definition and could never fail.
    #[test]
    fn published_surface_is_the_vae_decode_contract() {
        assert_eq!(MODEL_ID, "minimax_h3");
        // 32, not the VAE-only 16: diffusers' `canvas_multiple` is
        // `vae_spatial_compression_ratio * patch_size[2]` = 16 * 2.
        assert_eq!(SIZE_MULTIPLE, 32);
        assert_eq!(VAE_RATIO, 16);
        assert_eq!(VAE_RATIO_T, 4);
        assert_eq!(LATENT_CHANNELS, 24);
        assert_eq!(LATENTS_MEAN.len(), LATENT_CHANNELS);
        assert_eq!(LATENTS_STD.len(), LATENT_CHANNELS);
    }

    /// The audio half's published envelope: 32 kHz stereo, 32 latent channels, 40 Hz tokens.
    #[test]
    fn audio_surface_is_the_published_envelope() {
        let cfg = MiniMaxH3AudioVaeConfig::default();
        assert_eq!(AUDIO_SAMPLE_RATE, 32_000);
        assert_eq!(AUDIO_OUTPUT_CHANNELS, 2);
        assert_eq!(AUDIO_LATENT_CHANNELS, 32);
        assert_eq!(AUDIO_LATENTS_MEAN.len(), AUDIO_LATENT_CHANNELS);
        assert_eq!(AUDIO_LATENTS_STD.len(), AUDIO_LATENT_CHANNELS);
        // 40 Hz tokens: a 5-second clip is 200 tokens, and each token is 800 samples.
        assert_eq!(cfg.hop_length(), 800);
        assert_eq!(cfg.token_rate_hz() as u32, AUDIO_TOKEN_RATE_HZ);
        assert_eq!(5 * AUDIO_TOKEN_RATE_HZ, 200);
    }
}
