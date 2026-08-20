//! # mlx-gen-minimax-h3
//!
//! Native-Rust / MLX inference for **MiniMax-H3 (Hailuo 3.0)** (`MiniMaxAI/MiniMax-H3`) — a joint
//! audio+video generation family.
//!
//! # The weights are NOT Apache-2.0 (corrected in sc-17147)
//!
//! This crate's own code is Apache-2.0; the **model** is not, and the two were conflated here until
//! the licence was read. `MiniMaxAI/MiniMax-H3` declares `license: other` /
//! `minimax-h3-community-license-agreement`, and that agreement is **territorially exclusive**: §I.3
//! and §I.5 define its "Applicable Territory" as worldwide **excluding the European Union, the
//! United Kingdom, the Republic of Korea and the United States of America**, and §V.4 forbids use
//! outside it. §IV.1 also imposes a $20 M revenue ceiling. The bundled `text_encoder/` is the one
//! exception: it is byte-identical upstream Qwen3-VL-32B-Instruct under Apache-2.0, which the
//! LICENSE itself says. Both halves are rowed in
//! `gen_core::license::components` — see `MINIMAX_H3_COMMUNITY` for the transcription.
//!
//! ## Video VAE (sc-17140)
//!
//! - [`config`] carries the VAE geometry, the temporal knobs and the per-channel latent
//!   statistics, reconciling the two published configs (neither is sufficient alone);
//! - [`chunking`] models the `clip_length` / `token_drop` temporal chunk plan as pure integer
//!   arithmetic;
//! - [`rope`] implements the 3-D **partial** rotary embedding and its normalized token ids;
//! - [`blocks`] and [`decoder`] implement the 36-layer transformer decoder;
//! - [`vae`] assembles the de-normalize → chunk → decode → cross-fade decode path.
//!
//! The video VAE is unusual: the encoder is a 3-D causal CNN but the **decoder is a transformer**
//! (36 layers, 2048 dim, ~5.2 B params) that performs all 16× spatial and 4× temporal upsampling
//! in its output projection. Chunk/overlap/drop bookkeeping — not the blocks — is where that port
//! is most easily wrong, so it is isolated and asserted independently of any tensor math.
//!
//! ## Audio VAE (sc-17141)
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
//! ## Text encoder (sc-17143)
//!
//! - [`text_encoder`] runs **Qwen3-VL-32B** and returns the `context` the DiT consumes: the hidden
//!   state after 50 of its 64 layers, one row per presentation token. The conditioner applies **no
//!   chat template** and adds no special tokens, so nothing is sliced off the front (sc-18741).
//!
//! Two findings from that slice are load-bearing elsewhere in the epic. First, the checkpoint is
//! **upstream `Qwen/Qwen3-VL-32B-Instruct`, byte-identical** across all 14 shards — only
//! `tokenizer_config.json` differs — so the TE need not be rehosted, and decoder layers 50-63 plus
//! `lm_head` are never executed. Second, `<d>` and six sibling specials are declared *only* in that
//! one changed file, get their ids assigned positionally at load time, and land on **untrained**
//! embedding rows.
//!
//! ## DiT block stack (sc-17144)
//!
//! - [`dit`] carries the 50-layer **single-stream** transformer block, its AdaLN modulation, the
//!   2-layer text token refiner and the 3-axis **MM-RoPE** — including the `(t, h, w)` coordinate
//!   conventions each modality carries, of which the audio one (no height; pinned to the two
//!   extremes of the video's width grid; channel-major on the video clock) is the highest-risk
//!   detail in the port.
//!
//! Video, audio and text tokens share **one** stack and one attention document; there is no
//! cross-attention anywhere in MiniMax-H3. The attention inner width (7168) is *wider* than the
//! residual stream (5376), and only 96 of each head's 128 channels rotate.
//!
//! ## AdaLN precompute + evict (sc-17145)
//!
//! - [`dit::adaln`] projects the **whole schedule's** modulation up front and then releases the
//!   `adaln_proj` weights — 13.01 B of the DiT's ~33 B, **26.02 GB at bf16**, taking
//!   denoise-resident from ~62 GB to ~36 GB.
//!
//! The AdaLN weights are a function of the timestep embedding only, never of the tokens, which is
//! what makes this possible. Two details decide whether an implementation is correct or merely
//! plausible. MiniMax-H3's packed sequence carries **several timesteps at once** — video (which the
//! text rows share), audio, `fl2va` conditioning (0.999) and `ref2va` reference audio (1.0) all sit
//! at different `t` inside one forward pass — so
//! the table is keyed on distinct timesteps across the run, not on steps. And MLX is lazily
//! evaluated, so a precompute that is not forced still references the weights it was supposed to
//! free: [`dit::adaln::AdaLnCache::precompute_and_evict`] owns that ordering, and pairs the drop
//! with an explicit allocator drain because `get_active_memory` alone cannot tell a released buffer
//! from a cached one.
//!
//! ## Joint audio+video denoise (sc-17146)
//!
//! - [`denoise`] runs the **joint dual-modality loop**: one packed sequence, **one** transformer
//!   pass per step (the checkpoint is guidance-distilled — there is no CFG), video stepped on a
//!   12.0 sigma shift and audio on 3.0, with the `fl2va` keyframe anchors written past.
//!
//! Three things it owns that nothing else can state. [`denoise::geometry`] carries **the AV time
//! alignment**: both modalities advance one shared MM-RoPE clock at 40 rotary units per second
//! (`24 fps · 5/3` and `40 latents/s · 1`), so drift can only enter through a mis-derived
//! latent-frame count — which is checked, not assumed. [`denoise::schedule`] reproduces five ways
//! `MiniMaxH3Scheduler` departs from stock rectified flow, of which the **reversed velocity sign**
//! (`x0 = x_t + σ·v`) and the two-source σ in one Euler update are the ones a port copied from any
//! other in-tree family gets silently wrong. And [`denoise::packing`] owns the row order, the three
//! index tensors — `video_indices` **skips the audio block** — the tags and the scatter.
//!
//! ## Read [`layout`] before porting anything else in this family
//!
//! MiniMax publishes the checkpoint in the **converted diffusers layout**, and that conversion
//! applies tensor transforms which are *shape-identical* to their inputs — a swapped gated-FFN
//! projection and a fused-QKV reorder. No shape check, exhaustive-key-mapping proof or checksum can
//! see them; only an explicit assertion can. [`layout`] states the contract, backs it with
//! [`layout::split_gate_value`], and records how sc-18740 shipped a functionally wrong 36-layer
//! decoder past a fully green parity suite. **The DiT (sc-17144) carries the same FFN swap plus a
//! grouped-QKV reorder**, and the candle twins (sc-17154 / sc-17155) inherit both.
//!
//! ## `t2va` first light — the DiT's input/output projections and the pipeline (sc-17147)
//!
//! - [`dit::heads`] carries the **17** tensors of `transformer/` that are neither a block nor a
//!   refiner entry: the two patch projections, `context_embedder`, the timestep MLP, the final
//!   modulated norm and the two output heads;
//! - [`dit::model`] assembles the whole `MiniMaxH3Transformer3DModel` and implements
//!   [`denoise::JointVelocity`] over it;
//! - [`pipeline`] owns request geometry, latent packing, the cancellable render core and the
//!   delivered soundtrack's length policy;
//! - [`model`] is the `gen-core` [`Generator`](mlx_gen::gen_core::Generator), registered with
//!   `mlx-gen-catalog`.
//!
//! Three things that slice owns and nothing else can state. The published checkpoint is **mixed
//! precision** and the 17 are where it lives — twelve of them ship float32 while the 50-block stack
//! is bfloat16 — so they load at their published dtype rather than the stack's. `norm_out` is keyed
//! on the **bare timestep index**, not on the blocks' `timestep · MODALITY_NUM + tag`, and the two
//! index tensors are derived from one another so they cannot drift. And the delivered soundtrack is
//! **fitted to the picture** rather than muxed at its own length: see [`pipeline::fit_audio_to_video`]
//! for the ±8.33 ms the audio-latent rounding produces and why the correction belongs on the audio.
//!
//! ## Quantized tiers (sc-17150)
//!
//! - [`convert`] packs the DiT offline into the `q4` / `q8` artifacts
//!   `SceneWorks/minimax-h3-mlx` publishes, **one source shard at a time** — the whole-map shape
//!   every sibling converter uses would hold ~101 GB at q8 on a host this epic has already
//!   OOM-killed once;
//! - [`quant`] is the consume side: packed weights auto-detect on `{base}.scales`, so one loader
//!   serves every tier with no dense transient.
//!
//! Three things this slice owns. **Nothing quantizes at load** — [`model::load`] *reconciles*
//! `spec.quantize` against the staged tier's `config.json` marker and refuses to pack 66_280_430_080
//! dense bytes on the way in, so a tier is a hosting decision rather than an install-time peak.
//! **Two components tier**: the DiT ([`model::DIT_COMPONENT`]) and, since sc-19120, the text
//! encoder ([`model::TEXT_ENCODER_COMPONENT`], packed tiers that took the conditioning stage from
//! 53.07 GB dense to 14.43 GB at q4) — each redirected individually, so a tiered install is the
//! upstream root plus one or both redirected components at independent tiers. Both VAEs and the
//! tokenizer are dense in every tier and shared. And the dense-by-policy set is a **correctness**
//! boundary, not a size trade — [`dit::heads`] reads those tensors with a raw `Weights::require`,
//! which would load u32 codes as floats with no error at all (sc-14980).
//!
//! ## LoRA adapters and the turbo alpha (sc-18724)
//!
//! - [`adapters`] carries the key→module map for the **diffusers** key space the lightx2v turbo
//!   checkpoints publish, and the alpha resolution those checkpoints need: rank from the factor
//!   shapes, alpha from the top-level `__metadata__`, and a fallback of **8** — upstream's
//!   `DEFAULT_LORA_ALPHA` — rather than the rank.
//!
//! That fallback is the whole slice. `lightx2v/Minimax-h3-Turbo` stamps alpha as a bare
//! `__metadata__` string with no `rank` key, no per-target `.alpha` and no `lora_adapter_metadata`
//! blob, and **both** pre-existing in-tree alpha paths mishandle that silently: the PEFT one folds an
//! `alpha=8, rank=128` file 16× too strong, `parse_rank_alpha` 128×. Alpha also differs *per file
//! inside one repo* — 128 on the 4-step 768p export, 8 on the 8-step and ref2v ones, absent on
//! `4step_v0.1` — so it is read from each file rather than pinned per family. Two further facts:
//! 24 of each file's 624 tensors target the **token refiner**, which is a second reason it cannot be
//! stubbed; and the `_comfyui_` twin of each file is a different module shape (fused `qkv_proj`,
//! swapped SwiGLU halves), detected on the keys and **converted** (sc-19443) rather than
//! half-applied — folding one un-converted is shape-valid and numerically wrong.

pub mod adapters;
pub mod alias_free;
pub mod audio_config;
pub mod audio_vae;
pub mod audio_vae_encoder;
pub mod block_stream;
pub mod blocks;
pub mod chunking;
pub mod conditioning;
pub mod config;
pub mod convert;
pub mod cost;
pub mod decoder;
pub mod denoise;
pub mod dit;
pub mod keyframe;
pub mod layout;
pub mod memory_strategy;
pub mod model;
pub mod pipeline;
pub mod quant;
pub mod reference;
pub mod rope;
pub mod spatial_tiling;
pub mod tensor;
pub mod text_encoder;
pub mod vae;
pub mod vae_encoder;

pub use adapters::{
    adapter_target_paths, alpha_rank_fold, apply_minimax_h3_adapters, is_comfyui_key_space,
    normalize_minimax_h3_key, resolve_alpha, resolve_rank, MiniMaxH3LoraReport, ALPHA_METADATA_KEY,
    BLOCK_TARGETS, DEFAULT_LORA_ALPHA, PREFIXES, RANK_METADATA_KEY,
};
pub use alias_free::{kaiser_sinc_filter1d, Activation1d, LowPassFilter1d, SnakeBeta, UpSample1d};
pub use audio_config::{
    BigVganConfig, MiniMaxH3AudioVaeConfig, ACTIVATION_KERNEL_SIZE, ACTIVATION_RESAMPLE_RATIO,
    AUDIO_LATENTS_MEAN, AUDIO_LATENTS_STD, AUDIO_LATENT_CHANNELS, AUDIO_OUTPUT_CHANNELS,
    AUDIO_SAMPLE_RATE, AUDIO_TOKEN_RATE_HZ,
};
pub use audio_vae::MiniMaxH3AudioVae;
pub use audio_vae_encoder::{AudioDiagonalGaussian, MiniMaxH3AudioVaeEncoder};
pub use block_stream::{precompute_adaln_windowed, DitBlockStream};
pub use chunking::{ChunkSpan, TemporalGeometry, TemporalPlan};
pub use conditioning::{
    build_condition_rows, encode_keyframe_condition, fp16_round_trip, keyframe_condition_rows,
    scale_noise, validate_condition_arity, KeyframeNoise, KEYFRAME_ENCODE_SEED,
    KEYFRAME_NOISE_AUG_T,
};
pub use config::{
    MiniMaxH3VaeConfig, CLIP_LENGTH, DECODER_HEAD_DIM, DECODER_NUM_HEADS, DECODER_NUM_LAYERS,
    DECODER_NUM_REGISTER_TOKENS, DECODER_ROPE_DIM_RATIO, DECODER_ROPE_THETA, LATENTS_MEAN,
    LATENTS_STD, LATENT_CHANNELS, TOKEN_DROP, VAE_RATIO, VAE_RATIO_T,
};
pub use cost::{
    largest_writable_frame_count, packed_seq_len, rows_per_latent_frame, writable_duration_seconds,
    DitSequenceCost, MAX_WRITABLE_ELEMS,
};
pub use decoder::ViT3dDecoder;
pub use denoise::{
    adaln_schedule, align_num_frames, audio_latent_num_frames, denoise_av, rope_clocks_agree,
    shift_sigma, video_latent_num_frames, DenoiseModality, JointGeometry, JointSchedule, JointStep,
    JointVelocity, PackedLayout, RowClass, SigmaSchedule, AUDIO_LATENTS_PER_SECOND,
    AUDIO_SIGMA_SHIFT, AUDIO_TAG, KEYFRAME_NOISE_AUG, LEGAL_FRAME_COUNTS, MAX_AV_DRIFT_SECONDS,
    MINIMAX_H3_FPS, MIN_INFERENCE_STEPS, NUM_ROW_CLASSES, REFERENCE_AUDIO_TIMESTEP,
    ROPE_UNITS_PER_SECOND, TEXT_TAG, VIDEO_SIGMA_SHIFT, VIDEO_TAG,
};
pub use dit::{
    AdaLayerNormOut, AdaLnCache, AdaLnModulation, AdaLnProjection, AdaLnResidency, DitBlock,
    DitProjections, JointDit, LinearBias, MiniMaxH3Dit, MiniMaxH3DitConfig, MmRope, MmRopeTables,
    NormOutModulation, ScheduleKey, TimestepEmbedder, TimestepSchedule, TokenRefiner,
    TokenRefinerBlock, MODALITY_NUM, PUBLISHED_DIT_TENSORS,
};
pub use keyframe::{
    anchors_for, fit_keyframe, fit_keyframes, keyframe_to_vae_pixels, resolve_canvas_size,
    resolve_keyframe_canvas, round_half_to_even, KeyframeFit, MAX_ASPECT_RATIO, MIN_ASPECT_RATIO,
};
pub use layout::{
    split_gate_value, GatedFfnLayout, AUDIO_VAE_IS_UNCONVERTED, PUBLISHED_GATED_FFN_LAYOUT,
};
pub use model::{descriptor, load, MODEL_ID as GENERATOR_ID};
pub use pipeline::{
    align_frames_for_duration, fit_audio_to_video, frames_to_images, initial_latents,
    patchify_video_latents, render_latents, resolve_geometry, revert_pixel_normalization,
    t2va_layout, unpack_audio_rows, unpatchify_video_rows, RenderedLatents, RequestGeometry,
    CANVAS_MAX_PIXELS, CANVAS_SHORT_EDGE, MAX_CANVAS_EDGE, MAX_DURATION_SECONDS,
    MIN_DURATION_SECONDS, PATCH_SIZE, PIXEL_MEAN, PIXEL_STD, SMALLEST_LEGAL_FRAMES, SPATIAL_STRIDE,
};
pub use reference::{
    AudioReference, Ref2VaReference, Ref2VaReferences, ReferenceCounts, ReferenceKind,
    ReferencePresentation, VideoReference, MAX_AUDIO_REFERENCES, MAX_IMAGE_REFERENCES,
    MAX_TOTAL_REFERENCES, MAX_VIDEO_REFERENCES, REFERENCE_IMAGE_SHORT_EDGE,
};
pub use rope::{create_token_ids, Rope3d, RopeTables};
pub use text_encoder::{
    encode_grounded, encode_grounded_from_vision, minimax_h3_vision_config, run_vision,
    GroundedVision, MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, SpecialTokens,
    APPLIES_CHAT_TEMPLATE, LM_PREFIX, MINIMAX_ADDED_SPECIALS, NUM_HIDDEN_LAYERS, SELECT_HIDDEN,
    VISION_PREFIX,
};
pub use vae::{split_fused_qkv, MiniMaxH3VideoVae};
pub use vae_encoder::{
    reflect_pad_axis, stitch_tiles, zero_pad_front_time, CausalConv3d, DiagonalGaussian,
    DownBlock3d, FrameIsolatedGroupNorm, ResnetBlock3d, TilePlan, VideoEncoder3d,
    ENCODER_TEMPORAL_PADDING, ENCODER_TILING_IS_ON_BY_DEFAULT, LOGVAR_CLAMP,
    TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE,
};

/// The published model id this crate targets.
pub const MODEL_ID: &str = "minimax_h3";

/// Pixel alignment a generated canvas must satisfy — the VAE's `VAE_RATIO` spatial compression
/// **times the DiT's width patch of 2**, i.e. 32.
///
/// Consumers read this for `requiresDimensionsMultipleOf`, so it must be the number the engine
/// actually enforces, not the looser VAE-only one. A 16-aligned canvas survives the VAE and then has
/// an odd number of latent columns, which has no patched representation at all
/// ([`pipeline::SPATIAL_STRIDE`]). Until sc-17147 there was no pipeline to enforce anything and this
/// advertised 16.
pub const SIZE_MULTIPLE: u32 = VAE_RATIO as u32 * 2;

/// Add the MLX MiniMax-H3 generator to an explicit media registry builder.
///
/// The three memory registrations join the ladder (sc-18659). `register_memory_strategy` errors at
/// `build()` for an id with no matching generator, and the contract fixture and behavior seams error
/// for an id with no matching memory strategy, so the four are wired together or not at all.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_memory_strategy(memory_strategy::MEMORY_REGISTRATION)
        .register_memory_contract_fixture(memory_strategy::MEMORY_CONTRACT_FIXTURE)
        .register_memory_behavior(memory_strategy::MEMORY_BEHAVIOR)
}

/// Build the complete explicit MLX MiniMax-H3 provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    /// The explicit catalog surface: one generator, at the published id.
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().expect("registry");
        let ids: Vec<&str> = registry.generators().map(|g| (g.descriptor)().id).collect();
        assert_eq!(ids, vec![super::MODEL_ID]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published compression contract, and the stride the generator enforces.
    #[test]
    fn published_surface_is_the_vae_decode_contract() {
        assert_eq!(MODEL_ID, "minimax_h3");
        assert_eq!(
            SIZE_MULTIPLE, 32,
            "VAE 16x times the DiT's width patch of 2"
        );
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
