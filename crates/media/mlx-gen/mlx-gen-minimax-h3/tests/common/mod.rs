#![allow(dead_code)]
//! Shared helpers. `mod common;` is compiled into every test binary, so only a subset is used by
//! any one of them.

use std::path::PathBuf;

use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{BigVganConfig, MiniMaxH3AudioVaeConfig, MiniMaxH3VaeConfig};

// --- Cross-backend fixture geometry (sc-19496) ---------------------------------------------------
//
// Every fixture under `tests/fixtures/` here is byte-identical to the one `candle-gen-minimax-h3`
// commits under the same name. Both lanes therefore load the same bytes through their own hand-typed
// geometry, so a drift in either one leaves both lanes internally consistent and both parity suites
// green while the two backends compare tensors dumped at one shape against a model built at another.
// Nothing could see that: the two crates cannot import each other, because this one builds on macOS
// only.
//
// The numbers the two lanes must agree on are declared here as `SHARED_FIXTURE_*` constants, and the
// constructors below build from them rather than repeating literals. `check_cross_backend_geometry`
// in `scripts/check-workspace.py` compares every `SHARED_FIXTURE_*` declaration in this file against
// the candle crate's, by name set and by value — so the doc comments that used to assert agreement
// in prose now describe something enforced.

/// DiT attention heads — `MiniMaxH3DitConfig::num_attention_heads`.
pub const SHARED_FIXTURE_DIT_NUM_ATTENTION_HEADS: i32 = 4;
/// DiT per-head width. `heads · head_dim` (96) stays wider than `hidden_size` (64), as it is in the
/// shipped model.
pub const SHARED_FIXTURE_DIT_ATTENTION_HEAD_DIM: i32 = 24;
/// DiT residual width.
pub const SHARED_FIXTURE_DIT_HIDDEN_SIZE: i32 = 64;
/// DiT transformer depth.
pub const SHARED_FIXTURE_DIT_NUM_LAYERS: i32 = 2;
/// DiT refiner depth.
pub const SHARED_FIXTURE_DIT_NUM_REFINER_LAYERS: i32 = 2;
/// DiT feed-forward width.
pub const SHARED_FIXTURE_DIT_FFN_DIM: i32 = 32;
/// DiT text-conditioning width.
pub const SHARED_FIXTURE_DIT_TEXT_DIM: i32 = 40;
/// DiT sinusoidal timestep-embedding width.
pub const SHARED_FIXTURE_DIT_FREQ_DIM: i32 = 16;
/// DiT timestep-embedding hidden width.
pub const SHARED_FIXTURE_DIT_TIME_EMBED_HIDDEN_DIM: i32 = 64;
/// DiT timestep-embedding output width.
pub const SHARED_FIXTURE_DIT_TIME_EMBED_DIM: i32 = 48;
/// DiT rotary width. 2 keeps the rotary **partial** — `2 · 3 · rope_freq_dim` (12) of the 24 head
/// channels — which is the path under test.
pub const SHARED_FIXTURE_DIT_ROPE_FREQ_DIM: i32 = 2;

/// Packed text rows in the DiT fixture's `layout.*` tensors.
pub const SHARED_FIXTURE_LAYOUT_NUM_TEXT_TOKENS: i32 = 5;
/// Audio latents **per channel**.
pub const SHARED_FIXTURE_LAYOUT_NUM_AUDIO_LATENTS: i32 = 3;
/// Channels the soundtrack is packed channel-major over.
pub const SHARED_FIXTURE_LAYOUT_AUDIO_CHANNELS: i32 = 2;
/// Target video latent frames.
pub const SHARED_FIXTURE_LAYOUT_NUM_LATENT_FRAMES: i32 = 3;
/// Target latent height.
pub const SHARED_FIXTURE_LAYOUT_LATENT_HEIGHT: i32 = 4;
/// Target latent width.
pub const SHARED_FIXTURE_LAYOUT_LATENT_WIDTH: i32 = 6;

/// Video-VAE latent channels — the shipped 24, not a shrunk stand-in.
pub const SHARED_FIXTURE_VAE_LATENT_CHANNELS: i32 = 24;
/// Video-VAE decoded channels (RGB).
pub const SHARED_FIXTURE_VAE_OUT_CHANNELS: i32 = 3;
/// Video-VAE transformer depth.
pub const SHARED_FIXTURE_VAE_NUM_LAYERS: usize = 2;
/// Video-VAE attention heads.
pub const SHARED_FIXTURE_VAE_NUM_HEADS: i32 = 2;
/// Video-VAE per-head width.
pub const SHARED_FIXTURE_VAE_HEAD_DIM: i32 = 16;
/// Video-VAE register tokens.
pub const SHARED_FIXTURE_VAE_NUM_REGISTER_TOKENS: i32 = 4;
/// Video-VAE feed-forward multiplier.
pub const SHARED_FIXTURE_VAE_FFN_MULT: i32 = 4;
/// Video-VAE rotary base.
pub const SHARED_FIXTURE_VAE_ROPE_THETA: f32 = 100.0;
/// Video-VAE rotary coverage, the shipped ratio.
pub const SHARED_FIXTURE_VAE_ROPE_DIM_RATIO: f32 = 0.75;
/// Video-VAE decoder norm epsilon.
pub const SHARED_FIXTURE_VAE_NORM_EPS: f32 = 1e-5;
/// Video-VAE spatial patch — the decode fixture's spatial cumprod.
pub const SHARED_FIXTURE_VAE_PATCH_SIZE: i32 = 2;
/// Video-VAE temporal patch — the **production** value, so the chunk plan is the real one.
pub const SHARED_FIXTURE_VAE_PATCH_SIZE_T: i32 = 4;
/// Video-VAE clip length — the **production** 17, the `17n + 5` lattice's period.
pub const SHARED_FIXTURE_VAE_CLIP_LENGTH: i32 = 17;
/// Video-VAE encoded channels (RGB).
pub const SHARED_FIXTURE_VAE_IN_CHANNELS: i32 = 3;
/// Video-VAE encoder level widths for the decode fixture.
pub const SHARED_FIXTURE_VAE_BLOCK_OUT_CHANNELS: [i32; 4] = [32, 32, 32, 32];
/// Video-VAE ResNet blocks per level.
pub const SHARED_FIXTURE_VAE_LAYERS_PER_BLOCK: usize = 1;
/// Video-VAE spatial strides per level for the decode fixture.
pub const SHARED_FIXTURE_VAE_SPATIAL_DOWNSAMPLE_FACTORS: [i32; 4] = [2, 1, 1, 1];
/// Video-VAE temporal strides per level for the decode fixture.
pub const SHARED_FIXTURE_VAE_TEMPORAL_DOWNSAMPLE_FACTORS: [i32; 4] = [1, 2, 2, 1];
/// Video-VAE encoder GroupNorm groups.
pub const SHARED_FIXTURE_VAE_NORM_NUM_GROUPS: i32 = 32;
/// Video-VAE encoder norm epsilon.
pub const SHARED_FIXTURE_VAE_ENCODER_NORM_EPS: f32 = 1e-6;

/// Encode-fixture spatial patch: every level is spatial-strided there, so the cumprod is 4.
pub const SHARED_FIXTURE_ENCODE_PATCH_SIZE: i32 = 4;
/// Encode-fixture level widths — the last level changes width so `conv_shortcut` is built.
pub const SHARED_FIXTURE_ENCODE_BLOCK_OUT_CHANNELS: [i32; 4] = [32, 32, 32, 64];
/// Encode-fixture spatial strides.
pub const SHARED_FIXTURE_ENCODE_SPATIAL_DOWNSAMPLE_FACTORS: [i32; 4] = [1, 2, 2, 1];
/// Encode-fixture temporal strides.
pub const SHARED_FIXTURE_ENCODE_TEMPORAL_DOWNSAMPLE_FACTORS: [i32; 4] = [1, 2, 2, 1];

/// Audio-VAE sample rate — the shipped 32 kHz, which is what selects the production BigVGAN branch.
pub const SHARED_FIXTURE_AUDIO_SAMPLE_RATE: u32 = 32_000;
/// Audio-VAE output channels.
pub const SHARED_FIXTURE_AUDIO_OUTPUT_CHANNELS: u16 = 2;
/// Audio-VAE latent channels — the shipped 32.
pub const SHARED_FIXTURE_AUDIO_LATENT_CHANNELS: i32 = 32;
/// Audio-VAE encoder width.
pub const SHARED_FIXTURE_AUDIO_ENCODER_DIM: i32 = 8;
/// Audio-VAE encoder strides.
pub const SHARED_FIXTURE_AUDIO_ENCODER_RATES: [i32; 2] = [2, 2];
/// Audio-VAE decoder strides — the shipped seven upsample stages.
pub const SHARED_FIXTURE_AUDIO_DECODER_RATES: [i32; 7] = [5, 5, 2, 2, 2, 2, 2];
/// Audio-VAE decoder width, shrunk to the smallest that survives all seven halvings.
pub const SHARED_FIXTURE_AUDIO_DECODER_DIM: i32 = 128;
/// Audio-VAE attention projection.
pub const SHARED_FIXTURE_AUDIO_ATTN_PROJ: bool = true;
/// Audio-VAE decoder branch.
pub const SHARED_FIXTURE_AUDIO_DECODER_TYPE: &str = "bigvgan";
/// BigVGAN mel bands the fixture was dumped with.
pub const SHARED_FIXTURE_AUDIO_BIGVGAN_NUM_MELS: i32 = 64;
/// BigVGAN decoder width the fixture was dumped with.
pub const SHARED_FIXTURE_AUDIO_BIGVGAN_DECODER_DIM: i32 = 128;

/// The committed video-VAE parity fixture, produced by `tools/dump_minimax_h3_video_vae.py`
/// running the **official diffusers** `AutoencoderKLMiniMaxH3` — the converted-checkpoint layout
/// production loads. It carries the pre-conversion `src.` tensors alongside, so
/// `video_vae_parity.rs` can assert the committed weights really are in the converted layout
/// (sc-18740: a fixture dumped from the reference modules through a pure rename made the whole
/// suite a false green).
pub const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/video_vae_decode.safetensors"
);

/// The committed video-VAE **encode** fixture (sc-17148), produced by
/// `tools/dump_minimax_h3_video_vae_encode.py`.
///
/// # Why the encode half has its own file
///
/// Its geometry cannot be the decode fixture's. The encoder must not **crop**: a level that
/// strides time without also striding space convolves a 3-wide kernel with no spatial padding and
/// loses two columns instead of halving, which breaks the tiled encode outright (the stitch
/// assumes `latent = pixel / ratio`). The original MiniMax module asserts `time_stride in [1, 2]`,
/// so `patch_size_t` 4 needs TWO time-strided levels — each of which must therefore also be
/// spatial-strided, making the spatial cumprod 4 rather than [`FIXTURE`]'s 2.
///
/// Regenerating [`FIXTURE`] at that geometry was the obvious move and is the wrong one. Its bytes
/// are **shared verbatim with `candle-gen-minimax-h3`** (sc-17154), whose `cross_backend.rs`
/// digests them, and re-randomizing it perturbs every decoder weight — which measurably eroded
/// that crate's `a_gated_ffn_half_swap_is_loud_against_the_mlx_record` mutation gate from ~1.2e-1
/// to 8.4e-2 against its 1e-1 floor. A second file keeps this slice's change inside this slice.
pub const ENCODE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/video_vae_encode.safetensors"
);

/// The committed audio parity fixture, produced by `tools/dump_minimax_h3_audio_vae.py` running
/// the same snapshot's `FL2VA/audio_vae` bundle (`DacAudioVAE` / `BigVGAN` / `SnakeBeta` /
/// `kaiser_sinc_filter1d`).
pub const AUDIO_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/audio_vae_decode.safetensors"
);

/// The committed text-encoder parity fixture, produced by `tools/dump_minimax_h3_te.py` running
/// the **transformers** `Qwen3VLTextModel` — an independent reference graph — at tiny dims.
///
/// Carries the select-layer context AND both neighbouring layers' contexts, so an off-by-one in
/// either direction is a failure rather than a plausible-looking tensor. Its safetensors metadata
/// additionally pins the **presentation** — the official conditioner's own
/// `tokenizer(prompt, add_special_tokens=False)` ids for a probe prompt, alongside the ids
/// sc-17143's chat-template render produced as an explicit negative control — plus the
/// special-token id map read from the shipped `tokenizer_config.json`.
pub const TE_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/te_context.safetensors"
);

/// The committed DiT parity fixture, produced by `tools/dump_minimax_h3_dit.py` running the
/// **official diffusers** `MiniMaxH3Transformer3DModel` — the converted-checkpoint layout
/// production loads (sc-18740's Rule 3).
///
/// Carries the pre-conversion `src.` tensors alongside the published ones — the raw per-head
/// interleaved fused QKV, its reordered `[q_all; k_all; v_all]` form, and the gate-first
/// `mlp.fc1` — each round-tripped through the OFFICIAL conversion functions in the generator, so
/// `dit_parity.rs` can assert which transform produced the published bytes rather than trusting a
/// comment. Its `layout.*` tensors are the reference's own `build_packed_sequence` output, which
/// is what pins the audio-token position convention.
pub const DIT_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/dit_block.safetensors"
);

/// The tiny DiT geometry the fixture was dumped at. Mirrors `dump_minimax_h3_dit.py`.
///
/// Built from the `SHARED_FIXTURE_DIT_*` constants, which `check_cross_backend_geometry` holds equal
/// to the candle lane's — the two lanes load byte-identical fixture bytes and must agree about their
/// shape before either compares a tensor.
///
/// Two structural properties of the shipped model are preserved rather than shrunk away:
/// `heads · head_dim` (96) stays **wider** than `hidden_size` (64), as 7168 > 5376; and
/// `2 · 3 · rope_freq_dim` (12) stays **narrower** than `attention_head_dim` (24), as 96 < 128, so
/// the partial-rotary path is the one under test.
pub fn dit_fixture_config() -> mlx_gen_minimax_h3::MiniMaxH3DitConfig {
    mlx_gen_minimax_h3::MiniMaxH3DitConfig {
        num_attention_heads: SHARED_FIXTURE_DIT_NUM_ATTENTION_HEADS,
        attention_head_dim: SHARED_FIXTURE_DIT_ATTENTION_HEAD_DIM,
        hidden_size: SHARED_FIXTURE_DIT_HIDDEN_SIZE,
        num_layers: SHARED_FIXTURE_DIT_NUM_LAYERS,
        num_refiner_layers: SHARED_FIXTURE_DIT_NUM_REFINER_LAYERS,
        ffn_dim: SHARED_FIXTURE_DIT_FFN_DIM,
        text_dim: SHARED_FIXTURE_DIT_TEXT_DIM,
        freq_dim: SHARED_FIXTURE_DIT_FREQ_DIM,
        time_embed_hidden_dim: SHARED_FIXTURE_DIT_TIME_EMBED_HIDDEN_DIM,
        time_embed_dim: SHARED_FIXTURE_DIT_TIME_EMBED_DIM,
        rope_freq_dim: SHARED_FIXTURE_DIT_ROPE_FREQ_DIM,
        ..mlx_gen_minimax_h3::MiniMaxH3DitConfig::default()
    }
}

/// The packed layout `dump_minimax_h3_dit.py` built the goldens over.
pub struct DitLayout {
    /// Text rows.
    pub num_text_tokens: i32,
    /// Audio latents **per channel**.
    pub num_audio_latents: i32,
    /// Channels the soundtrack is packed channel-major over.
    pub audio_channels: i32,
    /// Target video latent frames.
    pub num_latent_frames: i32,
    /// Target latent height.
    pub latent_height: i32,
    /// Target latent width.
    pub latent_width: i32,
}

/// 5 text rows, 3 audio latents over 2 channels, and 3 latent frames of a 4×6 latent at patch
/// `[1, 2, 2]` — 29 packed rows. Kept in one place so a fixture regenerated at different dims
/// fails loudly rather than drifting.
pub const DIT_LAYOUT: DitLayout = DitLayout {
    num_text_tokens: SHARED_FIXTURE_LAYOUT_NUM_TEXT_TOKENS,
    num_audio_latents: SHARED_FIXTURE_LAYOUT_NUM_AUDIO_LATENTS,
    audio_channels: SHARED_FIXTURE_LAYOUT_AUDIO_CHANNELS,
    num_latent_frames: SHARED_FIXTURE_LAYOUT_NUM_LATENT_FRAMES,
    latent_height: SHARED_FIXTURE_LAYOUT_LATENT_HEIGHT,
    latent_width: SHARED_FIXTURE_LAYOUT_LATENT_WIDTH,
};

/// The tiny text-encoder geometry the fixture was dumped at. Mirrors `dump_minimax_h3_te.py`:
/// `head_dim` deliberately != `hidden_size / num_heads` (32 vs 16), exactly as the real model has
/// 128 != 5120/64, and `select_hidden` < `num_layers` so the unused-tail trim is exercised.
pub fn te_fixture_config() -> mlx_gen_minimax_h3::MiniMaxH3TeConfig {
    mlx_gen_minimax_h3::MiniMaxH3TeConfig {
        hidden_size: 64,
        num_layers: 6,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 32,
        intermediate_size: 128,
        rms_norm_eps: 1e-6,
        rope_theta: 5_000_000.0,
        vocab_size: 256,
        select_hidden: 4,
        ..mlx_gen_minimax_h3::MiniMaxH3TeConfig::qwen3_vl_32b()
    }
}

/// The tiny audio geometry the fixture was dumped at.
///
/// Only the *width* is shrunk — `decoder_dim` 1024 → 128 (the smallest that survives all seven
/// halvings) and `latent_dim` 2048 → 64. Everything that drives the arithmetic is the shipped
/// value: `sample_rate` 32000, which is what selects the production BigVGAN branch, so all seven
/// upsample stages, the `[3,7,11]` × `[1,3,5]` AMP table, the 800× hop and the 12-tap Kaiser-sinc
/// filters are the real ones — as are the 32 latent channels and their de-normalization
/// statistics.
///
/// Built by hand rather than through `from_source_files`, because the dump passes `latent_dim`
/// explicitly instead of letting it derive from `encoder_dim · 2^len(encoder_rates)`; the
/// derivation itself is covered by the config unit tests and by the real-weight smoke.
pub fn audio_fixture_config() -> MiniMaxH3AudioVaeConfig {
    let shipped = MiniMaxH3AudioVaeConfig::default();
    MiniMaxH3AudioVaeConfig {
        sample_rate: SHARED_FIXTURE_AUDIO_SAMPLE_RATE,
        output_channels: SHARED_FIXTURE_AUDIO_OUTPUT_CHANNELS,
        latent_channels: SHARED_FIXTURE_AUDIO_LATENT_CHANNELS,
        encoder_dim: SHARED_FIXTURE_AUDIO_ENCODER_DIM,
        encoder_rates: SHARED_FIXTURE_AUDIO_ENCODER_RATES.to_vec(),
        decoder_rates: SHARED_FIXTURE_AUDIO_DECODER_RATES.to_vec(),
        decoder_dim: SHARED_FIXTURE_AUDIO_DECODER_DIM,
        attn_proj: SHARED_FIXTURE_AUDIO_ATTN_PROJ,
        decoder_type: SHARED_FIXTURE_AUDIO_DECODER_TYPE.into(),
        // The REAL 32-entry de-normalization statistics, not placeholders.
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        bigvgan: BigVganConfig::for_sample_rate(
            SHARED_FIXTURE_AUDIO_SAMPLE_RATE,
            SHARED_FIXTURE_AUDIO_BIGVGAN_NUM_MELS,
            SHARED_FIXTURE_AUDIO_BIGVGAN_DECODER_DIM,
        )
        .expect("32 kHz is supported"),
    }
}

/// NCL `[B, C, T]` → MLX-native NLC `[B, T, C]`. The audio fixture stores the reference's NCL
/// tensors; the port's public API is NCL too, so this is only for the internal NLC modules.
pub fn to_nlc(x: &Array) -> Array {
    x.transpose_axes(&[0, 2, 1]).unwrap()
}

/// The tiny geometry the fixture was dumped at. Structurally identical to the shipped model —
/// same partial-rotary ratio, same 24 latent channels, and the **production** temporal knobs
/// (`clip_length 17`, `patch_size_t 4`) so the chunk plan is the real one; only the width and
/// depth are shrunk so the weights are committable.
pub fn fixture_config(token_drop: i32) -> MiniMaxH3VaeConfig {
    let shipped = MiniMaxH3VaeConfig::default();
    MiniMaxH3VaeConfig {
        latent_channels: SHARED_FIXTURE_VAE_LATENT_CHANNELS,
        out_channels: SHARED_FIXTURE_VAE_OUT_CHANNELS,
        num_layers: SHARED_FIXTURE_VAE_NUM_LAYERS,
        num_heads: SHARED_FIXTURE_VAE_NUM_HEADS,
        head_dim: SHARED_FIXTURE_VAE_HEAD_DIM,
        num_register_tokens: SHARED_FIXTURE_VAE_NUM_REGISTER_TOKENS,
        ffn_mult: SHARED_FIXTURE_VAE_FFN_MULT,
        rope_theta: SHARED_FIXTURE_VAE_ROPE_THETA,
        rope_dim_ratio: SHARED_FIXTURE_VAE_ROPE_DIM_RATIO,
        norm_eps: SHARED_FIXTURE_VAE_NORM_EPS,
        patch_size: SHARED_FIXTURE_VAE_PATCH_SIZE,
        patch_size_t: SHARED_FIXTURE_VAE_PATCH_SIZE_T,
        clip_length: SHARED_FIXTURE_VAE_CLIP_LENGTH,
        token_drop,
        // The REAL 24-entry de-normalization statistics, not placeholders.
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        // The encoder fields the struct gained in sc-17148. This fixture carries NO `encoder.*`
        // tensors — it is the decode golden and its bytes are shared with
        // `candle-gen-minimax-h3` — so these only have to satisfy `validate_encoder`'s
        // "the cumprods are the patch sizes" rule. `MiniMaxH3VideoVae::from_weights` loads the
        // encode half only when it is present; see [`ENCODE_FIXTURE`] for the one that has it.
        in_channels: SHARED_FIXTURE_VAE_IN_CHANNELS,
        block_out_channels: SHARED_FIXTURE_VAE_BLOCK_OUT_CHANNELS.to_vec(),
        layers_per_block: SHARED_FIXTURE_VAE_LAYERS_PER_BLOCK,
        spatial_downsample_factors: SHARED_FIXTURE_VAE_SPATIAL_DOWNSAMPLE_FACTORS.to_vec(),
        temporal_downsample_factors: SHARED_FIXTURE_VAE_TEMPORAL_DOWNSAMPLE_FACTORS.to_vec(),
        norm_num_groups: SHARED_FIXTURE_VAE_NORM_NUM_GROUPS,
        encoder_norm_eps: SHARED_FIXTURE_VAE_ENCODER_NORM_EPS,
    }
}

/// The geometry [`ENCODE_FIXTURE`] was dumped at, mirroring
/// `tools/dump_minimax_h3_video_vae_encode.py`.
///
/// Differs from [`fixture_config`] in exactly the ways the encode half forces (see
/// [`ENCODE_FIXTURE`]): every downsampling level is spatial-strided, so nothing crops and the
/// spatial cumprod — and therefore `patch_size` — is 4. The width also changes at the last level,
/// so `conv_shortcut` is actually built rather than left unexercised.
pub fn encode_fixture_config(token_drop: i32) -> MiniMaxH3VaeConfig {
    MiniMaxH3VaeConfig {
        patch_size: SHARED_FIXTURE_ENCODE_PATCH_SIZE,
        block_out_channels: SHARED_FIXTURE_ENCODE_BLOCK_OUT_CHANNELS.to_vec(),
        spatial_downsample_factors: SHARED_FIXTURE_ENCODE_SPATIAL_DOWNSAMPLE_FACTORS.to_vec(),
        temporal_downsample_factors: SHARED_FIXTURE_ENCODE_TEMPORAL_DOWNSAMPLE_FACTORS.to_vec(),
        ..fixture_config(token_drop)
    }
}

/// The tile geometry [`ENCODE_FIXTURE`]'s tiled golden was dumped at — deliberately smaller than
/// the shipped 256/64 so a committable canvas spans more than one tile.
pub fn encode_fixture_tiles(f: &Weights) -> (i32, i32) {
    let t = f
        .require("const.encode_tile")
        .expect("the encode fixture carries const.encode_tile")
        .as_slice::<i32>()
        .to_vec();
    assert_eq!(t.len(), 2, "const.encode_tile is [tile_size, min_overlap]");
    (t[0], t[1])
}

/// `(max|a-b| / peak|b|, mean|a-b| / mean|b|)` over the full tensors — the peak- and
/// mean-relative error the parity gates use.
pub fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let a = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let b = b
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let diff = mlx_rs::ops::subtract(&a, &b).unwrap().abs().unwrap();
    let bb = b.abs().unwrap();
    let peak: f32 = bb.max(None).unwrap().item();
    let mean: f32 = bb.mean(None).unwrap().item();
    let max_d: f32 = diff.max(None).unwrap().item();
    let mean_d: f32 = diff.mean(None).unwrap().item();
    (max_d / peak.max(1e-12), mean_d / mean.max(1e-12))
}

/// Cosine similarity between two tensors, flattened.
///
/// Reported alongside [`rel`] wherever a **layout** error is under test rather than a numeric one.
/// sc-18740's gate/value half-swap leaves the output norm essentially unchanged (89 vs 85 on real
/// weights) so magnitude, std and checksum assertions are all blind to it — but it also keeps
/// cosine at 0.73-0.78 rather than ~0, because `silu(a)·b` and `silu(b)·a` share sign structure.
/// **Neither metric alone is sufficient**: cosine is scale-invariant and the relative max-abs-diff
/// is the one that actually exposes the swap, so the tests print both and gate on both.
pub fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let b = b
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .flatten(None, None)
        .unwrap();
    let dot: f32 = mlx_rs::ops::multiply(&a, &b)
        .unwrap()
        .sum(None)
        .unwrap()
        .item();
    let na: f32 = a.square().unwrap().sum(None).unwrap().item::<f32>().sqrt();
    let nb: f32 = b.square().unwrap().sum(None).unwrap().item::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// L2 norm of a tensor — reported (never asserted on) in the half-swap tests, as the direct
/// demonstration that magnitude cannot see a layout error.
pub fn l2_norm(x: &Array) -> f32 {
    x.as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .square()
        .unwrap()
        .sum(None)
        .unwrap()
        .item::<f32>()
        .sqrt()
}

/// Assert `got` matches `want` within the parity tolerance, reporting the actual error.
pub fn assert_parity(got: &Array, want: &Array, tol: f32, what: &str) {
    assert_eq!(got.shape(), want.shape(), "{what}: shape");
    let (peak, mean) = rel(got, want);
    assert!(
        peak < tol,
        "{what}: peak-relative error {peak:.3e} (mean {mean:.3e}) exceeds {tol:.1e}"
    );
}

/// **The materialization instrument** (sc-19452, shared by sc-19505). Whether MLX has already
/// *materialized* this array — whether the compute that produces it has actually happened, rather
/// than sitting in an unscheduled graph waiting for somebody to force it.
///
/// `_mlx_array_is_available` is mlx-c's accessor for `mlx::core::array::is_available()`: the status
/// is `available`, or it is `evaluated` with the completion event signalled or absent. A
/// **synchronous** `mlx_rs::transforms::eval` — the only kind this crate uses — attaches no event
/// and marks its outputs `evaluated` before returning, so the reading is exact and clock-free. That
/// is where the load-insensitivity comes from, and it is conditional: if a production `eval` here
/// were ever switched to `async_eval`, this would become a race against the event signal and would
/// have to `wait()` first. Anyone making that change has to revisit every caller of this function.
///
/// Two precision notes, because the obvious paraphrases are both wrong:
///
/// * it is **not** "`false` for anything an op returned". MLX short-circuits identity ops — a
///   `reshape` to the same shape, a full-range `slice`, an `astype` to the same dtype — by
///   returning the *input*, which keeps the input's status. A view defers no compute, so the
///   reading stays semantically right, but the mechanism is not "every op yields unscheduled";
/// * it is **not** a pure read. `is_available()` detaches the event and promotes the status on the
///   shared descriptor, non-atomically. Harmless under this repo's forced `RUST_TEST_THREADS = 1`,
///   but do not treat it as a race-free probe of an `Array` shared across threads.
///
/// mlx-rs exposes no safe wrapper, so the call goes through `mlx-sys` — the same binding crate
/// `mlx_rs::Array` is built on, so [`Array::as_ptr`] hands back exactly the `mlx_array` this
/// signature wants. Two obligations, not one: the handle must outlive the call, which the `&Array`
/// borrow guarantees; and some mlx-rs op must already have run on this thread, because mlx-rs
/// installs its error handler lazily inside its own wrappers and mlx-c's default handler is
/// `exit(-1)` rather than a status return. Every caller must therefore read *after* it has built
/// or loaded the arrays under test, which every current call site does.
///
/// # Why this lives here rather than in one test file
///
/// Two test binaries read it — `pipeline.rs` (sc-19452) and `joint_denoise.rs` (sc-19505) — and the
/// preconditions above are subtle enough that a second copy of them would drift. `mod common;` is
/// compiled into each binary, so there is one function and one set of caveats.
pub fn is_materialized(array: &Array) -> bool {
    let mut available = false;
    let status = unsafe { mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr()) };
    assert_eq!(
        status, 0,
        "_mlx_array_is_available failed on a live array handle"
    );
    available
}

/// Standard deviation of a tensor — a golden whose expected output is ~constant proves nothing.
pub fn std_dev(x: &Array) -> f32 {
    let x = x.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let mean: f32 = x.mean(None).unwrap().item();
    let centered = mlx_rs::ops::subtract(&x, Array::from_f32(mean)).unwrap();
    let var: f32 = centered.square().unwrap().mean(None).unwrap().item();
    var.sqrt()
}

// --- sc-19449: the AdaLN precompute / denoise phase pair ---------------------------------------

/// Multiple of the **retained modulation table** that the precompute's own transient has to stay
/// under — the denominator of assertion (g) in [`assert_adaln_phase_envelope`].
///
/// Integer arithmetic on purpose: these are byte counts in the tens of gigabytes, where an `f64`
/// round-trip is avoidable noise in a message a reviewer has to trust.
///
/// Measured `precompute_transient / table` is **2.222x** on the synthetic f32 8-layer stack and
/// **1.331x** on the real bf16 50-layer `transformer/`. `peak_precompute` is bit-stable run to run
/// at both (`166_627_328 B` over four synthetic runs whose `peak_denoise` moved 37%), so `3` is
/// 1.35x of headroom at the tighter of the two. This is the **only** empirical number left in the
/// envelope — everything else follows from the identity documented on
/// [`assert_adaln_phase_envelope`].
pub const ADALN_TRANSIENT_TABLE_MULTIPLE: usize = 3;

/// The fixed fraction of `released` that assertion (f) was stated against before sc-19449's
/// review, as `num/den`. **Retired** — (f) is now stated against `released −
/// denoise_working_set`, which is geometry-free where a fixed fraction is not.
///
/// Kept, and used only by `adaln_evict_memory.rs`'s arm 1b, which holds a denoise working set of
/// half the eviction live and shows that the retired form goes red under it while the shipped form
/// still closes. That is the design decision as an executing measurement rather than as a comment.
pub const ADALN_RETIRED_GAP_NUM: usize = 3;
/// Denominator of [`ADALN_RETIRED_GAP_NUM`].
pub const ADALN_RETIRED_GAP_DEN: usize = 4;

/// One AdaLN precompute-and-evict measurement: both phase peaks, and the residency either side of
/// the evict. Every field is MLX **active** bytes.
///
/// # Why this is a shared type rather than assertions written twice
///
/// Until sc-19449, `tests/adaln_evict_memory.rs` and `tests/adaln_evict_real_weights.rs` each
/// computed `peak_precompute`, printed it, and asserted **nothing** against it. Their (a)-(d) pin
/// the post-evict drop against the *pre-evict resident* — a residency, not a phase peak — so the
/// two tests positioned to measure the precompute/denoise relationship measured it and let it go.
/// That is how `convert.rs` came to carry the claim that AdaLN precompute is free because "the
/// precompute peak stays under the denoise peak … it holds, comfortably", which sc-18659
/// retracted: the one in-repo measurement of the precompute side
/// (`memory_strategy::ADALN_EVICTED_BYTES`, 64.56 → 38.70 GB active) says the precompute side sits
/// *above* the residency that follows, i.e. the inequality ran the other way.
///
/// [`assert_adaln_phase_envelope`] now pins the pair, and both files call it, so the synthetic and
/// the real-weight measurement cannot drift into pinning different things.
pub struct AdaLnPhases<'a> {
    /// Tier, denoise geometry and cache schedule this pair was taken at. The bounds below no
    /// longer depend on any of the three (see [`assert_adaln_phase_envelope`]); the label is on
    /// the record line so a reader can reproduce the measurement and so the record says what was
    /// measured rather than leaving it to be guessed.
    pub scale: &'a str,
    /// Active bytes with every `adaln_proj` still resident, before `precompute_and_evict`.
    pub active_before: usize,
    /// Active bytes after the evict and the allocator drain.
    pub active_after: usize,
    /// High-water of the precompute+evict phase — peak reset immediately before it, read
    /// immediately after.
    pub peak_precompute: usize,
    /// High-water of the phase that runs *after* the evict — peak reset again, then a real block
    /// forward against the cached table.
    pub peak_denoise: usize,
    /// Projection bytes `precompute_and_evict` reported releasing.
    pub released: usize,
    /// Bytes of the modulation table the precompute retains in their place.
    pub table: usize,
}

impl AdaLnPhases<'_> {
    /// `peak_precompute − peak_denoise`, saturating. The quantity sc-19449 is about.
    pub fn gap(&self) -> usize {
        self.peak_precompute.saturating_sub(self.peak_denoise)
    }

    /// What the precompute phase adds on top of the residency it starts from.
    pub fn precompute_transient(&self) -> usize {
        self.peak_precompute.saturating_sub(self.active_before)
    }

    /// What the denoise phase adds on top of the residency it starts from — activations, the rope
    /// tables, the gathered modulation. This is the term that moves with **geometry**, and it is
    /// also the noisiest reading here: it moved 4.1 → 8.2 MB over four synthetic runs.
    pub fn denoise_working_set(&self) -> usize {
        self.peak_denoise.saturating_sub(self.active_after)
    }

    /// Active bytes the evict actually removed. Below `released` by the table the precompute
    /// retained in the projections' place — the term assertion (f) reduces to.
    pub fn freed(&self) -> usize {
        self.active_before.saturating_sub(self.active_after)
    }

    fn x(&self, bytes: usize) -> f64 {
        bytes as f64 / (self.released.max(1)) as f64
    }

    /// The single greppable record line. The label says what was measured; the two ratios the
    /// bounds are stated against — `gap`/`released` and `transient`/`table` — are on it so a
    /// future run at a new tier or schedule can be compared to this one without re-deriving.
    pub fn record(&self) -> String {
        format!(
            "ADALN PHASE PAIR [{}]: peak_precompute {} B, peak_denoise {} B, gap {} B ({:.3}x \
             released); precompute transient {} B ({:.3}x released, {:.3}x table), denoise \
             working set {} B ({:.3}x); active {} -> {} B, released {} B, table {} B",
            self.scale,
            self.peak_precompute,
            self.peak_denoise,
            self.gap(),
            self.x(self.gap()),
            self.precompute_transient(),
            self.x(self.precompute_transient()),
            self.precompute_transient() as f64 / self.table.max(1) as f64,
            self.denoise_working_set(),
            self.x(self.denoise_working_set()),
            self.active_before,
            self.active_after,
            self.released,
            self.table,
        )
    }
}

/// Pin the AdaLN precompute/denoise phase relationship (sc-19449).
///
/// # The identity both bounds are stated against
///
/// `mlx_reset_peak_memory` sets the counter to **zero** — `MetalAllocator::reset_peak_memory` in
/// `mlx/backend/metal/allocator.h` is `peak_memory_ = 0`, not `= active_memory_` — and the
/// allocator raises it back to the then-current active on the very next `malloc`
/// (`peak_memory_ = std::max(peak_memory_, active_memory_)` in `allocator.cpp`). Both read at the
/// MLX revision this workspace pins. The precompute's first allocation therefore happens while
/// every `adaln_proj` is still resident, so the precompute phase's high-water cannot be below the
/// pre-evict residency — assertion (e). The denoise phase's window opens *after* the evict, so its
/// high-water is the post-evict residency plus its own working set. Substituting the definitions
/// of all four derived terms:
///
/// ```text
/// gap = peak_precompute − peak_denoise
///     = (active_before + precompute_transient) − (active_after + denoise_working_set)
///     = freed + precompute_transient − denoise_working_set
/// ```
///
/// That is an **identity**: it holds by the definitions of the terms, at every tier, every
/// geometry and every schedule. sc-19449's review found the first cut of (f)/(g) stated as fixed
/// fractions of `released` instead, which made both of them tier-, geometry- and schedule-bound —
/// (f) had only 2.8% of margin at bf16 render geometry, and (g)'s `1/2` was 1.3x from red at q4.
/// Both are now stated against the identity, which removes all three dependences by construction.
/// The constants did not need tightening; the **denominators** were wrong.
///
/// # (f) `gap + denoise_working_set >= released`
///
/// Substituting the identity, (f) is exactly
///
/// ```text
/// precompute_transient  >=  released − freed
/// ```
///
/// — the precompute's own high-water covers the bytes it did **not** hand back, i.e. the table it
/// retained in the projections' place. It holds by the ordering `precompute_and_evict` exists to
/// guarantee: the table is materialized while every projection is still resident, and only then is
/// anything dropped. Not vacuous — it is red for the lazy-eval trap this file's arm 2 demonstrates
/// (nothing freed, no transient, `released` still reported), red for a peak read from the wrong
/// side of the evict, and red for an over-reported `released`.
///
/// * **Geometry-free.** `denoise_working_set` enters `gap` through `peak_denoise` and enters the
///   left-hand side directly, with opposite signs, so it cancels exactly. So does the run-to-run
///   noise in `peak_denoise`, which is ±37% at synthetic scale and would otherwise have to be
///   absorbed by slack. The measured margin is 4_030_464 B synthetic and 52_025_344 B real, and
///   both are constant in the working set.
/// * **Tier-free.** `released` and `freed` are both the projections' bytes at the tier they were
///   loaded at and move together; the difference between them is the retained table, whose dtype
///   is the *compute* dtype (`quant::compute_dtype` — bf16 off a packed tier's scales) rather than
///   the tier's bit width.
/// * **Schedule-free.** The schedule sets `modulation_rows`, which scales `table` and
///   `precompute_transient` together; the margin is 0.32x-1.14x of `table` at the two measured
///   points, i.e. it tracks the table rather than the eviction.
///
/// Written additively rather than as `gap >= released − denoise_working_set` so that neither side
/// saturates: past the crossing at `denoise_working_set = released` the subtractive form would go
/// quietly vacuous, where this one still requires the eviction to be accounted for.
///
/// # (g) `precompute_transient <= ADALN_TRANSIENT_TABLE_MULTIPLE · table`
///
/// The corrected form of the retracted claim: the precompute is **not** free — it pays for the
/// full pre-evict residency, which (e) pins — but its own transient on top of that residency is a
/// small multiple of the table it keeps, so the lever is net-positive at its own worst instant.
///
/// Against `table`, not against `released`, because the two scale in opposite directions:
///
/// * `table` grows linearly with `modulation_rows` — 51 rows for the 8-evaluation schedule these
///   tests use, roughly 600 for a 100-step render — and so does `precompute_transient`;
/// * `released` **falls** with the tier. q4 packs `adaln_proj` at 4 bits plus one f16 scale and
///   bias per group of 64, so it drops from 26.02 GB to ≈7.3 GB (≈6.5 GB if the group metadata is
///   not counted).
///
/// Combining them, a q4 render on a 600-row schedule puts `precompute_transient / released` at
/// ≈0.35x (≈0.40x on the 4x model) against the retired `1/2` ceiling — 1.3x-1.4x of headroom on a
/// bound that reads as 60x loose at the geometry it was measured at (0.008x real). Stated against
/// `table`, the same quantity is 1.331x real and 2.222x synthetic at *any* tier and schedule.
///
/// # What is still empirical
///
/// Only (g)'s multiple. The identity is exact, (e) follows from the allocator's peak semantics and
/// (f) from the precompute's ordering; `precompute_transient / table` is measured — 2.222x
/// synthetic, 1.331x real — and is the projection-output intermediates on top of the table that is
/// kept. Every record line reports it. A changed precompute ordering, or a first measurement at a
/// tier or schedule not covered here, does not invalidate (e) or (f), but it does need that ratio
/// read off the record line before [`ADALN_TRANSIENT_TABLE_MULTIPLE`] can be trusted at `3`.
pub fn assert_adaln_phase_envelope(p: &AdaLnPhases<'_>) {
    println!("  {}", p.record());

    // (e) STAGE WINDOW. The precompute phase's high-water is at least the residency it began
    //     with. This is what makes `peak_precompute` a *phase* reading at all: if the peak were
    //     reset after the evict, or read from the wrong side of it, this is the assertion that
    //     notices. Attributing a peak to the wrong stage is the error that produced two wrong
    //     handoffs on this epic, and nothing here could see it before.
    assert!(
        p.peak_precompute >= p.active_before,
        "the precompute phase peaked at {} B, BELOW the {} B that was resident when the phase \
         opened — the peak was not read over the precompute window, so nothing here is a \
         per-stage measurement",
        p.peak_precompute,
        p.active_before
    );

    // (f) THE IDENTITY CLOSES. `gap + denoise_working_set >= released`: the drop from the
    //     precompute peak to the denoise peak, plus whatever the denoise phase added back,
    //     accounts for the whole eviction. Reduces to `precompute_transient >= released − freed`,
    //     i.e. the precompute's high-water covers the table it retained instead of releasing —
    //     true by the ordering `precompute_and_evict` guarantees, at any tier, geometry or
    //     schedule. It also carries the direction the retracted claim had backwards: below the
    //     crossing, `gap` cannot be the eviction unless `peak_precompute > peak_denoise`.
    //
    //     NOT a fixed fraction of `released`. That was the first cut, and it made the bound
    //     geometry- and tier-bound: 2.8% of margin at bf16 render geometry, and it would false-red
    //     outright at q4. `adaln_evict_memory.rs`'s arm 1b measures both halves of that.
    let covered = p.gap().saturating_add(p.denoise_working_set());
    assert!(
        covered >= p.released,
        "the precompute peak {} B is only {} B above the denoise peak {} B; with the denoise \
         phase's own working set of {} B that accounts for {} B of a {} B eviction. \
         `gap = freed + precompute_transient − denoise_working_set` is an identity, so this can \
         only fail if the precompute's transient ({} B) no longer covers what it kept rather than \
         released ({} B) — the table was not materialized before the evict, the peak was read \
         from the wrong side of it, or `released` is over-reported",
        p.peak_precompute,
        p.gap(),
        p.peak_denoise,
        p.denoise_working_set(),
        covered,
        p.released,
        p.precompute_transient(),
        p.released.saturating_sub(p.freed())
    );

    // (g) TRANSIENT CEILING. The corrected form of the retracted claim: the precompute is NOT
    //     free — it pays for the full pre-evict residency, which (e) pins — but its own transient
    //     is a small multiple of the table it keeps, so the lever is net-positive at its own worst
    //     instant and the precompute stage is not a new process ceiling.
    //
    //     Against `table`, not against `released`. The retired `1/2 · released` read as 60x loose
    //     (0.008x measured) and was not: `table` grows with the schedule's modulation rows while
    //     `released` falls with the tier, so a q4 render on a 600-row schedule sits at ≈0.35x of
    //     it. A tighter constant would have been wrong; a different denominator is right. Stated
    //     against `table` the ratio is 1.331x real / 2.222x synthetic at any tier and schedule.
    let ceiling = ADALN_TRANSIENT_TABLE_MULTIPLE * p.table;
    assert!(
        p.precompute_transient() <= ceiling,
        "the precompute added {} B on top of the {} B it started from — {:.3}x the {} B \
         modulation table it retains, wanted <= {}x ({} B). Its transient is no longer tracking \
         the table it keeps, so the projection-output intermediates have grown and the evict is \
         less clearly net-positive at the precompute instant. That is {:.3}x the {} B released",
        p.precompute_transient(),
        p.active_before,
        p.precompute_transient() as f64 / p.table.max(1) as f64,
        p.table,
        ADALN_TRANSIENT_TABLE_MULTIPLE,
        ceiling,
        p.x(p.precompute_transient()),
        p.released
    );
}

/// The MiniMax-H3 snapshot root, from `MINIMAX_H3_SNAPSHOT`.
///
/// Deliberately **panics** when unset. These are `#[ignore]`d tests: if this returned `None` and
/// the test quietly returned, `cargo test -- --ignored` would print `ok` in 0.00s and a reviewer
/// would read a skipped test as a passing one. Inference never derives a cache location or
/// self-fetches (epic 13657), so the path must be supplied explicitly.
pub fn snapshot() -> PathBuf {
    let raw = std::env::var("MINIMAX_H3_SNAPSHOT").unwrap_or_default();
    assert!(
        !raw.is_empty(),
        "MINIMAX_H3_SNAPSHOT must point at a MiniMaxAI/MiniMax-H3 snapshot root (the dir holding \
         `vae/`). This test is #[ignore]d and asserts rather than skips so that a missing \
         snapshot cannot be mistaken for a pass."
    );
    let path = PathBuf::from(raw);
    assert!(
        path.join("vae").join("config.json").is_file(),
        "MINIMAX_H3_SNAPSHOT={} has no vae/config.json",
        path.display()
    );
    path
}

// --- sc-18661: rung-3 bounded-attention measurement receipts -------------------------------------

/// One measured `(tier x duration)` cell of the rung-3 in-forward comparison.
///
/// Never a single number, and never one row: `frames` is an **exact-match** evidence key
/// (`mlx_fit_gate.rs:1381`) and fitted-curve extrapolation scales by area only, so no measurement at
/// one duration may stand in for another. The tier is carried on every cell rather than assumed from
/// the run, because the harness is pointed at a staged DiT directory and the same binary measures
/// whichever tier that directory holds.
#[derive(Debug, Clone)]
pub struct BoundedAttentionCell {
    pub tier: String,
    pub width: u32,
    pub height: u32,
    pub frames: i32,
    pub seq: i32,
    pub heads: i32,
    pub unbounded_peak_bytes: u64,
    pub head_chunked_peak_bytes: u64,
    pub query_chunked_peak_bytes: u64,
    /// The head axis at the **geometry-derived** budget: one head per chunk, query axis whole.
    pub bit_exact_head_peak_bytes: u64,
    /// That budget, in score elements — `B · Sq²`, i.e. exactly one head's score domain.
    pub bit_exact_head_budget: u64,
    pub head_chunks: usize,
    /// Query-row blocks the 64 Mi **head** arm fell back to. `1` means the head split alone fit.
    pub head_query_row_blocks: usize,
    pub query_row_blocks: usize,
    pub unbounded_fastest_eval_s: f64,
    pub head_chunked_fastest_eval_s: f64,
    pub query_chunked_fastest_eval_s: f64,
    pub bit_exact_head_fastest_eval_s: f64,
    pub head_relative_max_abs: f32,
    pub query_relative_max_abs: f32,
    pub bit_exact_head_relative_max_abs: f32,
}

impl BoundedAttentionCell {
    fn to_json(&self) -> serde_json::Value {
        let delta = |bounded: u64| bounded as f64 / self.unbounded_peak_bytes.max(1) as f64 - 1.0;
        serde_json::json!({
            "tier": self.tier,
            "width": self.width,
            "height": self.height,
            "frames": self.frames,
            "packed_seq_len": self.seq,
            "attention_heads": self.heads,
            "attention_chunk_size": mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET,
            "unbounded_peak_bytes": self.unbounded_peak_bytes,
            "head_chunked_peak_bytes": self.head_chunked_peak_bytes,
            "query_chunked_peak_bytes": self.query_chunked_peak_bytes,
            "bit_exact_head_peak_bytes": self.bit_exact_head_peak_bytes,
            "bit_exact_head_budget": self.bit_exact_head_budget,
            "head_chunked_peak_delta": delta(self.head_chunked_peak_bytes),
            "query_chunked_peak_delta": delta(self.query_chunked_peak_bytes),
            "bit_exact_head_peak_delta": delta(self.bit_exact_head_peak_bytes),
            "head_chunks": self.head_chunks,
            "head_query_row_blocks": self.head_query_row_blocks,
            "query_row_blocks": self.query_row_blocks,
            "unbounded_fastest_eval_seconds": self.unbounded_fastest_eval_s,
            "head_chunked_fastest_eval_seconds": self.head_chunked_fastest_eval_s,
            "query_chunked_fastest_eval_seconds": self.query_chunked_fastest_eval_s,
            "bit_exact_head_fastest_eval_seconds": self.bit_exact_head_fastest_eval_s,
            "head_relative_max_abs": self.head_relative_max_abs,
            "query_relative_max_abs": self.query_relative_max_abs,
            "bit_exact_head_relative_max_abs": self.bit_exact_head_relative_max_abs,
        })
    }
}

/// Emit the rung-3 measurement receipt: to `MINIMAX_H3_EVIDENCE_OUT` when set, and always to stderr.
///
/// # Why this is not a `MEMORY_EVIDENCE_V1` line, stated rather than skipped
///
/// `MemoryRunOutcome::to_json_line` is the canonical receipt the SceneWorks side ingests into
/// `docs/generated/memory-calibration-evidence.json`, and it **requires a positive
/// `predicted_peak_bytes`** — a peak from the provider's fitted curve, checked against the observed
/// one. MiniMax-H3 has no fitted curve yet: `synthesize_estimate_ladder` needs the calibration
/// campaign that **sc-17153** owns, which is the last story of this chain, after this one. Minting a
/// `MEMORY_EVIDENCE_V1` record here would mean inventing the prediction half, and a fabricated
/// prediction validating against itself is precisely the kind of receipt the evidence gate exists to
/// refuse.
///
/// So this emits the measured half in a self-describing, versioned form, with every axis
/// `MemoryEvidenceKey` will need (tier, geometry, frames, strategy parameter) already on each cell,
/// and sc-17153 pairs it with the prediction. That is a sequencing fact recorded in both stories, not
/// an evidence gap discovered here.
pub fn write_bounded_attention_evidence(cells: &[BoundedAttentionCell]) {
    let document = serde_json::json!({
        "schema": "MINIMAX_H3_RUNG3_MEASUREMENT_V1",
        "story": "sc-18661",
        "provider": "minimax-h3",
        "backend": "mlx-metal",
        "phase": "denoise",
        "note": "Denoise-phase activation peak per attention plan, measured on a real 50-block DiT \
                 with synthetic drive. Conditioning and decode are separate phases with their own \
                 high-water marks and are deliberately outside the measured window.",
        "cells": cells.iter().map(BoundedAttentionCell::to_json).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&document).expect("serialize rung-3 receipt");
    eprintln!("\nMINIMAX_H3_RUNG3_MEASUREMENT_V1\n{text}");
    if let Ok(path) = std::env::var("MINIMAX_H3_EVIDENCE_OUT") {
        if !path.trim().is_empty() {
            std::fs::write(path.trim(), &text).expect("write rung-3 receipt");
        }
    }
}

/// A human-readable tier label for a staged DiT directory, from the tier's **own** `config.json`
/// quantization marker — the same authority `crate::model::reconcile_tier` treats as decisive —
/// rather than from the directory name, which a caller chooses freely.
pub fn describe_dit_tier(dir: &std::path::Path) -> String {
    match mlx_gen::quant::packed_quant_bits_at(dir) {
        Ok(Some(bits)) => format!("q{bits}"),
        Ok(None) => "bf16".to_owned(),
        Err(error) => panic!(
            "{}: cannot read the staged tier's quantization marker: {error}. An unlabelled tier \
             would make every measured cell unattributable.",
            dir.display()
        ),
    }
}

// --- sc-18662: rung-4 bounded-transformer-residency receipts and stack walks --------------------

/// A bare walk of a **resident** DiT block stack, modulation-free.
///
/// Rung 4 bounds *weight* residency, so the walk is deliberately the thinnest thing that touches
/// every block's tensors: `norm1 -> attn -> norm2 -> ff`, no AdaLN modulation and no rotary. Adding
/// them would put a 3.87 GB modulation table and a rotary grid inside a measurement window that the
/// window does not bound, diluting the ratio under test — and neither is a function of which blocks
/// are materialized, which is the question.
///
/// The windowed twin is [`walk_windowed_blocks`] and it runs the identical per-block call, so the
/// two arms differ in exactly one variable.
pub fn walk_resident_blocks(
    dit: &mlx_gen_minimax_h3::dit::MiniMaxH3Dit,
    hidden: &Array,
) -> mlx_gen::Result<Array> {
    let mut h = hidden.clone();
    for block in dit.blocks() {
        h = block.forward_body(&h)?;
    }
    Ok(h)
}

/// [`walk_resident_blocks`] over a bounded window — one block materialized at a time at `window` 1.
pub fn walk_windowed_blocks(
    stream: &mlx_gen_minimax_h3::DitBlockStream,
    plan: &mlx_gen::block_residency::BlockPlan,
    hidden: &Array,
) -> mlx_gen::Result<Array> {
    mlx_gen::block_residency::run_windowed(
        plan,
        &mlx_gen::CancelFlag::default(),
        hidden.clone(),
        || stream.open(),
        |mut h: Array, view: &mut Weights, range: std::ops::Range<usize>| {
            for index in range {
                let block = stream.materialize(view, index)?;
                h = block.forward_body(&h)?;
            }
            Ok(h)
        },
        |h: &Array| mlx_rs::transforms::eval([h]).map_err(Into::into),
    )
}

/// **Absolute per-constant bound on a declared peak (sc-17153, rider 1 of activity 20066).**
///
/// This replaces the `|measured/declared − 1| < 0.25` **ratio** bands that graded rung 4's
/// constants until sc-17153's re-measurement pass. Two defects motivated the change, both recorded
/// in the sc-18662 review:
///
/// 1. The ratio form was *multiplicatively asymmetric* — it admitted a declared constant up to
///    `1/(1 − 0.25) ≈ 1.333x` the measured one while allowing measured only `1.25x` declared.
/// 2. More seriously, a band on a **quotient** of two constants is blind to *pair-proportional*
///    drift: `RUNG4_TE_RESIDENT_PEAK_BYTES` and `RUNG4_TE_WINDOWED_PEAK_BYTES` (or a
///    `RUNG4_DIT_MEASURED_PEAKS` row's pair) moving by the same factor cancel exactly. The comments
///    of the day claimed the absolute request-peak bound caught that class; it does not — it reads
///    neither constant. So the class was caught by **nothing**.
///
/// Bounding each constant against its own measurement closes both: proportional drift moves each
/// term and is caught on both, and the bound is symmetric in bytes.
///
/// **Tolerance** is the harness's own committed convention — the larger of 256 MiB absolute or 5 %
/// relative (the `compare-reuse` tolerance in the **SceneWorks** repo's
/// `docs/memory-calibration-harness.md`; this repo has no copy of that document) — reused here so
/// this rung is graded the way every other cross-run memory comparison in the fleet is, rather than
/// by a number invented for it. The absolute floor is what makes it usable on the small windowed
/// constants, where 5 % of ~1.4 GB would be tighter than the counter's own resolution.
///
/// Measured spread this bound was sized against (sc-17153 terminal campaign, real weights, Metal):
/// the q4 DiT windowed peak was **bit-identical** across two runs, its resident peak moved 0.36 %
/// (~43 MB), and the q8 pair reproduced its declared constants to within 5 KB and 36 KB. The
/// "~1 GB of run-to-run spread" the retired comments cited was not reproduced on either arm.
///
/// # Two limits of this bound, stated rather than discovered later
///
/// 1. **It is single-machine.** Every figure it grades was taken on one `rw-mage` host. The label
///    `rw-mage` is on *both* Macs, so a scheduler that lands a real-weight arm on the other one is
///    grading against numbers nobody has replicated there. Nothing here detects that; the failure
///    mode is a red with a stale-constant message that names the wrong cause.
/// 2. **On the tightest cell it is narrower than that cell's own recorded history.**
///    `RUNG4_TE_WINDOWED_PEAK_BYTES` takes the 256 MiB floor, which is 5.96 % of the constant,
///    while the sc-18662 comments record window-1 peaks of 5.30 GB and 6.24-6.34 GB for what was
///    nominally the same cell — roughly ±9 %. That history is **not** evidence of counter noise at
///    the current pin: it was taken on mlx-rs `932beb4e6`, and the pin has since moved to
///    `7151a9b27` (see [`RUNG4_TE_WINDOWED_PEAK_BYTES`](mlx_gen_minimax_h3::memory_strategy::RUNG4_TE_WINDOWED_PEAK_BYTES)).
///    At `7151a9b27` the same cell reproduced across **eight** runs within ~500 KB (±0.01 %), which
///    is why the shared tolerance is kept rather than widened to cover a spread from a different
///    MLX build. A pin bump is the event that should be expected to move it again.
pub fn assert_declared_peak_constant(what: &str, declared: u64, measured: u64) {
    const ABSOLUTE_FLOOR_BYTES: u64 = 256 * 1024 * 1024;
    let tolerance = ABSOLUTE_FLOOR_BYTES.max((declared as f64 * 0.05) as u64);
    let delta = declared.abs_diff(measured);
    assert!(
        delta <= tolerance,
        "{what}: declared {declared} B against a measured {measured} B — off by {delta} B, over \
         the {tolerance} B bound (larger of 256 MiB or 5 %). The constant is stale, and the \
         contract's rung-4 declaration and the survey's requestPeak row are written from it. This \
         is a per-constant bound precisely so that proportional drift of a resident/windowed pair \
         cannot cancel out of a quotient."
    );
}

/// Emit a rung-4 measurement receipt: to `MINIMAX_H3_EVIDENCE_OUT` when set, always to stderr.
///
/// Same shape and the same sequencing caveat as [`write_bounded_attention_evidence`]: the measured
/// half is emitted here in a versioned, self-describing form carrying every axis
/// `MemoryEvidenceKey` needs, and sc-17153 pairs it with the fitted prediction that
/// `MemoryRunOutcome` requires.
///
/// (see [`assert_declared_peak_constant`] for the per-constant bound these receipts are graded by)
///
/// `cells` are `(window, peak_bytes, wall_seconds)`. **The wall figure is recorded but must not be
/// asserted on**: this Mac is also the `nax-macos` runner and has been measured at load average
/// 80-109, so a duration taken here is a sample of the fleet's contention, not of the rung. The peak
/// is a per-process allocator high-water mark and does not read machine load at all.
pub fn write_rung4_evidence(
    component: &str,
    tier: &str,
    cells: &[(usize, u64, f64)],
    resident_peak_bytes: u64,
) {
    let document = serde_json::json!({
        "schema": "MINIMAX_H3_RUNG4_MEASUREMENT_V1",
        "story": "sc-18662",
        "provider": "minimax-h3",
        "backend": "mlx-metal",
        "component": component,
        "tier": tier,
        "resident_peak_bytes": resident_peak_bytes,
        "wall_seconds_are_advisory": "measured on a machine that is also the nax-macos CI runner; \
                                      no assertion reads them",
        "cells": cells.iter().map(|(window, peak, wall)| serde_json::json!({
            "transformer_window_size": window,
            "peak_bytes": peak,
            "peak_ratio_below_resident": resident_peak_bytes as f64 / (*peak).max(1) as f64,
            "wall_seconds": wall,
        })).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&document).expect("serialize rung-4 receipt");
    eprintln!("\nMINIMAX_H3_RUNG4_MEASUREMENT_V1\n{text}");
    if let Ok(path) = std::env::var("MINIMAX_H3_EVIDENCE_OUT") {
        if !path.trim().is_empty() {
            let path = format!("{}.rung4-{component}-{tier}.json", path.trim());
            std::fs::write(path, &text).expect("write rung-4 receipt");
        }
    }
}

/// Emit the rung-4 **request-peak** receipt (sc-18662): the per-stage peaks of one end-to-end
/// `generate` on the deferred path, with the streamed selection on the request.
///
/// Same shape and the same sequencing caveat as [`write_rung4_evidence`]: the measured half only —
/// sc-17153 pairs it with the fitted prediction `MemoryRunOutcome` requires. The distinguishing
/// axis is `request_peak_bytes`: the MAX over the render's stage peaks, which is the quantity the
/// SceneWorks survey's `requestPeak` axis asks about and which no per-arm cell can stand in for.
#[allow(clippy::too_many_arguments)]
pub fn write_rung4_request_evidence(
    dit_tier: &str,
    te_tier: &str,
    width: u32,
    height: u32,
    frames: u32,
    steps: u32,
    stages: &[(&'static str, usize, usize)],
    request_peak_bytes: usize,
) {
    let document = serde_json::json!({
        "schema": "MINIMAX_H3_RUNG4_REQUEST_MEASUREMENT_V1",
        "story": "sc-18662",
        "provider": "minimax-h3",
        "backend": "mlx-metal",
        "task": "t2va",
        "transformer_window_size": 1,
        "transformer_window_component": "both",
        "dit_tier": dit_tier,
        "te_tier": te_tier,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "request_peak_bytes": request_peak_bytes,
        "wall_seconds_are_advisory": "measured on a machine that is also the nax-macos CI runner; \
                                      no assertion reads them",
        "stages": stages.iter().map(|(name, peak, held)| serde_json::json!({
            "stage": name,
            "peak_bytes": peak,
            "active_plus_cache_at_close_bytes": held,
        })).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&document).expect("serialize rung-4 request receipt");
    eprintln!("\nMINIMAX_H3_RUNG4_REQUEST_MEASUREMENT_V1\n{text}");
    if let Ok(path) = std::env::var("MINIMAX_H3_EVIDENCE_OUT") {
        if !path.trim().is_empty() {
            let path = format!("{}.rung4-request-{dit_tier}-{frames}f.json", path.trim());
            std::fs::write(path, &text).expect("write rung-4 request receipt");
        }
    }
}
