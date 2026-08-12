#![allow(dead_code)]
//! Shared helpers. `mod common;` is compiled into every test binary, so only a subset is used by
//! any one of them.

use std::path::PathBuf;

use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::{BigVganConfig, MiniMaxH3AudioVaeConfig, MiniMaxH3VaeConfig};

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
/// Two structural properties of the shipped model are preserved rather than shrunk away:
/// `heads · head_dim` (96) stays **wider** than `hidden_size` (64), as 7168 > 5376; and
/// `2 · 3 · rope_freq_dim` (12) stays **narrower** than `attention_head_dim` (24), as 96 < 128, so
/// the partial-rotary path is the one under test.
pub fn dit_fixture_config() -> mlx_gen_minimax_h3::MiniMaxH3DitConfig {
    mlx_gen_minimax_h3::MiniMaxH3DitConfig {
        num_attention_heads: 4,
        attention_head_dim: 24,
        hidden_size: 64,
        num_layers: 2,
        num_refiner_layers: 2,
        ffn_dim: 32,
        text_dim: 40,
        freq_dim: 16,
        time_embed_hidden_dim: 64,
        time_embed_dim: 48,
        rope_freq_dim: 2,
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
    num_text_tokens: 5,
    num_audio_latents: 3,
    audio_channels: 2,
    num_latent_frames: 3,
    latent_height: 4,
    latent_width: 6,
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
        sample_rate: 32_000,
        output_channels: 2,
        latent_channels: 32,
        encoder_dim: 8,
        encoder_rates: vec![2, 2],
        decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
        decoder_dim: 128,
        attn_proj: true,
        decoder_type: "bigvgan".into(),
        // The REAL 32-entry de-normalization statistics, not placeholders.
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        bigvgan: BigVganConfig::for_sample_rate(32_000, 64, 128).expect("32 kHz is supported"),
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
        latent_channels: 24,
        out_channels: 3,
        num_layers: 2,
        num_heads: 2,
        head_dim: 16,
        num_register_tokens: 4,
        ffn_mult: 4,
        rope_theta: 100.0,
        rope_dim_ratio: 0.75,
        norm_eps: 1e-5,
        patch_size: 2,
        patch_size_t: 4,
        clip_length: 17,
        token_drop,
        // The REAL 24-entry de-normalization statistics, not placeholders.
        latents_mean: shipped.latents_mean.clone(),
        latents_std: shipped.latents_std.clone(),
        // The encoder fields the struct gained in sc-17148. This fixture carries NO `encoder.*`
        // tensors — it is the decode golden and its bytes are shared with
        // `candle-gen-minimax-h3` — so these only have to satisfy `validate_encoder`'s
        // "the cumprods are the patch sizes" rule. `MiniMaxH3VideoVae::from_weights` loads the
        // encode half only when it is present; see [`ENCODE_FIXTURE`] for the one that has it.
        in_channels: 3,
        block_out_channels: vec![32, 32, 32, 32],
        layers_per_block: 1,
        spatial_downsample_factors: vec![2, 1, 1, 1],
        temporal_downsample_factors: vec![1, 2, 2, 1],
        norm_num_groups: 32,
        encoder_norm_eps: 1e-6,
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
        patch_size: 4,
        block_out_channels: vec![32, 32, 32, 64],
        spatial_downsample_factors: vec![1, 2, 2, 1],
        temporal_downsample_factors: vec![1, 2, 2, 1],
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

/// Standard deviation of a tensor — a golden whose expected output is ~constant proves nothing.
pub fn std_dev(x: &Array) -> f32 {
    let x = x.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let mean: f32 = x.mean(None).unwrap().item();
    let centered = mlx_rs::ops::subtract(&x, Array::from_f32(mean)).unwrap();
    let var: f32 = centered.square().unwrap().mean(None).unwrap().item();
    var.sqrt()
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
