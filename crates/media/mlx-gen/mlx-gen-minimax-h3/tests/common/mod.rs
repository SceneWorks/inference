#![allow(dead_code)]
//! Shared helpers. `mod common;` is compiled into every test binary, so only a subset is used by
//! any one of them.

use std::path::PathBuf;

use mlx_rs::Array;

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
    }
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
