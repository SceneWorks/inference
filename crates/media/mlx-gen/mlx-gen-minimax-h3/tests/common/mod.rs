#![allow(dead_code)]
//! Shared helpers. `mod common;` is compiled into every test binary, so only a subset is used by
//! any one of them.

use std::path::PathBuf;

use mlx_rs::Array;

use mlx_gen_minimax_h3::MiniMaxH3VaeConfig;

/// The committed parity fixture, produced by `tools/dump_minimax_h3_video_vae.py` running the
/// Apache-2.0 reference implementation from the MiniMax-H3 snapshot.
pub const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/video_vae_decode.safetensors"
);

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
