//! Velocity-convention regression guard for mlx-gen-anima (sc-10515).
//!
//! The Anima DiT is a **standard flow denoiser**: for a flow-matched sample
//! `x_σ = (1 − σ)·x0 + σ·ε` it predicts the flow velocity `v ≈ ε − x0`, embedding the **raw σ** as its
//! timestep. `run_flow_sampler` (`TimestepConvention::Sigma`) consumes that output directly and
//! integrates `x + (σ_next − σ)·v` — **no** negation, **no** `1 − σ`/`σ·1000` timestep rescale (see
//! `pipeline.rs`). A sign flip or a bad timestep scale silently collapses generation to a wash/noise.
//!
//! This is the ONLY direct guard on that sign/timestep convention, so it lives in its **own
//! integration-test binary**: it materialises many arrays, and mlx-rs shares one Metal default stream
//! across a test binary, so running it alongside the other real-weights tests can cross-contaminate.
//! Its own binary sidesteps that entirely. Real-weights-gated + `#[ignore]`d exactly like
//! `tests/real_weights.rs`, so it never runs in CI. Run it alone with:
//!   cargo test -p mlx-gen-anima --test velocity_convention -- --ignored --nocapture
//!
//! Measurement (reproduces the port-time check): VAE-encode the checkpoint's shipped real anime image
//! `example.png` to a KNOWN on-manifold latent `x0`, re-noise it to `x_σ` at a mid σ, run the DiT, and
//! measure `cos(v_pred, ε − x0)`. On a correct port this is strongly positive (≈0.9+); a negated
//! velocity flips it strongly negative, and a `σ·1000` timestep collapses it toward 0.

use std::path::PathBuf;

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::WeightsSource;
use mlx_gen_anima::pipeline::AnimaPipeline;
use mlx_gen_anima::Variant;

/// Glob the Anima snapshot's `split_files/` dir from the HF cache (no hardcoded sha). Mirrors
/// `tests/real_weights.rs`.
fn split_files() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--circlestone-labs--Anima/snapshots");
    std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .find_map(|e| {
            let p = e.path().join("split_files");
            p.join("diffusion_models").is_dir().then_some(p)
        })
}

/// The checkpoint's shipped 1024² real anime sample (sibling of `split_files/`).
fn example_png(split: &std::path::Path) -> PathBuf {
    split.parent().expect("snapshot root").join("example.png")
}

/// Load `example.png` → NCHW `[1, 3, H, W]` f32 normalised to `[-1, 1]` (the VAE encode input range).
fn image_to_nchw(path: &std::path::Path) -> Array {
    let img = image::open(path).expect("open example.png").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut chw = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                chw[c * h * w + y * w + x] = p[c] as f32 / 127.5 - 1.0;
            }
        }
    }
    Array::from_slice(&chw, &[1, 3, h as i32, w as i32])
}

/// Cosine similarity over the full flattened tensors. Both inputs must be contiguous f32 (they are —
/// each is the fresh result of an elementwise op / dtype cast, not a strided view).
fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.as_slice::<f32>();
    let b = b.as_slice::<f32>();
    assert_eq!(a.len(), b.len());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

#[test]
#[ignore = "needs the circlestone-labs/Anima snapshot; real weights + Metal"]
fn dit_predicts_flow_velocity_eps_minus_x0() {
    let split = split_files().expect("Anima snapshot");
    let pipeline =
        AnimaPipeline::from_source(&WeightsSource::Dir(split.clone()), Variant::Base).unwrap();
    let dit = &pipeline.components().dit;
    let vae = &pipeline.components().vae;

    // KNOWN on-manifold latent: VAE-encode the shipped real anime image → [1, 16, 1, 128, 128].
    let x0 = vae.encode(&image_to_nchw(&example_png(&split))).unwrap();

    // Flow-matched sample x_σ = (1 − σ)·x0 + σ·ε at a mid σ (raw, matching the DiT timestep = σ).
    let sigma = 0.5f32;
    let eps =
        random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1234).unwrap())).unwrap();
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s_scale = Array::from_slice(&[sigma], &[1]);
    let x_sigma = add(
        multiply(&x0, &one_minus).unwrap(),
        multiply(&eps, &s_scale).unwrap(),
    )
    .unwrap();

    // Predict velocity (cond-only, no CFG — the sign convention is independent of guidance).
    let cond = pipeline
        .encode_prompt("an anime girl with long silver hair, detailed illustration, masterpiece")
        .unwrap();
    let sigma_ts = Array::from_slice(&[sigma], &[1]);
    let v_pred = dit
        .forward(&x_sigma, &sigma_ts, &cond, Dtype::Bfloat16)
        .unwrap()
        .as_dtype(Dtype::Float32) // contiguous f32 copy
        .unwrap();

    // Flow-match target: v = ε − x0.
    let target = subtract(&eps, &x0).unwrap();

    let cos = cosine(&v_pred, &target);
    println!("[velocity_convention] cos(v_pred, eps - x0) = {cos:.4} (sigma = {sigma})");
    // A correct port aligns strongly with ε − x0 (≈0.9+). A negated velocity gives ≈ −0.9; a σ·1000
    // timestep collapses toward 0. 0.5 is a wide, unambiguous floor that only a correct sign/timestep
    // clears — the mutation guard (negate `proj_out` in CosmosDiT::forward) drives cos strongly
    // negative, well under this floor.
    assert!(
        cos > 0.5,
        "DiT velocity does not align with the flow target ε − x0 (cos {cos:.4}); a sign or timestep \
         convention error would produce this"
    );
}
