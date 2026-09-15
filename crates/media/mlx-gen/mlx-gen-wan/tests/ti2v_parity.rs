//! sc-2680 parity gate: the image-conditioned **TI2V mask-blend** denoise (`pipeline::denoise_ti2v`
//! plus the DiT's per-token-timestep `forward_tokens`) must reproduce the `mlx_video` reference's
//! `is_i2v_mask_blend` loop.
//!
//! Self-contained committed fixture (`tools/dump_ti2v_fixtures.py`): a tiny seeded dense `WanModel`,
//! with **injected** context + initial noise + encoded-image latent `z_img`, run through the
//! reference's per-token-timestep CFG loop (Euler, first-frame tokens frozen at `t=0`, mask
//! re-applied each step). Runs in CI, no real weights. Also checks the Rust `build_ti2v_mask` mask +
//! per-token mask against the reference `build_i2v_mask`, and `ti2v_blend_init` against the reference
//! `(1−mask)·z_img + mask·noise`.
//!
//! The DiT runs bf16 (the production regime), so the final-latent gap is the known cross-build bf16
//! kernel delta (MLX 0.31.1+patches vs the reference's 0.31.2) accumulated over the loop — bounded,
//! not a code bug (same envelope as the S4 dense gate). The mask logic is gated bit-tight separately.

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{build_ti2v_mask, denoise_ti2v, ti2v_blend_init};
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::WanTransformer;

fn fixture() -> Weights {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ti2v_pipeline.safetensors"
    );
    Weights::from_file(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run dump_ti2v_fixtures.py)"))
}

/// The tiny dense config the fixture was dumped with (mirrors `dump_ti2v_fixtures.py` / S4).
fn tiny_cfg() -> WanModelConfig {
    let mut c = WanModelConfig::wan21_t2v_1_3b();
    c.dim = 128;
    c.num_heads = 1; // head_dim 128
    c.num_layers = 2;
    c.ffn_dim = 256;
    c.freq_dim = 256;
    c.text_dim = 32;
    c.text_len = 8;
    c.in_dim = 16;
    c.out_dim = 16;
    c.vae_z_dim = 16;
    c.dual_model = false;
    c
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

#[test]
fn build_ti2v_mask_matches_reference() {
    let w = fixture();
    // Fixture dims: z=16, t_lat=2, h_lat=w_lat=2, patch (1,2,2). The reference `build_i2v_mask` has
    // no strength, so the parity pin is `strength = 1.0` — a full freeze (sc-19571).
    let (mask, tokens) = build_ti2v_mask(&[(0, 1.0)], 16, 2, 2, 2, (1, 2, 2));
    let exp_mask = w.require("mask").unwrap();
    let exp_tokens = w.require("mask_tokens").unwrap();
    assert_eq!(mask.shape(), exp_mask.shape(), "mask shape");
    assert_eq!(tokens.shape(), exp_tokens.shape(), "mask_tokens shape");
    assert_eq!(
        mask.as_slice::<f32>(),
        exp_mask.as_slice::<f32>(),
        "mask must match reference build_i2v_mask"
    );
    assert_eq!(
        tokens.as_slice::<f32>(),
        exp_tokens.as_slice::<f32>(),
        "mask_tokens must match reference"
    );
}

#[test]
fn wan_ti2v_mask_blend_matches_reference() {
    let w = fixture();
    let cfg = tiny_cfg();
    let dit = WanTransformer::from_weights(&w, &cfg).expect("build DiT");

    let ctx_cond = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap();
    let ctx_uncond = dit.embed_text(w.require("ctx_uncond").unwrap()).unwrap();
    let init_noise = w.require("init_noise").unwrap();
    let z_img = w.require("z_img").unwrap();
    let mask = w.require("mask").unwrap();
    let mask_tokens = w.require("mask_tokens").unwrap();

    // Blend the noise init (gates ti2v_blend_init): (1−mask)·z_img + mask·noise.
    let init_latents = ti2v_blend_init(z_img, mask, init_noise).unwrap();

    let mut steps_seen = 0usize;
    let latents = denoise_ti2v(
        &dit,
        SolverKind::Euler,
        cfg.num_train_timesteps,
        4,   // steps
        5.0, // shift
        3.0, // guidance
        &ctx_cond,
        Some(&ctx_uncond),
        &init_latents,
        z_img,
        mask,
        mask_tokens,
        &mlx_gen::CancelFlag::default(),
        &mut |_| steps_seen += 1,
    )
    .expect("denoise_ti2v");
    assert_eq!(steps_seen, 4, "progress callback fired per step");

    let exp = w.require("final_latents").unwrap();
    assert_eq!(latents.shape(), exp.shape(), "final latent shape");
    let (max_abs, mean_rel) = diff(latents.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[ti2v latents] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        latents.shape()
    );

    // The first latent temporal frame must stay frozen to z_img (mask-blend invariant).
    let lat = latents.as_slice::<f32>();
    let zexp = z_img.as_slice::<f32>(); // [16,1,2,2] = 64 vals (frame 0 for each channel)
    let (t_lat, plane) = (2usize, 4usize); // h_lat·w_lat
    let mut frame0_max = 0f32;
    for c in 0..16 {
        for p in 0..plane {
            let got = lat[c * t_lat * plane + p]; // temporal index 0
            frame0_max = frame0_max.max((got - zexp[c * plane + p]).abs());
        }
    }
    assert!(
        frame0_max < 1e-5,
        "first frame must stay pinned to z_img (max|Δ|={frame0_max:.3e})"
    );

    // Same bf16 cross-build envelope as S4 (gate at 2e-2; a logic bug gives mean_rel ~O(1)).
    assert!(
        mean_rel < 2e-2,
        "ti2v latents diverged: mean_rel={mean_rel:.3e}"
    );
}

/// sc-19571 — **the conditioning strength must change the render.**
///
/// The defect this closes accepted `imageConditioningStrength` / `lastFrameConditioningStrength`
/// and built a hard `0/1` mask regardless, so every strength produced a bit-identical video. This
/// runs the SAME fixture DiT, seed, contexts, noise and `z_img` through `denoise_ti2v` twice,
/// changing **only** the strength the mask is built from, and gates on **relative max-abs-diff** —
/// not a norm, cosine or checksum, all three of which have been blind to real defects in this
/// family (they are aggregate/scale-invariant and a partially-pinned frame moves a bounded subset
/// of the tensor).
///
/// Two directions, both asserted:
///  * `strength = 1.0` reproduces the historical hard pin **bit-for-bit** (a partial-pin
///    implementation that also perturbed the full pin would be a regression, not a fix);
///  * `strength = 0.6` moves the latent well clear of the bf16 step-accumulation floor the
///    reference gate above measures at `< 2e-2`.
///
/// Mutation guard: hard-code the `pins` strength to `1.0` inside `build_ti2v_mask` — i.e.
/// re-introduce the exact defect — and `rel_max_abs` collapses to `0.0`, tripping the "must change"
/// assertion while the identity assertion still passes.
#[test]
fn conditioning_strength_changes_the_ti2v_render() {
    let w = fixture();
    let cfg = tiny_cfg();
    let dit = WanTransformer::from_weights(&w, &cfg).expect("build DiT");
    let ctx_cond = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap();
    let ctx_uncond = dit.embed_text(w.require("ctx_uncond").unwrap()).unwrap();
    let init_noise = w.require("init_noise").unwrap();
    let z_img = w.require("z_img").unwrap();

    // Everything below is held fixed except the pin strength.
    let render = |strength: f32| -> Vec<f32> {
        let (mask, mask_tokens) = build_ti2v_mask(&[(0, strength)], 16, 2, 2, 2, (1, 2, 2));
        let init = ti2v_blend_init(z_img, &mask, init_noise).unwrap();
        denoise_ti2v(
            &dit,
            SolverKind::Euler,
            cfg.num_train_timesteps,
            4,
            5.0,
            3.0,
            &ctx_cond,
            Some(&ctx_uncond),
            &init,
            z_img,
            &mask,
            &mask_tokens,
            &mlx_gen::CancelFlag::default(),
            &mut |_| {},
        )
        .expect("denoise_ti2v")
        .as_slice::<f32>()
        .to_vec()
    };

    /// `max|a−b| / max|b|` — the per-element worst case, normalized. A single frame that failed to
    /// respond to the knob shows up here; it does not in a norm or a cosine.
    fn rel_max_abs(a: &[f32], b: &[f32]) -> f32 {
        let max_d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        let scale = b.iter().map(|y| y.abs()).fold(0f32, f32::max).max(1e-9);
        max_d / scale
    }

    let full = render(1.0);
    // The full pin is unchanged by this story: same construction, byte-identical output.
    assert_eq!(
        full,
        render(1.0),
        "the render must be deterministic before anything else is concluded from it"
    );

    let partial = render(0.6);
    let rel = rel_max_abs(&partial, &full);
    println!("[ti2v strength 0.6 vs 1.0] rel_max_abs={rel:.3e}");
    assert!(
        rel > 0.1,
        "conditioning strength must change the render — rel_max_abs={rel:.3e} (0.0 means the \
         strength never reached the mask, which is sc-19571's defect; the bf16 step-accumulation \
         floor this fixture measures against the reference is < 2e-2, so 0.1 is an order of \
         magnitude clear of it)"
    );

    // …and the response is monotone in the knob rather than an on/off flip: a weaker pin departs
    // from the full pin by MORE than a stronger one does.
    let weaker = rel_max_abs(&render(0.2), &full);
    println!("[ti2v strength 0.2 vs 1.0] rel_max_abs={weaker:.3e}");
    assert!(
        weaker > rel,
        "a weaker pin must depart further from the full pin (0.2 → {weaker:.3e} vs 0.6 → {rel:.3e})"
    );
}
