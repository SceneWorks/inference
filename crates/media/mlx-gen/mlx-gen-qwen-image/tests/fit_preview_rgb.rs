//! Re-derive `preview::RGB_FACTORS` / `RGB_BIAS` — the producer for the linear latent→RGB fit.
//!
//! # Why this exists
//!
//! `preview.rs` carries the fit as source constants and tells a future maintainer to **"refit
//! whenever the VAE lineage changes, by dumping the two halves of a real render and re-solving."**
//! Until this file, that instruction named a procedure with no implementation: nothing in the tree
//! rendered the pairs, and nothing solved the system. The constants were therefore unreproducible —
//! not merely un-re-run, but *un-checkable*, since a reviewer had no way to ask whether R² = 0.9586
//! was right.
//!
//! That is the same defect class as the single-host parity goldens (issue #311), and worse in one
//! respect: a golden is a gitignored test input, while these are shipping constants in `src/`.
//!
//! # What it does
//!
//! Renders the fit corpus, then solves `decoded_rgb ≈ latent · M + b` by ordinary least squares:
//!
//! 1. render each (prompt, seed) at 1024², 8-step Lightning;
//! 2. take the **final unpacked latent** `[1, 16, 128, 128]` — the same tensor `preview::project`
//!    sees, so the fit is over exactly the input it will be applied to;
//! 3. decode it, denormalize to `[0, 1]` the way [`decoded_to_image`] does, and **8×-average-pool**
//!    to latent resolution, giving one RGB target per latent position;
//! 4. accumulate the normal equations over every position of every render and solve.
//!
//! # It reports rather than asserts agreement
//!
//! The committed constants were fit from a corpus that no longer exists, so this cannot claim to
//! reproduce them: a re-solve here is a *different sample* of the same VAE, not a replay. Asserting
//! equality would fail on sampling noise and mean nothing when it passed. So the test asserts the
//! property that is actually invariant — the fit explains most of the variance ([`R2_FLOOR`]) — and
//! **prints** both the new constants and their delta against the committed ones for a human to read.
//!
//! A large delta is a finding, not a failure: it means the committed fit does not describe this VAE
//! at this configuration, and the printed block is what should replace it.
//!
//! ```sh
//! QWEN_IMAGE_SNAPSHOT=/path/to/Qwen-Image \
//!   cargo test -p mlx-gen-qwen-image --release --test fit_preview_rgb -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use mlx_gen::{CancelFlag, PreviewSink};
use mlx_gen_qwen_image::pipeline::{encode_prompt, unpack_latents, LATENT_CHANNELS, SPATIAL_SCALE};
use mlx_gen_qwen_image::sampler::lightning_sigmas;
use mlx_gen_qwen_image::{create_noise, denoise_with_progress, loader};
use mlx_rs::ops::{add, maximum, mean_axes, minimum, multiply};
use mlx_rs::Dtype;

use mlx_gen::array::scalar;

/// The fit corpus. Two prompts and two seeds, matching what `preview.rs` records for the committed
/// fit — a corpus that is *stated* there but was never expressible in code until now.
///
/// Two is deliberate rather than lazy: the fit is a global linear map, so the risk is a corpus whose
/// colour statistics are narrow (one palette, one lighting), not one that is small. The two prompts
/// are chosen to disagree — warm artificial light against cool daylight — so a channel that only
/// matters in one of them still carries weight.
const CORPUS: [(&str, u64); 2] = [
    (
        "a highly detailed photograph of a bustling night market food stall, glowing paper \
         lanterns, a neon sign reading OPEN, steam rising from a wok, reflections on wet \
         cobblestones, shallow depth of field, 35mm",
        0,
    ),
    (
        "a wide landscape photograph of a snow-covered mountain range at midday under a clear blue \
         sky, pine forest in the foreground, crisp shadows, high dynamic range",
        1,
    ),
];

const W: u32 = 1024;
const H: u32 = 1024;
const STEPS: usize = 8;

/// Lightning is CFG-free — the distilled schedule takes no negative branch.
const GUIDANCE: f32 = 1.0;

/// The floor the solved fit must clear.
///
/// Set below the committed 0.9586 rather than at it: this is a different corpus, so the honest
/// question is "does a linear map still explain this latent space", not "does it match to 4 decimal
/// places". A fit that drops under this is telling you the linear approximation itself has stopped
/// working for this VAE — at which point the preview needs a different projection, not new numbers.
const R2_FLOOR: f64 = 0.90;

/// The constants currently in `preview.rs`, for the delta report. Kept here rather than made `pub`
/// in the crate: this file is the only reader, and widening a provider's public surface to let a
/// test look at it is the wrong trade.
const COMMITTED_FACTORS: [[f32; 3]; 16] = [
    [-0.00986379, 0.0257554, 0.211834],
    [-0.00150066, -0.00355605, 0.00219657],
    [0.0881243, 0.0565462, 0.0390654],
    [0.166173, 0.180288, 0.0838119],
    [0.0081918, -0.00272948, -0.0139806],
    [0.0276023, -0.0379166, -0.0372937],
    [-0.144053, -0.167288, -0.107295],
    [-0.0423725, -0.004423, 0.00174681],
    [-0.0705916, -0.0879479, -0.17535],
    [-0.0603724, 0.0326614, 0.0934403],
    [0.0473827, 0.121914, 0.0651104],
    [0.0138456, 0.0267495, 0.0120851],
    [-0.0844989, -0.0160223, 0.0123298],
    [-0.0162293, -0.0335703, -0.018524],
    [0.111816, 0.050061, 0.0724697],
    [0.0448471, 0.0208121, 0.0407526],
];
const COMMITTED_BIAS: [f32; 3] = [0.406258, 0.385829, 0.287052];

fn snapshot() -> PathBuf {
    PathBuf::from(std::env::var("QWEN_IMAGE_SNAPSHOT").unwrap_or_else(|_| {
        panic!(
            "set QWEN_IMAGE_SNAPSHOT to the required snapshot dir; inference never self-fetches or \
             derives a cache location (epic 13657)"
        )
    }))
}

/// One render's `(latent [n, 16], rgb [n, 3])` pair, host-side as f64 for the accumulation.
struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
}

/// Render one corpus entry and return its aligned latent/RGB samples.
fn render_pair(root: &Path, prompt: &str, seed: u64) -> Pair {
    let tf = loader::load_transformer(root).unwrap();
    let te = loader::load_text_encoder(root).unwrap();
    let tok = loader::load_tokenizer(root).unwrap();
    let pos = encode_prompt(&tok, &te, prompt, "qwen_image").unwrap();
    drop(te);

    let sigmas = lightning_sigmas(STEPS);
    let cancel = CancelFlag::default();
    let latents = create_noise(seed, W, H).unwrap();
    let packed = denoise_with_progress(
        &tf,
        None,
        &sigmas,
        seed,
        latents,
        &pos,
        None,
        GUIDANCE,
        W,
        H,
        0,
        &cancel,
        &PreviewSink::default(),
        &mut |_| {},
    )
    .unwrap();
    drop(tf);

    // The exact tensor `preview::project` receives, so the fit's domain is the projection's domain.
    let unpacked = unpack_latents(&packed, W, H).unwrap();
    let vae = loader::load_vae(root).unwrap();
    let decoded = vae
        .decode(&unpacked)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    drop(vae);

    let lh = (H / SPATIAL_SCALE) as i32;
    let lw = (W / SPATIAL_SCALE) as i32;
    let c = LATENT_CHANNELS;

    // Latent [1, 16, lh, lw] -> [n, 16], the same reshape/transpose `project` performs.
    let x = unpacked
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[c, lh * lw])
        .unwrap()
        .transpose_axes(&[1, 0])
        .unwrap();

    // Decode [1, 3, H, W] -> [0, 1] exactly as `decoded_to_image` denormalizes, then average-pool
    // 8x8 blocks down to latent resolution so every sample is one latent position and the RGB it
    // corresponds to. Average rather than subsample: a point sample would fit the projection to one
    // arbitrary pixel of each block instead of the block's colour.
    let half = scalar(0.5);
    let y = add(multiply(&decoded, half.clone()).unwrap(), half).unwrap();
    let y = minimum(maximum(y, scalar(0.0)).unwrap(), scalar(1.0)).unwrap();
    let s = SPATIAL_SCALE as i32;
    let y = y
        .reshape(&[1, 3, lh, s, lw, s])
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let y = mean_axes(&y, &[3, 5], false).unwrap(); // [1, 3, lh, lw]
    let y = y
        .reshape(&[3, lh * lw])
        .unwrap()
        .transpose_axes(&[1, 0])
        .unwrap();

    let n = (lh * lw) as usize;
    Pair {
        latent: x.as_slice::<f32>().iter().map(|&v| v as f64).collect(),
        rgb: y.as_slice::<f32>().iter().map(|&v| v as f64).collect(),
        n,
    }
}

/// Solve `A z = b` for each of the 3 RGB columns by Gauss-Jordan with partial pivoting.
///
/// 17x17 (16 channels + intercept), so a dense host-side solve costs nothing and avoids depending on
/// a backend linalg surface for what is arithmetic.
// Index form on purpose: Gauss-Jordan advances `a` and `b` in lockstep by row and column, and the
// textbook shape is what makes an off-by-one visible here. An iterator rewrite hides exactly the
// indices a reader needs to check.
#[allow(clippy::needless_range_loop)]
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
    let n = a.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())
            .unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        let d = a[col][col];
        assert!(
            d.abs() > 1e-12,
            "normal equations are singular at column {col}"
        );
        for k in col..n {
            a[col][k] /= d;
        }
        for k in 0..3 {
            b[col][k] /= d;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            for k in 0..3 {
                b[row][k] -= f * b[col][k];
            }
        }
    }
    b
}

#[test]
#[ignore = "needs the Qwen/Qwen-Image snapshot (QWEN_IMAGE_SNAPSHOT); renders 2x1024^2"]
#[allow(clippy::needless_range_loop)] // see `solve` — normal-equation assembly indexes in lockstep
fn fit_preview_rgb_factors() {
    let root = snapshot();
    let c = LATENT_CHANNELS as usize;
    let dim = c + 1; // + intercept

    // Normal equations, accumulated across renders so peak memory is one render's samples.
    let mut xtx = vec![vec![0.0f64; dim]; dim];
    let mut xty = vec![[0.0f64; 3]; dim];
    let mut sum_y = [0.0f64; 3];
    let mut sum_y2 = [0.0f64; 3];
    let mut total = 0usize;
    let mut pairs: Vec<Pair> = Vec::new();

    for (prompt, seed) in CORPUS {
        let pair = render_pair(&root, prompt, seed);
        eprintln!("  rendered seed {seed}: {} samples", pair.n);
        for i in 0..pair.n {
            let mut row = [0.0f64; 17];
            row[..c].copy_from_slice(&pair.latent[i * c..(i + 1) * c]);
            row[c] = 1.0;
            let y = &pair.rgb[i * 3..i * 3 + 3];
            for a in 0..dim {
                for b in a..dim {
                    xtx[a][b] += row[a] * row[b];
                }
                for (k, &yk) in y.iter().enumerate() {
                    xty[a][k] += row[a] * yk;
                }
            }
            for (k, &yk) in y.iter().enumerate() {
                sum_y[k] += yk;
                sum_y2[k] += yk * yk;
            }
        }
        total += pair.n;
        pairs.push(pair);
    }
    // Mirror the upper triangle — the accumulation above only filled `b >= a`.
    for a in 0..dim {
        for b in 0..a {
            xtx[a][b] = xtx[b][a];
        }
    }

    let coef = solve(xtx, xty);

    // R^2 over the whole corpus, computed against the SAME samples the system was built from.
    let mut ss_res = [0.0f64; 3];
    for pair in &pairs {
        for i in 0..pair.n {
            let lat = &pair.latent[i * c..(i + 1) * c];
            for k in 0..3 {
                let mut pred = coef[c][k];
                for (j, &l) in lat.iter().enumerate() {
                    pred += l * coef[j][k];
                }
                let d = pair.rgb[i * 3 + k] - pred;
                ss_res[k] += d * d;
            }
        }
    }
    let mut r2 = [0.0f64; 3];
    for k in 0..3 {
        let mean = sum_y[k] / total as f64;
        let ss_tot = sum_y2[k] - total as f64 * mean * mean;
        r2[k] = 1.0 - ss_res[k] / ss_tot;
    }
    let r2_overall = 1.0
        - ss_res.iter().sum::<f64>()
            / (0..3)
                .map(|k| {
                    let mean = sum_y[k] / total as f64;
                    sum_y2[k] - total as f64 * mean * mean
                })
                .sum::<f64>();

    // The block to paste into `preview.rs`, in its exact source form.
    println!(
        "\n// {total} samples, {} renders, {STEPS}-step Lightning at {W}x{H}",
        CORPUS.len()
    );
    println!(
        "// R^2 = {r2_overall:.4} (r {:.4}, g {:.4}, b {:.4})",
        r2[0], r2[1], r2[2]
    );
    println!("const RGB_FACTORS: [[f32; 3]; {c}] = [");
    for row in coef.iter().take(c) {
        println!("    [{:.6}, {:.6}, {:.6}],", row[0], row[1], row[2]);
    }
    println!("];");
    println!(
        "const RGB_BIAS: [f32; 3] = [{:.6}, {:.6}, {:.6}];",
        coef[c][0], coef[c][1], coef[c][2]
    );

    // Delta against what ships, so drift is visible without being fatal (see the module docs).
    let mut max_factor_delta = 0.0f64;
    for (j, row) in COMMITTED_FACTORS.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            max_factor_delta = max_factor_delta.max((coef[j][k] - v as f64).abs());
        }
    }
    let max_bias_delta = (0..3)
        .map(|k| (coef[c][k] - COMMITTED_BIAS[k] as f64).abs())
        .fold(0.0f64, f64::max);
    println!(
        "\nvs committed: max |Δfactor| = {max_factor_delta:.6}, max |Δbias| = {max_bias_delta:.6}"
    );
    println!(
        "(committed fit reports R^2 = 0.9586 over 32768 samples; its corpus no longer exists, so \
         this is a fresh sample of the same VAE rather than a replay — read the deltas, not an \
         equality)"
    );

    assert!(
        r2_overall > R2_FLOOR,
        "linear latent->RGB fit explains only R^2 = {r2_overall:.4} (floor {R2_FLOOR}); the \
         approximation itself has stopped working for this VAE, which needs a different projection \
         rather than new constants"
    );
}
