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
//! # sc-17515: the first execution measured R² = 0.0114, and the fit was never the problem
//!
//! sc-17284 gave this file its first run anywhere and it failed its own floor by two orders of
//! magnitude, with a re-solved intercept identical in all three channels to four decimals. Two
//! causes were proposed — a degenerate corpus, or a moved VAE lineage — and **neither held**. The
//! defect was in this file, on the two lines that read the samples back to the host:
//!
//! `mlx-rs` `as_slice` returns the array's **physical buffer, ignoring strides**, so a
//! `transpose_axes`-terminated array reads back in its *pre-transpose* order. Both host reads below
//! ended in a transpose. Sample *i* therefore paired 16 consecutive values of latent channel 0 with
//! 3 consecutive pixels of the red channel — unrelated quantities, hence R² ≈ 0, and three targets
//! drawn from one channel, hence the three identical intercepts. The flattening `reshape` that fixes
//! it is the same one `mlx-gen-z-image`'s sibling producer already carried; of the seven MLX
//! producers this was the only one that read a transpose-terminated array without it.
//!
//! Measured on nax-macos against `SceneWorks/qwen-image-mlx@8080a417` bf16, 32768 samples:
//!
//! | corpus | physical read (the bug) | logical read (fixed) |
//! |---|---|---|
//! | no LoRA, as this file shipped | 0.0114 | 0.9766 |
//! | + 8-step Lightning LoRA | 0.0027 | 0.9521 |
//!
//! Applying the LoRA made the *broken* number worse, not better, which is what eliminated the
//! degenerate-corpus theory outright; the renders were coherent photographs either way (std 48–77,
//! 246–256 distinct levels). And the **committed** constants, scored unchanged on that corpus, come
//! back at R² = 0.9450 — so the VAE has not moved out from under them either, and they were left
//! alone. The `RGB_FACTORS` note in `src/preview.rs` carries the same figures.
//!
//! # The corpus applies the Lightning LoRA
//!
//! Both this file and `preview.rs` describe the fit as "8-step Lightning", but the corpus used to
//! take only [`lightning_sigmas`] — the distilled *schedule* without the distillation *adapter*.
//! That is not what the constants document, and it is measurable: without the LoRA the re-solve
//! lands 0.26 from the committed factors, with it 0.082. Since both clear the floor, this is a
//! fidelity fix rather than a way to go green.
//!
//! ```sh
//! MLX_GEN_QWEN_SNAPSHOT=/path/to/Qwen-Image/bf16 \
//!   QWEN_LIGHTNING_SNAPSHOT=/path/to/models--lightx2v--Qwen-Image-Lightning/snapshots/<rev> \
//!   cargo test -p mlx-gen-qwen-image --release --test fit_preview_rgb -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use mlx_gen::{AdapterKind, AdapterSpec, CancelFlag, PreviewSink};
use mlx_gen_qwen_image::adapters::apply_qwen_adapters;
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
    PathBuf::from(std::env::var("MLX_GEN_QWEN_SNAPSHOT").unwrap_or_else(|_| {
        panic!(
            "set MLX_GEN_QWEN_SNAPSHOT to the required snapshot dir; inference never self-fetches or \
             derives a cache location (epic 13657)"
        )
    }))
}

/// The pinned 8-step lightx2v Lightning LoRA, which [`CORPUS`] renders through.
///
/// `QWEN_LIGHTNING_SNAPSHOT` wins when set — the same precedence, and for the same reason, as
/// `lightning_render_real_weights`: `ensure_model_snapshot.py` verifies ONE revision, while the
/// `MLX_GEN_MODELS_ROOT` fallback only knows which revisions are cached, not which one was verified.
/// Without the override a lane can therefore verify the pin and then load a different revision
/// sitting beside it (sc-17284). The fallback is a local-dev convenience; CI always sets the
/// variable.
///
/// Panics rather than skipping on a missing variable. A producer that returns early on an unset
/// variable reports `1 passed` in 0.00 s and gates nothing — the silent-skip shape sc-17250 found
/// and sc-17284 removed from `perf.rs`.
fn lightning_lora() -> PathBuf {
    const FILE: &str = "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors";
    if let Ok(dir) = std::env::var("QWEN_LIGHTNING_SNAPSHOT") {
        let path = PathBuf::from(dir).join(FILE);
        assert!(
            path.exists(),
            "QWEN_LIGHTNING_SNAPSHOT is set but {} is missing",
            path.display()
        );
        return path;
    }
    let root = std::env::var("MLX_GEN_MODELS_ROOT").unwrap_or_else(|_| {
        panic!(
            "set QWEN_LIGHTNING_SNAPSHOT (or MLX_GEN_MODELS_ROOT) so the corpus can render through \
             the 8-step Lightning LoRA the committed fit documents; inference never self-fetches or \
             derives a cache location (epic 13657)"
        )
    });
    let snaps = PathBuf::from(root)
        .join("models--lightx2v--Qwen-Image-Lightning")
        .join("snapshots");
    // Sorted, not `read_dir` order: with several revisions cached, taking whichever entry the
    // filesystem yields first makes the corpus — and therefore every number this file prints —
    // depend on directory iteration order. Sorted is at least run-to-run stable. It is still not
    // this pin, which is why CI sets the variable rather than relying on this arm.
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .unwrap_or_else(|e| panic!("read {}: {e}; set QWEN_LIGHTNING_SNAPSHOT", snaps.display()))
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path().join(FILE))
        .filter(|path| path.exists())
        .collect();
    candidates.sort();
    candidates.into_iter().next().unwrap_or_else(|| {
        panic!(
            "{FILE} not cached under {}; set QWEN_LIGHTNING_SNAPSHOT",
            snaps.display()
        )
    })
}

/// One render's `(latent [n, 16], rgb [n, 3])` pair, host-side as f64 for the accumulation.
struct Pair {
    latent: Vec<f64>,
    rgb: Vec<f64>,
    n: usize,
}

/// Render one corpus entry and return its aligned latent/RGB samples.
fn render_pair(root: &Path, lora: &Path, prompt: &str, seed: u64) -> Pair {
    let mut tf = loader::load_transformer(root).unwrap();
    // The distilled schedule needs its distillation adapter: `lightning_sigmas` alone is only half
    // of the "8-step Lightning" configuration the committed fit records. 720 of the host's 840
    // per-block targets is this file's full designed surface, not a partial install (sc-17518).
    let report = apply_qwen_adapters(
        &mut tf,
        &[AdapterSpec::new(lora.to_path_buf(), 1.0, AdapterKind::Lora)],
    )
    .unwrap_or_else(|e| panic!("Lightning LoRA {} failed to apply: {e}", lora.display()));
    assert!(
        report.unmatched_paths.is_empty(),
        "Lightning LoRA left {} unmatched target(s): {:?}",
        report.unmatched_paths.len(),
        report.unmatched_paths
    );
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
    //
    // `project` then hands that straight to `matmul`, which is stride-aware, so it never needs what
    // the trailing reshape below does. This file does, because it reads the samples to the host:
    // `as_slice` exposes physical storage for a transpose view, and without flattening first the
    // accumulation pairs latent channel 0 with the red channel and scores R² ≈ 0 (sc-17515).
    let x = unpacked
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[c, lh * lw])
        .unwrap()
        .transpose_axes(&[1, 0])
        .unwrap()
        .reshape(&[lh * lw * c])
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
        .unwrap()
        // Same contiguity pass as `x` above, for the same reason.
        .reshape(&[lh * lw * 3])
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
#[ignore = "needs the Qwen/Qwen-Image snapshot (MLX_GEN_QWEN_SNAPSHOT); renders 2x1024^2"]
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

    let lora = lightning_lora();
    eprintln!("  Lightning LoRA: {}", lora.display());
    for (prompt, seed) in CORPUS {
        let pair = render_pair(&root, &lora, prompt, seed);
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

    // Score the SHIPPING projection unchanged on this corpus. This is the number the coefficient
    // deltas above cannot give you: the latent channels are correlated, so OLS can move a
    // coefficient a long way without moving a prediction, and |Δfactor| alone therefore cannot say
    // whether a preview got worse. Clamped, because `project_latents` clamps before the 8-bit round
    // — this scores what is actually drawn (sc-17515).
    let mut committed_ss_res = [0.0f64; 3];
    let mut abs_sum = 0.0f64;
    for pair in &pairs {
        for i in 0..pair.n {
            let lat = &pair.latent[i * c..(i + 1) * c];
            for k in 0..3 {
                let mut pred = COMMITTED_BIAS[k] as f64;
                for (j, &l) in lat.iter().enumerate() {
                    pred += l * COMMITTED_FACTORS[j][k] as f64;
                }
                let d = pair.rgb[i * 3 + k] - pred.clamp(0.0, 1.0);
                committed_ss_res[k] += d * d;
                abs_sum += d.abs() * 255.0;
            }
        }
    }
    let committed_r2 = 1.0
        - committed_ss_res.iter().sum::<f64>()
            / (0..3)
                .map(|k| {
                    let mean = sum_y[k] / total as f64;
                    sum_y2[k] - total as f64 * mean * mean
                })
                .sum::<f64>();
    let mean_abs = abs_sum / (total * 3) as f64;
    println!(
        "\ncommitted constants, scored unchanged on this corpus: R^2 = {committed_r2:.4}, \
         mean |ΔRGB| = {mean_abs:.2}/255"
    );

    assert!(
        r2_overall > R2_FLOOR,
        "linear latent->RGB fit explains only R^2 = {r2_overall:.4} (floor {R2_FLOOR}); the \
         approximation itself has stopped working for this VAE, which needs a different projection \
         rather than new constants"
    );

    // The drift gate the module docs asked for and nothing implemented. The assertion above only
    // asks whether SOME linear map fits; it passes just as happily after a VAE re-pin that leaves
    // the shipping constants describing a model that no longer exists. This one asks whether the
    // constants users actually get are still right, which is the question that reaches a preview.
    //
    // Same floor deliberately: a projection is either explaining most of the variance or it is not,
    // and a second threshold would just be a number to argue about. Measured 0.9450 on
    // `qwen-image-mlx@8080a417` bf16 (mean |ΔRGB| 12.58/255), against a 0.9633 in-sample refit — so
    // the incumbent gives up 2.6/255 of mean error to a best-case re-solve, which is why sc-17515
    // left it alone rather than re-baselining shipping constants onto a two-render sample.
    assert!(
        committed_r2 > R2_FLOOR,
        "the COMMITTED preview constants explain only R^2 = {committed_r2:.4} (floor {R2_FLOOR}) \
         on this snapshot — the VAE lineage has moved out from under `preview::RGB_FACTORS` and \
         every live preview is being projected through a stale fit. Paste the block printed above \
         into src/preview.rs"
    );
}
