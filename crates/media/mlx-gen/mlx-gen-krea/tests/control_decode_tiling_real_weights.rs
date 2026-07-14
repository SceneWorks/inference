//! sc-11747 (epic 8459): **e2e budget-fit + decode-parity** for the Krea 2 pose-control tiled Qwen-VAE
//! decode. This story's acceptance (folded in from sc-11865): prove that once the decode-tiling lever
//! engages, a real `krea_2_turbo_control` q4/1024² render actually FITS a ~24 GiB budget, and that the
//! tiled decode reconstructs the untiled image (no blend seams → coherent + pose-locked).
//!
//! **No 32 GB Mac needed** — the budget is EMULATED on any large-memory Metal Mac by lowering the MLX
//! memory limit so [`mlx_gen::memory::safe_budget_gib`] reports ~24 GiB (epic 7819's user-GPU-cap knob,
//! `mlx_rs::memory::set_memory_limit`). The limit is a *soft* backpressure point (allocations above it
//! still succeed, MLX just evicts cache to stay near it), so the render never hard-OOMs; the measured
//! `get_peak_memory` is the true concurrent high-water, which is exactly what a real 24 GiB device would
//! OOM on. The original limit is restored on exit.
//!
//! `#[ignore]`d — needs the real snapshots (env overrides, else the HF cache), same as
//! `control_memory_calibration_real_weights.rs`, whose rig (base_dir/overlay/fixed_image + the
//! Progress-callback peak split) this reuses:
//!   - base — `SceneWorks/krea-2-turbo-mlx/bf16`, env `KREA_CONTROL_DIR` (quantized to q4 at load).
//!   - overlay — `SceneWorks/krea2-pose-controlnet-beta/control_step5000.safetensors`, env
//!     `KREA_CONTROL_OVERLAY`.
//!
//! Run:
//! ```text
//! cargo test -p mlx-gen-krea --release --test control_decode_tiling_real_weights -- \
//!   --ignored --nocapture
//! ```

use mlx_gen_krea::memory::plan_control_decode_tiling;
use mlx_gen_krea::{load_vae, Krea2Config};

use mlx_gen::tiling::{SpatialTiling, TilingConfig};
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{
    clear_cache, get_memory_limit, get_peak_memory, reset_peak_memory, set_memory_limit,
};
use std::cell::Cell;
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The S0 recipe branch-block count (the overlay ships N=7).
const BRANCH_BLOCKS: usize = 7;

/// The emulated device budget: a 32 GB Mac's ~24 GiB usable unified memory — the story's target point.
const EMULATED_SAFE_GIB: f64 = 24.0;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

fn hf_snapshot(model: &str) -> PathBuf {
    let snaps = home()
        .join(".cache/huggingface/hub")
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {model}: {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn base_dir() -> PathBuf {
    std::env::var("KREA_CONTROL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("models--SceneWorks--krea-2-turbo-mlx").join("bf16"))
}

fn overlay() -> PathBuf {
    std::env::var("KREA_CONTROL_OVERLAY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            hf_snapshot("models--SceneWorks--krea2-pose-controlnet-beta")
                .join("control_step5000.safetensors")
        })
}

/// A deterministic RGB pose stand-in (shape-driven measurement → content is irrelevant).
fn fixed_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 255 / w.max(1)) as u8);
            pixels.push((y * 255 / h.max(1)) as u8);
            pixels.push(((x + y) * 127 / (w + h).max(1)) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// RAII guard: lower the MLX memory limit to emulate a `safe_gib` budget, restore it on drop so the
/// change never leaks into sibling tests. `safe_budget_gib = min(limit·0.85, maxBufferLength)`, so we set
/// `limit = safe / 0.85` (maxBufferLength on a large Mac is far higher, so it does not clamp).
struct BudgetGuard {
    previous: usize,
}
impl BudgetGuard {
    fn emulate(safe_gib: f64) -> Self {
        let previous = get_memory_limit();
        let limit_bytes = ((safe_gib / mlx_gen::memory::SAFE_FRAC) * GIB) as usize;
        set_memory_limit(limit_bytes);
        Self { previous }
    }
}
impl Drop for BudgetGuard {
    fn drop(&mut self) {
        set_memory_limit(self.previous);
    }
}

/// **(a) + (b)** — at the emulated ~24 GiB budget, a real q4/1024² pose-control render must engage decode
/// tiling and keep its measured concurrent peak under the budget. Driven under `Sequential` (text phase
/// dropped before the heavy phase) so the peaks are ex-text, matching the estimator's `*_ex_text_gib`
/// semantics and the calibration rig.
#[test]
#[ignore = "needs SceneWorks/krea-2-turbo-mlx bf16 + the pose overlay (KREA_CONTROL_DIR / KREA_CONTROL_OVERLAY)"]
fn q4_1024_control_render_fits_emulated_24gib_budget() {
    let _budget = BudgetGuard::emulate(EMULATED_SAFE_GIB);
    let safe = mlx_gen::memory::safe_budget_gib();
    assert!(
        (EMULATED_SAFE_GIB - 1.0..=EMULATED_SAFE_GIB + 1.0).contains(&safe),
        "emulated safe budget {safe:.2} GiB should be ~{EMULATED_SAFE_GIB} (raise the MLX limit if a \
         per-buffer cap clamped it)"
    );

    // (a) The gate engages at this budget for a q4/1024² decode (Sequential ⇒ no co-resident text).
    let cfg = Krea2Config::turbo();
    let tiling = plan_control_decode_tiling(
        safe,
        &cfg,
        BRANCH_BLOCKS,
        Some(Quant::Q4),
        None,
        1024,
        1024,
        false,
    )
    .expect("decode-tiling gate must not error at ~24 GiB");
    assert!(
        tiling.is_some(),
        "decode tiling must engage for q4/1024² at a ~24 GiB budget"
    );

    // Drive a real render under the emulated budget and measure the concurrent peak.
    let spec = LoadSpec::new(WeightsSource::Dir(base_dir()))
        .with_control(WeightsSource::File(overlay()))
        .with_offload_policy(OffloadPolicy::Sequential)
        .with_quant(Quant::Q4);
    let model = mlx_gen_krea::provider_registry()
        .expect("build explicit Krea provider registry")
        .load("krea_2_turbo_control", &spec)
        .expect("load krea_2_turbo_control (q4)");

    let req = GenerationRequest {
        prompt: "a person standing in a studio, photograph".into(),
        width: 1024,
        height: 1024,
        seed: Some(1234),
        steps: Some(8),
        conditioning: vec![Conditioning::Control {
            image: fixed_image(512, 512),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        ..Default::default()
    };

    // Whole-render peak: reset before generate, read after. Under Sequential every stage is ex-text, so
    // the high-water is max(load, denoise, tiled-decode) — the number a real 24 GiB device OOMs on.
    let denoise_peak = Cell::new(0usize);
    reset_peak_memory();
    let out = {
        let denoise_peak = &denoise_peak;
        model
            .generate(&req, &mut |p: Progress| {
                if let Progress::Decoding = p {
                    denoise_peak.set(get_peak_memory());
                }
            })
            .expect("generate")
    };
    let peak_gib = get_peak_memory() as f64 / GIB;
    let denoise_gib = denoise_peak.get() as f64 / GIB;

    let images = match out {
        GenerationOutput::Images(v) => v,
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!(images.len(), 1);
    // (c-lite) the render produced a real, non-degenerate image (not all-black / NaN-collapsed).
    let img = &images[0];
    assert_eq!((img.width, img.height), (1024, 1024));
    let (min, max) = img
        .pixels
        .iter()
        .fold((255u8, 0u8), |(lo, hi), &p| (lo.min(p), hi.max(p)));
    assert!(
        max > min,
        "decoded image is a constant field ({min}={max}) — decode collapsed"
    );

    drop(model);
    clear_cache();

    println!(
        "\n=== sc-11747 budget-fit: q4/1024² control render @ emulated {EMULATED_SAFE_GIB} GiB ===\n\
         denoise peak {denoise_gib:.2} GiB | whole-render peak {peak_gib:.2} GiB (budget {safe:.2})"
    );

    // (b) The measured peak stays under the emulated budget — the whole point of the decode-tiling lever.
    assert!(
        peak_gib <= safe,
        "render peak {peak_gib:.2} GiB exceeded the emulated {safe:.2} GiB budget — tiling did not \
         bound the decode spike"
    );
}

/// **(c)** — decode PARITY: the tiled Qwen-VAE decode must reconstruct the single-pass (untiled) decode
/// within a tight tolerance on the REAL VAE weights, proving the trapezoidal blend is seam-artifact-free
/// (the sc-4998/sc-5690 precedent). Runs at the production overlap (64 px) so it validates the exact
/// config the budget gate emits. VAE-only (no DiT) so it is cheap.
#[test]
#[ignore = "needs the Krea snapshot's vae/ (KREA_CONTROL_DIR / SceneWorks/krea-2-turbo-mlx bf16)"]
fn tiled_decode_matches_untiled_on_real_vae() {
    let vae = load_vae(base_dir()).expect("load Qwen-Image VAE from the Krea snapshot vae/");

    // A realistic latent: VAE-encode the fixed 1024² image → normalized latent [1,16,1,128,128].
    let image_nchw =
        mlx_gen::img2img::preprocess_init_image(&fixed_image(1024, 1024), 1024, 1024).unwrap();
    let latent = vae.encode(&image_nchw).expect("encode → latent");

    let untiled = vae.decode(&latent).expect("single-pass decode");
    // Force tiling: 512 px tiles (64 latent) over the 1024² (128 latent) image, 64 px overlap.
    let cfg = TilingConfig {
        spatial: Some(SpatialTiling {
            tile_px: 512,
            overlap_px: 64,
        }),
        temporal: None,
    };
    let tiled = vae.decode_tiled(&latent, &cfg, None).expect("tiled decode");

    untiled.eval().unwrap();
    tiled.eval().unwrap();
    assert_eq!(
        untiled.shape(),
        tiled.shape(),
        "tiled decode changed the output shape"
    );

    // `decode()` returns a transposed (non-contiguous) NCTHW view, while `decode_tiled` ends with an
    // explicit `contiguous`. `as_slice` reads the *physical* buffer, so the two must be materialized in
    // the SAME logical order before an elementwise compare — a `reshape([-1])` round-trip forces that
    // (matching `vae_real_weights.rs`). Without it the strided read is scrambled and the compare is bogus.
    let untiled = untiled.reshape(&[-1]).unwrap();
    let tiled = tiled.reshape(&[-1]).unwrap();
    let (u, t) = (untiled.as_slice::<f32>(), tiled.as_slice::<f32>());
    let (mut max_abs, mut sum_abs) = (0f32, 0f64);
    for (a, b) in u.iter().zip(t) {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    let mean_abs = sum_abs / u.len() as f64;
    println!(
        "\n=== sc-11747 decode parity (real VAE, 512px/64 overlap tiles) ===\n\
         max|Δ| {max_abs:.4e} | mean|Δ| {mean_abs:.4e}  (decode output ~[-1,1])"
    );

    // The decode output is ~[-1,1] and is later clipped + quantized to u8; a few 1e-2 of residual at the
    // tile seams is < 1 gray level and imperceptible. A blown blend (seam artifact) would be O(0.1+).
    assert!(
        max_abs < 3.0e-2,
        "tiled decode diverges from untiled at the seams (max|Δ|={max_abs:.3e}) — blend not seam-free"
    );
    assert!(
        mean_abs < 3.0e-3,
        "tiled decode mean drift {mean_abs:.3e} too high — blend/normalization off"
    );

    // Sanity: the untiled decode is itself non-degenerate (guards a both-NaN false pass).
    assert!(
        u.iter().any(|v| v.is_finite() && *v != u[0]),
        "untiled decode is constant/NaN — parity comparison would be vacuous"
    );
}
