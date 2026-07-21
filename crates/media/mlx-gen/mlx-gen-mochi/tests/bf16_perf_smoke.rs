//! Real-weight **precision perf/liveness** smoke for the Mochi 1 AsymmDiT (sc-12629). `#[ignore]`d —
//! run by hand on an Apple-Silicon Mac with the `SceneWorks/mochi-1-mlx` turnkey cached.
//!
//! It loads the selected tier (`MOCHI_TIER`, default `bf16`) twice — once at [`Precision::Fp32`] (the
//! old hardcoded compute path) and once at [`Precision::Bf16`] (the production default this story
//! wires) — runs an identical seeded render at each, and prints the per-step denoise wall time from the
//! `Progress::Step` callback (which isolates the DiT forward from load / text-encode / VAE-decode).
//! Both must render a live, non-degenerate clip; the bf16 run must not be slower. This is the on-device
//! evidence behind the diagnosis that the DiT was needlessly running in f32.
//!
//! `precision` is orthogonal to the tier: it sets the **activation / dense-tensor** compute dtype, not
//! the weight quantization. So on a `q4`/`q8` tier this measures 4-bit-weights × f32-activations vs
//! 4-bit-weights × bf16-activations — the quantized weights are identical in both runs.
//!
//! ```text
//! hf download SceneWorks/mochi-1-mlx --include 'q4/*' 'bf16/*' 'text_encoder/*' 'tokenizer/*' 'vae/*'
//! MOCHI_TIER=q4 cargo test -p mlx-gen-mochi --release mochi_bf16_vs_f32_perf -- --ignored --nocapture
//! ```
//! Overrides: `MOCHI_TIER` (bf16/q4/q8, default bf16), `MOCHI_W`/`MOCHI_H` (default 848×480),
//! `MOCHI_FRAMES` (default 13 = 6·2+1), `MOCHI_STEPS` (default 8), `MOCHI_MODEL_DIR` (else the cached
//! turnkey snapshot).

use std::path::{Path, PathBuf};
use std::time::Instant;

use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, Precision, Progress, WeightsSource,
};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Whether `dir` is a Mochi turnkey root — at least one tier's `transformer/` is present.
fn is_root(dir: &Path) -> bool {
    ["bf16", "q4", "q8"]
        .iter()
        .any(|t| dir.join(t).join("transformer").is_dir())
}

/// The cached `SceneWorks/mochi-1-mlx` root (tier subdirs + shared `text_encoder`/`tokenizer`/`vae`).
fn model_root() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MOCHI_MODEL_DIR") {
        let root = PathBuf::from(dir.trim());
        if is_root(&root) {
            return Some(root);
        }
    }
    let home = std::env::var("MLX_GEN_MODELS_ROOT").ok()?;
    let snapshots = PathBuf::from(home).join("models--SceneWorks--mochi-1-mlx/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|dir| is_root(dir))
}

/// Mean absolute per-pixel delta between two frames (0–255 scale) — a liveness probe.
fn mean_abs_delta(a: &Image, b: &Image) -> f64 {
    assert_eq!(a.pixels.len(), b.pixels.len());
    let sum: f64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(x, y)| (*x as f64 - *y as f64).abs())
        .sum();
    sum / a.pixels.len() as f64
}

/// Per-pixel std of one frame — a degenerate-decode probe (flat/black → ~0).
fn image_std(f: &Image) -> f64 {
    let n = f.pixels.len() as f64;
    let mean: f64 = f.pixels.iter().map(|p| *p as f64).sum::<f64>() / n;
    (f.pixels
        .iter()
        .map(|p| (*p as f64 - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// Load `tier` (`bf16`/`q4`/`q8`) at `precision`, run one seeded render, and return (per-step times,
/// frames). The tier dir's `split_model.json` drives the weight quantization; `precision` is the
/// orthogonal **activation / dense-tensor** compute dtype — so a Q4 tier at `Bf16` is 4-bit weights ×
/// bf16 activations (the quantized weights are unchanged either way).
fn render(
    root: &Path,
    tier: &str,
    precision: Precision,
    w: u32,
    h: u32,
    frames: u32,
    steps: u32,
) -> (Vec<f64>, Vec<Image>) {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root.join(tier)));
    spec.precision = precision;
    let generator = mlx_gen_mochi::load(&spec).expect("load mochi tier");

    let req = GenerationRequest {
        prompt: "a calico kitten padding through tall sunlit grass, shallow depth of field".into(),
        width: w,
        height: h,
        count: 1,
        frames: Some(frames),
        fps: Some(30),
        seed: Some(42),
        steps: Some(steps),
        guidance: Some(4.5),
        ..Default::default()
    };

    let mut step_times = Vec::new();
    let mut last = Instant::now();
    let output = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { .. } = p {
                let now = Instant::now();
                step_times.push((now - last).as_secs_f64());
                last = now;
            }
        })
        .expect("generate");
    let GenerationOutput::Video { frames, .. } = output else {
        panic!("expected video output");
    };
    (step_times, frames)
}

#[test]
#[ignore = "real-weight MLX perf smoke; needs a SceneWorks/mochi-1-mlx tier cached + a Metal Mac"]
fn mochi_bf16_vs_f32_perf() {
    let tier = env_or("MOCHI_TIER", "bf16"); // bf16 | q4 | q8 — the weight quantization tier
    let w: u32 = env_or("MOCHI_W", "848").parse().unwrap();
    let h: u32 = env_or("MOCHI_H", "480").parse().unwrap();
    let frames: u32 = env_or("MOCHI_FRAMES", "13").parse().unwrap();
    let steps: u32 = env_or("MOCHI_STEPS", "8").parse().unwrap();
    assert_eq!(frames % 6, 1, "frames must be 6k+1");

    let Some(root) = model_root() else {
        panic!("no Mochi turnkey cached — hf download SceneWorks/mochi-1-mlx --include 'bf16/*' 'text_encoder/*' 'tokenizer/*' 'vae/*'");
    };
    println!(
        "[perf] root {}  tier {tier}  {w}x{h} x{frames}f @ {steps} steps",
        root.display()
    );

    // The first Progress::Step interval also carries any lazy first-step warmup, so report the median
    // step time (robust to that outlier) as the per-step DiT cost at each precision.
    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };

    let summarize = |label: &str, precision: Precision| -> f64 {
        let total = Instant::now();
        let (step_times, frames_out) = render(&root, &tier, precision, w, h, frames, steps);
        let wall = total.elapsed().as_secs_f64();
        let med = median(step_times.clone());
        let motion: Vec<f64> = frames_out
            .windows(2)
            .map(|p| mean_abs_delta(&p[0], &p[1]))
            .collect();
        let span = mean_abs_delta(&frames_out[0], &frames_out[frames_out.len() - 1]);
        // Print the numbers first — the perf reading is the point; liveness is a coarse sanity gate.
        println!(
            "[perf] {label:>4}: total {wall:6.1}s | median step {med:6.3}s | steps {:?} | span {span:.2}",
            step_times.iter().map(|t| format!("{t:.2}")).collect::<Vec<_>>(),
        );
        // Coarse liveness: every frame decodes with real variance (not NaN/flat/black) and the clip
        // moves overall. A lenient span floor — this is a perf smoke, not the motion-quality bar the
        // worker smoke owns (a few-step render legitimately has near-static adjacent pairs).
        for (i, f) in frames_out.iter().enumerate() {
            assert!(
                image_std(f) > 2.0,
                "{label} frame {i} degenerate (std {:.2})",
                image_std(f)
            );
        }
        assert!(
            span > 1.5,
            "{label} clip barely moves (span {span:.2}): {motion:?}"
        );
        med
    };

    // f32 first, then bf16 (each generator drops before the next load, so peaks don't stack).
    let f32_step = summarize("f32", Precision::Fp32);
    let bf16_step = summarize("bf16", Precision::Bf16);

    println!(
        "[perf] per-step DiT speedup (f32/bf16): {:.2}x  ({:.3}s -> {:.3}s)",
        f32_step / bf16_step,
        f32_step,
        bf16_step
    );
    assert!(
        bf16_step <= f32_step * 1.05,
        "bf16 per-step {bf16_step:.3}s must not be slower than f32 {f32_step:.3}s"
    );
}
