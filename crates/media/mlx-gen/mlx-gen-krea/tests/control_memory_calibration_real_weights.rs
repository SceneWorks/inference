//! sc-11847 (epic 8459): **on-Metal calibration** of the two Krea pose-control memory-adaptation cost
//! coefficients in [`mlx_gen_krea::memory`] — `DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN` and
//! `DECODE_SPIKE_BYTES_PER_PIXEL`. sc-11750 seeded them from the candle #480 CUDA profile; this test
//! MEASURES the real MLX/Metal concurrent peaks of a `krea_2_turbo_control` render and asserts the
//! estimator still **over-predicts** the measured peak at every tested (resolution, base tier) — the
//! Wan/PiD never-under-shoot guard (sc-4998, sc-10087).
//!
//! `#[ignore]`d — needs the real snapshots (env overrides, else the HF cache), exactly like
//! `sequential_residency_real_weights.rs`:
//!   - base — `SceneWorks/krea-2-turbo-mlx/<tier>` (`bf16`/`q4`), env `KREA_CONTROL_DIR` (a bf16 dir;
//!     the q4 case quantizes it at load via `with_quant`, matching the sc-11727 packed-base path).
//!   - overlay — `SceneWorks/krea2-pose-controlnet-beta/control_step5000.safetensors`, env
//!     `KREA_CONTROL_OVERLAY`.
//!
//! Run:
//! ```text
//! cargo test -p mlx-gen-krea --release --test control_memory_calibration_real_weights -- \
//!   --ignored --nocapture
//! ```
//!
//! ## How the two stage peaks are isolated (no production-code change)
//!
//! The render is driven under [`OffloadPolicy::Sequential`], so the Qwen3-VL text/vision phase is
//! ENCODED then DROPPED before the heavy phase (DiT + pose branch + VAE) materializes. The measured
//! active-memory high-water during the heavy phase is therefore **ex-text**, matching the estimator's
//! `*_ex_text_gib` semantics (the policy carries the text footprint separately as the residency lever).
//!
//! Denoise and decode are split with the existing [`Progress`] callback (`sampler::step_gate` emits
//! `Step` at the START of each step, BEFORE its forward; `render_control_from` emits `Decoding` at the
//! exact `run_flow_sampler → decode_latents` seam):
//!   * at `Step { current: 1, .. }` — `reset_peak_memory()` to open the denoise window (excludes the
//!     pre-sampler pose VAE-encode + prep transient);
//!   * at `Decoding` — read `get_peak_memory()` = **denoise stage peak**, then `reset_peak_memory()`;
//!   * after `generate()` returns — read `get_peak_memory()` = **decode stage peak**.
//!
//! `reset_peak_memory` rebases the high-water to current active, so each read is the true active
//! high-water of its window — the peak IS the resident-weight + stage-transient concurrent footprint,
//! which is exactly what an OOM cares about. (Note: `get_active_memory()` BETWEEN forwards reads ~0 on
//! MLX/Metal — safetensors weights are mmap'd into unified memory and are not counted as active
//! allocator bytes until a forward streams them through — so the resident floor is only observable via
//! the peak, e.g. the 512² denoise peak where the per-step activation is smallest. This is why the
//! `RESIDENT_OVERHEAD_GIB` term in `memory.rs` is fit to the low-resolution peak, not to a snapshot.)
//!
//! ## Calibration (documented in `memory.rs`)
//!
//! With ≥2 resolutions the physical coefficient is the SLOPE `Δpeak / Δshape`, which cancels the
//! resident constant — independent of any resident estimate:
//!   * `DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN ≈ (denoise_peak_hi − denoise_peak_lo)·GiB /
//!     ((tokens_hi − tokens_lo)·hidden)`
//!   * `DECODE_SPIKE_BYTES_PER_PIXEL ≈ (decode_peak_hi − decode_peak_lo)·GiB / (px_hi − px_lo)`
//!
//! The test prints these backed-out slopes (per tier) alongside the absolute measured/estimator peaks
//! so a re-fit is a copy of the printed numbers into `memory.rs` (rounded UP), after which this test's
//! over-predict assertions hold.

use mlx_gen_krea::memory::{control_denoise_peak_ex_text_gib, qwen_vae_decode_peak_ex_text_gib};
use mlx_gen_krea::Krea2Config;

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy,
    Progress, Quant, WeightsSource,
};
use mlx_rs::memory::{clear_cache, get_peak_memory, reset_peak_memory};
use std::cell::Cell;
use std::path::PathBuf;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// The S0 recipe branch-block count (the overlay ships N=7).
const BRANCH_BLOCKS: usize = 7;

/// Upper bound on `estimator / measured` at every tested point — the estimator must over-predict, but
/// not wastefully (it would tile/adapt too soon). The story's documented target is ≤ ~1.3×; the measured
/// re-fit (sc-11847) lands every tested point in ~[1.03×, 1.16×], so 1.3× also guards against drift.
const MAX_OVERPREDICT: f64 = 1.3;

/// First snapshot dir under an HF-cache `models--…` entry.
fn hf_snapshot(model: &str) -> PathBuf {
    let snaps = std::path::PathBuf::from(std::env::var("MLX_GEN_MODELS_ROOT").expect("set MLX_GEN_MODELS_ROOT to the explicit models root (holds models--*/snapshots); inference never self-fetches or derives a cache location (epic 13657)"))
        .join(model)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {model}: {}", snaps.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// The dense bf16 base dir (`KREA_CONTROL_DIR` → the HF-cache turbo turnkey's `bf16` subdir).
fn base_dir() -> PathBuf {
    std::env::var("KREA_CONTROL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_snapshot("models--SceneWorks--krea-2-turbo-mlx").join("bf16"))
}

/// The pose overlay checkpoint (`KREA_CONTROL_OVERLAY` → the beta HF snapshot).
fn overlay() -> PathBuf {
    std::env::var("KREA_CONTROL_OVERLAY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            hf_snapshot("models--SceneWorks--krea2-pose-controlnet-beta")
                .join("control_step5000.safetensors")
        })
}

/// A deterministic RGB pose stand-in. Memory peaks are shape-driven, so the pixel content is
/// irrelevant to the measurement — a reproducible synthetic image suffices (same convention as
/// `sequential_residency_real_weights.rs::fixed_image`).
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

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Comma-separated resolution list (`KREA_CAL_SIZES`, default `512,768,1024`). The story asks for ≥2
/// (768²/1024²); 512² is included because it is the tightest point — the fixed resident floor dominates
/// there, so it is where the estimator is most at risk of under-shooting (and where the sc-11847 re-fit
/// found the original CUDA prior DID under-shoot).
fn sizes() -> Vec<u32> {
    std::env::var("KREA_CAL_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<u32>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![512, 768, 1024])
}

/// Denoise image-token count for a `size × size` render — the latent is `[16, H/8, W/8]`, patchified
/// 2×2 → `(size/16)²` tokens. Mirrors `memory::denoise_tokens`.
fn tokens(size: u32) -> f64 {
    (size as f64 / 16.0).floor() * (size as f64 / 16.0).floor()
}

/// One measured render: the ex-text denoise stage peak and decode stage peak (GiB).
#[derive(Clone, Copy, Debug)]
struct Measured {
    size: u32,
    denoise_gib: f64,
    decode_gib: f64,
}

/// Load `krea_2_turbo_control` (base + overlay) at `tier` under `Sequential` and measure the two stage
/// peaks for a `size × size` render.
fn measure(size: u32, tier: Option<Quant>) -> Measured {
    let mut spec = LoadSpec::new(WeightsSource::Dir(base_dir()))
        .with_control(WeightsSource::File(overlay()))
        .with_offload_policy(OffloadPolicy::Sequential);
    if let Some(q) = tier {
        spec = spec.with_quant(q);
    }
    let model = mlx_gen_krea::provider_registry()
        .expect("build explicit Krea provider registry")
        .load("krea_2_turbo_control", &spec)
        .unwrap_or_else(|e| panic!("load krea_2_turbo_control ({tier:?}): {e}"));

    let req = GenerationRequest {
        prompt: "a person standing in a studio, photograph".into(),
        width: size,
        height: size,
        seed: Some(1234),
        steps: Some(env_u32("KREA_CAL_STEPS", 8)),
        conditioning: vec![Conditioning::Control {
            image: fixed_image(512, 512),
            kind: ControlKind::Pose,
            scale: Some(0.6),
        }],
        ..Default::default()
    };

    // `Cell` interior mutability so the `FnMut` progress callback can record without a mutable borrow
    // that would outlive `generate`. Reset the peak at Step{1} (which `step_gate` fires BEFORE the first
    // forward) to open the denoise window — this excludes the pre-sampler pose VAE-encode + prep
    // transient so the denoise peak is the denoise loop alone.
    let denoise_peak = Cell::new(0usize);

    reset_peak_memory();
    let out = {
        let denoise_peak = &denoise_peak;
        model
            .generate(&req, &mut |p: Progress| match p {
                Progress::Step { current: 1, .. } => {
                    reset_peak_memory();
                }
                // The run_flow_sampler → decode_latents seam: the window since Step{1} was the denoise.
                Progress::Decoding => {
                    denoise_peak.set(get_peak_memory());
                    reset_peak_memory();
                }
                _ => {}
            })
            .expect("generate")
    };
    // The window since `Decoding` was the VAE decode.
    let decode_peak = get_peak_memory();

    match out {
        GenerationOutput::Images(v) => assert_eq!(v.len(), 1, "expected a single image"),
        other => panic!("expected Images, got {other:?}"),
    }
    drop(model);
    clear_cache();

    let m = Measured {
        size,
        denoise_gib: denoise_peak.get() as f64 / GIB,
        decode_gib: decode_peak as f64 / GIB,
    };
    assert!(
        m.denoise_gib > 0.0 && m.decode_gib > 0.0,
        "a stage peak was never recorded (progress callback shape changed?): {m:?}"
    );
    m
}

/// Run the full (resolution × tier) sweep for one base tier, print the measured-vs-estimator table +
/// the backed-out slopes, and assert the estimator over-predicts within [`MAX_OVERPREDICT`] at each
/// point.
fn calibrate_tier(label: &str, tier: Option<Quant>) {
    let cfg = Krea2Config::turbo();
    let sizes = sizes();
    let ms: Vec<Measured> = sizes.iter().map(|&s| measure(s, tier)).collect();

    println!("\n=== krea_2_turbo_control memory calibration — base tier {label} ===");
    println!(
        "{:>6} | {:>10} {:>10} | {:>10} {:>10} | {:>7} {:>7}",
        "size", "denoise", "decode", "est_dn", "est_dc", "dn_x", "dc_x"
    );
    for m in &ms {
        let est_dn =
            control_denoise_peak_ex_text_gib(&cfg, BRANCH_BLOCKS, tier, None, m.size, m.size);
        let est_dc =
            qwen_vae_decode_peak_ex_text_gib(&cfg, BRANCH_BLOCKS, tier, None, m.size, m.size);
        println!(
            "{:>6} | {:>10.3} {:>10.3} | {:>10.3} {:>10.3} | {:>7.3} {:>7.3}",
            m.size,
            m.denoise_gib,
            m.decode_gib,
            est_dn,
            est_dc,
            est_dn / m.denoise_gib,
            est_dc / m.decode_gib,
        );
    }

    // Backed-out physical slopes (needs ≥2 resolutions). Print the coefficient each stage implies.
    if ms.len() >= 2 {
        let lo = ms.first().unwrap();
        let hi = ms.last().unwrap();
        let d_tok = (tokens(hi.size) - tokens(lo.size)) * cfg.hidden_size as f64;
        let d_px = (hi.size as f64 * hi.size as f64) - (lo.size as f64 * lo.size as f64);
        if d_tok > 0.0 {
            let act_coeff = (hi.denoise_gib - lo.denoise_gib) * GIB / d_tok;
            println!(
                "  backed-out DENOISE_ACT_BYTES_PER_TOKEN_HIDDEN ≈ {act_coeff:.1} B/elem  \
                 (from {}²→{}²)",
                lo.size, hi.size
            );
        }
        if d_px > 0.0 {
            let spike_coeff = (hi.decode_gib - lo.decode_gib) * GIB / d_px;
            println!(
                "  backed-out DECODE_SPIKE_BYTES_PER_PIXEL     ≈ {spike_coeff:.1} B/px    \
                 (from {}²→{}²)",
                lo.size, hi.size
            );
        }
    }

    // The guard: never under-shoot, and not wastefully over. Resident sanity: the estimator's resident
    // term must also cover the measured ex-text resident (else an untested resolution could under-shoot
    // even with a fitted slope).
    for m in &ms {
        let est_dn =
            control_denoise_peak_ex_text_gib(&cfg, BRANCH_BLOCKS, tier, None, m.size, m.size);
        let est_dc =
            qwen_vae_decode_peak_ex_text_gib(&cfg, BRANCH_BLOCKS, tier, None, m.size, m.size);
        assert!(
            est_dn >= m.denoise_gib,
            "{label} {}²: denoise estimate {est_dn:.3} UNDER-SHOOTS measured {:.3} GiB (OOM risk)",
            m.size,
            m.denoise_gib
        );
        assert!(
            est_dc >= m.decode_gib,
            "{label} {}²: decode estimate {est_dc:.3} UNDER-SHOOTS measured {:.3} GiB (OOM risk)",
            m.size,
            m.decode_gib
        );
        assert!(
            est_dn <= m.denoise_gib * MAX_OVERPREDICT,
            "{label} {}²: denoise estimate {est_dn:.3} over-predicts measured {:.3} GiB by \
             >{MAX_OVERPREDICT}× — coefficient too conservative",
            m.size,
            m.denoise_gib
        );
        assert!(
            est_dc <= m.decode_gib * MAX_OVERPREDICT,
            "{label} {}²: decode estimate {est_dc:.3} over-predicts measured {:.3} GiB by \
             >{MAX_OVERPREDICT}× — coefficient too conservative",
            m.size,
            m.decode_gib
        );
    }
}

#[test]
#[ignore = "needs SceneWorks/krea-2-turbo-mlx bf16 + the pose overlay (KREA_CONTROL_DIR / KREA_CONTROL_OVERLAY)"]
fn calibrate_bf16_base() {
    calibrate_tier("bf16", None);
}

#[test]
#[ignore = "needs SceneWorks/krea-2-turbo-mlx bf16 + the pose overlay; quantizes the base to q4 at load"]
fn calibrate_q4_base() {
    calibrate_tier("q4", Some(Quant::Q4));
}
