//! sc-12381 — **the real-trunk VRAM reading for sc-12301**, on the community INT8-ConvRot Krea 2 DiT.
//!
//! sc-12301 fixed `QLinear::convrot_int8` building a `CublasLt` (and its eager 32 MiB workspace) **per
//! projection**, and measured the mechanism weights-free: 32.00 MiB/handle, extrapolating to ~7.00 GiB
//! across the ~224 projections `eight_bit_linear.rs` documents. That extrapolation is *a measured
//! per-projection cost × a documented count* — **not** a reading on a real trunk. This is the reading.
//!
//! # Measured (2026-07-16, exclusive sm_120, 1024²/8-step, seed 42)
//!
//! | | steady (resident) | overall-peak (load+denoise+decode) |
//! |---|---:|---:|
//! | pre-sc-12301 (per-projection handle) | 37.212 GB | 50.327 GB |
//! | post-sc-12301 (shared `Int8Context`) | **29.763 GB** | **42.878 GB** |
//! | delta | **7.449 GB** | **7.449 GB** |
//!
//! 7.449 GB = 7104 MiB = **222 × 32 MiB**, against the 223 predicted (224 projections − the one handle
//! now shared). The delta is *identical* in steady and peak — exactly what a constant resident buffer
//! must do: shift the whole curve without interacting with activation. That equality is the independent
//! check that the 7.449 GB really is duplicated workspace and nothing else.
//!
//! **The peak also found a live bug (sc-12425):** the SceneWorks manifest ships
//! `vramGbByTier.int8-convrot = 31.0` for a tier whose measured peak is **42.9 GB** — a gate that admits
//! 32/40 GB cards to a render they cannot fit. That figure was never measured (the manifest says
//! *"ESTIMATE ... pending measurement"*); this harness is how it gets re-measured.
//!
//! ```text
//! cargo run -p candle-gen-krea --example krea-convrot-vram --features cuda --release -- \
//!   <canonical_snapshot_dir> <convrot_dit.safetensors> [W] [H] [steps] [seed]
//! ```
//!
//! **Deliberately uses only public API** (`pipeline::load_components_convrot` + `pipeline::render` +
//! `testkit::VramProbe`), so the identical file runs against the pre-fix tree to produce the BEFORE.
//! That is the whole point: a before/after on the same trunk, same box, same run shape — rather than
//! one reading and a subtraction.
//!
//! # Two different numbers — do not confuse them
//!
//! * **`steady`** (after load, before denoise) — where the duplicated cuBLASLt workspace shows up. The
//!   int8 leg is built eagerly at construction (F-121 / sc-11208), so the sc-12301 delta is fully
//!   visible here. This is the sc-12301 story number.
//! * **`overall-peak`** (load + denoise + decode) — what the SceneWorks manifest's
//!   `vramGbByTier` actually means. Its own comment pins the semantic: *"1024²/8-step via the
//!   candle-gen nvidia-smi peak monitor: q4 packed **overall-peak** 26.4 GB"*. A load-only figure is
//!   NOT comparable to `q4: 26.4` and must never be written into that gate — it would under-gate the
//!   tier and OOM at generate time, the same failure mode as the unmeasured `int8-convrot: 31.0`
//!   estimate this run exists to replace.
//!
//! Defaults are 1024²/8-step precisely so `overall-peak` is directly comparable to the q4 figure.

use std::path::PathBuf;
use std::time::Instant;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{GenerationRequest, Progress};
use candle_gen::testkit::VramProbe;
use candle_gen_krea::pipeline;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let root = PathBuf::from(a.get(1).expect("arg1: canonical Krea 2 snapshot dir"));
    let convrot = PathBuf::from(a.get(2).expect("arg2: convrot DiT .safetensors"));
    // 1024²/8-step by default: the exact shape the manifest's q4 overall-peak (26.4 GB) was measured
    // at, so this tier's number is comparable to its siblings rather than to itself.
    let width: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let height: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let steps: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(42);

    let device = Device::new_cuda(0).expect("cuda:0");

    // Baseline on the physical GPU candle's logical cuda:0 actually renders on (sc-12107): deriving it
    // from CUDA_VISIBLE_DEVICES is what stops a 2-GPU box sampling the idle card while loading the busy
    // one. `assert_idle` refuses to publish a peak off a contended GPU.
    let mut probe = VramProbe::start_rendered().assert_idle(2.0);

    eprintln!("[sc-12381] loading INT8-ConvRot trunk...");
    let t0 = Instant::now();
    let phase = probe.phase();
    let comps =
        pipeline::load_components_convrot(&root, &convrot, &device).expect("load_components_convrot");
    probe.end_load(phase);
    let load_s = t0.elapsed().as_secs_f64();

    // The render: this is what turns `steady` into a manifest-comparable `overall-peak`.
    let req = GenerationRequest {
        prompt: "a photorealistic red apple on a wooden table, studio lighting".into(),
        width,
        height,
        count: 1,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    eprintln!("[sc-12381] rendering {width}x{height} / {steps} steps / seed {seed}...");
    let t1 = Instant::now();
    let gen = probe.phase();
    let mut noop = |_p: Progress| {};
    let imgs = pipeline::render(&comps, &req, &device, &mut noop).expect("render");
    probe.end_gen(gen);
    let render_s = t1.elapsed().as_secs_f64();
    assert_eq!(imgs.len(), 1, "one image expected");

    let report = probe.report().assert_trustworthy(2.0);
    eprintln!("[sc-12381] load {load_s:.1} s | render {render_s:.1} s");
    eprintln!("[sc-12381] {report}");
    // STEADY = the sc-12301 workspace number. PEAK = the manifest `vramGbByTier` number.
    eprintln!(
        "[sc-12381] STEADY_GB={:.3} PEAK_GB={:.3} LOAD_PEAK_GB={:.3} BASELINE_GB={:.3}",
        report.steady_gb, report.peak_gb, report.load_peak_gb, report.baseline_gb
    );

    // Keep the trunk alive until after the report is taken: dropping it first would free the very
    // workspace this example exists to measure.
    drop(comps);
}
