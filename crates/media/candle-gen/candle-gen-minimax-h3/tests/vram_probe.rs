//! **The measured `vramGbByTier` harness for `minimax_h3` on candle/CUDA (sc-17156).**
//!
//! The sibling of `candle-gen-wan/tests/vram_probe.rs`, and it follows the Wan 5B precedent
//! deliberately rather than inventing a second protocol. Each run prints one machine-parseable line
//! to scrape into SceneWorks' `config/manifests/builtin.models.jsonc`.
//!
//! ```sh
//! MINIMAX_H3_VRAM_DIR=<staged snapshot root> \
//! MINIMAX_H3_VRAM_TIER=bf16 \
//! CANDLE_GEN_OFFLOAD=sequential \
//!   cargo test -p candle-gen-minimax-h3 --features cuda --release minimax_h3_vram -- \
//!     --ignored --nocapture
//! ```
//!
//! # The five protocol rules, and why each one is not optional
//!
//! 1. **An idle card.** [`VramProbe::assert_trustworthy`] panics above a 1 GB baseline: a co-tenant
//!    process makes the high-water a fact about the box rather than about the model.
//! 2. **One tier per process.** cudarc's caching allocator never returns pages to the driver, so a
//!    second tier measured in the same process re-reports the first tier's high-water. There is one
//!    test per tier and each must be run alone.
//! 3. **The model's own shipped default geometry and step count.** 768x1344, the lattice floor of
//!    124 frames, [`DEFAULT_STEPS`]. A number measured at a geometry the model does not ship at is
//!    not the number the admission gate needs.
//! 4. **The nvidia-smi POOL high-water is what ships**, base-10 GB — not the lower driver-mempool
//!    `USED_MEM_HIGH`. Gating at the pool high-water needs no assumption that the pool packs down
//!    under memory pressure on a smaller card, and that unproven assumption is exactly what forced
//!    the Wan A14B to defer its q8/bf16 rows as `measured: false`. Both numbers are printed, so the
//!    scrape can record the concurrent-live figure as evidence without gating on it.
//! 5. **`CANDLE_GEN_OFFLOAD=sequential` is belt-and-braces here, not the mechanism.** This provider
//!    *forces* [`OffloadPolicy::Sequential`] (see `crate::model::OFFLOAD_POLICY`), which is what
//!    makes the measured number a property of the render rather than of the request. The env var is
//!    set anyway so the invocation matches the Wan runbook line for line.
//!
//! # Which phase owns the peak
//!
//! Unlike Wan — where the denoise bound — H3 has a **36-layer transformer video-VAE decoder** that
//! is a real candidate for the peak, so the probe splits the render at `Progress::Decoding` and
//! reports the two sides separately. The manifest comment must name the winner; a single overall
//! number cannot say which stage it describes, and the answer decides whether a smaller card is
//! helped by decode tiling (sc-18660) or by nothing at all.
//!
//! # Why there is no `q4` / `q8` test yet
//!
//! This lane advertises no `supported_quants`: the pre-quantized tiers are packed by the MLX lane's
//! `convert` and candle has no tier loader for them. [`minimax_h3_vram_bf16`] is therefore the only
//! honest measurement this crate can take today, and a `vramGbByTier` shipped with a `q4` row would
//! be a number no candle render can reach. The tier tests land with the tier loader.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress, WeightsSource,
};
use candle_gen::testkit::{cuda_mempool_used_high_bytes, reset_cuda_mempool_high_water, VramProbe};

use candle_gen_minimax_h3::model::{DEFAULT_STEPS, MODEL_ID};
use candle_gen_minimax_h3::pipeline::{CANVAS_SHORT_EDGE, SMALLEST_LEGAL_FRAMES};

/// Max idle-baseline VRAM (GB) tolerated before the sampled peak is considered contaminated.
const MAX_BASELINE_GB: f64 = 1.0;

/// The **logical** CUDA device candle renders on (`cuda:0`). The driver API respects
/// `CUDA_VISIBLE_DEVICES`, so logical 0 is the physical card candle uses.
const CANDLE_LOGICAL_DEVICE: i32 = 0;

/// A render whose middle frame is flatter than this is degenerate (black / uniform), which means the
/// engine failed silently and the peak describes a broken run.
const DEGENERATE_STD_FLOOR: f64 = 3.0;

/// The long edge of the shipped canvas: `768 · 16/9` snapped to the 32 stride.
const CANVAS_LONG_EDGE: u32 = 1344;

fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

fn env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    env(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Per-pixel standard deviation of an RGB frame — the non-degenerate gate.
fn frame_std(img: &Image) -> f64 {
    let n = img.pixels.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = img.pixels.iter().map(|&p| f64::from(p)).sum::<f64>() / n;
    (img.pixels
        .iter()
        .map(|&p| (f64::from(p) - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

fn measure(tier: &str) {
    let dir: PathBuf = env("MINIMAX_H3_VRAM_DIR")
        .expect(
            "MINIMAX_H3_VRAM_DIR must point at a staged MiniMax-H3 snapshot root (real files, not \
             HF blob symlinks: candle's memmap cannot traverse Windows reparse points)",
        )
        .into();
    assert!(
        dir.join("transformer").is_dir(),
        "{} has no transformer/ — this is not a MiniMax-H3 snapshot root",
        dir.display()
    );

    let width: u32 = env_or("MINIMAX_H3_VRAM_W", CANVAS_LONG_EDGE);
    let height: u32 = env_or("MINIMAX_H3_VRAM_H", CANVAS_SHORT_EDGE);
    let frames: u32 = env_or("MINIMAX_H3_VRAM_FRAMES", SMALLEST_LEGAL_FRAMES as u32);
    let steps: u32 = env_or("MINIMAX_H3_VRAM_STEPS", DEFAULT_STEPS);
    let prompt = env("MINIMAX_H3_VRAM_PROMPT").unwrap_or_else(|| {
        "a cellist playing on a rooftop at golden hour, the city humming below".to_owned()
    });

    // The load spec deliberately asks for **Resident**. This provider forces `Sequential`, and the
    // measurement is only meaningful if that force is what actually happens — so the probe exercises
    // the forcing path rather than politely requesting the policy it wants measured.
    let spec =
        LoadSpec::new(WeightsSource::Dir(dir.clone())).with_offload_policy(OffloadPolicy::Resident);

    let req = GenerationRequest {
        prompt,
        width,
        height,
        frames: Some(frames),
        steps: Some(steps),
        seed: Some(0),
        ..Default::default()
    };

    eprintln!(
        "[h3-vram] {MODEL_ID} tier={tier} dir={}\n[h3-vram] {width}x{height} frames={frames} \
         steps={steps} offload_requested={:?}",
        dir.display(),
        spec.offload_policy
    );

    let mut probe = VramProbe::start_rendered();
    if !reset_cuda_mempool_high_water(CANDLE_LOGICAL_DEVICE) {
        eprintln!(
            "[h3-vram] WARNING: could not reset the driver mempool USED_MEM_HIGH watermark; the \
             *MemHighGib numbers still read the pool high-water (fresh process ⇒ starts at 0)"
        );
    }

    // Bracketed separately even though this provider's `load` is paths-only, so the report PROVES
    // load-peak ~0 rather than assuming it — a load that started materializing weights would show
    // up here rather than being absorbed into the render peak.
    let load_phase = probe.phase();
    let registry =
        candle_gen_minimax_h3::provider_registry().expect("minimax-h3 provider registry");
    let generator = registry
        .load(MODEL_ID, &spec)
        .unwrap_or_else(|e| panic!("load {MODEL_ID} ({tier}) from {}: {e}", dir.display()));
    probe.end_load(load_phase);

    // Split the render at the DECODE boundary. For H3 the 36-layer transformer VAE decoder is a real
    // candidate for the peak, unlike Wan where the denoise bound — so a single overall number cannot
    // say which stage it describes, and the manifest comment has to.
    let mut pre_decode_gb = 0.0f64;
    let mut denoise_high_bytes = 0u64;
    let started = std::time::Instant::now();
    let mut on_progress = |p: Progress| {
        if matches!(p, Progress::Decoding) {
            pre_decode_gb = probe.current_peak_gb();
            denoise_high_bytes = cuda_mempool_used_high_bytes(CANDLE_LOGICAL_DEVICE).unwrap_or(0);
            // Reset so the decode's concurrent-live peak is measured in isolation.
            reset_cuda_mempool_high_water(CANDLE_LOGICAL_DEVICE);
        }
    };
    let out = generator
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("generate {MODEL_ID} ({tier}): {e}"));
    let secs = started.elapsed().as_secs_f64();

    let decode_high_bytes = cuda_mempool_used_high_bytes(CANDLE_LOGICAL_DEVICE).unwrap_or(0);
    let report = probe.finish();
    report.assert_trustworthy(MAX_BASELINE_GB);

    let (out_frames, out_fps, out_audio) = match out {
        GenerationOutput::Video { frames, fps, audio } => (frames, fps, audio),
        other => panic!("{MODEL_ID} must produce Video, got {other:?}"),
    };
    assert_eq!(
        out_frames.len(),
        frames as usize,
        "the render must deliver the requested frame count"
    );
    assert_eq!(out_fps, 24, "MiniMax-H3 renders at 24 fps");
    let audio =
        out_audio.expect("MiniMax-H3 is a JOINT model — a render without audio is a defect");
    assert!(
        audio.samples.iter().any(|s| *s != 0.0),
        "the soundtrack is all zeros: the audio VAE produced silence, so this peak describes a \
         broken run"
    );

    // A degenerate (black / uniform) clip would produce a perfectly plausible peak for a render that
    // silently failed. Gate on the MIDDLE frame: the first frame of a keyframe-free render is the
    // one most likely to be structured by accident.
    let std = frame_std(&out_frames[out_frames.len() / 2]);
    assert!(
        std > DEGENERATE_STD_FLOOR,
        "middle-frame std {std:.2} is under the degenerate floor {DEGENERATE_STD_FLOOR} — this \
         peak describes a broken render, not a coherent one"
    );

    let denoise_high_gib = gib(denoise_high_bytes);
    let decode_high_gib = gib(decode_high_bytes);
    let true_mem_high_gib = denoise_high_gib.max(decode_high_gib);
    let decode_gb = (report.peak_gb - pre_decode_gb).max(0.0);
    let owner = if decode_gb > 0.5 { "decode" } else { "denoise" };

    eprintln!(
        "\n[h3-vram] {MODEL_ID} {tier}: {report} | TRUE concurrent-live peak (USED_MEM_HIGH) \
         {true_mem_high_gib:.2} GiB = max(denoise {denoise_high_gib:.2}, decode \
         {decode_high_gib:.2}) | pre-decode {pre_decode_gb:.1} GB | decode adds (nvidia-smi) \
         {decode_gb:.1} GB | peak owner: {owner} | {} frames + {:.2}s audio in {secs:.0}s | \
         middle-frame std {std:.1}",
        out_frames.len(),
        audio.samples.len() as f64
            / f64::from(audio.sample_rate.max(1))
            / f64::from(audio.channels.max(1)),
    );
    // Machine-parseable — scrape `[[H3_VRAM]]`. `peakGb` is the nvidia-smi POOL high-water (base-10
    // GB, the manifest unit); `trueMemHighGib` is the driver mempool USED_MEM_HIGH concurrent-live
    // peak (base-2 GiB), split into `denoiseMemHighGib` / `decodeMemHighGib` so the owning stage is
    // attributable rather than guessed.
    println!(
        "[[H3_VRAM]] {{\"model\":\"{MODEL_ID}\",\"tier\":\"{tier}\",\"peakGb\":{:.3},\
         \"trueMemHighGib\":{true_mem_high_gib:.2},\"denoiseMemHighGib\":{denoise_high_gib:.2},\
         \"decodeMemHighGib\":{decode_high_gib:.2},\"peakOwner\":\"{owner}\",\
         \"preDecodeGb\":{pre_decode_gb:.1},\"decodeGb\":{decode_gb:.1},\
         \"steadyGb\":{:.1},\"loadPeakGb\":{:.1},\"baselineGb\":{:.2},\
         \"vramMeasuredPixels\":{},\"frames\":{},\"width\":{width},\"height\":{height},\
         \"steps\":{steps},\"middleFrameStd\":{std:.1},\"seconds\":{secs:.0}}}",
        report.peak_gb,
        report.steady_gb,
        report.load_peak_gb,
        report.baseline_gb,
        u64::from(width) * u64::from(height),
        out_frames.len(),
    );
}

/// The dense bf16 checkpoint at the shipped 768x1344 canvas and the 124-frame lattice floor.
///
/// The only tier this lane can measure honestly today — see the module docs on why there is no
/// `q4` / `q8` companion yet.
#[test]
#[ignore = "sc-17156 VRAM campaign; needs a staged MiniMax-H3 snapshot in MINIMAX_H3_VRAM_DIR + an idle CUDA GPU"]
fn minimax_h3_vram_bf16() {
    measure("bf16");
}
