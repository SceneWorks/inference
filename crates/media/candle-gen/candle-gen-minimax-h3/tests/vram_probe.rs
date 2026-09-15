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
//! # The three tiers, and how a tier run differs from the `bf16` one
//!
//! This lane advertises `supported_quants: [Q4, Q8]` as of sc-20267 — `crate::tier` resolves the
//! per-tier component directories and `crate::quant` builds each packed Linear straight from the MLX
//! affine triple — so all three tiers are now measurable here. There is one `#[ignore]`d test per
//! tier ([`minimax_h3_vram_bf16`], [`minimax_h3_vram_q4`], [`minimax_h3_vram_q8`]) and protocol rule
//! 2 applies with full force: **one tier per process**, each run alone.
//!
//! A tier run needs the tier **staged**, because MiniMax-H3 never quantizes at load — the DiT's
//! 66.28 GB of dense bytes plus the growing packed output will not co-reside, so every tier ships
//! pre-quantized and `spec.quantize` is an *assertion* about what is on disk. Two extra env vars
//! carry that:
//!
//! ```sh
//! MINIMAX_H3_VRAM_DIR=<snapshot root holding vae/, audio_vae/, tokenizer/> \
//! MINIMAX_H3_VRAM_TIER=q4 \
//! MINIMAX_H3_VRAM_DIT_DIR=<the tier's transformer/ dir> \
//! MINIMAX_H3_VRAM_TE_DIR=<the tier's text_encoder/ dir> \
//! CANDLE_GEN_OFFLOAD=sequential \
//!   cargo test -p candle-gen-minimax-h3 --features cuda --release minimax_h3_vram_q4 -- \
//!     --ignored --nocapture
//! ```
//!
//! Both tier dirs default to `MINIMAX_H3_VRAM_DIR`'s own `transformer/` / `text_encoder/`, which is
//! the flat-snapshot case and exactly what the `bf16` run wants. The two are staged **independently**
//! (sc-19120): the manifest ships per-tier text encoders, but the engine deliberately does not couple
//! the TE's tier to the DiT's, so a measurement of a packed DiT beside a dense TE is a legal
//! configuration and the probe does not force them to match.
//!
//! **What these runs are for, and what they are not.** `crate::memory_strategy`'s per-tier byte
//! constants are DECLARED from the manifest's hosted subtree sizes, not measured — they are on-disk
//! tier footprints. These tests are what produce the *runtime* `vramGbByTier` rows, and until they
//! have been run on an idle CUDA card there is no measured candle ceiling for any tier. The manifest
//! records that honestly today (`"measured": false`, no `vramGbByTier`), and a row must not be
//! written from arithmetic.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, OffloadPolicy, Progress, Quant,
    WeightsSource,
};
use candle_gen::testkit::{
    cuda_mempool_used_high_bytes, probe_gpu, reset_cuda_mempool_high_water, used_mib, VramProbe,
};

use candle_gen_minimax_h3::model::DEFAULT_STEPS;
use candle_gen_minimax_h3::pipeline::{CANVAS_SHORT_EDGE, SMALLEST_LEGAL_FRAMES};
// `MODEL_ID` comes from the crate ROOT, not from `model::` — `model.rs` only `use`s it privately,
// so `candle_gen_minimax_h3::model::MODEL_ID` does not name a public path and will not compile.
use candle_gen_minimax_h3::MODEL_ID;

/// Max idle-baseline VRAM (GB) tolerated before the sampled peak is considered contaminated.
const MAX_BASELINE_GB: f64 = 1.0;

/// The **logical** CUDA device candle renders on (`cuda:0`). The driver API respects
/// `CUDA_VISIBLE_DEVICES`, so logical 0 is the physical card candle uses.
const CANDLE_LOGICAL_DEVICE: i32 = 0;

/// A render whose middle frame is flatter than this is degenerate (black / uniform), which means the
/// engine failed silently and the peak describes a broken run.
const DEGENERATE_STD_FLOOR: f64 = 3.0;

/// How much the decode must add over the pre-decode sample before the peak is attributed to it.
///
/// Both sides of the difference are **over the idle baseline** (see `measure`), so this is a noise
/// margin, not a bias allowance. It has to stay well under the smallest decode this could plausibly
/// see — the video VAE alone is 10.42 GB — while being above nvidia-smi's sampling jitter.
const DECODE_OWNER_THRESHOLD_GB: f64 = 0.5;

/// Force the structured receipt onto its own line after libtest's `test ...` prefix.
const H3_VRAM_RECEIPT_PREFIX: &str = "\n[[H3_VRAM]] ";

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

fn measure(tier: &str, quant: Option<Quant>) {
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
    let mut spec =
        LoadSpec::new(WeightsSource::Dir(dir.clone())).with_offload_policy(OffloadPolicy::Resident);

    // **Stage the tier, and assert it.** H3 never quantizes at load, so a `q4` number can only be
    // measured against a pre-quantized `q4` DiT. `spec.quantize` makes it an assertion: if the staged
    // dir is not really that tier, `crate::tier`'s reconcile fails the load rather than quietly
    // measuring a different tier and publishing the figure under this one's name.
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    // Both default to the flat snapshot's own dirs, which is what the `bf16` run wants.
    let dit_dir: PathBuf = env("MINIMAX_H3_VRAM_DIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("transformer"));
    let te_dir: PathBuf = env("MINIMAX_H3_VRAM_TE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| dir.join("text_encoder"));
    spec.components.insert(
        "transformer".to_owned(),
        WeightsSource::Dir(dit_dir.clone()),
    );
    spec.components.insert(
        "text_encoder".to_owned(),
        WeightsSource::Dir(te_dir.clone()),
    );
    eprintln!(
        "[h3-vram] staged transformer={} text_encoder={}",
        dit_dir.display(),
        te_dir.display()
    );

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
    // The PHYSICAL ordinal `start_rendered` sampled — derived from `CUDA_VISIBLE_DEVICES`, so a
    // multi-GPU box cannot render on one card while this samples another. The box behind the
    // `cuda` runner label has two cards and `default_device()` pins logical `cuda:0`.
    let gpu = probe_gpu();
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
    // **Absolute** device usage at the decode boundary — NOT baseline-subtracted, because the probe
    // is sampled directly here and `report.baseline_gb` is not available until the report is folded.
    // It is put on the same footing as `report.peak_gb` below, before anything is derived from it.
    let mut pre_decode_abs_gb = 0.0f64;
    let mut denoise_high_bytes = 0u64;
    let started = std::time::Instant::now();
    // **The generate phase must be BRACKETED**, exactly as the Wan sibling brackets it. `report()`
    // folds the overall peak from the phases that were closed; a render measured without an
    // `end_gen` reports the LOAD peak as the overall peak, and this provider's load is paths-only,
    // so the published number would have been ~0 for a render that really peaks above 60 GB.
    let generate_phase = probe.phase();
    let mut on_progress = |p: Progress| {
        if matches!(p, Progress::Decoding) {
            // Sample the device directly rather than asking the probe: the probe's overall peak is
            // not folded until `end_gen`, so mid-render it does not yet know about this phase.
            // Base-10 GB, the manifest unit, matching the Wan probe's derivation line for line.
            pre_decode_abs_gb = used_mib(gpu).unwrap_or(0) as f64 * 1048576.0 / 1.0e9;
            denoise_high_bytes = cuda_mempool_used_high_bytes(CANDLE_LOGICAL_DEVICE).unwrap_or(0);
            // Reset so the decode's concurrent-live peak is measured in isolation.
            reset_cuda_mempool_high_water(CANDLE_LOGICAL_DEVICE);
        }
    };
    let out = generator
        .generate(&req, &mut on_progress)
        .unwrap_or_else(|e| panic!("generate {MODEL_ID} ({tier}): {e}"));
    probe.end_gen(generate_phase);
    let secs = started.elapsed().as_secs_f64();

    let decode_high_bytes = cuda_mempool_used_high_bytes(CANDLE_LOGICAL_DEVICE).unwrap_or(0);
    let report = probe.report().assert_trustworthy(MAX_BASELINE_GB);

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
    // **Put both sides of the difference on the same footing before differencing them.**
    // `report.peak_gb` is over the idle baseline (`VramProbe::report` subtracts it); the mid-render
    // sample above is absolute. Differencing them directly carried up to a full baseline of bias —
    // and `MAX_BASELINE_GB` is 1.0 GB, twice `DECODE_OWNER_THRESHOLD_GB`, so on a card with a
    // tolerated-but-nonzero baseline the bias alone could flip `peakOwner` from decode to denoise.
    // The story asks the manifest comment to NAME the peak-owning phase, so that is a published
    // fact, not a log line — the Wan sibling shares this derivation but only prints it.
    let pre_decode_gb = (pre_decode_abs_gb - report.baseline_gb).max(0.0);
    let decode_gb = (report.peak_gb - pre_decode_gb).max(0.0);
    let owner = if decode_gb > DECODE_OWNER_THRESHOLD_GB {
        "decode"
    } else {
        "denoise"
    };

    eprintln!(
        "\n[h3-vram] {MODEL_ID} {tier}: {report} | TRUE concurrent-live peak (USED_MEM_HIGH) \
         {true_mem_high_gib:.2} GiB = max(denoise {denoise_high_gib:.2}, decode \
         {decode_high_gib:.2}) | pre-decode {pre_decode_gb:.1} GB over baseline (abs \
         {pre_decode_abs_gb:.1}) | decode adds (nvidia-smi) {decode_gb:.1} GB | peak owner: \
         {owner} | {} frames + {:.2}s audio in {secs:.0}s | middle-frame std {std:.1}",
        out_frames.len(),
        audio.samples.len() as f64
            / f64::from(audio.sample_rate.max(1))
            / f64::from(audio.channels.max(1)),
    );
    // Machine-parseable — scrape `[[H3_VRAM]]`. `peakGb` is the nvidia-smi POOL high-water (base-10
    // GB, the manifest unit); `trueMemHighGib` is the driver mempool USED_MEM_HIGH concurrent-live
    // peak (base-2 GiB), split into `denoiseMemHighGib` / `decodeMemHighGib` so the owning stage is
    // attributable rather than guessed. `preDecodeGb` is over the idle baseline, like `peakGb`;
    // `preDecodeAbsGb` is the raw device sample it came from, kept as provenance for the scrape.
    println!(
        "{H3_VRAM_RECEIPT_PREFIX}{{\"model\":\"{MODEL_ID}\",\"tier\":\"{tier}\",\"peakGb\":{:.3},\
         \"trueMemHighGib\":{true_mem_high_gib:.2},\"denoiseMemHighGib\":{denoise_high_gib:.2},\
         \"decodeMemHighGib\":{decode_high_gib:.2},\"peakOwner\":\"{owner}\",\
         \"preDecodeGb\":{pre_decode_gb:.1},\"preDecodeAbsGb\":{pre_decode_abs_gb:.1},\
         \"decodeGb\":{decode_gb:.1},\
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

#[test]
fn vram_receipt_prefix_forces_a_fresh_log_line() {
    assert_eq!(H3_VRAM_RECEIPT_PREFIX, "\n[[H3_VRAM]] ");
}

/// The dense bf16 checkpoint at the shipped 768x1344 canvas and the 124-frame lattice floor.
///
/// No `spec.quantize`: `None` means "the caller asserted nothing", which is the correct request for a
/// dense snapshot. Asserting `Bf16` is not expressible through `spec.quantize` at all — gen-core's
/// `Quant` has no dense variant — and the resolver treats an absent request as unasserted rather than
/// as a demand for denseness.
#[test]
#[ignore = "sc-17156 VRAM campaign; needs a staged MiniMax-H3 snapshot in MINIMAX_H3_VRAM_DIR + an idle CUDA GPU"]
fn minimax_h3_vram_bf16() {
    measure("bf16", None);
}

/// The **`q4`** tier at the same geometry (sc-20267).
///
/// Needs `MINIMAX_H3_VRAM_DIT_DIR` (and optionally `MINIMAX_H3_VRAM_TE_DIR`) pointed at a
/// pre-quantized `q4` subtree — see the module docs. `spec.quantize = Q4` makes that an assertion, so
/// a mis-staged dir fails the load instead of publishing a bf16 number under the `q4` row.
///
/// **Run alone.** cudarc's caching allocator never returns pages to the driver, so a second tier
/// measured in the same process re-reports the first tier's high-water (protocol rule 2).
#[test]
#[ignore = "sc-17156 VRAM campaign; needs a staged q4 MiniMax-H3 tier + an idle CUDA GPU; run alone"]
fn minimax_h3_vram_q4() {
    measure("q4", Some(Quant::Q4));
}

/// The **`q8`** tier at the same geometry (sc-20267). Run alone — see [`minimax_h3_vram_q4`].
#[test]
#[ignore = "sc-17156 VRAM campaign; needs a staged q8 MiniMax-H3 tier + an idle CUDA GPU; run alone"]
fn minimax_h3_vram_q8() {
    measure("q8", Some(Quant::Q8));
}
