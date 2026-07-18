//! sc-12402 — per-(model x tier) VRAM measurement for the candle Wan video engines, so SceneWorks'
//! fit gate can predict the real PEAK instead of the sc-12344 weights FLOOR.
//!
//! The Wan sibling of `candle-gen-flux2`'s `run_probed_offload_ab` (sc-10868) and the harness behind
//! the shipped `candle.vramGbByTier` blocks for flux2 (sc-10920) / qwen-image (sc-10969) / qwen-edit
//! (sc-11019). Same instrument — [`candle_gen::testkit::VramProbe`], a device-level `nvidia-smi
//! memory.used` sampler over a recorded idle baseline (sc-9094) — pointed at the Wan providers.
//!
//! # What is measured, and why it is the OVERALL peak
//!
//! `vramGbByTier` is the number the card must physically hold, so the reported quantity is
//! `VramReport::peak_gb`: the max across load + denoise + VAE decode, over the idle baseline.
//!
//! **Wan's `load` is LAZY** (`components: Mutex::new(None)` — lib.rs / wan14b.rs), so unlike the
//! eager FLUX.2 path the load phase allocates NOTHING on the device and `load_peak`/`steady` read
//! ~0. Everything — the component build and the denoise — lands inside the generate phase. That is
//! not a defect in the measurement: it mirrors the z_image manifest note ("candle loads lazily, so
//! load-peak is negligible — the denoise peak dominates"). The phases are still bracketed
//! separately so the report proves that rather than assuming it.
//!
//! # ONE TIER PER PROCESS — mandatory
//!
//! candle's CUDA backend uses cudarc's stream-ordered caching allocator: there is no `empty_cache`
//! and freed pages never return to the driver. A second measurement in the same process therefore
//! reuses the first tier's pool and re-reports the FIRST tier's high-water. Run exactly one
//! `wan_vram_*` per `cargo test` invocation (the same rule `footprint_measure.rs` enforces for the
//! MLX counters, and why the flux2 A/B demands separate processes).
//!
//! # The snapshot must be REAL FILES, not HF blob symlinks
//!
//! An HF cache snapshot stores each shard as a relative symlink into `blobs/`, which candle's memmap
//! cannot traverse on Windows (os-error-448, reparse point). Hardlink-stage the tier onto the same
//! volume first (see the campaign runner) and point `WAN_VRAM_DIR` at the staged tree. Same
//! requirement the flux2 harness carries.
//!
//! # The recipe is the PRODUCTION recipe, deliberately
//!
//! A peak measured at a toy geometry would under-predict the gate it feeds, so each engine defaults
//! to exactly what SceneWorks' candle video lane dispatches (`video_jobs.rs::candle_wan_sampling` +
//! the manifest `defaults`), snapped onto Wan's `4k+1` temporal lattice:
//!
//! | engine             | geometry   | frames             | steps | guidance     | Lightning |
//! |--------------------|------------|--------------------|-------|--------------|-----------|
//! | `wan2_2_ti2v_5b`   | 832x480    | 121 (5s @ 24fps)   | 20    | engine (5.0) | n/a       |
//! | `wan2_2_t2v_14b`   | 1280x720   | 81  (5s @ 16fps)   | 4     | 1.0 (CFG-off)| REQUIRED  |
//! | `wan2_2_i2v_14b`   | 1280x720   | 81  (5s @ 16fps)   | 4     | 1.0 (CFG-off)| REQUIRED  |
//!
//! **The A14B Lightning pair is not optional here.** `candle_wan_lightning_on` is a default-ON
//! toggle, so the 4-step distill rides EVERY default A14B job: measuring without it would record a
//! tier no user runs. On a packed q4/q8 tier it applies additively (`AdaptLinear`, base stays
//! packed); on the dense tier it FOLDS via `merge_adapters`, which peaks heavier (the Qwen-Edit
//! Lightning precedent, sc-11066). Omit it only via `WAN_VRAM_LIGHTNING=0`, for a deliberate A/B.
//!
//! # Run
//!
//! ```text
//! # one tier per process; GPU must be idle (the probe asserts baseline < 1 GB)
//! $env:WAN_VRAM_DIR="E:/staged/wan-t2v-a14b-q4"
//! $env:WAN_VRAM_TIER="q4"
//! $env:WAN_VRAM_LORA_HIGH="E:/staged/lightning-t2v/high_noise_model.safetensors"
//! $env:WAN_VRAM_LORA_LOW="E:/staged/lightning-t2v/low_noise_model.safetensors"
//! cargo test -p candle-gen-wan --features cuda --release wan_vram_t2v_14b -- --ignored --nocapture
//! ```
//!
//! Each run prints one machine-parseable line to scrape into `builtin.models.jsonc`:
//! `[[WAN_VRAM]] {"model":...,"tier":...,"peakGb":...,"steadyGb":...,"loadPeakGb":...}`.

#![cfg(feature = "cuda")]

use std::path::PathBuf;

use candle_gen::gen_core::{
    AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec,
    MoeExpert, Progress, WeightsSource,
};
use candle_gen::testkit::{
    cuda_mempool_used_high_bytes, reset_cuda_mempool_high_water, VramProbe,
};

/// Max idle-baseline VRAM (GB) tolerated before the sampled peak is considered contaminated by
/// another process. Matches the flux2 harness's `assert_trustworthy(1.0)`.
const MAX_BASELINE_GB: f64 = 1.0;

/// The **logical** CUDA device candle renders on (`cuda:0`). The driver API respects
/// `CUDA_VISIBLE_DEVICES`, so logical 0 is the physical card candle uses — NOT the physical nvidia-smi
/// ordinal (`probe_gpu`). The `USED_MEM_HIGH` mempool probe (sc-12818) reads this device's default pool.
const CANDLE_LOGICAL_DEVICE: i32 = 0;

/// Bytes → GiB (base-2, the unit the campaign quotes the concurrent-live peak in).
fn gib(bytes: u64) -> f64 {
    bytes as f64 / 1_073_741_824.0
}

/// A render whose middle frame is flatter than this is degenerate (black / uniform), which means the
/// engine failed silently and the peak describes a broken run. Mirrors `smoke_support`'s
/// `DEGENERATE_STD_FLOOR_DEFAULT` on the worker side.
const DEGENERATE_STD_FLOOR: f64 = 3.0;

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
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    (img.pixels
        .iter()
        .map(|&p| {
            let d = p as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n)
        .sqrt()
}

/// A synthetic first frame for the I2V conditioning — a smooth diagonal gradient with a centered
/// block. VRAM is content-independent (the latent shape is fixed by the geometry), so a fixture
/// would buy nothing; this keeps the harness self-contained. `WAN_VRAM_IMAGE` overrides it.
fn synthetic_first_frame(w: u32, h: u32) -> Image {
    let (wu, hu) = (w as usize, h as usize);
    let mut pixels = vec![0u8; wu * hu * 3];
    for y in 0..hu {
        for x in 0..wu {
            let i = (y * wu + x) * 3;
            pixels[i] = ((x * 255) / wu.max(1)) as u8;
            pixels[i + 1] = ((y * 255) / hu.max(1)) as u8;
            let centered = x > wu / 3 && x < 2 * wu / 3 && y > hu / 3 && y < 2 * hu / 3;
            pixels[i + 2] = if centered { 220 } else { 40 };
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// The production recipe for one engine — see the module table. Kept as data so a drift in
/// SceneWorks' `candle_wan_sampling` is a one-line change here.
struct Recipe {
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    steps: u32,
    /// `None` ⇒ let the engine apply its own config default (the 5B's guide 5.0). `Some(1.0)` is
    /// the A14B Lightning CFG-off preset.
    guidance: Option<f32>,
    /// A14B MoE ⇒ the mandatory 4-step Lightning distill pair.
    moe: bool,
    /// I2V ⇒ needs a `Conditioning::Reference` first frame.
    i2v: bool,
}

fn recipe(engine_id: &str) -> Recipe {
    match engine_id {
        // manifest defaults: 832x480, 5s @ 24fps (sc-12319 fast path); candle interim 20 steps, CFG on.
        "wan2_2_ti2v_5b" => Recipe {
            width: 832,
            height: 480,
            frames: 121,
            fps: 24,
            steps: 20,
            guidance: None,
            moe: false,
            i2v: false,
        },
        // manifest defaults: 1280x720, 5s @ 16fps; Lightning default-on ⇒ 4 steps / CFG-off.
        "wan2_2_t2v_14b" => Recipe {
            width: 1280,
            height: 720,
            frames: 81,
            fps: 16,
            steps: 4,
            guidance: Some(1.0),
            moe: true,
            i2v: false,
        },
        "wan2_2_i2v_14b" => Recipe {
            width: 1280,
            height: 720,
            frames: 81,
            fps: 16,
            steps: 4,
            guidance: Some(1.0),
            moe: true,
            i2v: true,
        },
        other => panic!("no recipe for {other}"),
    }
}

/// Build the Lightning distill adapter pair from the staged file paths. Mirrors the worker's
/// `candle_resolve_wan_adapters`: strength 1.0, tagged per MoE expert.
fn lightning_adapters() -> Vec<AdapterSpec> {
    // WAN_VRAM_LIGHTNING=0 ⇒ deliberate no-distill A/B.
    if env_or::<u32>("WAN_VRAM_LIGHTNING", 1) == 0 {
        return Vec::new();
    }
    let high = env("WAN_VRAM_LORA_HIGH").unwrap_or_else(|| {
        panic!(
            "set WAN_VRAM_LORA_HIGH / WAN_VRAM_LORA_LOW to the staged lightx2v/Wan2.2-Lightning \
             high+low files for this architecture (T2V ⇒ Seko-V1.1, I2V ⇒ Seko-V1 — they are NOT \
             cross-compatible). The A14B recipe bakes the 4-step distill on every default job, so a \
             measurement without it records a tier nobody runs. Set WAN_VRAM_LIGHTNING=0 to opt out."
        )
    });
    let low = env("WAN_VRAM_LORA_LOW").expect("set WAN_VRAM_LORA_LOW alongside WAN_VRAM_LORA_HIGH");
    vec![
        AdapterSpec::new(PathBuf::from(high), 1.0, AdapterKind::Lora)
            .with_moe_expert(MoeExpert::High),
        AdapterSpec::new(PathBuf::from(low), 1.0, AdapterKind::Lora)
            .with_moe_expert(MoeExpert::Low),
    ]
}

/// Measure one (engine x tier): probe the idle baseline, run ONE real production-recipe generation,
/// and print the scrape line. See the module docs for the one-tier-per-process rule.
fn measure(engine_id: &str) {
    let dir = env("WAN_VRAM_DIR").unwrap_or_else(|| {
        panic!(
            "set WAN_VRAM_DIR to a hardlink-staged (real-file, NOT raw HF blob symlink) snapshot: \
             the tier subdir for a packed q4/q8 tier, or the dense diffusers snapshot root for bf16"
        )
    });
    let tier = env("WAN_VRAM_TIER").unwrap_or_else(|| "unknown".to_owned());
    let r = recipe(engine_id);
    let (width, height) = (
        env_or("WAN_VRAM_W", r.width),
        env_or("WAN_VRAM_H", r.height),
    );
    let frames = env_or("WAN_VRAM_FRAMES", r.frames);
    let steps = env_or("WAN_VRAM_STEPS", r.steps);

    let adapters = if r.moe {
        lightning_adapters()
    } else {
        Vec::new()
    };
    // `spec.quantize` is deliberately unset: on Wan the tier IS the directory (the packed-detect
    // loaders read the `.scales` marker off tensor content), so `quantize` is a no-op tier-select
    // marker the engine ignores — see `load`'s sc-10025 note. Passing it would imply it selects
    // something here.
    let spec =
        LoadSpec::new(WeightsSource::Dir(PathBuf::from(&dir))).with_adapters(adapters.clone());

    let conditioning = if r.i2v {
        let image = match env("WAN_VRAM_IMAGE") {
            Some(p) => {
                let rgb = image::open(&p)
                    .unwrap_or_else(|e| panic!("open {p}: {e}"))
                    .to_rgb8();
                Image {
                    width: rgb.width(),
                    height: rgb.height(),
                    pixels: rgb.into_raw(),
                }
            }
            None => synthetic_first_frame(width, height),
        };
        vec![Conditioning::Reference {
            image,
            strength: None,
        }]
    } else {
        Vec::new()
    };

    let req = GenerationRequest {
        prompt: env("WAN_VRAM_PROMPT").unwrap_or_else(|| {
            "a fluffy cat walking across a sunny garden, cinematic, shallow depth of field"
                .to_owned()
        }),
        width,
        height,
        count: 1,
        seed: Some(42),
        steps: Some(steps),
        guidance: r.guidance,
        frames: Some(frames),
        fps: Some(r.fps),
        conditioning,
        ..Default::default()
    };

    eprintln!(
        "[wan-vram] {engine_id} tier={tier} dir={dir}\n[wan-vram] {width}x{height} frames={frames} \
         fps={} steps={steps} guidance={:?} adapters={} (lightning)",
        r.fps,
        r.guidance,
        adapters.len()
    );

    // Baseline on the physical GPU candle's logical cuda:0 renders on (derived from
    // CUDA_VISIBLE_DEVICES — a multi-GPU box must not render on one card and sample another).
    let mut probe = VramProbe::start_rendered();

    // sc-12818: reset the driver mempool USED_MEM_HIGH watermark up front so the DENOISE-phase read
    // (at Progress::Decoding) captures the load+denoise true concurrent-live peak from a clean slate.
    // This is the accurate number — nvidia-smi's ~40 ms polling under-samples the brief attention /
    // im2col / decode transients ~2× — and it is split into denoise vs decode below (each reported
    // alongside the nvidia-smi peak) to ATTRIBUTE the peak: the campaign's fixed ~30 GiB A14B floor is
    // owned by the denoise attention, not the VAE decode (so a bf16 VAE, sc-12818, does not move it).
    if !reset_cuda_mempool_high_water(CANDLE_LOGICAL_DEVICE) {
        eprintln!(
            "[wan-vram] WARNING: could not reset the driver mempool USED_MEM_HIGH watermark; \
             the *MemHighGib numbers still read the pool high-water (fresh process ⇒ starts at 0)"
        );
    }

    // Bracketed separately even though Wan's load is lazy, so the report PROVES load-peak ~0 rather
    // than assuming it.
    let load_phase = probe.phase();
    let generator = candle_gen_wan::provider_registry()
        .expect("wan provider registry")
        .load(engine_id, &spec)
        .unwrap_or_else(|e| panic!("load {engine_id} ({tier}) from {dir}: {e}"));
    probe.end_load(load_phase);

    // Split the generate phase at the DECODE boundary. Wan's peak has two candidate owners — the
    // denoise working set (card-INDEPENDENT) and the z48 vae22 decode (card-ADAPTIVE: its tiler
    // budgets `0.85 x TOTAL VRAM`, so it expands to fill the card) — and a single overall peak cannot
    // say which one it is. `Progress::Decoding` fires exactly when the denoise ends and the decode
    // begins, so sampling the device there yields `weights + denoise` and the remainder is the decode.
    // Without this split a measured peak cannot be attributed, and attributing it is the whole
    // question: a denoise-owned peak is a legitimate per-tier constant, a decode-owned one is a fact
    // about the measuring card (sc-12402).
    let gpu = candle_gen::testkit::probe_gpu();
    let pre_decode_mib = std::sync::atomic::AtomicU64::new(0);
    // sc-12818: the denoise-phase USED_MEM_HIGH, captured at the decode boundary (0 = not captured).
    let denoise_high_bytes = std::sync::atomic::AtomicU64::new(0);
    let generate_phase = probe.phase();
    let t0 = std::time::Instant::now();
    let output = generator
        .generate(&req, &mut |p: Progress| match p {
            Progress::Step { current, total } => {
                eprint!("\r[wan-vram] step {current}/{total}   ");
            }
            Progress::Decoding => {
                let mib = candle_gen::testkit::used_mib(gpu).unwrap_or(0);
                pre_decode_mib.store(mib, std::sync::atomic::Ordering::Relaxed);
                // sc-12818: read the driver's true DENOISE-phase concurrent-live peak (accurate where
                // the nvidia-smi `mib` above under-samples the attention transient), then RESET the
                // watermark so the remaining work (the VAE decode) is measured in isolation.
                let dh = cuda_mempool_used_high_bytes(CANDLE_LOGICAL_DEVICE).unwrap_or(0);
                denoise_high_bytes.store(dh, std::sync::atomic::Ordering::Relaxed);
                reset_cuda_mempool_high_water(CANDLE_LOGICAL_DEVICE);
                eprintln!(
                    "\n[wan-vram] denoise done, decode starts — nvidia-smi high-water here: {mib} MiB \
                     ({:.1} GiB) | denoise USED_MEM_HIGH {:.2} GiB (true concurrent-live)",
                    mib as f64 * 1048576.0 / 1073741824.0,
                    gib(dh),
                );
            }
            Progress::Loading(_) => {}
        })
        .unwrap_or_else(|e| panic!("{engine_id} ({tier}) generate: {e}"));
    probe.end_gen(generate_phase);
    let secs = t0.elapsed().as_secs_f32();
    // sc-12818: the driver's honest concurrent-live peaks (GiB, base-2) — the accurate re-baseline for
    // the campaign's ~2×-understated nvidia-smi numbers. `decode_high` is measured from the reset at the
    // decode boundary (⇒ the isolated VAE-decode true peak, incl. resident weights/latent); `denoise_high`
    // was captured there. The overall true peak is the max — and attributes which stage owns the floor.
    let denoise_high = denoise_high_bytes.load(std::sync::atomic::Ordering::Relaxed);
    let decode_high = cuda_mempool_used_high_bytes(CANDLE_LOGICAL_DEVICE).unwrap_or(0);
    let denoise_high_gib = gib(denoise_high);
    let decode_high_gib = gib(decode_high);
    let true_mem_high_gib = denoise_high_gib.max(decode_high_gib);
    let pre_decode_gb =
        pre_decode_mib.load(std::sync::atomic::Ordering::Relaxed) as f64 * 1048576.0 / 1.0e9;

    let out_frames = match output {
        GenerationOutput::Video { frames, .. } => frames,
        GenerationOutput::Images(_) => panic!("expected video, got images"),
    };
    assert!(!out_frames.is_empty(), "engine returned no frames");
    // Guard the CREDIBILITY of the number: a black/uniform clip means the run failed silently and
    // its peak describes nothing worth recording in the manifest.
    let std = frame_std(&out_frames[out_frames.len() / 2]);
    assert!(
        std > DEGENERATE_STD_FLOOR,
        "{engine_id} ({tier}) render looks degenerate (middle-frame std {std:.2}) — the measured \
         peak would be bogus"
    );

    let report = probe.report().assert_trustworthy(MAX_BASELINE_GB);
    // Attribute the peak. `decodeGb` is the decode's marginal contribution over the denoise
    // high-water; when it is ~0 the peak is denoise-owned (and the card-adaptive tiler never moved
    // it), when it dominates the peak is a fact about this card's VRAM, not about the model.
    let decode_gb = (report.peak_gb - pre_decode_gb).max(0.0);
    eprintln!(
        "\n[wan-vram] {engine_id} {tier}: {report} | TRUE concurrent-live peak (USED_MEM_HIGH) \
         {true_mem_high_gib:.2} GiB = max(denoise {denoise_high_gib:.2}, decode {decode_high_gib:.2}) \
         | pre-decode {pre_decode_gb:.1} GB | decode adds (nvidia-smi) {decode_gb:.1} GB | {} frames in \
         {secs:.0}s | middle-frame std {std:.1}",
        out_frames.len()
    );
    // Machine-parseable — scrape `[[WAN_VRAM]]`. `peakGb` is the nvidia-smi high-water (base-10 GB, the
    // manifest unit); `trueMemHighGib` is the driver mempool USED_MEM_HIGH concurrent-live peak (base-2
    // GiB, sc-12818) — the accurate number the nvidia-smi poll under-samples ~2× — split into
    // `denoiseMemHighGib` / `decodeMemHighGib` so the owning stage is attributable.
    println!(
        "[[WAN_VRAM]] {{\"model\":\"{engine_id}\",\"tier\":\"{tier}\",\"peakGb\":{:.1},\
         \"trueMemHighGib\":{true_mem_high_gib:.2},\"denoiseMemHighGib\":{denoise_high_gib:.2},\
         \"decodeMemHighGib\":{decode_high_gib:.2},\"preDecodeGb\":{pre_decode_gb:.1},\
         \"decodeGb\":{decode_gb:.1},\"vaeBudgetGib\":\"{}\",\
         \"steadyGb\":{:.1},\"loadPeakGb\":{:.1},\"baselineGb\":{:.2},\"frames\":{},\
         \"width\":{width},\"height\":{height},\"steps\":{steps},\"seconds\":{secs:.0}}}",
        report.peak_gb,
        env("WAN_VAE_BUDGET_GIB").unwrap_or_else(|| "auto(0.85xTOTAL)".to_owned()),
        report.steady_gb,
        report.load_peak_gb,
        report.baseline_gb,
        out_frames.len(),
    );
}

/// Dense TI2V-5B (single transformer, z48 VAE) at the shipped 832x480 / 5 s default.
#[test]
#[ignore = "sc-12402 VRAM campaign; needs a staged wan_2_2 tier in WAN_VRAM_DIR + an idle CUDA GPU"]
fn wan_vram_ti2v_5b() {
    measure("wan2_2_ti2v_5b");
}

/// T2V-A14B dual-expert MoE at the shipped 1280x720 / 5 s default + the mandatory Lightning distill.
#[test]
#[ignore = "sc-12402 VRAM campaign; needs a staged wan_2_2_t2v_14b tier + Lightning pair + an idle CUDA GPU"]
fn wan_vram_t2v_14b() {
    measure("wan2_2_t2v_14b");
}

/// I2V-A14B dual-expert MoE (channel-concat conditioning) at the shipped default + Lightning.
#[test]
#[ignore = "sc-12402 VRAM campaign; needs a staged wan_2_2_i2v_14b tier + Lightning pair + an idle CUDA GPU"]
fn wan_vram_i2v_14b() {
    measure("wan2_2_i2v_14b");
}

