//! Real-weight LoRA/LoKr evidence for Qwen-Image 2.1 on MLX (sc-24163, epic sc-24107 Phase 2) —
//! every test is `#[ignore]`d and needs the pinned snapshots plus the Metal GPU:
//!
//! * `MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT` — `Qwen/Qwen-Image-2.1` @ `790c9263…` (the dense bf16 base
//!   the trainer trains over and the bf16 render tier);
//! * `MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT` — `SceneWorks/qwen-image-2-1-mlx` @ `1691de01…`,
//!   holding the complete `q8/` and `q4/` tier snapshots;
//! * `QWEN_IMAGE_2_1_RENDER_OUT` — the evidence directory (PNGs, adapters, one JSON per test).
//!
//! Inference never self-fetches or derives a cache location (epic 13657): the dispatch-only
//! `qwen-image-2-1-lora-mlx` profile of `.github/workflows/real-weights.yml` (job
//! `mlx-qwen-image-2-1`) materializes both snapshots and runs every test here by name, in this
//! order — the stacking test consumes the two adapters the training tests write:
//!
//! 1. [`t2i_lora_trains_reloads_and_moves_every_tier`] — a short text-to-image **LoRA** run on a
//!    synthetic single-palette style, saved, reloaded through `LoadSpec::adapters`, and rendered
//!    with and without it at bf16, q8 and q4 on the same seed.
//! 2. [`edit_lokr_trains_and_moves_two_reference_edits_every_tier`] — a representative short instruction-edit
//!    **LoKr** run on one-reference edit pairs, then the same with/without comparison on a held-out
//!    two-reference edit at every tier.
//! 3. [`stacked_adapters_apply_with_independent_weights`] — both trained files stacked on one
//!    load, each at its own strength.
//! 4. [`third_party_lora_applies_strictly_and_moves_every_tier`] — a public third-party 2.1 LoRA
//!    (operator-supplied via `QWEN_IMAGE_2_1_THIRD_PARTY_LORA`; the lane runs it only when the
//!    dispatch names one).
//! 5. [`imported_adapters_move_t2i_and_two_reference_edit_every_tier`] — the hash-pinned CUDA
//!    LoRA/LoKr and preserved MLX 1000-step LoRA, plus this candidate's corrected edit LoKr in
//!    `edit`/`full` phases, on both routes at every tier. `probe` runs two training steps per mode
//!    without renders; `edit` reuses the completed T2I file; `imports` runs only donor/public cells.
//!    The retained style donor's T2I cells use its original style-only request and three dedicated
//!    bare bases; the terminal edit phase therefore has 51 renders. Other donor requests stay fixed.
//! 6. [`diagnostic_reused_t2i_style_direction`] — explicitly diagnostic-only six style renders
//!    (three bare/adapted pairs), reusing the immutable original donor without training.
//!
//! EVERY TEST WRITES ITS EVIDENCE BEFORE IT ASSERTS. The PNGs and the `<test>.json` metrics
//! (per-render mean |Δ|, the learned-direction scores, MLX active peaks, process footprint and the
//! crate's own predicted overlay / training footprint) land first, so a red run still leaves the
//! numbers a reviewer needs to tell a defect from a threshold that was sized wrong.
//!
//! THE THRESHOLDS ARE FIRST-RUN FLOORS, NOT CALIBRATED BARS. Each one is the weakest claim that
//! still rules out the failure it names (an adapter that never reached the forward, a tier that
//! silently dropped it, a stacked strength that leaks into its neighbour); the JSON carries the
//! measured margin so they can be re-sized from evidence.
//!
//! A physical watchdog admits training from its exact preflight, measured baseline overhead,
//! available RAM and Metal working-set policy. Physical bytes are sampled in the background;
//! allocator bytes are foreground stage snapshots with limited timestamp-based attribution. An
//! explicit `QWEN_IMAGE_2_1_FOOTPRINT_CEILING_GB` further constrains that ceiling. Inference retains
//! the 100 GB default. Aborts leave an exact-byte receipt before stopping the process.
//!
//! ```sh
//! MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT=…/models--Qwen--Qwen-Image-2.1/snapshots/790c9263… \
//! MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT=…/models--SceneWorks--qwen-image-2-1-mlx/snapshots/1691de01… \
//! QWEN_IMAGE_2_1_RENDER_OUT=~/SceneWorks/lora-evidence-sc-24163 \
//!   cargo test --locked --release -p mlx-gen-qwen-image-2-1 --test integration \
//!   lora_real_weights::t2i_lora_trains_reloads_and_moves_every_tier \
//!   -- --ignored --exact --nocapture --test-threads 1
//! ```

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use mlx_gen::gen_core::weightsmeta::safetensors_file_metadata;
use mlx_gen::gen_core::Conditioning;
use mlx_gen::runtime::{AdapterKind, AdapterSpec};
use mlx_gen::{
    GenerationOutput, GenerationRequest, Image, LoadSpec, NetworkType, Quant, Trainer,
    TrainingConfig, TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_qwen_image_2_1::memory_strategy::memory_strategy_contract;
use mlx_gen_qwen_image_2_1::{provider_registry, QwenImage21Trainer, TRAINER_ID};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::e2e_real_weights::phys_footprint;

#[path = "support/training_peaks.rs"]
mod training_peaks;

const ID: &str = "qwen_image_2_1";

// ── thresholds (first-run floors; see the module docs) ──────────────────────────────────────────

/// Mean |Δ| per RGB byte (0–255) an adapter must move a same-seed render by. A residual that never
/// reaches the forward, or a tier that drops it, measures exactly 0; any trained adapter that
/// reaches the DiT moves an 8-step render by far more than 2/255.
const ADAPTER_MOVES_FLOOR: f64 = 2.0;
/// Mean |Δ| one stacked file at strength 1 must add over the other alone — the weaker of the two
/// files (the edit LoKr, on a text-to-image render) must still be visible in the stack.
const STACK_MOVES_FLOOR: f64 = 0.5;
/// Mean RGB distance to the nearest training-palette colour ([`palette_distance`]) by which the T2I
/// LoRA's render must be CLOSER to the palette than the bare base's, same prompt and seed — the "it
/// learned the style" direction, not just "something changed". A no-op adapter measures exactly 0.
/// The first-run floor was a palette-FRACTION gain of 0.05, which is binary per pixel: a 1e-4 x 300
/// step run (inference run 37124582021) moved every tier toward the palette (distance 164.7 ->
/// 161.8 bf16, 164.5 -> 161.7 q8, 159.8 -> 158.7 q4) while the fraction stayed 0.000 -> 0.000.
const PALETTE_DISTANCE_GAIN_FLOOR: f64 = 1.0;
/// Mean |Δ| per byte by which the edit adapter's output must be CLOSER to the trained transform of
/// the held-out source than the bare base's output is.
const EDIT_GAIN_FLOOR: f64 = 1.0;
/// Slack on top of a prediction before an observed peak counts as an under-prediction: MLX's
/// active peak carries per-step transients the derived models round, and the preflight prints its
/// figures to one decimal of a GiB.
const PREDICTION_SLACK_BYTES: u64 = 1 << 30;
/// A render is a picture, not a flat field: pixel standard deviation floor (the candle twin's
/// installed-tier floor).
const NON_DEGENERATE_STD: f64 = 8.0;

// ── geometry ─────────────────────────────────────────────────────────────────────────────────────

const RENDER_EDGE: u32 = 768;
const RENDER_STEPS: u32 = 8;
const SEED: u64 = 24163;
const TRAIN_EDGE: u32 = 512;
/// Learning rate of both training runs: the product presets' rate for every Qwen-Image 2.1 target
/// (`qwen_image_2_1_lora.*` / `qwen_image_2_1_edit_lora.*` balanced = 1e-4). The first real-weight
/// run (inference run 37122808826) trained the T2I LoRA at 1e-3 and DIVERGED within five AdamW
/// updates — loss 0.26 / 0.06 / 0.23 at steps 1-3, then 4.77 and 8.02 at steps 6-7, then a flat
/// ~1.7-1.9 (the "predict nothing" level of a flow-matching velocity) to step 100 — and its adapter
/// rendered the same white field with green blobs at every tier (pixel std 35.7 vs the base's 77.4,
/// mean |Δ| 134/255, palette fraction 0.00003). The evidence trains at the rate a user gets.
const TRAIN_LR: f32 = 1e-4;
const T2I_ADAPTER: &str = "qwen21_t2i_lora.safetensors";
const EDIT_ADAPTER: &str = "qwen21_edit_lokr.safetensors";

#[path = "support/edit_protocol.rs"]
pub(crate) mod edit_protocol;
#[path = "support/edit_training_balanced64.rs"]
mod edit_training_balanced64;
#[path = "support/style_protocol.rs"]
mod style_protocol;
use edit_protocol::{EDIT_INSTRUCTION, T2I_EVAL_PROMPT, TRAIN_EDIT_INSTRUCTION};

/// The T2I training style: concentric rings in exactly these three colours.
const PALETTE: [[u8; 3]; 3] = style_protocol::PALETTE;
/// RGB distance under which a pixel counts as "a palette colour".
const PALETTE_RADIUS: f64 = 60.0;

// ── environment ──────────────────────────────────────────────────────────────────────────────────

fn required_dir(var: &str, what: &str) -> PathBuf {
    let path = PathBuf::from(std::env::var(var).unwrap_or_else(|_| {
        panic!("set {var} to {what}; inference never self-fetches (epic 13657)")
    }));
    assert!(path.is_dir(), "{var}={} is not a directory", path.display());
    path
}

fn snapshot() -> PathBuf {
    required_dir(
        "MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT",
        "the pinned Qwen/Qwen-Image-2.1 snapshot dir",
    )
}

fn tier_snapshot() -> PathBuf {
    required_dir(
        "MLX_GEN_QWEN_IMAGE_2_1_TIER_SNAPSHOT",
        "the pinned SceneWorks/qwen-image-2-1-mlx snapshot dir (holding q8/ and q4/)",
    )
}

pub(crate) fn out_dir() -> PathBuf {
    let dir = std::env::var("QWEN_IMAGE_2_1_RENDER_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn env_u32(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .map(|v| {
            v.parse()
                .unwrap_or_else(|_| panic!("{var}={v} is not a u32"))
        })
        .unwrap_or(default)
}

fn training_steps(var: &str, default: u32) -> u32 {
    let probe = std::env::var("QWEN_IMAGE_2_1_PROBE_ONLY").as_deref() == Ok("1");
    let steps = env_u32(var, if probe { 2 } else { default });
    assert!(
        steps > 0 && (!probe || steps <= 3),
        "{var}: a probe requires 1–3 steps"
    );
    steps
}

#[test]
fn edit_instruction_and_target_use_independent_rgb_levels() {
    let source = image::RgbImage::from_pixel(1, 1, image::Rgb([0, 85, 170]));
    assert_eq!(edit_transform(&source).get_pixel(0, 0).0, [255, 170, 85]);
    assert!(EDIT_INSTRUCTION.contains("independently"));
    assert!(EDIT_INSTRUCTION.contains("keep the result in colour"));
    assert_eq!(
        TRAIN_EDIT_INSTRUCTION,
        EDIT_INSTRUCTION
            .strip_suffix(edit_protocol::PALETTE_ROLE)
            .unwrap()
    );
}

/// The three render tiers: the dense bf16 base, and the published packed q8 / q4 snapshots.
fn tiers() -> [(&'static str, Option<Quant>); 3] {
    [
        ("bf16", None),
        ("q8", Some(Quant::Q8)),
        ("q4", Some(Quant::Q4)),
    ]
}

pub(crate) fn tier_spec(label: &str, quant: Option<Quant>) -> LoadSpec {
    match quant {
        None => LoadSpec::new(WeightsSource::Dir(snapshot())),
        Some(quant) => {
            LoadSpec::new(WeightsSource::Dir(tier_snapshot().join(label))).with_quant(quant)
        }
    }
}

// ── memory: footprint guard + per-phase peaks ───────────────────────────────────────────────────

/// A 50 ms `phys_footprint` sampler that keeps a resettable phase high-water mark and aborts the
/// process above the ceiling (writing a marker into the evidence dir first).
#[path = "support/physical_watchdog.rs"]
pub(crate) mod physical_watchdog;

pub(crate) fn host_census() -> (physical_watchdog::Host, Value) {
    use std::process::Command;
    let command = |program: &str, args: &[&str]| {
        let result = Command::new(program)
            .args(args)
            .output()
            .expect("host census command starts");
        assert!(
            result.status.success(),
            "host census {program} failed: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        String::from_utf8(result.stdout).expect("host census is UTF-8")
    };
    let total = command("sysctl", &["-n", "hw.memsize"])
        .trim()
        .parse::<u64>()
        .unwrap();
    let vm = command("vm_stat", &[]);
    let pressure = command("sysctl", &["-n", "kern.memorystatus_vm_pressure_level"])
        .trim()
        .parse::<u64>()
        .unwrap();
    let pressure_query = command("memory_pressure", &["-Q"]);
    // Once on the main thread before weights, matching e2e_real_weights::memory_line.
    // Never read limits by changing them from the background sampler.
    let cache_limit = mlx_rs::memory::set_cache_limit(0);
    mlx_rs::memory::set_cache_limit(cache_limit);
    let host = physical_watchdog::Host {
        total,
        available: physical_watchdog::reclaimable_bytes(&vm).unwrap(),
        pressure,
        recommended: mlx_gen::memory::recommended_working_set_bytes().unwrap_or(0),
        mlx_limit: mlx_rs::memory::get_memory_limit() as u64,
        baseline_physical: phys_footprint().0,
        baseline_active: mlx_rs::memory::get_active_memory() as u64,
        baseline_cache: mlx_rs::memory::get_cache_memory() as u64,
        cache_limit: cache_limit as u64,
    };
    let receipt = json!({"totalBytes": host.total, "reclaimableBytes": host.available,
        "recommendedWorkingSetBytes": host.recommended, "mlxMemoryLimitBytes": host.mlx_limit,
        "initialCacheLimitBytes": host.cache_limit, "pressureLevel": pressure,
        "baselinePhysBytes": host.baseline_physical, "baselineActiveBytes": host.baseline_active,
        "baselineCacheBytes": host.baseline_cache, "vmStat": vm, "memoryPressureQuery": pressure_query,
        "physicalSamplerScope": "background_physical_and_host_only_no_MLX_calls",
        "allocatorAttribution": "foreground_non_atomic_stage_snapshots_joined_by_unixMillis_no_added_eval_or_sync"});
    (host, receipt)
}

pub(crate) struct Footprint {
    phase_max: Arc<AtomicU64>,
    ceiling: Arc<AtomicU64>,
    explicit_cap: Option<u64>,
    out: PathBuf,
    stop: Option<std::sync::mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Footprint {
    pub(crate) fn start(out: &Path) -> Self {
        use std::io::Write;
        let explicit_cap = std::env::var("QWEN_IMAGE_2_1_FOOTPRINT_CEILING_GB")
            .ok()
            .map(|v| {
                let gb: f64 = v
                    .parse()
                    .expect("physical ceiling is a finite positive number");
                assert!(
                    gb.is_finite() && gb > 0.0,
                    "physical ceiling must be positive and finite"
                );
                (gb * 1e9) as u64
            });
        let (_, census) = host_census();
        std::fs::write(
            out.join("host-memory-census.json"),
            serde_json::to_vec_pretty(&census).unwrap(),
        )
        .unwrap();
        let ceiling = Arc::new(AtomicU64::new(explicit_cap.unwrap_or(100_000_000_000)));
        let phase_max = Arc::new(AtomicU64::new(0));
        let shared = phase_max.clone();
        let shared_ceiling = ceiling.clone();
        let directory = out.to_path_buf();
        let (stop, rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            // A failed sampler must stop the GPU process, not silently leave it unguarded.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut file =
                    std::fs::File::create(directory.join("physical-allocator-samples.jsonl"))
                        .unwrap();
                let mut ticks = 0u64;
                let mut pressure = census["pressureLevel"].as_u64().unwrap();
                let mut available = census["reclaimableBytes"].as_u64().unwrap();
                loop {
                    // No MLX calls on this thread: upstream counters are plain C++ scalars.
                    // Foreground stage snapshots align by timestamp with this physical trace.
                    let (fp, lifetime) = phys_footprint();
                    shared.fetch_max(fp, Ordering::Relaxed);
                    if ticks.is_multiple_of(20) {
                        let read = |program: &str, args: &[&str]| {
                            std::process::Command::new(program)
                                .args(args)
                                .output()
                                .ok()
                                .filter(|v| v.status.success())
                                .and_then(|v| String::from_utf8(v.stdout).ok())
                        };
                        pressure = read("sysctl", &["-n", "kern.memorystatus_vm_pressure_level"])
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                        available = read("vm_stat", &[])
                            .and_then(|v| physical_watchdog::reclaimable_bytes(&v).ok())
                            .unwrap_or(0);
                    }
                    ticks += 1;
                    let ceiling = shared_ceiling.load(Ordering::Relaxed);
                    let sample = json!({
                        "unixMillis": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis(),
                        "physFootprintBytes": fp, "lifetimePhysPeakBytes": lifetime,
                        "physicalCeilingBytes": ceiling, "pressureLevel": pressure,
                        "reclaimableBytes": available,
                    });
                    writeln!(file, "{sample}").unwrap();
                    file.flush().unwrap();
                    if fp > ceiling || pressure != 1 || available == 0 {
                        let message = format!("physical watchdog aborted: footprint={fp} bytes ceiling={ceiling} bytes pressure={pressure} available={available} bytes\n");
                        eprint!("{message}");
                        let _ = std::fs::write(
                            directory.join("FOOTPRINT_CEILING_EXCEEDED.txt"),
                            message,
                        );
                        let _ = std::fs::write(
                            directory.join("physical-watchdog-abort.json"),
                            serde_json::to_vec_pretty(&sample).unwrap(),
                        );
                        std::process::abort();
                    }
                    match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                        _ => break,
                    }
                }
            }));
            if result.is_err() {
                let message = "physical watchdog sampler failed; refusing unguarded GPU work\n";
                eprint!("{message}");
                let _ = std::fs::write(directory.join("MEMORY_SAMPLER_FAILED.txt"), message);
                std::process::abort();
            }
        });
        Self {
            phase_max,
            ceiling,
            explicit_cap,
            out: out.to_path_buf(),
            stop: Some(stop),
            handle: Some(handle),
        }
    }

    fn admit_training(&self, preflight: &Value) {
        let (mut host, census) = host_census();
        host.cache_limit = host
            .cache_limit
            .min(preflight["requestedCacheLimitBytes"].as_u64().unwrap());
        let envelope = preflight["peakBytes"].as_u64().unwrap();
        let result = physical_watchdog::admit(host, envelope, self.explicit_cap);
        let receipt = json!({"host": census, "preflight": preflight,
            "requestedPoolBoundBytes": host.cache_limit, "physicalCeilingBytes": result.as_ref().ok(),
            "refusal": result.as_ref().err(), "policy": "preflight_plus_measured_overhead_and_cache_constrained_by_available_RAM_recommended_MLX_and_explicit_cap"});
        std::fs::write(
            self.out.join("physical-admission.json"),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
        let ceiling =
            result.expect("selected training case cannot safely fit; see physical-admission.json");
        self.ceiling.store(ceiling, Ordering::Relaxed);
        eprintln!("admitted training: preflight={envelope} physicalCeiling={ceiling} bytes");
    }

    #[allow(dead_code)] // Used only by the cfg(test) library diagnostic, not this integration binary.
    pub(crate) fn admit_numeric(&self, active_envelope: u64, free_cache: u64) {
        let (host, census) = host_census();
        // Reserve the entire frozen allowance even if the allocator was already
        // capped lower. This estimate never modifies the actual allocator policy.
        let cap = self.explicit_cap.unwrap_or(100_000_000_000);
        let result =
            physical_watchdog::admit_numeric_full(host, active_envelope, free_cache, Some(cap));
        let receipt = json!({"kind": "DIAGNOSTIC_ONLY", "host": census,
            "activeEnvelopeBytes": active_envelope, "freeCacheAllowanceBytes": free_cache,
            "actualAllocatorCacheLimitBytes": host.cache_limit,
            "explicitOrDefaultCapBytes": cap, "physicalCeilingBytes": result.as_ref().ok(),
            "refusal": result.as_ref().err(), "reservesUnchanged": true});
        write_json(&self.out, "numeric-physical-admission", &receipt);
        let ceiling = result.expect("numeric diagnostic cannot safely fit; receipt retained");
        self.ceiling.store(ceiling, Ordering::Relaxed);
    }

    pub(crate) fn begin(&self) {
        mlx_rs::memory::reset_peak_memory();
        self.phase_max.store(0, Ordering::Relaxed);
    }

    pub(crate) fn end(&self) -> (u64, u64) {
        let (fp, _) = phys_footprint();
        (
            mlx_rs::memory::get_peak_memory() as u64,
            self.phase_max.load(Ordering::Relaxed).max(fp),
        )
    }
}

impl Drop for Footprint {
    fn drop(&mut self) {
        drop(self.stop.take());
        if let Some(handle) = self.handle.take() {
            handle.join().expect("memory sampler finishes");
        }
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

// ── image metrics ────────────────────────────────────────────────────────────────────────────────

pub(crate) fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    assert_eq!((a.width, a.height), (b.width, b.height), "geometry differs");
    assert_eq!(a.pixels.len(), b.pixels.len(), "byte length differs");
    let sum: u64 = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| u64::from(x.abs_diff(y)))
        .sum();
    sum as f64 / a.pixels.len().max(1) as f64
}

fn pixel_std(img: &Image) -> f64 {
    let n = img.pixels.len().max(1) as f64;
    let mean = img.pixels.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    (img.pixels
        .iter()
        .map(|&v| (f64::from(v) - mean).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

/// Fraction of rows whose luma second difference reads as uninitialised memory (the candle twin's
/// band-corruption detector, sc-24114): a rendered scene is ~3, stale device memory 20–35.
fn static_row_fraction(img: &Image) -> f64 {
    let w = img.width as usize;
    let luma =
        |px: &[u8]| 0.299 * f32::from(px[0]) + 0.587 * f32::from(px[1]) + 0.114 * f32::from(px[2]);
    let rows = img
        .pixels
        .chunks_exact(w * 3)
        .filter(|row| {
            let l: Vec<f32> = row.chunks_exact(3).map(luma).collect();
            let hf = l
                .windows(3)
                .map(|t| (t[2] - 2.0 * t[1] + t[0]).abs())
                .sum::<f32>()
                / (w - 2) as f32;
            hf > 20.0
        })
        .count();
    rows as f64 / img.height.max(1) as f64
}

/// Mean RGB distance from each pixel to its nearest [`PALETTE`] colour — the continuous twin of
/// [`palette_fraction`]: it moves with every partial shift toward the style, not only once a pixel
/// lands inside [`PALETTE_RADIUS`].
fn palette_distance(img: &Image) -> f64 {
    style_protocol::palette_distance(&img.pixels)
}

fn palette_fraction(img: &Image) -> f64 {
    let near = img
        .pixels
        .chunks_exact(3)
        .filter(|px| {
            PALETTE.iter().any(|c| {
                let d: f64 = px
                    .iter()
                    .zip(c)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum();
                d.sqrt() <= PALETTE_RADIUS
            })
        })
        .count();
    near as f64 / (img.pixels.len() / 3).max(1) as f64
}

/// The sanity facts every render reports, and the two that must hold for it to count as a
/// picture at all (checked by [`assert_sane`], after the evidence is written).
fn image_facts(img: &Image) -> Value {
    json!({
        "width": img.width,
        "height": img.height,
        "pixelStd": pixel_std(img),
        "staticRowFraction": static_row_fraction(img),
    })
}

fn assert_sane(label: &str, img: &Image) {
    let std = pixel_std(img);
    assert!(
        std > NON_DEGENERATE_STD,
        "{label}: a near-flat field (pixel std {std:.2}) — a packed site read as dense, or an \
         adapter that blew the forward up"
    );
    let static_rows = static_row_fraction(img);
    assert!(
        static_rows <= 0.25,
        "{label}: {:.1}% of rows are noise — the render is band-corrupted",
        100.0 * static_rows
    );
}

fn save_png(path: &Path, img: &Image) {
    image::save_buffer(
        path,
        &img.pixels,
        img.width,
        img.height,
        image::ColorType::Rgb8,
    )
    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

pub(crate) fn write_json(out: &Path, name: &str, value: &Value) {
    let path = out.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(value).unwrap()).unwrap();
    eprintln!("wrote {}", path.display());
}

pub(crate) fn sha256_file(path: &Path) -> String {
    let mut file = std::fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer).unwrap();
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    format!("{:x}", hash.finalize())
}

fn dataset_receipt(items: &[TrainingItem]) -> Value {
    let file = |path: &Path| {
        json!({
            "file": path.file_name().unwrap().to_string_lossy(),
            "bytes": std::fs::metadata(path).unwrap().len(), "sha256": sha256_file(path),
        })
    };
    let rows: Vec<Value> = items.iter().enumerate().map(|(index, item)| json!({
        "index": index, "target": file(&item.image_path), "caption": item.caption,
        "referenceCount": item.reference_image_paths.len(),
        "orderedReferences": item.reference_image_paths.iter().map(|path| file(path)).collect::<Vec<_>>(),
    })).collect();
    // Hash binds item/reference order, captions and exact source bytes, independent of host paths.
    let hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&rows).unwrap()));
    json!({"sha256": hash, "hashSchema": "sha256(serde_json ordered rows with file byte hashes)", "items": rows})
}

#[test]
fn dataset_hash_binds_caption_bytes_and_reference_order() {
    let guard = tempfile::tempdir().unwrap();
    let dir = guard.path();
    let paths: Vec<_> = ["target.png", "source.png", "key.png"]
        .iter()
        .map(|name| dir.join(name))
        .collect();
    for (index, path) in paths.iter().enumerate() {
        std::fs::write(path, [index as u8]).unwrap();
    }
    let mut items = vec![TrainingItem::edit_pair(
        paths[0].clone(),
        EDIT_INSTRUCTION.into(),
        paths[1..].to_vec(),
    )];
    let original = dataset_receipt(&items);
    assert_eq!(original, dataset_receipt(&items));
    items[0].reference_image_paths.reverse();
    assert_ne!(original["sha256"], dataset_receipt(&items)["sha256"]);
    items[0].reference_image_paths.reverse();
    items[0].caption = TRAIN_EDIT_INSTRUCTION.into();
    assert_ne!(original["sha256"], dataset_receipt(&items)["sha256"]);
    items[0].caption = EDIT_INSTRUCTION.into();
    std::fs::write(&paths[1], [99]).unwrap();
    assert_ne!(original["sha256"], dataset_receipt(&items)["sha256"]);
}

// ── synthetic datasets (deterministic; no fetched or checked-in photographs) ─────────────────────

/// A tiny LCG so every synthetic image is reproducible bit for bit.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

pub(crate) fn to_image(img: image::RgbImage) -> Image {
    Image {
        width: img.width(),
        height: img.height(),
        pixels: img.into_raw(),
    }
}

/// One T2I style image: concentric rings in [`PALETTE`] around a per-image centre and width.
fn ring_image(index: u32, edge: u32) -> image::RgbImage {
    let (cx, cy) = (
        (edge / 4 + (index * 97) % (edge / 2)) as f32,
        (edge / 4 + (index * 61) % (edge / 2)) as f32,
    );
    let width = (14 + 5 * index) as f32;
    image::RgbImage::from_fn(edge, edge, |x, y| {
        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
        image::Rgb(PALETTE[((d / width) as usize + index as usize) % PALETTE.len()])
    })
}

/// One edit source: a vertical gradient with five seeded discs of seeded colours.
pub(crate) fn edit_source(seed: u64, edge: u32) -> image::RgbImage {
    let mut rng = Lcg(seed);
    let discs: Vec<(f32, f32, f32, [u8; 3])> = (0..5)
        .map(|_| {
            (
                (rng.next() % edge) as f32,
                (rng.next() % edge) as f32,
                (edge / 10 + rng.next() % (edge / 5)) as f32,
                [
                    (rng.next() % 256) as u8,
                    (rng.next() % 256) as u8,
                    (rng.next() % 256) as u8,
                ],
            )
        })
        .collect();
    let base = [
        (rng.next() % 256) as u8,
        (rng.next() % 256) as u8,
        (rng.next() % 256) as u8,
    ];
    image::RgbImage::from_fn(edge, edge, |x, y| {
        for &(cx, cy, r, c) in &discs {
            if (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) <= r * r {
                return image::Rgb(c);
            }
        }
        let t = y as f32 / edge as f32;
        image::Rgb(base.map(|v| (f32::from(v) * (1.0 - 0.6 * t)) as u8))
    })
}

/// Reference 2 is a color palette, not a competing monochrome layout.
pub(crate) fn edit_key(edge: u32) -> image::RgbImage {
    image::RgbImage::from_fn(edge, edge, |x, y| {
        image::Rgb(edit_protocol::palette_pixel(x, y, edge))
    })
}

/// The trained edit: invert, then posterize every channel to the key's four levels.
pub(crate) fn edit_transform(src: &image::RgbImage) -> image::RgbImage {
    let mut out = src.clone();
    for px in out.pixels_mut() {
        for v in px.0.iter_mut() {
            *v = edit_protocol::transformed_channel(*v);
        }
    }
    out
}

// ── shared run helpers ───────────────────────────────────────────────────────────────────────────

/// Load `spec` through the explicit catalog, render `req` once, write `<out>/<label>.png`, and
/// return the image plus its timing / memory / predicted-overlay facts.
pub(crate) fn render(
    label: &str,
    spec: &LoadSpec,
    req: &GenerationRequest,
    guard: &Footprint,
    out: &Path,
) -> (Image, Value) {
    let contract = memory_strategy_contract(ID, spec)
        .unwrap_or_else(|e| panic!("{label}: the memory contract refused the spec: {e}"));
    let predicted_overlay = contract.asset_facts.overlay_bytes;
    let predicted_resident = contract.total_resident_bytes();
    guard.begin();
    let started = Instant::now();
    let generator = provider_registry()
        .unwrap()
        .load(ID, spec)
        .unwrap_or_else(|e| panic!("{label}: load failed: {e}"));
    let loaded = started.elapsed().as_secs_f64();
    let output = generator
        .generate(req, &mut |_| {})
        .unwrap_or_else(|e| panic!("{label}: render failed: {e}"));
    let seconds = started.elapsed().as_secs_f64();
    drop(generator);
    let (mlx_peak, footprint_max) = guard.end();
    mlx_rs::memory::clear_cache();
    let GenerationOutput::Images(mut images) = output else {
        panic!("{label}: images expected");
    };
    let image = images.remove(0);
    assert_eq!(
        (image.width, image.height),
        (req.width, req.height),
        "{label}"
    );
    save_png(&out.join(format!("{label}.png")), &image);
    eprintln!(
        "{label}: {seconds:.1}s (load {loaded:.1}s) mlx_peak={:.2} GiB footprint_max={:.2} GiB \
         predicted_overlay={:.2} GiB",
        gib(mlx_peak),
        gib(footprint_max),
        gib(predicted_overlay)
    );
    let facts = json!({
        "label": label,
        "png": format!("{label}.png"),
        "seconds": seconds,
        "loadSeconds": loaded,
        "mlxActivePeakBytes": mlx_peak,
        "physFootprintMaxBytes": footprint_max,
        "predictedOverlayBytes": predicted_overlay,
        "predictedResidentBytes": predicted_resident,
        "image": image_facts(&image),
    });
    (image, facts)
}

pub(crate) fn adapter(path: &Path, scale: f32, kind: AdapterKind) -> AdapterSpec {
    AdapterSpec::new(path.to_path_buf(), scale, kind)
}

/// The crate's own derived training footprint for `req`, read off the preflight refusal a 1 KiB
/// budget forces (no weight is read before the preflight). Returns the message and the parsed
/// `train ~X GiB` figure.
fn predicted_training_footprint(req: &TrainingRequest) -> (String, Option<f64>) {
    let mut probe =
        QwenImage21Trainer::load(&LoadSpec::new(WeightsSource::Dir(snapshot()))).unwrap();
    probe.set_memory_budget_override(Some(1 << 10));
    let message = match probe.train(req, &mut |_| {}) {
        Ok(_) => panic!("a 1 KiB training budget must be refused at the preflight"),
        Err(e) => e.to_string(),
    };
    let train_gib = message
        .split("train ~")
        .nth(1)
        .and_then(|rest| rest.split(" GiB").next())
        .and_then(|v| v.trim().parse::<f64>().ok());
    (message, train_gib)
}

/// What one training run measured.
struct Trained {
    adapter: PathBuf,
    facts: Value,
    /// Observed MLX active peak of the step phase, and the preflight's predicted train phase.
    train_peak: u64,
    predicted_train_bytes: u64,
    predicted_peak_bytes: u64,
    remaining_steps_peak: u64,
}

/// Run `req` on a fresh trainer through the registry, recording the caching-phase and the
/// subsequent step peaks. The first observation overlaps caching and step 1 because the trainer
/// reports progress after each completed step. Retain and price that conservative overlap against
/// the full preflight envelope, and subsequent steps against its train phase. Copy
/// the adapter to `canonical` if the trainer wrote it elsewhere.
fn train(req: &TrainingRequest, guard: &Footprint, canonical: &Path, log_every: u32) -> Trained {
    // The tests run serially. Scope the explicit trace output to this evidence directory and
    // restore the caller's environment even on an unwinding assertion.
    struct DiagnosticOutput(Option<std::ffi::OsString>);
    impl Drop for DiagnosticOutput {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                std::env::set_var("QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS_OUT", previous);
            } else {
                std::env::remove_var("QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS_OUT");
            }
        }
    }
    let _diagnostics =
        DiagnosticOutput(std::env::var_os("QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS_OUT"));
    std::env::set_var("QWEN_IMAGE_2_1_TRAINING_DIAGNOSTICS_OUT", &guard.out);
    for name in ["training-preflight.json", "training-stages.jsonl"] {
        let previous = guard.out.join(name);
        if previous.exists() {
            std::fs::remove_file(previous).expect("remove previous owned diagnostic receipt");
        }
    }
    let (predicted_message, predicted_train_gib) = predicted_training_footprint(req);
    let preflight: Value = serde_json::from_slice(
        &std::fs::read(guard.out.join("training-preflight.json")).expect("exact preflight receipt"),
    )
    .expect("valid exact preflight receipt");
    let predicted_train_bytes = preflight["trainBytes"].as_u64().expect("exact train bytes");
    let predicted_peak_bytes = preflight["peakBytes"]
        .as_u64()
        .expect("exact envelope bytes");
    guard.admit_training(&preflight);
    let predicted_peak_gib = predicted_message
        .split("derived peak memory is ~")
        .nth(1)
        .and_then(|rest| rest.split(" GiB").next())
        .and_then(|v| v.parse::<f64>().ok());
    eprintln!("preflight (forced 1 KiB budget): {predicted_message}");

    let mut trainer = provider_registry()
        .unwrap()
        .load_trainer(TRAINER_ID, &LoadSpec::new(WeightsSource::Dir(snapshot())))
        .expect("the bf16 snapshot loads as a trainer");
    trainer
        .validate(req)
        .unwrap_or_else(|e| panic!("validate refused the run: {e}"));
    guard.begin();
    let started = Instant::now();
    let mut losses: Vec<f32> = Vec::new();
    let mut caching_peaks: Option<(u64, u64)> = None;
    // MLX active memory right after step 1 returns: the run's resident set (DiT + adapter and
    // optimizer state + caches) with no step in flight. `train_peak − resident` is the step's own
    // transient, so an under-prediction splits into "resident" vs "working set".
    let mut resident_after_first_step: Option<u64> = None;
    let mut samples: Vec<Value> = Vec::new();
    let output = trainer
        .train(req, &mut |p| {
            let memory = || {
                (
                    mlx_rs::memory::get_active_memory() as u64,
                    mlx_rs::memory::get_cache_memory() as u64,
                    mlx_rs::memory::get_peak_memory() as u64,
                    phys_footprint().0,
                )
            };
            match p {
                TrainingProgress::Training { step, loss, .. } => {
                    let first_observation = caching_peaks.is_none();
                    if first_observation {
                        // First step reported: everything before it was the staged caching (and
                        // step 1 itself).
                        caching_peaks = Some(guard.end());
                        resident_after_first_step =
                            Some(mlx_rs::memory::get_active_memory() as u64);
                    }
                    losses.push(loss);
                    if first_observation || step % log_every == 0 {
                        let (active, cache, peak, footprint) = memory();
                        eprintln!(
                            "step {step}: loss {loss:.4} ({:.0}s) active={:.2} cache={:.2} \
                             phase_peak={:.2} footprint={:.2} GiB",
                            started.elapsed().as_secs_f64(),
                            gib(active),
                            gib(cache),
                            gib(peak),
                            gib(footprint)
                        );
                        samples.push(json!({
                            "step": step,
                            "loss": loss,
                            "seconds": started.elapsed().as_secs_f64(),
                            "activeBytes": active,
                            "cacheBytes": cache,
                            "phasePeakBytes": peak,
                            "physFootprintBytes": footprint,
                        }));
                    }
                    if first_observation {
                        // Reset only AFTER sampling the completed first step's high-water mark.
                        guard.begin();
                    }
                }
                TrainingProgress::Preparing
                | TrainingProgress::LoadingModel
                | TrainingProgress::Caching { .. } => {
                    let (active, cache, peak, footprint) = memory();
                    eprintln!(
                        "{p:?} ({:.0}s) active={:.2} cache={:.2} peak={:.2} footprint={:.2} GiB",
                        started.elapsed().as_secs_f64(),
                        gib(active),
                        gib(cache),
                        gib(peak),
                        gib(footprint)
                    );
                }
                _ => {}
            }
        })
        .unwrap_or_else(|e| panic!("training failed: {e}"));
    let seconds = started.elapsed().as_secs_f64();
    let remaining_steps = guard.end();
    drop(trainer);
    mlx_rs::memory::clear_cache();
    let (caching_peak, caching_footprint) = caching_peaks.unwrap_or((0, 0));
    let (train_peak, train_footprint) = training_peaks::aggregate_training_peaks(
        (caching_peak, caching_footprint),
        remaining_steps,
    );
    let resident = resident_after_first_step.unwrap_or(0);
    eprintln!(
        "trained in {seconds:.0}s: conservative cache/step peak {:.2} GiB (resident after step 1 {:.2} + \
         high-water excess {:.2}) vs preflight train ~{} GiB",
        gib(train_peak),
        gib(resident),
        gib(train_peak.saturating_sub(resident)),
        predicted_train_gib.map_or("?".into(), |g| format!("{g:.1}"))
    );

    if output.adapter_path != canonical {
        std::fs::copy(&output.adapter_path, canonical).unwrap_or_else(|e| {
            panic!(
                "copy {} -> {}: {e}",
                output.adapter_path.display(),
                canonical.display()
            )
        });
    }
    let metadata = safetensors_file_metadata(canonical).unwrap();
    let cfg = &req.config;
    let stage_names = std::fs::read_to_string(guard.out.join("training-stages.jsonl"))
        .ok()
        .and_then(|text| {
            text.lines()
                .map(serde_json::from_str::<Value>)
                .collect::<Result<Vec<_>, _>>()
                .ok()
        })
        .unwrap_or_default()
        .into_iter()
        .filter_map(|event| event["stage"].as_str().map(str::to_owned))
        .collect::<std::collections::BTreeSet<_>>();
    let stage_trace_complete = [
        "pool_bound",
        "caption_cache_cleared",
        "latent_cache_cleared",
        "dit_loaded",
    ]
    .iter()
    .all(|stage| stage_names.contains(*stage))
        && (1..=output.steps).all(|step| {
            ["begin", "gradients_returned", "before_progress"]
                .iter()
                .all(|stage| stage_names.contains(&format!("step_{step}_{stage}")))
        });
    let facts = json!({
        "adapter": canonical.file_name().unwrap().to_string_lossy(),
        "adapterBytes": std::fs::metadata(canonical).unwrap().len(),
        "adapterSha256": sha256_file(canonical),
        "dataset": dataset_receipt(&req.items),
        "networkType": format!("{:?}", cfg.network_type),
        "rank": cfg.rank,
        "learningRate": cfg.learning_rate,
        "steps": cfg.steps,
        "stepsRun": output.steps,
        "resolution": cfg.resolution,
        "gradientCheckpointing": cfg.gradient_checkpointing,
        "items": req.items.len(),
        "finalLoss": output.final_loss,
        "losses": losses,
        "seconds": seconds,
        "cachingMlxActivePeakBytes": caching_peak,
        "cachingPhysFootprintMaxBytes": caching_footprint,
        "trainMlxActivePeakBytes": train_peak,
        "trainPhysFootprintMaxBytes": train_footprint,
        "trainMeasurementScope": "caching_and_first_step_overlap_then_remaining_steps",
        "remainingStepsMlxActivePeakBytes": remaining_steps.0,
        "remainingStepsPhysFootprintMaxBytes": remaining_steps.1,
        "residentAfterFirstStepBytes": resident,
        "stepTransientPeakBytes": train_peak.saturating_sub(resident),
        "stepSamples": samples,
        "trainingStageTraceComplete": stage_trace_complete,
        "predictedPreflight": predicted_message,
        "predictedTrainPhaseGiB": predicted_train_gib,
        "predictedFullEnvelopeGiB": predicted_peak_gib,
        "predictedTrainPhaseBytes": predicted_train_bytes,
        "predictedFullEnvelopeBytes": predicted_peak_bytes,
        "physicalWatchdogCeilingBytes": guard.ceiling.load(Ordering::Relaxed),
        "metadata": metadata,
    });
    Trained {
        adapter: canonical.to_path_buf(),
        facts,
        train_peak,
        predicted_train_bytes,
        predicted_peak_bytes,
        remaining_steps_peak: remaining_steps.0,
    }
}

/// The assertions every trained run must satisfy, checked after its evidence is on disk.
fn assert_trained(label: &str, trained: &Trained, steps: u32, edit: bool) {
    let facts = &trained.facts;
    assert_eq!(facts["stepsRun"], json!(steps), "{label}: steps run");
    assert_eq!(
        facts["trainingStageTraceComplete"],
        json!(true),
        "{label}: cache, DiT and every completed step require a valid diagnostic trace"
    );
    let losses = facts["losses"].as_array().unwrap();
    assert_eq!(losses.len(), steps as usize, "{label}: one loss per step");
    assert!(
        losses
            .iter()
            .all(|l| l.as_f64().is_some_and(f64::is_finite)),
        "{label}: a non-finite loss"
    );
    let meta = &facts["metadata"];
    assert_eq!(meta["family"], json!("qwen-image-2-1"), "{label}: {meta}");
    assert_eq!(
        meta["baseModel"],
        json!("qwen_image_2_1"),
        "{label}: {meta}"
    );
    assert!(
        meta["license"]
            .as_str()
            .is_some_and(|l| l.contains("Qwen Research License")),
        "{label}: {meta}"
    );
    if edit {
        assert_eq!(meta["trainingMode"], json!("edit"), "{label}: {meta}");
    } else {
        assert!(meta.get("trainingMode").is_none(), "{label}: {meta}");
    }
}

/// The preflight must cover all observations: the cache/first-step overlap uses the full envelope,
/// and later steps use its training phase. Checked LAST in each training test, after every
/// render-side assertion, so a red run still exercises (and reports) the adapter's own checks.
fn assert_preflight_covers_step(label: &str, trained: &Trained) {
    assert!(
        trained.remaining_steps_peak <= trained.predicted_train_bytes + PREDICTION_SLACK_BYTES,
        "{label}: remaining training steps exceed their exact derived train phase"
    );
    assert!(
        trained.train_peak <= trained.predicted_peak_bytes + PREDICTION_SLACK_BYTES,
        "{label}: caching/first-step plus remaining steps peaked at {} bytes, over the exact full \
         envelope {} bytes (+{} bytes historical slack) — the training footprint under-predicts",
        trained.train_peak,
        trained.predicted_peak_bytes,
        PREDICTION_SLACK_BYTES
    );
}

/// The assertion every adapted render must satisfy against its own predicted overlay: the observed
/// active peak may exceed the bare render's by no more than the overlay the contract priced.
fn assert_overlay_not_underpredicted(label: &str, base: &Value, adapted: &Value) {
    let base_peak = base["mlxActivePeakBytes"].as_u64().unwrap();
    let adapted_peak = adapted["mlxActivePeakBytes"].as_u64().unwrap();
    let predicted = adapted["predictedOverlayBytes"].as_u64().unwrap();
    assert!(
        adapted_peak <= base_peak + predicted + PREDICTION_SLACK_BYTES,
        "{label}: the adapted render peaked {:.2} GiB above the bare one, over the predicted \
         overlay {:.2} GiB (+{:.0} GiB slack)",
        gib(adapted_peak.saturating_sub(base_peak)),
        gib(predicted),
        gib(PREDICTION_SLACK_BYTES)
    );
}

pub(crate) fn t2i_request() -> GenerationRequest {
    GenerationRequest {
        prompt: T2I_EVAL_PROMPT.to_owned(),
        width: RENDER_EDGE,
        height: RENDER_EDGE,
        steps: Some(RENDER_STEPS),
        seed: Some(SEED),
        ..Default::default()
    }
}

fn original_style_request() -> GenerationRequest {
    GenerationRequest {
        prompt: style_protocol::ORIGINAL_STYLE_PROMPT.to_owned(),
        ..t2i_request()
    }
}

fn direction_request_facts(request: &GenerationRequest) -> Value {
    json!({"prompt": request.prompt, "width": request.width, "height": request.height,
        "steps": request.steps, "seed": request.seed, "guidance": request.guidance,
        "strength": request.strength, "count": request.count,
        "negativePrompt": request.negative_prompt, "trueCfg": request.true_cfg,
        "conditioningCount": request.conditioning.len()})
}

fn verify_original_style_donor(entry: &Value, file: &Path) {
    assert_eq!(entry["name"], style_protocol::DONOR_NAME);
    assert_eq!(entry["kind"], "lora");
    assert_eq!(entry["origin"], "MLX run 37127726908");
    assert_eq!(entry["sha256"], style_protocol::DONOR_SHA256);
    assert_eq!(entry["size"], style_protocol::DONOR_BYTES);
    assert_eq!(
        std::fs::metadata(file).unwrap().len(),
        style_protocol::DONOR_BYTES
    );
    assert_eq!(sha256_file(file), style_protocol::DONOR_SHA256);
}

// ── 1. text-to-image LoRA ────────────────────────────────────────────────────────────────────────

/// A short text-to-image LoRA run on real weights, then the same-seed with/without comparison at
/// bf16, q8 and q4. Asserts, per tier: both renders are pictures; the adapter moves the render by
/// at least [`ADAPTER_MOVES_FLOOR`]; it moves it TOWARD the training palette by at least
/// [`PALETTE_DISTANCE_GAIN_FLOOR`]; the observed overlay does not exceed the priced one. At bf16 a
/// strength-0 load must be byte-identical to the bare render (the change is the adapter's, not
/// nondeterminism). The run itself must complete every step with finite losses, carry the 2.1
/// provenance, and peak inside its own preflight's derived train phase.
///
/// `QWEN_IMAGE_2_1_LORA_T2I_STEPS` (default 1000 — at the product rate [`TRAIN_LR`], ~1.45 s a
/// step on an M5 Max at 512 px, ~24 min; 300 steps moved the palette distance only 1-3) scales the
/// run without a code edit.
#[test]
#[ignore]
fn t2i_lora_trains_reloads_and_moves_every_tier() {
    let out = out_dir().join("t2i");
    std::fs::create_dir_all(&out).unwrap();
    let guard = Footprint::start(&out);
    let adapters = out_dir().join("adapters");
    std::fs::create_dir_all(&adapters).unwrap();
    let data = out.join("dataset");
    std::fs::create_dir_all(&data).unwrap();

    const SUBJECTS: [&str; 8] = [
        "a lighthouse",
        "a teapot",
        "a city skyline",
        "a sailing boat",
        "a mountain",
        "a bicycle",
        "a cat",
        "a tree",
    ];
    let items: Vec<TrainingItem> = SUBJECTS
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let path = data.join(format!("style_{i}.png"));
            ring_image(i as u32, TRAIN_EDGE).save(&path).unwrap();
            TrainingItem::captioned(
                path,
                format!("zxq style, {subject} drawn as concentric teal and orange rings on yellow"),
            )
        })
        .collect();
    let steps = training_steps("QWEN_IMAGE_2_1_LORA_T2I_STEPS", 1000);
    let req = TrainingRequest {
        items,
        config: TrainingConfig {
            rank: 16,
            alpha: 16.0,
            learning_rate: TRAIN_LR,
            steps,
            gradient_checkpointing: true,
            resolution: TRAIN_EDGE,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".into(),
            network_type: NetworkType::Lora,
            ..Default::default()
        },
        output_dir: adapters.clone(),
        file_name: T2I_ADAPTER.into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    let trained = train(&req, &guard, &adapters.join(T2I_ADAPTER), 10);

    if std::env::var("QWEN_IMAGE_2_1_PROBE_ONLY").as_deref() == Ok("1") {
        assert!(
            steps <= 3,
            "a bounded probe must not train more than three steps"
        );
        write_json(&out_dir(), "t2i_probe", &trained.facts);
        assert_trained("t2i probe", &trained, steps, false);
        assert_preflight_covers_step("t2i probe", &trained);
        return;
    }

    let request = t2i_request();
    let mut cases = Vec::new();
    let mut images = Vec::new();
    for (tier, quant) in tiers() {
        let spec = tier_spec(tier, quant);
        let (base, base_facts) = render(&format!("{tier}_base"), &spec, &request, &guard, &out);
        let (adapted, adapted_facts) = render(
            &format!("{tier}_lora"),
            &spec
                .clone()
                .with_adapters(vec![adapter(&trained.adapter, 1.0, AdapterKind::Lora)]),
            &request,
            &guard,
            &out,
        );
        let zero = (tier == "bf16").then(|| {
            render(
                &format!("{tier}_lora_scale0"),
                &spec.clone().with_adapters(vec![adapter(
                    &trained.adapter,
                    0.0,
                    AdapterKind::Lora,
                )]),
                &request,
                &guard,
                &out,
            )
        });
        cases.push(json!({
            "tier": tier,
            "base": base_facts,
            "adapted": adapted_facts,
            "scale0": zero.as_ref().map(|(_, facts)| facts.clone()),
            "meanAbsDiff": mean_abs_diff(&adapted, &base),
            "paletteFractionBase": palette_fraction(&base),
            "paletteFractionAdapted": palette_fraction(&adapted),
            "paletteDistanceBase": palette_distance(&base),
            "paletteDistanceAdapted": palette_distance(&adapted),
            "scale0IdenticalToBase": zero.as_ref().map(|(img, _)| img.pixels == base.pixels),
        }));
        images.push((tier, base, adapted));
    }
    write_json(
        &out_dir(),
        "t2i_lora",
        &json!({"training": trained.facts, "renders": cases}),
    );

    assert_trained("t2i lora", &trained, steps, false);
    for ((tier, base, adapted), case) in images.iter().zip(&cases) {
        assert_sane(&format!("{tier} base"), base);
        assert_sane(&format!("{tier} lora"), adapted);
        let moved = case["meanAbsDiff"].as_f64().unwrap();
        assert!(
            moved >= ADAPTER_MOVES_FLOOR,
            "{tier}: the trained LoRA moved the same-seed render by only {moved:.3}/255"
        );
        let (db, da) = (
            case["paletteDistanceBase"].as_f64().unwrap(),
            case["paletteDistanceAdapted"].as_f64().unwrap(),
        );
        assert!(
            da + PALETTE_DISTANCE_GAIN_FLOOR <= db,
            "{tier}: the LoRA did not move the render toward its training palette (mean distance \
             {db:.2} -> {da:.2}, floor -{PALETTE_DISTANCE_GAIN_FLOOR})"
        );
        assert_overlay_not_underpredicted(tier, &case["base"], &case["adapted"]);
        if let Some(identical) = case["scale0IdenticalToBase"].as_bool() {
            assert!(
                identical,
                "{tier}: a strength-0 LoRA must render byte-identically to the bare base"
            );
        }
    }
    assert_preflight_covers_step("t2i lora", &trained);
}

// ── 2. instruction-edit LoKr on two references ───────────────────────────────────────────────────

/// A representative short instruction-edit **LoKr** run on six one-reference edit pairs (image 1
/// is the source; target is the source inverted and posterized), then a held-out
/// two-reference edit with and without the adapter at bf16, q8 and q4. Asserts, per tier: both
/// renders are pictures; the adapter moves the edit by at least [`ADAPTER_MOVES_FLOOR`]; its
/// output is CLOSER to the trained transform of the held-out source than the bare base's by at
/// least [`EDIT_GAIN_FLOOR`]; the observed overlay (LoKr materializes dense deltas on a dense base)
/// does not exceed the priced one. The run must carry the edit marker and peak inside its own
/// preflight.
///
/// `QWEN_IMAGE_2_1_LORA_EDIT_STEPS` (default 120 at [`TRAIN_LR`], ~30 s a step on an M5 Max, ~60
/// min; 40 historical two-reference steps did not yet move the held-out edit toward the transform)
/// scales this representative run: each step attends over one 1024-px-fitted reference.
/// Evaluation, imports and stacking retain their ordered two-reference protocol and thresholds.
#[test]
#[ignore]
fn edit_lokr_trains_and_moves_two_reference_edits_every_tier() {
    let out = out_dir().join("edit");
    std::fs::create_dir_all(&out).unwrap();
    let guard = Footprint::start(&out);
    let adapters = out_dir().join("adapters");
    std::fs::create_dir_all(&adapters).unwrap();
    let data = out.join("dataset");
    std::fs::create_dir_all(&data).unwrap();

    let key_path = data.join("key.png");
    let key = edit_key(TRAIN_EDGE);
    key.save(&key_path).unwrap();
    let mut raw_rgb_audits = Vec::new();
    let items: Vec<TrainingItem> = (0..edit_training_balanced64::ITEMS)
        .map(|i| {
            let src = image::RgbImage::from_fn(TRAIN_EDGE, TRAIN_EDGE, |x, y| {
                image::Rgb(edit_training_balanced64::training_pixel(i, x, y, TRAIN_EDGE))
            });
            let target = edit_transform(&src);
            let audit = edit_training_balanced64::audit(i, src.as_raw(), target.as_raw());
            raw_rgb_audits.push(json!({"index":i,"sourceRawRgbSha256":audit.source_sha256,
                "targetRawRgbSha256":audit.target_sha256,"targetColours":64,"pixelsPerTargetColour":4096,
                "distinctSourceBytesPerChannel":256,"pixelsPerSourceBytePerChannel":1024}));
            let src_path = data.join(format!("src_{i}.png"));
            let tgt_path = data.join(format!("tgt_{i}.png"));
            src.save(&src_path).unwrap();
            target.save(&tgt_path).unwrap();
            TrainingItem::edit_pair(tgt_path, TRAIN_EDIT_INSTRUCTION.into(), vec![src_path])
        })
        .collect();
    let steps = training_steps("QWEN_IMAGE_2_1_LORA_EDIT_STEPS", 120);
    let req = TrainingRequest {
        items,
        config: TrainingConfig {
            rank: 16,
            alpha: 16.0,
            learning_rate: TRAIN_LR,
            steps,
            gradient_checkpointing: true,
            resolution: TRAIN_EDGE,
            save_every: 0,
            seed: 42,
            optimizer: "adamw".into(),
            network_type: NetworkType::Lokr,
            ..Default::default()
        },
        output_dir: adapters.clone(),
        file_name: EDIT_ADAPTER.into(),
        trigger_words: Vec::new(),
        cancel: Default::default(),
    };
    assert!(req.items.iter().all(
        |item| item.caption == TRAIN_EDIT_INSTRUCTION && item.reference_image_paths.len() == 1
    ));
    assert!(
        edit_training_balanced64::recipe_is_fixed(
            edit_training_balanced64::Recipe {
                items: req.items.len(),
                rank: req.config.rank,
                alpha: req.config.alpha,
                learning_rate: req.config.learning_rate,
                seed: req.config.seed,
                checkpointing: req.config.gradient_checkpointing,
                edge: req.config.resolution,
                references: req.items[0].reference_image_paths.len(),
                steps: req.config.steps,
                lokr: req.config.network_type == NetworkType::Lokr,
                adamw: req.config.optimizer == "adamw",
            },
            std::env::var("QWEN_IMAGE_2_1_PROBE_ONLY").as_deref() == Ok("1")
        ),
        "balanced64 training recipe drift refused before admission"
    );
    let protocol = json!({
        "kind": "representative_one_reference_training_two_reference_evaluation",
        "trainingReferenceCount": 1, "evaluationReferenceCount": 2,
        "trainingCaption": TRAIN_EDIT_INSTRUCTION, "evaluationCaption": EDIT_INSTRUCTION,
        "trainingTargetEdge": TRAIN_EDGE, "trainingReferenceFittedEdge": 1024,
        "evaluationTargetEdge": RENDER_EDGE, "stepsRequested": steps,
        "dataset": dataset_receipt(&req.items), "evaluationKeySha256": sha256_file(&key_path),
        "trainingDataRecipe": {"version":edit_training_balanced64::VERSION,"frozenPlanSha256":edit_training_balanced64::PLAN_SHA256,
            "rawRgbAudits":raw_rgb_audits,"heldoutSource99UsedForTraining":false,
            "itemExposuresAt120RoundRobinSteps":20,"targetAuthority":"unchanged edit_transform(source)",
            "scopeLimit":"colour coverage does not remove512/768, one/two reference or denseBF16/packedQ4 gaps"},
        "historicalTwoReferenceTrainingEvidence": "retained only at its original source SHA",
    });
    // Persist the disclosed protocol before admission, so even an explicit refusal is attributable.
    write_json(&out, "edit-training-protocol", &protocol);
    let mut trained = train(&req, &guard, &adapters.join(EDIT_ADAPTER), 1);
    trained.facts["editProtocol"] = protocol;

    if std::env::var("QWEN_IMAGE_2_1_PROBE_ONLY").as_deref() == Ok("1") {
        assert!(
            steps <= 3,
            "a bounded probe must not train more than three steps"
        );
        write_json(&out_dir(), "edit_probe", &trained.facts);
        assert_trained("edit probe", &trained, steps, true);
        assert_preflight_covers_step("edit probe", &trained);
        return;
    }

    // The held-out two-reference edit: unseen source plus the levels key named by evaluation.
    let eval_src = edit_source(99, RENDER_EDGE);
    eval_src.save(out.join("eval_source.png")).unwrap();
    let expected = to_image(edit_transform(&eval_src));
    save_png(&out.join("eval_expected.png"), &expected);
    let request = GenerationRequest {
        prompt: EDIT_INSTRUCTION.to_owned(),
        width: RENDER_EDGE,
        height: RENDER_EDGE,
        steps: Some(RENDER_STEPS),
        seed: Some(SEED),
        conditioning: vec![Conditioning::MultiReference {
            images: vec![to_image(eval_src), to_image(key)],
        }],
        ..Default::default()
    };
    let mut cases = Vec::new();
    let mut images = Vec::new();
    for (tier, quant) in tiers() {
        let spec = tier_spec(tier, quant);
        let (base, base_facts) = render(&format!("{tier}_base"), &spec, &request, &guard, &out);
        let (adapted, adapted_facts) = render(
            &format!("{tier}_lokr"),
            &spec
                .clone()
                .with_adapters(vec![adapter(&trained.adapter, 1.0, AdapterKind::Lokr)]),
            &request,
            &guard,
            &out,
        );
        cases.push(json!({
            "tier": tier,
            "base": base_facts,
            "adapted": adapted_facts,
            "meanAbsDiff": mean_abs_diff(&adapted, &base),
            "errorToExpectedBase": mean_abs_diff(&base, &expected),
            "errorToExpectedAdapted": mean_abs_diff(&adapted, &expected),
        }));
        images.push((tier, base, adapted));
    }
    write_json(
        &out_dir(),
        "edit_lokr",
        &json!({"training": trained.facts, "renders": cases}),
    );

    assert_trained("edit lokr", &trained, steps, true);
    for ((tier, base, adapted), case) in images.iter().zip(&cases) {
        assert_sane(&format!("{tier} base edit"), base);
        assert_sane(&format!("{tier} lokr edit"), adapted);
        let moved = case["meanAbsDiff"].as_f64().unwrap();
        assert!(
            moved >= ADAPTER_MOVES_FLOOR,
            "{tier}: the trained edit LoKr moved the same-seed edit by only {moved:.3}/255"
        );
        let (eb, ea) = (
            case["errorToExpectedBase"].as_f64().unwrap(),
            case["errorToExpectedAdapted"].as_f64().unwrap(),
        );
        assert!(
            ea + EDIT_GAIN_FLOOR <= eb,
            "{tier}: the edit LoKr did not move the output toward the trained transform (error to \
             expected {eb:.2} -> {ea:.2}, floor -{EDIT_GAIN_FLOOR})"
        );
        assert_overlay_not_underpredicted(tier, &case["base"], &case["adapted"]);
    }
    assert_preflight_covers_step("edit lokr", &trained);
}

// ── 3. two stacked adapters, independent strengths ───────────────────────────────────────────────

/// Both trained files on ONE bf16 load, each at its own strength, on the T2I eval render. Asserts:
/// `[t2i@1, edit@0]` is byte-identical to `[t2i@1]` and `[t2i@0, edit@1]` to `[edit@1]` (a
/// strength reaches only its own file); `[t2i@1, edit@1]` differs from each single file by at
/// least [`STACK_MOVES_FLOOR`] (both contribute); halving the T2I strength inside the stack moves
/// the render by at least [`STACK_MOVES_FLOOR`] (strengths are applied, not just switched). Needs
/// the two adapters the training tests above wrote to `<out>/adapters/`.
#[test]
#[ignore]
fn stacked_adapters_apply_with_independent_weights() {
    let out = out_dir().join("stack");
    std::fs::create_dir_all(&out).unwrap();
    let guard = Footprint::start(&out);
    let adapters = out_dir().join("adapters");
    let (t2i, edit) = (adapters.join(T2I_ADAPTER), adapters.join(EDIT_ADAPTER));
    for file in [&t2i, &edit] {
        assert!(
            file.is_file(),
            "{} is missing — run the two training tests first (the lane orders them)",
            file.display()
        );
    }
    let spec = tier_spec("bf16", None);
    let request = t2i_request();
    let stacks: [(&str, Vec<AdapterSpec>); 6] = [
        ("t2i_1", vec![adapter(&t2i, 1.0, AdapterKind::Lora)]),
        ("edit_1", vec![adapter(&edit, 1.0, AdapterKind::Lokr)]),
        (
            "t2i_1_edit_0",
            vec![
                adapter(&t2i, 1.0, AdapterKind::Lora),
                adapter(&edit, 0.0, AdapterKind::Lokr),
            ],
        ),
        (
            "t2i_0_edit_1",
            vec![
                adapter(&t2i, 0.0, AdapterKind::Lora),
                adapter(&edit, 1.0, AdapterKind::Lokr),
            ],
        ),
        (
            "t2i_1_edit_1",
            vec![
                adapter(&t2i, 1.0, AdapterKind::Lora),
                adapter(&edit, 1.0, AdapterKind::Lokr),
            ],
        ),
        (
            "t2i_05_edit_1",
            vec![
                adapter(&t2i, 0.5, AdapterKind::Lora),
                adapter(&edit, 1.0, AdapterKind::Lokr),
            ],
        ),
    ];
    let mut rendered = BTreeMap::new();
    let mut facts = Vec::new();
    for (label, stack) in stacks {
        let (img, f) = render(
            label,
            &spec.clone().with_adapters(stack),
            &request,
            &guard,
            &out,
        );
        facts.push(f);
        rendered.insert(label, img);
    }
    fn pick<'m>(rendered: &'m BTreeMap<&str, Image>, label: &str) -> &'m Image {
        rendered
            .get(label)
            .unwrap_or_else(|| panic!("no render labelled {label}"))
    }
    let r = |label: &str| pick(&rendered, label);
    let comparisons = json!({
        "t2i_1_edit_0 == t2i_1": r("t2i_1_edit_0").pixels == r("t2i_1").pixels,
        "t2i_0_edit_1 == edit_1": r("t2i_0_edit_1").pixels == r("edit_1").pixels,
        "meanAbsDiff(t2i_1_edit_1, t2i_1)": mean_abs_diff(r("t2i_1_edit_1"), r("t2i_1")),
        "meanAbsDiff(t2i_1_edit_1, edit_1)": mean_abs_diff(r("t2i_1_edit_1"), r("edit_1")),
        "meanAbsDiff(t2i_05_edit_1, t2i_1_edit_1)":
            mean_abs_diff(r("t2i_05_edit_1"), r("t2i_1_edit_1")),
    });
    write_json(
        &out_dir(),
        "stacked_adapters",
        &json!({"renders": facts, "comparisons": comparisons}),
    );

    for (label, img) in &rendered {
        assert_sane(label, img);
    }
    assert!(
        comparisons["t2i_1_edit_0 == t2i_1"].as_bool().unwrap(),
        "a strength-0 second file changed the stack's render — strengths are not per file"
    );
    assert!(
        comparisons["t2i_0_edit_1 == edit_1"].as_bool().unwrap(),
        "a strength-0 first file changed the stack's render — strengths are not per file"
    );
    for key in [
        "meanAbsDiff(t2i_1_edit_1, t2i_1)",
        "meanAbsDiff(t2i_1_edit_1, edit_1)",
        "meanAbsDiff(t2i_05_edit_1, t2i_1_edit_1)",
    ] {
        let moved = comparisons[key].as_f64().unwrap();
        assert!(
            moved >= STACK_MOVES_FLOOR,
            "{key} = {moved:.3}/255 is under the {STACK_MOVES_FLOOR} floor — one stacked file or \
             its strength did not reach the render"
        );
    }
}

// ── 4. a public third-party 2.1 LoRA ─────────────────────────────────────────────────────────────

/// A published third-party Qwen-Image 2.1 adapter installs through the strict seam (an unmatched
/// key fails the load by name) and moves a same-seed render at every tier by at least
/// [`ADAPTER_MOVES_FLOOR`], with both renders pictures and the overlay not under-predicted.
///
/// Operator-supplied, never fetched here: `QWEN_IMAGE_2_1_THIRD_PARTY_LORA` is the materialized
/// file (the lane downloads a pinned `owner/repo@<sha>:<file>` dispatch input into the hub cache),
/// `_KIND` is `lora` (default) or `lokr`, `_SCALE` defaults to 1, `_PROMPT` to the T2I eval prompt
/// without the training trigger. Panics when unset — the lane selects it only when the dispatch
/// names an adapter, so an unset variable can never turn into a 0.00 s green.
#[test]
#[ignore]
fn third_party_lora_applies_strictly_and_moves_every_tier() {
    let file = PathBuf::from(std::env::var("QWEN_IMAGE_2_1_THIRD_PARTY_LORA").expect(
        "set QWEN_IMAGE_2_1_THIRD_PARTY_LORA to a materialized third-party Qwen-Image 2.1 adapter",
    ));
    assert!(file.is_file(), "{} is not a file", file.display());
    let kind = match std::env::var("QWEN_IMAGE_2_1_THIRD_PARTY_LORA_KIND")
        .unwrap_or_else(|_| "lora".into())
        .as_str()
    {
        "lora" => AdapterKind::Lora,
        "lokr" => AdapterKind::Lokr,
        other => panic!("QWEN_IMAGE_2_1_THIRD_PARTY_LORA_KIND={other}: lora or lokr"),
    };
    let scale: f32 = std::env::var("QWEN_IMAGE_2_1_THIRD_PARTY_LORA_SCALE")
        .ok()
        .map(|v| v.parse().expect("a float scale"))
        .unwrap_or(1.0);
    let prompt = std::env::var("QWEN_IMAGE_2_1_THIRD_PARTY_LORA_PROMPT")
        .unwrap_or_else(|_| "a lighthouse on a rocky coast at dusk".into());

    let out = out_dir().join("third_party");
    std::fs::create_dir_all(&out).unwrap();
    let guard = Footprint::start(&out);
    let request = GenerationRequest {
        prompt,
        ..t2i_request()
    };
    let metadata = safetensors_file_metadata(file.as_path()).ok();
    let mut cases = Vec::new();
    let mut images = Vec::new();
    for (tier, quant) in tiers() {
        let spec = tier_spec(tier, quant);
        let (base, base_facts) = render(&format!("{tier}_base"), &spec, &request, &guard, &out);
        let (adapted, adapted_facts) = render(
            &format!("{tier}_third_party"),
            &spec
                .clone()
                .with_adapters(vec![adapter(&file, scale, kind)]),
            &request,
            &guard,
            &out,
        );
        cases.push(json!({
            "tier": tier,
            "base": base_facts,
            "adapted": adapted_facts,
            "meanAbsDiff": mean_abs_diff(&adapted, &base),
        }));
        images.push((tier, base, adapted));
    }
    write_json(
        &out_dir(),
        "third_party_lora",
        &json!({
            "adapter": file.display().to_string(),
            "kind": format!("{kind:?}"),
            "scale": scale,
            "metadata": metadata,
            "renders": cases,
        }),
    );

    for ((tier, base, adapted), case) in images.iter().zip(&cases) {
        assert_sane(&format!("{tier} base"), base);
        assert_sane(&format!("{tier} third-party"), adapted);
        let moved = case["meanAbsDiff"].as_f64().unwrap();
        assert!(
            moved >= ADAPTER_MOVES_FLOOR,
            "{tier}: the third-party adapter moved the same-seed render by only {moved:.3}/255"
        );
        assert_overlay_not_underpredicted(tier, &case["base"], &case["adapted"]);
    }
}

/// Hash-pinned CUDA LoRA/LoKr and the completed MLX 1000-step LoRA must install and move BOTH
/// routes at all three tiers. Every cell is written before assertions; no missing environment
/// or empty manifest may skip an advertised part of this matrix.
#[test]
#[ignore]
fn imported_adapters_move_t2i_and_two_reference_edit_every_tier() {
    let manifest_path = PathBuf::from(
        std::env::var("QWEN_IMAGE_2_1_IMPORT_MANIFEST")
            .expect("set QWEN_IMAGE_2_1_IMPORT_MANIFEST to the materialized hash-pinned manifest"),
    );
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let directory = PathBuf::from(manifest["directory"].as_str().unwrap());
    let entries = manifest["adapters"].as_array().unwrap();
    assert_eq!(
        entries.len(),
        3,
        "the complete transfer matrix has three adapters"
    );
    let mut imports = Vec::new();
    for entry in entries {
        let file_name = entry["file"].as_str().unwrap();
        assert!(Path::new(file_name).components().count() == 1);
        let file = directory.join(file_name);
        let actual = format!("{:x}", Sha256::digest(std::fs::read(&file).unwrap()));
        assert_eq!(
            actual,
            entry["sha256"].as_str().unwrap(),
            "transferred adapter hash"
        );
        if entry["name"] == style_protocol::DONOR_NAME {
            verify_original_style_donor(entry, &file);
        }
        let kind = match entry["kind"].as_str().unwrap() {
            "lora" => AdapterKind::Lora,
            "lokr" => AdapterKind::Lokr,
            other => panic!("unknown transferred kind {other}"),
        };
        imports.push((entry["name"].as_str().unwrap(), file, kind));
    }
    assert_eq!(
        imports
            .iter()
            .map(|x| x.0)
            .collect::<std::collections::BTreeSet<_>>(),
        ["cuda_lora", "cuda_lokr", "mlx_t2i_1000_steps"]
            .into_iter()
            .collect()
    );
    // The corrected edit run is a fourth identity produced by this same candidate. Keep its
    // hash/provenance with the matrix, and exercise T2I too rather than relying on the edit
    // learning check to certify both routes. An imports-only dispatch covers the pinned donors.
    let corrected_edit = std::env::var("QWEN_IMAGE_2_1_CORRECTED_EDIT_ADAPTER")
        .ok()
        .map(|file| {
            let file = PathBuf::from(file);
            let metadata = safetensors_file_metadata(&file).unwrap();
            assert_eq!(
                metadata.get("trainingMode").map(String::as_str),
                Some("edit")
            );
            assert_eq!(
                metadata.get("family").map(String::as_str),
                Some("qwen-image-2-1")
            );
            let sha = format!("{:x}", Sha256::digest(std::fs::read(&file).unwrap()));
            imports.push(("mlx_corrected_edit_lokr", file.clone(), AdapterKind::Lokr));
            json!({"file": file, "sha256": sha, "metadata": metadata,
                "sourceCandidate": std::env::var("GITHUB_SHA").ok()})
        });
    let out = out_dir().join("imports");
    std::fs::create_dir_all(&out).unwrap();
    let guard = Footprint::start(&out);
    let edit = GenerationRequest {
        prompt: EDIT_INSTRUCTION.to_owned(),
        conditioning: vec![Conditioning::MultiReference {
            images: vec![
                to_image(edit_source(99, RENDER_EDGE)),
                to_image(edit_key(RENDER_EDGE)),
            ],
        }],
        ..t2i_request()
    };
    let mut cases = Vec::new();
    let mut images = Vec::new();
    for (tier, quant) in tiers() {
        for (mode, request) in [("t2i", t2i_request()), ("two_reference_edit", edit.clone())] {
            let spec = tier_spec(tier, quant);
            let (base, base_facts) = render(
                &format!("{tier}_{mode}_base"),
                &spec,
                &request,
                &guard,
                &out,
            );
            // The retained style donor was trained/evaluated before the mixed edit-prefix
            // diagnostic. Its learned direction uses that frozen request and its own base.
            let style_request = original_style_request();
            let style_base = (mode == "t2i").then(|| {
                render(
                    &format!("{tier}_t2i_original_style_base"),
                    &spec,
                    &style_request,
                    &guard,
                    &out,
                )
            });
            for (name, file, kind) in &imports {
                let label = format!("{tier}_{mode}_{name}");
                let (paired_base, paired_base_facts, paired_request) =
                    if style_protocol::uses_original_style_request(name, mode) {
                        let (image, facts) = style_base.as_ref().unwrap();
                        (image, facts, &style_request)
                    } else {
                        (&base, &base_facts, &request)
                    };
                let (adapted, adapted_facts) = render(
                    &label,
                    &spec.clone().with_adapters(vec![adapter(file, 1.0, *kind)]),
                    paired_request,
                    &guard,
                    &out,
                );
                cases.push(json!({"tier": tier, "mode": mode, "adapter": name,
                    "base": paired_base_facts, "adapted": adapted_facts,
                    "request": direction_request_facts(paired_request),
                    "requestProtocol": if style_protocol::uses_original_style_request(name, mode) {
                        "original-1000-step-style" } else if mode == "t2i" {
                        "common-edit-prefix" } else { "ordered-two-reference-edit" },
                    "meanAbsDiff": mean_abs_diff(&adapted, paired_base),
                    "paletteDistanceGain": palette_distance(paired_base) - palette_distance(&adapted),
                }));
                images.push((label, paired_base.clone(), adapted));
            }
        }
    }
    write_json(
        &out_dir(),
        "imported_adapters",
        &json!({"manifest": manifest, "correctedEditAdapter": corrected_edit, "renders": cases,
            "additionalOriginalStyleBaseRenders": 3, "stylePalette": PALETTE}),
    );
    assert_eq!(
        cases.len(),
        imports.len() * 6,
        "every adapter × two routes × three tiers"
    );
    for ((label, base, adapted), case) in images.iter().zip(&cases) {
        assert_sane(label, base);
        assert_sane(label, adapted);
        assert!(
            case["meanAbsDiff"].as_f64().unwrap() >= ADAPTER_MOVES_FLOOR,
            "{label}: imported adapter must move the same-seed render"
        );
        assert_overlay_not_underpredicted(label, &case["base"], &case["adapted"]);
        if case["adapter"] == "mlx_t2i_1000_steps" && case["mode"] == "t2i" {
            assert!(case["paletteDistanceGain"].as_f64().unwrap() >= PALETTE_DISTANCE_GAIN_FLOOR,
                "{label}: the preserved 1000-step adapter must still move toward its learned palette");
        }
    }
}

/// Fixed style-protocol diagnosis: six renders, no training or terminal acceptance.
#[test]
#[ignore = "needs pinned snapshots, immutable original style donor and Metal"]
fn diagnostic_reused_t2i_style_direction() {
    let source = std::env::var("GITHUB_SHA").expect("record exact diagnostic source SHA");
    assert!(source.len() == 40 && source.bytes().all(|b| b.is_ascii_hexdigit()));
    let manifest_path = PathBuf::from(
        std::env::var("QWEN_IMAGE_2_1_IMPORT_MANIFEST")
            .expect("exact hash-pinned original donor manifest required"),
    );
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let directory = PathBuf::from(manifest["directory"].as_str().unwrap());
    assert!(directory.is_absolute() && directory.is_dir());
    let entries = manifest["adapters"].as_array().unwrap();
    assert_eq!(entries.len(), 3);
    let donors: Vec<_> = entries
        .iter()
        .filter(|e| e["name"] == style_protocol::DONOR_NAME)
        .collect();
    assert_eq!(
        donors.len(),
        1,
        "exactly one immutable original style donor"
    );
    let entry = donors[0];
    let basename = entry["file"].as_str().unwrap();
    assert!(Path::new(basename).components().count() == 1);
    assert_eq!(Path::new(basename).file_name().unwrap(), basename);
    let file = directory.join(basename);
    verify_original_style_donor(entry, &file);

    let out = out_dir().join("style-protocol");
    std::fs::create_dir_all(&out).unwrap();
    let request = original_style_request();
    let provenance = json!({"trainingSource": style_protocol::TRAINING_SOURCE,
        "trainingRun": style_protocol::TRAINING_RUN, "trainingSteps": 1000,
        "donorSha256": style_protocol::DONOR_SHA256, "donorBytes": style_protocol::DONOR_BYTES});
    write_json(
        &out,
        "DIAGNOSTIC_ONLY",
        &json!({
        "purpose": "DIAGNOSTIC_ONLY", "acceptanceEvidence": false, "retrain": false,
        "sourceCandidate": source, "trainingProvenance": provenance,
        "request": direction_request_facts(&request), "adapterStrength": 1.0,
        "stylePalette": PALETTE, "renderCount": 6,
        "movementFloor": ADAPTER_MOVES_FLOOR, "paletteGainFloor": PALETTE_DISTANCE_GAIN_FLOOR}),
    );
    let guard = Footprint::start(&out);
    let mut cases = Vec::new();
    let mut images = Vec::new();
    for (tier, quant) in tiers() {
        let spec = tier_spec(tier, quant);
        let (base, base_facts) =
            render(&format!("{tier}_style_base"), &spec, &request, &guard, &out);
        let label = format!("{tier}_style_mlx_t2i_1000_steps");
        let (adapted, adapted_facts) = render(
            &label,
            &spec.with_adapters(vec![adapter(&file, 1.0, AdapterKind::Lora)]),
            &request,
            &guard,
            &out,
        );
        cases.push(json!({"tier": tier, "mode": "t2i", "adapter": style_protocol::DONOR_NAME,
            "base": base_facts, "adapted": adapted_facts, "request": direction_request_facts(&request),
            "requestProtocol": "original-1000-step-style", "adapterSha256": sha256_file(&file),
            "meanAbsDiff": mean_abs_diff(&adapted, &base),
            "paletteDistanceBase": palette_distance(&base),
            "paletteDistanceAdapted": palette_distance(&adapted),
            "paletteDistanceGain": palette_distance(&base) - palette_distance(&adapted)}));
        images.push((label, base, adapted));
    }
    // Persist all six fixed cells before a learned-direction assertion can fail.
    write_json(
        &out,
        "style-direction",
        &json!({
        "purpose": "DIAGNOSTIC_ONLY", "acceptanceEvidence": false, "retrain": false,
        "sourceCandidate": source, "trainingProvenance": provenance, "donor": entry,
        "stylePalette": PALETTE, "renderCount": 6, "renders": cases}),
    );
    assert_eq!(cases.len(), 3);
    for ((label, base, adapted), case) in images.iter().zip(&cases) {
        assert_sane(label, base);
        assert_sane(label, adapted);
        assert_overlay_not_underpredicted(label, &case["base"], &case["adapted"]);
        assert!(case["meanAbsDiff"].as_f64().unwrap() >= ADAPTER_MOVES_FLOOR);
        assert!(
            case["paletteDistanceGain"].as_f64().unwrap() >= PALETTE_DISTANCE_GAIN_FLOOR,
            "{label}: original style donor must move toward its frozen training palette"
        );
    }
}

/// A fixed 19-render investigation reuses completed training from the rejected candidate.
/// Passing this test never makes it terminal acceptance: provenance and all results say so.
#[test]
#[ignore = "needs pinned snapshots, exact diagnostic donor receipt and Metal"]
fn diagnostic_reused_edit_adapter_semantics() {
    const ORIGINAL_SOURCE: &str = "b3ec3f0b8ee6880dc55e6433a35dc60116709752";
    const DONOR_SHA: &str = "c233129e9a64e384850331c9804b5b496d5208c08fe34aca4c680920fddb03d7";
    const RECEIPT_SHA: &str = "fa5b6234870f6ac0784f893067524407178ee9ee1678c50bc6bd6713d008e333";
    let diagnostic_source =
        std::env::var("GITHUB_SHA").expect("record the exact diagnostic source SHA");
    assert!(
        diagnostic_source.len() == 40 && diagnostic_source.bytes().all(|b| b.is_ascii_hexdigit())
    );
    let manifest_path = PathBuf::from(
        std::env::var("QWEN_IMAGE_2_1_DIAGNOSTIC_MANIFEST")
            .expect("exact hash-pinned diagnostic manifest required"),
    );
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest["purpose"], "DIAGNOSTIC_ONLY");
    assert_eq!(manifest["acceptanceEvidence"], false);
    assert_eq!(
        manifest["trainingProvenance"]["trainingSourceMain"],
        ORIGINAL_SOURCE
    );
    assert_eq!(
        manifest["trainingProvenance"]["trainingRun"],
        37214050997_u64
    );
    assert_eq!(
        manifest["trainingProvenance"]["trainingJob"],
        111470848550_u64
    );
    assert_eq!(manifest["trainingProvenance"]["steps"], 120);
    let directory = PathBuf::from(manifest["directory"].as_str().unwrap());
    let entries = manifest["adapters"].as_array().unwrap();
    assert_eq!(entries.len(), 3);
    let mut files = BTreeMap::new();
    for entry in entries {
        let name = entry["name"].as_str().unwrap();
        let basename = entry["file"].as_str().unwrap();
        assert_eq!(Path::new(basename).file_name().unwrap(), basename);
        let path = directory.join(basename);
        let expected = match name {
            "cuda_lokr" => "201bffa58dc8a1b61ec6399845109ebc3741220f9942f6caa0bfdc85695abf96",
            "mlx_corrected_edit_lokr" => DONOR_SHA,
            "training_receipt" => RECEIPT_SHA,
            _ => panic!("unexpected diagnostic input"),
        };
        assert_eq!(entry["sha256"], expected);
        assert_eq!(sha256_file(&path), expected);
        assert_eq!(
            std::fs::metadata(&path).unwrap().len(),
            entry["size"].as_u64().unwrap()
        );
        assert!(files.insert(name, path).is_none());
    }
    let original: Value =
        serde_json::from_slice(&std::fs::read(&files["training_receipt"]).unwrap()).unwrap();
    let training = &original["training"];
    assert_eq!(training["stepsRun"], 120);
    assert_eq!(training["steps"], 120);
    assert_eq!(training["adapterSha256"], DONOR_SHA);
    assert_eq!(training["trainingStageTraceComplete"], true);
    let losses = training["losses"].as_array().unwrap();
    assert_eq!(losses.len(), 120);
    assert!(losses
        .iter()
        .all(|x| x.as_f64().is_some_and(f64::is_finite)));
    assert_eq!(training["editProtocol"]["trainingReferenceCount"], 1);
    assert_eq!(training["editProtocol"]["evaluationReferenceCount"], 2);
    assert_eq!(
        training["editProtocol"]["trainingCaption"],
        TRAIN_EDIT_INSTRUCTION
    );
    assert_eq!(
        training["dataset"]["sha256"],
        manifest["trainingProvenance"]["datasetSha256"]
    );
    for name in ["cuda_lokr", "mlx_corrected_edit_lokr"] {
        let metadata = safetensors_file_metadata(&files[name]).unwrap();
        for (key, value) in [
            ("family", "qwen-image-2-1"),
            ("baseModel", ID),
            ("trainingMode", "edit"),
            ("networkType", "lokr"),
            ("rank", "16"),
            ("alpha", "16"),
            (
                "license",
                "Qwen Research License Agreement (research/evaluation only)",
            ),
        ] {
            assert_eq!(metadata.get(key).map(String::as_str), Some(value));
        }
    }
    let out = out_dir().join("diagnostic");
    std::fs::create_dir_all(&out).unwrap();
    write_json(
        &out,
        "DIAGNOSTIC_ONLY",
        &json!({"kind": "DIAGNOSTIC_ONLY", "acceptanceEvidence": false,
        "trainingProvenance": manifest["trainingProvenance"], "diagnosticSource": diagnostic_source,
        "trainingCaption": TRAIN_EDIT_INSTRUCTION, "evaluationCaption": EDIT_INSTRUCTION,
        "t2iPrompt": T2I_EVAL_PROMPT, "trainingReferenceCount": 1, "evaluationReferenceCount": 2,
        "seed": SEED, "steps": RENDER_STEPS, "strength": 1.0, "renderCount": 19}),
    );
    let key = edit_key(RENDER_EDGE);
    key.save(out.join("palette-key.png")).unwrap();
    let expected = to_image(edit_transform(&edit_source(99, RENDER_EDGE)));
    let edit_request = GenerationRequest {
        prompt: EDIT_INSTRUCTION.to_owned(),
        conditioning: vec![Conditioning::MultiReference {
            images: vec![
                to_image(edit_source(99, RENDER_EDGE)),
                to_image(key.clone()),
            ],
        }],
        ..t2i_request()
    };
    let guard = Footprint::start(&out);
    let mut learned = Vec::new();
    let mut transfer = Vec::new();
    let mut zero = Vec::new();
    let mut failures = Vec::new();
    for (tier, quant) in tiers() {
        let spec = tier_spec(tier, quant);
        let (edit_base, edit_base_facts) = render(
            &format!("{tier}_edit_base"),
            &spec,
            &edit_request,
            &guard,
            &out,
        );
        let (edit_adapted, edit_adapted_facts) = render(
            &format!("{tier}_edit_mlx_lokr"),
            &spec.clone().with_adapters(vec![adapter(
                &files["mlx_corrected_edit_lokr"],
                1.0,
                AdapterKind::Lokr,
            )]),
            &edit_request,
            &guard,
            &out,
        );
        let base_error = mean_abs_diff(&edit_base, &expected);
        let adapted_error = mean_abs_diff(&edit_adapted, &expected);
        let movement = mean_abs_diff(&edit_base, &edit_adapted);
        if adapted_error + EDIT_GAIN_FLOOR > base_error || movement < ADAPTER_MOVES_FLOOR {
            failures.push(format!("{tier}: learned edit movement={movement}, expected error {base_error}->{adapted_error}"));
        }
        diagnostic_render_checks(
            &format!("{tier}_edit"),
            &edit_base_facts,
            &edit_adapted_facts,
            &mut failures,
        );
        learned.push(json!({"tier": tier, "base": edit_base_facts, "adapted": edit_adapted_facts,
            "meanAbsDiff": movement, "errorToExpectedBase": base_error, "errorToExpectedAdapted": adapted_error}));
        let (t2i_base, t2i_base_facts) = render(
            &format!("{tier}_t2i_base"),
            &spec,
            &t2i_request(),
            &guard,
            &out,
        );
        for name in ["cuda_lokr", "mlx_corrected_edit_lokr"] {
            let (image, facts) = render(
                &format!("{tier}_t2i_{name}"),
                &spec
                    .clone()
                    .with_adapters(vec![adapter(&files[name], 1.0, AdapterKind::Lokr)]),
                &t2i_request(),
                &guard,
                &out,
            );
            let movement = mean_abs_diff(&t2i_base, &image);
            if movement < ADAPTER_MOVES_FLOOR {
                failures.push(format!("{tier}_t2i_{name}: movement={movement}"));
            }
            diagnostic_render_checks(
                &format!("{tier}_t2i_{name}"),
                &t2i_base_facts,
                &facts,
                &mut failures,
            );
            transfer.push(json!({"tier": tier, "adapter": name, "base": t2i_base_facts, "adapted": facts, "meanAbsDiff": movement}));
            if tier == "bf16" {
                for (mode, request, base, base_facts) in [
                    ("t2i", t2i_request(), &t2i_base, &t2i_base_facts),
                    (
                        "two_reference_edit",
                        edit_request.clone(),
                        &edit_base,
                        &edit_base_facts,
                    ),
                ] {
                    let label = format!("bf16_{mode}_{name}_zero");
                    let (image, facts) = render(
                        &label,
                        &spec.clone().with_adapters(vec![adapter(
                            &files[name],
                            0.0,
                            AdapterKind::Lokr,
                        )]),
                        &request,
                        &guard,
                        &out,
                    );
                    let same = image.pixels == base.pixels;
                    if !same {
                        failures.push(format!("{label}: scale0 changed output"));
                    }
                    diagnostic_render_checks(&label, base_facts, &facts, &mut failures);
                    zero.push(json!({"mode": mode, "adapter": name, "base": base_facts, "zero": facts, "samePixelsAsBase": same}));
                }
            }
        }
    }
    let render_count = learned.len() * 2 + 3 + transfer.len() + zero.len();
    if render_count != 19 {
        failures.push(format!("expected19 renders, captured{render_count}"));
    }
    write_json(
        &out_dir(),
        "diagnostic_semantics",
        &json!({"kind": "DIAGNOSTIC_ONLY", "acceptanceEvidence": false,
        "renderCount": render_count, "diagnosticSource": diagnostic_source,
        "trainingProvenance": manifest["trainingProvenance"], "originalTrainingReceipt": original,
        "donors": manifest["adapters"], "paletteKeySha256": sha256_file(&out.join("palette-key.png")),
        "paletteKeyRawRgbSha256": format!("{:x}", Sha256::digest(key.as_raw())),
        "trainingCaption": TRAIN_EDIT_INSTRUCTION, "evaluationCaption": EDIT_INSTRUCTION, "t2iPrompt": T2I_EVAL_PROMPT,
        "thresholds": {"movement": ADAPTER_MOVES_FLOOR, "editGain": EDIT_GAIN_FLOOR,
            "pixelStd": NON_DEGENERATE_STD, "overlaySlackBytes": PREDICTION_SLACK_BYTES},
        "learnedEdit": learned, "t2iTransfers": transfer, "zeroControls": zero, "failures": failures}),
    );
    assert!(
        failures.is_empty(),
        "DIAGNOSTIC_ONLY failures retained: {failures:?}"
    );
}

fn diagnostic_render_checks(
    label: &str,
    base: &Value,
    adapted: &Value,
    failures: &mut Vec<String>,
) {
    for facts in [base, adapted] {
        if facts["image"]["pixelStd"].as_f64().unwrap() <= NON_DEGENERATE_STD
            || facts["image"]["staticRowFraction"].as_f64().unwrap() > 0.25
        {
            failures.push(format!("{label}: image sanity"));
        }
    }
    if adapted["mlxActivePeakBytes"].as_u64().unwrap()
        > base["mlxActivePeakBytes"].as_u64().unwrap()
            + adapted["predictedOverlayBytes"].as_u64().unwrap()
            + PREDICTION_SLACK_BYTES
    {
        failures.push(format!("{label}: overlay prediction"));
    }
}
