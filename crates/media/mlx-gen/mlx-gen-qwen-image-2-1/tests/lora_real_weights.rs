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
//! `qwen-image-2-1` profile of `.github/workflows/real-weights.yml` (job `mlx-qwen-image-2-1`)
//! materializes both snapshots and runs every test here by name, in this order — the stacking test
//! consumes the two adapters the training tests write:
//!
//! 1. [`t2i_lora_trains_reloads_and_moves_every_tier`] — a short text-to-image **LoRA** run on a
//!    synthetic single-palette style, saved, reloaded through `LoadSpec::adapters`, and rendered
//!    with and without it at bf16, q8 and q4 on the same seed.
//! 2. [`edit_lokr_trains_on_two_references_and_moves_every_tier`] — a short instruction-edit
//!    **LoKr** run on two-reference edit pairs, then the same with/without comparison on a held-out
//!    two-reference edit at every tier.
//! 3. [`stacked_adapters_apply_with_independent_weights`] — both trained files stacked on one
//!    load, each at its own strength.
//! 4. [`third_party_lora_applies_strictly_and_moves_every_tier`] — a public third-party 2.1 LoRA
//!    (operator-supplied via `QWEN_IMAGE_2_1_THIRD_PARTY_LORA`; the lane runs it only when the
//!    dispatch names one).
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
//! A footprint guard aborts the process above `QWEN_IMAGE_2_1_FOOTPRINT_CEILING_GB` (default 100 GB
//! of `phys_footprint`): the host has been kernel-panicked by an unguarded MLX run before, and an
//! abort is recoverable where a panic of the box is not.
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

use crate::e2e_real_weights::phys_footprint;

const ID: &str = "qwen_image_2_1";

// ── thresholds (first-run floors; see the module docs) ──────────────────────────────────────────

/// Mean |Δ| per RGB byte (0–255) an adapter must move a same-seed render by. A residual that never
/// reaches the forward, or a tier that drops it, measures exactly 0; any trained adapter that
/// reaches the DiT moves an 8-step render by far more than 2/255.
const ADAPTER_MOVES_FLOOR: f64 = 2.0;
/// Mean |Δ| one stacked file at strength 1 must add over the other alone — the weaker of the two
/// files (the edit LoKr, on a text-to-image render) must still be visible in the stack.
const STACK_MOVES_FLOOR: f64 = 0.5;
/// Absolute gain in the fraction of pixels near the training palette the T2I LoRA must produce
/// over the bare base on the same prompt and seed — the "it learned the style" direction, not just
/// "something changed".
const PALETTE_GAIN_FLOOR: f64 = 0.05;
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
const T2I_ADAPTER: &str = "qwen21_t2i_lora.safetensors";
const EDIT_ADAPTER: &str = "qwen21_edit_lokr.safetensors";

const T2I_EVAL_PROMPT: &str = "zxq style, a lighthouse on a rocky coast at dusk";
const EDIT_INSTRUCTION: &str =
    "zxq edit: invert the colours of image 1 and posterize them to the four grey levels of image 2";

/// The T2I training style: concentric rings in exactly these three colours.
const PALETTE: [[u8; 3]; 3] = [[0, 150, 150], [240, 120, 20], [250, 220, 60]];
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

fn out_dir() -> PathBuf {
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

/// The three render tiers: the dense bf16 base, and the published packed q8 / q4 snapshots.
fn tiers() -> [(&'static str, Option<Quant>); 3] {
    [
        ("bf16", None),
        ("q8", Some(Quant::Q8)),
        ("q4", Some(Quant::Q4)),
    ]
}

fn tier_spec(label: &str, quant: Option<Quant>) -> LoadSpec {
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
struct Footprint {
    phase_max: Arc<AtomicU64>,
}

impl Footprint {
    fn start(out: &Path) -> Self {
        let ceiling_gb: f64 = std::env::var("QWEN_IMAGE_2_1_FOOTPRINT_CEILING_GB")
            .ok()
            .map(|v| {
                v.parse()
                    .expect("QWEN_IMAGE_2_1_FOOTPRINT_CEILING_GB is a number")
            })
            .unwrap_or(100.0);
        let ceiling = (ceiling_gb * 1e9) as u64;
        let phase_max = Arc::new(AtomicU64::new(0));
        let shared = phase_max.clone();
        let marker = out.join("FOOTPRINT_CEILING_EXCEEDED.txt");
        std::thread::spawn(move || loop {
            let (fp, _) = phys_footprint();
            shared.fetch_max(fp, Ordering::Relaxed);
            if fp > ceiling {
                let msg = format!(
                    "phys_footprint {:.2} GB exceeded the {ceiling_gb:.1} GB ceiling; aborting \
                     before the host does\n",
                    fp as f64 / 1e9
                );
                eprint!("{msg}");
                let _ = std::fs::write(&marker, &msg);
                std::process::abort();
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        });
        Self { phase_max }
    }

    /// Start a phase: reset MLX's active-peak counter and the footprint high-water mark.
    fn begin(&self) {
        mlx_rs::memory::reset_peak_memory();
        self.phase_max.store(0, Ordering::Relaxed);
    }

    /// `(MLX active peak, phys_footprint max)` since [`begin`](Self::begin), in bytes.
    fn end(&self) -> (u64, u64) {
        let (fp, _) = phys_footprint();
        (
            mlx_rs::memory::get_peak_memory() as u64,
            self.phase_max.load(Ordering::Relaxed).max(fp),
        )
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

// ── image metrics ────────────────────────────────────────────────────────────────────────────────

fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
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

fn write_json(out: &Path, name: &str, value: &Value) {
    let path = out.join(format!("{name}.json"));
    std::fs::write(&path, serde_json::to_string_pretty(value).unwrap()).unwrap();
    eprintln!("wrote {}", path.display());
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

fn to_image(img: image::RgbImage) -> Image {
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
fn edit_source(seed: u64, edge: u32) -> image::RgbImage {
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

/// The edit key (reference 2): four vertical grey bands — the posterize levels.
fn edit_key(edge: u32) -> image::RgbImage {
    image::RgbImage::from_fn(edge, edge, |x, _| {
        let level = (x * 4 / edge).min(3) as u8 * 85;
        image::Rgb([level, level, level])
    })
}

/// The trained edit: invert, then posterize every channel to the key's four levels.
fn edit_transform(src: &image::RgbImage) -> image::RgbImage {
    let mut out = src.clone();
    for px in out.pixels_mut() {
        for v in px.0.iter_mut() {
            *v = ((255 - *v) / 64).min(3) * 85;
        }
    }
    out
}

// ── shared run helpers ───────────────────────────────────────────────────────────────────────────

/// Load `spec` through the explicit catalog, render `req` once, write `<out>/<label>.png`, and
/// return the image plus its timing / memory / predicted-overlay facts.
fn render(
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

fn adapter(path: &Path, scale: f32, kind: AdapterKind) -> AdapterSpec {
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
    predicted_train_gib: Option<f64>,
}

/// Run `req` on a fresh trainer through the registry, recording the caching-phase and the
/// step-phase MLX peaks separately (the phases never overlap), the losses and the wall time; copy
/// the adapter to `canonical` if the trainer wrote it elsewhere.
fn train(req: &TrainingRequest, guard: &Footprint, canonical: &Path) -> Trained {
    let (predicted_message, predicted_train_gib) = predicted_training_footprint(req);
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
    let output = trainer
        .train(req, &mut |p| {
            if let TrainingProgress::Training { step, loss, .. } = p {
                if caching_peaks.is_none() {
                    // First step reported: everything before it was the staged caching.
                    caching_peaks = Some(guard.end());
                    guard.begin();
                }
                losses.push(loss);
                if step % 10 == 0 {
                    eprintln!(
                        "step {step}: loss {loss:.4} ({:.0}s)",
                        started.elapsed().as_secs_f64()
                    );
                }
            }
        })
        .unwrap_or_else(|e| panic!("training failed: {e}"));
    let seconds = started.elapsed().as_secs_f64();
    let (train_peak, train_footprint) = guard.end();
    drop(trainer);
    mlx_rs::memory::clear_cache();
    let (caching_peak, caching_footprint) = caching_peaks.unwrap_or((0, 0));

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
    let facts = json!({
        "adapter": canonical.file_name().unwrap().to_string_lossy(),
        "adapterBytes": std::fs::metadata(canonical).unwrap().len(),
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
        "predictedPreflight": predicted_message,
        "predictedTrainPhaseGiB": predicted_train_gib,
        "metadata": metadata,
    });
    Trained {
        adapter: canonical.to_path_buf(),
        facts,
        train_peak,
        predicted_train_gib,
    }
}

/// The assertions every trained run must satisfy, checked after its evidence is on disk.
fn assert_trained(label: &str, trained: &Trained, steps: u32, edit: bool) {
    let facts = &trained.facts;
    assert_eq!(facts["stepsRun"], json!(steps), "{label}: steps run");
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
    // The preflight is what stops the OS killing a worker mid-run, so its train-phase figure must
    // not under-predict the step phase it guards.
    let predicted = trained
        .predicted_train_gib
        .unwrap_or_else(|| panic!("{label}: could not read the preflight's train figure"));
    let predicted_bytes = (predicted * (1u64 << 30) as f64) as u64;
    assert!(
        trained.train_peak <= predicted_bytes + PREDICTION_SLACK_BYTES,
        "{label}: the step phase peaked at {:.2} GiB, over the preflight's derived {predicted:.1} \
         GiB (+{:.0} GiB slack) — the training footprint under-predicts",
        gib(trained.train_peak),
        gib(PREDICTION_SLACK_BYTES)
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

fn t2i_request() -> GenerationRequest {
    GenerationRequest {
        prompt: T2I_EVAL_PROMPT.to_owned(),
        width: RENDER_EDGE,
        height: RENDER_EDGE,
        steps: Some(RENDER_STEPS),
        seed: Some(SEED),
        ..Default::default()
    }
}

// ── 1. text-to-image LoRA ────────────────────────────────────────────────────────────────────────

/// A short text-to-image LoRA run on real weights, then the same-seed with/without comparison at
/// bf16, q8 and q4. Asserts, per tier: both renders are pictures; the adapter moves the render by
/// at least [`ADAPTER_MOVES_FLOOR`]; it moves it TOWARD the training palette by at least
/// [`PALETTE_GAIN_FLOOR`]; the observed overlay does not exceed the priced one. At bf16 a
/// strength-0 load must be byte-identical to the bare render (the change is the adapter's, not
/// nondeterminism). The run itself must complete every step with finite losses, carry the 2.1
/// provenance, and peak inside its own preflight's derived train phase.
///
/// `QWEN_IMAGE_2_1_LORA_T2I_STEPS` (default 100) scales the run without a code edit.
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
    let steps = env_u32("QWEN_IMAGE_2_1_LORA_T2I_STEPS", 100);
    let req = TrainingRequest {
        items,
        config: TrainingConfig {
            rank: 16,
            alpha: 16.0,
            learning_rate: 1e-3,
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
    let trained = train(&req, &guard, &adapters.join(T2I_ADAPTER));

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
        let (pb, pa) = (
            case["paletteFractionBase"].as_f64().unwrap(),
            case["paletteFractionAdapted"].as_f64().unwrap(),
        );
        assert!(
            pa >= pb + PALETTE_GAIN_FLOOR,
            "{tier}: the LoRA did not move the render toward its training palette ({pb:.3} -> \
             {pa:.3}, floor +{PALETTE_GAIN_FLOOR})"
        );
        assert_overlay_not_underpredicted(tier, &case["base"], &case["adapted"]);
        if let Some(identical) = case["scale0IdenticalToBase"].as_bool() {
            assert!(
                identical,
                "{tier}: a strength-0 LoRA must render byte-identically to the bare base"
            );
        }
    }
}

// ── 2. instruction-edit LoKr on two references ───────────────────────────────────────────────────

/// A short instruction-edit **LoKr** run on six two-reference edit pairs (image 1 = a source, image
/// 2 = the four-level key; target = the source inverted and posterized), then a held-out
/// two-reference edit with and without the adapter at bf16, q8 and q4. Asserts, per tier: both
/// renders are pictures; the adapter moves the edit by at least [`ADAPTER_MOVES_FLOOR`]; its
/// output is CLOSER to the trained transform of the held-out source than the bare base's by at
/// least [`EDIT_GAIN_FLOOR`]; the observed overlay (LoKr materializes dense deltas on a dense base)
/// does not exceed the priced one. The run must carry the edit marker and peak inside its own
/// preflight.
///
/// `QWEN_IMAGE_2_1_LORA_EDIT_STEPS` (default 40) scales the run: every step attends over two
/// 1024-px-fitted references (~8k latent tokens), so it is the expensive one.
#[test]
#[ignore]
fn edit_lokr_trains_on_two_references_and_moves_every_tier() {
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
    let items: Vec<TrainingItem> = (0..6u64)
        .map(|i| {
            let src = edit_source(1000 + i, TRAIN_EDGE);
            let src_path = data.join(format!("src_{i}.png"));
            let tgt_path = data.join(format!("tgt_{i}.png"));
            src.save(&src_path).unwrap();
            edit_transform(&src).save(&tgt_path).unwrap();
            TrainingItem::edit_pair(
                tgt_path,
                EDIT_INSTRUCTION.into(),
                vec![src_path, key_path.clone()],
            )
        })
        .collect();
    let steps = env_u32("QWEN_IMAGE_2_1_LORA_EDIT_STEPS", 40);
    let req = TrainingRequest {
        items,
        config: TrainingConfig {
            rank: 16,
            alpha: 16.0,
            learning_rate: 1e-3,
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
    let trained = train(&req, &guard, &adapters.join(EDIT_ADAPTER));

    // The held-out edit: a source the run never saw, the same key, the same instruction.
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
