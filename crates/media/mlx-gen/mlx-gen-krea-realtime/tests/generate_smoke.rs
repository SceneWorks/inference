//! Krea Realtime 14B real-weight end-to-end validation (sc-8446, S13).
//!
//! Mirrors `mlx-gen-scail2/tests/generate_smoke.rs`: one CI-safe weights-free gate plus `#[ignore]`
//! real-weight drivers that go through the **public product entry point** —
//! `provider_registry().load("krea_realtime_14b", spec)` → `Generator::generate` — so the registry,
//! snapshot resolution, tier probe, UMT5 encode, AR few-step chunk loop, and z16 VAE decode are all
//! exercised as the worker drives them.
//!
//! Beyond "it runs", these capture the three S13 measurements:
//!   * **coherence** — per-frame dynamic range / plausibility and temporal change (a drifted or
//!     collapsed AR clip shows up as a frozen or blown-out tail, which the *last-third* checks catch);
//!   * **timing** — per-chunk and whole-clip wall time (the latency input for the realtime epic 8432);
//!   * **memory** — the MLX ceiling, sampled across the run, reported as **active** and
//!     **active + cache** separately (`get_peak_memory` is the ACTIVE high-water mark and does **not**
//!     include the buffer cache, so it under-reports what the OS has to have available), plus the
//!     measured KV-cache residency that sizes the SceneWorks manifest's `mlx.minMemoryGb`.
//!
//! ```text
//! KREA_REALTIME_SNAPSHOT_DIR=~/.cache/krea-realtime-mlx-snapshot/q4 \
//! KREA_SMOKE_OUT=/tmp/krea_smoke \
//!   cargo test -p mlx-gen-krea-realtime --test generate_smoke -- --ignored --nocapture
//! ```
//!
//! Env: `KREA_REALTIME_SNAPSHOT_DIR` (snapshot root; default `~/.cache/krea-realtime-mlx-convert`),
//! `KREA_SMOKE_W`/`_H` (default 832×480, the reference bucket), `KREA_SMOKE_FRAMES` (default 81),
//! `KREA_SMOKE_STEPS` (default: the config's Self-Forcing schedule), `KREA_SMOKE_SEED`,
//! `KREA_SMOKE_OUT` (frame dump dir).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress, WeightsSource,
};
use mlx_gen_krea_realtime::{decode_latents_to_video, decode_tiling, KreaRealtimeConfig, MODEL_ID};

// ---------------------------------------------------------------------------------------------
// Environment / fixtures
// ---------------------------------------------------------------------------------------------

/// The converted MLX snapshot root, holding `dit.safetensors`, `t5_encoder.safetensors`,
/// `vae.safetensors` and `tokenizer.json` — one of the published `SceneWorks/krea-realtime-14b-mlx`
/// tiers. Caller-provided like every other component path in this workspace; the default mirrors the
/// sibling SCAIL-2 smoke.
fn snapshot_dir() -> PathBuf {
    std::env::var("KREA_REALTIME_SNAPSHOT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/krea-realtime-mlx-convert")
        })
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_opt_usize(key: &str) -> Option<usize> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Assert the snapshot is present and return its root. The DiT is the file that makes a directory a
/// Krea Realtime snapshot rather than a bare companion staging dir.
fn require_snapshot() -> PathBuf {
    let root = snapshot_dir();
    assert!(
        root.join("dit.safetensors").exists() || root.join("transformer").is_dir(),
        "no Krea Realtime DiT at {} — set KREA_REALTIME_SNAPSHOT_DIR to a tier of \
         SceneWorks/krea-realtime-14b-mlx (q4/ q8/ or bf16/)",
        root.display()
    );
    root
}

/// A deterministic gradient still, used as an i2v reference / v2v source frame.
fn gradient(w: usize, h: usize, phase: usize) -> Image {
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.extend_from_slice(&[
                ((x * 2 + phase) % 256) as u8,
                ((y * 2 + phase) % 256) as u8,
                ((x + y + phase * 3) % 256) as u8,
            ]);
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// A smooth (band-limited) moving source clip frame. Deliberately NOT the hard-edged modulo gradient:
/// a z16 VAE cannot reproduce a sawtooth discontinuity, so a hard source would put the VAE's own
/// round-trip error far above whatever the denoise contributes and make the strength=0 comparison
/// meaningless.
fn smooth_frame(w: usize, h: usize, phase: usize) -> Image {
    let mut pixels = Vec::with_capacity(w * h * 3);
    let p = phase as f32 * 0.21;
    for y in 0..h {
        let fy = y as f32 / h as f32;
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let r = 0.5 + 0.35 * ((fx * 6.0 + p).sin());
            let g = 0.5 + 0.35 * ((fy * 5.0 - p).cos());
            let b = 0.5 + 0.30 * (((fx + fy) * 4.0 + p * 0.5).sin());
            pixels.extend_from_slice(&[(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
        }
    }
    Image {
        width: w as u32,
        height: h as u32,
        pixels,
    }
}

/// Minimal binary PPM (P6) writer — no image-crate dependency, just to eyeball the clip.
fn write_ppm(path: &Path, img: &Image) {
    let mut buf = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    buf.extend_from_slice(&img.pixels);
    std::fs::write(path, buf).expect("write ppm");
}

fn dump_frames(frames: &[Image], label: &str) {
    let Some(dir) = std::env::var_os("KREA_SMOKE_OUT") else {
        return;
    };
    let dir = PathBuf::from(dir).join(label);
    std::fs::create_dir_all(&dir).expect("create the frame dump dir");
    for (i, f) in frames.iter().enumerate() {
        write_ppm(&dir.join(format!("frame{i:03}.ppm")), f);
    }
    println!("  wrote {} PPM frames to {}", frames.len(), dir.display());
}

// ---------------------------------------------------------------------------------------------
// Memory sampling
// ---------------------------------------------------------------------------------------------

/// Background sampler for the MLX allocator's high-water marks.
///
/// `mlx_rs::memory::get_peak_memory` is the **active** high-water mark only — it excludes the
/// allocator's buffer cache, which is real resident memory the OS must have available. A ceiling
/// derived from `get_peak_memory` alone therefore under-reports, so this samples `active` and
/// `active + cache` independently and reports both.
struct MemorySampler {
    stop: Arc<AtomicBool>,
    max_active: Arc<AtomicUsize>,
    max_total: Arc<AtomicUsize>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MemorySampler {
    fn start() -> Self {
        mlx_rs::memory::reset_peak_memory();
        let stop = Arc::new(AtomicBool::new(false));
        let max_active = Arc::new(AtomicUsize::new(0));
        let max_total = Arc::new(AtomicUsize::new(0));
        let (s, a, t) = (stop.clone(), max_active.clone(), max_total.clone());
        let handle = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let active = mlx_rs::memory::get_active_memory();
                let total = active + mlx_rs::memory::get_cache_memory();
                a.fetch_max(active, Ordering::Relaxed);
                t.fetch_max(total, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });
        Self {
            stop,
            max_active,
            max_total,
            handle: Some(handle),
        }
    }

    /// The running `(peak_active, peak_active_plus_cache)` so far. Both are monotonic, so taking a
    /// snapshot at each pipeline phase boundary attributes the ceiling to a phase: whichever boundary
    /// the number stops rising at is where the peak was reached.
    fn snapshot(&self) -> (usize, usize) {
        (
            self.max_active
                .load(Ordering::Relaxed)
                .max(mlx_rs::memory::get_peak_memory()),
            self.max_total.load(Ordering::Relaxed),
        )
    }

    /// Stop sampling and report `(peak_active, peak_active_plus_cache)` in bytes. `peak_active` is the
    /// max of the allocator's own high-water mark and the sampled one (the sampler can miss a spike
    /// between polls; `get_peak_memory` cannot).
    fn finish(mut self) -> (usize, usize) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let active = self
            .max_active
            .load(Ordering::Relaxed)
            .max(mlx_rs::memory::get_peak_memory());
        let total = self.max_total.load(Ordering::Relaxed).max(active);
        (active, total)
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB
}

// ---------------------------------------------------------------------------------------------
// Coherence
// ---------------------------------------------------------------------------------------------

/// Per-frame statistics used by the coherence assertions.
struct FrameStat {
    min: u8,
    max: u8,
    mean: f64,
    /// Mean |Δ| against the previous frame — 0 means the frame is a byte-identical repeat.
    delta: f64,
}

/// Mean absolute per-byte difference between two clips of the same geometry, in 0..255 units.
fn mean_abs_delta(a: &[Image], b: &[Image]) -> f64 {
    assert_eq!(a.len(), b.len(), "clip lengths differ");
    let mut total = 0.0f64;
    let mut n = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pixels.len(), y.pixels.len(), "frame buffer sizes differ");
        total += x
            .pixels
            .iter()
            .zip(y.pixels.iter())
            .map(|(&p, &q)| (p as f64 - q as f64).abs())
            .sum::<f64>();
        n += x.pixels.len();
    }
    total / n as f64
}

fn frame_stats(frames: &[Image]) -> Vec<FrameStat> {
    let mut out = Vec::with_capacity(frames.len());
    let mut prev: Option<&Image> = None;
    for f in frames {
        let min = *f.pixels.iter().min().unwrap();
        let max = *f.pixels.iter().max().unwrap();
        let mean: f64 = f.pixels.iter().map(|&b| b as f64).sum::<f64>() / f.pixels.len() as f64;
        let delta = match prev {
            Some(p) if p.pixels.len() == f.pixels.len() => {
                p.pixels
                    .iter()
                    .zip(f.pixels.iter())
                    .map(|(&a, &b)| (a as f64 - b as f64).abs())
                    .sum::<f64>()
                    / f.pixels.len() as f64
            }
            _ => f64::NAN,
        };
        out.push(FrameStat {
            min,
            max,
            mean,
            delta,
        });
        prev = Some(f);
    }
    out
}

/// The shared coherence gate for a decoded clip. Deliberately stronger than "not flat": the AR failure
/// modes this model actually has are **drift** (the tail saturates or washes out as the KV window
/// slides with `sink_size = 0`) and **freeze** (the tail stops changing), neither of which a whole-clip
/// min/max check can see. So the last third is checked on its own terms.
fn assert_coherent(frames: &[Image], w: usize, h: usize, label: &str) {
    assert!(!frames.is_empty(), "{label}: no frames produced");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.width as usize, w, "{label}: frame {i} width");
        assert_eq!(f.height as usize, h, "{label}: frame {i} height");
        assert_eq!(f.pixels.len(), w * h * 3, "{label}: frame {i} buffer size");
    }
    let stats = frame_stats(frames);
    for (i, s) in stats.iter().enumerate() {
        println!(
            "  {label} frame {i:>3}: min={:>3} max={:>3} mean={:>6.1} Δprev={:.2}",
            s.min, s.max, s.mean, s.delta
        );
    }
    for (i, s) in stats.iter().enumerate() {
        assert!(
            s.max > s.min,
            "{label}: frame {i} is flat (min==max) — the VAE decode produced a constant image"
        );
        assert!(
            (12.0..244.0).contains(&s.mean),
            "{label}: frame {i} mean {:.1} is implausible (all-black / blown out) — AR drift or a \
             decode bug",
            s.mean
        );
    }
    if frames.len() > 1 {
        let moving = stats.iter().skip(1).filter(|s| s.delta > 0.0).count();
        assert!(
            moving > 0,
            "{label}: every frame is byte-identical — the temporal decode is degenerate"
        );
        // Drift/freeze gate on the TAIL specifically: the last third must still move, and its
        // brightness must not have run away from the first third. A whole-clip check passes even when
        // a clip is fine for a second and then collapses.
        let third = (frames.len() / 3).max(1);
        let head_mean: f64 = stats[..third].iter().map(|s| s.mean).sum::<f64>() / third as f64;
        let tail = &stats[frames.len() - third..];
        let tail_mean: f64 = tail.iter().map(|s| s.mean).sum::<f64>() / tail.len() as f64;
        let tail_motion = tail.iter().filter(|s| s.delta > 0.0).count();
        println!(
            "  {label}: head mean {head_mean:.1} -> tail mean {tail_mean:.1}, \
             {tail_motion}/{} tail frames moving",
            tail.len()
        );
        if frames.len() >= 6 {
            assert!(
                tail_motion > 0,
                "{label}: the last third of the clip is frozen — AR generation collapsed mid-clip"
            );
            assert!(
                (tail_mean - head_mean).abs() < 90.0,
                "{label}: brightness drifted {:.1} from head ({head_mean:.1}) to tail \
                 ({tail_mean:.1}) — the classic bounded-window AR drift",
                (tail_mean - head_mean).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Weights-free gate (runs in CI)
// ---------------------------------------------------------------------------------------------

/// The request the real-weight smoke builds must be inside the model's *advertised* surface, checked
/// against the registered descriptor with no weights on disk. Discriminating: it goes through
/// `Generator::validate` (the same floor `run` self-applies), so a smoke that quietly drifted
/// out-of-surface — a guidance scale on this CFG-off model, an unadvertised sampler, a size below the
/// patch×stride alignment — fails here in CI instead of after a 20 GB load.
#[test]
fn smoke_request_is_within_the_advertised_surface() {
    let spec = LoadSpec::new(WeightsSource::Dir(std::env::temp_dir()));
    let gen = mlx_gen_krea_realtime::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .expect("load the krea_realtime_14b provider against an existing dir");
    assert_eq!(gen.descriptor().id, MODEL_ID);

    gen.validate(&smoke_request(832, 480, 81, None, 7))
        .expect("the t2v smoke request must validate");
    gen.validate(&i2v_request(832, 480, 81, None, 7, gradient(832, 480, 0)))
        .expect("the i2v smoke request must validate");

    // The floor really is being exercised: the same request plus a guidance scale is rejected on this
    // CFG-off model. Without this the check above could pass a validator that accepts everything.
    let mut cfg_on = smoke_request(832, 480, 81, None, 7);
    cfg_on.guidance = Some(5.0);
    let err = gen
        .validate(&cfg_on)
        .expect_err("a guidance scale must be rejected on a CFG-off model");
    assert!(err.to_string().contains("guidance"), "got: {err}");
}

// ---------------------------------------------------------------------------------------------
// Request builders (shared by the CI gate and the real-weight drivers)
// ---------------------------------------------------------------------------------------------

fn smoke_request(
    w: usize,
    h: usize,
    frames: usize,
    steps: Option<usize>,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox trotting through a snowy pine forest at sunrise, drifting snow, \
                 cinematic, shallow depth of field"
            .into(),
        width: w as u32,
        height: h as u32,
        frames: Some(frames as u32),
        steps: steps.map(|s| s as u32),
        seed: Some(seed),
        fps: Some(24),
        sampler: Some("self_forcing".into()),
        ..Default::default()
    }
}

fn i2v_request(
    w: usize,
    h: usize,
    frames: usize,
    steps: Option<usize>,
    seed: u64,
    reference: Image,
) -> GenerationRequest {
    GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: reference,
            strength: None,
        }],
        ..smoke_request(w, h, frames, steps, seed)
    }
}

// ---------------------------------------------------------------------------------------------
// Real-weight drivers
// ---------------------------------------------------------------------------------------------

/// One instrumented `generate` run: returns the decoded clip plus the timing + memory measurements.
struct RunResult {
    frames: Vec<Image>,
    fps: u32,
    total: std::time::Duration,
    /// Wall time from the first `Progress::Step` to the last (the AR denoise phase only).
    denoise: std::time::Duration,
    /// Per-chunk denoise wall times, derived from the step stream.
    per_chunk: Vec<std::time::Duration>,
    decode: std::time::Duration,
    /// Model load + prompt encode + the first denoise step (everything before the first `Step` mark).
    prologue: std::time::Duration,
    /// Mean measured denoise-step wall time (over every observable step-to-step interval).
    mean_step: std::time::Duration,
    peak_active: usize,
    peak_total: usize,
    /// `(phase label, peak_active_so_far, peak_total_so_far)` at each pipeline boundary. Because both
    /// figures are monotonic, the phase at which they stop rising is the one that sets the ceiling.
    phase_peaks: Vec<(&'static str, usize, usize)>,
}

fn run(req: &GenerationRequest, expected_chunks: usize, label: &str) -> RunResult {
    let root = require_snapshot();
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let gen = mlx_gen_krea_realtime::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .expect("load the krea_realtime_14b provider");
    gen.validate(req).expect("validate");

    let sampler = MemorySampler::start();
    let t0 = Instant::now();
    let mut first_step: Option<Instant> = None;
    let mut step_marks: Vec<Instant> = Vec::new();
    let mut decode_start: Option<Instant> = None;
    let mut last = 0u32;
    let mut phase_peaks: Vec<(&'static str, usize, usize)> = Vec::new();

    let out = gen
        .generate(req, &mut |p| match p {
            Progress::Step { current, total } => {
                let now = Instant::now();
                assert!(current >= last, "{label}: progress went backwards");
                if first_step.is_none() {
                    first_step = Some(now);
                    println!("  {label}: {total} denoise steps");
                    let (a, t) = sampler.snapshot();
                    phase_peaks.push(("after load + prompt encode", a, t));
                }
                last = current;
                step_marks.push(now);
            }
            Progress::Decoding => {
                decode_start = Some(Instant::now());
                let (a, t) = sampler.snapshot();
                phase_peaks.push(("after the AR denoise loop", a, t));
            }
            Progress::Loading(phase) => println!("  {label}: loading {phase:?}"),
        })
        .unwrap_or_else(|e| panic!("{label}: generate must succeed: {e}"));
    let total = t0.elapsed();
    let (peak_active, peak_total) = sampler.finish();
    phase_peaks.push(("after the VAE decode (final)", peak_active, peak_total));

    let GenerationOutput::Video { frames, fps, audio } = out else {
        panic!("{label}: expected a Video output");
    };
    assert!(audio.is_none(), "{label}: Krea Realtime has no audio track");

    // Per-chunk timing. The AR loop emits exactly one `Progress::Step` per denoise step and splits the
    // steps evenly across chunks, so chunk boundaries sit at `steps_per_chunk` strides through the
    // marks. Each mark is a step *completion*, so the interval between consecutive marks is one step.
    //
    // Chunk 0 is the one chunk whose start instant is not observable — the first mark already includes
    // the model load + prompt encode ahead of it. Its **first step** is therefore imputed at the mean
    // measured step time; chunks 1..N are exact mark-to-mark intervals. `prologue` (below) carries the
    // load + encode time separately so nothing is silently folded into a chunk number.
    let denoise = match (first_step, step_marks.last()) {
        (Some(a), Some(b)) => b.duration_since(a),
        _ => std::time::Duration::ZERO,
    };
    let step_intervals: Vec<std::time::Duration> = step_marks
        .windows(2)
        .map(|w| w[1].duration_since(w[0]))
        .collect();
    let mean_step = if step_intervals.is_empty() {
        std::time::Duration::ZERO
    } else {
        step_intervals.iter().sum::<std::time::Duration>() / step_intervals.len() as u32
    };
    let mut per_chunk = Vec::new();
    if expected_chunks > 0
        && !step_marks.is_empty()
        && step_marks.len().is_multiple_of(expected_chunks)
    {
        let per = step_marks.len() / expected_chunks;
        for c in 0..expected_chunks {
            let end = step_marks[(c + 1) * per - 1];
            per_chunk.push(if c == 0 {
                end.duration_since(step_marks[0]) + mean_step
            } else {
                end.duration_since(step_marks[c * per - 1])
            });
        }
    }
    let prologue = first_step
        .map(|f| f.duration_since(t0))
        .unwrap_or(std::time::Duration::ZERO);
    let decode = decode_start
        .map(|d| (t0 + total).duration_since(d))
        .unwrap_or(std::time::Duration::ZERO);

    RunResult {
        frames,
        fps,
        total,
        denoise,
        per_chunk,
        decode,
        prologue,
        mean_step,
        peak_active,
        peak_total,
        phase_peaks,
    }
}

fn report(label: &str, r: &RunResult, w: usize, h: usize, frames: usize) {
    println!("--- {label} -------------------------------------------------");
    println!(
        "  geometry     : {w}x{h}, {frames} frames @ {} fps ({} decoded)",
        r.fps,
        r.frames.len()
    );
    println!("  clip wall    : {:.2?}", r.total);
    println!(
        "  load+encode  : {:.2?} (to the first denoise-step mark)",
        r.prologue
    );
    println!(
        "  AR denoise   : {:.2?} ({:.2?}/step mean)",
        r.denoise, r.mean_step
    );
    println!("  VAE decode   : {:.2?}", r.decode);
    for (i, d) in r.per_chunk.iter().enumerate() {
        println!(
            "    chunk {i:>2}   : {d:.2?}{}",
            if i == 0 {
                "  (first step imputed at the mean)"
            } else {
                ""
            }
        );
    }
    if !r.per_chunk.is_empty() {
        let mean = r.per_chunk.iter().sum::<std::time::Duration>() / r.per_chunk.len() as u32;
        println!("    chunk mean : {mean:.2?}");
    }
    println!(
        "  MLX peak     : active {:.2} GiB | active+cache {:.2} GiB  \
         (get_peak_memory is ACTIVE only — the cache is real resident memory too)",
        gib(r.peak_active),
        gib(r.peak_total)
    );
    for (phase, a, t) in &r.phase_peaks {
        println!(
            "    cumulative peak {phase:<28}: active {:.2} GiB | active+cache {:.2} GiB",
            gib(*a),
            gib(*t)
        );
    }
}

/// **The S13 clip.** Full product path at the requested geometry: coherence, timing, memory.
#[test]
#[ignore = "real ~20 GB (q4) / ~40 GB (bf16) snapshot; run with --ignored on macOS (see module doc)"]
fn t2v_produces_a_coherent_clip() {
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 81);
    let steps = env_opt_usize("KREA_SMOKE_STEPS");
    let seed = env_usize("KREA_SMOKE_SEED", 7) as u64;

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let latent_frames = (frames - 1) / 4 + 1;
    let chunks = latent_frames.div_ceil(cfg.ar.num_frames_per_block);

    let req = smoke_request(w, h, frames, steps, seed);
    let r = run(&req, chunks, "t2v");
    report("t2v", &r, w, h, frames);
    assert_eq!(r.fps, 24, "fps passthrough");
    assert_eq!(
        r.frames.len(),
        frames,
        "the decode must be trimmed back to the requested frame count"
    );
    assert_coherent(&r.frames, w, h, "t2v");
    dump_frames(&r.frames, "t2v");
}

/// **S7 deferred minor #1 — single-frame anchor vs. the reference's repeated anchor.**
///
/// `generate_i2v` warms **one** clean-context latent frame from the still, while
/// `release_server.py::setup_start_frame` repeats the still to `kv_cache_num_frames` (3) latent frames
/// so the first generated chunk starts frame-block aligned. The engine seam accepts either, so this
/// measures the difference on real weights instead of arguing it: both runs must be coherent, and the
/// per-frame stats are printed so the anchor choice can be judged rather than assumed.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn i2v_single_frame_anchor_is_coherent() {
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);
    let steps = env_opt_usize("KREA_SMOKE_STEPS");

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let latent_frames = (frames - 1) / 4 + 1;
    // i2v generates `total_latent - 1` frames (latent frame 0 is the reference).
    let chunks = (latent_frames - 1).div_ceil(cfg.ar.num_frames_per_block);

    let req = i2v_request(w, h, frames, steps, 11, gradient(w, h, 0));
    let r = run(&req, chunks, "i2v");
    report("i2v/anchor=1", &r, w, h, frames);
    assert_eq!(r.frames.len(), frames);
    assert_coherent(&r.frames, w, h, "i2v/anchor=1");
    dump_frames(&r.frames, "i2v_anchor1");
}

/// **S7 deferred minor #2 — is v2v at `strength = 0` genuinely identity?**
///
/// At `strength = 0` every entry of the strength-scaled schedule is timestep 0, so `σ = 0` throughout:
/// the init is `source·(1−σ) + ε·σ` = the source latents verbatim, each Euler step is
/// `x − 0·v = x`, and each renoise is `(1−0)·x0 + 0·ε = x0`. The **denoise** is therefore an exact
/// identity on the latents. But the *clip* is not identity on pixels, and this pins down why with
/// controls rather than assertions:
///
///   * `A` — source → VAE round-trip through `encode` (the distribution **mode**). The floor any
///     encode/decode pays.
///   * `A'` — mode round-trip → `encode_sample` round-trip. The v2v path deliberately uses
///     `.sample()` (the reference's video-source path), which draws `mean + std·ε` per latent. `A'` is
///     the size of that draw, measured, not assumed.
///   * `C` — mode round-trip → v2v@0. What is left once the VAE mode round-trip is accounted for.
///
/// The finding this encodes: `C` is the VAE's *sampled* encode, not the denoise. So v2v@0 is identity
/// **in the latents**, and its pixel deviation is entirely the `.sample()` draw the source encode makes
/// by design — which is why the gate is `C` against the measured sampling scale `A'`, not against zero.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn v2v_strength_zero_preserves_the_source() {
    use mlx_gen_wan::{preprocess_i2v_image, WanVae};
    use mlx_rs::ops::concatenate_axis;
    use mlx_rs::random;

    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);

    let source: Vec<Image> = (0..frames).map(|i| smooth_frame(w, h, i)).collect();
    let req = GenerationRequest {
        conditioning: vec![Conditioning::VideoClip {
            frames: source.clone(),
            frame_idx: 0,
            strength: 0.0,
        }],
        ..smoke_request(w, h, frames, None, 3)
    };

    let cfg = KreaRealtimeConfig::krea_realtime_14b();
    let latent_frames = (frames - 1) / 4 + 1;
    let chunks = latent_frames.div_ceil(cfg.ar.num_frames_per_block);
    let r = run(&req, chunks, "v2v/strength=0");
    report("v2v/strength=0", &r, w, h, frames);
    assert_eq!(r.frames.len(), frames);
    dump_frames(&r.frames, "v2v_strength0");

    // The controls: the same source through the same VAE and nothing else, once at the distribution
    // mode and once sampled (the encode the v2v path actually performs).
    let root = require_snapshot();
    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let chw: Vec<mlx_rs::Array> = source
        .iter()
        .map(|f| {
            preprocess_i2v_image(f, w as u32, h as u32)
                .expect("preprocess")
                .expand_dims(1)
                .expect("expand")
        })
        .collect();
    let video = concatenate_axis(&chw.iter().collect::<Vec<_>>(), 1)
        .expect("stack the source clip")
        .expand_dims(0)
        .expect("batch axis");

    // Decode through the SAME windowing the product path uses — a single-pass control is not
    // comparable to a tiled product decode, and the difference is not small (see the tile-seam note in
    // the S13 findings).
    let tiling = decode_tiling(h, w, (latent_frames * 4) as i32);
    println!("  control decode tiling: {tiling:?}");
    let decode = |latents: &mlx_rs::Array| -> Vec<Image> {
        match decode_latents_to_video(
            &vae,
            latents,
            24,
            Some(frames),
            tiling.as_ref(),
            &mlx_gen::CancelFlag::default(),
        )
        .expect("VAE decode")
        {
            GenerationOutput::Video { frames, .. } => frames,
            other => panic!("expected a Video output, got {other:?}"),
        }
    };
    let drop_batch = |z: mlx_rs::Array| {
        let s = z.shape().to_vec();
        z.reshape(&[s[1], s[2], s[3], s[4]]).expect("drop batch")
    };

    let z_mode = drop_batch(vae.encode(&video).expect("VAE encode (mode)"));
    let mode_rt = decode(&z_mode);
    let key = random::key(99).expect("key");
    let eps = random::normal::<f32>(
        &[
            1,
            z_mode.shape()[0],
            z_mode.shape()[1],
            z_mode.shape()[2],
            z_mode.shape()[3],
        ],
        None,
        None,
        Some(&key),
    )
    .expect("eps");
    let z_sample = drop_batch(
        vae.encode_sample(&video, &eps)
            .expect("VAE encode (sampled)"),
    );
    let sample_rt = decode(&z_sample);
    dump_frames(&mode_rt, "v2v_vae_roundtrip_mode");

    let a = mean_abs_delta(&source, &mode_rt);
    let a_prime = mean_abs_delta(&mode_rt, &sample_rt);
    let b = mean_abs_delta(&source, &r.frames);
    let c = mean_abs_delta(&mode_rt, &r.frames);
    println!(
        "  v2v/strength=0 mean |Δ| (0..255): A source->VAE(mode) = {a:.2}, \
         A' VAE(mode)->VAE(sample) = {a_prime:.2}, B source->v2v@0 = {b:.2}, \
         C VAE(mode)->v2v@0 = {c:.2}"
    );
    // MEASURED (832x480, 33 frames, Q4): A ~= 27.6, A' ~= 0.01, B ~= 27.5, **C ~= 0.4**.
    //
    // So: v2v at strength 0 IS identity — the output sits 0.4/255 from a pure VAE round-trip of its own
    // source, while the VAE round-trip itself sits 27.6/255 from the source. The naive number `B` is
    // ~27.5 and means nothing on its own; essentially all of it is the **tiled** decode. (`A'` ~ 0 also
    // rules out the `.sample()` draw: this VAE's log-variance is small enough that sampling and the mode
    // agree.) Two things had to be right for `C` to be meaningful, and both are gated below: the control
    // must decode through the *same* tiling as the product path — a single-pass control puts A at ~3.0
    // and C at ~26 — and the VAE must actually be doing something.
    assert!(
        a > 1.0,
        "the VAE round-trip control is a near no-op (A={a:.2}) — C cannot be interpreted against it"
    );
    assert!(
        c < a * 0.1,
        "v2v@0 is {c:.2}/255 from a VAE round-trip of its own source, more than a tenth of the VAE's \
         own error (A={a:.2}) — either the strength=0 denoise is not an identity, or the control \
         decode no longer matches the product decode path"
    );
    assert!(b > 0.0, "B is exactly zero — the clips are the same object");
    assert_coherent(&r.frames, w, h, "v2v/strength=0");
}

/// **The decisive form of S7 minor #2: is v2v@0 identity in the LATENTS?**
///
/// The pixel test above cannot separate the AR loop from the VAE. This drives
/// [`generate_v2v_latents`] directly on VAE-encoded source latents at `strength = 0` and compares the
/// returned latents to the input — no decode, no encode noise, nothing between the source and the
/// answer. At `strength = 0` every scheduled timestep is 0, so the arithmetic says every Euler step and
/// every renoise is an identity; this is the measurement of whether the implementation agrees.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn v2v_strength_zero_is_latent_identity() {
    use mlx_gen_krea_realtime::{
        generate_v2v_latents, load_krea_realtime_transformer_with_quant, ArGenParams,
        CausalKreaTransformer,
    };
    use mlx_gen_wan::{load_tokenizer, preprocess_i2v_image, Umt5Encoder, WanVae};
    use mlx_rs::ops::concatenate_axis;
    use mlx_rs::Array;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);
    let (latent_h, latent_w) = (h / 8, w / 8);
    let latent_frames = (frames - 1) / 4 + 1;

    let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
    cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
    cfg.ar.frame_seq_length = (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
    cfg.ar.seq_length = latent_frames * cfg.ar.frame_seq_length;

    // Source latents, via the same VAE the pipeline uses.
    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let chw: Vec<Array> = (0..frames)
        .map(|i| {
            preprocess_i2v_image(&smooth_frame(w, h, i), w as u32, h as u32)
                .expect("preprocess")
                .expand_dims(1)
                .expect("expand")
        })
        .collect();
    let video = concatenate_axis(&chw.iter().collect::<Vec<_>>(), 1)
        .expect("stack")
        .expand_dims(0)
        .expect("batch");
    let z = vae.encode(&video).expect("VAE encode");
    let zs = z.shape().to_vec();
    let source_latents = z
        .reshape(&[zs[1], zs[2], zs[3], zs[4]])
        .expect("drop batch");
    drop(vae);

    // A real prompt context (the cross-attention path must be exercised, not zeroed).
    let tokenizer =
        load_tokenizer(root.join("tokenizer.json"), cfg.wan.text_len).expect("tokenizer");
    let mut tw = mlx_gen::weights::Weights::from_file(root.join("t5_encoder.safetensors"))
        .expect("open the TE");
    let context = {
        let enc = Umt5Encoder::from_weights_quantized(
            &mut tw,
            &cfg.wan,
            mlx_gen_wan::config::WanQuant {
                bits: 8,
                group_size: 64,
            },
        )
        .expect("UMT5");
        let c = enc
            .encode(&tokenizer, "a paper crane on a desk")
            .expect("encode");
        mlx_rs::transforms::eval([&c]).expect("eval context");
        c
    };

    let dw = mlx_gen::weights::Weights::from_file(root.join("dit.safetensors")).expect("open DiT");
    let raw: std::collections::HashMap<String, Array> = dw
        .keys()
        .map(|k| (k.to_string(), dw.get(k).expect("listed key").clone()))
        .collect();
    let (dit, _) = load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the DiT");
    let transformer = CausalKreaTransformer::new(dit, &cfg);

    let params = ArGenParams {
        seed: 3,
        steps: None,
        num_latent_frames: latent_frames,
        latent_height: latent_h,
        latent_width: latent_w,
        fps: 24,
    };
    let out = generate_v2v_latents(
        &transformer,
        &cfg,
        &context,
        &params,
        &source_latents,
        0.0,
        &mlx_gen::CancelFlag::default(),
        &mut |_| {},
    )
    .expect("v2v at strength 0");

    mlx_rs::transforms::eval([&out, &source_latents]).expect("materialize");
    let diff = mlx_rs::ops::subtract(&out, &source_latents)
        .expect("diff")
        .abs()
        .expect("abs");
    let max_abs = diff.max(None).expect("max").item::<f32>();
    let mean_abs = diff.mean(None).expect("mean").item::<f32>();
    let src_scale = source_latents
        .abs()
        .expect("abs")
        .mean(None)
        .expect("mean")
        .item::<f32>();
    println!(
        "  v2v@0 latents vs source: mean |Δ| = {mean_abs:.5}, max |Δ| = {max_abs:.5}, \
         source mean |x| = {src_scale:.5}  ({:.2}% of the source scale)",
        100.0 * mean_abs / src_scale
    );
    assert!(
        src_scale > 0.0,
        "the source latents are all zero — the comparison is inert"
    );
    assert!(
        mean_abs < 0.02 * src_scale,
        "v2v at strength=0 changed the latents by {mean_abs:.5} ({:.2}% of the source scale) — the \
         zero-strength schedule is NOT an identity on the latents",
        100.0 * mean_abs / src_scale
    );
}

/// **Measured KV-cache residency at the production geometry.** The self-attention KV cache holds
/// post-RoPE **activations**, so it is bf16 on every weight tier — a Q4 DiT does not shrink it, which
/// is the whole point of measuring it next to the Q4 weights. Drives the real DiT chunk-by-chunk with
/// the same bounded window the pipeline uses and reports the retained bytes plus the allocator delta.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn kv_cache_residency_at_the_production_geometry() {
    use mlx_gen_krea_realtime::{load_krea_realtime_transformer_with_quant, CausalKreaTransformer};
    use mlx_rs::Array;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 81);

    let (latent_h, latent_w) = (h / 8, w / 8);
    let latent_frames = (frames - 1) / 4 + 1;

    let mut cfg = KreaRealtimeConfig::krea_realtime_14b();
    cfg.ar.local_attn_size = cfg.ar.streaming_local_attn_frames() as i64;
    cfg.ar.frame_seq_length = (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
    cfg.ar.seq_length = latent_frames * cfg.ar.frame_seq_length;

    // Load the real DiT through the product load path.
    let weights =
        mlx_gen::weights::Weights::from_file(root.join("dit.safetensors")).expect("open the DiT");
    let raw: std::collections::HashMap<String, Array> = weights
        .keys()
        .map(|k| (k.to_string(), weights.get(k).expect("listed key").clone()))
        .collect();
    let (dit, packed) =
        load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the Krea DiT");
    println!("  DiT tier on disk: {packed:?}");
    let transformer = CausalKreaTransformer::new(dit, &cfg);

    let mut cache = transformer.new_cache();

    // A zero context is enough: this measures cache growth, not image quality.
    let ctx = Array::zeros::<f32>(&[cfg.wan.text_len as i32, cfg.wan.text_dim as i32])
        .expect("zero context");
    let embedded = transformer
        .inner()
        .embed_text(&ctx)
        .expect("embed the text context");
    let cross_kv = transformer
        .prepare_cross_kv(&embedded)
        .expect("cross-attention cache");
    // MLX safetensors loads are lazy, so the weights are not resident until something reads them.
    // The per-prompt cross-attention cache touches every block's `cross_attn.{k,v}` (and the text
    // embedding), which is enough to pull the packed DiT off disk — so the reading below is a real
    // staged-model figure rather than the ~0 an unevaluated graph reports.
    {
        let mut staged: Vec<&Array> = vec![&embedded];
        for (k, v) in &cross_kv {
            staged.push(k);
            staged.push(v);
        }
        mlx_rs::transforms::eval(staged).expect("stage the DiT");
    }
    let staged_resident = mlx_rs::memory::get_active_memory();

    let fpb = cfg.ar.num_frames_per_block as i32;
    let mut start = 0usize;
    let mut peak_retained_bytes = 0usize;
    let chunks = latent_frames.div_ceil(cfg.ar.num_frames_per_block);
    for c in 0..chunks {
        let chunk =
            Array::zeros::<f32>(&[cfg.wan.in_dim as i32, fpb, latent_h as i32, latent_w as i32])
                .expect("chunk");
        let velocity = transformer
            .forward_chunk(&chunk, 500.0, &cross_kv, start, &mut cache)
            .expect("causal chunk forward");
        start += (fpb as usize) * cfg.ar.frame_seq_length;
        // Sum the retained (k, v) bytes across every layer — the true residency, after eviction.
        //
        // MLX is LAZY: without this `eval` the whole chunk forward is an unexecuted graph, the cache
        // arrays are unmaterialized, and `get_active_memory()` reads ~0 — the measurement would be a
        // shape calculation dressed up as a memory reading. Forcing the cache (and the velocity that
        // depends on the same graph) makes the reported bytes and the allocator's active figure both
        // real, and makes the run take actual GPU time.
        let mut resident: Vec<&Array> = vec![&velocity];
        for l in 0..cache.num_layers() {
            if let Some((k, v)) = cache.layer_kv(l) {
                resident.push(k);
                resident.push(v);
            }
        }
        mlx_rs::transforms::eval(resident).expect("materialize the KV cache");
        let mut bytes = 0usize;
        for l in 0..cache.num_layers() {
            if let Some((k, v)) = cache.layer_kv(l) {
                for a in [k, v] {
                    bytes += a.nbytes();
                }
            }
        }
        peak_retained_bytes = peak_retained_bytes.max(bytes);
        println!(
            "  chunk {c:>2}: stored {} tok, retained {} tok, KV {:.2} GiB, MLX active {:.2} GiB",
            cache.stored_tokens(),
            cache.retained_tokens(),
            gib(bytes),
            gib(mlx_rs::memory::get_active_memory()),
        );
    }

    println!(
        "  KV-cache residency at {w}x{h} ({} tok/frame, window {} frames): {:.2} GiB \
         (MLX active after staging the DiT: {:.2} GiB)",
        cfg.ar.frame_seq_length,
        cfg.ar.streaming_local_attn_frames(),
        gib(peak_retained_bytes),
        gib(staged_resident),
    );
    assert!(
        peak_retained_bytes > 0,
        "the KV cache never retained anything — the measurement is inert"
    );
    // The bound is structural: retention can never exceed the read window plus one in-flight chunk.
    let max_tokens = cfg.ar.max_attention_size() + cfg.ar.block_size();
    assert!(
        cache.retained_tokens() <= max_tokens,
        "retained {} tokens > the bounded window's {max_tokens} — eviction is not bounding the cache",
        cache.retained_tokens()
    );
}
