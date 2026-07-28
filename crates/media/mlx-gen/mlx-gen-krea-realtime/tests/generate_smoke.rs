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
/// allocator's buffer cache. So this samples `active` and `active + cache` independently.
///
/// Two caveats the reported numbers must be read with:
/// * `active + cache` is **polled at 50 ms** and can miss a shorter spike, so it is a **lower bound**.
///   (`active` is not: it takes the max of the sampler and MLX's own exact high-water mark.)
/// * That cut both ways in the derivation. The cache is *reclaimable* — MLX frees cached buffers under
///   pressure to stay inside `get_memory_limit()` — so the figure a machine must actually satisfy is
///   the **active** peak, and `mlx.minMemoryGb` is derived from that. `active + cache` records what the
///   allocator was willing to hold when RAM was abundant; it is the reason not to quote
///   `get_peak_memory` as "the memory this uses", not itself the requirement.
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

/// Per-**column** mean |Δ| across the whole clip (sc-15325). A whole-frame mean averages a spatial
/// seam away: a tile boundary is a handful of columns out of hundreds, so a badly-blended spatial tile
/// can cost well under 0.1/255 on the frame mean while being plainly visible. This is the metric that
/// can *see* a seam — a spike at a multiple of the spatial tile stride is the signature.
fn mean_abs_delta_columns(a: &[Image], b: &[Image]) -> Vec<f64> {
    assert_eq!(a.len(), b.len(), "clip lengths differ");
    let w = a.first().map(|f| f.width as usize).unwrap_or(0);
    let mut sums = vec![0.0f64; w.max(1)];
    let mut n = vec![0usize; w.max(1)];
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x.pixels.len(), y.pixels.len(), "frame buffer sizes differ");
        for (i, (&p, &q)) in x.pixels.iter().zip(y.pixels.iter()).enumerate() {
            let col = (i / 3) % w.max(1);
            sums[col] += (p as f64 - q as f64).abs();
            n[col] += 1;
        }
    }
    sums.iter()
        .zip(n.iter())
        .map(|(s, &c)| s / c.max(1) as f64)
        .collect()
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
/// modes this model can show are **drift** (the tail saturates or washes out) and **freeze** (the tail
/// stops changing), neither of which a whole-clip min/max check can see. So the last third is checked
/// on its own terms.
///
/// This is a *structural* floor, not the long-clip coherence measurement — that is
/// [`long_clip_coherence_under_the_bounded_window`] (sc-15127), which measured that the drift mode is
/// **not** caused by the bounded KV window sliding with an empty sink.
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

/// **sc-15325 regression guard — the tiled decode must never starve the temporal receptive field.**
///
/// Weights-free arithmetic over the *product* policy (`decode_tiling`), at every bucket the old
/// policy collapsed at, under a budget too small for a full-frame single pass — i.e. exactly the
/// regime the defect lived in. The old policy emitted an 8-output-frame window = **2 latent frames**
/// (overlap 1, the clamp maximum there) at all of these, so this test is red on it by construction;
/// it is a property gate, not a snapshot of today's numbers.
///
/// The invariant: *if* a temporal tile is emitted, it spans ≥ `MIN_TEMPORAL_TILE_LATENT_FRAMES`
/// latent frames with ≥ `MIN_TEMPORAL_TILE_LATENT_OVERLAP` latent frames of blend. A plan with no
/// temporal tiling is trivially fine — the decoder sees the whole sequence.
#[test]
fn decode_tiling_never_starves_the_temporal_receptive_field() {
    use mlx_gen::tiling::{MIN_TEMPORAL_TILE_LATENT_FRAMES, MIN_TEMPORAL_TILE_LATENT_OVERLAP};
    const TEMPORAL_SCALE: i32 = 4; // VaeTiling::WAN

    // Pin the budget so this gates the policy, not the host's free memory.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "12");
    for (w, h) in [
        (832usize, 480usize),
        (640, 384),
        (512, 512),
        (768, 512),
        (1280, 720),
        (480, 832),
        (512, 384),
    ] {
        let cfg = decode_tiling(h, w, 84)
            .unwrap_or_else(|e| panic!("{w}x{h} must stay decodable within 12 GiB: {e}"))
            .unwrap_or_else(|| panic!("{w}x{h}/84f cannot fit a 12 GiB single pass"));
        let Some(t) = cfg.temporal else { continue };
        let lat_tile = t.tile_frames / TEMPORAL_SCALE;
        let lat_over = (t.overlap_frames / TEMPORAL_SCALE).min(lat_tile - 1);
        assert!(
            lat_tile >= MIN_TEMPORAL_TILE_LATENT_FRAMES,
            "{w}x{h}: temporal tile {} output frames = {lat_tile} LATENT frames, under the \
             {MIN_TEMPORAL_TILE_LATENT_FRAMES}-frame receptive-field floor — this is the sc-15325 \
             defect (2 latent frames measured 18.5/255 vs single-pass, 26.6% worst-frame clipping)",
            t.tile_frames
        );
        assert!(
            lat_over >= MIN_TEMPORAL_TILE_LATENT_OVERLAP,
            "{w}x{h}: temporal overlap {} output frames = {lat_over} LATENT frames after the tile-1 \
             clamp, under the {MIN_TEMPORAL_TILE_LATENT_OVERLAP}-frame blend floor",
            t.overlap_frames
        );
    }
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
}

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
    /// Mean denoise-step wall time over **non-boundary** intervals — the cost of one denoise step, with
    /// the unobservable KV-recompute forward excluded. This is the figure epic 8432 should use.
    mean_step: std::time::Duration,
    /// The same mean including boundary intervals (each of which also contains a KV recompute), kept so
    /// the inflation is visible rather than silently corrected away.
    mean_step_with_boundaries: std::time::Duration,
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
                    // NB: this fires at the FIRST `Progress::Step`, which is emitted *after* that
                    // step's compute and eval — so it already includes one full DiT forward and one
                    // chunk of KV, not just load + encode. Named for what it measures.
                    phase_peaks.push(("through the first denoise step", a, t));
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
    // Per-step time. An interval that SPANS a chunk boundary also contains the S5 clean-context
    // KV-recompute forward, which emits no `Progress` — including those inflates the per-step figure by
    // ~17% (measured 6.2 vs 5.3 s at 832x480). The realtime epic wants the cost of a denoise step, so
    // boundary intervals are excluded here and the recompute is reported inside the per-chunk time,
    // where it belongs.
    let all_intervals: Vec<std::time::Duration> = step_marks
        .windows(2)
        .map(|w| w[1].duration_since(w[0]))
        .collect();
    let per_chunk_steps = if expected_chunks > 0 && !step_marks.is_empty() {
        step_marks.len() / expected_chunks
    } else {
        0
    };
    let in_chunk: Vec<std::time::Duration> = all_intervals
        .iter()
        .enumerate()
        // interval i sits between step i and step i+1; it crosses a boundary when step i+1 starts a chunk
        .filter(|(i, _)| per_chunk_steps == 0 || !(i + 1).is_multiple_of(per_chunk_steps))
        .map(|(_, d)| *d)
        .collect();
    let mean_of = |v: &[std::time::Duration]| {
        if v.is_empty() {
            std::time::Duration::ZERO
        } else {
            v.iter().sum::<std::time::Duration>() / v.len() as u32
        }
    };
    let mean_step = mean_of(&in_chunk);
    let mean_step_with_boundaries = mean_of(&all_intervals);
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
        mean_step_with_boundaries,
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
        "  AR denoise   : {:.2?} ({:.2?}/step mean, excl. chunk boundaries; {:.2?} incl. — a \
         boundary interval also carries the KV-recompute forward, which emits no progress event)",
        r.denoise, r.mean_step, r.mean_step_with_boundaries
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

/// **The S13 product-path run:** timing, memory, and the *structural* coherence floor.
///
/// ⚠️ **This gate does NOT certify image quality, and the clip it produces is known to be visibly
/// corrupted** (sc-15325 — the tiled VAE decode blows ~26% of one frame in eight to near-white). The
/// assertions below check flatness, plausible mean, tail motion and gross brightness drift; **none of
/// those is violated by the corruption**, which is precisely why it went unnoticed in the first cut of
/// this work. `report_artifacts` is therefore called on the product clip so the clipping figure is at
/// least *printed* by the story's headline run — a green tick here means "the pipeline ran and produced
/// a structurally sane clip", not "the clip looks right". Tightening this into a real quality gate
/// depends on sc-15325 landing, since today it would fail by design.
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
    // Print what `assert_coherent` structurally cannot see (sc-15325). Compare the clipping figure
    // against the ~0.08% a single-pass decode of the same latents achieves.
    report_artifacts(&r.frames, "t2v/product-path");
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
    let tiling = decode_tiling(h, w, (latent_frames * 4) as i32).expect("plan the decode tiling");
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

/// **sc-15325 real-weight regression guard — the new policy reaches single-pass quality at the OLD
/// policy's memory peak; the old policy did not.**
///
/// Real z16 VAE, real encode/decode, the same latents decoded every way. Three arms:
///
///  * **single-pass** — the reference every tiled decode approximates (guarded to stay under the z16
///    write cap; past it MLX writes silently wrong pixels and the reference is garbage, sc-15402);
///  * **the OLD policy**, reproduced verbatim from the deleted `DECODE_TILE_BUDGET_PXFRAMES`
///    arithmetic — this is what makes the test a *guard* rather than a snapshot: it is red on the
///    shipped-before code by construction, because that code emitted exactly this config;
///  * **the NEW policy**, read from `decode_tiling` under a budget pinned to the OLD policy's measured
///    peak (~20 GiB at 832×480), so "did the fix cost memory?" is answered by the same run.
///
/// It then walks a candidate ladder (temporal-only vs. spatially-relieved) for the record, reporting
/// mean abs err, highlight clipping and MLX active peak per decode.
///
/// ⚠️ **Read this test's SPATIAL rows as a memory measurement, not as quality evidence.** Its source is
/// [`smooth_frame`] — a 5-6-cycle sinusoid with essentially no energy above DC, chosen so the VAE's own
/// round-trip error would not swamp the v2v comparisons this module also runs. That stimulus
/// *structurally cannot* exhibit a spatial seam or a starved spatial receptive field: there is no
/// high-frequency content for either to destroy. The peaks it reports are real (memory does not care
/// what the pixels are) and its temporal rows are diagnostic (the z16 collapse is a low-frequency
/// content failure, which this source does show). But the claim "shrinking the spatial tile is nearly
/// free" is evidenced by [`decode_tiling_sweep_against_single_pass`], which runs the same ladder on
/// **real generated latents** and adds a per-column metric that a whole-frame mean would hide.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn decode_policy_matches_single_pass_at_the_old_memory_peak() {
    use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig, VaeTiling};
    use mlx_gen_wan::{preprocess_i2v_image, WanVae};
    use mlx_rs::ops::concatenate_axis;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let frames = env_usize("KREA_SMOKE_FRAMES", 33);
    let latent_frames = (frames - 1) / 4 + 1;
    let out_frames = (latent_frames * 4) as i32;
    // ⚠️ A single-pass z16 decode is only a valid reference below the write cap (56 output frames at
    // 832×480). Past it MLX writes silently wrong pixels and every tiled candidate looks catastrophic.
    let write_cap = VaeTiling::WAN.writable_frame_cap(h as i32, w as i32);
    assert!(
        out_frames as i64 <= write_cap,
        "needs a valid single-pass reference: {out_frames} output frames exceeds the z16 write cap \
         {write_cap} at {w}x{h}"
    );

    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let chw: Vec<mlx_rs::Array> = (0..frames)
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
    let latents = z
        .reshape(&[zs[1], zs[2], zs[3], zs[4]])
        .expect("drop batch");

    let decode = |tiling: Option<&TilingConfig>| -> Vec<Image> {
        match decode_latents_to_video(
            &vae,
            &latents,
            24,
            Some(frames),
            tiling,
            &mlx_gen::CancelFlag::default(),
        )
        .expect("VAE decode")
        {
            GenerationOutput::Video { frames, .. } => frames,
            other => panic!("expected a Video output, got {other:?}"),
        }
    };
    // Peak per decode: "does the fix work" and "is the fix affordable" answered by the same run.
    let peak_of = |f: &dyn Fn() -> Vec<Image>| -> (Vec<Image>, usize) {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let out = f();
        (out, mlx_rs::memory::get_peak_memory())
    };

    /// The removed policy, verbatim: a 3.5e6 output px·frame budget snapped to the ×4 stride, with
    /// the sc-8446 overlap floor. At every bucket ≥ ~233k px/frame this is 8 output frames = 2 latent.
    fn old_policy(out_h: usize, out_w: usize, out_frames: i32) -> TilingConfig {
        const BUDGET_PXFRAMES: i64 = 3_500_000;
        let px_per_frame = (out_h as i64) * (out_w as i64);
        let budget_frames = BUDGET_PXFRAMES / px_per_frame.max(1);
        let write_cap = VaeTiling::WAN.writable_frame_cap(out_h as i32, out_w as i32);
        let win = budget_frames.min(write_cap) as i32;
        let tile_frames = (win / 4 * 4).clamp(8, out_frames.max(8));
        TilingConfig::temporal_only(tile_frames, (tile_frames / 4).max(4))
    }

    let (single, single_peak) = peak_of(&|| decode(None));
    let (single_clip_mean, single_clip_worst) = clip_stats(&single);
    println!(
        "  single-pass (reference): clipping {single_clip_mean:.2}% / {single_clip_worst:.2}%, \
         peak {:.2} GiB",
        gib(single_peak)
    );

    let old = old_policy(h, w, out_frames);
    let ot = old.temporal.expect("the old policy is temporal-only");
    let (old_out, old_peak) = peak_of(&|| decode(Some(&old)));
    let old_err = mean_abs_delta(&single, &old_out);
    let (old_clip_mean, old_clip_worst) = clip_stats(&old_out);
    println!(
        "  OLD policy tile {}/overlap {} (latent {}/{}): mean abs err {old_err:.2}/255, clipping \
         {old_clip_mean:.2}% / {old_clip_worst:.2}%, peak {:.2} GiB",
        ot.tile_frames,
        ot.overlap_frames,
        ot.tile_frames / 4,
        (ot.overlap_frames / 4).min(ot.tile_frames / 4 - 1),
        gib(old_peak)
    );

    // Pin the NEW policy's budget to what the OLD one actually cost, so the comparison is like-for-like
    // on memory and the quality result is not bought with a bigger machine.
    let budget_gib = env_usize("KREA_DECODE_BUDGET_GIB", gib(old_peak).ceil() as usize);
    std::env::set_var("WAN_VAE_BUDGET_GIB", budget_gib.to_string());
    let new = decode_tiling(h, w, out_frames)
        .expect("the new policy must be feasible at the old policy's peak")
        .expect("the clip must still tile at that budget");
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    let (new_out, new_peak) = peak_of(&|| decode(Some(&new)));
    let new_err = mean_abs_delta(&single, &new_out);
    let (new_clip_mean, new_clip_worst) = clip_stats(&new_out);
    println!(
        "  NEW policy {new:?} (budget pinned {budget_gib} GiB): mean abs err {new_err:.2}/255, \
         clipping {new_clip_mean:.2}% / {new_clip_worst:.2}%, peak {:.2} GiB",
        gib(new_peak)
    );
    dump_frames(&single, "policy_single_pass");
    dump_frames(&old_out, "policy_old");
    dump_frames(&new_out, "policy_new");

    // --- the record: temporal-only vs. spatially-relieved at the SAME latent tile ------------------
    println!("  --- candidate ladder (for the operating-point record) ---");
    let ladder: [(i32, i32, Option<i32>); 8] = [
        (8, 4, None),        // the old policy: latent 2 / 1
        (16, 8, None),       // latent 4 / 2
        (32, 8, None),       // latent 8 / 2, full frame
        (32, 16, None),      // latent 8 / 4, full frame
        (32, 16, Some(448)), // latent 8 / 4, spatially relieved
        (32, 16, Some(320)), // latent 8 / 4, spatially relieved further
        (32, 16, Some(256)), // latent 8 / 4, 32 latent px spatial tiles
        (32, 16, Some(192)), // latent 8 / 4, the SMALLEST spatial candidate (24 latent px)
    ];
    for (tile, overlap, tile_px) in ladder {
        let cfg = TilingConfig {
            spatial: tile_px.map(|tile_px| SpatialTiling {
                tile_px,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: tile,
                overlap_frames: overlap,
            }),
        };
        let (out, peak) = peak_of(&|| decode(Some(&cfg)));
        let (cm, cw) = clip_stats(&out);
        println!(
            "    latent tile {}/overlap {} spatial {:?}: mean abs err {:.2}/255, clipping \
             {cm:.2}% / {cw:.2}%, peak {:.2} GiB",
            tile / 4,
            overlap / 4,
            tile_px,
            mean_abs_delta(&single, &out),
            gib(peak)
        );
    }

    // --- gates ------------------------------------------------------------------------------------
    // 1. The harness reproduces the defect: the OLD policy is materially worse than single-pass. If
    //    this ever goes green the measurement stopped measuring, and gate 2 proves nothing.
    assert!(
        old_err > 5.0,
        "the OLD policy now matches single-pass to {old_err:.2}/255 — this guard is no longer \
         reproducing sc-15325 and gate 2 below is vacuous"
    );
    // 2. The NEW policy is at single-pass level, not merely better.
    assert!(
        new_err < 4.0 && new_err < old_err * 0.35,
        "the new decode policy is {new_err:.2}/255 from single-pass (old: {old_err:.2}) — sc-15325 \
         requires approximately single-pass quality at the shipping buckets"
    );
    // 3. ...on the metric a viewer actually sees. The old policy blew out whole frames.
    assert!(
        new_clip_worst <= single_clip_worst + 1.5,
        "the new policy's worst-frame highlight clipping is {new_clip_worst:.2}% against \
         {single_clip_worst:.2}% single-pass — highlights are still blowing out"
    );
    // 4. ...and it did not buy that with memory: the pinned budget must have been honoured.
    assert!(
        gib(new_peak) <= budget_gib as f64 * 1.15,
        "the new policy peaked at {:.2} GiB against a {budget_gib} GiB budget — the selector's cost \
         model is under-predicting and `mlx.minMemoryGb` cannot be trusted",
        gib(new_peak)
    );
}

/// **sc-15325 root cause: decode the SAME real t2v latents single-pass vs. every tiling candidate.**
///
/// This is the experiment the whole diagnosis rests on. It generates a real clip's latents once, then
/// decodes them repeatedly — single-pass (the reference every tiled decode approximates) and across a
/// sweep of `(tile_frames, overlap_frames)` — and reports, per decode: mean |Δ| vs single-pass, the
/// worst per-frame Δ and where it lands, highlight-clipping %, and mean saturation. If the shipped
/// tiling's period-8 artifacts vanish single-pass, the tiled decode is the mechanism; the sweep then
/// says whether the fix is *overlap* (a blending problem) or *tile size* (a temporal receptive-field
/// problem), which are different bugs with different fixes.
///
/// It also carries the **spatial** ladder, because the fix's affordability claim ("relieve the memory
/// on the spatial axis instead") has to be evidenced on the same real latents rather than on a
/// band-limited synthetic source that cannot show a seam. Measured (832×480, 36 output frames, latent
/// tile 8 / overlap 4):
///
/// | spatial tile | mean abs err | per-column mean abs err | clipping mean / worst | MLX active peak |
/// |---|---|---|---|---|
/// | none (full frame) | 1.954 /255 | 1.629 … 2.440 | 0.14 % / 1.05 % | 75.77 GiB |
/// | 448 px | 2.021 | 1.696 … 2.505 | 0.14 % / 1.04 % | 38.57 GiB |
/// | 320 px | 2.054 | 1.719 … 2.531 | 0.13 % / 1.04 % | 20.08 GiB |
/// | 256 px | 2.075 | 1.735 … 2.579 | 0.13 % / 1.04 % | 12.99 GiB |
/// | 192 px | 2.121 | 1.769 … 2.630 | 0.13 % / 1.02 % | 7.49 GiB |
///
/// **10.1× less memory for 0.17/255, and no seam**: the per-column error shifts uniformly rather than
/// spiking at the tile stride (worst column 1.24× the clip mean at the smallest tile). Contrast the
/// temporal axis on the same clip: latent 2 → 4 → 8 is 17.09 → 6.43 → 1.95 /255.
#[test]
#[ignore = "real snapshot; run with --ignored on macOS (see module doc)"]
fn decode_tiling_sweep_against_single_pass() {
    use mlx_gen::tiling::{SpatialTiling, TemporalTiling, TilingConfig};
    use mlx_gen_krea_realtime::{
        generate_latents, load_krea_realtime_transformer_with_quant, ArGenParams,
        CausalKreaTransformer,
    };
    use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanVae};
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

    // --- Generate one real clip's latents (the same content the shipped clip shows). ---
    let latents = {
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
                .encode(
                    &tokenizer,
                    "a red fox trotting through a snowy pine forest at sunrise, drifting snow, \
                     cinematic, shallow depth of field",
                )
                .expect("encode");
            mlx_rs::transforms::eval([&c]).expect("eval context");
            c
        };
        let dw =
            mlx_gen::weights::Weights::from_file(root.join("dit.safetensors")).expect("open DiT");
        let raw: std::collections::HashMap<String, Array> = dw
            .keys()
            .map(|k| (k.to_string(), dw.get(k).expect("listed key").clone()))
            .collect();
        let (dit, _) = load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the DiT");
        let transformer = CausalKreaTransformer::new(dit, &cfg);
        let params = ArGenParams {
            seed: 7,
            steps: None,
            num_latent_frames: latent_frames,
            latent_height: latent_h,
            latent_width: latent_w,
            fps: 24,
        };
        let l = generate_latents(
            &transformer,
            &cfg,
            &context,
            &params,
            &mlx_gen::CancelFlag::default(),
            &mut |_| {},
        )
        .expect("generate latents");
        mlx_rs::transforms::eval([&l]).expect("materialize latents");
        l
    };
    mlx_rs::memory::clear_cache();

    // --- Decode the SAME latents every way. ---
    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");
    let decode = |tiling: Option<&TilingConfig>| -> Vec<Image> {
        match decode_latents_to_video(
            &vae,
            &latents,
            24,
            Some(frames),
            tiling,
            &mlx_gen::CancelFlag::default(),
        )
        .expect("VAE decode")
        {
            GenerationOutput::Video { frames, .. } => frames,
            other => panic!("expected a Video output, got {other:?}"),
        }
    };

    // Peak is measured per decode: tile size is the decode's memory bound, so "is the fix
    // affordable" is answered by the same run that answers "does the fix work".
    let peak_of = |f: &dyn Fn() -> Vec<Image>| -> (Vec<Image>, usize) {
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let out = f();
        (out, mlx_rs::memory::get_peak_memory())
    };

    // ⚠️ The single-pass reference is only valid BELOW the z16 write bound. At 832x480 the cap is 56
    // output frames (`96 · f · h · w <= i32::MAX`); an 84-frame clip is 1.5x over it, and MLX writes
    // silently WRONG results past that — a corrupt reference that would make every tiled candidate look
    // catastrophically bad (measured: it collapses saturation 0.33 -> 0.07 and inflates every mean |Δ|
    // to 58-71/255). Refuse to compare against garbage.
    let out_frames_total = (latent_frames * 4) as i64;
    let write_cap = mlx_gen::tiling::VaeTiling::WAN.writable_frame_cap(h as i32, w as i32);
    assert!(
        out_frames_total <= write_cap,
        "this sweep needs a VALID single-pass reference, but {out_frames_total} output frames exceeds \
         the z16 write cap of {write_cap} at {w}x{h} — past it the untiled decode is silently wrong. \
         Re-run with KREA_SMOKE_FRAMES <= {}",
        write_cap - 3
    );

    let (single, single_peak) = peak_of(&|| decode(None));
    dump_frames(&single, "sweep_single_pass");
    println!("--- single-pass (the reference) ---");
    println!("    [single] MLX active peak {:.2} GiB", gib(single_peak));
    report_artifacts(&single, "single");

    // The product policy, pinned to a small budget so the sweep is comparable run-to-run rather than
    // reflecting whatever this host happened to have free.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "20");
    let shipped = decode_tiling(h, w, (latent_frames * 4) as i32)
        .expect("the product policy must be feasible")
        .expect("the clip must tile at a 20 GiB budget");
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
    println!("--- product policy at a 20 GiB budget: {shipped:?} ---");
    // ⚠️ The product plan is NOT necessarily temporal. Since sc-15325 the selector relieves memory on
    // the spatial axis, and at a short clip it will happily return a spatial-only plan — which is the
    // *best* possible answer for this defect (no temporal tiling at all ⇒ the decoder sees the whole
    // sequence). Assuming a temporal tile here is how this test first went red after the fix.
    let ship_t = shipped.temporal;
    let mut candidates: Vec<(i32, i32)> = Vec::new();
    if let Some(t) = ship_t {
        candidates.push((t.tile_frames, t.overlap_frames));
    }
    for c in [(8, 4), (8, 8), (16, 4), (16, 8), (32, 8), (32, 16)] {
        if !candidates.contains(&c) {
            candidates.push(c);
        }
    }

    // The full-frame latent-8/4 decode is the reference the SPATIAL ladder below is compared against
    // (it isolates "what does shrinking the spatial tile cost?" from "what does the temporal tile
    // cost?"), so keep it rather than paying a second 70-GiB-class decode for it.
    let mut full_frame_32_16: Option<(Vec<Image>, usize)> = None;
    for (tile, overlap) in candidates {
        let cfg = TilingConfig {
            spatial: None,
            temporal: Some(TemporalTiling {
                tile_frames: tile,
                overlap_frames: overlap,
            }),
        };
        let (out, peak) = peak_of(&|| decode(Some(&cfg)));
        let d = mean_abs_delta(&single, &out);
        let per: Vec<f64> = single
            .iter()
            .zip(out.iter())
            .map(|(a, b)| {
                a.pixels
                    .iter()
                    .zip(b.pixels.iter())
                    .map(|(&x, &y)| (x as f64 - y as f64).abs())
                    .sum::<f64>()
                    / a.pixels.len() as f64
            })
            .collect();
        let (worst_i, worst) =
            per.iter().enumerate().fold(
                (0usize, 0.0f64),
                |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc },
            );
        let is_product =
            ship_t.is_some_and(|t| (t.tile_frames, t.overlap_frames) == (tile, overlap));
        let label = if is_product {
            format!("tile {tile:>2} / overlap {overlap:>2}  [PRODUCT temporal tile]")
        } else {
            format!("tile {tile:>2} / overlap {overlap:>2}")
        };
        println!(
            "  {label}: latent tile {} / overlap {} -> mean |Δ| {d:.2}/255, worst frame {worst_i} \
             at {worst:.2}",
            tile / 4,
            overlap / 4
        );
        println!(
            "    [t{tile}o{overlap}] MLX active peak {:.2} GiB",
            gib(peak)
        );
        report_artifacts(&out, &format!("t{tile}o{overlap}"));
        if is_product {
            dump_frames(&out, "sweep_product_temporal");
        }
        if (tile, overlap) == (32, 16) {
            dump_frames(&out, "sweep_best");
            full_frame_32_16 = Some((out, peak));
        }
    }

    // --- the SPATIAL ladder, on the same real latents ---------------------------------------------
    //
    // sc-15325's affordability claim ("the memory floor drops ~6-10× and it costs almost nothing")
    // used to be evidenced only by `decode_policy_matches_single_pass_at_the_old_memory_peak`, whose
    // source is `smooth_frame` — a 5-6-cycle sinusoid with essentially zero energy above DC. That
    // stimulus *structurally cannot* show a spatial seam or a starved spatial receptive field, so it
    // could not establish the claim it was cited for. These rows are the real-latent version: same
    // clip, same latents, same latent tile 8 / overlap 4, varying only the spatial tile.
    //
    // Both metrics are reported. The whole-frame mean answers "how much error"; the per-COLUMN mean
    // answers "is any of it concentrated at a tile boundary", which the frame mean averages away.
    let (full_out, full_peak) =
        full_frame_32_16.expect("the (32, 16) full-frame row must have run");
    let full_err = mean_abs_delta(&single, &full_out);
    let full_cols = mean_abs_delta_columns(&single, &full_out);
    let col_span = |c: &[f64]| -> (f64, f64) {
        let lo = c.iter().cloned().fold(f64::MAX, f64::min);
        let hi = c.iter().cloned().fold(0.0f64, f64::max);
        (lo, hi)
    };
    let (full_lo, full_hi) = col_span(&full_cols);
    println!("  --- spatial ladder at latent tile 8 / overlap 4 (real latents) ---");
    println!(
        "    spatial none (full frame): mean |Δ| {full_err:.3}/255, per-column {full_lo:.3}..\
         {full_hi:.3}, peak {:.2} GiB",
        gib(full_peak)
    );
    let mut spatial_rows: Vec<(i32, f64, f64, f64)> = Vec::new(); // (px, err, col_hi, peak_gib)
    for tile_px in [448i32, 320, 256, 192] {
        let cfg = TilingConfig {
            spatial: Some(SpatialTiling {
                tile_px,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 32,
                overlap_frames: 16,
            }),
        };
        let (out, peak) = peak_of(&|| decode(Some(&cfg)));
        let err = mean_abs_delta(&single, &out);
        let cols = mean_abs_delta_columns(&single, &out);
        let (lo, hi) = col_span(&cols);
        let (cm, cw) = clip_stats(&out);
        println!(
            "    spatial {tile_px:>3} px: mean |Δ| {err:.3}/255, per-column {lo:.3}..{hi:.3}, \
             clipping {cm:.2}% / {cw:.2}%, peak {:.2} GiB",
            gib(peak)
        );
        spatial_rows.push((tile_px, err, hi, gib(peak)));
    }

    // Gate 1 — the spatial axis is nearly free. The SMALLEST candidate must stay close to the
    // full-frame decode at the same temporal tile; if shrinking the spatial tile ever starts costing
    // real error, "relieve memory spatially" stops being the right answer and the floor has to become
    // a memory trade again.
    let (small_px, small_err, small_hi, small_peak) = *spatial_rows
        .last()
        .expect("the ladder ran at least one row");
    assert!(
        small_err <= full_err + 1.0,
        "the smallest spatial candidate ({small_px} px) is {small_err:.3}/255 against \
         {full_err:.3}/255 full-frame at the same latent tile — spatial relief is no longer nearly \
         free, so sc-15325's affordability argument needs re-deriving"
    );
    // Gate 2 — and none of that error is a SEAM. A trapezoidally blended spatial tile should raise
    // every column a little, not spike at the stride. Compared against the row's own column floor so
    // this measures concentration, not content.
    assert!(
        small_hi <= small_err * 2.0,
        "the {small_px} px spatial plan has a worst column of {small_hi:.3}/255 against a clip mean \
         of {small_err:.3} — that concentration is the signature of a spatial seam, which the \
         whole-frame mean would have hidden"
    );
    // Gate 3 — and it actually bought the memory it is claimed to buy.
    assert!(
        small_peak < gib(full_peak) * 0.5,
        "the {small_px} px spatial plan peaked at {small_peak:.2} GiB against {:.2} GiB full-frame — \
         sc-15325's claim is that the spatial axis is where the memory comes from",
        gib(full_peak)
    );

    // sc-15325 is fixed, so the gate inverts: the full plan the PRODUCT emits — spatial and temporal
    // together, whatever the selector chose — must now track single-pass. (The `8 / 4` row above still
    // shows the old 2-latent-frame window failing: the corruption is reproducible on demand, it is
    // simply no longer selectable.) The old window is re-measured here as the control, so a run where
    // this gate passes because the *harness* stopped discriminating is caught rather than believed.
    let (product_out, product_peak) = peak_of(&|| decode(Some(&shipped)));
    let d_product = mean_abs_delta(&single, &product_out);
    dump_frames(&product_out, "sweep_product_plan");
    println!(
        "  PRODUCT plan {shipped:?}: mean abs err {d_product:.2}/255, peak {:.2} GiB",
        gib(product_peak)
    );
    report_artifacts(&product_out, "product");

    let old_window = TilingConfig {
        spatial: None,
        temporal: Some(TemporalTiling {
            tile_frames: 8,
            overlap_frames: 4,
        }),
    };
    let d_old = mean_abs_delta(&single, &decode(Some(&old_window)));
    assert!(
        d_old > 5.0,
        "the pre-sc-15325 window is now only {d_old:.2}/255 from single-pass — this sweep has \
         stopped reproducing the defect, so the gate below proves nothing"
    );
    assert!(
        d_product < 4.0 && d_product < d_old * 0.35,
        "the product decode plan is {d_product:.2}/255 from single-pass (old window: {d_old:.2}) — \
         sc-15325 has regressed"
    );
}

/// Highlight-clipping statistics, returned rather than printed: `(mean %, worst-frame %)` of pixels
/// whose brightest channel is >= 250. This is what a viewer sees blow out, and it is the metric that
/// separates a starved tile from a healthy one — so a test has to be able to assert on it (sc-15325).
fn clip_stats(frames: &[Image]) -> (f64, f64) {
    let pcts: Vec<f64> = frames
        .iter()
        .map(|f| {
            let n = f.pixels.len() / 3;
            let clipped = f
                .pixels
                .chunks_exact(3)
                .filter(|px| px.iter().map(|&c| c as u32).max().unwrap_or(0) >= 250)
                .count();
            100.0 * clipped as f64 / n.max(1) as f64
        })
        .collect();
    if pcts.is_empty() {
        return (0.0, 0.0);
    }
    let mean = pcts.iter().sum::<f64>() / pcts.len() as f64;
    let worst = pcts.iter().cloned().fold(0.0f64, f64::max);
    (mean, worst)
}

/// Per-clip artifact statistics the naive min/max/mean misses: highlight clipping and saturation are
/// what a viewer actually sees blow out, and the period is what identifies the mechanism.
fn report_artifacts(frames: &[Image], label: &str) {
    let mut clip_pcts = Vec::with_capacity(frames.len());
    let mut sats = Vec::with_capacity(frames.len());
    for f in frames {
        let n = f.pixels.len() / 3;
        let mut clipped = 0usize;
        let mut sat_sum = 0.0f64;
        for px in f.pixels.chunks_exact(3) {
            let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
            let mx = r.max(g).max(b);
            let mn = r.min(g).min(b);
            if mx >= 250.0 {
                clipped += 1;
            }
            sat_sum += if mx > 0.0 { (mx - mn) / mx } else { 0.0 };
        }
        clip_pcts.push(100.0 * clipped as f64 / n as f64);
        sats.push(sat_sum / n as f64);
    }
    let worst_clip = clip_pcts.iter().cloned().fold(0.0f64, f64::max);
    let worst_i = clip_pcts
        .iter()
        .enumerate()
        .fold(
            (0usize, 0.0f64),
            |a, (i, &v)| if v > a.1 { (i, v) } else { a },
        )
        .0;
    let mean_clip = clip_pcts.iter().sum::<f64>() / clip_pcts.len() as f64;
    let sat0 = sats.first().copied().unwrap_or(0.0);
    let satn = sats.last().copied().unwrap_or(0.0);
    println!(
        "    [{label}] clipping: mean {mean_clip:.2}% / worst {worst_clip:.2}% at frame {worst_i}; \
         saturation {sat0:.3} -> {satn:.3}"
    );
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
    // MLX safetensors loads are lazy, so weights are not resident until something reads them. This
    // evaluation touches only the text embedding and every block's `cross_attn.{k,v}` — **2 of the 10
    // per-block Linears and none of the FFN, roughly 15% of the DiT's bytes** (~1.2 of 7.8 GiB at Q4).
    // So the figure below is a PARTIAL-staging reading, not the DiT's footprint; it is reported as such
    // and the full residency is the per-chunk `MLX active` column, which is measured after a real
    // forward has pulled everything in.
    {
        let mut staged: Vec<&Array> = vec![&embedded];
        for (k, v) in &cross_kv {
            staged.push(k);
            staged.push(v);
        }
        mlx_rs::transforms::eval(staged).expect("stage the DiT");
    }
    let partially_staged = mlx_rs::memory::get_active_memory();

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
         (MLX active after PARTIAL staging -- text embed + cross-attn k/v only, ~15% of the DiT: \
         {:.2} GiB)",
        cfg.ar.frame_seq_length,
        cfg.ar.streaming_local_attn_frames(),
        gib(peak_retained_bytes),
        gib(partially_staged),
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

// ---------------------------------------------------------------------------------------------
// sc-15127 (S18) — long-clip coherence under the bounded KV window
// ---------------------------------------------------------------------------------------------
//
// The question is NOT "does the clip change over time" — a video is supposed to. It is "does the
// clip's *established scene* survive the KV window rolling out from under it". So the metric is
// built to be blind to ordinary motion and sensitive to a one-way excursion:
//
//   * The descriptor is a set of **scene-level** statistics that ordinary camera/subject motion
//     perturbs in a bounded, oscillating way, and that a degenerating AR loop moves one way. It spans
//     three axes deliberately, because a single-axis descriptor has large blind spots (see
//     [`the_drift_metric_catches_the_plausible_ar_failure_shapes`]): **tone** (luma mean, contrast),
//     **colour** (saturation and the two opponent-colour means, which a hue rotation moves and a
//     luma/saturation descriptor cannot see), and **space** (the spread of a 5x5 grid of block luma
//     means, which a localized subject collapse moves and every global moment cannot see).
//   * The **baseline** is the pre-roll segment: the output frames produced before the first KV
//     eviction. Those frames saw the full history, so their frame-to-frame fluctuation IS the
//     ordinary-motion scale for this clip, measured on this clip.
//   * The verdict statistic is **trend AND excursion**, both gated. The OLS trend catches a ramp; it
//     is structurally blind to a *plateau* (the loop settling into a degenerate attractor, which is
//     the most likely AR failure), which is exactly what the excursion catches. The excursion's own
//     failure mode — an oscillating scene whose baseline sits at an extreme of its cycle — is handled
//     by suppressing it when it is small relative to the pre-roll fluctuation scale (`z`), not by
//     discarding the statistic.
//   * The **per-roll-bucket** table is printed alongside, so a non-linear collapse is visible rather
//     than averaged into a slope.
//
// **What is NOT available as a control, and why the budget is absolute.** A bounded window over a long
// clip evicts *by definition* — so there is no such thing as a zero-roll run at the shipped window and
// the shipped clip length. The checkpoint's global window (row E) is *not* that null: changing
// `local_attn_size` changes the attention mask, hence the sampling trajectory, hence the video. Rows
// are not the same clip; they are the same *prompt and seed* under different attention. Row E is
// retained as an out-of-regime reference (it is what the Mac bound replaces) and is explicitly **not**
// a motion floor. What remains valid:
//   * a **within-regime dose-response**: row D is the same bounded local-attention path at a 2.5×
//     wider window (fewer rolls). If rolling causes drift, fewer rolls must drift less.
//   * a **within-regime zero-eviction rate floor** (row Z): the shipped window on a clip short enough
//     that it never rolls. Its clip is too short for a length-scaled trend to be comparable, so what it
//     contributes is a *slope* (per 100 output frames) — the rate ordinary motion moves the descriptor
//     under this exact attention regime.
//   * an **absolute budget** ([`DRIFT_BUDGET`]), justified against the metric's measured floor and its
//     measured response to the plausible failure shapes, and gated in CI on both.

/// Grid resolution for the spatial component of [`scene_descriptor`]: `SPATIAL_GRID`^2.
const SPATIAL_GRID: usize = 5;

/// Number of components in [`scene_descriptor`].
const N_DESC: usize = 6;

/// Per-frame scene descriptor in 0..255 units.
///
/// Three axes, because each covers a failure shape the others are blind to:
/// * **tone** — `luma-mean`, `contrast` (luma std).
/// * **colour** — `saturation` (max−min chroma), and the two opponent means `R−G` and `B−(R+G)/2`.
///   A hue rotation at constant luma and constant saturation moves *only* the opponent pair; without
///   them the descriptor cannot see a colour cast developing.
/// * **space** — `spatial-sd`, the standard deviation of a `SPATIAL_GRID`^2 grid of block luma means.
///   A subject collapsing in the centre while the surround compensates leaves every *global* moment
///   unchanged; it moves this one hard.
///
/// Deliberately still statistics, not a per-pixel comparison against frame 0 — that measures motion,
/// which is exactly the signal that must NOT drive this conclusion.
fn scene_descriptor(f: &Image) -> [f64; N_DESC] {
    let w = (f.width as usize).max(1);
    let h = (f.height as usize).max(1);
    let n = (f.pixels.len() / 3).max(1);
    let (mut sum, mut sumsq, mut sat, mut rg, mut by) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut blk = [0.0f64; SPATIAL_GRID * SPATIAL_GRID];
    let mut blk_n = [0usize; SPATIAL_GRID * SPATIAL_GRID];
    for (i, px) in f.pixels.chunks_exact(3).enumerate() {
        let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        sum += y;
        sumsq += y * y;
        sat += r.max(g).max(b) - r.min(g).min(b);
        rg += r - g;
        by += b - 0.5 * (r + g);
        let bx = (i % w) * SPATIAL_GRID / w;
        let byi = (i / w).min(h - 1) * SPATIAL_GRID / h;
        let bi = byi.min(SPATIAL_GRID - 1) * SPATIAL_GRID + bx.min(SPATIAL_GRID - 1);
        blk[bi] += y;
        blk_n[bi] += 1;
    }
    let mean = sum / n as f64;
    let var = (sumsq / n as f64 - mean * mean).max(0.0);
    let block_means: Vec<f64> = blk
        .iter()
        .zip(blk_n.iter())
        .filter(|(_, &c)| c > 0)
        .map(|(&s, &c)| s / c as f64)
        .collect();
    [
        mean,
        var.sqrt(),
        sat / n as f64,
        rg / n as f64,
        by / n as f64,
        std_f64(&block_means),
    ]
}

/// Names of the [`scene_descriptor`] components, for the printed table.
const DESCRIPTOR_NAMES: [&str; N_DESC] = [
    "luma-mean",
    "contrast",
    "saturation",
    "opp-R-G",
    "opp-B-Y",
    "spatial-sd",
];

/// The absolute one-way drift budget, in 0..255 units, for both the trend and the z-gated excursion.
///
/// **Absolute, not derived from a control row** — see the module comment: no zero-roll run exists at
/// the shipped window and the shipped clip length, so there is no valid within-regime floor to subtract.
/// The number is instead pinned from both sides, and both sides are gated in CI by
/// [`the_drift_metric_separates_drift_from_ordinary_motion`] and
/// [`the_drift_metric_catches_the_plausible_ar_failure_shapes`]:
/// * **from below** — the metric reads a violently moving clip and a violently jittery clip at
///   ≤ 5/255, and the within-regime zero-eviction row Z bounds this content's own rate;
/// * **from above** — every plausible AR failure shape (linear wash-out ramp, plateauing wash-out,
///   localized subject collapse, hue rotation) scores above it, several of them by 2–6×.
///
/// The two sides are measured, not asserted, and they bracket a **narrow** range:
/// * floor — motion and jitter both score **2.81/255**;
/// * ceiling — the *weakest* of the four failure shapes (a 180° hue rotation at constant luma and
///   constant chroma magnitude, whose scene-mean chroma partially cancels across pixels) scores
///   **11.37/255**. Everything else scores 21–80.
///
/// So the budget must lie in `(2.81, 11.37)`. 8/255 ≈ 3.1% of the range sits at 2.85× the measured
/// floor and 0.70× the weakest detection, and is about where a *sustained one-way* shift in scene
/// luma, colour cast or spatial structure starts to read as the picture changing rather than the
/// subject moving. Both margins are asserted in the two gates above, so this cannot silently rot into
/// a number that admits gross drift or rejects ordinary motion.
const DRIFT_BUDGET: f64 = 8.0;

/// Minimum `|z|` for an excursion to count. Below this the tail's offset from the baseline is inside
/// the clip's own pre-roll fluctuation, i.e. it is where a moving scene happens to be in its cycle, and
/// gating on it would flag every real clip. (Measured on the synthetic motion control, whose raw
/// excursion is large and whose `z` is not.)
const EXCURSION_Z_MIN: f64 = 3.0;

fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_f64(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean_f64(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Ordinary-least-squares slope of `y` against its own index, scaled to **units per 100 samples**.
fn slope_per_100(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let n = y.len() as f64;
    let xbar = (n - 1.0) / 2.0;
    let ybar = mean_f64(y);
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &v) in y.iter().enumerate() {
        let dx = i as f64 - xbar;
        num += dx * (v - ybar);
        den += dx * dx;
    }
    if den == 0.0 {
        0.0
    } else {
        100.0 * num / den
    }
}

/// The drift summary for one descriptor component of one clip.
#[derive(Clone)]
struct ComponentDrift {
    /// The total *one-way* movement the rolling portion of the clip accumulates, in 0..255 units: the
    /// OLS trend over the post-roll segment × that segment's length.
    ///
    /// Catches a ramp. **Structurally blind to a plateau** — a loop that washes out by frame 54 and
    /// then holds has a small trend and a large [`excursion`](Self::excursion), which is why both are
    /// gated.
    trend: f64,
    /// Signed offset of the clip's final segment from the pre-roll baseline, in 0..255 units.
    ///
    /// Catches a plateau. Its own failure mode is an oscillating (i.e. ordinary, moving) scene whose
    /// baseline window happens to sit at an extreme of its cycle: that shows a large offset while going
    /// nowhere — measured at +11.7/255 on the synthetic motion control. That is handled by
    /// [`gated_excursion`](Self::gated_excursion), not by discarding the statistic.
    excursion: f64,
    /// [`excursion`](Self::excursion) in units of the pre-roll frame-to-frame fluctuation.
    z: f64,
    /// OLS slope over the post-roll segment, per 100 output frames. **Length-independent**, so this is
    /// the statistic that can be compared across rows of different clip length (row Z).
    slope_post: f64,
    /// OLS slope over the pre-roll baseline segment, per 100 output frames.
    slope_pre: f64,
    /// Per-bucket signed excursion from the baseline (bucket = one window-roll's worth of frames).
    /// Printed so a non-linear collapse is visible rather than averaged into a single slope.
    buckets: Vec<f64>,
}

impl ComponentDrift {
    /// [`excursion`](Self::excursion), suppressed when it is inside the clip's own pre-roll
    /// fluctuation scale (see [`EXCURSION_Z_MIN`]).
    fn gated_excursion(&self) -> f64 {
        if self.z.abs() < EXCURSION_Z_MIN {
            0.0
        } else {
            self.excursion
        }
    }
}

/// Score one clip's drift. `pre_len` is the number of leading output frames generated *before* the
/// first KV eviction (the baseline), `bucket` the frames one window roll contributes.
fn score_drift(series: &[f64], pre_len: usize, bucket: usize) -> ComponentDrift {
    let pre_len = pre_len.min(series.len()).max(1);
    let pre = &series[..pre_len];
    let base = mean_f64(pre);
    // Floor the motion scale so a pathologically still baseline cannot manufacture an infinite z.
    // 0.25/255 is well under any perceptible change and under this decode's own noise floor.
    let scale = std_f64(pre).max(0.25);
    let tail_start = series.len().saturating_sub(pre_len).max(pre_len);
    let tail = &series[tail_start.min(series.len().saturating_sub(1))..];
    let excursion = mean_f64(tail) - base;
    let mut buckets = Vec::new();
    let bucket = bucket.max(1);
    let mut i = pre_len;
    while i < series.len() {
        let end = (i + bucket).min(series.len());
        buckets.push(mean_f64(&series[i..end]) - base);
        i = end;
    }
    let post = &series[pre_len.min(series.len())..];
    let slope_post = slope_per_100(post);
    ComponentDrift {
        trend: slope_post * post.len() as f64 / 100.0,
        excursion,
        z: excursion / scale,
        slope_post,
        slope_pre: slope_per_100(pre),
        buckets,
    }
}

/// Score a whole clip: one [`ComponentDrift`] per [`scene_descriptor`] component.
fn score_clip(frames: &[Image], pre_len: usize, bucket: usize) -> [ComponentDrift; N_DESC] {
    let d: Vec<[f64; N_DESC]> = frames.iter().map(scene_descriptor).collect();
    std::array::from_fn(|c| {
        let series: Vec<f64> = d.iter().map(|x| x[c]).collect();
        score_drift(&series, pre_len, bucket)
    })
}

/// Largest absolute one-way component trend, in 0..255 units.
fn worst_trend(d: &[ComponentDrift; N_DESC]) -> f64 {
    d.iter().map(|c| c.trend.abs()).fold(0.0, f64::max)
}

/// Largest absolute **z-gated** component excursion, in 0..255 units.
fn worst_excursion(d: &[ComponentDrift; N_DESC]) -> f64 {
    d.iter()
        .map(|c| c.gated_excursion().abs())
        .fold(0.0, f64::max)
}

/// Largest absolute component slope over the rolling portion, per 100 output frames. This is the only
/// statistic comparable across clips of *different length* (row Z is 24 output frames, row A is 180).
fn worst_slope(d: &[ComponentDrift; N_DESC]) -> f64 {
    d.iter().map(|c| c.slope_post.abs()).fold(0.0, f64::max)
}

/// The clip's headline drift figure: worst trend **or** worst gated excursion, whichever is larger.
/// Gating on the max is what makes the budget a "trend AND excursion" gate.
fn worst_drift(d: &[ComponentDrift; N_DESC]) -> f64 {
    worst_trend(d).max(worst_excursion(d))
}

fn print_drift(label: &str, d: &[ComponentDrift; N_DESC]) {
    for (name, c) in DESCRIPTOR_NAMES.iter().zip(d.iter()) {
        println!(
            "    {label} {name:<11}: TREND {:+7.2}/255 | excursion {:+7.2} (z {:+6.2}, gated \
             {:+7.2}), slope pre {:+7.3} -> post {:+7.3} per 100f",
            c.trend,
            c.excursion,
            c.z,
            c.gated_excursion(),
            c.slope_pre,
            c.slope_post
        );
        let b: Vec<String> = c.buckets.iter().map(|v| format!("{v:+.2}")).collect();
        println!("      per-roll buckets: [{}]", b.join(", "));
    }
}

/// Apply a per-frame pixel transform, for building the synthetic drift stimuli.
fn map_frames(
    n: usize,
    w: usize,
    h: usize,
    mut f: impl FnMut(usize, f64, &mut Image),
) -> Vec<Image> {
    (0..n)
        .map(|i| {
            let t = if n > 1 {
                i as f64 / (n - 1) as f64
            } else {
                0.0
            };
            let mut img = smooth_frame(w, h, i);
            f(i, t, &mut img);
            img
        })
        .collect()
}

/// **CI gate for the metric itself.** A metric that cannot tell drift from motion is not evidence, so
/// this drives [`score_clip`] on two synthetic clips with the *same* total excursion budget: one that
/// oscillates (ordinary motion) and one that ramps one way (drift). The gate is that the drift score
/// separates them — it is red if the descriptor, the baseline, or the bucketing stops discriminating.
///
/// It also pins the **lower** side of [`DRIFT_BUDGET`]: motion and noise must both land well under it,
/// or the budget is not a budget.
#[test]
fn the_drift_metric_separates_drift_from_ordinary_motion() {
    const W: usize = 128;
    const H: usize = 96;
    const N: usize = 180;
    const PRE: usize = 24;
    const BUCKET: usize = 12;

    // Ordinary motion: a moving scene whose global statistics oscillate. Amplitude is deliberately
    // LARGE (the frames swing right across the range) so this is not passing by being static.
    let motion: Vec<Image> = (0..N).map(|i| smooth_frame(W, H, i)).collect();
    let m = score_clip(&motion, PRE, BUCKET);
    print_drift("motion", &m);

    // Drift: the same clip with a one-way wash-out applied — luma pulled up and saturation pulled
    // down, both proportional to the frame index, i.e. exactly the bounded-window AR failure mode.
    let drift = map_frames(N, W, H, |_, t, f| {
        for px in f.pixels.chunks_exact_mut(3) {
            let g = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
            for c in px.iter_mut() {
                // Pull each channel toward a rising grey: desaturates and brightens together.
                let v = *c as f64;
                *c = (v + t * 0.85 * (g + 60.0 - v)).clamp(0.0, 255.0) as u8;
            }
        }
    });
    let d = score_clip(&drift, PRE, BUCKET);
    print_drift("drift", &d);

    let m_worst = worst_drift(&m);
    let d_worst = worst_drift(&d);
    assert!(
        m_worst * 2.5 < DRIFT_BUDGET,
        "the ordinary-motion clip scored {m_worst:.2}/255 against a {DRIFT_BUDGET:.2} budget — that \
         is under a 2.5x margin, so the budget is not comfortably above the metric's own motion \
         floor and a no-drift verdict would be resting on noise"
    );
    assert!(
        d_worst > 25.0,
        "the one-way wash-out clip scored only {d_worst:.2}/255 — the metric cannot see the drift \
         mode it exists to detect, so a green long-clip run would prove nothing"
    );
    assert!(
        d_worst > m_worst * 4.0,
        "drift {d_worst:.2}/255 vs motion {m_worst:.2}/255 — the separation is too small for the \
         long-clip verdict to rest on"
    );
    // The RAW head->tail offset is explicitly not gated: on this very stimulus it reads the moving clip
    // as drifting harder than several plausible real-drift budgets. That is what `EXCURSION_Z_MIN`
    // exists for, and this pins both halves — the raw offset is big, and the gated one is not.
    let m_offset = m.iter().map(|c| c.excursion.abs()).fold(0.0, f64::max);
    assert!(
        m_offset > worst_trend(&m) * 3.0,
        "the moving control's raw offset ({m_offset:.2}/255) is no longer much larger than its trend \
         ({:.2}) — this stimulus has stopped demonstrating why the raw offset is unusable, so the \
         z-gate is no longer evidenced",
        worst_trend(&m)
    );
    assert!(
        worst_excursion(&m) < DRIFT_BUDGET,
        "the z-gate let the moving control's offset through at {:.2}/255 — EXCURSION_Z_MIN is too \
         low to keep ordinary motion out of the verdict",
        worst_excursion(&m)
    );
    // A second, INDEPENDENT way the metric could be fooled: per-frame noise. A jittery clip has huge
    // frame-to-frame variance and goes nowhere, and a metric that reported that as drift would flag
    // every real clip. Same stimulus, plus a large zero-mean per-frame brightness jitter.
    let mut lcg = 0x2545_F491_4F6C_DD1Du64;
    let jitter = map_frames(N, W, H, |_, _, f| {
        lcg = lcg
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((lcg >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 60.0;
        for c in f.pixels.iter_mut() {
            *c = (*c as f64 + j).clamp(0.0, 255.0) as u8;
        }
    });
    let j = score_clip(&jitter, PRE, BUCKET);
    print_drift("jitter", &j);
    let j_worst = worst_drift(&j);
    assert!(
        j_worst < DRIFT_BUDGET,
        "a zero-mean per-frame jitter scored {j_worst:.2}/255 — the metric is reading noise as \
         drift, so it cannot support a drift verdict either"
    );
    // ...and that jitter really was violent, so this control is not passing by being tame.
    let j_delta = mean_f64(
        &frame_stats(&jitter)
            .iter()
            .skip(1)
            .map(|s| s.delta)
            .collect::<Vec<_>>(),
    );
    let m_delta = mean_f64(
        &frame_stats(&motion)
            .iter()
            .skip(1)
            .map(|s| s.delta)
            .collect::<Vec<_>>(),
    );
    assert!(
        j_delta > m_delta * 2.0,
        "the jitter control moves {j_delta:.2}/255 per frame against the plain motion clip's \
         {m_delta:.2} — it is not the noisy stimulus it claims to be"
    );
}

/// **CI gate for the metric's blind spots.** The linear full-amplitude ramp above is the shape an OLS
/// trend is *maximally* good at; passing on it is not a sensitivity floor. This drives [`score_clip`]
/// on three failure shapes an autoregressive loop is at least as likely to produce, each of which
/// defeats a single-statistic, single-axis descriptor:
///
/// 1. **Plateauing wash-out** — the loop settles into a degenerate attractor: brightness ramps for a
///    third of the clip and then holds. The OLS trend over the post-roll segment is structurally small;
///    only the excursion sees it. (Measured at TREND ≈ 11.9 against excursion ≈ 32.)
/// 2. **Localized subject collapse** — the centre of the frame goes dark while the surround brightens
///    to hold the global mean. Every *global* moment is unmoved; only the spatial component sees it.
/// 3. **Hue rotation at constant luma and saturation** — a colour cast develops. Luma, contrast and
///    max−min saturation are all unmoved; only the opponent-colour components see it.
///
/// Each must exceed [`DRIFT_BUDGET`], and — because "it scored high on *something*" is not enough —
/// each is also asserted to be caught by the *specific* statistic/axis that exists for it, so the
/// coverage cannot silently rot away.
#[test]
fn the_drift_metric_catches_the_plausible_ar_failure_shapes() {
    const W: usize = 128;
    const H: usize = 96;
    const N: usize = 180;
    const PRE: usize = 24;
    const BUCKET: usize = 12;

    // 1. Plateau: +40/255 wash-out that starts at the first eviction (frame `PRE` — a real AR drift
    //    cannot begin before the window first rolls), is fully in place by frame 54, and then holds.
    let plateau = map_frames(N, W, H, |i, _, f| {
        let a = 40.0 * ((i.saturating_sub(PRE) as f64 / (54 - PRE) as f64).min(1.0));
        for c in f.pixels.iter_mut() {
            *c = (*c as f64 + a).clamp(0.0, 255.0) as u8;
        }
    });
    let p = score_clip(&plateau, PRE, BUCKET);
    print_drift("plateau", &p);
    assert!(
        worst_drift(&p) > DRIFT_BUDGET,
        "a permanent +40/255 wash-out that plateaus scored {:.2}/255, inside the {DRIFT_BUDGET:.2} \
         budget — the metric is blind to the most likely AR failure shape",
        worst_drift(&p)
    );
    assert!(
        worst_excursion(&p) > worst_trend(&p),
        "the plateau was caught by the TREND ({:.2}) not the excursion ({:.2}) — this stimulus has \
         stopped demonstrating why the excursion is gated, so MAJOR-4 is no longer evidenced",
        worst_trend(&p),
        worst_excursion(&p)
    );

    // 1b. STEP plateau: the same +40/255 wash-out, but the loop falls into the degenerate attractor
    //     within one AR chunk of the first eviction and then holds. This is where an OLS trend is
    //     *genuinely* blind — a step early in the segment leaves almost no slope — so it is the
    //     stimulus that makes the excursion gate load-bearing rather than merely redundant.
    let step = map_frames(N, W, H, |i, _, f| {
        if i >= PRE + 4 {
            for c in f.pixels.iter_mut() {
                *c = (*c as f64 + 40.0).clamp(0.0, 255.0) as u8;
            }
        }
    });
    let st = score_clip(&step, PRE, BUCKET);
    print_drift("step", &st);
    assert!(
        worst_trend(&st) < DRIFT_BUDGET,
        "the step-onset plateau scored {:.2}/255 on the TREND alone — this stimulus no longer \
         demonstrates the OLS blind spot, so it cannot evidence why the excursion is gated",
        worst_trend(&st)
    );
    assert!(
        worst_drift(&st) > DRIFT_BUDGET,
        "a +40/255 wash-out that lands one chunk after the first eviction and holds scored {:.2}/255 \
         — the gate is blind to a degenerate attractor, which is the most likely AR failure",
        worst_drift(&st)
    );

    // 2. Localized collapse: the central 12% of the frame fades to black while the surround brightens
    //    to keep the GLOBAL luma mean where it started. Statistically invisible to global moments.
    let (cx, cy) = (W as f64 / 2.0, H as f64 / 2.0);
    let r2 = 0.12 * (W * H) as f64 / std::f64::consts::PI;
    let localized = map_frames(N, W, H, |_, t, f| {
        let (mut inside, mut outside) = (0usize, 0usize);
        let mut lost = 0.0f64;
        for (i, px) in f.pixels.chunks_exact(3).enumerate() {
            let (x, y) = ((i % W) as f64 - cx, (i / W) as f64 - cy);
            let luma = 0.299 * px[0] as f64 + 0.587 * px[1] as f64 + 0.114 * px[2] as f64;
            if x * x + y * y <= r2 {
                inside += 1;
                lost += t * luma;
            } else {
                outside += 1;
            }
        }
        let comp = lost / outside.max(1) as f64;
        let _ = inside;
        for (i, px) in f.pixels.chunks_exact_mut(3).enumerate() {
            let (x, y) = ((i % W) as f64 - cx, (i / W) as f64 - cy);
            if x * x + y * y <= r2 {
                for c in px.iter_mut() {
                    *c = (*c as f64 * (1.0 - t)).clamp(0.0, 255.0) as u8;
                }
            } else {
                for c in px.iter_mut() {
                    *c = (*c as f64 + comp).clamp(0.0, 255.0) as u8;
                }
            }
        }
    });
    let l = score_clip(&localized, PRE, BUCKET);
    print_drift("localized", &l);
    assert!(
        worst_drift(&l) > DRIFT_BUDGET,
        "a localized subject collapse that holds the global mean scored {:.2}/255, inside the \
         {DRIFT_BUDGET:.2} budget — global moments alone cannot see subject-identity loss",
        worst_drift(&l)
    );
    // ...and it is the SPATIAL component that caught it, which is why that component exists.
    let spatial = l[5].trend.abs().max(l[5].gated_excursion().abs());
    assert!(
        spatial > DRIFT_BUDGET,
        "the spatial component scored only {spatial:.2}/255 on a localized collapse — the block-luma \
         spread is not doing the job it was added for"
    );

    // 3. Hue rotation at constant luma AND constant chroma magnitude: rotate the (Cb, Cr) opponent
    //    pair through 180 degrees over the clip. Luma is preserved exactly by reconstruction and the
    //    chroma *magnitude* is preserved by the rotation, so neither luma-mean, contrast, saturation
    //    nor the spatial spread can see it — only the opponent pair, which ends sign-reversed.
    let hue = map_frames(N, W, H, |_, t, f| {
        let (s, c) = (std::f64::consts::PI * t).sin_cos();
        for px in f.pixels.chunks_exact_mut(3) {
            let (r, g, b) = (px[0] as f64, px[1] as f64, px[2] as f64);
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let (cb, cr) = (b - y, r - y);
            let (cb2, cr2) = (cb * c - cr * s, cb * s + cr * c);
            let (r2, b2) = (y + cr2, y + cb2);
            let g2 = (y - 0.299 * r2 - 0.114 * b2) / 0.587;
            px[0] = r2.clamp(0.0, 255.0) as u8;
            px[1] = g2.clamp(0.0, 255.0) as u8;
            px[2] = b2.clamp(0.0, 255.0) as u8;
        }
    });
    let hu = score_clip(&hue, PRE, BUCKET);
    print_drift("hue", &hu);
    assert!(
        worst_drift(&hu) > DRIFT_BUDGET,
        "a 180-degree hue rotation at constant luma scored {:.2}/255, inside the {DRIFT_BUDGET:.2} \
         budget — the descriptor cannot see a colour cast",
        worst_drift(&hu)
    );
    let opponent = hu[3..5]
        .iter()
        .map(|c| c.trend.abs().max(c.gated_excursion().abs()))
        .fold(0.0, f64::max);
    assert!(
        opponent > DRIFT_BUDGET,
        "the opponent-colour components scored only {opponent:.2}/255 on a hue rotation — they are \
         not doing the job they were added for"
    );

    // The budget's UPPER pin: the weakest of these four shapes bounds how high `DRIFT_BUDGET` may be
    // set. Assert the margin explicitly, so raising the budget past what the metric can actually
    // detect turns this gate red instead of silently widening the blind spot.
    let weakest = [&p, &st, &l, &hu]
        .into_iter()
        .map(worst_drift)
        .fold(f64::INFINITY, f64::min);
    assert!(
        weakest > DRIFT_BUDGET * 1.3,
        "the weakest plausible failure shape scores {weakest:.2}/255 against a {DRIFT_BUDGET:.2} \
         budget — under a 1.3x margin, so the budget is too close to the metric's detection limit \
         for a no-drift verdict to mean anything"
    );

    // No descriptor component may be INERT: a channel that returns the same number on every stimulus
    // is contributing nothing and would silently narrow the coverage this gate claims. Compare each
    // component's trend across the four stimuli and require it to actually move somewhere.
    let motion: Vec<Image> = (0..N).map(|i| smooth_frame(W, H, i)).collect();
    let mo = score_clip(&motion, PRE, BUCKET);
    for (c, name) in DESCRIPTOR_NAMES.iter().enumerate().map(|(i, n)| (i, *n)) {
        let vals = [
            mo[c].trend,
            p[c].trend,
            l[c].trend,
            hu[c].trend,
            mo[c].excursion,
            p[c].excursion,
            l[c].excursion,
            hu[c].excursion,
        ];
        let span = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - vals.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            span > 1.0,
            "descriptor component `{name}` moves only {span:.4}/255 across four different failure \
             shapes — it is inert and the descriptor is narrower than it claims"
        );
    }
}

/// One measured row of the S18 sweep, as printed by the gated real-weight run.
struct S18Row {
    label: String,
    /// The verdict statistic: worst trend or worst z-gated excursion (see [`worst_drift`]).
    drift: f64,
    trend: f64,
    excursion: f64,
    /// Worst per-100-frame slope — the only length-comparable statistic, used for row Z.
    slope: f64,
    rolls: usize,
    /// MLX active peak for this row's AR loop — the sink/window memory cost, measured.
    peak: usize,
    /// Mean % of pixels at or above 250 in the brightest channel — the independent quality metric.
    clip_mean: f64,
    /// Mean |Δ| per frame over the first / last third: the freeze check.
    head_motion: f64,
    tail_motion: f64,
    row: char,
    seed: u64,
}

/// One row of the S18 sweep. Rows share a prompt, seed and geometry, but **not** a clip: changing
/// `local_attn_size` changes the attention mask and therefore the sampling trajectory. Rows D and Z are
/// the within-regime controls (same bounded local-attention path); row E is an out-of-regime reference.
struct DriftRun {
    row: char,
    label: &'static str,
    window: i64,
    sink: usize,
    /// Latent frames for this row. Row Z is deliberately short — short enough that the shipped window
    /// never evicts — so only its *slope* is comparable with the long rows.
    latent_frames: Option<usize>,
}

/// **sc-15127 (S18) — does a long batch clip drift as the bounded KV window rolls, and does a
/// first-chunk sink anchor fix it?**
///
/// Generates the clip once per (row, seed) — varying only the KV read window, the sink anchor and the
/// seed — scores each with [`score_clip`], and prints the table the verdict rests on. Each row's MLX
/// peak is reported too, so the sink's KV cost is measured rather than asserted: a sink is permanently
/// resident, and the bounded window exists precisely because KV is expensive on this host.
///
/// **Rows.** `A` shipped (window 6, sink 0) · `B` sink 1 · `C` sink 3 · `D` wide window (the
/// within-regime dose-response) · `E` the checkpoint's global window (an out-of-regime *reference*, not
/// a floor — see the module comment) · `Z` the within-regime zero-eviction rate floor (shipped window,
/// clip short enough never to roll).
///
/// **Driving it.** `KREA_S18_ROWS` (default `ABCDEZ`) selects rows; `KREA_S18_SEEDS` (comma-separated,
/// default `7`) selects seeds. Each (row, seed) prints one `S18CELL` TSV line, so a long sweep can be
/// run in pieces and re-aggregated without holding a five-hour process open.
///
/// ⚠️ **Geometry — the global reference has to fit in memory.** Row E runs the checkpoint's *global*
/// window, whose KV is `latent_frames × frame_seq_length` tokens at ≈546 KB/token. At the 832×480
/// reference bucket a 45-latent-frame clip is 70,200 tokens ≈ 38 GiB of KV before activations, and this
/// 128 GiB host **SIGKILLs** it (measured, sc-15127: jetsam at step 39/75) — which is exactly the
/// ~27 GB-of-KV problem [`mac_ar_config`](mlx_gen_krea_realtime::mac_ar_config) exists to dodge, so it
/// is a finding rather than a harness bug. Run row E only at a bucket where it fits
/// (`KREA_SMOKE_W=640 KREA_SMOKE_H=384` → 960 tok/frame → 43,200 tokens ≈ 15 GiB, measured to
/// complete). The bounded rows run at both buckets, and **both buckets are recorded** — see
/// [`the_recorded_s18_sweep_is_what_the_docs_claim`].
///
/// ⚠️ Must be run on a tree that has sc-15325 (the tiled-decode fix). Before it, the decode injected an
/// 8-output-frame-period artifact that a drift metric reads as AR drift; an earlier attempt at this
/// exact measurement drew the wrong conclusion from it.
#[test]
#[ignore = "real snapshot, hours of GPU; run with --ignored on macOS (see module doc)"]
fn long_clip_coherence_under_the_bounded_window() {
    use mlx_gen_krea_realtime::{
        generate_latents, load_krea_realtime_transformer_with_quant, ArGenParams,
        CausalKreaTransformer,
    };
    use mlx_gen_wan::{load_tokenizer, Umt5Encoder, WanVae};
    use mlx_rs::Array;

    let root = require_snapshot();
    let w = env_usize("KREA_SMOKE_W", 832);
    let h = env_usize("KREA_SMOKE_H", 480);
    let long_lat = env_usize("KREA_S18_LATENT_FRAMES", 45);
    let seeds: Vec<u64> = std::env::var("KREA_S18_SEEDS")
        .unwrap_or_else(|_| "7".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    assert!(!seeds.is_empty(), "KREA_S18_SEEDS parsed to nothing");
    let want_rows = std::env::var("KREA_S18_ROWS").unwrap_or_else(|_| "ABCDEZ".to_string());
    let (latent_h, latent_w) = (h / 8, w / 8);

    let base = KreaRealtimeConfig::krea_realtime_14b();
    let fpb = base.ar.num_frames_per_block; // 3 latent frames per AR chunk
    let shipped_window = base.ar.streaming_local_attn_frames() as i64; // 6 latent frames
    let wide_window = shipped_window * 2 + 3; // 15 latent frames — the dose-response control
                                              // Row Z: the longest clip the SHIPPED window never evicts on. Chunk `c` commits `fpb*(c+1)` latent
                                              // frames and evicts once that exceeds the window, so `shipped_window` latent frames is exactly it.
    let zero_roll_lat = shipped_window as usize;

    // Every long row is scored over the SAME split, fixed by the shipped window's geometry: the
    // baseline is the output frames the *shipped* configuration produces before its first eviction, and
    // the trend is measured over everything after. A per-row split would make the rows incomparable.
    let pre_len = shipped_window as usize * 4;
    let bucket = fpb * 4; // one AR chunk == one window roll == 12 output frames
                          // Row Z's whole clip is `pre_len` frames long, so it needs its own (shorter) baseline. Its trend is
                          // therefore NOT comparable with the long rows — only its slope is, which is what it is used for.
    let z_pre_len = fpb * 4;

    let all_runs = [
        DriftRun {
            row: 'A',
            label: "A shipped    (window  6, sink 0)",
            window: shipped_window,
            sink: 0,
            latent_frames: None,
        },
        DriftRun {
            row: 'B',
            label: "B anchor 1f  (window  6, sink 1)",
            window: shipped_window,
            sink: 1,
            latent_frames: None,
        },
        DriftRun {
            row: 'C',
            label: "C anchor 3f  (window  6, sink 3)",
            window: shipped_window,
            sink: fpb,
            latent_frames: None,
        },
        DriftRun {
            row: 'D',
            label: "D wide win   (window 15, sink 0)",
            window: wide_window,
            sink: 0,
            latent_frames: None,
        },
        DriftRun {
            row: 'E',
            label: "E global ref (global window, sink 0)",
            window: -1,
            sink: 0,
            latent_frames: None,
        },
        DriftRun {
            row: 'Z',
            label: "Z zero-roll  (window  6, sink 0, short)",
            window: shipped_window,
            sink: 0,
            latent_frames: Some(zero_roll_lat),
        },
    ];
    let runs: Vec<&DriftRun> = all_runs
        .iter()
        .filter(|r| want_rows.contains(r.row))
        .collect();
    assert!(
        !runs.is_empty(),
        "KREA_S18_ROWS `{want_rows}` selected no rows"
    );

    // --- Encode the prompt ONCE, then drop the ~11 GB text encoder before any DiT is resident. ---
    let prompt = "a red fox trotting through a snowy pine forest at sunrise, drifting snow, \
                  cinematic, shallow depth of field";
    let context = {
        let tokenizer =
            load_tokenizer(root.join("tokenizer.json"), base.wan.text_len).expect("tokenizer");
        let mut tw = mlx_gen::weights::Weights::from_file(root.join("t5_encoder.safetensors"))
            .expect("open the TE");
        let enc = Umt5Encoder::from_weights_quantized(
            &mut tw,
            &base.wan,
            mlx_gen_wan::config::WanQuant {
                bits: 8,
                group_size: 64,
            },
        )
        .expect("UMT5");
        let c = enc.encode(&tokenizer, prompt).expect("encode");
        // MLX is lazy: without this the context is an unexecuted graph held across the TE's drop.
        mlx_rs::transforms::eval([&c]).expect("eval context");
        c
    };
    mlx_rs::memory::clear_cache();

    let vw =
        mlx_gen::weights::Weights::from_file(root.join("vae.safetensors")).expect("open the VAE");
    let vae = WanVae::from_weights(&vw).expect("load the z16 Wan VAE");

    // Pin the decode budget so every row decodes through the same plan (the comparison is between AR
    // configs; a host-dependent decode plan would be a second free variable).
    std::env::set_var("WAN_VAE_BUDGET_GIB", "20");

    let mut results: Vec<S18Row> = Vec::new();
    for r in &runs {
        for &seed in &seeds {
            let lat = r.latent_frames.unwrap_or(long_lat);
            let row_pre_len = if r.latent_frames.is_some() {
                z_pre_len
            } else {
                pre_len
            };
            let mut cfg = base.clone();
            cfg.ar.local_attn_size = r.window;
            cfg.ar.sink_size = r.sink;
            cfg.ar.frame_seq_length =
                (latent_h / cfg.wan.patch_size.1) * (latent_w / cfg.wan.patch_size.2);
            cfg.ar.seq_length = lat * cfg.ar.frame_seq_length;

            let chunks = lat.div_ceil(fpb);
            // Chunk `c` commits latent frames up to `fpb*(c+1)`; it evicts once that exceeds the
            // window. A global window (`local_attn_size < 0`) never evicts.
            let rolls = if r.window < 0 {
                0
            } else {
                (0..chunks)
                    .filter(|c| fpb * (c + 1) > r.window as usize)
                    .count()
            };

            println!(
                "=== {} seed {seed} | {lat} latent frames, {chunks} chunks, {rolls} evicting, \
                 baseline {row_pre_len} output frames, window {} tok",
                r.label,
                cfg.ar.max_attention_size()
            );

            mlx_rs::memory::clear_cache();
            mlx_rs::memory::reset_peak_memory();
            let t0 = Instant::now();
            let latents = {
                let dw = mlx_gen::weights::Weights::from_file(root.join("dit.safetensors"))
                    .expect("DiT");
                let raw: std::collections::HashMap<String, Array> = dw
                    .keys()
                    .map(|k| (k.to_string(), dw.get(k).expect("listed key").clone()))
                    .collect();
                let (dit, _) =
                    load_krea_realtime_transformer_with_quant(raw, &cfg).expect("load the DiT");
                let transformer = CausalKreaTransformer::new(dit, &cfg);
                let params = ArGenParams {
                    seed,
                    steps: None,
                    num_latent_frames: lat,
                    latent_height: latent_h,
                    latent_width: latent_w,
                    fps: 24,
                };
                // Print a per-step mark. This sweep is long enough that a silent run is
                // indistinguishable from a hung one, and the per-step cadence is also how a
                // memory-contended row (the global reference especially) announces itself.
                let mut tstep = Instant::now();
                let l = generate_latents(
                    &transformer,
                    &cfg,
                    &context,
                    &params,
                    &mlx_gen::CancelFlag::default(),
                    &mut |p| {
                        if let Progress::Step { current, total } = p {
                            let now = Instant::now();
                            println!(
                                "    step {current:>3}/{total} (+{:.1?}, {:.1?} elapsed)",
                                now.duration_since(tstep),
                                t0.elapsed()
                            );
                            tstep = now;
                        }
                    },
                )
                .expect("generate latents");
                // MLX is lazy — without this the AR loop has not actually run and the peak below is a
                // shape calculation. Three measurements in this epic read ~0 for exactly this reason.
                mlx_rs::transforms::eval([&l]).expect("materialize latents");
                l
            };
            let ar_peak = mlx_rs::memory::get_peak_memory();
            let ar_wall = t0.elapsed();
            mlx_rs::memory::clear_cache();

            let out_frames = lat * 4;
            let tiling =
                decode_tiling(h, w, out_frames as i32).expect("decode policy must be feasible");
            let frames = match decode_latents_to_video(
                &vae,
                &latents,
                24,
                None,
                tiling.as_ref(),
                &mlx_gen::CancelFlag::default(),
            )
            .expect("VAE decode")
            {
                GenerationOutput::Video { frames, .. } => frames,
                other => panic!("expected a Video output, got {other:?}"),
            };
            drop(latents);
            mlx_rs::memory::clear_cache();

            assert_eq!(frames.len(), out_frames, "{}: decoded frame count", r.label);
            println!(
                "  AR wall {:.1?} ({:.1?}/chunk), MLX active peak {:.2} GiB, decode plan {tiling:?}",
                ar_wall,
                ar_wall / chunks as u32,
                gib(ar_peak)
            );
            report_artifacts(&frames, r.label);
            let d = score_clip(&frames, row_pre_len, bucket);
            print_drift(r.label, &d);
            dump_frames(
                &frames,
                &format!("s18_{}_w{}_sink{}_{lat}_s{seed}", r.row, r.window, r.sink),
            );

            // Two INDEPENDENT cross-checks, because "the drift statistic is small" is not by itself a
            // quality claim:
            //   * highlight clipping — what a viewer actually sees blow out, and a metric with no
            //     construction in common with `score_clip`.
            //   * tail motion — a clip that has stopped moving also has no trend. Freeze and drift are
            //     the two AR failure modes, and a gate that only sees one can be passed by the other.
            let (clip_mean, _) = clip_stats(&frames);
            let deltas: Vec<f64> = frame_stats(&frames)
                .iter()
                .skip(1)
                .map(|s| s.delta)
                .collect();
            let third = (deltas.len() / 3).max(1);
            let head_motion = mean_f64(&deltas[..third]);
            let tail_motion = mean_f64(&deltas[deltas.len() - third..]);
            println!(
                "    motion: head {head_motion:.2}/255 -> tail {tail_motion:.2}/255 per frame; \
                 clipping mean {clip_mean:.2}%"
            );
            // Machine-readable, so a sweep split across processes can be re-aggregated.
            println!(
                "S18CELL\t{}\t{seed}\t{w}x{h}\t{lat}\t{rolls}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t\
                 {:.4}\t{:.4}\t{:.4}",
                r.row,
                worst_drift(&d),
                worst_trend(&d),
                worst_excursion(&d),
                worst_slope(&d),
                ar_peak,
                clip_mean,
                head_motion,
                tail_motion
            );
            results.push(S18Row {
                label: format!("{} s{seed}", r.label),
                drift: worst_drift(&d),
                trend: worst_trend(&d),
                excursion: worst_excursion(&d),
                slope: worst_slope(&d),
                rolls,
                peak: ar_peak,
                clip_mean,
                head_motion,
                tail_motion,
                row: r.row,
                seed,
            });
        }
    }
    std::env::remove_var("WAN_VAE_BUDGET_GIB");

    println!("=== sc-15127 S18 summary ({w}x{h}) ==========================================");
    println!(
        "  row                                       rolls   drift   trend   excur  slope/100f  \
         peak GiB  clip%  tail motion"
    );
    for r in &results {
        println!(
            "  {:<41} {:>5}  {:>6.2}  {:>6.2}  {:>6.2}  {:>10.3}  {:>8.2}  {:>5.2}  {:>8.2}",
            r.label,
            r.rolls,
            r.drift,
            r.trend,
            r.excursion,
            r.slope,
            gib(r.peak),
            r.clip_mean,
            r.tail_motion
        );
    }

    // FREEZE gate, every row. A frozen tail is the other way a bounded window can ruin a long clip,
    // and it would read as *excellent* on a drift metric — so a no-drift verdict is only meaningful
    // alongside this.
    for r in &results {
        assert!(
            r.tail_motion > 1.0 && r.tail_motion > r.head_motion * 0.2,
            "{}: the clip's tail moves {:.2}/255 per frame against a head of {:.2} — it has frozen, \
             so its low drift score means nothing",
            r.label,
            r.tail_motion,
            r.head_motion
        );
    }
    // CROSS-CHECK gate: highlight clipping, an independent metric, must agree that the shipped bounded
    // window is a healthy operating point. Judged **within regime** against the wider bounded window
    // (row D) — comparing it against the global row would be comparing against a different attention
    // regime, and against an absolute ceiling, so a run where everything clips cannot pass by being
    // uniformly bad.
    const CLIP_CEILING: f64 = 8.0;
    let mean_clip = |row: char| {
        let v: Vec<f64> = results
            .iter()
            .filter(|r| r.row == row)
            .map(|r| r.clip_mean)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    };
    if let Some(a) = mean_clip('A') {
        assert!(
            a <= CLIP_CEILING,
            "the shipped bounded window clips {a:.2}% of pixels — over the {CLIP_CEILING:.1}% \
             ceiling, so it is not a healthy operating point whatever the drift statistic says"
        );
        if let Some(d) = mean_clip('D') {
            assert!(
                a <= d * 2.0,
                "the shipped window clips {a:.2}% against the wider bounded window's {d:.2}% — the \
                 within-regime dose-response says tightening the window blows highlights, which the \
                 drift statistic must not be allowed to paper over"
            );
        }
    }
    // Row Z's contribution: the within-regime, zero-eviction rate. The shipped row's rate must not be
    // wildly above the rate the SAME attention regime shows with no evictions at all.
    let mean_slope = |row: char| {
        let v: Vec<f64> = results
            .iter()
            .filter(|r| r.row == row)
            .map(|r| r.slope)
            .collect();
        (!v.is_empty()).then(|| mean_f64(&v))
    };
    if let (Some(a), Some(z)) = (mean_slope('A'), mean_slope('Z')) {
        println!(
            "  within-regime rate floor: zero-eviction row Z {z:.3}/100f vs shipped row A \
             {a:.3}/100f ({:.2}x)",
            a / z.max(1e-6)
        );
    }

    let sweep = S18Sweep {
        bucket: format!("{w}x{h}"),
        cells: results
            .iter()
            .map(|r| S18Cell {
                row: r.row,
                seed: r.seed,
                rolls: r.rolls,
                trend: r.trend,
                excursion: r.excursion,
            })
            .collect(),
    };
    println!("  {}", sweep.summary());
    if want_rows.contains('A') {
        match sweep.verdict() {
            Ok(v) => println!("  VERDICT: {v}"),
            Err(e) => panic!("{e}"),
        }
    } else {
        println!(
            "  (partial sweep: rows `{want_rows}` do not include the shipped row A, so no verdict \
             is computed — re-aggregate the S18CELL lines)"
        );
    }
}

/// One measured (row, seed) cell of the S18 sweep.
#[derive(Clone, Copy, Debug)]
struct S18Cell {
    /// `A` shipped · `B` sink 1 · `C` sink 3 · `D` wide window · `E` global reference · `Z` zero-roll.
    row: char,
    seed: u64,
    rolls: usize,
    /// Worst absolute one-way component trend, 0..255.
    trend: f64,
    /// Worst absolute z-gated component excursion, 0..255.
    excursion: f64,
}

impl S18Cell {
    /// The gated statistic: trend AND excursion must be inside budget, i.e. their max must be.
    fn drift(&self) -> f64 {
        self.trend.abs().max(self.excursion.abs())
    }
}

/// A measured S18 sweep at one geometry bucket, and the decision rule applied to it. Split out of the
/// real-weight driver so the **rule** is gated in CI rather than only exercised on the gated GPU run.
struct S18Sweep {
    bucket: String,
    cells: Vec<S18Cell>,
}

impl S18Sweep {
    fn of(&self, row: char) -> Vec<f64> {
        self.cells
            .iter()
            .filter(|c| c.row == row)
            .map(|c| c.drift())
            .collect()
    }

    fn mean(&self, row: char) -> Option<f64> {
        let v = self.of(row);
        (!v.is_empty()).then(|| mean_f64(&v))
    }

    /// The between-seed uncertainty on a row's **mean**: twice the standard error, i.e. roughly a 95%
    /// half-interval. `None` if the row has fewer than two seeds — in which case there is **no
    /// variance estimate at all**, which the verdict must say out loud.
    ///
    /// Deliberately not the max−min range: that *grows* with the number of seeds, so a rule gated on
    /// it would get harder to satisfy the more evidence you collected. This shrinks as 1/√n, which is
    /// what makes "add seeds until the comparison resolves" a real option.
    fn spread(&self, row: char) -> Option<f64> {
        let v = self.of(row);
        (v.len() >= 2).then(|| 2.0 * std_f64(&v) / (v.len() as f64).sqrt())
    }

    /// The row with the smaller mean of `B`/`C`, i.e. the sink anchor that did best.
    fn best_sink(&self) -> Option<(char, f64)> {
        ['B', 'C']
            .iter()
            .filter_map(|&r| self.mean(r).map(|m| (r, m)))
            .fold(None, |acc: Option<(char, f64)>, x| match acc {
                Some(a) if a.1 <= x.1 => Some(a),
                _ => Some(x),
            })
    }

    fn summary(&self) -> String {
        let f = |r: char| match (self.mean(r), self.spread(r)) {
            (None, _) => "-".to_string(),
            (Some(m), None) => format!("{m:.2} (n=1)"),
            (Some(m), Some(s)) => format!("{m:.2} ±{s:.2} (n={}, 2*SEM)", self.of(r).len()),
        };
        format!(
            "{} — A shipped {} | B sink1 {} | C sink3 {} | D wide {} | E global-ref {} | budget \
             {DRIFT_BUDGET:.2}/255 (absolute)",
            self.bucket,
            f('A'),
            f('B'),
            f('C'),
            f('D'),
            f('E'),
        )
    }

    /// The verdict, or the reason this sweep cannot support one.
    ///
    /// The rule is deliberately conservative about **statistical power**: a single seed per row gives
    /// no variance estimate, and this sweep's config-to-config differences are of the same order as its
    /// seed-to-seed scatter. So the rule refuses to *rank configs* unless the margin it would rank them
    /// by exceeds the measured scatter; what it will still do with one seed is state whether gross
    /// drift was observed, which is a much weaker claim and is labelled as such.
    fn verdict(&self) -> std::result::Result<String, String> {
        let shipped_rolls = self
            .cells
            .iter()
            .filter(|c| c.row == 'A')
            .map(|c| c.rolls)
            .max()
            .ok_or_else(|| "no shipped (A) row was measured — nothing to conclude".to_string())?;
        if shipped_rolls < 10 {
            return Err(format!(
                "the shipped row only rolled the window {shipped_rolls} times — that is not a long \
                 clip, so it cannot answer the long-clip question. Raise KREA_S18_LATENT_FRAMES."
            ));
        }
        if self.cells.iter().any(|c| c.row == 'E' && c.rolls != 0) {
            return Err(
                "the global reference row rolled the window — it is not a reference".into(),
            );
        }
        if self.cells.iter().any(|c| c.row == 'Z' && c.rolls != 0) {
            return Err(
                "the zero-roll row Z evicted — its clip is too long to be a zero-eviction floor"
                    .into(),
            );
        }
        let a = self.mean('A').expect("row A measured above");
        let margin = (a - DRIFT_BUDGET).abs();
        // Which side of the budget the SHIPPED row is on is decided by the shipped row's own
        // between-seed spread, not by the noisiest row in the sweep.
        let a_spread = self.spread('A');
        let resolvable = match a_spread {
            None => false,
            Some(s) => s < margin,
        };

        if !resolvable {
            let power = match a_spread {
                None => "one seed per row — NO variance estimate".to_string(),
                Some(s) => format!("between-seed spread {s:.2}/255 vs a {margin:.2}/255 margin"),
            };
            // Underpowered to rank, but a *gross* result is still reportable: if every measured
            // bounded row is far inside the budget, "no gross drift observed" survives the scatter.
            let worst_bounded = ['A', 'B', 'C', 'D']
                .iter()
                .filter_map(|&r| self.mean(r))
                .fold(0.0f64, f64::max);
            if worst_bounded <= DRIFT_BUDGET / 2.0 {
                return Ok(format!(
                    "underpowered but no gross drift — every bounded row sits at or below \
                     {worst_bounded:.2}/255 against a {DRIFT_BUDGET:.2} budget over \
                     {shipped_rolls} rolls ({power}). This does NOT rank the configs and does not \
                     license a claim about any drift mode outside the measured descriptor."
                ));
            }
            return Err(format!(
                "UNDERPOWERED: the shipped row is {a:.2}/255 against a {DRIFT_BUDGET:.2} budget \
                 ({power}), and the bounded rows reach {worst_bounded:.2}. The sweep cannot resolve \
                 which side of the budget the shipped config is on. Add seeds \
                 (KREA_S18_SEEDS) before concluding."
            ));
        }

        if a <= DRIFT_BUDGET {
            // No drift: the within-regime dose-response must agree — a wider window (half the rolls)
            // cannot be materially worse, or "no drift" was an artifact of an insensitive metric.
            if let Some(d) = self.mean('D') {
                if d > DRIFT_BUDGET {
                    return Err(format!(
                        "the shipped window scored {a:.2}/255 (inside the {DRIFT_BUDGET:.2} budget) \
                         but the WIDER window scored {d:.2} — the within-regime dose-response \
                         disagrees, so this run does not support a no-drift verdict"
                    ));
                }
            }
            Ok(format!(
                "coherent — the shipped bounded window holds the clip over {shipped_rolls} window \
                 rolls at {a:.2}/255 against a {DRIFT_BUDGET:.2}/255 absolute budget, with the \
                 between-seed scatter under the margin. No sink anchor warranted for the drift \
                 modes this descriptor covers."
            ))
        } else {
            // Drift is real. **Attribution first**: sc-15127 asks whether the *bounded window* costs
            // coherence, and "the clip drifts" does not answer that. The within-regime dose-response
            // (row D, the same local-attention path at a 2.5x wider window and fewer rolls) is what
            // decides it. If halving the roll count does not materially reduce the drift, eviction is
            // not the demonstrated mechanism — and then a first-chunk sink, whose whole rationale is
            // to survive eviction, is not what the evidence points at, whatever the sink rows read.
            if let Some(d) = self.mean('D') {
                let unc = self.spread('A').unwrap_or(0.0) + self.spread('D').unwrap_or(0.0);
                if d + unc >= a {
                    let d_rolls = self
                        .cells
                        .iter()
                        .find(|c| c.row == 'D')
                        .map(|c| c.rolls)
                        .unwrap_or(0);
                    let sinks = ['B', 'C']
                        .iter()
                        .filter_map(|&r| self.mean(r).map(|m| format!("{r} {m:.2}")))
                        .collect::<Vec<_>>()
                        .join(", ");
                    let global = match self.mean('E') {
                        Some(e) => format!(" The zero-eviction global reference reads {e:.2}."),
                        None => String::new(),
                    };
                    return Ok(format!(
                        "drift is real ({a:.2}/255 over {shipped_rolls} rolls against a \
                         {DRIFT_BUDGET:.2} budget) but it is NOT attributable to the bounded KV \
                         window: the within-regime dose-response runs the wrong way — at {d_rolls} \
                         rolls the 2.5x WIDER window scores {d:.2}/255, no better than the shipped \
                         window's {a:.2} (combined 2*SEM {unc:.2}).{global} Fewer evictions do not \
                         buy coherence, so eviction is not the mechanism and a first-chunk sink \
                         anchor is not indicated. No sink is wired. Sink rows for the record: \
                         {sinks}. The drift itself needs its own investigation."
                    ));
                }
            }
            // The remaining question is whether the anchored rows repair it — a SECOND comparison
            // with its own power problem, so it gets its own resolvability check.
            let Some((row, best_sink)) = self.best_sink() else {
                return Err(format!(
                    "the shipped window drifted {a:.2}/255 past the {DRIFT_BUDGET:.2} budget but no \
                     sink row was measured — run rows B and C before concluding"
                ));
            };
            let threshold = a * 0.6;
            // The repair claim is a comparison between two noisy means, so the gap must clear the
            // scatter of BOTH rows before it can be called either way.
            let combined = self.spread('A').unwrap_or(f64::INFINITY)
                + self.spread(row).unwrap_or(f64::INFINITY);
            if (threshold - best_sink).abs() < combined {
                return Ok(format!(
                    "drift is real ({a:.2}/255 over {shipped_rolls} rolls against a \
                     {DRIFT_BUDGET:.2} budget), and the sink anchor's effect on it is NOT resolvable \
                     at this sample size: the best sink row ({row}) is {best_sink:.2}/255 against a \
                     {threshold:.2} repair threshold, a gap of {:.2} inside a combined between-seed \
                     scatter of {combined:.2}. No sink is wired — permanently-resident KV must not be \
                     bought on an unresolved comparison.",
                    (threshold - best_sink).abs()
                ));
            }
            if best_sink >= threshold {
                return Ok(format!(
                    "drift is real ({a:.2}/255 over {shipped_rolls} rolls against a \
                     {DRIFT_BUDGET:.2} budget) and a first-chunk sink anchor does NOT repair it: the \
                     best sink row ({row}) only reached {best_sink:.2}/255 against a {threshold:.2} \
                     repair threshold. No sink is wired; sc-15127 needs a different anchor."
                ));
            }
            Ok(format!(
                "drift is real ({a:.2}/255 over {shipped_rolls} rolls against a {DRIFT_BUDGET:.2} \
                 budget) and a first-chunk sink anchor repairs it to {best_sink:.2}/255 (row {row})."
            ))
        }
    }
}

/// **CI gate for the S18 decision rule.** The real-weight run happens on a gated host; the rule it
/// applies to those numbers must not be un-tested. Drives [`S18Sweep::verdict`] over every outcome it
/// distinguishes, including the two power-related ones added after review.
#[test]
fn the_s18_verdict_rule_distinguishes_its_outcomes() {
    // Build a sweep with `n` seeds per row, each row's values spread by `spread` around its mean.
    let sweep = |means: [f64; 5], spread: f64, n: usize| {
        let mut cells = Vec::new();
        for (i, row) in ['A', 'B', 'C', 'D', 'E'].iter().enumerate() {
            for k in 0..n {
                let off = if n < 2 {
                    0.0
                } else {
                    spread * (k as f64 / (n - 1) as f64 - 0.5)
                };
                cells.push(S18Cell {
                    row: *row,
                    seed: k as u64,
                    rolls: if *row == 'E' { 0 } else { 13 },
                    trend: means[i] + off,
                    excursion: 0.0,
                });
            }
        }
        S18Sweep {
            bucket: "test".into(),
            cells,
        }
    };

    // 1. Coherent: every bounded row sits far under the budget, with a resolvable margin.
    let v = sweep([2.0, 2.1, 1.9, 2.2, 1.8], 0.4, 3)
        .verdict()
        .expect("a flat, replicated sweep must yield a verdict");
    assert!(v.starts_with("coherent"), "got: {v}");

    // 2. Drift, repaired by the anchor: the shipped row is far past the budget and the sinks pull it
    //    most of the way back.
    let v = sweep([40.0, 8.0, 6.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("a repaired-drift sweep must yield a verdict");
    assert!(v.starts_with("drift is real"), "got: {v}");

    // 2b. Drift that is NOT attributable to the window: the wider window (fewer rolls) is no better.
    //     The rule must say so and must not reach for a sink, even though the sink rows look great.
    let v = sweep([40.0, 8.0, 6.0, 41.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unattributed drift is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(v.contains("NOT attributable"), "got: {v}");
    assert!(!v.contains("repairs it to"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");

    // 3. Drift the sink does NOT repair — a real finding, but it must never read as "ship a sink".
    let v = sweep([40.0, 38.0, 39.0, 20.0, 2.0], 2.0, 3)
        .verdict()
        .expect("an unrepaired drift is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(v.contains("does NOT repair it"), "got: {v}");
    assert!(!v.contains("repairs it to"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");

    // 3b. Drift where the sink comparison is swamped by seed scatter — must say so, and must still
    //     refuse to wire a sink.
    let v = sweep([40.0, 22.0, 26.0, 20.0, 2.0], 12.0, 3)
        .verdict()
        .expect("an unresolvable repair comparison is still a conclusion");
    assert!(v.starts_with("drift is real"), "got: {v}");
    assert!(v.contains("NOT resolvable"), "got: {v}");
    assert!(v.contains("No sink is wired"), "got: {v}");

    // 4. Incoherent evidence: the shipped row looks clean but the WIDER window looks worse. That
    //    ordering is impossible under the drift hypothesis, so the run proves nothing either way.
    let e = sweep([3.0, 3.0, 3.0, 40.0, 2.0], 0.4, 3)
        .verdict()
        .expect_err("contradictory rows must not produce a no-drift verdict");
    assert!(e.contains("dose-response"), "got: {e}");

    // 5. UNDERPOWERED, near the budget: one seed per row and a shipped row close to the budget. The
    //    rule must refuse rather than pick a side — this is the guard the review demanded.
    let e = sweep([13.0, 8.0, 9.0, 12.0, 2.0], 0.0, 1)
        .verdict()
        .expect_err("a single-seed sweep near the budget must not produce a verdict");
    assert!(e.contains("UNDERPOWERED"), "got: {e}");
    assert!(e.contains("NO variance estimate"), "got: {e}");

    // 5b. ...and the same means WITH enough seeds still refuse while the shipped row's own scatter
    //     swamps its margin against the budget.
    let e = sweep([13.0, 8.0, 9.0, 12.0, 2.0], 12.0, 3)
        .verdict()
        .expect_err("a sweep whose seed scatter swamps its margin must not produce a verdict");
    assert!(e.contains("between-seed spread"), "got: {e}");

    // 6. Underpowered but unambiguously clean: one seed, but everything is far under the budget. That
    //    is reportable — as a narrowed claim that says so.
    let v = sweep([2.0, 2.0, 2.0, 2.0, 2.0], 0.0, 1)
        .verdict()
        .expect("a single-seed but grossly clean sweep must yield a narrowed verdict");
    assert!(v.starts_with("underpowered but no gross drift"), "got: {v}");
    assert!(v.contains("does NOT rank the configs"), "got: {v}");

    // 7. And the sweep must have been long enough / the reference must really be a reference.
    let mut short = sweep([2.0, 2.0, 2.0, 2.0, 2.0], 0.4, 3);
    for c in short.cells.iter_mut().filter(|c| c.row == 'A') {
        c.rolls = 4;
    }
    assert!(short.verdict().unwrap_err().contains("not a long clip"));
    let mut bad_ref = sweep([2.0, 2.0, 2.0, 2.0, 2.0], 0.4, 3);
    for c in bad_ref.cells.iter_mut().filter(|c| c.row == 'E') {
        c.rolls = 3;
    }
    assert!(bad_ref.verdict().unwrap_err().contains("not a reference"));
    // 8. No shipped row at all — refuse, do not index off the end.
    let empty = S18Sweep {
        bucket: "test".into(),
        cells: vec![],
    };
    assert!(empty.verdict().unwrap_err().contains("nothing to conclude"));
}

// --- The recorded measurement -----------------------------------------------------------------
//
// The gated real-weight sweep runs on one host; the numbers it produced are recorded here so the
// documented conclusion is gated by the data rather than by prose. Regenerate with:
//
//   KREA_REALTIME_SNAPSHOT_DIR=... KREA_SMOKE_W=640 KREA_SMOKE_H=384 KREA_S18_SEEDS=7,11,23 \
//     cargo test -p mlx-gen-krea-realtime --test generate_smoke -- --ignored --nocapture \
//     long_clip_coherence_under_the_bounded_window
//
// and paste the `S18CELL` lines below.

/// The prefix of the verdict the crate documentation claims. Changing the documented conclusion means
/// changing this, which means [`the_recorded_s18_sweep_is_what_the_docs_claim`] re-checks it against
/// the recorded data.
const RECORDED_VERDICT_PREFIX: &str = "drift is real";

/// Measured cells at **640×384** (the bucket where the global reference row fits in 128 GiB).
///
/// Row E is `n = 1`: its 45.0 GiB MLX peak drove this 128 GiB host into enough swap to fill the boot
/// volume, so the remaining two seeds were abandoned. It is a *reference*, not a control, and nothing
/// the verdict decides turns on it — the attribution is decided by row D, which is within regime and
/// replicated.
const MEASURED_640: &[S18Cell] = &[
    S18Cell {
        row: 'A',
        seed: 7,
        rolls: 13,
        trend: 19.9907,
        excursion: 23.0413,
    },
    S18Cell {
        row: 'A',
        seed: 11,
        rolls: 13,
        trend: 31.0541,
        excursion: 19.3187,
    },
    S18Cell {
        row: 'A',
        seed: 23,
        rolls: 13,
        trend: 21.7860,
        excursion: 28.4337,
    },
    S18Cell {
        row: 'B',
        seed: 7,
        rolls: 13,
        trend: 11.6090,
        excursion: 17.4084,
    },
    S18Cell {
        row: 'B',
        seed: 11,
        rolls: 13,
        trend: 28.5211,
        excursion: 12.3800,
    },
    S18Cell {
        row: 'B',
        seed: 23,
        rolls: 13,
        trend: 11.8207,
        excursion: 10.3263,
    },
    S18Cell {
        row: 'C',
        seed: 7,
        rolls: 13,
        trend: 17.6473,
        excursion: 14.7965,
    },
    S18Cell {
        row: 'C',
        seed: 11,
        rolls: 13,
        trend: 15.7251,
        excursion: 0.0,
    },
    S18Cell {
        row: 'C',
        seed: 23,
        rolls: 13,
        trend: 6.2537,
        excursion: 5.8270,
    },
    S18Cell {
        row: 'D',
        seed: 7,
        rolls: 10,
        trend: 27.5739,
        excursion: 21.8348,
    },
    S18Cell {
        row: 'D',
        seed: 11,
        rolls: 10,
        trend: 33.0361,
        excursion: 23.0505,
    },
    S18Cell {
        row: 'D',
        seed: 23,
        rolls: 10,
        trend: 31.1090,
        excursion: 30.5435,
    },
    S18Cell {
        row: 'E',
        seed: 7,
        rolls: 0,
        trend: 34.0619,
        excursion: 18.5963,
    },
    S18Cell {
        row: 'Z',
        seed: 7,
        rolls: 0,
        trend: 1.3928,
        excursion: 0.0,
    },
    S18Cell {
        row: 'Z',
        seed: 11,
        rolls: 0,
        trend: 3.3254,
        excursion: 0.0,
    },
    S18Cell {
        row: 'Z',
        seed: 23,
        rolls: 0,
        trend: 5.9501,
        excursion: 0.0,
    },
];

/// Measured cells at **832×480** — the crate default and a shipping bucket. Row E is absent by
/// necessity: the global window at this bucket SIGKILLs a 128 GiB host.
const MEASURED_832: &[S18Cell] = &[
    S18Cell {
        row: 'A',
        seed: 7,
        rolls: 13,
        trend: 45.8218,
        excursion: 47.1323,
    },
    S18Cell {
        row: 'A',
        seed: 11,
        rolls: 13,
        trend: 27.7189,
        excursion: 34.5532,
    },
    S18Cell {
        row: 'A',
        seed: 23,
        rolls: 13,
        trend: 29.2038,
        excursion: 36.0093,
    },
    S18Cell {
        row: 'B',
        seed: 7,
        rolls: 13,
        trend: 25.6222,
        excursion: 34.1733,
    },
    S18Cell {
        row: 'B',
        seed: 11,
        rolls: 13,
        trend: 23.7267,
        excursion: 31.0055,
    },
    S18Cell {
        row: 'B',
        seed: 23,
        rolls: 13,
        trend: 18.8356,
        excursion: 26.8110,
    },
    S18Cell {
        row: 'C',
        seed: 7,
        rolls: 13,
        trend: 14.6706,
        excursion: 22.8868,
    },
    S18Cell {
        row: 'C',
        seed: 11,
        rolls: 13,
        trend: 11.0461,
        excursion: 19.4879,
    },
    S18Cell {
        row: 'C',
        seed: 23,
        rolls: 13,
        trend: 21.1164,
        excursion: 27.9238,
    },
    S18Cell {
        row: 'D',
        seed: 7,
        rolls: 10,
        trend: 47.4356,
        excursion: 47.0857,
    },
    S18Cell {
        row: 'D',
        seed: 11,
        rolls: 10,
        trend: 35.2790,
        excursion: 39.0004,
    },
    S18Cell {
        row: 'D',
        seed: 23,
        rolls: 10,
        trend: 22.8338,
        excursion: 22.2395,
    },
];

/// **CI gate on the recorded measurement.** The numbers the crate documentation cites live in
/// [`MEASURED_640`] / [`MEASURED_832`], and this applies the *same* [`S18Sweep::verdict`] rule to them
/// that the gated GPU run applies to fresh ones. Editing the docs' conclusion without re-measuring, or
/// pasting in a sweep that does not actually support it, turns this red.
///
/// It also gates the thing the review caught: **both buckets are recorded and both are checked.** The
/// 832×480 bucket is the crate default and a shipping bucket; a result there that disagrees with
/// 640×384 cannot be set aside.
#[test]
fn the_recorded_s18_sweep_is_what_the_docs_claim() {
    for (bucket, cells) in [("640x384", MEASURED_640), ("832x480", MEASURED_832)] {
        assert!(
            !cells.is_empty(),
            "the {bucket} sweep is not recorded — the documented conclusion has no data behind it"
        );
        let sweep = S18Sweep {
            bucket: bucket.to_string(),
            cells: cells.to_vec(),
        };
        println!("{}", sweep.summary());
        let v = sweep.verdict();
        println!("  {bucket}: {v:?}");
        assert_eq!(
            v.as_ref().map(|s| s.starts_with(RECORDED_VERDICT_PREFIX)),
            Ok(true),
            "the recorded {bucket} sweep yields `{v:?}`, not the documented \
             `{RECORDED_VERDICT_PREFIX}...` — src/t2v.rs and src/lib.rs are claiming something the \
             data does not say"
        );
        // Replication is the other half of what the review demanded: a recorded bucket with one seed
        // per row cannot support a ranking claim, so it must not be recorded as if it could.
        for row in ['A', 'B', 'C', 'D'] {
            // ...and they must be DIFFERENT seeds. Three cells from one seed is determinism, not
            // replication, and would give a spread of zero that makes every comparison look resolved.
            let mut seeds: Vec<u64> = sweep
                .cells
                .iter()
                .filter(|c| c.row == row)
                .map(|c| c.seed)
                .collect();
            let n = seeds.len();
            seeds.sort_unstable();
            seeds.dedup();
            assert_eq!(
                seeds.len(),
                n,
                "{bucket} row {row} records a repeated seed — that is determinism, not replication"
            );
            if !sweep.of(row).is_empty() {
                assert!(
                    sweep.of(row).len() >= 3,
                    "{bucket} row {row} is recorded with only {} seed(s) — the documented \
                     conclusion would rest on an unreplicated cell",
                    sweep.of(row).len()
                );
            }
        }
    }
}
